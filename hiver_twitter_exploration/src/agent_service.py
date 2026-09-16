"""
Support Agent Service (Unified Real-Time Orchestrator)
======================================================

Purpose:
  Unifies all pipeline stages (Phase 6 Intent Classification, Phase 7 Historical
  Dialogue Retrieval, Phase 8 Safe Reply Drafting, and Phase 9 Risk-Aware Routing)
  into a single, high-performance, real-time Python service.

Key Features:
  - Loads models and sparse matrices once into memory at initialization.
  - Sub-50ms inference per customer query on CPU.
  - Full traceability: returns detailed diagnostic signals from each pipeline stage.
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from src.reply_safety import detect_safety_flags, is_restricted
from src.reply_drafter import draft_reply_for_inquiry, sanitize_text
from src.routing_policy import evaluate_routing_decision, parse_bool


def normalize_customer_input(text: str) -> str:
    """
    Normalizes incoming customer tweet text:
      - Strips leading @mentions (e.g. @AppleSupport)
      - Standardizes URLs to [URL]
      - Normalizes whitespace
    """
    if not text:
        return ""
    t = str(text).strip()
    # Strip leading @mentions
    t = re.sub(r'^(?:@\w+\s*)+', '', t)
    # Replace URLs
    t = re.sub(r'https?://\S+|www\.\S+', '[URL]', t)
    # Normalize excessive whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def get_lexical_similarity_band(score: float) -> str:
    """Assigns standardized lexical similarity band based on Phase 7 criteria."""
    if score < 0.30:
        return "insufficient_lexical_evidence"
    elif score < 0.50:
        return "weak_lexical_evidence"
    elif score < 0.70:
        return "moderate_lexical_evidence"
    else:
        return "high_lexical_similarity"


class SupportAgentService:
    """
    Unified real-time service orchestrating intent classification,
    evidence retrieval, safe reply drafting, and risk-aware routing.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent.parent / "models"
        self.models_dir = Path(models_dir)

        print(f"[SupportAgentService] Initializing models from {self.models_dir}...")

        # 1. Load Intent Classification Artifacts
        vec_path = self.models_dir / "tfidf_vectorizer.joblib"
        cls_path = self.models_dir / "tfidf_logistic_regression.joblib"
        if not vec_path.exists() or not cls_path.exists():
            raise FileNotFoundError(f"Classification models missing in {self.models_dir}")

        self.intent_vectorizer = joblib.load(vec_path)
        self.intent_classifier = joblib.load(cls_path)
        self.intent_classes = list(self.intent_classifier.classes_)

        # 2. Load Retrieval Index Artifacts
        ret_vec_path = self.models_dir / "retrieval_tfidf_vectorizer.joblib"
        ret_mat_path = self.models_dir / "retrieval_customer_matrix.joblib"
        ret_meta_path = self.models_dir / "retrieval_metadata.joblib"
        if not ret_vec_path.exists() or not ret_mat_path.exists() or not ret_meta_path.exists():
            raise FileNotFoundError(f"Retrieval index models missing in {self.models_dir}")

        self.retrieval_vectorizer = joblib.load(ret_vec_path)
        self.retrieval_matrix = joblib.load(ret_mat_path)
        self.retrieval_meta = joblib.load(ret_meta_path)
        self.retrieval_total_inquiries = self.retrieval_matrix.shape[0]

        print(
            f"[SupportAgentService] Ready! Loaded {len(self.intent_classes)} intent classes and "
            f"{self.retrieval_total_inquiries:,} historical training dialogue turns."
        )

    def classify_intent(
        self, text_clean: str, review_confidence_threshold: float = 0.60
    ) -> Dict[str, Any]:
        """
        Runs TF-IDF feature extraction and Logistic Regression Softmax.
        """
        if not text_clean:
            return {
                "predicted_intent": "other_or_unclear",
                "predicted_confidence": 0.0,
                "needs_human_review": True,
                "top_3_candidates": [],
                "all_probabilities": {c: 0.0 for c in self.intent_classes},
            }

        X = self.intent_vectorizer.transform([text_clean])
        proba = self.intent_classifier.predict_proba(X)[0]

        ranked_indices = np.argsort(proba)[::-1]
        top_idx = ranked_indices[0]
        top_intent = self.intent_classes[top_idx]
        top_conf = float(proba[top_idx])

        top_3 = []
        for idx in ranked_indices[:3]:
            top_3.append({
                "intent": self.intent_classes[idx],
                "probability": round(float(proba[idx]), 4),
            })

        all_probs = {
            self.intent_classes[idx]: round(float(proba[idx]), 4)
            for idx in ranked_indices
        }

        needs_review = bool(top_conf < review_confidence_threshold)

        return {
            "predicted_intent": top_intent,
            "predicted_confidence": round(top_conf, 4),
            "needs_human_review": needs_review,
            "top_3_candidates": top_3,
            "all_probabilities": all_probs,
        }

    def retrieve_historical_evidence(
        self, text_clean: str, top_k: int = 3, min_similarity: float = 0.30
    ) -> Dict[str, Any]:
        """
        Computes cosine similarity against indexed unique historical customer inquiries.
        """
        if not text_clean:
            return {
                "best_similarity_score": 0.0,
                "lexical_similarity_band": "insufficient_lexical_evidence",
                "has_sufficient_historical_evidence": False,
                "retrieval_warning": "empty_query",
                "candidates": [],
            }

        Q = self.retrieval_vectorizer.transform([text_clean])
        if Q.nnz == 0:
            return {
                "best_similarity_score": 0.0,
                "lexical_similarity_band": "insufficient_lexical_evidence",
                "has_sufficient_historical_evidence": False,
                "retrieval_warning": "no_known_vocabulary",
                "candidates": [],
            }

        sim_scores = Q.dot(self.retrieval_matrix.T).toarray()[0]
        ranked_indices = np.argsort(sim_scores)[::-1]

        candidates = []
        for idx in ranked_indices[:top_k]:
            score = float(sim_scores[idx])
            meta_row = self.retrieval_meta.iloc[idx]
            band = get_lexical_similarity_band(score)
            has_evidence = bool(score >= min_similarity)

            candidates.append({
                "rank": len(candidates) + 1,
                "similarity_score": round(score, 4),
                "lexical_similarity_band": band,
                "has_sufficient_historical_evidence": has_evidence,
                "retrieved_customer_tweet_id": str(meta_row.get("customer_tweet_id", "")),
                "retrieved_customer_text_clean": sanitize_text(str(meta_row.get("customer_text_clean", ""))),
                "retrieved_brand_tweet_id": str(meta_row.get("brand_tweet_id", "")),
                "retrieved_brand_reply_clean": sanitize_text(str(meta_row.get("brand_reply_clean", ""))),
            })

        best_score = float(candidates[0]["similarity_score"]) if candidates else 0.0
        best_band = get_lexical_similarity_band(best_score)
        has_sufficient = bool(best_score >= min_similarity)

        return {
            "best_similarity_score": round(best_score, 4),
            "lexical_similarity_band": best_band,
            "has_sufficient_historical_evidence": has_sufficient,
            "retrieval_warning": None,
            "candidates": candidates,
        }

    def process_inquiry(
        self,
        text: str,
        tweet_id: str = "LIVE_QUERY",
        top_k: int = 3,
        confidence_threshold: float = 0.80,
        similarity_threshold: float = 0.50,
        adaptation_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Executes the full end-to-end support agent pipeline:
          1. Normalizes input text
          2. Classifies intent & confidence (Phase 6)
          3. Retrieves historical evidence (Phase 7)
          4. Drafts safe grounded reply (Phase 8)
          5. Evaluates risk-aware routing policy (Phase 9)
        """
        clean_text = normalize_customer_input(text)

        # Stage 1 & 2: Intent Classification
        intent_res = self.classify_intent(clean_text)
        pred_intent = intent_res["predicted_intent"]
        pred_conf = intent_res["predicted_confidence"]
        needs_review = intent_res["needs_human_review"]

        # Stage 3: Historical Dialogue Retrieval
        retrieval_res = self.retrieve_historical_evidence(clean_text, top_k=top_k)
        best_sim = retrieval_res["best_similarity_score"]
        lexical_band = retrieval_res["lexical_similarity_band"]
        has_evidence = retrieval_res["has_sufficient_historical_evidence"]
        candidates = retrieval_res["candidates"]

        # Stage 4: Safe Reply Drafting
        draft_res = draft_reply_for_inquiry(
            customer_tweet_id=tweet_id,
            customer_text_clean=clean_text,
            predicted_intent=pred_intent,
            predicted_confidence=pred_conf,
            needs_human_review=needs_review,
            best_similarity_score=best_sim,
            lexical_similarity_band=lexical_band,
            has_sufficient_historical_evidence=has_evidence,
            retrieval_candidates=candidates,
            similarity_adaptation_threshold=adaptation_threshold,
        )

        # Stage 5: Risk-Aware Escalation Routing
        inquiry_dict = {
            "predicted_intent": pred_intent,
            "predicted_confidence": pred_conf,
            "best_similarity_score": best_sim,
            "draft_mode": draft_res["draft_mode"],
            "restricted_draft": draft_res["restricted_draft"],
            "needs_human_review": needs_review,
            "safety_flags": draft_res["safety_flags"],
        }

        routing_res = evaluate_routing_decision(
            inquiry=inquiry_dict,
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold,
        )

        return {
            "query": {
                "raw_text": text,
                "clean_text": clean_text,
                "tweet_id": tweet_id,
            },
            "intent": intent_res,
            "retrieval": retrieval_res,
            "reply_draft": draft_res,
            "routing": routing_res,
            "parameters": {
                "confidence_threshold": confidence_threshold,
                "similarity_threshold": similarity_threshold,
                "adaptation_threshold": adaptation_threshold,
            },
        }
