#!/usr/bin/env python3
"""
Run Complete Support Agent Inference on Golden Inquiries (Phase 10)
===================================================================

Purpose:
  Executes blind, end-to-end agent inference on the protected 200-row golden set.
  Combines Phase 6 intent classification, Phase 7 train-only retrieval, Phase 8
  reply drafting, and Phase 9 routing policy.

Strict Anti-Leakage & Anti-Tamper Guard:
  This script REFUSES to run unless:
    1. data/golden/golden_labels_freeze_manifest.json exists.
    2. Manifest row count is exactly 200.
    3. The SHA-256 of data/golden/adjudication_sheet.csv matches the manifest exactly.
    4. All 200 rows in adjudication_sheet.csv have final_status == 'DONE'.
  This guarantees that human golden labels cannot be altered after seeing predictions.

Inference Pipeline:
  - Input: Only golden_id, customer_tweet_id, customer_text_clean.
  - Final human labels are strictly NEVER read or used during inference.
  - Output: outputs/final_evaluation/golden_agent_predictions.csv.
  - Every row marked decision_status = 'offline_candidate_not_sent'.

Usage:
  python scripts/run_golden_agent_inference.py
"""

import sys
import os
import json
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib

# Ensure Windows stdout handles UTF-8 gracefully
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reply_safety import detect_safety_flags, is_restricted
from src.reply_drafter import draft_reply_for_inquiry, sanitize_text
from src.routing_policy import evaluate_routing_decision, DECISION_STATUS_OFFLINE


def compute_sha256(filepath: Path) -> str:
    """Computes standard SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_golden_freeze_guard(
    manifest_path: Path,
    adjudication_path: Path,
) -> Tuple[bool, str]:
    """
    Validates that golden human labels are frozen, completed, and untampered.
    Returns (is_valid, explanation_message).
    """
    if not manifest_path.exists():
        return False, (
            f"FREEZE GUARD ERROR: Freeze manifest not found at {manifest_path}.\n"
            f"Human golden labels must be completed, reconciled, and frozen via `scripts/freeze_golden_labels.py` "
            f"before golden inference is permitted."
        )

    if not adjudication_path.exists():
        return False, f"Adjudication sheet not found at {adjudication_path}."

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return False, f"Failed to parse freeze manifest JSON: {e}"

    # 1. Manifest row count check
    if manifest.get("golden_row_count") != 200:
        return False, f"Manifest golden_row_count is {manifest.get('golden_row_count')}, expected 200."

    # 2. SHA-256 Checksum check
    current_sha256 = compute_sha256(adjudication_path)
    expected_sha256 = manifest.get("adjudication_sha256")
    if current_sha256 != expected_sha256:
        return False, (
            f"ANTI-TAMPER CHECKSUM MISMATCH!\n"
            f"Adjudication sheet SHA-256 ({current_sha256}) does not match frozen manifest SHA-256 ({expected_sha256}).\n"
            f"Human labels have been modified since the manifest was created."
        )

    # 3. Adjudication status check
    df_adj = pd.read_csv(adjudication_path, low_memory=False)
    if len(df_adj) != 200:
        return False, f"Adjudication sheet row count ({len(df_adj)}) is not 200."

    status_counts = df_adj["final_status"].fillna("MISSING").value_counts().to_dict()
    if status_counts.get("DONE", 0) != 200:
        return False, f"Not all golden rows have final_status == 'DONE'. Current: {status_counts}."

    return True, "Freeze guard verified: Golden human labels are frozen and valid."


def run_agent_inference_on_dataframe(
    df_queries: pd.DataFrame,
    models_dir: Path,
    top_k: int = 3,
    confidence_threshold: float = 0.80,
    similarity_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Executes blind end-to-end agent inference over an input inquiry dataframe.
    """
    # 1. Load Intent Classifier (Phase 6)
    vec_path = models_dir / "tfidf_vectorizer.joblib"
    clf_path = models_dir / "tfidf_logistic_regression.joblib"
    intent_vectorizer = joblib.load(vec_path)
    intent_classifier = joblib.load(clf_path)
    intent_classes = list(intent_classifier.classes_)

    # 2. Load Retrieval Index (Phase 7 - Train-only)
    ret_vec_path = models_dir / "retrieval_tfidf_vectorizer.joblib"
    ret_mat_path = models_dir / "retrieval_customer_matrix.joblib"
    ret_meta_path = models_dir / "retrieval_metadata.joblib"
    ret_vectorizer = joblib.load(ret_vec_path)
    ret_train_matrix = joblib.load(ret_mat_path)
    ret_meta_df = joblib.load(ret_meta_path)

    # Pre-index for thread exclusion
    cust_id_to_indices = defaultdict(list)
    group_id_to_indices = defaultdict(list)
    for idx, row in ret_meta_df.iterrows():
        cid = row.get("customer_tweet_id")
        gid = row.get("conversation_root_or_group_id")
        if cid and str(cid) not in ("nan", ""):
            cust_id_to_indices[str(cid)].append(idx)
        if gid and str(gid) not in ("nan", ""):
            group_id_to_indices[str(gid)].append(idx)

    # 3. Classify Intents
    in_texts = df_queries["customer_text_clean"].fillna("").astype(str).tolist()
    X_intent = intent_vectorizer.transform(in_texts)
    intent_probs = intent_classifier.predict_proba(X_intent)

    # 4. Retrieval & Drafting
    Q_queries = ret_vectorizer.transform(in_texts)
    sim_matrix = Q_queries.dot(ret_train_matrix.T).toarray()

    results = []

    for i, (_, q_row) in enumerate(df_queries.iterrows()):
        q_cust_id = str(q_row.get("customer_tweet_id", ""))
        q_text_raw = str(q_row.get("customer_text_clean", ""))
        q_group_id = str(q_row.get("conversation_root_or_group_id", ""))
        golden_id = str(q_row.get("golden_id", f"sample_{i+1:03d}"))

        # Intent predictions
        probs = intent_probs[i]
        top_idx = int(np.argmax(probs))
        pred_intent = intent_classes[top_idx]
        pred_conf = float(probs[top_idx])
        needs_review = bool(pred_conf < 0.60)

        # Retrieval candidates
        q_vec = Q_queries[i]
        has_vocab = bool(q_vec.nnz > 0)
        candidates = []
        best_sim = 0.0

        if has_vocab:
            scores = sim_matrix[i]
            if q_cust_id in cust_id_to_indices:
                for excl_idx in cust_id_to_indices[q_cust_id]:
                    scores[excl_idx] = -1.0
            if q_group_id in group_id_to_indices:
                for excl_idx in group_id_to_indices[q_group_id]:
                    scores[excl_idx] = -1.0

            num_cand = min(top_k * 4, len(scores))
            top_part = np.argpartition(scores, -num_cand)[-num_cand:]
            top_sorted = top_part[np.argsort(scores[top_part])[::-1]]

            seen_cust = set()
            for c_idx in top_sorted:
                s = float(scores[c_idx])
                if s < 0.0:
                    continue
                c_row = ret_meta_df.iloc[c_idx]
                cid = str(c_row["customer_tweet_id"])
                if cid in seen_cust:
                    continue
                seen_cust.add(cid)
                candidates.append({
                    "retrieved_customer_tweet_id": cid,
                    "retrieved_brand_tweet_id": str(c_row.get("brand_tweet_id", "")),
                    "retrieved_brand_reply_clean": str(c_row.get("brand_reply_clean", "")),
                    "similarity_score": s,
                })
                if len(candidates) == top_k:
                    break

            if candidates:
                best_sim = candidates[0]["similarity_score"]

        # Lexical similarity band
        if best_sim < 0.30:
            band = "insufficient_lexical_evidence"
        elif best_sim < 0.50:
            band = "weak_lexical_evidence"
        elif best_sim < 0.70:
            band = "moderate_lexical_evidence"
        else:
            band = "high_lexical_similarity"

        has_sufficient = bool(best_sim >= 0.30)

        # Phase 8: Draft reply
        draft_dict = draft_reply_for_inquiry(
            customer_tweet_id=q_cust_id,
            customer_text_clean=q_text_raw,
            predicted_intent=pred_intent,
            predicted_confidence=pred_conf,
            needs_human_review=needs_review,
            best_similarity_score=best_sim,
            lexical_similarity_band=band,
            has_sufficient_historical_evidence=has_sufficient,
            retrieval_candidates=candidates,
            similarity_adaptation_threshold=similarity_threshold,
        )

        # Phase 9: Routing policy
        routing_dict = evaluate_routing_decision(
            inquiry=draft_dict,
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold,
        )

        results.append({
            "golden_id": golden_id,
            "customer_tweet_id": q_cust_id,
            "customer_text_clean": draft_dict["customer_text_clean"],
            "predicted_intent": draft_dict["predicted_intent"],
            "predicted_confidence": draft_dict["predicted_confidence"],
            "draft_reply": draft_dict["draft_reply"],
            "draft_mode": draft_dict["draft_mode"],
            "restricted_draft": draft_dict["restricted_draft"],
            "safety_flags": draft_dict["safety_flags"],
            "best_similarity_score": draft_dict["best_similarity_score"],
            "lexical_similarity_band": draft_dict["lexical_similarity_band"],
            "evidence_customer_tweet_ids": draft_dict["evidence_customer_tweet_ids"],
            "evidence_brand_tweet_ids": draft_dict["evidence_brand_tweet_ids"],
            "agent_action": routing_dict["action"],
            "primary_reason_code": routing_dict["primary_reason_code"],
            "all_reason_codes": routing_dict["all_reason_codes"],
            "decision_explanation": routing_dict["decision_explanation"],
            "decision_status": routing_dict["decision_status"],
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Run Support Agent Inference on Golden Set (Phase 10).")
    parser.add_argument("--golden-input", type=Path, default=REPO_ROOT / "data" / "golden" / "golden_set_200.csv")
    parser.add_argument("--adjudication-file", type=Path, default=REPO_ROOT / "data" / "golden" / "adjudication_sheet.csv")
    parser.add_argument("--manifest-file", type=Path, default=REPO_ROOT / "data" / "golden" / "golden_labels_freeze_manifest.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "final_evaluation" / "golden_agent_predictions.csv")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--skip-freeze-guard-for-testing", action="store_true", help="Internal flag strictly for non-golden test fixtures.")
    args = parser.parse_args()

    print("=" * 75)
    print("PHASE 10: GOLDEN BENCHMARK AGENT INFERENCE")
    print("=" * 75)
    print(f"Golden Input     : {args.golden_input}")
    print(f"Adjudication File: {args.adjudication_file}")
    print(f"Freeze Manifest  : {args.manifest_file}")
    print(f"Output File      : {args.output}")
    print("=" * 75)

    # Enforce Freeze Guard
    if not args.skip_freeze_guard_for_testing:
        guard_ok, guard_msg = verify_golden_freeze_guard(args.manifest_file, args.adjudication_file)
        if not guard_ok:
            print("\n[EXECUTION BLOCKED BY GOLDEN FREEZE GUARD]")
            print(guard_msg)
            print("\nIntegrity Safeguard: Golden inference cannot proceed until human labels are completed and frozen.")
            sys.exit(1)
        else:
            print(f"\n[Guard Verified] {guard_msg}")
    else:
        print("\n[WARNING] Freeze guard bypassed for internal test fixture. MUST NOT BE USED FOR PRODUCTION GOLDEN INFERENCE.")

    if not args.golden_input.exists():
        raise FileNotFoundError(f"Golden input file not found: {args.golden_input}")

    df_gold = pd.read_csv(args.golden_input, low_memory=False)
    print(f"\n[DataLoader] Loaded {len(df_gold)} golden inquiry queries.")

    # Execute blind inference
    df_preds = run_agent_inference_on_dataframe(
        df_queries=df_gold,
        models_dir=args.models_dir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\n[Saving] Successfully saved {len(df_preds)} golden predictions to: {args.output}")

    # Summary
    print("\nGolden Inference Summary:")
    print(df_preds["agent_action"].value_counts().to_string())
    print("\nPrimary Reasons:")
    print(df_preds["primary_reason_code"].value_counts().to_string())
    print("\nGolden inference finished successfully.")


if __name__ == "__main__":
    main()
