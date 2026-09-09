#!/usr/bin/env python3
"""
Draft Grounded Customer Support Replies (Phase 8)
=================================================

Purpose:
  Combines Phase 6 intent classification and Phase 7 historical dialogue retrieval
  with deterministic safety rules to produce grounded, safe, and auditable draft
  replies for incoming customer inquiries.

Pipeline:
  Customer message
  + predicted intent and confidence (Phase 6 TF-IDF + Logistic Regression)
  + retrieved historical AppleSupport response patterns (Phase 7 train-only index)
  + deterministic safety restrictions (Phase 8 reply_safety)
          ↓
  Safe draft reply with evidence IDs and grounding notes

Strict Partition Rules:
  - Retrieval index and classifier are train-only.
  - Development drafts are generated exclusively on applesupport_validation.csv.
  - Never load applesupport_test.csv or golden_set_200.csv.

Usage:
  python scripts/draft_grounded_replies.py `
    --input data/processed/applesupport_validation.csv `
    --output outputs/replies/validation_drafted_replies.csv `
    --top-k 3
"""

import sys
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib

# Ensure repo root is on sys.path so src can be imported cleanly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reply_safety import detect_safety_flags, is_restricted
from src.reply_drafter import draft_reply_for_inquiry, sanitize_text

# Configure Windows console stdout
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def get_lexical_similarity_band(score: float) -> str:
    """Assigns standardized lexical similarity band."""
    if score < 0.30:
        return "insufficient_lexical_evidence"
    elif score < 0.50:
        return "weak_lexical_evidence"
    elif score < 0.70:
        return "moderate_lexical_evidence"
    else:
        return "high_lexical_similarity"


def check_input_isolation(input_path: Path) -> None:
    """Strict check preventing accidental loading of test or golden partitions."""
    name = input_path.name.lower()
    if "test" in name or "golden" in name or "blind" in name:
        raise ValueError(
            f"DATA PROTECTION ERROR: Prohibited input file '{input_path.name}'. "
            f"Drafting in Phase 8 is permitted only on validation data."
        )


def main():
    parser = argparse.ArgumentParser(description="Draft Grounded Customer Support Replies (Phase 8).")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "applesupport_validation.csv",
        help="Path to input customer inquiries CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "replies" / "validation_drafted_replies.csv",
        help="Path to save output drafted replies CSV.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory containing trained model and index artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of historical replies to retrieve per inquiry.",
    )
    parser.add_argument(
        "--adaptation-threshold",
        type=float,
        default=0.50,
        help="Similarity threshold required to adapt draft with historical pattern (default: 0.50).",
    )
    parser.add_argument(
        "--evidence-threshold",
        type=float,
        default=0.30,
        help="Similarity threshold for has_sufficient_historical_evidence flag (default: 0.30).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for batch retrieval computation.",
    )

    args = parser.parse_args()

    check_input_isolation(args.input)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 8: SAFE, HISTORICALLY GROUNDED REPLY DRAFTING")
    print("=" * 75)
    print(f"Input Query Dataset   : {args.input}")
    print(f"Output Draft File     : {args.output}")
    print(f"Models Directory      : {args.models_dir}")
    print(f"Top-K Retrieval       : {args.top_k}")
    print(f"Adaptation Threshold  : {args.adaptation_threshold:.2f} (>= 0.50 for historical pattern)")
    print(f"Evidence Threshold    : {args.evidence_threshold:.2f} (>= 0.30 for sufficient evidence)")
    print("=" * 75)

    # 1. Load Intent Classifier (Phase 6)
    print("\n[1/4] Loading Intent Classification Models (Phase 6)...", flush=True)
    clf_vec_path = args.models_dir / "tfidf_vectorizer.joblib"
    clf_model_path = args.models_dir / "tfidf_logistic_regression.joblib"
    for p in [clf_vec_path, clf_model_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing classifier artifact: {p}. Run Phase 6 first.")

    intent_vectorizer = joblib.load(clf_vec_path)
    intent_classifier = joblib.load(clf_model_path)
    intent_classes = list(intent_classifier.classes_)
    print(f"  Loaded classifier with {len(intent_classes)} intent classes.")

    # 2. Load Retrieval Index (Phase 7)
    print("\n[2/4] Loading Retrieval Index Artifacts (Phase 7)...", flush=True)
    ret_vec_path = args.models_dir / "retrieval_tfidf_vectorizer.joblib"
    ret_mat_path = args.models_dir / "retrieval_customer_matrix.joblib"
    ret_meta_path = args.models_dir / "retrieval_metadata.joblib"
    for p in [ret_vec_path, ret_mat_path, ret_meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing retrieval artifact: {p}. Run Phase 7 first.")

    ret_vectorizer = joblib.load(ret_vec_path)
    ret_train_matrix = joblib.load(ret_mat_path)
    ret_meta_df = joblib.load(ret_meta_path)
    print(f"  Loaded retrieval index: {ret_train_matrix.shape[0]:,} documents across {ret_train_matrix.shape[1]:,} features.")

    # Pre-index for O(1) thread exclusion
    cust_id_to_indices = defaultdict(list)
    group_id_to_indices = defaultdict(list)
    for idx, row in ret_meta_df.iterrows():
        cid = row.get('customer_tweet_id')
        gid = row.get('conversation_root_or_group_id')
        if cid and str(cid) not in ('nan', ''):
            cust_id_to_indices[str(cid)].append(idx)
        if gid and str(gid) not in ('nan', ''):
            group_id_to_indices[str(gid)].append(idx)

    # 3. Load Inquiries
    print(f"\n[3/4] Reading input inquiries from {args.input.name}...", flush=True)
    df_in = pd.read_csv(args.input, low_memory=False)
    total_inquiries = len(df_in)
    print(f"  Loaded {total_inquiries:,} customer inquiries.")

    # Run Intent Classification
    print("  Classifying intents and confidence scores...", flush=True)
    in_texts = df_in['customer_text_clean'].fillna('').astype(str).tolist()
    X_intent = intent_vectorizer.transform(in_texts)
    intent_probs = intent_classifier.predict_proba(X_intent)

    predicted_intents = []
    predicted_confidences = []
    needs_human_reviews = []

    for i in range(total_inquiries):
        probs = intent_probs[i]
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        predicted_intents.append(intent_classes[top_idx])
        predicted_confidences.append(top_conf)
        needs_human_reviews.append(bool(top_conf < 0.60))

    # 4. Batch Retrieval & Reply Drafting
    print(f"\n[4/4] Executing batch retrieval and reply drafting in chunks of {args.chunk_size}...", flush=True)
    draft_results = []

    for start_idx in range(0, total_inquiries, args.chunk_size):
        end_idx = min(start_idx + args.chunk_size, total_inquiries)
        chunk_df = df_in.iloc[start_idx:end_idx]
        chunk_texts = in_texts[start_idx:end_idx]

        # Transform chunk for retrieval
        Q_chunk = ret_vectorizer.transform(chunk_texts)
        # Dot product with train_matrix (norm="l2" assures cosine similarity)
        sim_matrix = Q_chunk.dot(ret_train_matrix.T).toarray()

        for local_i, (_, row) in enumerate(chunk_df.iterrows()):
            global_i = start_idx + local_i
            q_cust_id = str(row.get('customer_tweet_id', ''))
            q_text_raw = str(row.get('customer_text_clean', ''))
            q_group_id = str(row.get('conversation_root_or_group_id', ''))

            pred_intent = predicted_intents[global_i]
            pred_conf = predicted_confidences[global_i]
            needs_review = needs_human_reviews[global_i]

            # Retrieval candidates logic
            q_vec = Q_chunk[local_i]
            has_vocab = (q_vec.nnz > 0)

            candidates = []
            best_sim = 0.0

            if has_vocab:
                scores = sim_matrix[local_i]

                # Thread exclusion
                if q_cust_id in cust_id_to_indices:
                    for excl_idx in cust_id_to_indices[q_cust_id]:
                        scores[excl_idx] = -1.0
                if q_group_id in group_id_to_indices:
                    for excl_idx in group_id_to_indices[q_group_id]:
                        scores[excl_idx] = -1.0

                num_cand = min(args.top_k * 4, len(scores))
                top_part = np.argpartition(scores, -num_cand)[-num_cand:]
                top_sorted = top_part[np.argsort(scores[top_part])[::-1]]

                seen_cust = set()
                for c_idx in top_sorted:
                    s = float(scores[c_idx])
                    if s < 0.0:
                        continue
                    c_row = ret_meta_df.iloc[c_idx]
                    cid = str(c_row['customer_tweet_id'])
                    if cid in seen_cust:
                        continue
                    seen_cust.add(cid)
                    candidates.append({
                        "retrieved_customer_tweet_id": cid,
                        "retrieved_brand_tweet_id": str(c_row.get('brand_tweet_id', '')),
                        "retrieved_brand_reply_clean": str(c_row.get('brand_reply_clean', '')),
                        "similarity_score": s,
                    })
                    if len(candidates) == args.top_k:
                        break

                if candidates:
                    best_sim = candidates[0]["similarity_score"]

            band = get_lexical_similarity_band(best_sim)
            has_sufficient = bool(best_sim >= args.evidence_threshold)

            # Generate draft
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
                similarity_adaptation_threshold=args.adaptation_threshold,
            )
            draft_results.append(draft_dict)

        print(f"  Completed {end_idx:,} / {total_inquiries:,} drafts...", flush=True)

    df_out = pd.DataFrame(draft_results)

    # Ensure exact column ordering
    col_order = [
        "customer_tweet_id",
        "customer_text_clean",
        "predicted_intent",
        "predicted_confidence",
        "needs_human_review",
        "draft_reply",
        "draft_mode",
        "restricted_draft",
        "safety_flags",
        "evidence_customer_tweet_ids",
        "evidence_brand_tweet_ids",
        "best_similarity_score",
        "lexical_similarity_band",
        "has_sufficient_historical_evidence",
        "grounding_note",
    ]
    df_out = df_out[col_order]

    print(f"\n[Saving] Writing {len(df_out):,} drafted replies to {args.output}...", flush=True)
    df_out.to_csv(args.output, index=False, encoding='utf-8')

    # Summary Statistics
    print("\n" + "=" * 75)
    print("PHASE 8 DRAFTING SUMMARY REPORT")
    print("=" * 75)
    print(f"Total Inquiries Processed : {len(df_out):,}")
    print("\nDraft Mode Breakdown:")
    for mode, count in df_out['draft_mode'].value_counts().items():
        pct = (count / len(df_out)) * 100
        print(f"  {mode:<35} : {count:>6,} ({pct:5.2f}%)")

    print("\nRestricted Drafts:")
    restr_count = int(df_out['restricted_draft'].sum())
    restr_pct = (restr_count / len(df_out)) * 100
    print(f"  restricted_draft = True            : {restr_count:>6,} ({restr_pct:5.2f}%)")

    print("\nLexical Similarity Bands:")
    for band, count in df_out['lexical_similarity_band'].value_counts().items():
        pct = (count / len(df_out)) * 100
        print(f"  {band:<35} : {count:>6,} ({pct:5.2f}%)")

    print("\nTop Predicted Intents:")
    for intent, count in df_out['predicted_intent'].value_counts().items():
        pct = (count / len(df_out)) * 100
        print(f"  {intent:<35} : {count:>6,} ({pct:5.2f}%)")

    print("\nDraft generation completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
