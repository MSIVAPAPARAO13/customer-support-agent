#!/usr/bin/env python3
"""
Retrieve Similar Historical AppleSupport Dialogues (Phase 7 - Improved)
======================================================================

Purpose:
  For each incoming customer support inquiry, finds the top-k most similar
  unique historical customer inquiries from the train-only TF-IDF index, returning
  the actual sanitized AppleSupport replies as grounding evidence.

Key Enhancements (Phase 7 Fixes):
  1. UNIQUE INQUIRY RETRIEVAL:
     Retrieves from the deduplicated customer inquiry index (82,063 unique inquiries).
     Guarantees distinct `retrieved_customer_tweet_id` values in top-k results.
  2. CONSERVATIVE EVIDENCE THRESHOLD (0.30):
     Replaces initial 0.15 threshold with conservative 0.30.
  3. LEXICAL SIMILARITY BANDS:
     - similarity < 0.30  -> insufficient_lexical_evidence
     - 0.30 <= sim < 0.50 -> weak_lexical_evidence
     - 0.50 <= sim < 0.70 -> moderate_lexical_evidence
     - similarity >= 0.70 -> high_lexical_similarity
     (Note: high_lexical_similarity does NOT mean safe grounding or safe auto-handling.)
  4. UNKNOWN VOCABULARY SAFEGUARD:
     If a query contains zero recognized vocabulary words, assigns:
     similarity_score = 0.0, band = "insufficient_lexical_evidence",
     has_sufficient_historical_evidence = False, retrieval_warning = "no_known_vocabulary".
  5. SANITIZATION:
     Sanitizes all text (@mentions -> [USER], URLs -> [URL]).
  6. REVIEW SIGNAL:
     Sets `needs_human_relevance_review = True`.
  7. STRICT TEXT EXCLUSION OPTION:
     Provides `--strict-text-exclusion` flag to exclude candidates sharing
     exact normalized wording with the query for cross-conversation diagnostics.

Usage:
  python scripts/retrieve_similar_dialogues.py `
    --input data/processed/applesupport_validation.csv `
    --output outputs/retrieval/validation_retrieval_examples.csv `
    --top-k 3
"""

import sys
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import joblib

# Ensure Windows console stdout handles UTF-8 gracefully
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent


def sanitize_text(text: str) -> str:
    """Sanitizes text: replaces @mentions with [USER] and standardizes URLs to [URL]."""
    if not text or pd.isna(text):
        return ""
    t = re.sub(r'https?://\S+|www\.\S+', '[URL]', str(text))
    t = re.sub(r'@\w+', '[USER]', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def normalize_for_text_exclusion(text: str) -> str:
    """
    Conservative normalization for strict text exclusion:
      - lowercase
      - replace URLs
      - remove leading brand mentions
      - normalize whitespace
    """
    if not text or pd.isna(text):
        return ""
    t = str(text).lower().strip()
    t = re.sub(r'https?://\S+|www\.\S+', '[url]', t)
    # Remove leading brand handles like @applesupport or @115858
    t = re.sub(r'^(?:@\w+\s*)+', '', t)
    t = re.sub(r'@\w+', '[user]', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


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


def load_retrieval_artifacts(models_dir: Path) -> Tuple[Any, Any, pd.DataFrame]:
    """Loads vectorizer, customer sparse matrix, and metadata table."""
    vec_path = models_dir / "retrieval_tfidf_vectorizer.joblib"
    mat_path = models_dir / "retrieval_customer_matrix.joblib"
    meta_path = models_dir / "retrieval_metadata.joblib"

    for p in [vec_path, mat_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required retrieval index file: {p}. Run build_retrieval_index.py first.")

    print(f"[Index Loader] Loading TF-IDF vectorizer from {vec_path.name}...", flush=True)
    vectorizer = joblib.load(vec_path)
    print(f"[Index Loader] Loading sparse matrix from {mat_path.name}...", flush=True)
    matrix = joblib.load(mat_path)
    print(f"[Index Loader] Loading metadata from {meta_path.name}...", flush=True)
    meta_df = joblib.load(meta_path)

    return vectorizer, matrix, meta_df


def retrieve_batch(
    query_df: pd.DataFrame,
    vectorizer: Any,
    train_matrix: Any,
    meta_df: pd.DataFrame,
    top_k: int = 3,
    min_similarity: float = 0.30,
    strict_text_exclusion: bool = False,
    chunk_size: int = 1000
) -> pd.DataFrame:
    """
    Computes batched cosine similarity between query inquiries and indexed train matrix.
    Applies thread exclusion and optional strict text exclusion, returning top-k results.
    """
    total_queries = len(query_df)
    results = []

    # Pre-index train metadata for O(1) exclusion lookups
    print("[Index] Pre-indexing customer IDs, group IDs, and text hashes for O(1) exclusion...", flush=True)
    cust_id_to_indices = defaultdict(list)
    group_id_to_indices = defaultdict(list)
    text_to_indices = defaultdict(list)

    for idx, row in meta_df.iterrows():
        cid = row.get('customer_tweet_id')
        gid = row.get('conversation_root_or_group_id')
        if cid and str(cid) not in ('nan', ''):
            cust_id_to_indices[str(cid)].append(idx)
        if gid and str(gid) not in ('nan', ''):
            group_id_to_indices[str(gid)].append(idx)
        if strict_text_exclusion:
            norm_t = normalize_for_text_exclusion(row.get('customer_text_clean', ''))
            if norm_t:
                text_to_indices[norm_t].append(idx)

    mode_desc = "STRICT TEXT EXCLUSION" if strict_text_exclusion else "STANDARD EXCLUSION"
    print(f"\n[Retrieval Engine] Querying {total_queries:,} inputs [{mode_desc}] (top_k={top_k}, threshold={min_similarity:.2f})...", flush=True)

    for start_idx in range(0, total_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, total_queries)
        chunk = query_df.iloc[start_idx:end_idx]

        q_texts = chunk['customer_text_clean'].fillna('').astype(str).tolist()
        Q_chunk = vectorizer.transform(q_texts)

        # Batch matrix multiplication: Q_chunk is (C, 50000), train_matrix is (82063, 50000)
        # Because norm='l2', dot product is exact cosine similarity.
        sim_matrix = Q_chunk.dot(train_matrix.T).toarray()

        for local_i, (_, q_row) in enumerate(chunk.iterrows()):
            q_cust_id = str(q_row.get('customer_tweet_id', ''))
            q_text_raw = str(q_row.get('customer_text_clean', ''))
            q_text = sanitize_text(q_text_raw)
            q_group_id = str(q_row.get('conversation_root_or_group_id', ''))

            # Check if query has zero non-zero features in vectorizer
            q_vec = Q_chunk[local_i]
            has_vocab = (q_vec.nnz > 0)

            if not has_vocab:
                # Unknown vocabulary handling (Problem 3)
                for rank in range(1, top_k + 1):
                    results.append({
                        'query_customer_tweet_id': q_cust_id,
                        'query_customer_text_clean': q_text,
                        'query_conversation_group_id': q_group_id,
                        'rank': rank,
                        'similarity_score': 0.0,
                        'best_similarity_score': 0.0,
                        'lexical_similarity_band': "insufficient_lexical_evidence",
                        'has_sufficient_historical_evidence': False,
                        'retrieval_warning': "no_known_vocabulary",
                        'needs_human_relevance_review': True,
                        'retrieved_customer_tweet_id': "NONE",
                        'retrieved_brand_tweet_id': "NONE",
                        'retrieved_conversation_group_id': "NONE",
                        'retrieved_customer_text_clean': "No historical match found for unknown vocabulary query.",
                        'retrieved_brand_reply_clean': "NONE",
                        'retrieval_scope': 'train_only'
                    })
                continue

            scores = sim_matrix[local_i]

            # 1. Thread Exclusion: same tweet ID or same conversation group
            if q_cust_id in cust_id_to_indices:
                for excl_idx in cust_id_to_indices[q_cust_id]:
                    scores[excl_idx] = -1.0
            if q_group_id in group_id_to_indices:
                for excl_idx in group_id_to_indices[q_group_id]:
                    scores[excl_idx] = -1.0

            # 2. Strict Text Exclusion (Problem 4 diagnostic)
            if strict_text_exclusion:
                norm_q = normalize_for_text_exclusion(q_text_raw)
                if norm_q in text_to_indices:
                    for excl_idx in text_to_indices[norm_q]:
                        scores[excl_idx] = -1.0

            # Find top candidates using argpartition
            num_candidates = min(top_k * 4, len(scores))
            top_part = np.argpartition(scores, -num_candidates)[-num_candidates:]
            top_sorted = top_part[np.argsort(scores[top_part])[::-1]]

            valid_ranks = []
            seen_cust_ids = set()

            for candidate_idx in top_sorted:
                score = float(scores[candidate_idx])
                if score < 0.0:  # Excluded
                    continue
                cand_row = meta_df.iloc[candidate_idx]
                cid = str(cand_row['customer_tweet_id'])
                if cid in seen_cust_ids:
                    continue
                seen_cust_ids.add(cid)
                valid_ranks.append((candidate_idx, score))
                if len(valid_ranks) == top_k:
                    break

            best_score = valid_ranks[0][1] if valid_ranks else 0.0
            has_sufficient = bool(best_score >= min_similarity)
            band = get_lexical_similarity_band(best_score)
            warning_msg = "None" if has_sufficient else f"Best similarity score ({best_score:.4f}) below {min_similarity:.2f} threshold; weak or insufficient lexical evidence."

            for rank, (cand_idx, score) in enumerate(valid_ranks, start=1):
                cand_row = meta_df.iloc[cand_idx]
                results.append({
                    'query_customer_tweet_id': q_cust_id,
                    'query_customer_text_clean': q_text,
                    'query_conversation_group_id': q_group_id,
                    'rank': rank,
                    'similarity_score': round(score, 4),
                    'best_similarity_score': round(best_score, 4),
                    'lexical_similarity_band': band,
                    'has_sufficient_historical_evidence': has_sufficient,
                    'retrieval_warning': warning_msg,
                    'needs_human_relevance_review': True,
                    'retrieved_customer_tweet_id': str(cand_row['customer_tweet_id']),
                    'retrieved_brand_tweet_id': str(cand_row['brand_tweet_id']),
                    'retrieved_conversation_group_id': str(cand_row['conversation_root_or_group_id']),
                    'retrieved_customer_text_clean': sanitize_text(str(cand_row['customer_text_clean'])),
                    'retrieved_brand_reply_clean': sanitize_text(str(cand_row['brand_reply_clean'])),
                    'retrieval_scope': 'train_only'
                })

        print(f"  Processed {end_idx:,} / {total_queries:,} queries...", flush=True)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Retrieve Similar Historical AppleSupport Dialogues.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV containing customer inquiries (e.g. data/processed/applesupport_validation.csv)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for retrieved dialogue pairs"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of nearest historical dialogues to retrieve (default: 3)"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.30,
        help="Minimum similarity threshold for sufficient historical evidence (default: 0.30)"
    )
    parser.add_argument(
        "--strict-text-exclusion",
        action="store_true",
        help="Exclude candidate historical rows with identical normalized customer text as query"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory containing retrieval index artifacts"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional row limit for quick evaluation"
    )

    args = parser.parse_args()

    print("=" * 78)
    print("      PHASE 7: HISTORICAL DIALOGUE RETRIEVAL (CONSERVATIVE EVIDENCE)")
    print("=" * 78)

    # 1. Load artifacts
    vectorizer, matrix, meta_df = load_retrieval_artifacts(args.models_dir)

    # 2. Load input queries
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"\n[Data Loader] Loading queries from: {args.input.name}...", flush=True)
    df_queries = pd.read_csv(args.input, dtype=str)
    print(f"  - Loaded {len(df_queries):,} queries.", flush=True)

    if args.sample is not None and args.sample > 0 and args.sample < len(df_queries):
        print(f"  - Subsampling to {args.sample} queries as requested...", flush=True)
        df_queries = df_queries.head(args.sample)

    # 3. Retrieve
    df_retrieved = retrieve_batch(
        query_df=df_queries,
        vectorizer=vectorizer,
        train_matrix=matrix,
        meta_df=meta_df,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        strict_text_exclusion=args.strict_text_exclusion
    )

    # 4. Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_retrieved.to_csv(args.output, index=False)
    print(f"\n[Output] Saved {len(df_retrieved):,} retrieved rows to: {args.output}", flush=True)

    # 5. Diagnostic summary
    best_ranks = df_retrieved[df_retrieved['rank'] == 1]
    sufficient_count = best_ranks['has_sufficient_historical_evidence'].sum()
    total_q = len(best_ranks)
    pct_sufficient = sufficient_count / total_q if total_q > 0 else 0

    print(f"\n[Summary Statistics - Threshold {args.min_similarity:.2f}]")
    print(f"  - Evaluated queries: {total_q:,}")
    print(f"  - Queries with sufficient evidence (score >= {args.min_similarity:.2f}): {sufficient_count:,} ({pct_sufficient:.2%})")
    print(f"  - Queries with insufficient evidence (score < {args.min_similarity:.2f}): {total_q - sufficient_count:,} ({1.0 - pct_sufficient:.2%})")
    print(f"  - Mean best similarity score: {best_ranks['similarity_score'].mean():.4f}")
    print(f"  - Median best similarity score: {best_ranks['similarity_score'].median():.4f}")

    print("\n[Lexical Similarity Band Distribution]")
    for band, count in best_ranks['lexical_similarity_band'].value_counts().items():
        print(f"  * {band:<32}: {count:>6,} ({count/total_q:>6.2%})")

    print("\n" + "=" * 78)
    print("RETRIEVAL RUN COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
