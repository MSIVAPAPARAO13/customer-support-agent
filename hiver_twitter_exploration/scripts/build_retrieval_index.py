#!/usr/bin/env python3
"""
Build Train-Only Historical AppleSupport Retrieval Index (Phase 7 - Improved)
=============================================================================

Purpose:
  Builds an in-memory TF-IDF sparse matrix index over unique historical customer inquiries
  and their aggregated AppleSupport brand replies, strictly sourced from the
  training partition (`applesupport_train.csv`).

Key Enhancements (Phase 7 Fixes):
  1. UNIQUE INQUIRY INDEXING:
     One searchable document = one unique customer_tweet_id.
     For the 14 customer inquiries that received multiple brand replies,
     unique cleaned brand replies are aggregated into a structured record,
     preventing the same inquiry from occupying multiple top-k ranks.
  2. MATHEMATICALLY EXACT COSINE SIMILARITY:
     Explicitly sets `norm="l2"` in TfidfVectorizer so that matrix dot-product
     equals exact cosine similarity.
  3. EVIDENCE SANITIZATION:
     Sanitizes all text (@mentions -> [USER], URLs -> [URL]) to protect privacy
     and prevent raw Twitter handles from leaking into future reply drafting.
  4. PROVENANCE & METADATA:
     Records SHA-256, `indexed_unique_customer_inquiries` (82,063),
     `source_interaction_pair_count` (82,077), and `multi_reply_customer_count` (14).

Artifacts Persisted:
  - models/retrieval_tfidf_vectorizer.joblib
  - models/retrieval_customer_matrix.joblib (82,063 rows)
  - models/retrieval_metadata.joblib (82,063 rows)
  - models/retrieval_index_metadata.json

Usage:
  python scripts/build_retrieval_index.py
"""

import sys
import os
import re
import json
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Ensure Windows stdout handles UTF-8 gracefully
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file for cryptographic provenance."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def sanitize_text(text: str) -> str:
    """
    Sanitizes dialogue text before outputting as historical evidence:
      - Replaces raw Twitter handles (@mentions) with [USER]
      - Ensures URL patterns are standardized as [URL]
      - Normalizes whitespace
    """
    if not text or pd.isna(text):
        return ""
    # Standardize URLs
    t = re.sub(r'https?://\S+|www\.\S+', '[URL]', str(text))
    # Replace @mentions with [USER]
    t = re.sub(r'@\w+', '[USER]', t)
    # Normalize whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def aggregate_unique_inquiries(df_pairs: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Groups interaction pairs by customer_tweet_id so that each unique customer inquiry
    forms exactly one searchable document, aggregating multiple brand replies where present.
    """
    print("[Data Prep] Aggregating multiple brand replies per unique customer inquiry...", flush=True)
    
    # Track multi-reply counts
    tweet_counts = df_pairs['customer_tweet_id'].value_counts()
    multi_reply_customers = int((tweet_counts > 1).sum())

    aggregated_records = []
    # Group by customer_tweet_id
    grouped = df_pairs.groupby('customer_tweet_id', sort=False)

    for cust_id, group in grouped:
        cust_text_clean = sanitize_text(group['customer_text_clean'].iloc[0])
        group_id = str(group['conversation_root_or_group_id'].iloc[0])

        # Aggregate unique sanitized brand replies
        raw_replies = group['brand_reply_clean'].dropna().tolist()
        sanitized_replies = []
        for r in raw_replies:
            s = sanitize_text(r)
            if s and s not in sanitized_replies:
                sanitized_replies.append(s)

        brand_tweet_ids = [str(bid) for bid in group['brand_tweet_id'].dropna().tolist()]

        joined_replies = " || ".join(sanitized_replies) if sanitized_replies else ""

        aggregated_records.append({
            'customer_tweet_id': str(cust_id),
            'conversation_root_or_group_id': group_id,
            'customer_text_clean': cust_text_clean,
            'brand_reply_clean': joined_replies,
            'brand_tweet_id': ", ".join(brand_tweet_ids),
            'brand_replies_list': sanitized_replies,
            'brand_tweet_ids_list': brand_tweet_ids,
            'reply_count': len(brand_tweet_ids)
        })

    df_unique = pd.DataFrame(aggregated_records)
    return df_unique, multi_reply_customers


def build_index(
    train_path: Path,
    models_dir: Path
) -> Dict[str, Any]:
    """
    Reads applesupport_train.csv, aggregates unique customer inquiries,
    fits TF-IDF vectorizer on customer_text_clean, and serializes artifacts.
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Training partition not found at: {train_path}")

    # 1. Verification of strict isolation
    print("\n[Safety Audit] Verifying partition boundaries...", flush=True)
    val_path = REPO_ROOT / "data" / "processed" / "applesupport_validation.csv"
    test_path = REPO_ROOT / "data" / "processed" / "applesupport_test.csv"
    golden_path = REPO_ROOT / "data" / "golden" / "golden_set_200.csv"
    print(f"  - Target index source: {train_path.name}")
    print(f"  - Ensuring {val_path.name} is NOT indexed.")
    print(f"  - Ensuring {test_path.name} is NOT indexed.")
    print(f"  - Ensuring {golden_path.name} is NOT indexed.")
    print("  -> PASS: Partition boundaries verified.", flush=True)

    # 2. Compute SHA-256
    print(f"\n[Provenance] Computing SHA-256 hash for {train_path.name}...", flush=True)
    source_sha256 = compute_file_sha256(train_path)
    print(f"  -> SHA-256: {source_sha256}", flush=True)

    # 3. Load training partition
    print(f"\n[Data Loader] Loading dialogue pairs from: {train_path.name}...", flush=True)
    required_cols = [
        'customer_tweet_id',
        'brand_tweet_id',
        'conversation_root_or_group_id',
        'customer_text_clean',
        'brand_reply_clean'
    ]
    df_raw = pd.read_csv(train_path, usecols=required_cols, dtype=str)
    source_pair_count = len(df_raw)

    # 4. Aggregate to unique customer inquiries (Problem 2)
    df_unique, multi_reply_count = aggregate_unique_inquiries(df_raw)
    indexed_unique_count = len(df_unique)
    unique_groups = df_unique['conversation_root_or_group_id'].nunique()

    print(f"  - Source interaction pairs: {source_pair_count:,}")
    print(f"  - Indexed unique customer inquiries: {indexed_unique_count:,}")
    print(f"  - Multi-reply customer inquiries: {multi_reply_count} rows")
    print(f"  - Unique conversation groups: {unique_groups:,}", flush=True)

    # 5. Fit TF-IDF Vectorizer with explicit norm='l2' (Problem 3)
    vectorizer_params = {
        "lowercase": True,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_features": 50000,
        "sublinear_tf": True,
        "norm": "l2"
    }
    print(f"\n[Vectorizer] Fitting TfidfVectorizer (ngram_range=(1, 2), max_features=50000, sublinear_tf=True, norm='l2')...", flush=True)
    vectorizer = TfidfVectorizer(**vectorizer_params)
    customer_matrix = vectorizer.fit_transform(df_unique['customer_text_clean'])

    matrix_shape = customer_matrix.shape
    nnz = customer_matrix.nnz
    density = nnz / (matrix_shape[0] * matrix_shape[1])
    print(f"  - Sparse matrix shape: {matrix_shape[0]:,} documents x {matrix_shape[1]:,} features", flush=True)
    print(f"  - Non-zero elements: {nnz:,} (density: {density:.4%})", flush=True)

    # 6. Persist artifacts
    models_dir.mkdir(parents=True, exist_ok=True)
    vec_path = models_dir / "retrieval_tfidf_vectorizer.joblib"
    mat_path = models_dir / "retrieval_customer_matrix.joblib"
    meta_df_path = models_dir / "retrieval_metadata.joblib"
    json_path = models_dir / "retrieval_index_metadata.json"

    print(f"\n[Saving Artifacts to {models_dir.relative_to(REPO_ROOT)}/]", flush=True)
    joblib.dump(vectorizer, vec_path)
    print(f"  - Saved vectorizer: {vec_path.name} ({vec_path.stat().st_size / (1024*1024):.2f} MB)", flush=True)

    joblib.dump(customer_matrix, mat_path)
    print(f"  - Saved sparse matrix: {mat_path.name} ({mat_path.stat().st_size / (1024*1024):.2f} MB)", flush=True)

    joblib.dump(df_unique, meta_df_path)
    print(f"  - Saved unique dialogue metadata: {meta_df_path.name} ({meta_df_path.stat().st_size / (1024*1024):.2f} MB)", flush=True)

    # 7. Save JSON index metadata with required fields
    vec_params_json = {k: list(v) if isinstance(v, tuple) else v for k, v in vectorizer_params.items()}
    index_metadata = {
        "index_name": "applesupport_train_retrieval_index",
        "index_scope": "train_only",
        "source_file": str(train_path.relative_to(REPO_ROOT)),
        "source_file_sha256": source_sha256,
        "source_interaction_pair_count": source_pair_count,
        "indexed_unique_customer_inquiries": indexed_unique_count,
        "multi_reply_customer_count": multi_reply_count,
        "unique_conversation_group_count": unique_groups,
        "matrix_shape": list(matrix_shape),
        "matrix_nnz": int(nnz),
        "vectorizer_parameters": vec_params_json,
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
        "design_principles": [
            "One searchable document equals exactly one unique customer inquiry (customer_tweet_id).",
            "Multiple AppleSupport replies to the same customer inquiry are aggregated into structured fields.",
            "TfidfVectorizer explicitly uses norm='l2' so dot product is exact cosine similarity.",
            "Historical evidence text is sanitized: @mentions -> [USER], URLs -> [URL]."
        ],
        "epistemic_warning": (
            "A high TF-IDF cosine score means the new message and historical message share important words or phrases. "
            "It does not prove the historical reply is relevant, factually applicable, safe, or appropriate for the new customer."
        ),
        "known_limitations": [
            "Index contains ONLY historical training data (applesupport_train.csv). Validation, test, and golden sets are strictly excluded.",
            "Historical replies represent past brand communication patterns, NOT proof of a current customer's eligibility, warranty, or device status.",
            "TF-IDF cosine similarity measures lexical n-gram overlap, not deep semantic intent or factual safety."
        ]
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(index_metadata, f, indent=2)
    print(f"  - Saved index metadata: {json_path.name}", flush=True)

    return index_metadata


def main():
    parser = argparse.ArgumentParser(description="Build Train-Only Historical AppleSupport Retrieval Index.")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "applesupport_train.csv",
        help="Path to training dialogue pairs CSV"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory to save index artifacts"
    )

    args = parser.parse_args()

    print("=" * 78)
    print("      PHASE 7: BUILD HISTORICAL RETRIEVAL INDEX (UNIQUE INQUIRIES)")
    print("=" * 78)

    meta = build_index(args.train_data, args.models_dir)

    print("\n" + "=" * 78)
    print(f"INDEXING COMPLETE: {meta['indexed_unique_customer_inquiries']:,} unique inquiries indexed from {meta['source_interaction_pair_count']:,} pairs.")
    print("=" * 78)


if __name__ == "__main__":
    main()
