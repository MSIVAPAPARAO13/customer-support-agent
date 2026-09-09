#!/usr/bin/env python3
"""
Audit Historical AppleSupport Retrieval (Phase 7 - Improved)
============================================================

Purpose:
  Performs an auditable, reproducible review of the train-only retrieval engine
  using validation queries, generating human review sheets and audit reports
  with conservative lexical similarity bands and strict epistemic calibration.

Key Safeguards & Methodology Rules:
  1. HIGH SIMILARITY != STRONG GROUNDING:
     Explicitly documents that high TF-IDF cosine similarity reflects lexical
     n-gram overlap only. It does not prove the historical reply is relevant,
     factually applicable, safe, or appropriate for the new customer.
  2. CONSERVATIVE EVIDENCE THRESHOLD (0.30):
     Categorizes queries into:
       - < 0.30  : insufficient_lexical_evidence
       - 0.30-0.50: weak_lexical_evidence
       - 0.50-0.70: moderate_lexical_evidence
       - >= 0.70 : high_lexical_similarity
  3. HUMAN REVIEW SHEET (30 Queries x Top 3 = 90 Rows):
     Sets `needs_human_relevance_review = True` and `review_status = "NEEDS_HUMAN_REVIEW"`.
     Leaves evaluation columns blank for physical human review.
  4. NO GROUNDING CLAIMS:
     Retrieval quality is NOT claimed until human adjudication is completed.

Outputs:
  - outputs/retrieval/retrieval_review_sheet.csv
  - outputs/retrieval/validation_retrieval_audit.md

Usage:
  python scripts/audit_retrieval.py
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
import joblib

# Ensure Windows console stdout handles UTF-8 gracefully
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent

# Import retrieval engine components
sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from retrieve_similar_dialogues import (
    load_retrieval_artifacts,
    retrieve_batch,
    get_lexical_similarity_band,
    sanitize_text
)


def verify_isolation(
    train_meta_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> Tuple[bool, int, int]:
    """
    Verifies that validation queries do not exist in the training index.
    """
    train_tweet_ids = set(train_meta_df['customer_tweet_id'])
    val_tweet_ids = set(val_df['customer_tweet_id'])
    id_overlap = len(train_tweet_ids.intersection(val_tweet_ids))

    train_groups = set(train_meta_df['conversation_root_or_group_id'].dropna())
    val_groups = set(val_df['conversation_root_or_group_id'].dropna())
    group_overlap = len(train_groups.intersection(val_groups))

    is_isolated = (id_overlap == 0 and group_overlap == 0)
    return is_isolated, id_overlap, group_overlap


def generate_audit_report(
    indexed_unique_count: int,
    source_pair_count: int,
    multi_reply_count: int,
    total_val_queries: int,
    top_k: int,
    min_sim: float,
    df_all_retrieved: pd.DataFrame,
    df_audit_30: pd.DataFrame,
    is_isolated: bool,
    id_overlap: int,
    group_overlap: int,
    source_sha256: str,
    strict_report_path: Optional[Path] = None
) -> str:
    """Generates formatted markdown audit report."""
    best_ranks_all = df_all_retrieved[df_all_retrieved['rank'] == 1]
    mean_sim = best_ranks_all['similarity_score'].mean()
    median_sim = best_ranks_all['similarity_score'].median()
    pct_weak = (best_ranks_all['similarity_score'] < min_sim).mean() * 100
    bands = best_ranks_all['lexical_similarity_band'].value_counts().to_dict()

    lines = []
    lines.append("# Historical AppleSupport Dialogue Retrieval Audit Report (Phase 7 - Improved)\n")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Required Epistemic Disclaimer on Grounding & Accuracy**:")
    lines.append("> **Retrieval has been verified as train-only historical search. It has not yet been proven that retrieved reply patterns are relevant, safe, or beneficial. The 30-row human review sheet is required before making that claim.**")
    lines.append(">")
    lines.append("> **A high TF-IDF cosine score means the new message and historical message share important words or phrases. It does not prove the historical reply is relevant, factually applicable, safe, or appropriate for the new customer.**")
    lines.append("> A vague complaint with nearly identical profanity can receive similarity 1.0 but still provide poor grounding.")
    lines.append("> High lexical similarity (`similarity >= 0.70`) reflects **stronger lexical similarity only**, NOT safe grounding or safe auto-handling.\n")

    lines.append("## 1. Indexing & Querying Provenance\n")
    lines.append(f"- **Unique Customer Inquiries Indexed**: {indexed_unique_count:,} (`One searchable document = one unique customer_tweet_id`)")
    lines.append(f"- **Source Interaction Pairs Processed**: {source_pair_count:,} (`data/processed/applesupport_train.csv`)")
    lines.append(f"- **Multi-Reply Customer Inquiries**: {multi_reply_count} rows (unique brand replies aggregated into structured fields)")
    lines.append(f"- **Source File SHA-256**: `{source_sha256}`")
    lines.append(f"- **Validation Queries Evaluated**: {total_val_queries:,} external queries (`data/processed/applesupport_validation.csv`)")
    lines.append(f"- **Audit Sample Selected**: Exactly 30 validation queries (seed `42`), producing 90 candidate pairs (top-k = {top_k})")
    lines.append(f"- **Evidence Similarity Threshold**: {min_sim:.2f} (conservative)\n")

    lines.append("## 2. Partition Isolation & Anti-Leakage Verification\n")
    lines.append(f"- **Query Tweet ID Overlap with Index**: {id_overlap} records")
    lines.append(f"- **Conversation Group Overlap with Index**: {group_overlap} records")
    lines.append(f"- **Isolation Status**: {'CONFIRMED ZERO LEAKAGE' if is_isolated else 'LEAKAGE DETECTED'}")
    lines.append("- **Test & Golden Status**: Strictly 0 rows from `applesupport_test.csv` or `golden_set_200.csv` were loaded or queried.\n")

    lines.append("## 3. Lexical Similarity Score Distribution (Full Validation Set)\n")
    lines.append(f"- **Mean Best Similarity Score**: {mean_sim:.4f}")
    lines.append(f"- **Median Best Similarity Score**: {median_sim:.4f}")
    lines.append(f"- **Proportion Below Evidence Threshold (< {min_sim:.2f})**: {pct_weak:.2f}% ({int(pct_weak/100 * total_val_queries):,} / {total_val_queries:,} queries)")
    lines.append(f"- **Proportion with Sufficient Evidence (>= {min_sim:.2f})**: {100.0 - pct_weak:.2f}% ({total_val_queries - int(pct_weak/100 * total_val_queries):,} / {total_val_queries:,} queries)\n")

    lines.append("### Lexical Similarity Band Breakdown\n")
    lines.append("| Lexical Similarity Band | Query Count | % of Validation | Operational Interpretation |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(f"| `high_lexical_similarity` (>= 0.70) | {bands.get('high_lexical_similarity', 0):,} | {bands.get('high_lexical_similarity', 0)/total_val_queries:.2%} | Stronger lexical similarity only; relevance unverified |")
    lines.append(f"| `moderate_lexical_evidence` (0.50 - 0.70) | {bands.get('moderate_lexical_evidence', 0):,} | {bands.get('moderate_lexical_evidence', 0)/total_val_queries:.2%} | Substantial vocabulary overlap |")
    lines.append(f"| `weak_lexical_evidence` (0.30 - 0.50) | {bands.get('weak_lexical_evidence', 0):,} | {bands.get('weak_lexical_evidence', 0)/total_val_queries:.2%} | Partial vocabulary overlap |")
    lines.append(f"| `insufficient_lexical_evidence` (< 0.30) | {bands.get('insufficient_lexical_evidence', 0):,} | {bands.get('insufficient_lexical_evidence', 0)/total_val_queries:.2%} | **Escalation trigger: weak precedent** |\n")

    # 4. Five High Lexical Similarity Examples with Relevance Warning
    lines.append("## 4. Five De-Identified High Lexical Similarity Examples (Relevance Unverified)\n")
    lines.append("> [!WARNING]")
    lines.append("> **Relevance Unverified**: High cosine similarity indicates lexical n-gram overlap. It does NOT guarantee that the historical reply is appropriate or safe for the new customer.\n")
    
    good_queries = df_audit_30[df_audit_30['rank'] == 1].sort_values(by='similarity_score', ascending=False).head(5)
    for idx, (_, r) in enumerate(good_queries.iterrows(), start=1):
        lines.append(f"### Example {idx} (Similarity Score: {r['similarity_score']:.4f} | Band: `{r['lexical_similarity_band']}`)\n")
        lines.append(f"- **Validation Query (Tweet `{r['query_customer_tweet_id']}`):**\n  > \"{r['query_customer_text_clean']}\"")
        lines.append(f"- **Retrieved Historical Inquirer (Tweet `{r['retrieved_customer_tweet_id']}`):**\n  > \"{r['retrieved_customer_text_clean']}\"")
        lines.append(f"- **Actual AppleSupport Historical Reply (Tweet `{r['retrieved_brand_tweet_id']}`):**\n  > \"{r['retrieved_brand_reply_clean']}\"")
        lines.append(f"- **Audit Status**: `needs_human_relevance_review = True`\n")

    # 5. Five Insufficient / Weak Lexical Evidence Examples
    lines.append("## 5. Five De-Identified Insufficient / Weak Lexical Evidence Examples (Escalation Signals)\n")
    weak_queries = df_audit_30[df_audit_30['rank'] == 1].sort_values(by='similarity_score', ascending=True).head(5)
    for idx, (_, r) in enumerate(weak_queries.iterrows(), start=1):
        lines.append(f"### Example {idx} (Similarity Score: {r['similarity_score']:.4f} | Band: `{r['lexical_similarity_band']}`)\n")
        lines.append(f"- **Validation Query (Tweet `{r['query_customer_tweet_id']}`):**\n  > \"{r['query_customer_text_clean']}\"")
        lines.append(f"- **Retrieved Nearest Inquirer (Tweet `{r['retrieved_customer_tweet_id']}`):**\n  > \"{r['retrieved_customer_text_clean']}\"")
        lines.append(f"- **Actual AppleSupport Historical Reply (Tweet `{r['retrieved_brand_tweet_id']}`):**\n  > \"{r['retrieved_brand_reply_clean']}\"")
        lines.append(f"- **Audit Status**: Insufficient/weak evidence (`has_sufficient_historical_evidence = {r['has_sufficient_historical_evidence']}`) | Warning: `{r['retrieval_warning']}`\n")

    # 6. Train-Validation Repeated Wording Summary
    lines.append("## 6. Cross-Conversation Repeated Wording Limitation\n")
    lines.append("- A detailed audit is maintained in `train_validation_text_overlap_report.md`.")
    lines.append("- Independent customers frequently post identical short support inquiries across separate conversation trees.")
    lines.append("- While conversation isolation is preserved, repeated wording inflates lexical similarity on common phrases.")
    lines.append("- For novel phrasing in production, retrieval evidence will naturally fall back to lower similarity bands, triggering necessary human review.")
    lines.append("\n---\n*Audit report generated by scripts/audit_retrieval.py.*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit Historical AppleSupport Retrieval Engine.")
    parser.add_argument(
        "--val-data",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "applesupport_validation.csv",
        help="Path to validation dialogue CSV"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory containing retrieval index artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "retrieval",
        help="Directory to save audit review sheet and report"
    )
    parser.add_argument(
        "--all-retrieved-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "retrieval" / "validation_retrieval_examples.csv",
        help="Path to precomputed retrieval examples on full validation set"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Deterministic sample size for human review sheet (default: 30)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic query sampling (default: 42)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of nearest neighbours to retrieve per query (default: 3)"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.30,
        help="Similarity threshold for sufficient historical evidence (default: 0.30)"
    )

    args = parser.parse_args()

    print("=" * 78)
    print("      PHASE 7: AUDIT HISTORICAL DIALOGUE RETRIEVAL ENGINE (IMPROVED)")
    print("=" * 78)

    # 1. Load index artifacts
    vectorizer, matrix, meta_df = load_retrieval_artifacts(args.models_dir)

    # Load index JSON metadata
    json_path = args.models_dir / "retrieval_index_metadata.json"
    source_sha256 = "UNKNOWN"
    source_pairs = 82077
    multi_reply_count = 14
    indexed_unique_count = len(meta_df)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            idx_meta = json.load(f)
            source_sha256 = idx_meta.get("source_file_sha256", "UNKNOWN")
            source_pairs = idx_meta.get("source_interaction_pair_count", 82077)
            multi_reply_count = idx_meta.get("multi_reply_customer_count", 14)
            indexed_unique_count = idx_meta.get("indexed_unique_customer_inquiries", len(meta_df))

    # 2. Load validation data
    if not args.val_data.exists():
        raise FileNotFoundError(f"Validation data not found: {args.val_data}")

    print(f"\n[Data Loader] Loading validation queries from: {args.val_data.name}...", flush=True)
    df_val = pd.read_csv(args.val_data, dtype=str)
    total_val = len(df_val)
    print(f"  - Loaded {total_val:,} validation queries.", flush=True)

    # 3. Isolation check
    print("\n[Safety Audit] Checking cross-partition isolation...", flush=True)
    is_isolated, id_overlap, grp_overlap = verify_isolation(meta_df, df_val)
    print(f"  - Validation tweet ID overlap: {id_overlap}")
    print(f"  - Conversation group overlap: {grp_overlap}")
    if not is_isolated:
        raise ValueError("CRITICAL INTEGRITY FAILURE: Overlap detected between train index and validation queries!")
    print("  -> PASS: Zero overlap confirmed. Validation queries are external to train index.", flush=True)

    # 4. Check or generate full validation retrieval
    if args.all_retrieved_csv.exists():
        print(f"\n[Loading Full Validation Retrieval] Reading from: {args.all_retrieved_csv.name}...", flush=True)
        df_all_retrieved = pd.read_csv(args.all_retrieved_csv)
    else:
        print(f"\n[Computing Full Validation Retrieval] Computing top-{args.top_k} for all {total_val:,} queries...", flush=True)
        df_all_retrieved = retrieve_batch(
            query_df=df_val,
            vectorizer=vectorizer,
            train_matrix=matrix,
            meta_df=meta_df,
            top_k=args.top_k,
            min_similarity=args.min_similarity
        )
        args.all_retrieved_csv.parent.mkdir(parents=True, exist_ok=True)
        df_all_retrieved.to_csv(args.all_retrieved_csv, index=False)
        print(f"  Saved full validation retrieval to: {args.all_retrieved_csv.name}", flush=True)

    # 5. Deterministic sample of 30 queries for human review sheet
    print(f"\n[Sampling Review Cohort] Selecting {args.sample_size} validation queries with seed {args.seed}...", flush=True)
    df_val_sample = df_val.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)

    print(f"  Retrieving top {args.top_k} unique historical examples for {args.sample_size} queries...", flush=True)
    df_audit_retrieved = retrieve_batch(
        query_df=df_val_sample,
        vectorizer=vectorizer,
        train_matrix=matrix,
        meta_df=meta_df,
        top_k=args.top_k,
        min_similarity=args.min_similarity
    )

    # 6. Add blank human-review columns
    review_columns = [
        'query_customer_tweet_id',
        'query_customer_text_clean',
        'query_conversation_group_id',
        'rank',
        'similarity_score',
        'best_similarity_score',
        'lexical_similarity_band',
        'has_sufficient_historical_evidence',
        'retrieval_warning',
        'needs_human_relevance_review',
        'retrieved_customer_tweet_id',
        'retrieved_brand_tweet_id',
        'retrieved_conversation_group_id',
        'retrieved_customer_text_clean',
        'retrieved_brand_reply_clean',
        'retrieval_scope',
        'retrieval_relevant_yes_no',
        'historical_reply_pattern_usable_yes_no',
        'unsupported_or_unsafe_pattern_yes_no',
        'review_notes',
        'reviewer_id',
        'review_status'
    ]

    df_review_sheet = df_audit_retrieved.copy()
    df_review_sheet['retrieval_relevant_yes_no'] = ""
    df_review_sheet['historical_reply_pattern_usable_yes_no'] = ""
    df_review_sheet['unsupported_or_unsafe_pattern_yes_no'] = ""
    df_review_sheet['review_notes'] = ""
    df_review_sheet['reviewer_id'] = ""
    df_review_sheet['review_status'] = "NEEDS_HUMAN_REVIEW"

    # 7. Save Review Sheet and Audit Markdown Report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_sheet_path = args.output_dir / "retrieval_review_sheet.csv"
    audit_report_path = args.output_dir / "validation_retrieval_audit.md"

    df_review_sheet[review_columns].to_csv(review_sheet_path, index=False)
    print(f"\n[Saving Review Sheet] Saved {len(df_review_sheet)} rows to: {review_sheet_path.name}", flush=True)
    print(f"  All rows initialized with review_status = 'NEEDS_HUMAN_REVIEW'", flush=True)

    report_text = generate_audit_report(
        indexed_unique_count=indexed_unique_count,
        source_pair_count=source_pairs,
        multi_reply_count=multi_reply_count,
        total_val_queries=total_val,
        top_k=args.top_k,
        min_sim=args.min_similarity,
        df_all_retrieved=df_all_retrieved,
        df_audit_30=df_audit_retrieved,
        is_isolated=is_isolated,
        id_overlap=id_overlap,
        group_overlap=grp_overlap,
        source_sha256=source_sha256
    )

    audit_report_path.write_text(report_text, encoding="utf-8")
    print(f"[Saving Audit Report] Saved comprehensive audit to: {audit_report_path.name}", flush=True)

    print("\n" + "=" * 78)
    print("RETRIEVAL AUDIT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
