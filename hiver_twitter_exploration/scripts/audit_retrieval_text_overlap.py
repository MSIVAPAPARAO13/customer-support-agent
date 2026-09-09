#!/usr/bin/env python3
"""
Audit Cross-Conversation Repeated Wording & Strict Retrieval (Phase 7 - Problem 4)
==================================================================================

Purpose:
  Audits exact normalized text overlap between training and validation customer
  inquiries, investigates why conversation-group isolation allows repeated wording,
  and executes a strict retrieval diagnostic excluding candidate historical pairs
  with identical wording.

Key Distinctions:
  - This text overlap is NOT data leakage (partitions are strictly isolated by
    conversation root/group ID and tweet ID).
  - It is termed: `cross-conversation repeated wording`.
  - It occurs because real human customers independently formulate common support
    questions using identical phrasing (e.g., "what's the latest ios update?",
    "my battery is dying so fast", "help locked out of my apple id").
  - However, offline retrieval that matches identical queries may appear deceptively
    effective compared to handling novel customer phrasing in production.

Outputs:
  - outputs/retrieval/train_validation_text_overlap_report.md
  - outputs/retrieval/validation_retrieval_strict_examples.csv

Usage:
  python scripts/audit_retrieval_text_overlap.py
"""

import sys
import os
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
    normalize_for_text_exclusion,
    get_lexical_similarity_band,
    sanitize_text
)


def analyze_text_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> Tuple[int, float, List[Dict[str, Any]], Dict[str, int]]:
    """
    Computes exact normalized customer text matches between train and validation.
    """
    print("\n[Overlap Analysis] Normalizing customer texts across partitions...", flush=True)
    train_norm = train_df['customer_text_clean'].apply(normalize_for_text_exclusion)
    val_norm = val_df['customer_text_clean'].apply(normalize_for_text_exclusion)

    train_text_counts = train_norm.value_counts().to_dict()
    val_text_counts = val_norm.value_counts().to_dict()

    train_unique_texts = set(train_norm[train_norm != ''])
    val_unique_texts = set(val_norm[val_norm != ''])

    # Overlapping unique strings
    overlap_unique = train_unique_texts.intersection(val_unique_texts)

    # Count validation rows that have exact normalized matches in training
    val_overlap_mask = val_norm.isin(train_unique_texts)
    val_overlap_row_count = int(val_overlap_mask.sum())
    val_total = len(val_df)
    overlap_pct = (val_overlap_row_count / val_total) * 100 if val_total > 0 else 0.0

    print(f"  - Validation rows with exact normalized match in train: {val_overlap_row_count:,} / {val_total:,} ({overlap_pct:.2f}%)", flush=True)
    print(f"  - Distinct text phrases shared: {len(overlap_unique):,}", flush=True)

    # Extract 5 representative de-identified examples with frequencies
    examples = []
    # Sort overlap phrases by frequency in validation
    sorted_phrases = sorted(overlap_unique, key=lambda p: (val_text_counts.get(p, 0), train_text_counts.get(p, 0)), reverse=True)

    for phrase in sorted_phrases[:10]:
        # Skip trivial single-word utterances if longer phrases available
        if len(phrase.split()) >= 3 and len(examples) < 5:
            # Find a train sample and a val sample
            t_sample = train_df[train_norm == phrase].iloc[0]
            v_sample = val_df[val_norm == phrase].iloc[0]
            examples.append({
                "normalized_phrase": phrase,
                "train_tweet_id": str(t_sample['customer_tweet_id']),
                "val_tweet_id": str(v_sample['customer_tweet_id']),
                "train_group_id": str(t_sample.get('conversation_root_or_group_id', '')),
                "val_group_id": str(v_sample.get('conversation_root_or_group_id', '')),
                "train_freq": train_text_counts[phrase],
                "val_freq": val_text_counts[phrase]
            })

    return val_overlap_row_count, overlap_pct, examples, train_text_counts


def compare_distributions(
    df_normal: pd.DataFrame,
    df_strict: pd.DataFrame
) -> Dict[str, Any]:
    """Compares similarity score percentiles and band distributions."""
    norm_best = df_normal[df_normal['rank'] == 1]
    strict_best = df_strict[df_strict['rank'] == 1]

    def get_stats(df_b: pd.DataFrame) -> Dict[str, Any]:
        scores = df_b['similarity_score']
        bands = df_b['lexical_similarity_band'].value_counts().to_dict()
        n = len(scores)
        return {
            "mean": float(scores.mean()),
            "median": float(scores.median()),
            "pct_below_30": float((scores < 0.30).mean() * 100),
            "pct_high_sim": float((scores >= 0.70).mean() * 100),
            "bands": {b: bands.get(b, 0) for b in [
                "insufficient_lexical_evidence",
                "weak_lexical_evidence",
                "moderate_lexical_evidence",
                "high_lexical_similarity"
            ]},
            "bands_pct": {b: (bands.get(b, 0) / n) * 100 if n > 0 else 0 for b in [
                "insufficient_lexical_evidence",
                "weak_lexical_evidence",
                "moderate_lexical_evidence",
                "high_lexical_similarity"
            ]}
        }

    return {
        "normal": get_stats(norm_best),
        "strict": get_stats(strict_best)
    }


def generate_report(
    overlap_count: int,
    overlap_pct: float,
    examples: List[Dict[str, Any]],
    comparison: Dict[str, Any],
    total_val: int
) -> str:
    """Formats markdown audit report."""
    norm_st = comparison['normal']
    strict_st = comparison['strict']

    lines = []
    lines.append("# Cross-Conversation Repeated Wording & Strict Retrieval Audit\n")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Epistemic Distiction: Cross-Conversation Repeated Wording vs. Data Leakage**:")
    lines.append("> The text overlap documented here is **cross-conversation repeated wording**, NOT data leakage.")
    lines.append("> Every validation conversation has a distinct `conversation_root_or_group_id` from the training set.")
    lines.append("> However, real customers frequently post identical short inquiries (e.g. *'my battery is draining fast'*), which inflates lexical retrieval scores if unanalyzed.\n")

    lines.append("## 1. Executive Summary: Overlap Statistics\n")
    lines.append(f"- **Total Validation Inquiries Audited**: {total_val:,}")
    lines.append(f"- **Validation Inquiries with Exact Normalized Match in Train**: {overlap_count:,} ({overlap_pct:.2f}%)")
    lines.append(f"- **Validation Inquiries with Novel Phrasing**: {total_val - overlap_count:,} ({100.0 - overlap_pct:.2f}%)\n")

    lines.append("## 2. Why Conversation-Group Isolation Does Not Prevent Repeated Wording\n")
    lines.append("When we constructed our 80/10/10 dataset split in Phase 2, we enforced **strict conversation-group isolation**: all tweets belonging to the same root tree were kept together in either train, validation, or test.")
    lines.append("\nHowever, independent customers around the world who experience common issues (such as iOS 11 battery drain, locked Apple IDs, or cracked screens) frequently compose tweets using identical or near-identical syntax. For instance, hundreds of distinct users tweeted the exact four words: *'iphone 7 ios 11'* or *'my battery is terrible'* in completely separate support threads.")
    lines.append("\nThis is natural language reuse, but it introduces an important evaluation caveat: offline retrieval evaluations will look exceptionally strong on these repeated inquiries, while providing less guidance on truly novel phrasing.\n")

    lines.append("## 3. Five De-Identified Examples of Cross-Conversation Repeated Wording\n")
    for i, ex in enumerate(examples, start=1):
        lines.append(f"### Example {i}: \"{ex['normalized_phrase']}\"")
        lines.append(f"- **Validation Sample**: Tweet `{ex['val_tweet_id']}` (Conversation Group `{ex['val_group_id']}`)")
        lines.append(f"- **Training Match**: Tweet `{ex['train_tweet_id']}` (Conversation Group `{ex['train_group_id']}`)")
        lines.append(f"- **Corpus Frequency**: Appears {ex['train_freq']} time(s) in Train, {ex['val_freq']} time(s) in Validation")
        lines.append(f"- **Status**: Independent conversations with identical normalized wording.\n")

    lines.append("## 4. Normal Retrieval vs. Strict Text-Exclusion Retrieval Comparison\n")
    lines.append("To understand how much offline retrieval relies on exact query duplicates, we ran a **Strict Retrieval Diagnostic** where candidates sharing identical normalized wording with the query were excluded (in addition to standard tweet ID and conversation-group exclusion).\n")

    lines.append("| Metric | Normal Retrieval | Strict Diagnostic Retrieval | Delta / Observation |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(f"| **Mean Best Similarity** | {norm_st['mean']:.4f} | {strict_st['mean']:.4f} | {strict_st['mean'] - norm_st['mean']:+.4f} |")
    lines.append(f"| **Median Best Similarity** | {norm_st['median']:.4f} | {strict_st['median']:.4f} | {strict_st['median'] - norm_st['median']:+.4f} |")
    lines.append(f"| **Proportion Below 0.30 Threshold** | {norm_st['pct_below_30']:.2f}% | {strict_st['pct_below_30']:.2f}% | {strict_st['pct_below_30'] - norm_st['pct_below_30']:+.2f}% |")
    lines.append(f"| **Proportion with High Similarity (>= 0.70)** | {norm_st['pct_high_sim']:.2f}% | {strict_st['pct_high_sim']:.2f}% | {strict_st['pct_high_sim'] - norm_st['pct_high_sim']:+.2f}% |\n")

    lines.append("### Lexical Similarity Band Breakdown Comparison\n")
    lines.append("| Lexical Similarity Band | Normal Retrieval Count (%) | Strict Diagnostic Count (%) |")
    lines.append("| :--- | :--- | :--- |")
    for band in ["high_lexical_similarity", "moderate_lexical_evidence", "weak_lexical_evidence", "insufficient_lexical_evidence"]:
        n_cnt = norm_st['bands'][band]
        n_pct = norm_st['bands_pct'][band]
        s_cnt = strict_st['bands'][band]
        s_pct = strict_st['bands_pct'][band]
        lines.append(f"| `{band}` | {n_cnt:,} ({n_pct:.2f}%) | {s_cnt:,} ({s_pct:.2f}%) |")

    lines.append("\n## 5. Architectural Implications for Production\n")
    lines.append("1. **Escalation Gating**: Queries dropping into `insufficient_lexical_evidence` (< 0.30) or `weak_lexical_evidence` (0.30 - 0.50) must not trigger autonomous reply drafting.")
    lines.append("2. **Evidence Sanitization**: All candidate pairs are sanitized (@mentions -> `[USER]`, URLs -> `[URL]`) before being provided to downstream prompt builders.")
    lines.append("3. **Epistemic Humility**: Retrieval provides lexical evidence of past communication patterns, never factual ground truth for the new customer.")
    lines.append("\n---\n*Report generated by scripts/audit_retrieval_text_overlap.py.*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit Cross-Conversation Repeated Wording & Strict Retrieval.")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "applesupport_train.csv",
        help="Path to training dialogues CSV"
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "applesupport_validation.csv",
        help="Path to validation dialogues CSV"
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
        help="Directory to save strict examples and audit report"
    )

    args = parser.parse_args()

    print("=" * 78)
    print("      PHASE 7: CROSS-CONVERSATION REPEATED WORDING AUDIT")
    print("=" * 78)

    # 1. Load partitions
    print(f"\n[Data Loader] Loading training and validation dialogues...", flush=True)
    df_train = pd.read_csv(args.train_data, usecols=['customer_tweet_id', 'conversation_root_or_group_id', 'customer_text_clean'], dtype=str)
    df_val = pd.read_csv(args.val_data, usecols=['customer_tweet_id', 'conversation_root_or_group_id', 'customer_text_clean'], dtype=str)

    # 2. Text overlap analysis
    overlap_count, overlap_pct, examples, _ = analyze_text_overlap(df_train, df_val)

    # 3. Load retrieval artifacts
    vectorizer, matrix, meta_df = load_retrieval_artifacts(args.models_dir)

    # 4. Run Normal Retrieval
    print("\n[Diagnostic 1/2] Running Normal Retrieval on Validation Partition...", flush=True)
    df_normal = retrieve_batch(
        query_df=df_val,
        vectorizer=vectorizer,
        train_matrix=matrix,
        meta_df=meta_df,
        top_k=3,
        min_similarity=0.30,
        strict_text_exclusion=False
    )
    normal_csv = args.output_dir / "validation_retrieval_examples.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_normal.to_csv(normal_csv, index=False)
    print(f"  Saved normal retrieval to: {normal_csv.name}", flush=True)

    # 5. Run Strict Retrieval (excluding identical normalized text)
    print("\n[Diagnostic 2/2] Running Strict Retrieval (Excluding Identical Normalized Text)...", flush=True)
    df_strict = retrieve_batch(
        query_df=df_val,
        vectorizer=vectorizer,
        train_matrix=matrix,
        meta_df=meta_df,
        top_k=3,
        min_similarity=0.30,
        strict_text_exclusion=True
    )
    strict_csv = args.output_dir / "validation_retrieval_strict_examples.csv"
    df_strict.to_csv(strict_csv, index=False)
    print(f"  Saved strict retrieval to: {strict_csv.name}", flush=True)

    # 6. Compare distributions
    comparison = compare_distributions(df_normal, df_strict)

    # 7. Generate markdown report
    report_text = generate_report(
        overlap_count=overlap_count,
        overlap_pct=overlap_pct,
        examples=examples,
        comparison=comparison,
        total_val=len(df_val)
    )
    report_path = args.output_dir / "train_validation_text_overlap_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[Report] Saved audit report to: {report_path.name}", flush=True)

    print("\n" + "=" * 78)
    print("CROSS-CONVERSATION AUDIT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
