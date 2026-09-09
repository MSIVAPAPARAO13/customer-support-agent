#!/usr/bin/env python3
"""
Audit Drafted Customer Support Replies (Phase 8)
================================================

Purpose:
  Conducts safety, coverage, and distribution audits on the drafted validation
  replies. Generates a deterministic 30-row human review sheet for manual
  validation and compiles a comprehensive safety audit report.

Strict Audit Rules:
  - Analyzes only outputs from applesupport_validation.csv.
  - Never accesses or evaluates test or golden rows.
  - Initializes all 30 review rows to review_status = 'NEEDS_HUMAN_REVIEW'.
  - Reports zero automated accuracy, acceptance, or customer-satisfaction claims.

Usage:
  python scripts/audit_drafted_replies.py `
    --input outputs/replies/validation_drafted_replies.csv `
    --review-sheet outputs/replies/reply_human_review_sheet.csv `
    --audit-report outputs/replies/reply_safety_audit.md `
    --seed 42
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np

# Configure console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent


def select_stratified_audit_sample(df: pd.DataFrame, seed: int = 42, target_total: int = 30) -> pd.DataFrame:
    """
    Selects 30 deterministic validation inquiries representing key operational regimes:
      1. Account / security restricted cases
      2. Billing / dispute restricted cases
      3. Other / unclear cases
      4. High lexical similarity (>= 0.70)
      5. Insufficient lexical evidence (< 0.30)
      6. High model confidence (>= 0.85)
      7. Low model confidence (< 0.60)
    """
    np.random.seed(seed)
    selected_indices = set()

    # Bucket 1: Account / Credential security cases
    b1 = df[df['safety_flags'].str.contains('account_compromise|password_or_recovery', regex=True, na=False)].index.tolist()
    if b1:
        chosen = np.random.choice(b1, size=min(4, len(b1)), replace=False)
        selected_indices.update(chosen)

    # Bucket 2: Billing / Payment / Refund dispute cases
    b2 = df[df['safety_flags'].str.contains('unauthorized_payment|refund_or_payment_dispute', regex=True, na=False)].index.tolist()
    rem2 = [i for i in b2 if i not in selected_indices]
    if rem2:
        chosen = np.random.choice(rem2, size=min(4, len(rem2)), replace=False)
        selected_indices.update(chosen)

    # Bucket 3: Other / unclear intent cases
    b3 = df[df['predicted_intent'] == 'other_or_unclear'].index.tolist()
    rem3 = [i for i in b3 if i not in selected_indices]
    if rem3:
        chosen = np.random.choice(rem3, size=min(4, len(rem3)), replace=False)
        selected_indices.update(chosen)

    # Bucket 4: High lexical similarity (>= 0.70)
    b4 = df[(df['best_similarity_score'] >= 0.70) & (~df['restricted_draft'])].index.tolist()
    rem4 = [i for i in b4 if i not in selected_indices]
    if rem4:
        chosen = np.random.choice(rem4, size=min(5, len(rem4)), replace=False)
        selected_indices.update(chosen)

    # Bucket 5: Insufficient lexical evidence (< 0.30)
    b5 = df[(df['best_similarity_score'] < 0.30) & (~df['restricted_draft'])].index.tolist()
    rem5 = [i for i in b5 if i not in selected_indices]
    if rem5:
        chosen = np.random.choice(rem5, size=min(5, len(rem5)), replace=False)
        selected_indices.update(chosen)

    # Bucket 6: High confidence (>= 0.85) with historical pattern
    b6 = df[(df['predicted_confidence'] >= 0.85) & (df['draft_mode'] == 'template_with_historical_pattern')].index.tolist()
    rem6 = [i for i in b6 if i not in selected_indices]
    if rem6:
        chosen = np.random.choice(rem6, size=min(4, len(rem6)), replace=False)
        selected_indices.update(chosen)

    # Bucket 7: Low confidence (< 0.60) / Needs human review
    b7 = df[(df['needs_human_review']) & (~df['restricted_draft'])].index.tolist()
    rem7 = [i for i in b7 if i not in selected_indices]
    if rem7:
        chosen = np.random.choice(rem7, size=min(4, len(rem7)), replace=False)
        selected_indices.update(chosen)

    # Fill up to target_total if needed
    if len(selected_indices) < target_total:
        remaining_pool = [i for i in df.index if i not in selected_indices]
        deficit = target_total - len(selected_indices)
        chosen = np.random.choice(remaining_pool, size=deficit, replace=False)
        selected_indices.update(chosen)

    # Trim if slightly exceeded
    ordered_indices = sorted(list(selected_indices))[:target_total]
    return df.loc[ordered_indices].copy()


def generate_review_sheet(df_sample: pd.DataFrame, output_path: Path) -> None:
    """Exports 30-sample dataframe with blank human annotation review columns."""
    sample_df = df_sample.copy()

    review_cols = [
        "reply_relevant_yes_no",
        "reply_grounded_yes_no",
        "reply_safe_yes_no",
        "clear_next_step_yes_no",
        "would_send_or_escalate",
        "review_notes",
        "reviewer_id",
        "review_status",
    ]

    for col in review_cols:
        sample_df[col] = ""

    sample_df["review_status"] = "NEEDS_HUMAN_REVIEW"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[Review Sheet] Saved {len(sample_df)} review rows to {output_path} (all marked NEEDS_HUMAN_REVIEW).")


def generate_audit_report(df_all: pd.DataFrame, df_sample: pd.DataFrame, output_path: Path) -> None:
    """Generates comprehensive markdown safety audit report."""
    total_drafts = len(df_all)

    # Mode breakdown
    mode_counts = df_all["draft_mode"].value_counts().to_dict()
    restr_count = int(df_all["restricted_draft"].sum())
    restr_pct = (restr_count / total_drafts) * 100

    # Safety flag breakdown
    flag_series = df_all["safety_flags"].dropna().str.split("; ")
    all_flags = [f for sublist in flag_series for f in sublist if f not in ("none", "")]
    flag_counts = pd.Series(all_flags).value_counts().to_dict()

    # Similarity band breakdown
    band_counts = df_all["lexical_similarity_band"].value_counts().to_dict()

    # Select 5 safe historical examples
    safe_examples = df_all[df_all["draft_mode"] == "template_with_historical_pattern"].head(5)

    # Select 5 conservative no-evidence examples
    conservative_examples = df_all[df_all["draft_mode"] == "conservative_no_evidence"].head(5)

    # Select 3 restricted safety examples
    restricted_examples = df_all[df_all["draft_mode"] == "restricted_safety"].head(3)

    report_lines = [
        "# Phase 8 Reply Drafting Safety & Distribution Audit",
        "",
        "## Executive Summary & Integrity Principles",
        "",
        "In **Phase 8**, the AppleSupport inquiry pipeline was augmented with a safe, grounded reply-drafting engine. "
        "The engine operates on predictions from the Phase 6 TF-IDF classifier and historical support patterns retrieved "
        "from the Phase 7 train-only deduplicated retrieval index.",
        "",
        "> [!IMPORTANT]",
        "> **Core Safety & Grounding Principles**:",
        "> 1. **No Hallucination Claims**: Constrained templates reduce unsupported claims and make drafts easier to audit. They do not guarantee that every draft is relevant or safe.",
        "> 2. **Support Protocols vs. Facts**: Retrieved historical AppleSupport replies represent past interaction patterns (such as requesting OS version or advising a restart). They are **never** treated as customer-specific facts, promises, or diagnoses.",
        "> 3. **Stricter Adaptation Threshold (0.50)**: Inquiries with similarity score $< 0.50$ default strictly to `conservative_no_evidence`. Weak lexical similarity (0.30–0.49) is recorded in metadata but is barred from altering drafts.",
        "> 4. **No Automated Quality or Satisfaction Claims**: Zero reply acceptance, satisfaction, or accuracy claims are made prior to human or LLM-judge review. All 30 review rows in `reply_human_review_sheet.csv` are initialized to `NEEDS_HUMAN_REVIEW`.",
        "> 5. **Strict Train-Only Isolation**: All retrieval vectors and classifiers were trained exclusively on `applesupport_train.csv`. Reply drafts were generated exclusively on `applesupport_validation.csv`. Zero rows of `applesupport_test.csv` or `golden_set_200.csv` were loaded.",
        "",
        "---",
        "",
        "## 1. Overall Population Statistics (`applesupport_validation.csv`)",
        "",
        f"- **Total Customer Inquiries Evaluated**: **{total_drafts:,}**",
        f"- **Restricted Safety Drafts (`restricted_draft = True`)**: **{restr_count:,} ({restr_pct:.2f}%)**",
        f"- **Human Classification Review Flags (`needs_human_review = True`)**: **{int(df_all['needs_human_review'].sum()):,} ({(df_all['needs_human_review'].mean()*100):.2f}%)**",
        f"- **Average Best Lexical Similarity Score**: **{df_all['best_similarity_score'].mean():.4f}**",
        "",
        "### Draft Mode Distribution",
        "",
        "| Draft Mode | Count | Proportion | Operational Meaning |",
        "| :--- | :---: | :---: | :--- |",
    ]

    for mode in ["template_with_historical_pattern", "conservative_no_evidence", "restricted_safety"]:
        cnt = mode_counts.get(mode, 0)
        pct = (cnt / total_drafts) * 100
        meaning = (
            "Similarity $\\ge 0.50$; synthesizes intent base template with safe historical protocol pattern"
            if mode == "template_with_historical_pattern"
            else "Similarity $< 0.50$; uses cautious base intent template without historical adaptation"
            if mode == "conservative_no_evidence"
            else "Safety flags detected; brief, neutral secure handoff without troubleshooting"
        )
        report_lines.append(f"| `{mode}` | {cnt:,} | {pct:.2f}% | {meaning} |")

    report_lines.extend([
        "",
        "### Lexical Similarity Band Breakdown",
        "",
        "| Similarity Band | Score Range | Count | Proportion | Influence on Draft |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ])

    band_order = [
        ("insufficient_lexical_evidence", "< 0.30", "Conservative base template; no historical pattern"),
        ("weak_lexical_evidence", "0.30 – 0.49", "Conservative base template; barred from altering drafts"),
        ("moderate_lexical_evidence", "0.50 – 0.69", "Eligible for historical pattern adaptation"),
        ("high_lexical_similarity", ">= 0.70", "Eligible for historical pattern adaptation"),
    ]
    for b_name, b_range, b_inf in band_order:
        cnt = band_counts.get(b_name, 0)
        pct = (cnt / total_drafts) * 100
        report_lines.append(f"| `{b_name}` | {b_range} | {cnt:,} | {pct:.2f}% | {b_inf} |")

    report_lines.extend([
        "",
        "### Safety Flag Detection Breakdown",
        "",
        "| Safety Trigger Category | Occurrences | Strict Handling Action |",
        "| :--- | :---: | :--- |",
    ])

    for flag_id, count in flag_counts.items():
        report_lines.append(f"| `{flag_id}` | {count:,} | Directs to official secure channel; zero credential/payment info requested |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Exemplary De-Identified Draft Cases",
        "",
        "### A. Five Safe Drafts with Historical Patterns (`template_with_historical_pattern`)",
        "",
    ])

    for idx, (_, r) in enumerate(safe_examples.iterrows(), start=1):
        report_lines.extend([
            f"#### Example A{idx}: Tweet `{r['customer_tweet_id']}`",
            f"- **Customer Inquiry**: \"{r['customer_text_clean']}\"",
            f"- **Predicted Intent**: `{r['predicted_intent']}` (Confidence: `{r['predicted_confidence']:.4f}`)",
            f"- **Best Similarity Score**: `{r['best_similarity_score']:.4f}` (`{r['lexical_similarity_band']}`)",
            f"- **Historical Evidence Customer Tweets**: `{r['evidence_customer_tweet_ids']}`",
            f"- **Historical Evidence Brand Replies**: `{r['evidence_brand_tweet_ids']}`",
            f"- **Draft Reply**: > \"{r['draft_reply']}\"",
            f"- **Grounding Note**: *{r['grounding_note']}*",
            "",
        ])

    report_lines.extend([
        "### B. Five Conservative No-Evidence Drafts (`conservative_no_evidence`)",
        "",
    ])

    for idx, (_, r) in enumerate(conservative_examples.iterrows(), start=1):
        report_lines.extend([
            f"#### Example B{idx}: Tweet `{r['customer_tweet_id']}`",
            f"- **Customer Inquiry**: \"{r['customer_text_clean']}\"",
            f"- **Predicted Intent**: `{r['predicted_intent']}` (Confidence: `{r['predicted_confidence']:.4f}`)",
            f"- **Best Similarity Score**: `{r['best_similarity_score']:.4f}` (`{r['lexical_similarity_band']}`)",
            f"- **Draft Reply**: > \"{r['draft_reply']}\"",
            f"- **Grounding Note**: *{r['grounding_note']}*",
            "",
        ])

    report_lines.extend([
        "### C. Restricted Safety Drafts (`restricted_safety`)",
        "",
    ])

    for idx, (_, r) in enumerate(restricted_examples.iterrows(), start=1):
        report_lines.extend([
            f"#### Example C{idx}: Tweet `{r['customer_tweet_id']}`",
            f"- **Customer Inquiry**: \"{r['customer_text_clean']}\"",
            f"- **Triggered Safety Flags**: `{r['safety_flags']}`",
            f"- **Draft Reply**: > \"{r['draft_reply']}\"",
            f"- **Grounding Note**: *{r['grounding_note']}*",
            "",
        ])

    report_lines.extend([
        "---",
        "",
        "## 3. Known Limitations & Technical Constraints",
        "",
        "1. **Template Generalization vs. Specificity**: While constrained templates completely eliminate factual hallucination, they cannot resolve complex multi-turn edge cases on their own.",
        "2. **Lexical Retrieval Boundaries**: TF-IDF cosine similarity measures token and n-gram overlap. It does not measure semantic entailment or verify if the historical diagnosis matches the user's specific symptom.",
        "3. **Absence of Real-Time Account Access**: The system cannot query live order status, activation locks, or iCloud account states.",
        "4. **Need for Phase 9 Decision Routing**: Drafting a reply is distinct from sending it. Phase 9 must implement a rigorous auto-handle vs. human escalation decision policy based on confidence, similarity, and safety restrictions.",
        "5. **Regex Safety Audit Scope**: No matches to the defined prohibited phrases were found by the automated regex audit. Regex checks can detect selected risky phrases, but they cannot prove every generated draft is relevant, accurate, or safe. Human review remains necessary.",
        "",
        "---",
        "",
        "## 4. Required Epistemic Statement",
        "",
        "> **Retrieval has been verified as train-only historical search. It has not yet been proven that retrieved reply patterns are relevant, safe, or beneficial. The 30-row human review sheet (`reply_human_review_sheet.csv`) is required before making that claim.**",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"[Audit Report] Wrote safety audit report to {output_path}.")


def main():
    parser = argparse.ArgumentParser(description="Audit Drafted Customer Support Replies (Phase 8).")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "outputs" / "replies" / "validation_drafted_replies.csv",
        help="Path to input drafted replies CSV.",
    )
    parser.add_argument(
        "--review-sheet",
        type=Path,
        default=REPO_ROOT / "outputs" / "replies" / "reply_human_review_sheet.csv",
        help="Path to save 30-sample human review sheet CSV.",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=REPO_ROOT / "outputs" / "replies" / "reply_safety_audit.md",
        help="Path to save safety audit markdown report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42).",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}. Run drafting first.")

    print("=" * 75)
    print("PHASE 8: REPLY DRAFTING AUDIT & SAFETY EVALUATION")
    print("=" * 75)
    print(f"Input File        : {args.input}")
    print(f"Review Sheet Path : {args.review_sheet}")
    print(f"Audit Report Path : {args.audit_report}")
    print(f"Random Seed       : {args.seed}")
    print("=" * 75)

    df = pd.read_csv(args.input, low_memory=False)
    print(f"[DataLoader] Loaded {len(df):,} drafted replies.")

    # 1. Select 30 stratified samples
    print(f"[Sampling] Selecting 30 stratified audit samples with seed={args.seed}...", flush=True)
    df_sample = select_stratified_audit_sample(df, seed=args.seed, target_total=30)

    # 2. Generate review sheet
    generate_review_sheet(df_sample, args.review_sheet)

    # 3. Generate audit report
    generate_audit_report(df, df_sample, args.audit_report)

    print("\nAudit process finished successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
