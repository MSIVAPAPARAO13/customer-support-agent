#!/usr/bin/env python3
"""
Create Human-Judge Agreement Sheet (Phase 10)
============================================

Purpose:
  Generates a 40-row stratified human review sheet to benchmark agreement
  between human expert judgment and automated LLM-as-a-judge evaluations.

Sampling Protocol:
  - Stratified 40-row sample using seed 42 to balance operational actions
    and intent categories.
  - Blinded to human gold adjudication labels and model confidence probabilities.
  - Generates blank human score columns with `human_review_status = 'NEEDS_HUMAN_REVIEW'`.

Usage:
  python scripts/create_human_judge_agreement_sheet.py
  python scripts/create_human_judge_agreement_sheet.py --input <path_to_agent_predictions>
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


def create_human_agreement_sheet(
    input_file: Path,
    output_file: Path,
    sample_size: int = 40,
    seed: int = 42,
) -> bool:
    """
    Selects a stratified 40-item sample from agent predictions and creates a review template.
    """
    if not input_file.exists():
        print(f"[ERROR] Input agent predictions file not found: {input_file}")
        print("Note: In Phase 10 order, this sheet is created after golden agent predictions exist.")
        return False

    df = pd.read_csv(input_file, low_memory=False)
    total_available = len(df)
    print(f"[DataLoader] Loaded {total_available} records from {input_file.name}")

    action_col = "agent_action" if "agent_action" in df.columns else "action"
    intent_col = "predicted_intent" if "predicted_intent" in df.columns else None

    # Perform stratified sampling
    if total_available <= sample_size:
        sample_df = df.copy()
    else:
        # Stratify by action + intent if available
        if intent_col and intent_col in df.columns and action_col in df.columns:
            strata = df[action_col].astype(str) + "___" + df[intent_col].astype(str)
            # Sample proportionally or with group-based sampling
            sample_indices = []
            rng = np.random.default_rng(seed)
            grouped = df.groupby(strata)

            # Ensure at least 1 from each stratum if possible, then sample remaining
            for name, group in grouped:
                sample_indices.append(rng.choice(group.index, size=1)[0])

            remaining_needed = sample_size - len(sample_indices)
            remaining_indices = [i for i in df.index if i not in sample_indices]

            if remaining_needed > 0 and len(remaining_indices) >= remaining_needed:
                extra_picks = rng.choice(remaining_indices, size=remaining_needed, replace=False)
                sample_indices.extend(extra_picks)
            elif remaining_needed < 0:
                sample_indices = rng.choice(sample_indices, size=sample_size, replace=False).tolist()

            sample_df = df.loc[sample_indices].copy().reset_index(drop=True)
        else:
            sample_df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # Enforce exactly sample_size rows
    if len(sample_df) > sample_size:
        sample_df = sample_df.head(sample_size)

    # Format output sheet with blank human evaluation columns
    records = []
    for idx, row in sample_df.iterrows():
        sample_id = f"agreement_{idx + 1:02d}"
        golden_id = row.get("golden_id", f"sample_{idx + 1:02d}")
        tweet_id = row.get("customer_tweet_id", "")
        text = row.get("customer_text_clean", "")
        action = row.get(action_col, "")
        reasons = row.get("action_reason_codes", row.get("all_reason_codes", ""))
        draft = row.get("draft_reply", "")
        evidence = row.get("evidence_snippets", row.get("grounding_note", ""))

        records.append({
            "sample_id": sample_id,
            "golden_id": golden_id,
            "customer_tweet_id": tweet_id,
            "customer_text_clean": text,
            "agent_action": action,
            "action_reason_codes": reasons,
            "draft_reply": draft,
            "retrieved_evidence_snippets": evidence,
            # Human Review Fields (Intentionally Blank)
            "human_relevance": "",
            "human_historical_grounding": "",
            "human_safety_and_privacy": "",
            "human_clear_next_step": "",
            "human_routing_appropriateness": "",
            "human_overall_accept": "",
            "human_reason_codes": "",
            "human_reviewer_notes": "",
            "human_review_status": "NEEDS_HUMAN_REVIEW",
        })

    out_df = pd.DataFrame(records)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\n[AGREEMENT SHEET CREATED] Successfully wrote {len(out_df)} rows to:")
    print(f"  {output_file}")
    print("\nReview Status Breakdown:")
    print(out_df["human_review_status"].value_counts().to_string())
    print("\nAction Breakdown in Sample:")
    print(out_df["agent_action"].value_counts().to_string())
    return True


def main():
    parser = argparse.ArgumentParser(description="Create Human-Judge Agreement Review Sheet (Phase 10).")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "outputs" / "final_evaluation" / "golden_agent_predictions.csv",
        help="Path to agent predictions CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "final_evaluation" / "human_judge_agreement_sheet.csv",
        help="Path to output human agreement sheet.",
    )
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # If golden_agent_predictions does not exist yet, allow fallback to validation_agent_decisions for setup/testing
    input_file = args.input
    if not input_file.exists():
        fallback = REPO_ROOT / "outputs" / "agent" / "validation_agent_decisions.csv"
        if fallback.exists():
            print(f"[NOTICE] Golden predictions not found. Initializing agreement template from {fallback.name}.")
            input_file = fallback

    create_human_agreement_sheet(
        input_file=input_file,
        output_file=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
