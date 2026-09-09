#!/usr/bin/env python3
"""
Reconcile Golden Labels & Compute Inter-Annotator Agreement (Phase 4)
====================================================================

Purpose:
  Compares independent human annotation sheets (`annotator_a_blind.csv` and
  `annotator_b_blind.csv`), calculates raw agreement percentages and Cohen's Kappa,
  and exports any disagreements to `adjudication_sheet.csv` for human resolution.

Core Principles:
  1. Pure Python Implementation: Cohen's Kappa is calculated from scratch without
     requiring external machine learning packages (e.g. scikit-learn).
  2. Non-Destructive: Does not fabricate or guess labels. When sheets are in the
     initial unannotated state, reports the pending status and prepares the
     adjudication template cleanly.
  3. Strict Governance: Disagreements are left blank (`final_intent`, `final_action`)
     for a designated human adjudicator to resolve.

Outputs:
  - data/golden/adjudication_sheet.csv

Usage:
  python scripts/reconcile_golden_labels.py
"""

import sys
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Any

import pandas as pd


# -----------------------------------------------------------------------------
# 1. Pure Python Cohen's Kappa Calculator
# -----------------------------------------------------------------------------

def compute_cohens_kappa(labels_a: List[str], labels_b: List[str]) -> Tuple[float, float]:
    """
    Computes observed agreement (p_o) and Cohen's Kappa (kappa) between two annotators.
    Formula:
      kappa = (p_o - p_e) / (1.0 - p_e)
    Where:
      p_o = observed agreement ratio
      p_e = chance agreement ratio = sum(p_a_k * p_b_k)
    """
    n = len(labels_a)
    if n == 0 or len(labels_b) != n:
        return 0.0, 0.0

    # 1. Observed agreement (p_o)
    matches = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = matches / n

    # 2. Expected chance agreement (p_e)
    categories = set(labels_a).union(set(labels_b))
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)

    p_e = sum((count_a[cat] / n) * (count_b[cat] / n) for cat in categories)

    # 3. Kappa calculation
    if abs(1.0 - p_e) < 1e-9:
        # Edge case: perfect agreement on a single category
        kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return p_o, kappa


# -----------------------------------------------------------------------------
# 2. Reconciliation Engine
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Reconcile double-blind golden set annotations."
    )
    parser.add_argument("--sheet_a", default=None, help="Path to annotator_a_blind.csv")
    parser.add_argument("--sheet_b", default=None, help="Path to annotator_b_blind.csv")
    parser.add_argument("--output_csv", default=None, help="Path for adjudication_sheet.csv")

    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    path_a = Path(args.sheet_a).resolve() if args.sheet_a else project_root / "data" / "golden" / "annotator_a_blind.csv"
    path_b = Path(args.sheet_b).resolve() if args.sheet_b else project_root / "data" / "golden" / "annotator_b_blind.csv"
    out_path = Path(args.output_csv).resolve() if args.output_csv else project_root / "data" / "golden" / "adjudication_sheet.csv"

    print("=" * 78)
    print("      PHASE 4: DOUBLE-BLIND GOLDEN LABEL RECONCILIATION & AGREEMENT")
    print("=" * 78)
    print(f"Reading Annotator A Sheet: {path_a.name}")
    print(f"Reading Annotator B Sheet: {path_b.name}")

    if not path_a.exists() or not path_b.exists():
        print(f"Error: Missing annotation sheets in {project_root / 'data' / 'golden'}. Run create_golden_set.py first.")
        sys.exit(1)

    df_a = pd.read_csv(path_a, dtype=str)
    df_b = pd.read_csv(path_b, dtype=str)

    # Align sheets by golden_id
    merged = df_a.merge(
        df_b,
        on=['golden_id', 'customer_tweet_id', 'customer_text_clean'],
        suffixes=('_a', '_b')
    ).sort_values('golden_id').reset_index(drop=True)

    total_items = len(merged)
    print(f"Total Aligned Golden Items: {total_items}")

    # Check if annotations have been recorded
    # Labels are considered pending if human_intent is empty or 'NEEDS_HUMAN_LABEL'
    a_intents = merged['human_intent_a'].fillna("").str.strip()
    b_intents = merged['human_intent_b'].fillna("").str.strip()
    a_actions = merged['human_action_a'].fillna("").str.strip()
    b_actions = merged['human_action_b'].fillna("").str.strip()

    is_annotated_a = a_intents.ne("").sum() > 0
    is_annotated_b = b_intents.ne("").sum() > 0

    if not is_annotated_a or not is_annotated_b:
        print("\n[STATUS: AWAITING HUMAN ANNOTATION]")
        print("Both annotation sheets currently contain blank human-label fields.")
        print(f"Annotator A filled: {a_intents.ne('').sum()} / {total_items}")
        print(f"Annotator B filled: {b_intents.ne('').sum()} / {total_items}")
        print("\nInitializing adjudication template (adjudication_sheet.csv)...")

        adj_rows = []
        for _, row in merged.iterrows():
            adj_rows.append({
                'golden_id': row['golden_id'],
                'customer_tweet_id': row['customer_tweet_id'],
                'customer_text_clean': row['customer_text_clean'],
                'annotator_a_intent': row.get('human_intent_a', ''),
                'annotator_b_intent': row.get('human_intent_b', ''),
                'annotator_a_action': row.get('human_action_a', ''),
                'annotator_b_action': row.get('human_action_b', ''),
                'final_intent': "",
                'final_action': "",
                'final_escalation_reason': "",
                'adjudicator_id': "",
                'adjudication_notes': "",
                'final_status': "PENDING_HUMAN_LABELS"
            })

        adj_df = pd.DataFrame(adj_rows)
        adj_df.to_csv(out_path, index=False)
        print(f"Adjudication template generated: {out_path.name}")
        print("\nINSTRUCTIONS:")
        print("1. Open data/golden/annotator_a_blind.csv and record human intent and action.")
        print("2. Open data/golden/annotator_b_blind.csv and record human intent and action independently.")
        print("3. Re-run 'python scripts/reconcile_golden_labels.py' to compute Cohen's Kappa.")
        print("=" * 78 + "\n")
        return

    # When annotations exist: compute agreement & Cohen's Kappa
    print("\n[Step 1] Computing Inter-Annotator Agreement & Cohen's Kappa...")

    intent_p_o, intent_kappa = compute_cohens_kappa(a_intents.tolist(), b_intents.tolist())
    action_p_o, action_kappa = compute_cohens_kappa(a_actions.tolist(), b_actions.tolist())

    print(f"Intent Observed Agreement: {intent_p_o * 100:.2f}%")
    print(f"Intent Cohen's Kappa:      {intent_kappa:.4f}")
    print(f"Action Observed Agreement: {action_p_o * 100:.2f}%")
    print(f"Action Cohen's Kappa:      {action_kappa:.4f}")

    # Identify disagreements
    disagreements = merged[(a_intents != b_intents) | (a_actions != b_actions)].copy()
    agreements = merged[(a_intents == b_intents) & (a_actions == b_actions)].copy()

    print(f"\nAgreed Rows:    {len(agreements)} / {total_items} ({len(agreements)/total_items*100:.1f}%)")
    print(f"Disputed Rows:  {len(disagreements)} / {total_items} ({len(disagreements)/total_items*100:.1f}%)")

    # Build adjudication sheet
    adj_rows = []
    for _, row in merged.iterrows():
        g_id = row['golden_id']
        i_a = row['human_intent_a']
        i_b = row['human_intent_b']
        act_a = row['human_action_a']
        act_b = row['human_action_b']

        is_disputed = (i_a != i_b) or (act_a != act_b)

        adj_rows.append({
            'golden_id': g_id,
            'customer_tweet_id': row['customer_tweet_id'],
            'customer_text_clean': row['customer_text_clean'],
            'annotator_a_intent': i_a,
            'annotator_b_intent': i_b,
            'annotator_a_action': act_a,
            'annotator_b_action': act_b,
            'final_intent': i_a if not is_disputed else "",
            'final_action': act_a if not is_disputed else "",
            'final_escalation_reason': row.get('escalation_reason_a', '') if not is_disputed else "",
            'adjudicator_id': "CONSENSUS" if not is_disputed else "",
            'adjudication_notes': "Full agreement" if not is_disputed else "Awaiting human adjudication",
            'final_status': "RESOLVED_BY_CONSENSUS" if not is_disputed else "NEEDS_ADJUDICATION"
        })

    adj_df = pd.DataFrame(adj_rows)
    adj_df.to_csv(out_path, index=False)
    print(f"\nAdjudication sheet updated: {out_path.name}")
    print(f"  - Rows resolved by consensus: {len(agreements)}")
    print(f"  - Rows requiring adjudication: {len(disagreements)}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
