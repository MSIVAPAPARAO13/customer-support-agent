#!/usr/bin/env python3
"""
Create Golden Evaluation Set (Phase 4)
=====================================

Purpose:
  Deterministically samples exactly 200 unique customer inquiries from the protected
  test partition (`applesupport_test.csv`) to create a human-labeled evaluation benchmark.

Core Principles:
  1. Test Isolation: Sourced STRICTLY from `applesupport_test.csv`. Golden rows must
     NEVER appear in train/val splits and must NEVER enter future retrieval indexes.
  2. Double-Blind Sheets: Generates two separate annotation sheets (`annotator_a_blind.csv`
     and `annotator_b_blind.csv`) in different randomized orders. All historical brand
     replies, sampling buckets, and model predictions are completely omitted.
  3. Weak Sampling Buckets: Uses approximate keyword heuristics (`weak_sampling_bucket`)
     purely to enforce diversity across issue types and guarantee inclusion of critical
     safety/security cases (`is_safety_slice`). These buckets are NEVER treated as labels.
  4. Zero Fabrication: All human-label fields are initialized as empty with status
     `NEEDS_HUMAN_LABEL`.

Deliverables:
  - data/golden/golden_set_200.csv
  - data/golden/annotator_a_blind.csv
  - data/golden/annotator_b_blind.csv
  - data/golden/golden_set_sampling_report.md

Usage:
  python scripts/create_golden_set.py
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# 1. Weak Sampling Bucket Categorization & Safety Slicing
# -----------------------------------------------------------------------------

def assign_weak_sampling_bucket(text: str) -> str:
    """
    Assigns a customer inquiry to a broad heuristic bucket solely to ensure
    topical diversity during test sampling. These are NOT intent labels.
    """
    t = text.lower()
    if re.search(r'\b(?:apple id|password|passcode|locked|2fa|verification code|security question|hacked|stolen|unauthorized)\b', t):
        return 'account_access_bucket'
    if re.search(r'\b(?:charged|charge|refund|subscription|billing|receipt|cancel|money|purchased|payment|itunes\.com/bill)\b', t):
        return 'billing_purchase_bucket'
    if re.search(r'\b(?:ios 11|update|updating|updated|install|installation|upgrade|version|firmware)\b', t):
        return 'software_update_bucket'
    if re.search(r'\b(?:battery|charging|charger|screen|display|touch|overheat|overheating|crack|cracked|power|shutting down|dies)\b', t):
        return 'device_hardware_bucket'
    if re.search(r'\b(?:wifi|wi-fi|bluetooth|cellular|lte|data|signal|carrier|hotspot|airdrop)\b', t):
        return 'connectivity_network_bucket'
    if re.search(r'\b(?:app store|icloud|apple music|music|imessage|facetime|notes|photos|sync|download)\b', t):
        return 'apps_icloud_bucket'
    if re.search(r'\b(?:genius bar|apple store|repair|appointment|warranty|applecare|order|shipping|delivery)\b', t):
        return 'repair_order_bucket'
    return 'unclear_general_bucket'


def is_safety_critical(text: str) -> bool:
    """
    Flags messages containing high-risk keywords (security breaches, fraud,
    unauthorized payments, refunds) to ensure they are represented in evaluation.
    """
    t = text.lower()
    pattern = r'\b(?:hacked|stolen|compromised|unauthorized|fraud|refund|charged twice|payment dispute|chargeback|locked out)\b'
    return bool(re.search(pattern, t))


# -----------------------------------------------------------------------------
# 2. Main Sampling Pipeline
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Create 200-row human golden evaluation set from test split."
    )
    parser.add_argument(
        "--test_csv",
        default=None,
        help="Path to applesupport_test.csv (defaults to data/processed/applesupport_test.csv)."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Target output directory for golden set files (defaults to data/golden/)."
    )

    args = parser.parse_args()

    # Path resolution
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    test_path = Path(args.test_csv).resolve() if args.test_csv else project_root / "data" / "processed" / "applesupport_test.csv"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else project_root / "data" / "golden"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("       PHASE 4: HUMAN GOLDEN EVALUATION SET CREATION (200 ROWS)")
    print("=" * 78)
    print(f"Reading protected test split: {test_path}")

    if not test_path.exists():
        print(f"Error: {test_path} not found. Run Phase 2 first.")
        sys.exit(1)

    df_test = pd.read_csv(test_path, dtype=str)
    print(f"Total test split rows: {len(df_test):,}")

    # Ensure we use usable rows and unique customer tweet IDs
    df_usable = df_test[df_test['is_usable'].astype(str).str.lower() == 'true'].copy()
    unique_cust = df_usable.drop_duplicates(subset=['customer_tweet_id']).copy()
    print(f"Unique usable customer inquiries in test split: {len(unique_cust):,}")

    # Step 1: Assign weak sampling buckets and safety flags
    print("\n[Step 1] Categorizing candidate inquiries into weak sampling buckets...")
    unique_cust['weak_sampling_bucket'] = unique_cust['customer_text_clean'].apply(assign_weak_sampling_bucket)
    unique_cust['is_safety_slice'] = unique_cust['customer_text_clean'].apply(is_safety_critical)

    # Step 2: Stratified sampling targeting exactly 200 unique customer messages
    print("\n[Step 2] Executing deterministic stratified sampling (seed=42)...")
    target_per_bucket = {
        'software_update_bucket': 35,
        'device_hardware_bucket': 35,
        'apps_icloud_bucket': 25,
        'connectivity_network_bucket': 20,
        'account_access_bucket': 20,
        'billing_purchase_bucket': 20,
        'repair_order_bucket': 20,
        'unclear_general_bucket': 25
    }
    # Total = 35+35+25+20+20+20+20+25 = 200

    sampled_dfs = []
    for bucket_name, target_n in target_per_bucket.items():
        sub = unique_cust[unique_cust['weak_sampling_bucket'] == bucket_name]
        sub_safety = sub[sub['is_safety_slice']]
        sub_other = sub[~sub['is_safety_slice']]

        # Guarantee safety slice representation
        n_safety = min(len(sub_safety), max(2, int(target_n * 0.3))) if len(sub_safety) > 0 else 0
        n_other = target_n - n_safety

        s_part1 = sub_safety.sample(n=n_safety, random_state=42) if n_safety > 0 else pd.DataFrame()
        s_part2 = sub_other.sample(n=n_other, random_state=42) if n_other > 0 else pd.DataFrame()

        sampled_dfs.append(pd.concat([s_part1, s_part2]))

    # Combine and perform deterministic shuffle with seed 42
    golden_master = pd.concat(sampled_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    assert len(golden_master) == 200, f"Expected 200 samples, got {len(golden_master)}"

    # Assign golden IDs
    golden_master['golden_id'] = [f"golden_{i+1:03d}" for i in range(len(golden_master))]

    # Order master columns
    master_cols = [
        'golden_id',
        'customer_tweet_id',
        'conversation_root_or_group_id',
        'customer_text_clean',
        'customer_text_raw',
        'weak_sampling_bucket',
        'is_safety_slice'
    ]
    master_csv_path = output_dir / "golden_set_200.csv"
    golden_master[master_cols].to_csv(master_csv_path, index=False)
    print(f"Master golden set saved: {master_csv_path.name} (200 rows)")

    # Step 3: Build blind human annotation sheets
    print("\n[Step 3] Building double-blind annotation sheets for Annotator A and Annotator B...")

    blind_cols = [
        'golden_id',
        'customer_tweet_id',
        'customer_text_clean',
        'human_intent',
        'human_action',
        'escalation_reason',
        'annotation_notes',
        'annotator_id',
        'annotation_timestamp',
        'status'
    ]

    # Annotator A: randomized with seed 101
    sheet_a = golden_master[['golden_id', 'customer_tweet_id', 'customer_text_clean']].copy()
    sheet_a = sheet_a.sample(frac=1.0, random_state=101).reset_index(drop=True)
    sheet_a['human_intent'] = ""
    sheet_a['human_action'] = ""
    sheet_a['escalation_reason'] = ""
    sheet_a['annotation_notes'] = ""
    sheet_a['annotator_id'] = ""
    sheet_a['annotation_timestamp'] = ""
    sheet_a['status'] = "NEEDS_HUMAN_LABEL"

    sheet_a_path = output_dir / "annotator_a_blind.csv"
    sheet_a[blind_cols].to_csv(sheet_a_path, index=False)

    # Annotator B: randomized with seed 202 (different display order!)
    sheet_b = golden_master[['golden_id', 'customer_tweet_id', 'customer_text_clean']].copy()
    sheet_b = sheet_b.sample(frac=1.0, random_state=202).reset_index(drop=True)
    sheet_b['human_intent'] = ""
    sheet_b['human_action'] = ""
    sheet_b['escalation_reason'] = ""
    sheet_b['annotation_notes'] = ""
    sheet_b['annotator_id'] = ""
    sheet_b['annotation_timestamp'] = ""
    sheet_b['status'] = "NEEDS_HUMAN_LABEL"

    sheet_b_path = output_dir / "annotator_b_blind.csv"
    sheet_b[blind_cols].to_csv(sheet_b_path, index=False)

    print(f"Annotator A blind sheet: {sheet_a_path.name} (shuffled order, seed 101)")
    print(f"Annotator B blind sheet: {sheet_b_path.name} (shuffled order, seed 202)")

    # Step 4: Generate sampling report
    print("\n[Step 4] Writing golden_set_sampling_report.md...")
    report_path = output_dir / "golden_set_sampling_report.md"

    bucket_counts = golden_master['weak_sampling_bucket'].value_counts().to_dict()
    safety_count = int(golden_master['is_safety_slice'].sum())

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Golden Evaluation Set Sampling Audit Report (Phase 4)\n\n")

        f.write("## 1. Sampling Parameters & Provenance\n\n")
        f.write(f"- **Source Dataset**: Strictly `data/processed/applesupport_test.csv` (10,279 rows).\n")
        f.write(f"- **Unique Usable Test Candidates**: {len(unique_cust):,} unique customer tweets.\n")
        f.write(f"- **Sample Size**: Exactly **200 unique customer tweets** (`golden_001` through `golden_200`).\n")
        f.write(f"- **Random Seed**: `42` (deterministic reproducibility).\n")
        f.write(f"- **Safety Slice Count**: **{safety_count} critical safety/fraud/billing cases**.\n")
        f.write(f"- **Unique Tweet ID Integrity**: 200 distinct IDs (0 duplicates).\n\n")

        f.write("## 2. Weak Sampling Bucket Distribution\n\n")
        f.write("> [!NOTE]\n")
        f.write("> **Weak Bucket Disclaimer**: These buckets were used solely to enforce topical diversity during sampling. ")
        f.write("They were **never shown to human annotators** and are **never treated as ground-truth intent labels**.\n\n")

        f.write("| Weak Sampling Bucket | Count in Golden Set | % of Golden Set | Sampling Objective |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for bucket, cnt in sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (cnt / 200) * 100
            f.write(f"| `{bucket}` | {cnt} | {pct:.1f}% | Broad diversity coverage |\n")

        f.write("\n## 3. High-Priority Safety Slice\n\n")
        f.write(f"Exactly **{safety_count} messages** contain high-liability keywords related to account recovery, compromised credentials, unauthorized credit card charges, and payment disputes. ")
        f.write("These cases evaluate whether the automated agent safely escalates sensitive emergencies to human specialists.\n\n")

        f.write("## 4. Double-Blind Protocol Integrity\n\n")
        f.write("1. **Historical Brand Replies Omitted**: Annotators see only the customer text, ensuring they evaluate customer intent rather than Apple agent responses.\n")
        f.write("2. **Model Predictions Omitted**: Zero synthetic, keyword, or model predictions appear in the annotation sheets to prevent anchoring bias.\n")
        f.write("3. **Independent Shuffling**: Annotator A (seed 101) and Annotator B (seed 202) receive different presentation sequences to prevent order-effect correlation.\n")

    print(f"Sampling report saved: {report_path.name}")
    print("\n" + "=" * 78)
    print("                     GOLDEN SET CREATION COMPLETE")
    print("=" * 78)
    print(f"Sampled Messages:       200 unique customer inquiries from test split")
    print(f"Safety Slice Items:     {safety_count} critical security/billing cases")
    print(f"Double-Blind Files:     annotator_a_blind.csv & annotator_b_blind.csv")
    print(f"Next Action:            Human annotators must independently label blind sheets.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
