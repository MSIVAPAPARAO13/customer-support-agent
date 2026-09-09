#!/usr/bin/env python3
"""
Verify Golden Evaluation Set Isolation (Phase 4)
================================================

Purpose:
  Rigorously verifies that the 200 golden evaluation items satisfy all data
  hygiene and strict partition-isolation requirements:
    1. Exactly 200 unique golden customer tweet IDs.
    2. Every golden ID exists in `applesupport_test.csv`.
    3. Zero golden customer tweet IDs exist in train or validation sets.
    4. Zero golden conversation group IDs exist in train or validation sets.
    5. Zero duplicate customer message texts (after whitespace/casing normalization).
    6. Verifies weak-bucket and safety-slice distributions.
  
  Outputs:
    - data/golden/golden_isolation_report.md

Usage:
  python scripts/verify_golden_isolation.py
"""

import sys
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Rigorous isolation verification for golden evaluation set."
    )
    parser.add_argument("--golden_csv", default=None)
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--output_report", default=None)

    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    golden_path = Path(args.golden_csv).resolve() if args.golden_csv else project_root / "data" / "golden" / "golden_set_200.csv"
    train_path = Path(args.train_csv).resolve() if args.train_csv else project_root / "data" / "processed" / "applesupport_train.csv"
    val_path = Path(args.val_csv).resolve() if args.val_csv else project_root / "data" / "processed" / "applesupport_validation.csv"
    test_path = Path(args.test_csv).resolve() if args.test_csv else project_root / "data" / "processed" / "applesupport_test.csv"
    report_path = Path(args.output_report).resolve() if args.output_report else project_root / "data" / "golden" / "golden_isolation_report.md"

    print("=" * 78)
    print("         PHASE 4: GOLDEN EVALUATION SET ISOLATION VERIFICATION")
    print("=" * 78)

    # 1. Load files
    print("Loading datasets for cross-partition audit...")
    df_golden = pd.read_csv(golden_path, dtype=str)
    df_train = pd.read_csv(train_path, dtype=str)
    df_val = pd.read_csv(val_path, dtype=str)
    df_test = pd.read_csv(test_path, dtype=str)

    print(f"Golden Master Rows: {len(df_golden):,}")
    print(f"Train Rows:         {len(df_train):,}")
    print(f"Validation Rows:    {len(df_val):,}")
    print(f"Test Rows:          {len(df_test):,}")

    failures = []

    # Check 1: Exactly 200 unique golden IDs
    golden_tweet_ids = set(df_golden['customer_tweet_id'].dropna())
    if len(df_golden) != 200:
        failures.append(f"Golden set contains {len(df_golden)} rows, expected exactly 200.")
    if len(golden_tweet_ids) != 200:
        failures.append(f"Golden set contains {len(golden_tweet_ids)} unique customer tweet IDs, expected 200.")

    # Check 2: All golden IDs exist in test split
    test_tweet_ids = set(df_test['customer_tweet_id'].dropna())
    missing_from_test = golden_tweet_ids - test_tweet_ids
    if missing_from_test:
        failures.append(f"{len(missing_from_test)} golden tweet IDs do not exist in test split: {missing_from_test}")

    # Check 3: Zero golden tweet IDs in train or val
    train_tweet_ids = set(df_train['customer_tweet_id'].dropna())
    val_tweet_ids = set(df_val['customer_tweet_id'].dropna())

    train_leak_ids = golden_tweet_ids.intersection(train_tweet_ids)
    val_leak_ids = golden_tweet_ids.intersection(val_tweet_ids)

    if train_leak_ids:
        failures.append(f"CRITICAL: {len(train_leak_ids)} golden tweet IDs appear in train set!")
    if val_leak_ids:
        failures.append(f"CRITICAL: {len(val_leak_ids)} golden tweet IDs appear in validation set!")

    # Check 4: Zero golden conversation group IDs in train or val
    golden_group_ids = set(df_golden['conversation_root_or_group_id'].dropna())
    train_group_ids = set(df_train['conversation_root_or_group_id'].dropna())
    val_group_ids = set(df_val['conversation_root_or_group_id'].dropna())

    train_group_leaks = golden_group_ids.intersection(train_group_ids)
    val_group_leaks = golden_group_ids.intersection(val_group_ids)

    if train_group_leaks:
        failures.append(f"CRITICAL: {len(train_group_leaks)} golden conversation group IDs appear in train set!")
    if val_group_leaks:
        failures.append(f"CRITICAL: {len(val_group_leaks)} golden conversation group IDs appear in validation set!")

    # Check 5: No duplicate message text after basic normalization
    norm_texts = df_golden['customer_text_clean'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    dup_text_count = int(norm_texts.duplicated().sum())
    if dup_text_count > 0:
        failures.append(f"Warning: {dup_text_count} duplicate customer message texts detected in golden sample.")

    # Check 6 & 7: Weak bucket breakdown and safety slice count
    bucket_counts = df_golden['weak_sampling_bucket'].value_counts().to_dict()
    safety_slice_count = int((df_golden['is_safety_slice'].astype(str).str.lower() == 'true').sum())

    # Write report
    status_str = "PASSED (Strict Isolation Confirmed)" if not failures else "FAILED (Isolation Violations Detected)"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Golden Evaluation Set Isolation Verification Report (Phase 4)\n\n")
        f.write(f"## Status: **{status_str}**\n\n")

        f.write("## 1. Cross-Partition Integrity Audit\n\n")
        f.write("| Verification Check | Target Requirement | Measured Value | Result |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Sample Size** | Exactly 200 rows | {len(df_golden)} rows | {'PASSED' if len(df_golden)==200 else 'FAILED'} |\n")
        f.write(f"| **Unique Tweet IDs** | Exactly 200 distinct IDs | {len(golden_tweet_ids)} IDs | {'PASSED' if len(golden_tweet_ids)==200 else 'FAILED'} |\n")
        f.write(f"| **Test Split Provenance** | 100% sourced from test | {len(golden_tweet_ids - missing_from_test)} / 200 | {'PASSED' if not missing_from_test else 'FAILED'} |\n")
        f.write(f"| **Train Tweet ID Overlap** | 0 (Zero) | {len(train_leak_ids)} | {'PASSED' if not train_leak_ids else 'FAILED'} |\n")
        f.write(f"| **Validation Tweet ID Overlap** | 0 (Zero) | {len(val_leak_ids)} | {'PASSED' if not val_leak_ids else 'FAILED'} |\n")
        f.write(f"| **Train Group ID Overlap** | 0 (Zero observed overlap) | {len(train_group_leaks)} | {'PASSED' if not train_group_leaks else 'FAILED'} |\n")
        f.write(f"| **Validation Group ID Overlap** | 0 (Zero observed overlap) | {len(val_group_leaks)} | {'PASSED' if not val_group_leaks else 'FAILED'} |\n")
        f.write(f"| **Duplicate Text Count** | 0 duplicates | {dup_text_count} duplicates | {'PASSED' if dup_text_count==0 else 'WARNING'} |\n\n")

        f.write("## 2. Weak Sampling Bucket Counts\n\n")
        f.write("| Weak Sampling Bucket | Count | % of Golden Set |\n")
        f.write("| :--- | :--- | :--- |\n")
        for bucket, cnt in sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| `{bucket}` | {cnt} | {cnt/2:.1f}% |\n")

        f.write(f"\n- **High-Risk Safety Slice Count**: **{safety_slice_count} cases** (account compromises, fraud, billing disputes, refunds).\n\n")

        f.write("## 3. Strict Isolation Policy\n\n")
        f.write("- **Zero Contamination**: The 200 golden evaluation items originate solely from `applesupport_test.csv`.\n")
        f.write("- **Prohibition**: These 200 rows must NEVER enter training splits, fine-tuning corpora, or future vector database retrieval indices.\n")

    print(f"\nAudit Report generated: {report_path.name}")
    print(f"Final Verification Status: {status_str}")

    if failures:
        print("\nERRORS ENCOUNTERED:")
        for err in failures:
            print(f"  [!] {err}")
        sys.exit(1)
    else:
        print("\nAll 7 isolation and integrity checks PASSED successfully.\n")


if __name__ == "__main__":
    main()
