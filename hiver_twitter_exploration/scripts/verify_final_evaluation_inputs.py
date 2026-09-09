#!/usr/bin/env python3
"""
Verify Final Evaluation Inputs & Golden Set Readiness (Phase 10)
================================================================

Purpose:
  Validates that the protected 200-row human golden benchmark in
  `data/golden/adjudication_sheet.csv` meets all 8 strict integrity criteria
  before golden inference or final metric calculation is permitted.

Verification Criteria:
  1. Exactly 200 unique golden IDs.
  2. All 200 rows have final_status == 'DONE'.
  3. All final_intent values belong to the 8 approved taxonomy categories.
  4. All final_action values are either 'auto_handle' or 'escalate'.
  5. No missing final label fields (final_intent, final_action; escalation reason if escalate).
  6. Golden tweet IDs exist only in the protected test split (applesupport_test.csv).
  7. Zero golden tweet-ID overlap with train or validation splits.
  8. Zero golden conversation-group overlap with train or validation splits.

Failure Behavior:
  Exits with code 1 and a detailed diagnostic explanation if any rule is broken.
  Does NOT compute or report headline evaluation metrics.

Usage:
  python scripts/verify_final_evaluation_inputs.py
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

import pandas as pd

# Configure Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent

# Approved 8-intent taxonomy
APPROVED_INTENTS = {
    "software_update_or_os_issue",
    "device_performance_or_hardware",
    "connectivity_and_network",
    "apps_services_or_icloud",
    "account_access_and_apple_id",
    "billing_subscription_or_purchase",
    "repair_replacement_or_order",
    "other_or_unclear",
}

APPROVED_ACTIONS = {"auto_handle", "escalate"}


def verify_golden_inputs(
    adjudication_path: Path = REPO_ROOT / "data" / "golden" / "adjudication_sheet.csv",
    golden_set_path: Path = REPO_ROOT / "data" / "golden" / "golden_set_200.csv",
    train_path: Path = REPO_ROOT / "data" / "processed" / "applesupport_train.csv",
    val_path: Path = REPO_ROOT / "data" / "processed" / "applesupport_validation.csv",
    test_path: Path = REPO_ROOT / "data" / "processed" / "applesupport_test.csv",
) -> Tuple[bool, List[str]]:
    """
    Executes the 8-point verification suite against golden benchmark inputs.
    Returns (all_passed: bool, failure_messages: List[str]).
    """
    failures = []

    if not adjudication_path.exists():
        return False, [f"Missing adjudication sheet at {adjudication_path}"]
    if not golden_set_path.exists():
        return False, [f"Missing master golden set at {golden_set_path}"]
    if not test_path.exists():
        return False, [f"Missing test partition at {test_path}"]

    df_adj = pd.read_csv(adjudication_path, low_memory=False)
    df_gold = pd.read_csv(golden_set_path, low_memory=False)

    # Check 1: Exactly 200 unique golden IDs
    golden_ids = df_adj["golden_id"].dropna().tolist()
    if len(golden_ids) != 200 or len(set(golden_ids)) != 200:
        failures.append(
            f"Check 1 Failed: Expected exactly 200 unique golden IDs in adjudication sheet, "
            f"found {len(golden_ids)} rows ({len(set(golden_ids))} unique)."
        )

    # Check 2: All final_status values are 'DONE'
    status_counts = df_adj["final_status"].fillna("MISSING").value_counts().to_dict()
    done_count = status_counts.get("DONE", 0)
    if done_count != 200:
        failures.append(
            f"Check 2 Failed: All 200 rows must have final_status == 'DONE'. "
            f"Currently: {done_count}/200 are 'DONE'. Breakdown: {status_counts}. "
            f"Human adjudication is incomplete."
        )

    # Check 3: All final_intent values are valid taxonomy labels
    intents = df_adj["final_intent"].dropna().tolist()
    invalid_intents = [i for i in intents if i not in APPROVED_INTENTS]
    if len(intents) != 200 or invalid_intents:
        failures.append(
            f"Check 3 Failed: All 200 rows must have valid final_intent taxonomy categories. "
            f"Missing count: {200 - len(intents)}. Invalid labels found: {set(invalid_intents)}."
        )

    # Check 4: All final_action values are 'auto_handle' or 'escalate'
    actions = df_adj["final_action"].dropna().tolist()
    invalid_actions = [a for a in actions if a not in APPROVED_ACTIONS]
    if len(actions) != 200 or invalid_actions:
        failures.append(
            f"Check 4 Failed: All 200 rows must have final_action in {APPROVED_ACTIONS}. "
            f"Missing count: {200 - len(actions)}. Invalid actions found: {set(invalid_actions)}."
        )

    # Check 5: No missing final label fields
    missing_fields_mask = df_adj["final_intent"].isna() | df_adj["final_action"].isna()
    if missing_fields_mask.any():
        bad_ids = df_adj.loc[missing_fields_mask, "golden_id"].tolist()
        failures.append(f"Check 5 Failed: Found rows with missing final labels: {bad_ids[:5]}.")

    # Escalation reason check for escalate actions
    escalate_mask = df_adj["final_action"] == "escalate"
    missing_reason_mask = escalate_mask & df_adj["final_escalation_reason"].isna()
    if missing_reason_mask.any() and done_count == 200:
        bad_reason_ids = df_adj.loc[missing_reason_mask, "golden_id"].tolist()
        failures.append(f"Check 5 Failed: Escalate rows missing final_escalation_reason: {bad_reason_ids[:5]}.")

    # Check 6: Golden tweet IDs exist only in test split
    df_test = pd.read_csv(test_path, low_memory=False, usecols=["customer_tweet_id"])
    test_tweet_ids = set(df_test["customer_tweet_id"].astype(str))
    gold_tweet_ids = set(df_gold["customer_tweet_id"].astype(str))
    missing_from_test = gold_tweet_ids - test_tweet_ids
    if missing_from_test:
        failures.append(f"Check 6 Failed: Golden tweet IDs not found in test split: {len(missing_from_test)} rows.")

    # Check 7 & 8: Zero tweet-ID and conversation-group overlap with train or val
    if train_path.exists() and val_path.exists():
        df_train = pd.read_csv(train_path, low_memory=False, usecols=["customer_tweet_id", "conversation_root_or_group_id"])
        df_val = pd.read_csv(val_path, low_memory=False, usecols=["customer_tweet_id", "conversation_root_or_group_id"])

        train_cids = set(df_train["customer_tweet_id"].astype(str))
        val_cids = set(df_val["customer_tweet_id"].astype(str))
        train_gids = set(df_train["conversation_root_or_group_id"].astype(str))
        val_gids = set(df_val["conversation_root_or_group_id"].astype(str))

        gold_gids = set(df_gold["conversation_root_or_group_id"].astype(str))

        # Check 7: Tweet ID overlap
        cid_train_leak = gold_tweet_ids.intersection(train_cids)
        cid_val_leak = gold_tweet_ids.intersection(val_cids)
        if cid_train_leak or cid_val_leak:
            failures.append(
                f"Check 7 Failed: Tweet ID leakage detected! Train overlap: {len(cid_train_leak)}, "
                f"Val overlap: {len(cid_val_leak)}."
            )

        # Check 8: Conversation Group ID overlap
        gid_train_leak = gold_gids.intersection(train_gids)
        gid_val_leak = gold_gids.intersection(val_gids)
        if gid_train_leak or gid_val_leak:
            failures.append(
                f"Check 8 Failed: Conversation group leakage detected! Train overlap: {len(gid_train_leak)}, "
                f"Val overlap: {len(gid_val_leak)}."
            )

    return (len(failures) == 0), failures


def main():
    parser = argparse.ArgumentParser(description="Verify Final Evaluation Inputs (Phase 10).")
    parser.add_argument("--adjudication-file", type=Path, default=REPO_ROOT / "data" / "golden" / "adjudication_sheet.csv")
    parser.add_argument("--golden-master", type=Path, default=REPO_ROOT / "data" / "golden" / "golden_set_200.csv")
    args = parser.parse_args()

    print("=" * 75)
    print("PHASE 10: VERIFY FINAL EVALUATION INPUTS & GOLDEN READINESS")
    print("=" * 75)
    print(f"Adjudication Sheet : {args.adjudication_file}")
    print(f"Golden Master File : {args.golden_master}")
    print("=" * 75)

    passed, failures = verify_golden_inputs(args.adjudication_file, args.golden_master)

    if passed:
        print("\n[VERIFICATION PASSED] All 8 integrity criteria satisfied.")
        print("Golden labels are complete, validated, and isolated from training/validation.")
        print("Ready for label freezing and final evaluation.")
        sys.exit(0)
    else:
        print("\n[VERIFICATION FAILED] Golden evaluation set is NOT ready for evaluation.")
        print("The evaluation harness refuses to calculate final headline metrics due to the following:")
        for idx, msg in enumerate(failures, start=1):
            print(f"  [{idx}] {msg}")
        print("\nIntegrity Safeguard Enforced: Golden evaluation will NOT run until human labels are complete.")
        sys.exit(1)


if __name__ == "__main__":
    main()
