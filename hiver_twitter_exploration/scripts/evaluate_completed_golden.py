#!/usr/bin/env python3
"""
Evaluate Completed Golden Evaluation Set (Phase 5)
==================================================

Purpose:
  Computes standardized performance, safety, and operational routing metrics
  for intent classification and dialogue action routing against the protected
  200-row human-annotated golden set (`data/golden/golden_set_200.csv` and
  `data/golden/adjudication_sheet.csv`).

CRITICAL INTEGRITY ENFORCEMENT:
  This script implements strict safety guardrails and will REFUSE to execute unless:
    1. All 200 rows in `adjudication_sheet.csv` have `final_status == 'DONE'`.
    2. All 200 rows have valid, non-empty `final_intent` and `final_action`.
    3. Golden isolation verification passes (zero overlap with training/validation splits).
    4. Model predictions file contains every single golden_id exactly once (no missing, no dupes).

Calculated Metrics (when human adjudication is complete):
  - Overall Intent Accuracy
  - Macro-averaged F1 Score
  - Per-Intent Precision, Recall, F1, and Support
  - Intent Confusion Matrix (Truth vs. Prediction)
  - Action Accuracy (if predicted_action provided)
  - Unsafe Auto-Handle Count (model auto-handled when true label required human escalation)
  - Auto-Handle Coverage (% of queries handled without human escalation)
  - Selective Accuracy (accuracy strictly on auto-handled cohort)

Dependencies:
  Python standard library and Pandas only. No external ML dependencies.

Usage:
  python scripts/evaluate_completed_golden.py --predictions <path_to_predictions_csv>
"""

import sys
import os
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

# Ensure Windows stdout handles UTF-8 gracefully
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent

# Approved Taxonomy Categories
APPROVED_INTENTS = [
    "software_update_or_os_issue",
    "device_performance_or_hardware",
    "connectivity_and_network",
    "apps_services_or_icloud",
    "account_access_and_apple_id",
    "billing_subscription_or_purchase",
    "repair_replacement_or_order",
    "other_or_unclear"
]

APPROVED_ACTIONS = [
    "direct_troubleshoot",
    "route_to_specialist",
    "request_device_or_os_details",
    "escalate_to_human_agent",
    "other_or_unclear"
]


class GoldenIntegrityViolationError(Exception):
    """Raised when safety preconditions for golden set evaluation are not satisfied."""
    pass


def verify_golden_isolation(repo_root: Path) -> bool:
    """
    Verifies that no golden rows overlap with training or validation partitions.
    """
    golden_path = repo_root / "data" / "golden" / "golden_set_200.csv"
    train_path = repo_root / "data" / "processed" / "applesupport_train.csv"
    val_path = repo_root / "data" / "processed" / "applesupport_validation.csv"

    if not golden_path.exists():
        print(f"[Isolation Check] ERROR: {golden_path} does not exist.")
        return False

    golden_df = pd.read_csv(golden_path, dtype=str)
    golden_ids = set(golden_df['customer_tweet_id'])
    golden_groups = set(golden_df['conversation_root_or_group_id'].dropna())

    if train_path.exists():
        train_df = pd.read_csv(train_path, usecols=['customer_tweet_id', 'conversation_root_or_group_id'], dtype=str)
        t_id_overlap = golden_ids.intersection(set(train_df['customer_tweet_id']))
        t_grp_overlap = golden_groups.intersection(set(train_df['conversation_root_or_group_id'].dropna()))
        if t_id_overlap or t_grp_overlap:
            print(f"[Isolation Check] VIOLATION: Golden set leaks into training partition! "
                  f"IDs: {len(t_id_overlap)}, Groups: {len(t_grp_overlap)}")
            return False

    if val_path.exists():
        val_df = pd.read_csv(val_path, usecols=['customer_tweet_id', 'conversation_root_or_group_id'], dtype=str)
        v_id_overlap = golden_ids.intersection(set(val_df['customer_tweet_id']))
        v_grp_overlap = golden_groups.intersection(set(val_df['conversation_root_or_group_id'].dropna()))
        if v_id_overlap or v_grp_overlap:
            print(f"[Isolation Check] VIOLATION: Golden set leaks into validation partition! "
                  f"IDs: {len(v_id_overlap)}, Groups: {len(v_grp_overlap)}")
            return False

    return True


def check_adjudication_completeness(adjudication_path: Path) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validates that the human adjudication sheet is 100% complete.
    Returns: (is_valid, reason_message, loaded_df)
    """
    if not adjudication_path.exists():
        return False, f"Adjudication file does not exist at: {adjudication_path}", None

    df = pd.read_csv(adjudication_path, dtype=str)
    
    if len(df) != 200:
        return False, f"Adjudication sheet contains {len(df)} rows; exactly 200 rows expected.", None

    required_cols = ['golden_id', 'customer_tweet_id', 'final_intent', 'final_action', 'final_status']
    for col in required_cols:
        if col not in df.columns:
            return False, f"Missing required column '{col}' in adjudication sheet.", None

    # Condition 1: final_status == 'DONE' on all 200 rows
    non_done_mask = df['final_status'] != 'DONE'
    non_done_count = non_done_mask.sum()
    if non_done_count > 0:
        statuses = df['final_status'].value_counts().to_dict()
        return False, (
            f"Adjudication is INCOMPLETE. Found {non_done_count}/200 rows where final_status != 'DONE'. "
            f"Current status breakdown: {statuses}"
        ), None

    # Condition 2: valid non-empty final_intent and final_action
    missing_intent = df['final_intent'].isna() | (df['final_intent'].str.strip() == '')
    if missing_intent.any():
        return False, f"Found {missing_intent.sum()} rows with missing final_intent.", None

    missing_action = df['final_action'].isna() | (df['final_action'].str.strip() == '')
    if missing_action.any():
        return False, f"Found {missing_action.sum()} rows with missing final_action.", None

    # Validate against approved taxonomy lists
    invalid_intents = set(df['final_intent'].str.strip()) - set(APPROVED_INTENTS)
    if invalid_intents:
        return False, f"Found invalid intent labels not in taxonomy: {invalid_intents}", None

    invalid_actions = set(df['final_action'].str.strip()) - set(APPROVED_ACTIONS)
    if invalid_actions:
        return False, f"Found invalid action labels: {invalid_actions}", None

    return True, "All 200 rows adjudicated and validated.", df


def validate_predictions_format(
    predictions_path: Path,
    golden_df: pd.DataFrame
) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validates that model predictions file contains every golden_id exactly once.
    """
    if not predictions_path.exists():
        return False, f"Predictions file does not exist: {predictions_path}", None

    pred_df = pd.read_csv(predictions_path, dtype=str)

    # Check mapping identifier: either golden_id or customer_tweet_id
    if 'golden_id' not in pred_df.columns:
        if 'customer_tweet_id' in pred_df.columns:
            # Join golden_id from golden_df
            id_map = dict(zip(golden_df['customer_tweet_id'], golden_df['golden_id']))
            pred_df['golden_id'] = pred_df['customer_tweet_id'].map(id_map)
        else:
            return False, "Predictions file must contain either 'golden_id' or 'customer_tweet_id'.", None

    if 'predicted_intent' not in pred_df.columns:
        return False, "Predictions file must contain 'predicted_intent' column.", None

    expected_ids = set(golden_df['golden_id'])
    pred_ids = list(pred_df['golden_id'].dropna())

    if len(pred_ids) != 200:
        return False, f"Predictions file contains {len(pred_ids)} matched golden rows; exactly 200 required.", None

    if len(set(pred_ids)) != 200:
        return False, "Predictions file contains duplicate golden_id entries.", None

    missing_ids = expected_ids - set(pred_ids)
    if missing_ids:
        return False, f"Predictions file is missing {len(missing_ids)} golden IDs (e.g. {list(missing_ids)[:3]}).", None

    return True, "Predictions format valid.", pred_df


# -----------------------------------------------------------------------------
# Metric Calculations (Pure Python / Pandas)
# -----------------------------------------------------------------------------

def calculate_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str]
) -> pd.DataFrame:
    """Computes full confusion matrix: rows=True, cols=Predicted."""
    matrix = {c: [0] * len(classes) for c in classes}
    class_idx = {c: i for i, c in enumerate(classes)}

    for true_lbl, pred_lbl in zip(y_true, y_pred):
        if true_lbl in class_idx and pred_lbl in class_idx:
            t_idx = class_idx[true_lbl]
            matrix[pred_lbl][t_idx] += 1
        elif true_lbl in class_idx:
            # Unknown prediction defaults to other_or_unclear column if exists
            fallback = "other_or_unclear" if "other_or_unclear" in class_idx else classes[-1]
            matrix[fallback][class_idx[true_lbl]] += 1

    df_cm = pd.DataFrame(matrix, index=classes)
    df_cm.index.name = "True \\ Predicted"
    return df_cm


def compute_metrics(
    eval_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Computes all standard classification, macro F1, and action safety metrics.
    eval_df must contain:
      - final_intent
      - predicted_intent
      - final_action (optional for action metrics)
      - predicted_action (optional)
    """
    y_true = list(eval_df['final_intent'].str.strip())
    y_pred = list(eval_df['predicted_intent'].str.strip())
    n = len(y_true)

    # 1. Overall Intent Accuracy
    correct_intent = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    intent_accuracy = correct_intent / n if n > 0 else 0.0

    # 2. Per-intent Precision, Recall, F1
    per_intent = {}
    f1_scores = []
    
    for cls in APPROVED_INTENTS:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
        support = sum(1 for yt in y_true if yt == cls)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_intent[cls] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support
        }
        f1_scores.append(f1)

    # 3. Macro F1
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # 4. Confusion Matrix
    cm_df = calculate_confusion_matrix(y_true, y_pred, APPROVED_INTENTS)

    # 5. Action and Safety Metrics
    has_action_eval = 'final_action' in eval_df.columns and 'predicted_action' in eval_df.columns
    action_metrics = {}

    if has_action_eval:
        a_true = list(eval_df['final_action'].str.strip())
        a_pred = list(eval_df['predicted_action'].str.strip())

        # Action Accuracy
        correct_actions = sum(1 for at, ap in zip(a_true, a_pred) if at == ap)
        action_acc = correct_actions / n if n > 0 else 0.0

        # Auto-handle definition:
        # In customer support, 'escalate_to_human_agent' requires a human representative.
        # Any other action (e.g. direct_troubleshoot, request_device_details) represents
        # an automated handling path.
        auto_handled_indices = [i for i, ap in enumerate(a_pred) if ap != 'escalate_to_human_agent']
        auto_handle_count = len(auto_handled_indices)
        auto_handle_coverage = auto_handle_count / n if n > 0 else 0.0

        # Unsafe Auto-Handle Count:
        # Model predicted an automated action, but true label required human escalation!
        unsafe_indices = [i for i in auto_handled_indices if a_true[i] == 'escalate_to_human_agent']
        unsafe_count = len(unsafe_indices)

        # Selective Accuracy among Auto-Handled cases:
        # Intent accuracy restricted exclusively to auto-handled cases
        if auto_handle_count > 0:
            sel_correct = sum(1 for i in auto_handled_indices if y_true[i] == y_pred[i])
            selective_accuracy = sel_correct / auto_handle_count
        else:
            selective_accuracy = 0.0

        action_metrics = {
            "action_accuracy": action_acc,
            "unsafe_auto_handle_count": unsafe_count,
            "auto_handle_coverage": auto_handle_coverage,
            "selective_accuracy": selective_accuracy,
            "auto_handle_total": auto_handle_count
        }

    return {
        "intent_accuracy": intent_accuracy,
        "macro_f1": macro_f1,
        "per_intent": per_intent,
        "confusion_matrix": cm_df,
        "has_action_eval": has_action_eval,
        "action_metrics": action_metrics,
        "sample_count": n
    }


def df_to_markdown(df: pd.DataFrame) -> str:
    """Renders a pandas DataFrame to clean GitHub Flavored Markdown table without tabulate."""
    headers = [str(df.index.name or "")] + [str(c) for c in df.columns]
    rows = []
    for idx, row in df.iterrows():
        rows.append([str(idx)] + [str(val) for val in row.values])
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(val))
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    row_lines = ["| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |" for r in rows]
    return "\n".join([header_line, sep_line] + row_lines)


def format_markdown_report(metrics: Dict[str, Any], model_name: str) -> str:
    """Generates formatted markdown evaluation report."""
    lines = []
    lines.append(f"# Benchmark Evaluation Report: {model_name}\n")
    lines.append("## 1. Overall Performance Summary\n")
    lines.append(f"- **Evaluated Instances**: {metrics['sample_count']} human-adjudicated golden rows")
    lines.append(f"- **Overall Intent Accuracy**: {metrics['intent_accuracy']:.2%}")
    lines.append(f"- **Macro F1 Score**: {metrics['macro_f1']:.4f}\n")

    if metrics['has_action_eval']:
        act = metrics['action_metrics']
        lines.append("## 2. Dialogue Action & Safety Routing\n")
        lines.append(f"- **Action Accuracy**: {act['action_accuracy']:.2%}")
        lines.append(f"- **Auto-Handle Coverage**: {act['auto_handle_coverage']:.2%} ({act['auto_handle_total']}/{metrics['sample_count']} cases)")
        lines.append(f"- **Unsafe Auto-Handles (Safety Failures)**: {act['unsafe_auto_handle_count']} instances")
        lines.append(f"- **Selective Accuracy (Auto-Handled Cohort)**: {act['selective_accuracy']:.2%}\n")

    lines.append("## 3. Per-Intent Precision, Recall, and F1\n")
    lines.append("| Intent Category | Precision | Recall | F1 Score | Support |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for cls, row in metrics['per_intent'].items():
        lines.append(f"| `{cls}` | {row['precision']:.2%} | {row['recall']:.2%} | {row['f1']:.4f} | {row['support']} |")

    lines.append("\n## 4. Intent Confusion Matrix\n")
    lines.append(df_to_markdown(metrics['confusion_matrix']))
    lines.append("\n---\n*Verified against protected 200-row golden set.*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Predictions on Completed Adjudicated Golden Set.")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=False,
        default=None,
        help="Path to model predictions CSV file"
    )
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=REPO_ROOT / "data" / "golden" / "adjudication_sheet.csv",
        help="Path to adjudication sheet"
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Path to save markdown evaluation report"
    )
    parser.add_argument(
        "--dry-run-check",
        action="store_true",
        help="Perform safety & completeness verification checks without erroring"
    )

    args = parser.parse_args()

    print("=" * 78)
    print("      PHASE 5: PROTECTED GOLDEN SET EVALUATION RUNNER")
    print("=" * 78)

    # 1. Isolation verification check
    print("\n[Safety Check 1/3] Verifying golden set isolation...")
    if not verify_golden_isolation(REPO_ROOT):
        raise GoldenIntegrityViolationError(
            "CRITICAL INTEGRITY FAILURE: Golden set isolation check failed! "
            "Evaluation halted to prevent leakage."
        )
    print("-> PASS: Golden isolation verified (zero overlap with train/val).")

    # 2. Check human adjudication completeness
    print("\n[Safety Check 2/3] Checking human adjudication completeness...")
    is_complete, reason, golden_df = check_adjudication_completeness(args.adjudication)

    if not is_complete:
        print("\n" + "=" * 78)
        print("EVALUATION REFUSED: HUMAN GOLDEN ADJUDICATION IS INCOMPLETE")
        print("=" * 78)
        print(f"Reason: {reason}\n")
        print("Important Integrity Rules:")
        print("  - Do not evaluate models on unlabelled or partially annotated evaluation sets.")
        print("  - Do not write AI predictions into human label fields.")
        print("  - Do not report final accuracy, F1, or headline results until")
        print("    the adjudicated human golden labels are 100% complete.")
        print("=" * 78)
        
        if args.dry_run_check:
            print("\n[Dry Run]: Safety check caught incomplete status as expected. Clean exit.")
            sys.exit(0)
        else:
            sys.exit(1)

    print("-> PASS: All 200 golden rows have final_status == 'DONE' and valid taxonomy labels.")

    # 3. Validate predictions file
    print("\n[Safety Check 3/3] Validating predictions format and completeness...")
    if not args.predictions:
        print("ERROR: --predictions argument is required when adjudication is complete.")
        sys.exit(1)

    valid_preds, pred_reason, pred_df = validate_predictions_format(args.predictions, golden_df)
    if not valid_preds:
        print(f"ERROR: Invalid predictions file: {pred_reason}")
        sys.exit(1)
    print("-> PASS: Predictions contain all 200 golden_id records exactly once.")

    # 4. Merge predictions with golden truths
    eval_df = golden_df.merge(
        pred_df[['golden_id', 'predicted_intent'] + ([c for c in ['predicted_action'] if c in pred_df.columns])],
        on='golden_id',
        how='inner'
    )

    # 5. Compute metrics
    model_name = args.predictions.stem
    metrics = compute_metrics(eval_df)

    # 6. Render report
    report_text = format_markdown_report(metrics, model_name)
    print("\n" + "=" * 78)
    print(report_text)
    print("=" * 78)

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(report_text, encoding="utf-8")
        print(f"\nReport written to: {args.output_report}")


if __name__ == "__main__":
    main()
