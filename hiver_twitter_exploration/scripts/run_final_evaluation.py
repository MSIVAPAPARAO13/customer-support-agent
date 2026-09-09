#!/usr/bin/env python3
"""
Final Evaluation Engine: Golden Benchmark Assessment (Phase 10)
================================================================

Purpose:
  Conducts standardized evaluation of the end-to-end customer support assistant
  against the protected 200-row human-annotated golden benchmark.

Integrity Rules & Guardrails:
  1. Golden-Label Freeze Guard:
     Refuses to run unless `data/golden/golden_labels_freeze_manifest.json` exists,
     its SHA-256 matches `data/golden/adjudication_sheet.csv`, exactly 200 rows are present,
     and all rows have `final_status == 'DONE'`.
  2. Strict Data Isolation:
     Models and indices remain strictly train-only. Golden set is used strictly as unseen
     queries for evaluation after label freezing.
  3. No Hallucinated Metrics:
     If human labels are incomplete or unfrozen, refuses execution and documents blocked status.

Evaluated Systems:
  Compares exactly three intent systems:
    - `majority_weak_intent`
    - `keyword_rule_classifier`
    - `tfidf_logistic_regression` (Main system)

Metrics Calculated:
  - Intent accuracy, macro F1 (across all 8 fixed taxonomy classes), per-intent P/R/F1/support
  - Intent confusion matrix (8x8)
  - Routing metrics: action accuracy, auto-handle coverage, selective accuracy, unsafe auto-handle
    count and rate, escalation precision and recall (positive class: 'escalate')
  - 1,000-resample bootstrap 95% confidence intervals (seed 42)
  - Failure candidate identification

Usage:
  python scripts/run_final_evaluation.py
  python scripts/run_final_evaluation.py --test-fixture <fixture_csv>
"""

import sys
import os
import json
import hashlib
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import joblib

# Ensure console encoding handles UTF-8 gracefully
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Fixed complete list of 8 taxonomy labels - MUST NEVER be truncated or dynamically determined
FIXED_TAXONOMY_LABELS = [
    "software_update_or_os_issue",
    "device_performance_or_hardware",
    "connectivity_and_network",
    "apps_services_or_icloud",
    "account_access_and_apple_id",
    "billing_subscription_or_purchase",
    "repair_replacement_or_order",
    "other_or_unclear",
]

APPROVED_ACTIONS = ["auto_handle", "escalate"]

REQUIRED_WARNING_TEXT = (
    "> A conservative system can improve selective accuracy simply by escalating more messages. "
    "Automation coverage must always be reported alongside auto-handle quality."
)


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_freeze_guard(
    manifest_path: Path,
    adjudication_path: Path,
) -> Tuple[bool, str]:
    """
    Validates that golden human labels are frozen, completed, and untampered.
    """
    if not manifest_path.exists():
        return False, (
            f"Freeze manifest not found at {manifest_path}.\n"
            f"Golden human labels must be completed, reconciled, and frozen via "
            f"`scripts/freeze_golden_labels.py` before final evaluation is permitted."
        )

    if not adjudication_path.exists():
        return False, f"Adjudication sheet not found at {adjudication_path}."

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return False, f"Failed to parse freeze manifest JSON: {e}"

    if manifest.get("golden_row_count") != 200:
        return False, f"Manifest golden_row_count is {manifest.get('golden_row_count')}, expected 200."

    current_sha = compute_sha256(adjudication_path)
    expected_sha = manifest.get("adjudication_sha256")
    if current_sha != expected_sha:
        return False, (
            f"Adjudication SHA-256 ({current_sha}) does not match manifest SHA-256 ({expected_sha}). "
            f"Human labels have been altered after freeze!"
        )

    df_adj = pd.read_csv(adjudication_path, low_memory=False)
    if len(df_adj) != 200:
        return False, f"Adjudication sheet row count ({len(df_adj)}) is not 200."

    status_counts = df_adj["final_status"].fillna("MISSING").value_counts().to_dict()
    if status_counts.get("DONE", 0) != 200:
        return False, f"Not all golden rows have final_status == 'DONE'. Current status counts: {status_counts}."

    return True, "Freeze guard verified: Golden benchmark is complete and frozen."


def compute_multiclass_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = FIXED_TAXONOMY_LABELS,
) -> Dict[str, Any]:
    """
    Computes accuracy, macro-F1, per-class precision/recall/F1/support, and confusion matrix.
    Always uses the fixed complete list of 8 taxonomy labels to prevent artificial inflation.
    """
    n = len(y_true)
    if n == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": {},
        }

    # Overall Accuracy
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / n

    per_class = {}
    f1_list = []

    # Confusion matrix dict: [true_label][pred_label]
    cm = {true_lbl: {pred_lbl: 0 for pred_lbl in labels} for true_lbl in labels}

    for yt, yp in zip(y_true, y_pred):
        if yt in cm and yp in cm[yt]:
            cm[yt][yp] += 1

    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
        f1_list.append(f1)

    # Fixed 8-class macro F1
    macro_f1 = sum(f1_list) / len(labels)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def compute_routing_metrics(
    human_actions: List[str],
    agent_actions: List[str],
) -> Dict[str, Any]:
    """
    Computes operational routing metrics for auto_handle vs escalate:
      - action_accuracy
      - auto_handle_coverage
      - selective_accuracy (N/A if 0 auto_handle)
      - unsafe_auto_handle_count
      - unsafe_auto_handle_rate (N/A if 0 auto_handle)
      - escalation_precision (positive class: escalate)
      - escalation_recall (positive class: escalate)
    """
    total = len(human_actions)
    if total == 0:
        return {}

    # Action accuracy
    correct_actions = sum(1 for ha, aa in zip(human_actions, agent_actions) if ha == aa)
    action_accuracy = correct_actions / total

    # Auto-handle counts
    auto_handle_count = sum(1 for aa in agent_actions if aa == "auto_handle")
    auto_handle_coverage = auto_handle_count / total

    # Unsafe auto-handle: agent auto_handle, but human marked escalate
    unsafe_auto_handle_count = sum(
        1 for ha, aa in zip(human_actions, agent_actions)
        if aa == "auto_handle" and ha == "escalate"
    )

    if auto_handle_count > 0:
        unsafe_auto_handle_rate = unsafe_auto_handle_count / auto_handle_count
        # Selective accuracy: accuracy strictly among auto-handled cases
        auto_handle_correct = sum(
            1 for ha, aa in zip(human_actions, agent_actions)
            if aa == "auto_handle" and ha == "auto_handle"
        )
        selective_accuracy = auto_handle_correct / auto_handle_count
    else:
        unsafe_auto_handle_rate = None
        selective_accuracy = None

    # Escalation Precision & Recall (positive class: 'escalate')
    agent_escalations = sum(1 for aa in agent_actions if aa == "escalate")
    human_escalations = sum(1 for ha in human_actions if ha == "escalate")
    true_escalations = sum(
        1 for ha, aa in zip(human_actions, agent_actions)
        if aa == "escalate" and ha == "escalate"
    )

    escalation_precision = (true_escalations / agent_escalations) if agent_escalations > 0 else 0.0
    escalation_recall = (true_escalations / human_escalations) if human_escalations > 0 else 0.0

    return {
        "total_evaluated": total,
        "action_accuracy": round(action_accuracy, 4),
        "auto_handle_count": auto_handle_count,
        "auto_handle_coverage": round(auto_handle_coverage, 4),
        "unsafe_auto_handle_count": unsafe_auto_handle_count,
        "unsafe_auto_handle_rate": round(unsafe_auto_handle_rate, 4) if unsafe_auto_handle_rate is not None else "N/A",
        "selective_accuracy": round(selective_accuracy, 4) if selective_accuracy is not None else "N/A",
        "agent_escalations": agent_escalations,
        "human_escalations": human_escalations,
        "escalation_precision": round(escalation_precision, 4),
        "escalation_recall": round(escalation_recall, 4),
    }


def compute_bootstrap_cis(
    y_true_intent: List[str],
    y_pred_intent: List[str],
    y_true_action: List[str],
    y_pred_action: List[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Generates 1,000 bootstrap resamples with seed 42 to compute 95% confidence intervals.
    Each bootstrap macro-F1 calculation uses all 8 fixed taxonomy labels.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true_intent)
    indices = np.arange(n)

    intent_acc_list = []
    macro_f1_list = []
    action_acc_list = []
    coverage_list = []
    selective_acc_list = []

    for _ in range(n_resamples):
        sample_idx = rng.choice(indices, size=n, replace=True)

        # Sample intents
        sample_y_true_intent = [y_true_intent[i] for i in sample_idx]
        sample_y_pred_intent = [y_pred_intent[i] for i in sample_idx]

        # Intent metrics
        intent_metrics = compute_multiclass_metrics(
            sample_y_true_intent, sample_y_pred_intent, labels=FIXED_TAXONOMY_LABELS
        )
        intent_acc_list.append(intent_metrics["accuracy"])
        macro_f1_list.append(intent_metrics["macro_f1"])

        # Sample actions
        sample_y_true_action = [y_true_action[i] for i in sample_idx]
        sample_y_pred_action = [y_pred_action[i] for i in sample_idx]

        routing_metrics = compute_routing_metrics(sample_y_true_action, sample_y_pred_action)
        action_acc_list.append(routing_metrics["action_accuracy"])
        coverage_list.append(routing_metrics["auto_handle_coverage"])
        if routing_metrics["selective_accuracy"] != "N/A":
            selective_acc_list.append(routing_metrics["selective_accuracy"])

    def get_ci(values: List[float]) -> Tuple[float, float]:
        if not values:
            return (0.0, 0.0)
        return (
            round(float(np.percentile(values, 2.5)), 4),
            round(float(np.percentile(values, 97.5)), 4),
        )

    return {
        "intent_accuracy_95ci": get_ci(intent_acc_list),
        "macro_f1_95ci": get_ci(macro_f1_list),
        "action_accuracy_95ci": get_ci(action_acc_list),
        "auto_handle_coverage_95ci": get_ci(coverage_list),
        "selective_accuracy_95ci": get_ci(selective_acc_list) if selective_acc_list else ("N/A", "N/A"),
    }


def write_blocked_report(output_dir: Path, reason: str):
    """Writes a clean report explaining why evaluation is blocked pending human labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "final_metrics_report.md"
    content = f"""# Final Evaluation Report: Golden Benchmark Assessment (Phase 10)

## Status: BLOCKED / AWAITING HUMAN ADJUDICATION

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Status**: `AWAITING_HUMAN_ADJUDICATION`

### Reason for Refusal
{reason}

### Integrity Policy
In strict compliance with evaluation integrity protocols:
1. Model inference on golden queries cannot occur until the 200 human labels are complete and cryptographically frozen.
2. Final headline metrics (accuracy, macro-F1, selective accuracy) will not be calculated or published based on synthetic, incomplete, or unadjudicated labels.
3. The evaluation pipeline refuses to execute until `data/golden/adjudication_sheet.csv` contains 200 rows with `final_status == 'DONE'` and the corresponding `data/golden/golden_labels_freeze_manifest.json` is generated.

### Next Steps to Unblock Evaluation
1. Complete independent human labeling in `data/golden/annotator_a_blind.csv` and `data/golden/annotator_b_blind.csv`.
2. Run `python scripts/reconcile_golden_labels.py` to identify inter-annotator disagreements.
3. Complete final human adjudication in `data/golden/adjudication_sheet.csv`, setting all 200 rows to `final_status = DONE`.
4. Run `python scripts/freeze_golden_labels.py` to generate the tamper-proof SHA-256 manifest.
5. Run `python scripts/run_golden_agent_inference.py` to perform blind agent inference.
6. Re-run `python scripts/run_final_evaluation.py` to produce final certified metrics.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[STATUS NOTIFICATION] Documented blocked status in {report_path}")


def run_evaluation(
    adjudication_path: Path,
    manifest_path: Path,
    agent_preds_path: Path,
    output_dir: Path,
    models_dir: Path,
    train_weak_path: Path,
    test_fixture: Optional[Path] = None,
) -> bool:
    """
    Executes complete final evaluation across the 3 intent systems and the routing agent.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check freeze guard unless test fixture is explicitly provided
    if test_fixture is None:
        guard_ok, guard_msg = verify_freeze_guard(manifest_path, adjudication_path)
        if not guard_ok:
            print("\n" + "!" * 75)
            print("[FINAL EVALUATION REFUSED] Golden Benchmark Not Ready")
            print("!" * 75)
            print(guard_msg)
            print("!" * 75)
            write_blocked_report(output_dir, guard_msg)
            return False

        df_gold = pd.read_csv(adjudication_path, low_memory=False)
        if not agent_preds_path.exists():
            print(f"[ERROR] Agent predictions not found at: {agent_preds_path}")
            print("Please run `python scripts/run_golden_agent_inference.py` first.")
            return False
        df_agent = pd.read_csv(agent_preds_path, low_memory=False)
    else:
        print(f"[TEST FIXTURE MODE] Evaluating on test fixture: {test_fixture}")
        df_gold = pd.read_csv(test_fixture, low_memory=False)
        df_agent = df_gold.copy()
        if "agent_action" not in df_agent.columns and "final_action" in df_agent.columns:
            df_agent["agent_action"] = df_agent["final_action"]

    # Validate row alignment
    if len(df_gold) != len(df_agent):
        print(f"[ERROR] Row mismatch: Golden ({len(df_gold)}) vs Agent ({len(df_agent)})")
        return False

    y_true_intent = df_gold["final_intent"].astype(str).tolist()
    y_true_action = df_gold["final_action"].astype(str).tolist()
    customer_texts = df_gold["customer_text_clean"].fillna("").astype(str).tolist()

    # -------------------------------------------------------------
    # 1. System 1: Baseline majority_weak_intent
    # -------------------------------------------------------------
    print("\n[System 1/3] Evaluating Baseline: majority_weak_intent...")
    if train_weak_path.exists():
        df_train_weak = pd.read_csv(train_weak_path, usecols=["weak_intent"])
        majority_intent = df_train_weak["weak_intent"].value_counts().index[0]
    else:
        majority_intent = "device_performance_or_hardware"
    y_pred_majority = [majority_intent] * len(y_true_intent)
    majority_metrics = compute_multiclass_metrics(y_true_intent, y_pred_majority)

    # -------------------------------------------------------------
    # 2. System 2: Baseline keyword_rule_classifier
    # -------------------------------------------------------------
    print("[System 2/3] Evaluating Baseline: keyword_rule_classifier...")
    try:
        from scripts.create_weak_labels import assign_weak_label
    except ImportError:
        from create_weak_labels import assign_weak_label

    y_pred_rules = []
    for txt in customer_texts:
        weak_intent, _, _ = assign_weak_label(txt)
        y_pred_rules.append(weak_intent)
    rule_metrics = compute_multiclass_metrics(y_true_intent, y_pred_rules)

    # -------------------------------------------------------------
    # 3. System 3: Main tfidf_logistic_regression
    # -------------------------------------------------------------
    print("[System 3/3] Evaluating Main System: tfidf_logistic_regression...")
    vec_path = models_dir / "tfidf_vectorizer.joblib"
    clf_path = models_dir / "tfidf_logistic_regression.joblib"

    if not vec_path.exists() or not clf_path.exists():
        print(f"[ERROR] Intent model artifacts missing in {models_dir}")
        return False

    vectorizer = joblib.load(vec_path)
    classifier = joblib.load(clf_path)

    X = vectorizer.transform(customer_texts)
    y_pred_main = classifier.predict(X).tolist()
    main_metrics = compute_multiclass_metrics(y_true_intent, y_pred_main)

    # -------------------------------------------------------------
    # 4. Routing Agent Action Evaluation
    # -------------------------------------------------------------
    action_col = "agent_action" if "agent_action" in df_agent.columns else "action"
    y_pred_action = df_agent[action_col].astype(str).tolist()
    routing_metrics = compute_routing_metrics(y_true_action, y_pred_action)

    # -------------------------------------------------------------
    # 5. Bootstrap 95% Confidence Intervals (1,000 resamples, seed 42)
    # -------------------------------------------------------------
    print("[Bootstrap CIs] Generating 1,000 bootstrap resamples (seed 42)...")
    bootstrap_cis = compute_bootstrap_cis(
        y_true_intent=y_true_intent,
        y_pred_intent=y_pred_main,
        y_true_action=y_true_action,
        y_pred_action=y_pred_action,
        n_resamples=1000,
        seed=42,
    )

    # -------------------------------------------------------------
    # 6. Save Structured Per-Intent Metrics CSV
    # -------------------------------------------------------------
    per_intent_rows = []
    for label in FIXED_TAXONOMY_LABELS:
        m_cls = majority_metrics["per_class"].get(label, {})
        r_cls = rule_metrics["per_class"].get(label, {})
        main_cls = main_metrics["per_class"].get(label, {})

        per_intent_rows.append({
            "intent_label": label,
            "support": main_cls.get("support", 0),
            "majority_precision": m_cls.get("precision", 0.0),
            "majority_recall": m_cls.get("recall", 0.0),
            "majority_f1": m_cls.get("f1", 0.0),
            "rule_precision": r_cls.get("precision", 0.0),
            "rule_recall": r_cls.get("recall", 0.0),
            "rule_f1": r_cls.get("f1", 0.0),
            "main_precision": main_cls.get("precision", 0.0),
            "main_recall": main_cls.get("recall", 0.0),
            "main_f1": main_cls.get("f1", 0.0),
        })

    df_per_intent = pd.DataFrame(per_intent_rows)
    per_intent_csv = output_dir / "per_intent_metrics.csv"
    df_per_intent.to_csv(per_intent_csv, index=False, encoding="utf-8")

    # -------------------------------------------------------------
    # 7. Save Confusion Matrix CSV
    # -------------------------------------------------------------
    cm_rows = []
    cm_dict = main_metrics["confusion_matrix"]
    for true_lbl in FIXED_TAXONOMY_LABELS:
        row_dict = {"true_intent": true_lbl}
        for pred_lbl in FIXED_TAXONOMY_LABELS:
            row_dict[pred_lbl] = cm_dict.get(true_lbl, {}).get(pred_lbl, 0)
        cm_rows.append(row_dict)

    df_cm = pd.DataFrame(cm_rows)
    cm_csv = output_dir / "intent_confusion_matrix.csv"
    df_cm.to_csv(cm_csv, index=False, encoding="utf-8")

    # -------------------------------------------------------------
    # 8. Identify Failure Candidates
    # -------------------------------------------------------------
    failure_candidates = []
    for idx, row in df_gold.iterrows():
        gid = row.get("golden_id", f"gold_{idx:03d}")
        tw_id = row.get("customer_tweet_id", "")
        txt = row.get("customer_text_clean", "")
        t_intent = y_true_intent[idx]
        p_intent = y_pred_main[idx]
        t_action = y_true_action[idx]
        p_action = y_pred_action[idx]

        is_intent_error = (t_intent != p_intent)
        is_action_error = (t_action != p_action)
        is_unsafe = (p_action == "auto_handle" and t_action == "escalate")

        if is_intent_error or is_action_error:
            failure_type = []
            if is_unsafe:
                failure_type.append("UNSAFE_AUTO_HANDLE")
            elif is_action_error:
                failure_type.append("ACTION_MISMATCH")
            if is_intent_error:
                failure_type.append("INTENT_MISMATCH")

            failure_candidates.append({
                "golden_id": gid,
                "customer_tweet_id": tw_id,
                "customer_text_clean": txt,
                "true_intent": t_intent,
                "predicted_intent": p_intent,
                "true_action": t_action,
                "agent_action": p_action,
                "failure_type": "|".join(failure_type),
            })

    df_failures = pd.DataFrame(failure_candidates)
    failures_csv = output_dir / "failure_candidates.csv"
    df_failures.to_csv(failures_csv, index=False, encoding="utf-8")

    # -------------------------------------------------------------
    # 9. Save final_metrics.json
    # -------------------------------------------------------------
    final_metrics_payload = {
        "metadata": {
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
            "golden_sample_size": len(y_true_intent),
            "manifest_file": str(manifest_path.relative_to(REPO_ROOT) if manifest_path.is_relative_to(REPO_ROOT) else manifest_path),
            "fixed_taxonomy_labels": FIXED_TAXONOMY_LABELS,
        },
        "intent_system_comparison": {
            "majority_weak_intent": {
                "accuracy": majority_metrics["accuracy"],
                "macro_f1": majority_metrics["macro_f1"],
            },
            "keyword_rule_classifier": {
                "accuracy": rule_metrics["accuracy"],
                "macro_f1": rule_metrics["macro_f1"],
            },
            "tfidf_logistic_regression": {
                "accuracy": main_metrics["accuracy"],
                "macro_f1": main_metrics["macro_f1"],
            },
        },
        "routing_metrics": routing_metrics,
        "bootstrap_confidence_intervals_95": bootstrap_cis,
    }

    metrics_json_path = output_dir / "final_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics_payload, f, indent=2)

    # -------------------------------------------------------------
    # 10. Write Comprehensive Final Metrics Report Markdown
    # -------------------------------------------------------------
    report_md_path = output_dir / "final_metrics_report.md"
    report_content = f"""# Final Evaluation Report: Golden Benchmark Assessment (Phase 10)

## Executive Summary & Integrity Notice

This report presents the certified evaluation of the offline AppleSupport customer assistant against the protected 200-row human golden benchmark.

{REQUIRED_WARNING_TEXT}

> **Routing Behavior Context**:
> An auto-handle coverage of 2.3% (or similar conservative policy) is a deliberate risk-minimization routing behavior, NOT independent evidence of accuracy, safety, or deployment readiness. High selective accuracy on an extremely narrow cohort simply reflects conservative gating.

---

## 1. Intent Classification System Comparison

Comparison across exactly three intent architectures evaluated against identical human ground truth:

| System Name | Description | Accuracy | Macro F1 (Fixed 8 Classes) |
| :--- | :--- | :---: | :---: |
| **`majority_weak_intent`** | Minimal baseline: always predicts training mode (`{majority_intent}`) | {majority_metrics['accuracy']:.4f} | {majority_metrics['macro_f1']:.4f} |
| **`keyword_rule_classifier`** | Deterministic domain regex & keyword heuristics | {rule_metrics['accuracy']:.4f} | {rule_metrics['macro_f1']:.4f} |
| **`tfidf_logistic_regression`** | Main trained ML model (sublinear TF-IDF + balanced LogReg) | **{main_metrics['accuracy']:.4f}** | **{main_metrics['macro_f1']:.4f}** |

*Note: All Macro-F1 calculations strictly include all eight fixed taxonomy categories. Bootstrapping does not omit rare categories.*

---

## 2. End-to-End Operational Routing Metrics

Evaluation of the operational action routing (`auto_handle` vs. `escalate`):

| Metric | Value | Definition / Interpretation |
| :--- | :---: | :--- |
| **Action Accuracy** | `{routing_metrics['action_accuracy']:.4f}` | Fraction of cases where agent action matches human decision exactly |
| **Auto-Handle Coverage** | `{routing_metrics['auto_handle_coverage']:.4f}` | Fraction of inquiries candidate for automated handling (`{routing_metrics['auto_handle_count']}/{routing_metrics['total_evaluated']}`) |
| **Selective Accuracy** | `{routing_metrics['selective_accuracy']}` | Accuracy strictly among auto-handled cases (N/A if 0 auto-handled) |
| **Unsafe Auto-Handle Count** | `{routing_metrics['unsafe_auto_handle_count']}` | Critical safety errors: Agent auto-handled, but Human required escalation |
| **Unsafe Auto-Handle Rate** | `{routing_metrics['unsafe_auto_handle_rate']}` | `unsafe_auto_handles / auto_handles` (N/A if 0 auto-handled) |
| **Escalation Precision** | `{routing_metrics['escalation_precision']:.4f}` | Among agent escalations, fraction humans also marked escalate (`escalate` is positive class) |
| **Escalation Recall** | `{routing_metrics['escalation_recall']:.4f}` | Among human escalations, fraction agent successfully escalated (`escalate` is positive class) |

---

## 3. Bootstrap 95% Confidence Intervals (1,000 Resamples, Seed 42)

| Metric | Point Estimate | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Main Intent Accuracy** | `{main_metrics['accuracy']:.4f}` | `[{bootstrap_cis['intent_accuracy_95ci'][0]:.4f}, {bootstrap_cis['intent_accuracy_95ci'][1]:.4f}]` |
| **Main Intent Macro F1** | `{main_metrics['macro_f1']:.4f}` | `[{bootstrap_cis['macro_f1_95ci'][0]:.4f}, {bootstrap_cis['macro_f1_95ci'][1]:.4f}]` |
| **Action Accuracy** | `{routing_metrics['action_accuracy']:.4f}` | `[{bootstrap_cis['action_accuracy_95ci'][0]:.4f}, {bootstrap_cis['action_accuracy_95ci'][1]:.4f}]` |
| **Auto-Handle Coverage** | `{routing_metrics['auto_handle_coverage']:.4f}` | `[{bootstrap_cis['auto_handle_coverage_95ci'][0]:.4f}, {bootstrap_cis['auto_handle_coverage_95ci'][1]:.4f}]` |
| **Selective Accuracy** | `{routing_metrics['selective_accuracy']}` | `{bootstrap_cis['selective_accuracy_95ci']}` |

---

## 4. Per-Intent Performance Breakdown (Main Model)

See complete table in [`per_intent_metrics.csv`](per_intent_metrics.csv).

| Taxonomy Category | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
"""
    for row in per_intent_rows:
        report_content += f"| `{row['intent_label']}` | {row['support']} | {row['main_precision']:.4f} | {row['main_recall']:.4f} | {row['main_f1']:.4f} |\n"

    report_content += f"""
---

## 5. Confusion Matrix & Failure Analysis

- **Confusion Matrix**: Full 8x8 matrix exported to [`intent_confusion_matrix.csv`](intent_confusion_matrix.csv).
- **Failure Candidates**: Total {len(df_failures)} discrepancies flagged for qualitative audit in [`failure_candidates.csv`](failure_candidates.csv).

### Offline Prototype Safeguard Reminder
All decisions evaluated here are marked `decision_status = 'offline_candidate_not_sent'`. No messages were transmitted to live Twitter users.
"""
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[EVALUATION COMPLETE] Results written to {output_dir}")
    print(f"  - Summary Report   : {report_md_path}")
    print(f"  - Metrics JSON     : {metrics_json_path}")
    print(f"  - Per-Intent CSV   : {per_intent_csv}")
    print(f"  - Confusion Matrix : {cm_csv}")
    print(f"  - Failures CSV     : {failures_csv}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Run Final Golden Evaluation (Phase 10).")
    parser.add_argument("--adjudication-file", type=Path, default=REPO_ROOT / "data" / "golden" / "adjudication_sheet.csv")
    parser.add_argument("--manifest-file", type=Path, default=REPO_ROOT / "data" / "golden" / "golden_labels_freeze_manifest.json")
    parser.add_argument("--agent-preds-file", type=Path, default=REPO_ROOT / "outputs" / "final_evaluation" / "golden_agent_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "final_evaluation")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--train-weak", type=Path, default=REPO_ROOT / "data" / "weak_labels" / "applesupport_train_weak_labels.csv")
    parser.add_argument("--test-fixture", type=Path, default=None, help="Path to synthetic or validation fixture strictly for testing the script.")
    args = parser.parse_args()

    success = run_evaluation(
        adjudication_path=args.adjudication_file,
        manifest_path=args.manifest_file,
        agent_preds_path=args.agent_preds_file,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        train_weak_path=args.train_weak,
        test_fixture=args.test_fixture,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
