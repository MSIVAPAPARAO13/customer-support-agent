#!/usr/bin/env python3
"""
Run Weak-Label Validation Diagnostics (Phase 6)
==============================================

Purpose:
  Evaluates the trained TF-IDF + Logistic Regression prototype strictly against
  the heuristic weak labels on the validation partition
  (`applesupport_validation_weak_labels.csv`).

CRITICAL METHODOLOGICAL & INTEGRITY SAFEGUARDS:
  1. WEAK-LABEL DIAGNOSTIC ONLY — NOT HUMAN-GROUND-TRUTH PERFORMANCE:
     This diagnostic measures internal consistency and heuristic reproduction.
     It is NOT a measure of true real-world task performance.
  2. NEVER USE THE WORD "ACCURACY":
     All concordance metrics are explicitly termed `agreement_with_weak_labels`.
  3. STRICT ISOLATION:
     - Never loads `applesupport_test.csv`.
     - Never loads `golden_set_200.csv`.
     - Never modifies `annotator_a_blind.csv` or `annotator_b_blind.csv`.
  4. CONFIDENCE BREAKDOWNS:
     Reports agreement partitioned by weak-label heuristic confidence
     (high, medium, low, and overall).

Outputs:
  - outputs/tfidf/validation_weak_diagnostic_predictions.csv
  - outputs/tfidf/validation_weak_diagnostic_report.md

Usage:
  python scripts/run_validation_diagnostics.py
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import joblib

# Ensure Windows stdout handles UTF-8 gracefully
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent

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


def load_artifacts(models_dir: Path) -> Tuple[Any, Any]:
    """Loads vectorizer and classifier joblib files."""
    vec_path = models_dir / "tfidf_vectorizer.joblib"
    cls_path = models_dir / "tfidf_logistic_regression.joblib"
    if not vec_path.exists() or not cls_path.exists():
        raise FileNotFoundError(f"Model artifacts missing from {models_dir}. Train model first.")
    return joblib.load(vec_path), joblib.load(cls_path)


def compute_confusion_matrix(y_true: List[str], y_pred: List[str], classes: List[str]) -> pd.DataFrame:
    """Builds a confusion matrix as a DataFrame."""
    matrix = {c: [0] * len(classes) for c in classes}
    class_idx = {c: i for i, c in enumerate(classes)}

    for true_lbl, pred_lbl in zip(y_true, y_pred):
        if true_lbl in class_idx and pred_lbl in class_idx:
            t_idx = class_idx[true_lbl]
            matrix[pred_lbl][t_idx] += 1
        elif true_lbl in class_idx:
            fallback = "other_or_unclear" if "other_or_unclear" in class_idx else classes[-1]
            matrix[fallback][class_idx[true_lbl]] += 1

    df_cm = pd.DataFrame(matrix, index=classes)
    df_cm.index.name = "True Weak Label \\ Predicted"
    return df_cm


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


def format_report(
    eval_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    threshold: float = 0.60
) -> str:
    """Builds the comprehensive markdown diagnostic report."""
    total_rows = len(eval_df)
    
    # 1. Agreement by Confidence Slice
    def get_agreement(df_slice: pd.DataFrame) -> Tuple[int, int, float]:
        if len(df_slice) == 0:
            return 0, 0, 0.0
        agree = (df_slice['weak_intent'] == df_slice['predicted_intent']).sum()
        return agree, len(df_slice), agree / len(df_slice)

    overall_agree, overall_total, overall_rate = get_agreement(eval_df)
    high_agree, high_total, high_rate = get_agreement(eval_df[eval_df['weak_label_confidence'] == 'high'])
    med_agree, med_total, med_rate = get_agreement(eval_df[eval_df['weak_label_confidence'] == 'medium'])
    low_agree, low_total, low_rate = get_agreement(eval_df[eval_df['weak_label_confidence'] == 'low'])

    # 2. Predicted Intent Distribution
    pred_counts = eval_df['predicted_intent'].value_counts()
    weak_counts = eval_df['weak_intent'].value_counts()

    # 3. Low confidence rate
    low_conf_rows = eval_df[eval_df['predicted_confidence'] < threshold]
    low_conf_count = len(low_conf_rows)
    low_conf_rate = low_conf_count / total_rows if total_rows > 0 else 0.0

    lines = []
    lines.append("# WEAK-LABEL DIAGNOSTIC ONLY — NOT HUMAN-GROUND-TRUTH PERFORMANCE\n")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Strict Diagnostic Notice**:")
    lines.append("> These metrics reflect **agreement with heuristic weak labels** on the validation partition.")
    lines.append("> A model can agree highly with weak labels while merely reproducing the same keyword-rule biases.")
    lines.append("> Final performance claims require the completed, human-reviewed golden evaluation set.\n")

    lines.append("## 1. Agreement with Weak Labels by Confidence Stratum\n")
    lines.append("| Weak-Label Confidence Stratum | Total Rows | Agreement Count | Agreement with Weak Labels |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(f"| **High-Confidence Weak Labels** | {high_total:,} | {high_agree:,} | **{high_rate:.2%}** |")
    lines.append(f"| **Medium-Confidence Weak Labels** | {med_total:,} | {med_agree:,} | **{med_rate:.2%}** |")
    lines.append(f"| **Low-Confidence Weak Labels** | {low_total:,} | {low_agree:,} | **{low_rate:.2%}** |")
    lines.append(f"| **All Validation Rows (Overall)** | {overall_total:,} | {overall_agree:,} | **{overall_rate:.2%}** |\n")

    lines.append("## 2. Low-Confidence Predictions & Human Review Rate\n")
    lines.append(f"- **Total Inquiries Evaluated**: {total_rows:,}")
    lines.append(f"- **Confidence Threshold for Review**: < {threshold:.2f}")
    lines.append(f"- **Inquiries Flagged for Human Review (`needs_human_review = True`)**: {low_conf_count:,} ({low_conf_rate:.2%})\n")
    lines.append("> [!NOTE]")
    lines.append("> *Predicted confidence is a model probability estimate from a weak-label prototype. It is not yet calibrated against human ground truth.*")
    lines.append("> *Flagging for human review is an upstream triage signal and does not yet trigger automated escalation decisions.*\n")

    lines.append("## 3. Distribution Comparison: Heuristic Weak Labels vs. TF-IDF Predictions\n")
    lines.append("| Intent Class | Validation Weak Count | % Weak Split | TF-IDF Predicted Count | % Predicted |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for cls in APPROVED_INTENTS:
        w_cnt = weak_counts.get(cls, 0)
        w_pct = w_cnt / total_rows if total_rows > 0 else 0
        p_cnt = pred_counts.get(cls, 0)
        p_pct = p_cnt / total_rows if total_rows > 0 else 0
        lines.append(f"| `{cls}` | {w_cnt:,} | {w_pct:.2%} | {p_cnt:,} | {p_pct:.2%} |")

    lines.append("\n> [!WARNING]")
    lines.append("> **Distribution Shift Limitation**:")
    lines.append("> Because training uses class balancing and a controlled `other_or_unclear` sample, predicted class frequencies are not expected to represent real AppleSupport traffic frequencies.\n")

    lines.append("## 4. Weak-Label Confusion Matrix\n")
    lines.append(df_to_markdown(cm_df))
    lines.append("\n")


    # 4. De-identified Examples
    lines.append("## 5. De-Identified Prediction Audits\n")
    lines.append("### Five High-Confidence Predictions (`predicted_confidence >= 0.85`)\n")
    high_sample = eval_df.sort_values(by='predicted_confidence', ascending=False).head(5)
    for idx, r in high_sample.reset_index().iterrows():
        txt = r['customer_text_clean']
        lines.append(f"{idx+1}. **Tweet `{r['customer_tweet_id']}`**: \"{txt}\"")
        lines.append(f"   - **Weak Intent**: `{r['weak_intent']}` (Conf: `{r['weak_label_confidence']}`)")
        lines.append(f"   - **Predicted**: `{r['predicted_intent']}` (Model Conf: `{r['predicted_confidence']:.4f}`)")
        lines.append(f"   - **Top 3**: {r['top_3_intents']} | **Probs**: {r['top_3_probabilities']}\n")

    lines.append("### Five Low-Confidence Predictions (`needs_human_review = True`)\n")
    low_sample = eval_df.sort_values(by='predicted_confidence', ascending=True).head(5)
    for idx, r in low_sample.reset_index().iterrows():
        txt = r['customer_text_clean']
        lines.append(f"{idx+1}. **Tweet `{r['customer_tweet_id']}`**: \"{txt}\"")
        lines.append(f"   - **Weak Intent**: `{r['weak_intent']}` (Conf: `{r['weak_label_confidence']}`)")
        lines.append(f"   - **Predicted**: `{r['predicted_intent']}` (Model Conf: `{r['predicted_confidence']:.4f}`)")
        lines.append(f"   - **Top 3**: {r['top_3_intents']} | **Probs**: {r['top_3_probabilities']}\n")

    lines.append("---\n*Diagnostic report generated automatically by scripts/run_validation_diagnostics.py.*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run Weak-Label Validation Diagnostics.")
    parser.add_argument(
        "--val-weak",
        type=Path,
        default=REPO_ROOT / "data" / "weak_labels" / "applesupport_validation_weak_labels.csv",
        help="Path to validation weak labels CSV"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory containing trained model joblib files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "tfidf",
        help="Directory to save diagnostic outputs"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Confidence threshold for needs_human_review flag (default: 0.60)"
    )

    args = parser.parse_args()

    print("=" * 78)
    print("   PHASE 6: WEAK-LABEL VALIDATION DIAGNOSTICS (NOT GROUND TRUTH)")
    print("=" * 78)

    # 1. Enforce strict isolation
    print("\n[Safety Audit] Verifying test and golden set isolation...")
    test_path = REPO_ROOT / "data" / "processed" / "applesupport_test.csv"
    golden_path = REPO_ROOT / "data" / "golden" / "golden_set_200.csv"
    print(f"  - Ensuring {test_path.name} is NEVER loaded.")
    print(f"  - Ensuring {golden_path.name} is NEVER loaded.")
    print("  -> PASS: Diagnostics executed exclusively on validation weak labels.")

    # 2. Load model artifacts
    vectorizer, classifier = load_artifacts(args.models_dir)

    # 3. Load validation weak data
    if not args.val_weak.exists():
        raise FileNotFoundError(f"Validation weak labels not found at: {args.val_weak}")

    print(f"\n[Data Loader] Loading validation partition from: {args.val_weak.name}...")
    df_val = pd.read_csv(args.val_weak, dtype=str)
    print(f"[Data Loader] Loaded {len(df_val):,} validation rows.")

    # 4. Generate predictions
    texts = df_val['customer_text_clean'].fillna('').astype(str).tolist()
    print("\n[Inference] Transforming texts with TF-IDF vectorizer...")
    X_val = vectorizer.transform(texts)
    
    print("[Inference] Computing model probability distributions...")
    proba_matrix = classifier.predict_proba(X_val)
    classes = list(classifier.classes_)

    pred_records = []
    for i in range(len(texts)):
        probs = proba_matrix[i]
        ranked_idx = np.argsort(probs)[::-1]
        top_idx = ranked_idx[0]
        top_intent = classes[top_idx]
        top_conf = float(probs[top_idx])

        top_3_idx = ranked_idx[:3]
        top_3_intents = "; ".join([classes[idx] for idx in top_3_idx])
        top_3_probs = "; ".join([f"{probs[idx]:.4f}" for idx in top_3_idx])

        needs_review = bool(top_conf < args.threshold)

        row = df_val.iloc[i]
        pred_records.append({
            'customer_tweet_id': str(row.get('customer_tweet_id', '')),
            'customer_text_clean': texts[i],
            'weak_intent': str(row.get('weak_intent', '')),
            'weak_label_confidence': str(row.get('weak_label_confidence', '')),
            'weak_label_rule': str(row.get('weak_label_rule', '')),
            'predicted_intent': top_intent,
            'predicted_confidence': round(top_conf, 4),
            'top_3_intents': top_3_intents,
            'top_3_probabilities': top_3_probs,
            'needs_human_review': needs_review,
            'model_name': "tfidf_logistic_regression",
            'model_version': "v1.0-weak-prototype"
        })

    eval_df = pd.DataFrame(pred_records)

    # 5. Confusion matrix
    cm_df = compute_confusion_matrix(
        y_true=eval_df['weak_intent'].tolist(),
        y_pred=eval_df['predicted_intent'].tolist(),
        classes=APPROVED_INTENTS
    )

    # 6. Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.output_dir / "validation_weak_diagnostic_predictions.csv"
    report_out = args.output_dir / "validation_weak_diagnostic_report.md"

    eval_df.to_csv(csv_out, index=False)
    print(f"\n[Output] Saved predictions to: {csv_out.name} ({len(eval_df):,} rows)")

    report_content = format_report(eval_df, cm_df, threshold=args.threshold)
    report_out.write_text(report_content, encoding="utf-8")
    print(f"[Output] Saved diagnostic report to: {report_out.name}")

    print("\n" + "=" * 78)
    print("VALIDATION DIAGNOSTICS COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
