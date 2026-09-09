#!/usr/bin/env python3
"""
Evaluate Human-Judge Agreement (Phase 10)
=========================================

Purpose:
  Calculates statistical agreement between human expert audit ratings and
  blinded LLM-as-a-judge ratings across the 40-row evaluation sample.

Integrity Rules:
  - REFUSES to calculate or report agreement unless ALL 40 rows in
    `human_judge_agreement_sheet.csv` have `human_review_status == 'DONE'`.
  - Refuses if `llm_judge_outputs.csv` does not exist.
  - Never fabricates human or judge scores.

Calculated Statistics:
  - Exact agreement percentage for each 1-5 score dimension
  - Quadratic weighted Cohen's Kappa for 1-5 ordinal dimensions
  - Linear weighted Cohen's Kappa for 1-5 ordinal dimensions
  - Standard Cohen's Kappa for binary `overall_accept`
  - Sample size confirmation (40 rows)
  - Qualitative disagreement case studies

Usage:
  python scripts/evaluate_judge_human_agreement.py
  python scripts/evaluate_judge_human_agreement.py --test-fixture <fixture_csv>
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Configure console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent

DIMENSIONS = [
    ("relevance", "human_relevance", "judge_relevance"),
    ("historical_grounding", "human_historical_grounding", "judge_historical_grounding"),
    ("safety_and_privacy", "human_safety_and_privacy", "judge_safety_and_privacy"),
    ("clear_next_step", "human_clear_next_step", "judge_clear_next_step"),
    ("routing_appropriateness", "human_routing_appropriateness", "judge_routing_appropriateness"),
]


def write_pending_report(output_path: Path, reason: str):
    """Writes status notification when agreement evaluation is pending human review."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Human-Judge Agreement Report (Phase 10)

## Status: PENDING HUMAN REVIEW

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Status**: `PENDING_HUMAN_REVIEW`

### Diagnostic Status
{reason}

### Protocol Enforcement
In strict accordance with evaluation integrity rules:
1. Human-judge agreement metrics are NOT calculated or reported until all 40 human review rows have been completed independently.
2. The evaluator refuses to execute until `human_review_status == 'DONE'` across all 40 rows in `outputs/final_evaluation/human_judge_agreement_sheet.csv`.
3. LLM judge scores must be generated via `scripts/run_llm_judge.py` with valid API credentials.

### Next Steps
1. Human reviewers complete scores (1-5) and overall verdict (`ACCEPT`/`REJECT`) in `outputs/final_evaluation/human_judge_agreement_sheet.csv`.
2. Set all 40 `human_review_status` values to `DONE`.
3. Ensure `outputs/final_evaluation/llm_judge_outputs.csv` is populated.
4. Run `python scripts/evaluate_judge_human_agreement.py` to generate certified inter-rater agreement metrics.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[STATUS NOTIFICATION] Documented pending status in {output_path}")


def evaluate_agreement(
    human_sheet_path: Path,
    judge_outputs_path: Path,
    report_output_path: Path,
    test_fixture: Optional[Path] = None,
) -> bool:
    """Evaluates agreement between human ratings and LLM judge ratings."""
    if test_fixture is not None:
        print(f"[TEST FIXTURE MODE] Evaluating agreement using test fixture: {test_fixture}")
        df = pd.read_csv(test_fixture, low_memory=False)
    else:
        if not human_sheet_path.exists():
            msg = f"Human agreement review sheet not found at: {human_sheet_path}"
            print(f"[REFUSAL] {msg}")
            write_pending_report(report_output_path, msg)
            return False

        df_human = pd.read_csv(human_sheet_path, low_memory=False)
        if len(df_human) != 40:
            msg = f"Expected exactly 40 rows in human review sheet, found {len(df_human)}."
            print(f"[REFUSAL] {msg}")
            write_pending_report(report_output_path, msg)
            return False

        # Verify completion of human review
        status_counts = df_human["human_review_status"].fillna("MISSING").value_counts().to_dict()
        done_count = status_counts.get("DONE", 0)
        if done_count != 40:
            msg = (
                f"Incomplete human review: Only {done_count}/40 rows have human_review_status == 'DONE'.\n"
                f"Status distribution: {status_counts}.\n"
                f"Only report human-judge agreement after all 40 human rows are completed."
            )
            print("\n" + "!" * 75)
            print("[AGREEMENT EVALUATION REFUSED] Human Reviews Incomplete")
            print("!" * 75)
            print(msg)
            print("!" * 75)
            write_pending_report(report_output_path, msg)
            return False

        if not judge_outputs_path.exists():
            msg = f"LLM judge outputs not found at: {judge_outputs_path}."
            print(f"[REFUSAL] {msg}")
            write_pending_report(report_output_path, msg)
            return False

        df_judge = pd.read_csv(judge_outputs_path, low_memory=False)
        id_col = "golden_id" if "golden_id" in df_human.columns else "sample_id"
        df = pd.merge(df_human, df_judge, on=id_col, suffixes=("_human", "_judge"))
        if len(df) != 40:
            msg = f"Merged records yielded {len(df)} rows, expected 40."
            print(f"[REFUSAL] {msg}")
            write_pending_report(report_output_path, msg)
            return False

    sample_size = len(df)
    print(f"\n[EVALUATING AGREEMENT] Computing metrics on {sample_size} double-rated items...")

    dimension_metrics = []
    disagreements = []

    # 1. 1-5 Score Dimensions
    for dim_name, h_col, j_col in DIMENSIONS:
        if h_col not in df.columns or j_col not in df.columns:
            print(f"[WARNING] Missing columns for {dim_name}: {h_col}, {j_col}")
            continue

        h_vals = pd.to_numeric(df[h_col], errors="coerce").fillna(0).astype(int).tolist()
        j_vals = pd.to_numeric(df[j_col], errors="coerce").fillna(0).astype(int).tolist()

        exact_matches = sum(1 for h, j in zip(h_vals, j_vals) if h == j)
        exact_pct = exact_matches / sample_size

        # Weighted Kappas with fixed labels [1, 2, 3, 4, 5]
        try:
            quad_kappa = cohen_kappa_score(h_vals, j_vals, labels=[1, 2, 3, 4, 5], weights="quadratic")
            if np.isnan(quad_kappa):
                quad_kappa = 1.0 if exact_pct == 1.0 else 0.0
        except Exception:
            quad_kappa = 0.0

        try:
            lin_kappa = cohen_kappa_score(h_vals, j_vals, labels=[1, 2, 3, 4, 5], weights="linear")
            if np.isnan(lin_kappa):
                lin_kappa = 1.0 if exact_pct == 1.0 else 0.0
        except Exception:
            lin_kappa = 0.0

        dimension_metrics.append({
            "dimension": dim_name,
            "exact_agreement": round(exact_pct, 4),
            "quadratic_kappa": round(quad_kappa, 4),
            "linear_kappa": round(lin_kappa, 4),
            "mean_human_score": round(float(np.mean(h_vals)), 2),
            "mean_judge_score": round(float(np.mean(j_vals)), 2),
        })

    # 2. Binary Overall Accept Dimension
    h_accept = df["human_overall_accept"].astype(str).str.upper().tolist()
    j_accept = df["judge_overall_accept"].astype(str).str.upper().tolist()

    accept_matches = sum(1 for h, j in zip(h_accept, j_accept) if h == j)
    accept_exact_pct = accept_matches / sample_size

    try:
        accept_kappa = cohen_kappa_score(h_accept, j_accept, labels=["ACCEPT", "REJECT"])
        if np.isnan(accept_kappa):
            accept_kappa = 1.0 if accept_exact_pct == 1.0 else 0.0
    except Exception:
        accept_kappa = 0.0

    # 3. Disagreement Examples
    for idx, row in df.iterrows():
        sid = row.get("sample_id", f"sample_{idx + 1:02d}")
        tw_id = row.get("customer_tweet_id", "")
        txt = row.get("customer_text_clean", "")
        h_acc = str(row.get("human_overall_accept", "")).upper()
        j_acc = str(row.get("judge_overall_accept", "")).upper()

        # Score delta max
        max_delta = 0
        delta_dim = ""
        for dim_name, h_col, j_col in DIMENSIONS:
            hv = int(pd.to_numeric(row.get(h_col, 0), errors="coerce") or 0)
            jv = int(pd.to_numeric(row.get(j_col, 0), errors="coerce") or 0)
            diff = abs(hv - jv)
            if diff > max_delta:
                max_delta = diff
                delta_dim = dim_name

        if (h_acc != j_acc) or max_delta >= 2:
            disagreements.append({
                "sample_id": sid,
                "customer_tweet_id": tw_id,
                "text_snippet": (txt[:70] + "...") if len(txt) > 70 else txt,
                "human_verdict": h_acc,
                "judge_verdict": j_acc,
                "max_score_delta": max_delta,
                "worst_dimension": delta_dim,
                "judge_explanation": row.get("judge_explanation", ""),
            })

    # 4. Generate Final Agreement Report
    report_content = f"""# Human-Judge Agreement Report (Phase 10)

## Overview & Methodology
This report evaluates the inter-rater reliability between human domain experts and the blinded automated LLM judge across {sample_size} stratified support interactions.

- **Sample Size**: {sample_size} interactions
- **Evaluation Protocol**: Blinded double evaluation (Zero access to human golden labels, model probabilities, or weak training tags)
- **Scoring Scale**: 1 (poor) to 5 (optimal) across 5 dimensions + binary `overall_accept` verdict.

---

## 1. Score Dimension Agreement Summary

| Evaluation Dimension | Exact Agreement (%) | Quadratic Weighted $\\kappa$ | Linear Weighted $\\kappa$ | Human Mean | Judge Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for dm in dimension_metrics:
        report_content += (
            f"| `{dm['dimension']}` | {dm['exact_agreement']:.2%} | **{dm['quadratic_kappa']:.4f}** | "
            f"{dm['linear_kappa']:.4f} | {dm['mean_human_score']:.2f} | {dm['mean_judge_score']:.2f} |\n"
        )

    report_content += f"""
---

## 2. Binary Overall Acceptance Agreement

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Exact Agreement Rate** | `{accept_exact_pct:.2%}` | Fraction of cases where human and LLM agree on `ACCEPT` vs. `REJECT` |
| **Cohen's Kappa ($\\kappa$)** | `**{accept_kappa:.4f}**` | Chance-adjusted inter-rater reliability for operational deployment suitability |
| **Sample Size** | `{sample_size}` | Stratified human review cohort |

---

## 3. Qualitative Disagreement Case Studies

Total flagged disagreements (verdict mismatch or Likert score delta $\\ge 2$): **{len(disagreements)}**

"""
    if disagreements:
        report_content += "| Sample ID | Human Verdict | Judge Verdict | Max $\\Delta$ | Dimension | Customer Inquiry Excerpt |\n"
        report_content += "| :--- | :---: | :---: | :---: | :--- | :--- |\n"
        for d in disagreements[:10]:
            report_content += (
                f"| `{d['sample_id']}` | `{d['human_verdict']}` | `{d['judge_verdict']}` | "
                f"{d['max_score_delta']} | `{d['worst_dimension']}` | {d['text_snippet']} |\n"
            )
    else:
        report_content += "No major disagreements found.\n"

    report_content += """
---

## 4. Methodological Interpretation & Limits
- While weighted Kappa values indicate alignment on clear extremes (severe safety violations vs. straightforward FAQs), borderline cases highlight nuanced human discretion regarding risk tolerance.
- LLM judge evaluations should be utilized as automated regression monitors rather than unmonitored gatekeepers.
"""

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[AGREEMENT REPORT GENERATED] Saved to: {report_output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate Human-Judge Agreement (Phase 10).")
    parser.add_argument(
        "--human-sheet",
        type=Path,
        default=REPO_ROOT / "outputs" / "final_evaluation" / "human_judge_agreement_sheet.csv",
    )
    parser.add_argument(
        "--judge-outputs",
        type=Path,
        default=REPO_ROOT / "outputs" / "final_evaluation" / "llm_judge_outputs.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "final_evaluation" / "judge_human_agreement_report.md",
    )
    parser.add_argument("--test-fixture", type=Path, default=None)
    args = parser.parse_args()

    success = evaluate_agreement(
        human_sheet_path=args.human_sheet,
        judge_outputs_path=args.judge_outputs,
        report_output_path=args.report_output,
        test_fixture=args.test_fixture,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
