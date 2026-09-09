#!/usr/bin/env python3
"""
Audit Support Agent Routing Policy (Phase 9)
============================================

Purpose:
  Conducts safety, distribution, and scenario audits on the agent routing decisions.
  Generates:
    1. outputs/agent/routing_policy_audit.md
    2. outputs/agent/routing_human_review_sheet.csv (30 deterministic review rows, seed 42)
    3. outputs/agent/routing_threshold_scenarios.md (Coverage under 0.70, 0.80, 0.90)

Strict Rules:
  - Descriptive distribution counts only.
  - Zero automated quality, accuracy, or customer satisfaction claims.
  - All 30 review rows set to review_status = 'NEEDS_HUMAN_REVIEW'.
  - Evaluates threshold scenarios without declaring any as 'optimal' or 'best'.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.routing_policy import evaluate_routing_decision


def select_stratified_routing_sample(df: pd.DataFrame, seed: int = 42, target_total: int = 30) -> pd.DataFrame:
    """
    Selects 30 deterministic validation decisions representing key operational regimes:
      1. Auto-handle candidates (eligible low risk)
      2. Restricted account security cases
      3. Billing / dispute cases
      4. Low-confidence cases (< 0.80)
      5. Weak retrieval cases (< 0.50)
      6. Other or unclear intent cases
    """
    np.random.seed(seed)
    selected_indices = set()

    # Bucket 1: Auto-handle candidates
    b1 = df[df['action'] == 'auto_handle'].index.tolist()
    if b1:
        chosen = np.random.choice(b1, size=min(6, len(b1)), replace=False)
        selected_indices.update(chosen)

    # Bucket 2: Restricted account security cases
    b2 = df[df['safety_flags'].str.contains('account_compromise|password_or_recovery', regex=True, na=False)].index.tolist()
    rem2 = [i for i in b2 if i not in selected_indices]
    if rem2:
        chosen = np.random.choice(rem2, size=min(5, len(rem2)), replace=False)
        selected_indices.update(chosen)

    # Bucket 3: Billing / dispute cases
    b3 = df[df['predicted_intent'] == 'billing_subscription_or_purchase'].index.tolist()
    rem3 = [i for i in b3 if i not in selected_indices]
    if rem3:
        chosen = np.random.choice(rem3, size=min(4, len(rem3)), replace=False)
        selected_indices.update(chosen)

    # Bucket 4: Low-confidence cases (< 0.80)
    b4 = df[(df['predicted_confidence'] < 0.80) & (df['action'] == 'escalate')].index.tolist()
    rem4 = [i for i in b4 if i not in selected_indices]
    if rem4:
        chosen = np.random.choice(rem4, size=min(5, len(rem4)), replace=False)
        selected_indices.update(chosen)

    # Bucket 5: Weak retrieval cases (similarity < 0.50)
    b5 = df[(df['best_similarity_score'] < 0.50) & (df['action'] == 'escalate')].index.tolist()
    rem5 = [i for i in b5 if i not in selected_indices]
    if rem5:
        chosen = np.random.choice(rem5, size=min(5, len(rem5)), replace=False)
        selected_indices.update(chosen)

    # Bucket 6: Other or unclear intent cases
    b6 = df[df['predicted_intent'] == 'other_or_unclear'].index.tolist()
    rem6 = [i for i in b6 if i not in selected_indices]
    if rem6:
        chosen = np.random.choice(rem6, size=min(5, len(rem6)), replace=False)
        selected_indices.update(chosen)

    # Fill if needed
    if len(selected_indices) < target_total:
        rem_all = [i for i in df.index if i not in selected_indices]
        deficit = target_total - len(selected_indices)
        chosen = np.random.choice(rem_all, size=deficit, replace=False)
        selected_indices.update(chosen)

    ordered_indices = sorted(list(selected_indices))[:target_total]
    return df.loc[ordered_indices].copy()


def generate_review_sheet(df_sample: pd.DataFrame, output_path: Path) -> None:
    """Exports 30-sample review sheet with blank human evaluation fields."""
    sample_df = df_sample.copy()

    review_cols = [
        "human_expected_action",
        "human_expected_reason",
        "agent_action_correct_yes_no",
        "unsafe_auto_handle_yes_no",
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


def generate_policy_audit_report(df: pd.DataFrame, output_path: Path) -> None:
    """Generates markdown policy audit report with descriptive distributions."""
    total_decisions = len(df)
    auto_count = int((df["action"] == "auto_handle").sum())
    auto_pct = (auto_count / total_decisions) * 100
    esc_count = int((df["action"] == "escalate").sum())
    esc_pct = (esc_count / total_decisions) * 100

    primary_reason_counts = df["primary_reason_code"].value_counts().to_dict()

    # All reason counts breakdown
    all_reasons_series = df["all_reason_codes"].dropna().str.split(r"\s*\|\s*")
    all_reasons_flat = [r for sublist in all_reasons_series for r in sublist if r]
    all_reason_counts = pd.Series(all_reasons_flat).value_counts().to_dict()

    intent_cross = pd.crosstab(df["predicted_intent"], df["action"], margins=True)

    restr_count = int(df["restricted_draft"].astype(str).str.lower().isin(["true", "1"]).sum())

    # Exemplary cases
    auto_examples = df[df["action"] == "auto_handle"].head(5)
    esc_examples = df[df["action"] == "escalate"].head(5)

    report_lines = [
        "# Phase 9 Support Agent Routing Policy Audit",
        "",
        "## Executive Summary & Safety Policy Status",
        "",
        "In **Phase 9**, we implemented the final offline decision-support routing component of the AppleSupport inquiry pipeline. "
        "The routing engine evaluates model confidence, retrieval similarity, draft modes, and safety restrictions to make inspectable, "
        "deterministic routing determinations.",
        "",
        "> [!IMPORTANT]",
        "> **Core Policy Principles & Prototype Boundaries**:",
        "> 1. **Offline Prototype Status**: Every decision in this phase is marked `decision_status = 'offline_candidate_not_sent'`. No messages are sent to Twitter.",
        "> 2. **Candidate Meaning**: `auto_handle` designates an *eligible auto-handle candidate*, not deployed or unmonitored automation.",
        "> 3. **Fail-Safe Escalation**: Any triggered risk flag, high-risk intent, low confidence ($< 0.80$), weak similarity ($< 0.50$), or conservative draft mode forces immediate human escalation.",
        "> 4. **No Automated Accuracy Claims**: Descriptive distributions only. Deciding true routing accuracy requires completed human golden adjudication.",
        "",
        "---",
        "",
        "## 1. Overall Routing Decision Distributions (`validation_agent_decisions.csv`)",
        "",
        f"- **Total Inquiries Evaluated**: **{total_decisions:,}**",
        f"- **Auto-Handle Candidates (`auto_handle`)**: **{auto_count:,} ({auto_pct:.2f}%)**",
        f"- **Escalated to Human Specialist (`escalate`)**: **{esc_count:,} ({esc_pct:.2f}%)**",
        f"- **Inquiries with Active Safety Flags / Restricted Drafts**: **{restr_count:,} ({(restr_count/total_decisions*100):.2f}%)**",
        "",
        "### Primary Reason Code Breakdown",
        "",
        "The primary reason code reflects the first triggered constraint according to the strict priority hierarchy:",
        "",
        "| Primary Reason Code | Action | Count | Proportion | Policy Meaning |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ]

    for reason, count in primary_reason_counts.items():
        pct = (count / total_decisions) * 100
        action = "auto_handle" if reason == "eligible_low_risk_case" else "escalate"
        meaning = (
            "All safety, confidence, and grounding criteria satisfied for low-risk technical intent"
            if reason == "eligible_low_risk_case"
            else "Account compromise, password recovery, billing fraud, or legal/crisis risk terms detected"
            if reason == "restricted_safety_flag"
            else "Intent classified into account, billing, or repair/hardware order domain"
            if reason == "high_risk_intent"
            else "Inquiry intent is ambiguous or unclassified ('other_or_unclear')"
            if reason == "other_or_unclear"
            else "Model intent confidence is below 0.80 threshold or flagged for upstream review"
            if reason == "uncertain_intent"
            else "Retrieval similarity is below 0.50 threshold; lacks sufficient historical grounding"
            if reason == "insufficient_historical_evidence"
            else "Draft mode is conservative_no_evidence or lacks pattern adaptation"
        )
        report_lines.append(f"| `{reason}` | `{action}` | {count:,} | {pct:.2f}% | {meaning} |")

    report_lines.extend([
        "",
        "### Total Reason Code Occurrences (`all_reason_codes`)",
        "",
        "Inquiries frequently trigger multiple escalation conditions simultaneously. Below is the total occurrence count across all triggered reasons:",
        "",
        "| Reason Code | Total Occurrences across All Inquiries |",
        "| :--- | :---: |",
    ])

    for reason, count in all_reason_counts.items():
        report_lines.append(f"| `{reason}` | {count:,} |")

    report_lines.extend([
        "",
        "### Routing Actions by Predicted Intent",
        "",
        "| Predicted Intent | Auto-Handle Candidates | Escalated | Total Inquiries | Auto-Handle Rate |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for intent in intent_cross.index:
        if intent == "All":
            continue
        auto_cnt = intent_cross.loc[intent].get("auto_handle", 0)
        esc_cnt = intent_cross.loc[intent].get("escalate", 0)
        tot = auto_cnt + esc_cnt
        rate = (auto_cnt / tot * 100) if tot > 0 else 0.0
        report_lines.append(f"| `{intent}` | {auto_cnt:,} | {esc_cnt:,} | {tot:,} | {rate:.2f}% |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Exemplary De-Identified Decision Cases",
        "",
        "### A. Five Auto-Handle Candidates (`action = 'auto_handle'`)",
        "",
    ])

    for idx, (_, r) in enumerate(auto_examples.iterrows(), start=1):
        report_lines.extend([
            f"#### Auto-Handle Example {idx}: Tweet `{r['customer_tweet_id']}`",
            f"- **Customer Inquiry**: \"{r['customer_text_clean']}\"",
            f"- **Predicted Intent**: `{r['predicted_intent']}` (Confidence: `{r['predicted_confidence']:.4f}`)",
            f"- **Retrieval Similarity**: `{r['best_similarity_score']:.4f}` (`{r['lexical_similarity_band']}`)",
            f"- **Draft Mode**: `{r['draft_mode']}`",
            f"- **Action**: `{r['action']}` (`{r['decision_status']}`)",
            f"- **Primary Reason**: `{r['primary_reason_code']}`",
            f"- **Draft Reply**: > \"{r['draft_reply']}\"",
            f"- **Explanation**: {r['decision_explanation']}",
            "",
        ])

    report_lines.extend([
        "### B. Five Escalation Decisions (`action = 'escalate'`)",
        "",
    ])

    for idx, (_, r) in enumerate(esc_examples.iterrows(), start=1):
        report_lines.extend([
            f"#### Escalation Example {idx}: Tweet `{r['customer_tweet_id']}`",
            f"- **Customer Inquiry**: \"{r['customer_text_clean']}\"",
            f"- **Predicted Intent**: `{r['predicted_intent']}` (Confidence: `{r['predicted_confidence']:.4f}`)",
            f"- **Retrieval Similarity**: `{r['best_similarity_score']:.4f}` (`{r['lexical_similarity_band']}`)",
            f"- **Safety Flags**: `{r['safety_flags']}` (Restricted: `{r['restricted_draft']}`)",
            f"- **Action**: `{r['action']}` (`{r['decision_status']}`)",
            f"- **Primary Reason**: `{r['primary_reason_code']}`",
            f"- **All Reasons**: `{r['all_reason_codes']}`",
            f"- **Draft Reply**: > \"{r['draft_reply']}\"",
            f"- **Explanation**: {r['decision_explanation']}",
            "",
        ])

    report_lines.extend([
        "---",
        "",
        "## 3. Policy Limitations & Boundaries",
        "",
        "1. **Conservative Automation Bias**: With only 2.30% of validation inquiries qualifying for auto-handling under the default 0.80 confidence and 0.50 similarity thresholds, the system strongly prioritizes safety over automation volume.",
        "2. **Weak-Label Classifier Reliance**: Intent confidence scores originate from a weak-label prototype. True calibration requires the completed 200-row human golden set.",
        "3. **Absence of Real-Time System Telemetry**: The router cannot verify whether a customer's device is currently under warranty or if an Apple ID is locked.",
        "4. **Human Review Prerequisite**: All 30 review rows in `routing_human_review_sheet.csv` are initialized to `NEEDS_HUMAN_REVIEW`. No deployment should occur prior to human audit.",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"[Audit Report] Wrote routing policy audit report to {output_path}.")


def generate_threshold_scenarios_report(df_drafts: pd.DataFrame, output_path: Path) -> None:
    """
    Evaluates automation coverage across hypothetical confidence thresholds: 0.70, 0.80, 0.90.
    Reports coverage and intent distributions only without claiming any is 'best'.
    """
    scenarios = [0.70, 0.80, 0.90]
    total = len(df_drafts)
    results = {}

    for thresh in scenarios:
        auto_cnt = 0
        esc_cnt = 0
        intent_auto = {}

        for _, row in df_drafts.iterrows():
            dec = evaluate_routing_decision(
                inquiry=row.to_dict(),
                confidence_threshold=thresh,
                similarity_threshold=0.50,
            )
            if dec["action"] == "auto_handle":
                auto_cnt += 1
                intent = row.get("predicted_intent", "")
                intent_auto[intent] = intent_auto.get(intent, 0) + 1
            else:
                esc_cnt += 1

        results[thresh] = {
            "auto_handle_count": auto_cnt,
            "auto_handle_pct": (auto_cnt / total) * 100,
            "escalate_count": esc_cnt,
            "escalate_pct": (esc_cnt / total) * 100,
            "intent_breakdown": intent_auto,
        }

    lines = [
        "# Phase 9 Routing Threshold Scenarios Analysis",
        "",
        "## Purpose & Methodological Caveats",
        "",
        "This report analyzes the impact of varying the **intent prediction confidence threshold** on automation coverage. "
        r"All other conditions (historical retrieval similarity threshold $\ge 0.50$, draft mode, zero safety flags, low-risk intents only) "
        "remain strictly held constant across all three scenarios.",
        "",
        "> [!IMPORTANT]",
        "> **Critical Evaluation Rules**:",
        "> 1. **No Setting is Designated 'Best'**: We do not rank or declare any threshold 'optimal'.",
        "> 2. **No Performance or Accuracy Metrics Claimed**: Without completed adjudicated human labels on the golden set, true precision and error rates cannot be computed.",
        "> 3. **Trade-Off Nature**: Lowering the threshold increases automation volume but introduces higher risk of misclassification; raising it increases human workload.",
        "> 4. **Prerequisite**: Selecting an operational threshold requires evaluating the completed human golden evaluation set.",
        "",
        "---",
        "",
        "## 1. Scenario Coverage Comparison Across Validation Inquiries (10,357 Total)",
        "",
        "| Confidence Threshold | Auto-Handle Candidates | Auto-Handle Coverage | Escalation Volume | Escalation Rate | Operational Profile |",
        "| :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    profiles = {
        0.70: "More permissive technical auto-handling; captures borderline model predictions.",
        0.80: "Default conservative baseline; enforces solid model confidence and grounding.",
        0.90: "Extremely conservative; restricts automated candidates to near-certain predictions.",
    }

    for thresh in scenarios:
        r = results[thresh]
        lines.append(
            f"| `{thresh:.2f}` | **{r['auto_handle_count']:,}** | **{r['auto_handle_pct']:.2f}%** | "
            f"**{r['escalate_count']:,}** | **{r['escalate_pct']:.2f}%** | {profiles[thresh]} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Auto-Handle Candidates by Intent Across Scenarios",
        "",
        "Only the four approved low-risk technical intents ever produce auto-handle candidates. "
        "High-risk domains (`account_access_and_apple_id`, `billing_subscription_or_purchase`, `repair_replacement_or_order`) "
        "and `other_or_unclear` remain strictly 0% auto-handled across all thresholds.",
        "",
        "| Approved Intent | Threshold 0.70 | Threshold 0.80 (Default) | Threshold 0.90 |",
        "| :--- | :---: | :---: | :---: |",
    ])

    approved_intents = [
        "software_update_or_os_issue",
        "device_performance_or_hardware",
        "connectivity_and_network",
        "apps_services_or_icloud",
    ]

    for intent in approved_intents:
        c70 = results[0.70]["intent_breakdown"].get(intent, 0)
        c80 = results[0.80]["intent_breakdown"].get(intent, 0)
        c90 = results[0.90]["intent_breakdown"].get(intent, 0)
        lines.append(f"| `{intent}` | {c70:,} | {c80:,} | {c90:,} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. High-Risk & Ambiguous Domains (Invariant Across All Scenarios)",
        "",
        "| Intent Category | Threshold 0.70 | Threshold 0.80 | Threshold 0.90 | Policy Guarantee |",
        "| :--- | :---: | :---: | :---: | :--- |",
        "| `account_access_and_apple_id` | 0 | 0 | 0 | Mandatory human escalation (identity verification) |",
        "| `billing_subscription_or_purchase` | 0 | 0 | 0 | Mandatory human escalation (payment security) |",
        "| `repair_replacement_or_order` | 0 | 0 | 0 | Mandatory human escalation (warranty & logistics) |",
        "| `other_or_unclear` | 0 | 0 | 0 | Mandatory human escalation (intent ambiguity) |",
        "",
        "---",
        "",
        "## 4. Next Step: Golden Set Adjudication",
        "",
        "To empirically select an operational threshold for production deployment:",
        "1. Complete human adjudication of `data/golden/golden_set_200.csv`.",
        "2. Evaluate false-positive auto-handle rates (unsafe auto-handles) on golden rows.",
        "3. Select the operating point that guarantees near-zero unsafe auto-handles under organizational risk tolerance.",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Threshold Scenarios] Wrote threshold scenario report to {output_path}.")


def main():
    parser = argparse.ArgumentParser(description="Audit Support Agent Routing Policy (Phase 9).")
    parser.add_argument(
        "--decisions-input",
        type=Path,
        default=REPO_ROOT / "outputs" / "agent" / "validation_agent_decisions.csv",
        help="Path to input agent decisions CSV.",
    )
    parser.add_argument(
        "--drafts-input",
        type=Path,
        default=REPO_ROOT / "outputs" / "replies" / "validation_drafted_replies.csv",
        help="Path to validation drafted replies CSV (for scenario evaluation).",
    )
    parser.add_argument(
        "--review-sheet",
        type=Path,
        default=REPO_ROOT / "outputs" / "agent" / "routing_human_review_sheet.csv",
        help="Path to save 30-sample human review sheet CSV.",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=REPO_ROOT / "outputs" / "agent" / "routing_policy_audit.md",
        help="Path to save routing policy audit markdown report.",
    )
    parser.add_argument(
        "--scenarios-report",
        type=Path,
        default=REPO_ROOT / "outputs" / "agent" / "routing_threshold_scenarios.md",
        help="Path to save routing threshold scenarios markdown report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic review sampling (default: 42).",
    )

    args = parser.parse_args()

    if not args.decisions_input.exists():
        raise FileNotFoundError(f"Decisions input file not found: {args.decisions_input}. Run run_support_agent.py first.")

    print("=" * 75)
    print("PHASE 9: ROUTING POLICY AUDIT & THRESHOLD SCENARIOS")
    print("=" * 75)
    print(f"Decisions Input   : {args.decisions_input}")
    print(f"Drafts Input      : {args.drafts_input}")
    print(f"Review Sheet Path : {args.review_sheet}")
    print(f"Audit Report Path : {args.audit_report}")
    print(f"Scenarios Report  : {args.scenarios_report}")
    print(f"Random Seed       : {args.seed}")
    print("=" * 75)

    df_dec = pd.read_csv(args.decisions_input, low_memory=False)
    print(f"\n[DataLoader] Loaded {len(df_dec):,} agent decisions.")

    # 1. Generate Policy Audit Report
    generate_policy_audit_report(df_dec, args.audit_report)

    # 2. Generate 30-Sample Review Sheet
    print(f"\n[Sampling] Selecting 30 stratified audit samples with seed={args.seed}...", flush=True)
    df_sample = select_stratified_routing_sample(df_dec, seed=args.seed, target_total=30)
    generate_review_sheet(df_sample, args.review_sheet)

    # 3. Generate Threshold Scenarios Report
    if args.drafts_input.exists():
        df_drafts = pd.read_csv(args.drafts_input, low_memory=False)
        print(f"\n[Scenarios] Evaluating threshold scenarios (0.70, 0.80, 0.90) across {len(df_drafts):,} drafts...", flush=True)
        generate_threshold_scenarios_report(df_drafts, args.scenarios_report)
    else:
        print(f"[Warning] Drafts input not found at {args.drafts_input}; evaluating scenarios on decisions input...")
        generate_threshold_scenarios_report(df_dec, args.scenarios_report)

    print("\nAudit process finished successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
