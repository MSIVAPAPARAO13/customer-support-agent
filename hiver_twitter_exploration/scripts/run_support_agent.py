#!/usr/bin/env python3
"""
Run Support Agent Routing Decision Engine (Phase 9)
===================================================

Purpose:
  Evaluates all drafted validation replies against the risk-aware routing policy,
  assigning final offline actions ('auto_handle' or 'escalate') and detailed
  auditable reason codes.

Strict Partition Rules:
  - Sourced exclusively from outputs/replies/validation_drafted_replies.csv.
  - Never loads test, golden, or annotator blind files.
  - Every decision is marked decision_status = 'offline_candidate_not_sent'.

Usage:
  python scripts/run_support_agent.py `
    --input outputs/replies/validation_drafted_replies.csv `
    --output outputs/agent/validation_agent_decisions.csv `
    --confidence-threshold 0.80 `
    --similarity-threshold 0.50
"""

import sys
import os
import argparse
from pathlib import Path

import pandas as pd

# Configure console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.routing_policy import evaluate_routing_decision, DECISION_STATUS_OFFLINE


def check_input_isolation(input_path: Path) -> None:
    """Strict data isolation check to guard against accidental test/golden access."""
    name = input_path.name.lower()
    if "test" in name or "golden" in name or "blind" in name:
        raise ValueError(
            f"DATA ISOLATION ERROR: Prohibited input file '{input_path.name}'. "
            f"Routing in Phase 9 operates strictly on validation drafted replies."
        )


def main():
    parser = argparse.ArgumentParser(description="Run Support Agent Routing Decision Engine (Phase 9).")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "outputs" / "replies" / "validation_drafted_replies.csv",
        help="Path to input validation drafted replies CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "agent" / "validation_agent_decisions.csv",
        help="Path to save output agent decisions CSV.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.80,
        help="Minimum predicted confidence threshold for auto-handle eligibility (default: 0.80).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.50,
        help="Minimum retrieval similarity threshold for auto-handle eligibility (default: 0.50).",
    )

    args = parser.parse_args()

    check_input_isolation(args.input)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}. Run Phase 8 drafting first.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 9: RISK-AWARE AUTO-HANDLE VS. ESCALATION ROUTING ENGINE")
    print("=" * 75)
    print(f"Input File            : {args.input}")
    print(f"Output Decisions File : {args.output}")
    print(f"Confidence Threshold  : {args.confidence_threshold:.2f}")
    print(f"Similarity Threshold  : {args.similarity_threshold:.2f}")
    print("=" * 75)

    df_in = pd.read_csv(args.input, low_memory=False)
    total_inquiries = len(df_in)
    print(f"\n[DataLoader] Loaded {total_inquiries:,} drafted replies.")

    print("[Routing Engine] Evaluating policy rules across all inquiries...", flush=True)
    decision_rows = []

    for idx, row in df_in.iterrows():
        dec = evaluate_routing_decision(
            inquiry=row.to_dict(),
            confidence_threshold=args.confidence_threshold,
            similarity_threshold=args.similarity_threshold,
        )

        decision_rows.append({
            # Required primary columns
            "customer_tweet_id": row.get("customer_tweet_id"),
            "customer_text_clean": row.get("customer_text_clean"),
            "predicted_intent": row.get("predicted_intent"),
            "predicted_confidence": row.get("predicted_confidence"),
            "best_similarity_score": row.get("best_similarity_score"),
            "lexical_similarity_band": row.get("lexical_similarity_band"),
            "draft_reply": row.get("draft_reply"),
            "draft_mode": row.get("draft_mode"),
            "restricted_draft": row.get("restricted_draft"),
            "safety_flags": row.get("safety_flags"),
            "action": dec["action"],
            "decision_status": dec["decision_status"],
            "primary_reason_code": dec["primary_reason_code"],
            "all_reason_codes": dec["all_reason_codes"],
            "decision_explanation": dec["decision_explanation"],
            # Preserved evidence & provenance columns
            "needs_human_review": row.get("needs_human_review"),
            "has_sufficient_historical_evidence": row.get("has_sufficient_historical_evidence"),
            "evidence_customer_tweet_ids": row.get("evidence_customer_tweet_ids"),
            "evidence_brand_tweet_ids": row.get("evidence_brand_tweet_ids"),
            "grounding_note": row.get("grounding_note"),
        })

    df_out = pd.DataFrame(decision_rows)

    # Save to CSV
    df_out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\n[Saving] Successfully wrote {len(df_out):,} agent decisions to {args.output}.")

    # Summary Statistics
    auto_count = int((df_out["action"] == "auto_handle").sum())
    auto_pct = (auto_count / total_inquiries) * 100
    esc_count = int((df_out["action"] == "escalate").sum())
    esc_pct = (esc_count / total_inquiries) * 100

    print("\n" + "=" * 75)
    print("PHASE 9 ROUTING DECISION SUMMARY")
    print("=" * 75)
    print(f"Total Decisions Evaluated : {total_inquiries:,}")
    print(f"  auto_handle Candidates  : {auto_count:>6,} ({auto_pct:5.2f}%)")
    print(f"  escalate to Specialist  : {esc_count:>6,} ({esc_pct:5.2f}%)")

    print("\nPrimary Reason Breakdown:")
    for reason, count in df_out["primary_reason_code"].value_counts().items():
        pct = (count / total_inquiries) * 100
        print(f"  {reason:<35} : {count:>6,} ({pct:5.2f}%)")

    print("\nAction by Predicted Intent:")
    ct = pd.crosstab(df_out["predicted_intent"], df_out["action"], margins=True)
    print(ct.to_string())

    print("\nDecision execution completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
