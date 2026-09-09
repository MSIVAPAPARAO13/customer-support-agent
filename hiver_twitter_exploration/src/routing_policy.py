"""
Risk-Aware Routing Policy Engine (Phase 9)
===========================================

Purpose:
  Evaluates customer inquiry predictions, historical retrieval similarity,
  draft modes, and safety restrictions to make final offline routing decisions:
    - 'auto_handle' (eligible auto-handle candidate)
    - 'escalate' (requires human specialist review)
  Attaches inspectable reason codes and explanations.

Status:
  Offline decision-support prototype only.
  decision_status is always set to 'offline_candidate_not_sent'.
  No automated reply is ever sent to Twitter.

Design Principles:
  1. Fail-Safe Escalation: Any triggered risk, uncertainty, or weak evidence forces 'escalate'.
  2. Strict Priority Hierarchy: Reasons evaluated in deterministic order:
     restricted_safety_flag -> high_risk_intent -> other_or_unclear ->
     uncertain_intent -> insufficient_historical_evidence ->
     conservative_draft_mode -> eligible_low_risk_case.
  3. Robust Type Parsing: Safeguards against CSV string encodings ('False', 'True', 'none', '[]').
"""

import re
from typing import List, Dict, Any, Union, Optional


# Intent classification tiers
LOW_RISK_INTENTS = {
    "software_update_or_os_issue",
    "device_performance_or_hardware",
    "connectivity_and_network",
    "apps_services_or_icloud",
}

HIGH_RISK_INTENTS = {
    "account_access_and_apple_id",
    "billing_subscription_or_purchase",
    "repair_replacement_or_order",
}

# Reason code constants
REASON_RESTRICTED_SAFETY = "restricted_safety_flag"
REASON_HIGH_RISK_INTENT = "high_risk_intent"
REASON_OTHER_UNCLEAR = "other_or_unclear"
REASON_UNCERTAIN_INTENT = "uncertain_intent"
REASON_INSUFFICIENT_EVIDENCE = "insufficient_historical_evidence"
REASON_CONSERVATIVE_DRAFT_MODE = "conservative_draft_mode"
REASON_ELIGIBLE_LOW_RISK = "eligible_low_risk_case"

# Allowed reason codes
VALID_REASON_CODES = {
    REASON_RESTRICTED_SAFETY,
    REASON_HIGH_RISK_INTENT,
    REASON_OTHER_UNCLEAR,
    REASON_UNCERTAIN_INTENT,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_CONSERVATIVE_DRAFT_MODE,
    REASON_ELIGIBLE_LOW_RISK,
}

# Default decision status
DECISION_STATUS_OFFLINE = "offline_candidate_not_sent"


def parse_bool(value: Any) -> bool:
    """
    Safely parses boolean values from heterogeneous types (Python bool, int, string).
    Guards against bool("False") returning True.

    Conversions:
      True, "True", "true", 1, "1" -> True
      False, "False", "false", 0, "0", "", None, "none", "null" -> False
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value != 0)

    val_str = str(value).strip().lower()
    if val_str in ("true", "1", "yes", "t"):
        return True
    if val_str in ("false", "0", "no", "f", "", "none", "null", "nan"):
        return False

    return False


def parse_safety_flags(value: Any) -> List[str]:
    """
    Safely normalizes safety flags from string, list, or empty representation.

    Conversions:
      "", "none", "[]", None, "null", "nan" -> []
      "account_compromise|billing_fraud" -> ["account_compromise", "billing_fraud"]
      "account_compromise, billing_fraud" -> ["account_compromise", "billing_fraud"]
      "account_compromise; billing_fraud" -> ["account_compromise", "billing_fraud"]
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(f).strip() for f in value if str(f).strip() and str(f).strip().lower() not in ("none", "[]", "null")]

    val_str = str(value).strip()
    if not val_str or val_str.lower() in ("none", "[]", "null", "nan", "empty"):
        return []

    # Clean out surrounding brackets if present
    if val_str.startswith("[") and val_str.endswith("]"):
        val_str = val_str[1:-1].strip()

    if not val_str:
        return []

    # Split on commas, pipes, or semicolons
    raw_tokens = re.split(r"[,|;]\s*", val_str)
    normalized = []
    for tok in raw_tokens:
        clean_tok = tok.strip().strip("'\"").strip()
        if clean_tok and clean_tok.lower() not in ("none", "null", ""):
            normalized.append(clean_tok)

    return normalized


def evaluate_routing_decision(
    inquiry: Dict[str, Any],
    confidence_threshold: float = 0.80,
    similarity_threshold: float = 0.50,
) -> Dict[str, str]:
    """
    Evaluates a single inquiry row against the risk-aware routing policy.

    Inputs expected in inquiry dict:
      - predicted_intent: str
      - predicted_confidence: float or str
      - best_similarity_score: float or str
      - draft_mode: str
      - restricted_draft: bool, int, or str
      - safety_flags: list or str
      - needs_human_review: bool, int, or str

    Returns:
      {
        'action': 'auto_handle' | 'escalate',
        'decision_status': 'offline_candidate_not_sent',
        'primary_reason_code': str,
        'all_reason_codes': str,
        'decision_explanation': str
      }
    """
    # 1. Parse and normalize all input signals safely
    pred_intent = str(inquiry.get("predicted_intent", "")).strip()

    try:
        pred_conf = float(inquiry.get("predicted_confidence", 0.0))
    except (ValueError, TypeError):
        pred_conf = 0.0

    try:
        best_sim = float(inquiry.get("best_similarity_score", 0.0))
    except (ValueError, TypeError):
        best_sim = 0.0

    draft_mode = str(inquiry.get("draft_mode", "")).strip()
    restricted_draft = parse_bool(inquiry.get("restricted_draft"))
    needs_review = parse_bool(inquiry.get("needs_human_review"))
    safety_flags_list = parse_safety_flags(inquiry.get("safety_flags"))

    # 2. Check each escalation condition in strict priority order
    triggered_reasons = []

    # Condition 1: Restricted Safety Flag
    if restricted_draft or len(safety_flags_list) > 0:
        triggered_reasons.append(REASON_RESTRICTED_SAFETY)

    # Condition 2: High-Risk Intent
    if pred_intent in HIGH_RISK_INTENTS:
        triggered_reasons.append(REASON_HIGH_RISK_INTENT)

    # Condition 3: Other or Unclear Intent
    if pred_intent == "other_or_unclear" or not pred_intent:
        triggered_reasons.append(REASON_OTHER_UNCLEAR)

    # Condition 4: Uncertain Intent (Low Confidence or Upstream Review Flag)
    if pred_conf < confidence_threshold or needs_review:
        triggered_reasons.append(REASON_UNCERTAIN_INTENT)

    # Condition 5: Insufficient Historical Evidence
    if best_sim < similarity_threshold:
        triggered_reasons.append(REASON_INSUFFICIENT_EVIDENCE)

    # Condition 6: Conservative Draft Mode
    if draft_mode != "template_with_historical_pattern":
        triggered_reasons.append(REASON_CONSERVATIVE_DRAFT_MODE)

    # 3. Formulate Action & Explanation
    if not triggered_reasons:
        # All safety and confidence criteria satisfied: eligible auto-handle candidate
        action = "auto_handle"
        primary_reason = REASON_ELIGIBLE_LOW_RISK
        all_reasons = REASON_ELIGIBLE_LOW_RISK
        explanation = (
            f"Low-risk technical inquiry ({pred_intent}) with high intent confidence "
            f"({pred_conf:.4f} >= {confidence_threshold:.2f}), strong historical retrieval "
            f"({best_sim:.4f} >= {similarity_threshold:.2f}), safe draft mode, and zero safety flags. "
            f"Eligible for automated handling candidate."
        )
    else:
        # One or more escalation conditions triggered: escalate to human
        action = "escalate"
        primary_reason = triggered_reasons[0]
        all_reasons = " | ".join(triggered_reasons)

        if primary_reason == REASON_RESTRICTED_SAFETY:
            flags_str = ", ".join(safety_flags_list) if safety_flags_list else "restricted flag active"
            explanation = (
                f"Message contains safety-critical or account/billing risk language ({flags_str}). "
                f"Requires secure identity verification and human specialist handling; not eligible for auto-handling."
            )
        elif primary_reason == REASON_HIGH_RISK_INTENT:
            explanation = (
                f"Inquiry classified into high-risk domain '{pred_intent}'. "
                f"Account recovery, billing/subscription disputes, and repair/hardware orders require human specialist authorization."
            )
        elif primary_reason == REASON_OTHER_UNCLEAR:
            explanation = (
                "Customer intent is ambiguous or unclassified ('other_or_unclear'). "
                "Automated handling cannot safely diagnose or resolve unclassified inquiries."
            )
        elif primary_reason == REASON_UNCERTAIN_INTENT:
            explanation = (
                f"Intent prediction confidence ({pred_conf:.4f}) is below the required {confidence_threshold:.2f} threshold "
                f"or flagged for human review. Escalated to prevent misrouting."
            )
        elif primary_reason == REASON_INSUFFICIENT_EVIDENCE:
            explanation = (
                f"Historical dialogue retrieval similarity ({best_sim:.4f}) is below the {similarity_threshold:.2f} threshold. "
                f"Lacks sufficient historical grounding evidence for safe automated response."
            )
        elif primary_reason == REASON_CONSERVATIVE_DRAFT_MODE:
            explanation = (
                f"Draft mode is '{draft_mode}' rather than 'template_with_historical_pattern'. "
                f"Inquiries without safe historical pattern adaptation must be reviewed by a human."
            )
        else:
            explanation = f"Escalated due to policy constraint: {primary_reason}."

    return {
        "action": action,
        "decision_status": DECISION_STATUS_OFFLINE,
        "primary_reason_code": primary_reason,
        "all_reason_codes": all_reasons,
        "decision_explanation": explanation,
    }
