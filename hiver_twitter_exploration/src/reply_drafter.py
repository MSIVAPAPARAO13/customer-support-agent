"""
Safe Grounded Reply Drafter (Phase 8)
=====================================

Purpose:
  Combines predicted intent, confidence score, retrieved historical AppleSupport
  response patterns, and deterministic safety restrictions to generate safe,
  auditable draft customer-support replies.

Key Principles:
  1. Constrained Templates: Constrained templates reduce unsupported claims and make
     drafts easier to audit. They do not guarantee that every draft is relevant or safe.
  2. Grounding in Support Protocols, Not Facts: Historical replies influence general support
     response patterns (e.g., preparing device model and OS version). They do not assert
     customer-specific facts, promises, or diagnoses.
  3. Stricter Adaptation Threshold (0.50):
     - restricted_draft == True -> draft_mode = "restricted_safety"
     - restricted_draft == False and best_similarity_score < 0.50 -> draft_mode = "conservative_no_evidence"
     - restricted_draft == False and best_similarity_score >= 0.50 -> draft_mode = "template_with_historical_pattern"
     (A score between 0.30 and 0.49 is weak lexical evidence; it is recorded in metadata but does not alter drafts).
  4. No Hard-Coded URLs: Uses generic official channel guidance ("Apple’s official secure support").
  5. Zero Inventions: No promises of refunds, repairs, replacements, or technical resolutions.
  6. Zero Private Data Requests: Never requests passwords, PINs, card numbers, or government IDs.
  7. Sanitized Evidence: Replaces @mentions with [USER] and URLs with [URL].
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from src.reply_safety import detect_safety_flags, is_restricted, get_restricted_draft


# Base intent templates (Safe, cautious next steps without invented facts or URLs)
INTENT_BASE_TEMPLATES: Dict[str, str] = {
    "software_update_or_os_issue": (
        "We understand you are experiencing an issue related to software or updating. "
        "We recommend reviewing Apple’s official support resources for software updates. "
        "For troubleshooting tailored to your setup, please reach out through an official secure channel."
    ),
    "device_performance_or_hardware": (
        "We understand you are having concerns with your device’s performance or hardware. "
        "As an initial step, a basic device restart can often help refresh system processes. "
        "For a complete diagnostic check, please connect with Apple Support through an official secure channel."
    ),
    "connectivity_and_network": (
        "We understand you are encountering a connection issue. "
        "As a basic step, checking your local network settings or toggling Airplane Mode may help. "
        "If the connection difficulty continues, please contact Apple Support through an official secure channel."
    ),
    "apps_services_or_icloud": (
        "We understand you are experiencing trouble with an app or Apple service. "
        "We suggest checking Apple’s official System Status page and ensuring your apps are up to date. "
        "For direct account or service assistance, please reach out through an official secure support channel."
    ),
    "account_access_and_apple_id": (
        "For your security with Apple ID or account access, please use Apple’s official account-recovery "
        "or secure support options. Never share passwords, verification codes, or personal account details publicly."
    ),
    "billing_subscription_or_purchase": (
        "For questions regarding billing, purchases, or subscriptions, please review your purchase history "
        "through Apple’s official secure support options. Never post payment details or account credentials publicly."
    ),
    "repair_replacement_or_order": (
        "We understand you are inquiring about service, repair, or an order. "
        "Please visit Apple’s official support and service portal to explore available service options and appointment scheduling. "
        "Support agents cannot determine warranty eligibility or promise replacements over public social channels."
    ),
    "other_or_unclear": (
        "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, "
        "or contact Apple Support through an official secure channel. Please do not share private or account details publicly."
    ),
}


def sanitize_text(text: str) -> str:
    """Sanitizes text: replaces @mentions with [USER] and standardizes URLs to [URL]."""
    if not text:
        return ""
    t = re.sub(r'https?://\S+|www\.\S+', '[URL]', str(text))
    t = re.sub(r'@\w+', '[USER]', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def extract_historical_pattern(retrieved_replies: List[str]) -> Tuple[str, str, str]:
    """
    Analyzes sanitized historical brand replies to detect the dominant support pattern.
    Returns:
      (pattern_id, general_advice_phrase, grounding_note)
    """
    if not retrieved_replies:
        return (
            "none",
            "Please contact Apple Support through an official secure channel for assistance.",
            "No historical replies available to extract support patterns.",
        )

    combined_replies = " ".join(retrieved_replies).lower()

    # Pattern 1: Requesting device model and software/iOS version
    if re.search(r"\b(?:ios|version|model|device|what device|which version|update to)\b", combined_replies):
        return (
            "device_and_os",
            "To help troubleshoot, have your device model and current software version ready when contacting official Apple Support through a secure channel.",
            "Historical replies for similar training messages commonly requested device model and software version. This draft incorporates that general support protocol without asserting a customer-specific diagnosis.",
        )

    # Pattern 2: Suggesting restart / force restart
    if re.search(r"\b(?:restart|restarting|power off|turn off|reboot)\b", combined_replies):
        return (
            "restart_device",
            "As a general initial step, restarting your device can help resolve temporary issues. For continued assistance, contact Apple Support through an official secure channel.",
            "Historical replies for similar training messages commonly suggested a device restart. This draft references that general step without asserting a specific fix.",
        )

    # Pattern 3: Suggesting checking connection / network settings
    if re.search(r"\b(?:wi-?fi|network|cellular|bluetooth|airplane mode|settings)\b", combined_replies):
        return (
            "settings_or_network",
            "You can review your device settings or toggle Airplane Mode as a basic initial check. If the issue persists, reach out through an official secure channel.",
            "Historical replies for similar training messages suggested checking device settings or network connections. This draft references that general step safely.",
        )

    # Default pattern: Directing to secure support / direct messaging
    return (
        "secure_channel_handoff",
        "Please connect with Apple Support through an official secure channel so an advisor can review the specifics with you.",
        "Historical replies for similar training messages directed customers to secure support channels for investigation.",
    )


def draft_reply_for_inquiry(
    customer_tweet_id: str,
    customer_text_clean: str,
    predicted_intent: str,
    predicted_confidence: float,
    needs_human_review: bool,
    best_similarity_score: float,
    lexical_similarity_band: str,
    has_sufficient_historical_evidence: bool,
    retrieval_candidates: List[Dict[str, Any]],
    similarity_adaptation_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Generates a safe, grounded draft reply and records all evidence and safety flags.
    """
    raw_text = str(customer_text_clean or "")
    sanitized_query_text = sanitize_text(raw_text)

    # 1. Detect safety flags
    safety_flags = detect_safety_flags(raw_text)
    restricted = is_restricted(safety_flags)
    safety_flags_str = "; ".join(safety_flags) if safety_flags else "none"

    # Extract retrieved candidate IDs and sanitized replies
    evidence_cust_ids = []
    evidence_brand_ids = []
    retrieved_replies_clean = []

    for cand in retrieval_candidates:
        cid = str(cand.get("retrieved_customer_tweet_id", ""))
        bid = str(cand.get("retrieved_brand_tweet_id", ""))
        reply_txt = str(cand.get("retrieved_brand_reply_clean", ""))

        if cid and cid not in ("NONE", "nan", ""):
            evidence_cust_ids.append(cid)
        if bid and bid not in ("NONE", "nan", ""):
            evidence_brand_ids.append(bid)
        if reply_txt and reply_txt not in ("NONE", "nan", ""):
            retrieved_replies_clean.append(sanitize_text(reply_txt))

    evidence_cust_str = "; ".join(evidence_cust_ids) if evidence_cust_ids else "NONE"
    evidence_brand_str = "; ".join(evidence_brand_ids) if evidence_brand_ids else "NONE"

    # 2. Assign draft mode and construct draft reply based on strict criteria:
    #    - restricted_draft == True -> draft_mode = "restricted_safety"
    #    - restricted_draft == False and best_similarity_score < 0.50 -> draft_mode = "conservative_no_evidence"
    #    - restricted_draft == False and best_similarity_score >= 0.50 -> draft_mode = "template_with_historical_pattern"

    # Round similarity score to 4 decimal places for consistent threshold comparison
    rounded_sim = round(float(best_similarity_score), 4)

    if restricted:
        draft_mode = "restricted_safety"
        draft_reply, grounding_note = get_restricted_draft(safety_flags)

    elif rounded_sim < similarity_adaptation_threshold:
        draft_mode = "conservative_no_evidence"
        # Use cautious intent template or general handoff without historical pattern adaptation
        base_template = INTENT_BASE_TEMPLATES.get(
            predicted_intent, INTENT_BASE_TEMPLATES["other_or_unclear"]
        )
        draft_reply = base_template
        grounding_note = (
            f"Best lexical similarity ({rounded_sim:.4f}) is below the {similarity_adaptation_threshold:.2f} "
            f"adaptation threshold. Draft uses conservative base template without historical troubleshooting guidance."
        )

    else:
        draft_mode = "template_with_historical_pattern"
        # Identify historical response pattern from top-k replies
        pattern_id, pattern_advice, pattern_note = extract_historical_pattern(retrieved_replies_clean)

        # Base intent acknowledgement
        base_template = INTENT_BASE_TEMPLATES.get(
            predicted_intent, INTENT_BASE_TEMPLATES["other_or_unclear"]
        )

        # Synthesize base template with safe historical protocol pattern
        draft_reply = f"{base_template} {pattern_advice}"
        grounding_note = pattern_note

    return {
        "customer_tweet_id": str(customer_tweet_id),
        "customer_text_clean": sanitized_query_text,
        "predicted_intent": predicted_intent,
        "predicted_confidence": round(float(predicted_confidence), 4),
        "needs_human_review": bool(needs_human_review),
        "draft_reply": draft_reply,
        "draft_mode": draft_mode,
        "restricted_draft": bool(restricted),
        "safety_flags": safety_flags_str,
        "evidence_customer_tweet_ids": evidence_cust_str,
        "evidence_brand_tweet_ids": evidence_brand_str,
        "best_similarity_score": round(float(best_similarity_score), 4),
        "lexical_similarity_band": lexical_similarity_band,
        "has_sufficient_historical_evidence": bool(has_sufficient_historical_evidence),
        "grounding_note": grounding_note,
    }
