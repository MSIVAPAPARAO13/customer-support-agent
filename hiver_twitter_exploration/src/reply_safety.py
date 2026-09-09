"""
Safety & Risk Restriction Module (Phase 8)
===========================================

Purpose:
  Provides deterministic rule-based safety detection for incoming customer inquiries.
  Identifies high-risk categories such as account compromise, password recovery,
  unauthorized charges, payment disputes, legal escalation, and crisis/self-harm.

Design Principles:
  1. Risk-Aware Restrictions: Inquiries with safety flags are forced to restricted_draft = True
     and draft_mode = "restricted_safety".
  2. Brief, Neutral Handoffs: Generates brief, safe handoffs directing users to official, secure channels.
  3. No Unverified Advice: Never provides medical, crisis counseling, or legal advice.
  4. Zero Credential Requests: Never asks for passwords, PINs, full payment details, or government IDs.
  5. Zero Fact Claims: Never promises refunds, account restorations, or replacements.
  6. No Hard-Coded URLs: Recommends official secure channels without unverified links.
"""

import re
from typing import List, Tuple, Dict, Any


# Regex patterns for safety categories
SAFETY_PATTERNS: Dict[str, re.Pattern] = {
    "self_harm_or_danger": re.compile(
        r"\b(?:kill myself|suicide|suicidal|end my life|want to die|hurt myself|self[ -]?harm)\b",
        re.IGNORECASE,
    ),
    "threats_abuse_legal": re.compile(
        r"\b(?:lawsuit|sue you|sue apple|lawyer|attorney|police|court|legal action|litigation|better business bureau|bbb complaint|contacting my lawyer)\b",
        re.IGNORECASE,
    ),
    "account_compromise": re.compile(
        r"\b(?:hacked|hackers?|unauthorized access|someone logged into|account was compromised|stolen account|account breached|identity theft)\b",
        re.IGNORECASE,
    ),
    "password_or_recovery": re.compile(
        r"\b(?:forgot password|reset password|reset my password|locked out of apple id|two[- ]factor|verification code|security questions?|activation lock|passcode locked|disabled apple id|apple id is locked)\b",
        re.IGNORECASE,
    ),
    "unauthorized_payment_or_fraud": re.compile(
        r"\b(?:unauthorized charge|unauthorized purchase|fraud|fraudulent|scam|scammed|stolen card|charged without permission|unknown charge|unknown purchase|charged twice|unauthorized transaction)\b",
        re.IGNORECASE,
    ),
    "refund_or_payment_dispute": re.compile(
        r"\b(?:dispute charge|dispute the charge|want my refund|demand a refund|give me my refund|chargeback|billing dispute|refund status|refund my money|where is my refund)\b",
        re.IGNORECASE,
    ),
}


def detect_safety_flags(text: str) -> List[str]:
    """
    Scans customer text for predefined safety and risk trigger phrases.
    Returns a sorted list of matched safety flag identifiers.
    """
    if not text or not isinstance(text, str):
        return []

    flags = []
    for flag_name, pattern in SAFETY_PATTERNS.items():
        if pattern.search(text):
            flags.append(flag_name)

    return sorted(flags)


def is_restricted(safety_flags: List[str]) -> bool:
    """
    Evaluates whether the presence of safety flags requires a restricted draft.
    Any matched safety flag triggers restricted_draft = True.
    """
    return bool(len(safety_flags) > 0)


def get_restricted_draft(safety_flags: List[str]) -> Tuple[str, str]:
    """
    Generates a pre-approved, neutral, safe handoff response and grounding note
    tailored to the highest-severity matched safety flag.

    Returns:
      (draft_reply, grounding_note)
    """
    if not safety_flags:
        return (
            "Please contact Apple Support through an official secure channel for assistance.",
            "Standard fallback handoff; no safety flags detected.",
        )

    # Priority hierarchy for safety messaging:
    # 1. Self-harm / crisis (Highest priority)
    # 2. Threat / legal escalation
    # 3. Account compromise / password recovery
    # 4. Billing fraud / dispute
    if "self_harm_or_danger" in safety_flags:
        draft = (
            "If you or someone you know is in immediate danger or distress, please contact "
            "your local emergency services or a national crisis lifeline immediately. "
            "Customer support channels cannot provide crisis assistance."
        )
        note = (
            "Safety restriction triggered: self-harm/crisis terms detected. Provided emergency handoff; "
            "marked for urgent human safety escalation."
        )
        return draft, note

    if "threats_abuse_legal" in safety_flags:
        draft = (
            "For legal or formal dispute matters, please contact Apple through official corporate "
            "or legal channels. Social support channels cannot handle legal or regulatory notices."
        )
        note = (
            "Safety restriction triggered: legal/threat terms detected. Provided formal channel handoff; "
            "marked for human escalation."
        )
        return draft, note

    if "account_compromise" in safety_flags:
        draft = (
            "For your account security, please use Apple’s official account-recovery or support channels "
            "immediately. Do not share passwords, verification codes, or personal details publicly."
        )
        note = (
            "Safety restriction triggered: account compromise terms detected. Draft enforces strict "
            "credential security and directs to official recovery channels."
        )
        return draft, note

    if "password_or_recovery" in safety_flags:
        draft = (
            "For account access or password concerns, please use Apple’s official account-recovery "
            "or secure support channels. Never share passwords, verification codes, or personal credentials publicly."
        )
        note = (
            "Safety restriction triggered: password/recovery terms detected. Draft directs to secure "
            "account recovery without requesting private credentials."
        )
        return draft, note

    if "unauthorized_payment_or_fraud" in safety_flags or "refund_or_payment_dispute" in safety_flags:
        draft = (
            "For payment, purchase history, or refund concerns, please use Apple’s secure support "
            "or purchase-history options. Do not share payment details or card numbers publicly."
        )
        note = (
            "Safety restriction triggered: billing/fraud/dispute terms detected. Draft enforces payment "
            "privacy and directs to secure purchase support without asserting transaction status."
        )
        return draft, note

    # Generic safety handoff
    draft = (
        "Please contact Apple Support through an official secure channel for assistance with this inquiry. "
        "Do not share sensitive account or personal information publicly."
    )
    note = f"Safety restriction triggered: flags={'; '.join(safety_flags)}. Restricted draft applied."
    return draft, note
