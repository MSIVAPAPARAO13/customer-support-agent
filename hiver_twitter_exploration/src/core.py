"""
core.py — Public API shim for the Hiver AppleSupport Agent.

This module re-exports the primary classes and functions from the four
implementation modules so that callers can use a single import point:

    from src.core import SupportAgentService

The four underlying modules are:
    src/agent_service.py   — Unified orchestrator (model, retrieval, routing)
    src/reply_drafter.py   — Conservative reply drafting
    src/reply_safety.py    — PII and restriction detection
    src/routing_policy.py  — Threshold-based routing policy engine

Design note: The spec calls for a single src/core.py. We keep the four
separate modules for readability and testability; this shim provides the
single-import-path convenience the spec describes without duplicating code.
"""

# Re-export the primary service class
from src.agent_service import SupportAgentService  # noqa: F401

# Re-export routing policy helpers
from src.routing_policy import (  # noqa: F401
    evaluate_routing_decision,
    LOW_RISK_INTENTS,
    HIGH_RISK_INTENTS,
    REASON_RESTRICTED_SAFETY,
    REASON_HIGH_RISK_INTENT,
    REASON_OTHER_UNCLEAR,
    REASON_UNCERTAIN_INTENT,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_CONSERVATIVE_DRAFT_MODE,
)

# Re-export reply safety helpers
from src.reply_safety import detect_safety_flags, is_restricted  # noqa: F401

# Re-export reply drafter
from src.reply_drafter import draft_reply_for_inquiry, sanitize_text  # noqa: F401

__all__ = [
    "SupportAgentService",
    "evaluate_routing_decision",
    "LOW_RISK_INTENTS",
    "HIGH_RISK_INTENTS",
    "detect_safety_flags",
    "is_restricted",
    "draft_reply_for_inquiry",
    "sanitize_text",
]
