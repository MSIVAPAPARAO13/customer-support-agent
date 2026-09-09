#!/usr/bin/env python3
"""
Unit Test Suite for Risk-Aware Routing Policy (Phase 9)
======================================================

Tests:
  1. Hacked Apple ID -> escalate, restricted_safety_flag
  2. Unauthorized charge -> escalate, restricted_safety_flag
  3. Low-confidence Bluetooth issue -> escalate, uncertain_intent
  4. High-confidence Wi-Fi issue with similarity >= 0.50 -> auto_handle
  5. High-confidence repair request -> escalate, high_risk_intent
  6. other_or_unclear with high similarity -> escalate, other_or_unclear
  7. CSV string parsing test: 'False', 'none', '[]' -> auto_handle
  8. CSV string parsing test: 'True' -> escalate, restricted_safety_flag
  9. Dedicated parse_bool() edge case test
  10. Dedicated parse_safety_flags() edge case test
"""

import sys
from pathlib import Path

# Configure Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.routing_policy import (
    evaluate_routing_decision,
    parse_bool,
    parse_safety_flags,
    REASON_RESTRICTED_SAFETY,
    REASON_HIGH_RISK_INTENT,
    REASON_OTHER_UNCLEAR,
    REASON_UNCERTAIN_INTENT,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_CONSERVATIVE_DRAFT_MODE,
    REASON_ELIGIBLE_LOW_RISK,
    DECISION_STATUS_OFFLINE,
)


def test_parse_bool():
    """Validates robust boolean parsing across heterogeneous inputs."""
    # True representations
    assert parse_bool(True) is True
    assert parse_bool("True") is True
    assert parse_bool("true") is True
    assert parse_bool("TRUE") is True
    assert parse_bool(1) is True
    assert parse_bool("1") is True
    assert parse_bool("yes") is True

    # False representations
    assert parse_bool(False) is False
    assert parse_bool("False") is False
    assert parse_bool("false") is False
    assert parse_bool("FALSE") is False
    assert parse_bool(0) is False
    assert parse_bool("0") is False
    assert parse_bool("") is False
    assert parse_bool(None) is False
    assert parse_bool("none") is False
    assert parse_bool("null") is False
    assert parse_bool("nan") is False
    print("[PASS] test_parse_bool passed")


def test_parse_safety_flags():
    """Validates robust safety flags normalization."""
    assert parse_safety_flags("") == []
    assert parse_safety_flags("none") == []
    assert parse_safety_flags("[]") == []
    assert parse_safety_flags(None) == []
    assert parse_safety_flags("null") == []
    assert parse_safety_flags("nan") == []

    # Pipe separated
    assert parse_safety_flags("account_compromise|billing_fraud") == ["account_compromise", "billing_fraud"]
    # Comma separated
    assert parse_safety_flags("account_compromise, billing_fraud") == ["account_compromise", "billing_fraud"]
    # Semicolon separated
    assert parse_safety_flags("account_compromise; billing_fraud") == ["account_compromise", "billing_fraud"]
    # List input
    assert parse_safety_flags(["account_compromise", "billing_fraud"]) == ["account_compromise", "billing_fraud"]
    print("[PASS] test_parse_safety_flags passed")


def test_case_1_hacked_apple_id():
    """1. Hacked Apple ID: must escalate because of restricted safety."""
    row = {
        "customer_tweet_id": "101",
        "customer_text_clean": "My Apple ID was hacked and someone bought apps.",
        "predicted_intent": "account_access_and_apple_id",
        "predicted_confidence": 0.95,
        "best_similarity_score": 0.85,
        "draft_mode": "restricted_safety",
        "restricted_draft": True,
        "safety_flags": "account_compromise",
        "needs_human_review": False,
    }
    dec = evaluate_routing_decision(row)
    assert dec["action"] == "escalate"
    assert dec["primary_reason_code"] == REASON_RESTRICTED_SAFETY
    assert REASON_RESTRICTED_SAFETY in dec["all_reason_codes"]
    assert dec["decision_status"] == DECISION_STATUS_OFFLINE
    print("[PASS] Case 1 (Hacked Apple ID) passed")


def test_case_2_unauthorized_charge():
    """2. Unauthorized charge: must escalate because of restricted safety."""
    row = {
        "customer_tweet_id": "102",
        "customer_text_clean": "Why was my credit card charged twice without permission?",
        "predicted_intent": "billing_subscription_or_purchase",
        "predicted_confidence": 0.92,
        "best_similarity_score": 0.75,
        "draft_mode": "restricted_safety",
        "restricted_draft": True,
        "safety_flags": "unauthorized_payment_or_fraud",
        "needs_human_review": False,
    }
    dec = evaluate_routing_decision(row)
    assert dec["action"] == "escalate"
    assert dec["primary_reason_code"] == REASON_RESTRICTED_SAFETY
    assert dec["decision_status"] == DECISION_STATUS_OFFLINE
    print("[PASS] Case 2 (Unauthorized charge) passed")


def test_case_3_low_confidence_bluetooth():
    """3. Low-confidence Bluetooth issue: must escalate because of uncertain intent."""
    row = {
        "customer_tweet_id": "103",
        "customer_text_clean": "Bluetooth acting funny after the car drive.",
        "predicted_intent": "connectivity_and_network",
        "predicted_confidence": 0.65,  # < 0.80 threshold
        "best_similarity_score": 0.60,
        "draft_mode": "template_with_historical_pattern",
        "restricted_draft": False,
        "safety_flags": "none",
        "needs_human_review": True,
    }
    dec = evaluate_routing_decision(row)
    assert dec["action"] == "escalate"
    assert dec["primary_reason_code"] == REASON_UNCERTAIN_INTENT
    assert dec["decision_status"] == DECISION_STATUS_OFFLINE
    print("[PASS] Case 3 (Low-confidence Bluetooth) passed")


def test_case_4_high_confidence_wifi():
    """4. High-confidence Wi-Fi issue with similarity >= 0.50 and safe draft: may auto-handle."""
    row = {
        "customer_tweet_id": "104",
        "customer_text_clean": "My iPhone won't connect to my home Wi-Fi network.",
        "predicted_intent": "connectivity_and_network",
        "predicted_confidence": 0.88,
        "best_similarity_score": 0.70,
        "draft_mode": "template_with_historical_pattern",
        "restricted_draft": False,
        "safety_flags": "none",
        "needs_human_review": False,
    }
    dec = evaluate_routing_decision(row)
    assert dec["action"] == "auto_handle"
    assert dec["primary_reason_code"] == REASON_ELIGIBLE_LOW_RISK
    assert dec["all_reason_codes"] == REASON_ELIGIBLE_LOW_RISK
    assert dec["decision_status"] == DECISION_STATUS_OFFLINE
    print("[PASS] Case 4 (High-confidence Wi-Fi) passed")


def test_case_5_high_confidence_repair():
    """5. High-confidence repair request: must escalate because repair/replacement/order is high risk."""
    row = {
        "customer_tweet_id": "105",
        "customer_text_clean": "I cracked my screen and need to schedule a Genius Bar repair.",
        "predicted_intent": "repair_replacement_or_order",
        "predicted_confidence": 0.94,
        "best_similarity_score": 0.72,
        "draft_mode": "template_with_historical_pattern",
        "restricted_draft": False,
        "safety_flags": "none",
        "needs_human_review": False,
    }
    dec = evaluate_routing_decision(row)
    assert dec["action"] == "escalate"
    assert dec["primary_reason_code"] == REASON_HIGH_RISK_INTENT
    assert dec["decision_status"] == DECISION_STATUS_OFFLINE
    print("[PASS] Case 5 (High-confidence repair request) passed")


def test_case_6_other_or_unclear():
    """6. other_or_unclear with high similarity: must escalate."""
    row = {
        "customer_tweet_id": "106",
        "customer_text_clean": "Thanks for the response, appreciated.",
        "predicted_intent": "other_or_unclear",
        "predicted_confidence": 0.89,
        "best_similarity_score": 0.88,
        "draft_mode": "template_with_historical_pattern",
        "restricted_draft": False,
        "safety_flags": "none",
        "needs_human_review": False,
    }
    dec = evaluate_routing_decision(row)
    assert dec["action"] == "escalate"
    assert dec["primary_reason_code"] == REASON_OTHER_UNCLEAR
    assert dec["decision_status"] == DECISION_STATUS_OFFLINE
    print("[PASS] Case 6 (other_or_unclear) passed")


def test_case_7_csv_string_safe():
    """7. CSV string test: restricted_draft='False', needs_human_review='False', safety_flags='[]' -> auto_handle."""
    row = {
        "customer_tweet_id": "107",
        "customer_text_clean": "Wi-Fi toggle is greyed out.",
        "predicted_intent": "connectivity_and_network",
        "predicted_confidence": "0.85",
        "best_similarity_score": "0.65",
        "draft_mode": "template_with_historical_pattern",
        "restricted_draft": "False",  # string
        "safety_flags": "[]",          # string
        "needs_human_review": "False",# string
    }
    dec = evaluate_routing_decision(row)
    assert dec["action"] == "auto_handle"
    assert dec["primary_reason_code"] == REASON_ELIGIBLE_LOW_RISK
    assert dec["decision_status"] == DECISION_STATUS_OFFLINE
    print("[PASS] Case 7 (CSV string safe auto_handle) passed")


def test_case_8_csv_string_restricted():
    """8. CSV string test: restricted_draft='True' -> escalate, restricted_safety_flag."""
    row = {
        "customer_tweet_id": "108",
        "customer_text_clean": "Wi-Fi issue.",
        "predicted_intent": "connectivity_and_network",
        "predicted_confidence": "0.85",
        "best_similarity_score": "0.65",
        "draft_mode": "template_with_historical_pattern",
        "restricted_draft": "True",  # string
        "safety_flags": "[]",
        "needs_human_review": "False",
    }
    dec = evaluate_routing_decision(row)
    assert dec["action"] == "escalate"
    assert dec["primary_reason_code"] == REASON_RESTRICTED_SAFETY
    assert dec["decision_status"] == DECISION_STATUS_OFFLINE
    print("[PASS] Case 8 (CSV string restricted escalate) passed")


def main():
    print("=" * 60)
    print("RUNNING PHASE 9 ROUTING POLICY UNIT TESTS")
    print("=" * 60)
    test_parse_bool()
    test_parse_safety_flags()
    test_case_1_hacked_apple_id()
    test_case_2_unauthorized_charge()
    test_case_3_low_confidence_bluetooth()
    test_case_4_high_confidence_wifi()
    test_case_5_high_confidence_repair()
    test_case_6_other_or_unclear()
    test_case_7_csv_string_safe()
    test_case_8_csv_string_restricted()
    print("=" * 60)
    print("ALL 10 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
