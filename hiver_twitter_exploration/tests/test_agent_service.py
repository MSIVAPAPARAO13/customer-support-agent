"""
Unit Tests for Unified SupportAgentService
==========================================
"""

import unittest
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent_service import SupportAgentService, normalize_customer_input


class TestSupportAgentService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models_dir = REPO_ROOT / "models"
        cls.service = SupportAgentService(models_dir=models_dir)

    def test_normalize_customer_input(self):
        raw = "@AppleSupport @115858 My iPhone won't connect to Wi-Fi! https://t.co/xyz123   help please"
        clean = normalize_customer_input(raw)
        self.assertNotIn("@AppleSupport", clean)
        self.assertNotIn("@115858", clean)
        self.assertIn("[URL]", clean)
        self.assertTrue(clean.startswith("My iPhone"))

    def test_intent_probabilities(self):
        text = "My battery drains completely within one hour on iOS 11"
        res = self.service.classify_intent(text)
        self.assertIn(res["predicted_intent"], self.service.intent_classes)
        self.assertGreaterEqual(res["predicted_confidence"], 0.0)
        self.assertLessEqual(res["predicted_confidence"], 1.0)
        self.assertEqual(len(res["top_3_candidates"]), 3)
        prob_sum = sum(res["all_probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, delta=0.05)

    def test_hacked_account_escalates(self):
        text = "@AppleSupport My Apple ID has been hacked and my password was changed! Help!"
        res = self.service.process_inquiry(text)
        self.assertEqual(res["routing"]["action"], "escalate")
        self.assertTrue(
            res["routing"]["primary_reason_code"] in ["restricted_safety_flag", "high_risk_intent"]
        )
        self.assertTrue(res["reply_draft"]["restricted_draft"])

    def test_unauthorized_billing_escalates(self):
        text = "I was charged $9.99 for Apple Music subscription without my permission, refund me!"
        res = self.service.process_inquiry(text)
        self.assertEqual(res["routing"]["action"], "escalate")
        self.assertIn(
            res["routing"]["primary_reason_code"],
            ["restricted_safety_flag", "high_risk_intent"]
        )

    def test_empty_or_unknown_vocabulary_safely_handled(self):
        text = "asdfqwerzxcv12345"
        res = self.service.process_inquiry(text)
        self.assertEqual(res["routing"]["action"], "escalate")
        self.assertEqual(res["retrieval"]["best_similarity_score"], 0.0)
        self.assertEqual(res["retrieval"]["lexical_similarity_band"], "insufficient_lexical_evidence")

    def test_end_to_end_structure(self):
        text = "My iPhone 7 will not connect to my home Wi-Fi network after rebooting"
        res = self.service.process_inquiry(text)
        self.assertIn("query", res)
        self.assertIn("intent", res)
        self.assertIn("retrieval", res)
        self.assertIn("reply_draft", res)
        self.assertIn("routing", res)
        self.assertIn(res["routing"]["action"], ["auto_handle", "escalate"])
        self.assertIn("draft_reply", res["reply_draft"])


if __name__ == "__main__":
    unittest.main()
