# Phase 9: Risk-Aware Routing Policy Methodology

## Executive Summary

Phase 9 implements the final decision-support routing component of the AppleSupport conversational inquiry system. It serves as the operational governor that decides whether a drafted response may be considered an **auto-handle candidate** or must be **escalated to a human specialist**.

```text
Intent prediction (Phase 6)
+ historical retrieval strength (Phase 7)
+ safe reply draft mode (Phase 8)
+ deterministic safety restrictions (Phase 8)
        ↓
Routing Policy Engine (Phase 9)
        ↓
auto_handle candidate OR human escalation
+ decision_status: offline_candidate_not_sent
+ primary_reason_code & all_reason_codes
+ decision_explanation
```

> [!IMPORTANT]
> **Core Policy Boundaries & Epistemic Rules**:
> 1. **Drafting is NOT Sending**: Generating a fluent or polite draft does not make it safe or appropriate to send to a customer.
> 2. **Candidate Meaning**: `auto_handle` designates an *eligible auto-handle candidate*, not deployed or unmonitored automation. Every record in this prototype is labeled `decision_status = 'offline_candidate_not_sent'`.
> 3. **Fail-Safe Conservative Escalation**: Any uncertainty in intent, weakness in retrieval evidence, presence of high-risk subject matter, or active safety flag forces immediate escalation to human agents.
> 4. **No Automated Accuracy Claims**: Descriptive distributions only. True policy accuracy cannot be computed until human adjudication of the golden evaluation benchmark is complete.

---

## 1. Why an AI Reply Draft is Not Automatically Safe to Send

In customer support engineering, separating **generation (drafting)** from **execution (routing)** is a fundamental architectural best practice:
- A reply drafter answers: *"What is the safest, most helpful response pattern for this inquiry if we were to reply?"*
- The routing policy answers: *"Do we have sufficient confidence, precedent, and domain safety to allow this interaction to proceed without human eyes?"*

Even when drafts use constrained templates and grounded historical patterns, automated sending without human oversight introduces major operational hazards:
- **Misclassified Intent**: A customer asking to cancel an order might be misclassified as a software inquiry. Sending an automated software troubleshooting reply frustrates the user.
- **Novel Edge Cases**: An inquiry may use familiar words in an unprecedented combination that retrieval matches with false confidence.
- **Regulatory and Financial Exposure**: Premature automation on billing disputes, account takeovers, or hardware damage risks brand reputation and compliance failure.

Therefore, **drafting creates candidate evidence; routing determines operational risk.**

---

## 2. Multi-Signal Governance: How Decisions Combine

The routing engine evaluates five distinct signals simultaneously:

1. **Domain Risk Tier (`predicted_intent`)**:
   - **Low-Risk Technical Intents**: `software_update_or_os_issue`, `device_performance_or_hardware`, `connectivity_and_network`, `apps_services_or_icloud`. These inquiries involve general device troubleshooting and public self-service guidance.
   - **High-Risk Business Intents**: `account_access_and_apple_id`, `billing_subscription_or_purchase`, `repair_replacement_or_order`. These require secure credentials, payment verification, or hardware inspection.
   - **Ambiguous / Unclassified**: `other_or_unclear`. Unclassified inquiries lack a verified intent and cannot be automated.
2. **Intent Confidence (`predicted_confidence`)**:
   - Must meet or exceed **`0.80`** for auto-handling.
   - Inquiries with confidence $< 0.80$ or flagged with `needs_human_review = True` escalate immediately under `uncertain_intent`.
3. **Retrieval Grounding Strength (`best_similarity_score`)**:
   - Must meet or exceed **`0.50`** for auto-handling.
   - Inquiries with similarity $< 0.50$ escalate under `insufficient_historical_evidence`.
4. **Draft Mode (`draft_mode`)**:
   - Must equal **`template_with_historical_pattern`**.
   - Inquiries in `conservative_no_evidence` escalate under `conservative_draft_mode`.
5. **Safety Restrictions (`restricted_draft` & `safety_flags`)**:
   - Must have `restricted_draft == False` and empty normalized `safety_flags`.
   - Any detected account takeover, fraud, dispute, threat, or crisis language immediately escalates under `restricted_safety_flag`.

---

## 3. Strict Decision Priority Order

When an inquiry is evaluated, conditions are tested in a strict hierarchical order to establish the `primary_reason_code`. Additionally, **all** triggered constraints are recorded in `all_reason_codes`:

```text
Priority 1: restricted_safety_flag
  ↳ Triggered if restricted_draft is True or safety_flags is non-empty.

Priority 2: high_risk_intent
  ↳ Triggered if intent is account_access, billing, or repair/replacement.

Priority 3: other_or_unclear
  ↳ Triggered if intent is ambiguous or unclassified.

Priority 4: uncertain_intent
  ↳ Triggered if predicted_confidence < 0.80 or needs_human_review is True.

Priority 5: insufficient_historical_evidence
  ↳ Triggered if best_similarity_score < 0.50.

Priority 6: conservative_draft_mode
  ↳ Triggered if draft_mode != template_with_historical_pattern.

Priority 7: eligible_low_risk_case (Auto-Handle)
  ↳ Assigned only if ZERO escalation conditions are triggered.
```

---

## 4. Why High Confidence Alone is Not Enough

A common failure mode in naive ML deployments is relying exclusively on classifier probability (e.g., `confidence >= 0.80`) to trigger automation. 

In this system, high confidence alone is **insufficient**:
- A customer asking *"How do I dispute this fraudulent charge?"* may be classified into `billing_subscription_or_purchase` with **0.98 confidence**. Under naive confidence routing, this would auto-handle! In our policy, condition 2 (`high_risk_intent`) and condition 1 (`restricted_safety_flag`) force immediate escalation.
- A query may be classified as `software_update_or_os_issue` with 0.85 confidence, but historical retrieval similarity is only 0.22 (novel phrasing or edge case). Condition 5 (`insufficient_historical_evidence`) forces escalation.
- Only when **domain safety, high intent confidence, strong retrieval grounding, safe draft mode, and zero safety flags** converge is an inquiry designated `auto_handle`.

---

## 5. Threshold Scenarios & Golden-Set Prerequisite

In [`outputs/agent/routing_threshold_scenarios.md`](file:///c:/Users/msiva/Videos/HIVER/hiver_twitter_exploration/outputs/agent/routing_threshold_scenarios.md), we evaluated three confidence operating points:
- **`0.70`**: 280 auto-handle candidates (2.70% coverage)
- **`0.80` (Default)**: 238 auto-handle candidates (2.30% coverage)
- **`0.90`**: 180 auto-handle candidates (1.74% coverage)

### Methodological Rule:
**We do not declare any threshold "best" or "optimal."**

Because weak labels were used to train the initial classifier, model probabilities are uncalibrated estimates. Selecting a production operating threshold requires:
1. Completing the human adjudication of [`data/golden/golden_set_200.csv`](file:///c:/Users/msiva/Videos/HIVER/hiver_twitter_exploration/data/golden/golden_set_200.csv).
2. Auditing the 30-row review sheet ([`outputs/agent/routing_human_review_sheet.csv`](file:///c:/Users/msiva/Videos/HIVER/hiver_twitter_exploration/outputs/agent/routing_human_review_sheet.csv)).
3. Measuring the rate of *unsafe auto-handles* (false positives) against ground truth.
4. Setting the threshold where unsafe auto-handles are minimized according to enterprise risk tolerance.
