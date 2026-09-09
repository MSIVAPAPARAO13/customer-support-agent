# Phase 9 Support Agent Routing Policy Audit

## Executive Summary & Safety Policy Status

In **Phase 9**, we implemented the final offline decision-support routing component of the AppleSupport inquiry pipeline. The routing engine evaluates model confidence, retrieval similarity, draft modes, and safety restrictions to make inspectable, deterministic routing determinations.

> [!IMPORTANT]
> **Core Policy Principles & Prototype Boundaries**:
> 1. **Offline Prototype Status**: Every decision in this phase is marked `decision_status = 'offline_candidate_not_sent'`. No messages are sent to Twitter.
> 2. **Candidate Meaning**: `auto_handle` designates an *eligible auto-handle candidate*, not deployed or unmonitored automation.
> 3. **Fail-Safe Escalation**: Any triggered risk flag, high-risk intent, low confidence ($< 0.80$), weak similarity ($< 0.50$), or conservative draft mode forces immediate human escalation.
> 4. **No Automated Accuracy Claims**: Descriptive distributions only. Deciding true routing accuracy requires completed human golden adjudication.

---

## 1. Overall Routing Decision Distributions (`validation_agent_decisions.csv`)

- **Total Inquiries Evaluated**: **10,357**
- **Auto-Handle Candidates (`auto_handle`)**: **238 (2.30%)**
- **Escalated to Human Specialist (`escalate`)**: **10,119 (97.70%)**
- **Inquiries with Active Safety Flags / Restricted Drafts**: **74 (0.71%)**

### Primary Reason Code Breakdown

The primary reason code reflects the first triggered constraint according to the strict priority hierarchy:

| Primary Reason Code | Action | Count | Proportion | Policy Meaning |
| :--- | :---: | :---: | :---: | :--- |
| `other_or_unclear` | `escalate` | 5,921 | 57.17% | Inquiry intent is ambiguous or unclassified ('other_or_unclear') |
| `uncertain_intent` | `escalate` | 2,662 | 25.70% | Model intent confidence is below 0.80 threshold or flagged for upstream review |
| `insufficient_historical_evidence` | `escalate` | 1,073 | 10.36% | Retrieval similarity is below 0.50 threshold; lacks sufficient historical grounding |
| `high_risk_intent` | `escalate` | 389 | 3.76% | Intent classified into account, billing, or repair/hardware order domain |
| `eligible_low_risk_case` | `auto_handle` | 238 | 2.30% | All safety, confidence, and grounding criteria satisfied for low-risk technical intent |
| `restricted_safety_flag` | `escalate` | 74 | 0.71% | Account compromise, password recovery, billing fraud, or legal/crisis risk terms detected |

### Total Reason Code Occurrences (`all_reason_codes`)

Inquiries frequently trigger multiple escalation conditions simultaneously. Below is the total occurrence count across all triggered reasons:

| Reason Code | Total Occurrences across All Inquiries |
| :--- | :---: |
| `conservative_draft_mode` | 8,587 |
| `insufficient_historical_evidence` | 8,580 |
| `uncertain_intent` | 8,351 |
| `other_or_unclear` | 5,945 |
| `high_risk_intent` | 433 |
| `eligible_low_risk_case` | 238 |
| `restricted_safety_flag` | 74 |

### Routing Actions by Predicted Intent

| Predicted Intent | Auto-Handle Candidates | Escalated | Total Inquiries | Auto-Handle Rate |
| :--- | :---: | :---: | :---: | :---: |
| `account_access_and_apple_id` | 0 | 207 | 207 | 0.00% |
| `apps_services_or_icloud` | 16 | 428 | 444 | 3.60% |
| `billing_subscription_or_purchase` | 0 | 123 | 123 | 0.00% |
| `connectivity_and_network` | 18 | 457 | 475 | 3.79% |
| `device_performance_or_hardware` | 19 | 530 | 549 | 3.46% |
| `other_or_unclear` | 0 | 5,945 | 5,945 | 0.00% |
| `repair_replacement_or_order` | 0 | 103 | 103 | 0.00% |
| `software_update_or_os_issue` | 185 | 2,326 | 2,511 | 7.37% |

---

## 2. Exemplary De-Identified Decision Cases

### A. Five Auto-Handle Candidates (`action = 'auto_handle'`)

#### Auto-Handle Example 1: Tweet `40434`
- **Customer Inquiry**: "7 plus & ios 11"
- **Predicted Intent**: `software_update_or_os_issue` (Confidence: `0.9997`)
- **Retrieval Similarity**: `1.0000` (`high_lexical_similarity`)
- **Draft Mode**: `template_with_historical_pattern`
- **Action**: `auto_handle` (`offline_candidate_not_sent`)
- **Primary Reason**: `eligible_low_risk_case`
- **Draft Reply**: > "We understand you are experiencing an issue related to software or updating. We recommend reviewing Apple’s official support resources for software updates. For troubleshooting tailored to your setup, please reach out through an official secure channel. Please connect with Apple Support through an official secure channel so an advisor can review the specifics with you."
- **Explanation**: Low-risk technical inquiry (software_update_or_os_issue) with high intent confidence (0.9997 >= 0.80), strong historical retrieval (1.0000 >= 0.50), safe draft mode, and zero safety flags. Eligible for automated handling candidate.

#### Auto-Handle Example 2: Tweet `40480`
- **Customer Inquiry**: "IOS 11.1, And just my social media apps,"
- **Predicted Intent**: `software_update_or_os_issue` (Confidence: `0.8816`)
- **Retrieval Similarity**: `0.7127` (`high_lexical_similarity`)
- **Draft Mode**: `template_with_historical_pattern`
- **Action**: `auto_handle` (`offline_candidate_not_sent`)
- **Primary Reason**: `eligible_low_risk_case`
- **Draft Reply**: > "We understand you are experiencing an issue related to software or updating. We recommend reviewing Apple’s official support resources for software updates. For troubleshooting tailored to your setup, please reach out through an official secure channel. Please connect with Apple Support through an official secure channel so an advisor can review the specifics with you."
- **Explanation**: Low-risk technical inquiry (software_update_or_os_issue) with high intent confidence (0.8816 >= 0.80), strong historical retrieval (0.7127 >= 0.50), safe draft mode, and zero safety flags. Eligible for automated handling candidate.

#### Auto-Handle Example 3: Tweet `103217`
- **Customer Inquiry**: "iOS 11.1.2 - it’s up to date"
- **Predicted Intent**: `software_update_or_os_issue` (Confidence: `0.9516`)
- **Retrieval Similarity**: `0.7820` (`high_lexical_similarity`)
- **Draft Mode**: `template_with_historical_pattern`
- **Action**: `auto_handle` (`offline_candidate_not_sent`)
- **Primary Reason**: `eligible_low_risk_case`
- **Draft Reply**: > "We understand you are experiencing an issue related to software or updating. We recommend reviewing Apple’s official support resources for software updates. For troubleshooting tailored to your setup, please reach out through an official secure channel. Please connect with Apple Support through an official secure channel so an advisor can review the specifics with you."
- **Explanation**: Low-risk technical inquiry (software_update_or_os_issue) with high intent confidence (0.9516 >= 0.80), strong historical retrieval (0.7820 >= 0.50), safe draft mode, and zero safety flags. Eligible for automated handling candidate.

#### Auto-Handle Example 4: Tweet `110875`
- **Customer Inquiry**: "It’s iOS 11.1.1"
- **Predicted Intent**: `software_update_or_os_issue` (Confidence: `0.9998`)
- **Retrieval Similarity**: `1.0000` (`high_lexical_similarity`)
- **Draft Mode**: `template_with_historical_pattern`
- **Action**: `auto_handle` (`offline_candidate_not_sent`)
- **Primary Reason**: `eligible_low_risk_case`
- **Draft Reply**: > "We understand you are experiencing an issue related to software or updating. We recommend reviewing Apple’s official support resources for software updates. For troubleshooting tailored to your setup, please reach out through an official secure channel. To help troubleshoot, have your device model and current software version ready when contacting official Apple Support through a secure channel."
- **Explanation**: Low-risk technical inquiry (software_update_or_os_issue) with high intent confidence (0.9998 >= 0.80), strong historical retrieval (1.0000 >= 0.50), safe draft mode, and zero safety flags. Eligible for automated handling candidate.

#### Auto-Handle Example 5: Tweet `148278`
- **Customer Inquiry**: "Not at all... I’m on iOS 11.1.2"
- **Predicted Intent**: `software_update_or_os_issue` (Confidence: `0.9760`)
- **Retrieval Similarity**: `0.7203` (`high_lexical_similarity`)
- **Draft Mode**: `template_with_historical_pattern`
- **Action**: `auto_handle` (`offline_candidate_not_sent`)
- **Primary Reason**: `eligible_low_risk_case`
- **Draft Reply**: > "We understand you are experiencing an issue related to software or updating. We recommend reviewing Apple’s official support resources for software updates. For troubleshooting tailored to your setup, please reach out through an official secure channel. Please connect with Apple Support through an official secure channel so an advisor can review the specifics with you."
- **Explanation**: Low-risk technical inquiry (software_update_or_os_issue) with high intent confidence (0.9760 >= 0.80), strong historical retrieval (0.7203 >= 0.50), safe draft mode, and zero safety flags. Eligible for automated handling candidate.

### B. Five Escalation Decisions (`action = 'escalate'`)

#### Escalation Example 1: Tweet `725`
- **Customer Inquiry**: "I️ upgraded. I️t didn’t work."
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.5943`)
- **Retrieval Similarity**: `0.8431` (`high_lexical_similarity`)
- **Safety Flags**: `none` (Restricted: `False`)
- **Action**: `escalate` (`offline_candidate_not_sent`)
- **Primary Reason**: `other_or_unclear`
- **All Reasons**: `other_or_unclear | uncertain_intent`
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly. Please connect with Apple Support through an official secure channel so an advisor can review the specifics with you."
- **Explanation**: Customer intent is ambiguous or unclassified ('other_or_unclear'). Automated handling cannot safely diagnose or resolve unclassified inquiries.

#### Escalation Example 2: Tweet `730`
- **Customer Inquiry**: "Hello, internet. Can someone explain why this symbol keeps appearing on my phone and when I️ try to type the letter I️? Also [USER] [URL]"
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.6700`)
- **Retrieval Similarity**: `0.3411` (`weak_lexical_evidence`)
- **Safety Flags**: `none` (Restricted: `False`)
- **Action**: `escalate` (`offline_candidate_not_sent`)
- **Primary Reason**: `other_or_unclear`
- **All Reasons**: `other_or_unclear | uncertain_intent | insufficient_historical_evidence | conservative_draft_mode`
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly."
- **Explanation**: Customer intent is ambiguous or unclassified ('other_or_unclear'). Automated handling cannot safely diagnose or resolve unclassified inquiries.

#### Escalation Example 3: Tweet `1757`
- **Customer Inquiry**: "I️ just updated it and it’s still coming up [URL]"
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.5394`)
- **Retrieval Similarity**: `0.5628` (`moderate_lexical_evidence`)
- **Safety Flags**: `none` (Restricted: `False`)
- **Action**: `escalate` (`offline_candidate_not_sent`)
- **Primary Reason**: `other_or_unclear`
- **All Reasons**: `other_or_unclear | uncertain_intent`
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly. To help troubleshoot, have your device model and current software version ready when contacting official Apple Support through a secure channel."
- **Explanation**: Customer intent is ambiguous or unclassified ('other_or_unclear'). Automated handling cannot safely diagnose or resolve unclassified inquiries.

#### Escalation Example 4: Tweet `1759`
- **Customer Inquiry**: "Why is “I️” keep changing to this and how do I️ stop it🤦🏽‍♀️ [USER] [USER] #anybody [URL]"
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.7855`)
- **Retrieval Similarity**: `0.4895` (`weak_lexical_evidence`)
- **Safety Flags**: `none` (Restricted: `False`)
- **Action**: `escalate` (`offline_candidate_not_sent`)
- **Primary Reason**: `other_or_unclear`
- **All Reasons**: `other_or_unclear | uncertain_intent | insufficient_historical_evidence | conservative_draft_mode`
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly."
- **Explanation**: Customer intent is ambiguous or unclassified ('other_or_unclear'). Automated handling cannot safely diagnose or resolve unclassified inquiries.

#### Escalation Example 5: Tweet `1781`
- **Customer Inquiry**: "why can’t I change ringer volume with the buttons? Whose dumb idea was it to change that and how do they still have a job?"
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.4178`)
- **Retrieval Similarity**: `0.2430` (`insufficient_lexical_evidence`)
- **Safety Flags**: `none` (Restricted: `False`)
- **Action**: `escalate` (`offline_candidate_not_sent`)
- **Primary Reason**: `other_or_unclear`
- **All Reasons**: `other_or_unclear | uncertain_intent | insufficient_historical_evidence | conservative_draft_mode`
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly."
- **Explanation**: Customer intent is ambiguous or unclassified ('other_or_unclear'). Automated handling cannot safely diagnose or resolve unclassified inquiries.

---

## 3. Policy Limitations & Boundaries

1. **Conservative Automation Bias**: With only 2.30% of validation inquiries qualifying for auto-handling under the default 0.80 confidence and 0.50 similarity thresholds, the system strongly prioritizes safety over automation volume.
2. **Weak-Label Classifier Reliance**: Intent confidence scores originate from a weak-label prototype. True calibration requires the completed 200-row human golden set.
3. **Absence of Real-Time System Telemetry**: The router cannot verify whether a customer's device is currently under warranty or if an Apple ID is locked.
4. **Human Review Prerequisite**: All 30 review rows in `routing_human_review_sheet.csv` are initialized to `NEEDS_HUMAN_REVIEW`. No deployment should occur prior to human audit.

