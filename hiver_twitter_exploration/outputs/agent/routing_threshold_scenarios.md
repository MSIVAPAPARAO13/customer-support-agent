# Phase 9 Routing Threshold Scenarios Analysis

## Purpose & Methodological Caveats

This report analyzes the impact of varying the **intent prediction confidence threshold** on automation coverage. All other conditions (historical retrieval similarity threshold $\ge 0.50$, draft mode, zero safety flags, low-risk intents only) remain strictly held constant across all three scenarios.

> [!IMPORTANT]
> **Critical Evaluation Rules**:
> 1. **No Setting is Designated 'Best'**: We do not rank or declare any threshold 'optimal'.
> 2. **No Performance or Accuracy Metrics Claimed**: Without completed adjudicated human labels on the golden set, true precision and error rates cannot be computed.
> 3. **Trade-Off Nature**: Lowering the threshold increases automation volume but introduces higher risk of misclassification; raising it increases human workload.
> 4. **Prerequisite**: Selecting an operational threshold requires evaluating the completed human golden evaluation set.

---

## 1. Scenario Coverage Comparison Across Validation Inquiries (10,357 Total)

| Confidence Threshold | Auto-Handle Candidates | Auto-Handle Coverage | Escalation Volume | Escalation Rate | Operational Profile |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `0.70` | **280** | **2.70%** | **10,077** | **97.30%** | More permissive technical auto-handling; captures borderline model predictions. |
| `0.80` | **238** | **2.30%** | **10,119** | **97.70%** | Default conservative baseline; enforces solid model confidence and grounding. |
| `0.90` | **180** | **1.74%** | **10,177** | **98.26%** | Extremely conservative; restricts automated candidates to near-certain predictions. |

---

## 2. Auto-Handle Candidates by Intent Across Scenarios

Only the four approved low-risk technical intents ever produce auto-handle candidates. High-risk domains (`account_access_and_apple_id`, `billing_subscription_or_purchase`, `repair_replacement_or_order`) and `other_or_unclear` remain strictly 0% auto-handled across all thresholds.

| Approved Intent | Threshold 0.70 | Threshold 0.80 (Default) | Threshold 0.90 |
| :--- | :---: | :---: | :---: |
| `software_update_or_os_issue` | 219 | 185 | 142 |
| `device_performance_or_hardware` | 20 | 19 | 16 |
| `connectivity_and_network` | 19 | 18 | 13 |
| `apps_services_or_icloud` | 22 | 16 | 9 |

---

## 3. High-Risk & Ambiguous Domains (Invariant Across All Scenarios)

| Intent Category | Threshold 0.70 | Threshold 0.80 | Threshold 0.90 | Policy Guarantee |
| :--- | :---: | :---: | :---: | :--- |
| `account_access_and_apple_id` | 0 | 0 | 0 | Mandatory human escalation (identity verification) |
| `billing_subscription_or_purchase` | 0 | 0 | 0 | Mandatory human escalation (payment security) |
| `repair_replacement_or_order` | 0 | 0 | 0 | Mandatory human escalation (warranty & logistics) |
| `other_or_unclear` | 0 | 0 | 0 | Mandatory human escalation (intent ambiguity) |

---

## 4. Next Step: Golden Set Adjudication

To empirically select an operational threshold for production deployment:
1. Complete human adjudication of `data/golden/golden_set_200.csv`.
2. Evaluate false-positive auto-handle rates (unsafe auto-handles) on golden rows.
3. Select the operating point that guarantees near-zero unsafe auto-handles under organizational risk tolerance.

