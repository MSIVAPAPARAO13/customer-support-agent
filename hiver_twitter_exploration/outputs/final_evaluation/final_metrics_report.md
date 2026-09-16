# Final Evaluation Report: Golden Benchmark Assessment (Phase 10)

## Executive Summary & Integrity Notice

This report presents the certified evaluation of the offline AppleSupport customer assistant against the protected 200-row human golden benchmark.

> A conservative system can improve selective accuracy simply by escalating more messages. Automation coverage must always be reported alongside auto-handle quality.

> **Routing Behavior Context**:
> An auto-handle coverage of 2.3% (or similar conservative policy) is a deliberate risk-minimization routing behavior, NOT independent evidence of accuracy, safety, or deployment readiness. High selective accuracy on an extremely narrow cohort simply reflects conservative gating.

---

## 1. Intent Classification System Comparison

Comparison across exactly three intent architectures evaluated against identical human ground truth:

| System Name | Description | Accuracy | Macro F1 (Fixed 8 Classes) |
| :--- | :--- | :---: | :---: |
| **`majority_weak_intent`** | Minimal baseline: always predicts training mode (`other_or_unclear`) | 0.0450 | 0.0108 |
| **`keyword_rule_classifier`** | Deterministic domain regex & keyword heuristics | 0.4300 | 0.5332 |
| **`tfidf_logistic_regression`** | Main trained ML model (sublinear TF-IDF + balanced LogReg) | **0.5950** | **0.6630** |

*Note: All Macro-F1 calculations strictly include all eight fixed taxonomy categories. Bootstrapping does not omit rare categories.*

---

## 2. End-to-End Operational Routing Metrics

Evaluation of the operational action routing (`auto_handle` vs. `escalate`):

| Metric | Value | Definition / Interpretation |
| :--- | :---: | :--- |
| **Action Accuracy** | `0.3600` | Fraction of cases where agent action matches human decision exactly |
| **Auto-Handle Coverage** | `0.0200` | Fraction of inquiries candidate for automated handling (`4/200`) |
| **Selective Accuracy** | `0.75` | Accuracy strictly among auto-handled cases (N/A if 0 auto-handled) |
| **Unsafe Auto-Handle Count** | `1` | Critical safety errors: Agent auto-handled, but Human required escalation |
| **Unsafe Auto-Handle Rate** | `0.25` | `unsafe_auto_handles / auto_handles` (N/A if 0 auto-handled) |
| **Escalation Precision** | `0.3520` | Among agent escalations, fraction humans also marked escalate (`escalate` is positive class) |
| **Escalation Recall** | `0.9857` | Among human escalations, fraction agent successfully escalated (`escalate` is positive class) |

---

## 3. Bootstrap 95% Confidence Intervals (1,000 Resamples, Seed 42)

| Metric | Point Estimate | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Main Intent Accuracy** | `0.5950` | `[0.5250, 0.6601]` |
| **Main Intent Macro F1** | `0.6630` | `[0.5988, 0.7145]` |
| **Action Accuracy** | `0.3600` | `[0.2900, 0.4300]` |
| **Auto-Handle Coverage** | `0.0200` | `[0.0050, 0.0400]` |
| **Selective Accuracy** | `0.75` | `(0.0, 1.0)` |

---

## 4. Per-Intent Performance Breakdown (Main Model)

See complete table in [`per_intent_metrics.csv`](per_intent_metrics.csv).

| Taxonomy Category | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| `software_update_or_os_issue` | 58 | 0.8947 | 0.5862 | 0.7083 |
| `device_performance_or_hardware` | 35 | 0.8500 | 0.4857 | 0.6182 |
| `connectivity_and_network` | 14 | 0.8750 | 1.0000 | 0.9333 |
| `apps_services_or_icloud` | 34 | 0.7895 | 0.4412 | 0.5660 |
| `account_access_and_apple_id` | 19 | 0.9412 | 0.8421 | 0.8889 |
| `billing_subscription_or_purchase` | 12 | 1.0000 | 0.6667 | 0.8000 |
| `repair_replacement_or_order` | 19 | 1.0000 | 0.4737 | 0.6429 |
| `other_or_unclear` | 9 | 0.0822 | 0.6667 | 0.1463 |

---

## 5. Confusion Matrix & Failure Analysis

- **Confusion Matrix**: Full 8x8 matrix exported to [`intent_confusion_matrix.csv`](intent_confusion_matrix.csv).
- **Failure Candidates**: Total 153 discrepancies flagged for qualitative audit in [`failure_candidates.csv`](failure_candidates.csv).

### Offline Prototype Safeguard Reminder
All decisions evaluated here are marked `decision_status = 'offline_candidate_not_sent'`. No messages were transmitted to live Twitter users.
