# WEAK-LABEL DIAGNOSTIC ONLY — NOT HUMAN-GROUND-TRUTH PERFORMANCE

> [!IMPORTANT]
> **Strict Diagnostic Notice**:
> These metrics reflect **agreement with heuristic weak labels** on the validation partition.
> A model can agree highly with weak labels while merely reproducing the same keyword-rule biases.
> Final performance claims require the completed, human-reviewed golden evaluation set.

## 1. Agreement with Weak Labels by Confidence Stratum

| Weak-Label Confidence Stratum | Total Rows | Agreement Count | Agreement with Weak Labels |
| :--- | :--- | :--- | :--- |
| **High-Confidence Weak Labels** | 1,529 | 1,398 | **91.43%** |
| **Medium-Confidence Weak Labels** | 1,546 | 1,467 | **94.89%** |
| **Low-Confidence Weak Labels** | 7,282 | 5,890 | **80.88%** |
| **All Validation Rows (Overall)** | 10,357 | 8,755 | **84.53%** |

## 2. Low-Confidence Predictions & Human Review Rate

- **Total Inquiries Evaluated**: 10,357
- **Confidence Threshold for Review**: < 0.60
- **Inquiries Flagged for Human Review (`needs_human_review = True`)**: 5,632 (54.38%)

> [!NOTE]
> *Predicted confidence is a model probability estimate from a weak-label prototype. It is not yet calibrated against human ground truth.*
> *Flagging for human review is an upstream triage signal and does not yet trigger automated escalation decisions.*

## 3. Distribution Comparison: Heuristic Weak Labels vs. TF-IDF Predictions

| Intent Class | Validation Weak Count | % Weak Split | TF-IDF Predicted Count | % Predicted |
| :--- | :--- | :--- | :--- | :--- |
| `software_update_or_os_issue` | 2,096 | 20.24% | 2,511 | 24.24% |
| `device_performance_or_hardware` | 241 | 2.33% | 549 | 5.30% |
| `connectivity_and_network` | 355 | 3.43% | 475 | 4.59% |
| `apps_services_or_icloud` | 439 | 4.24% | 444 | 4.29% |
| `account_access_and_apple_id` | 83 | 0.80% | 207 | 2.00% |
| `billing_subscription_or_purchase` | 71 | 0.69% | 123 | 1.19% |
| `repair_replacement_or_order` | 47 | 0.45% | 103 | 0.99% |
| `other_or_unclear` | 7,025 | 67.83% | 5,945 | 57.40% |

> [!WARNING]
> **Distribution Shift Limitation**:
> Because training uses class balancing and a controlled `other_or_unclear` sample, predicted class frequencies are not expected to represent real AppleSupport traffic frequencies.

## 4. Weak-Label Confusion Matrix

| True Weak Label \ Predicted      | software_update_or_os_issue | device_performance_or_hardware | connectivity_and_network | apps_services_or_icloud | account_access_and_apple_id | billing_subscription_or_purchase | repair_replacement_or_order | other_or_unclear |
| -------------------------------- | --------------------------- | ------------------------------ | ------------------------ | ----------------------- | --------------------------- | -------------------------------- | --------------------------- | ---------------- |
| software_update_or_os_issue      | 1940                        | 55                             | 39                       | 21                      | 4                           | 2                                | 0                           | 35               |
| device_performance_or_hardware   | 4                           | 225                            | 0                        | 2                       | 0                           | 0                                | 1                           | 9                |
| connectivity_and_network         | 10                          | 1                              | 340                      | 1                       | 0                           | 1                                | 0                           | 2                |
| apps_services_or_icloud          | 13                          | 2                              | 6                        | 267                     | 35                          | 18                               | 0                           | 98               |
| account_access_and_apple_id      | 1                           | 0                              | 1                        | 1                       | 75                          | 0                                | 3                           | 2                |
| billing_subscription_or_purchase | 2                           | 1                              | 0                        | 1                       | 3                           | 64                               | 0                           | 0                |
| repair_replacement_or_order      | 0                           | 0                              | 0                        | 0                       | 0                           | 0                                | 46                          | 1                |
| other_or_unclear                 | 541                         | 265                            | 89                       | 151                     | 90                          | 38                               | 53                          | 5798             |


## 5. De-Identified Prediction Audits

### Five High-Confidence Predictions (`predicted_confidence >= 0.85`)

1. **Tweet `236400`**: "iPhone 6 iOS 11.0.2!!!"
   - **Weak Intent**: `software_update_or_os_issue` (Conf: `medium`)
   - **Predicted**: `software_update_or_os_issue` (Model Conf: `1.0000`)
   - **Top 3**: software_update_or_os_issue; connectivity_and_network; other_or_unclear | **Probs**: 1.0000; 0.0000; 0.0000

2. **Tweet `746865`**: "iPhone 7+ iOS 11.0.2"
   - **Weak Intent**: `software_update_or_os_issue` (Conf: `medium`)
   - **Predicted**: `software_update_or_os_issue` (Model Conf: `1.0000`)
   - **Top 3**: software_update_or_os_issue; connectivity_and_network; other_or_unclear | **Probs**: 1.0000; 0.0000; 0.0000

3. **Tweet `242819`**: "iPhone 7 & iOS 11.0.2"
   - **Weak Intent**: `software_update_or_os_issue` (Conf: `medium`)
   - **Predicted**: `software_update_or_os_issue` (Model Conf: `1.0000`)
   - **Top 3**: software_update_or_os_issue; connectivity_and_network; other_or_unclear | **Probs**: 1.0000; 0.0000; 0.0000

4. **Tweet `245342`**: "iPhone 7 iOS 11"
   - **Weak Intent**: `software_update_or_os_issue` (Conf: `medium`)
   - **Predicted**: `software_update_or_os_issue` (Model Conf: `1.0000`)
   - **Top 3**: software_update_or_os_issue; connectivity_and_network; other_or_unclear | **Probs**: 1.0000; 0.0000; 0.0000

5. **Tweet `1961559`**: "iPhone 7splus iOS 11.0.3"
   - **Weak Intent**: `software_update_or_os_issue` (Conf: `medium`)
   - **Predicted**: `software_update_or_os_issue` (Model Conf: `1.0000`)
   - **Top 3**: software_update_or_os_issue; connectivity_and_network; billing_subscription_or_purchase | **Probs**: 1.0000; 0.0000; 0.0000

### Five Low-Confidence Predictions (`needs_human_review = True`)

1. **Tweet `1879880`**: "My @115858 watch has been leaving marks and burning me for 6 months! Spoke with someone named heyray at #apple and he told me to send him pictures and videos and if it happens again to let him. I followed up via email 2 weeks after... 6 months later and still nothing"
   - **Weak Intent**: `other_or_unclear` (Conf: `low`)
   - **Predicted**: `billing_subscription_or_purchase` (Model Conf: `0.1745`)
   - **Top 3**: billing_subscription_or_purchase; other_or_unclear; account_access_and_apple_id | **Probs**: 0.1745; 0.1730; 0.1646

2. **Tweet `2869715`**: "Once 2nd call connected, tapped green bar at top... nothing. Click home and tap green bar, took me back to contact screen. Had 2nd call hang up on me bc I couldn’t hang up. 1st call was still on hold and no way for me to switch back or hang up. Had to reboot. (2/2)"
   - **Weak Intent**: `other_or_unclear` (Conf: `low`)
   - **Predicted**: `repair_replacement_or_order` (Model Conf: `0.1778`)
   - **Top 3**: repair_replacement_or_order; other_or_unclear; apps_services_or_icloud | **Probs**: 0.1778; 0.1719; 0.1487

3. **Tweet `118103`**: "I have a question, if I were to get a new computer, how would I transfer my songs on iTunes on my old computer to iTunes on my new computer? Asking because I may get a new computer"
   - **Weak Intent**: `apps_services_or_icloud` (Conf: `low`)
   - **Predicted**: `other_or_unclear` (Model Conf: `0.1782`)
   - **Top 3**: other_or_unclear; software_update_or_os_issue; billing_subscription_or_purchase | **Probs**: 0.1782; 0.1700; 0.1475

4. **Tweet `2799432`**: "My phone has stopped alerting me to any notifications that I get? At first I thought it was just only some apps or because my “do not disturb” feature was on, but now it’s constant? @AppleSupport"
   - **Weak Intent**: `other_or_unclear` (Conf: `low`)
   - **Predicted**: `apps_services_or_icloud` (Model Conf: `0.1843`)
   - **Top 3**: apps_services_or_icloud; other_or_unclear; device_performance_or_hardware | **Probs**: 0.1843; 0.1835; 0.1734

5. **Tweet `2747245`**: "Could you possibly give me some information on the iPhone X and 8? I️ really need a new phone and was thinking one of them would be nice."
   - **Weak Intent**: `other_or_unclear` (Conf: `low`)
   - **Predicted**: `other_or_unclear` (Model Conf: `0.1862`)
   - **Top 3**: other_or_unclear; software_update_or_os_issue; repair_replacement_or_order | **Probs**: 0.1862; 0.1579; 0.1343

---
*Diagnostic report generated automatically by scripts/run_validation_diagnostics.py.*