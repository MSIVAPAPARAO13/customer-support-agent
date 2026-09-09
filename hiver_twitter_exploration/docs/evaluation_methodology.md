# Final Evaluation Methodology & Verification Protocol (Phase 10)

## Overview & Evaluation Philosophy

This document defines the formal evaluation methodology, statistical metrics, isolation safeguards, and validation protocols for the offline Twitter AppleSupport assistant.

Evaluation is conducted against the protected, human-annotated 200-row golden benchmark (`data/golden/adjudication_sheet.csv`).

---

## 1. Critical Golden-Label Freeze Rule

To prevent cognitive leakage, p-hacking, or intentional/unintentional bias, **model inference on golden messages is strictly prohibited until human adjudication is complete and frozen**.

### Required Workflow Order:
```text
1. Complete Annotator A and Annotator B blind labels (data/golden/annotator_a_blind.csv, data/golden/annotator_b_blind.csv)
2. Reconcile disagreements (scripts/reconcile_golden_labels.py)
3. Fill final adjudication labels (data/golden/adjudication_sheet.csv)
4. Set all 200 final_status values to DONE
5. Freeze the final label file (scripts/freeze_golden_labels.py -> data/golden/golden_labels_freeze_manifest.json)
6. Run blind golden inference (scripts/run_golden_agent_inference.py)
7. Run final metrics (scripts/run_final_evaluation.py)
8. Run LLM judge and human-judge agreement (scripts/run_llm_judge.py, scripts/evaluate_judge_human_agreement.py)
```

### Freeze Manifest Specifications (`scripts/freeze_golden_labels.py`):
- Requires exactly 200 rows with `final_status == 'DONE'`.
- Validates that all `final_intent` categories belong to the approved 8-class taxonomy.
- Validates that all `final_action` decisions are either `auto_handle` or `escalate`.
- Computes SHA-256 cryptographic digest of `data/golden/adjudication_sheet.csv`.
- Writes `data/golden/golden_labels_freeze_manifest.json` containing:
  - `adjudication_file`
  - `adjudication_sha256`
  - `freeze_timestamp`
  - `golden_row_count`
  - `final_status_summary`
- Refuses to overwrite an existing manifest unless `--force` is explicitly provided.

### Golden-Inference Anti-Tamper Guard (`scripts/run_golden_agent_inference.py`):
Inference automatically verifies:
- `golden_labels_freeze_manifest.json` exists.
- Manifest row count equals 200.
- Current `adjudication_sheet.csv` SHA-256 checksum matches the manifest exactly.
- All 200 rows have `final_status == 'DONE'`.

If any condition fails, execution halts immediately with exit code 1.

---

## 2. Intent Classification Systems Evaluated

The evaluation benchmarks three distinct intent architectures against identical human ground truth:

1. **`majority_weak_intent` (Minimal Performance Floor)**:
   - Always predicts the training weak-label distribution mode (`device_performance_or_hardware`).
   - Represents the naive statistical floor that any viable learning system must outperform.
2. **`keyword_rule_classifier` (Domain Heuristic Benchmark)**:
   - Evaluates incoming text using deterministic regex patterns and domain keywords.
   - Represents rule-based engineering without statistical representation learning.
3. **`tfidf_logistic_regression` (Main System)**:
   - Sublinear TF-IDF vectorizer (1-2 ngrams, 30,000 max features) + class-balanced Logistic Regression trained on weak training labels.

### Macro-F1 Invariant Rule (Fixed 8 Taxonomy Classes)
All macro-averaged F1 calculations **strictly use the complete fixed list of eight taxonomy labels**:
```text
1. software_update_or_os_issue
2. device_performance_or_hardware
3. connectivity_and_network
4. apps_services_or_icloud
5. account_access_and_apple_id
6. billing_subscription_or_purchase
7. repair_replacement_or_order
8. other_or_unclear
```
> **Methodological Rule**: Under no circumstances may macro-F1 calculations dynamically drop rare or unobserved classes from the denominator. Bootstrapping must evaluate all eight classes to prevent artificial score inflation.

---

## 3. End-to-End Operational Routing Metrics

Operational decision routing (`auto_handle` vs. `escalate`) is evaluated against human adjudications:

| Metric | Definition | Formula |
| :--- | :--- | :--- |
| **Action Accuracy** | Overall decision accuracy | $\frac{\text{Count}(\text{Agent Action} = \text{Human Action})}{200}$ |
| **Auto-Handle Coverage** | Fraction of inquiries candidate for automated handling | $\frac{\text{Count}(\text{Agent Action} = \text{auto\_handle})}{200}$ |
| **Selective Accuracy** | Decision accuracy strictly on the auto-handled cohort | $\frac{\text{Count}(\text{Agent Action} = \text{Human Action} \land \text{Agent Action} = \text{auto\_handle})}{\text{Count}(\text{Agent Action} = \text{auto\_handle})}$ |
| **Unsafe Auto-Handle Count** | Model auto-handled when human required escalation | $\text{Count}(\text{Agent Action} = \text{auto\_handle} \land \text{Human Action} = \text{escalate})$ |
| **Unsafe Auto-Handle Rate** | Proportion of auto-handled cases that are unsafe | $\frac{\text{Unsafe Auto-Handle Count}}{\text{Count}(\text{Agent Action} = \text{auto\_handle})}$ |
| **Escalation Precision** | Precision for `escalate` (positive class) | $\frac{\text{Count}(\text{Agent Action} = \text{escalate} \land \text{Human Action} = \text{escalate})}{\text{Count}(\text{Agent Action} = \text{escalate})}$ |
| **Escalation Recall** | Recall for `escalate` (positive class) | $\frac{\text{Count}(\text{Agent Action} = \text{escalate} \land \text{Human Action} = \text{escalate})}{\text{Count}(\text{Human Action} = \text{escalate})}$ |

### Edge-Case Zero-Automation Handling
If the agent predicts zero auto-handle actions across the evaluation set:
- `selective_accuracy = N/A` (Undefined fraction)
- `unsafe_auto_handle_rate = N/A` (Undefined fraction)
- **Do not report them as zero.** Reporting 0% would falsely imply zero accuracy or zero risk, whereas the metric is mathematically undefined.

---

## 4. Required Warning on Selective Accuracy & Coverage

> [!WARNING]
> **A conservative system can improve selective accuracy simply by escalating more messages. Automation coverage must always be reported alongside auto-handle quality.**

### Operational Behaviour Interpretation:
- A low auto-handle coverage (e.g. 2.3%) is a **routing-policy behaviour**, NOT independent evidence of accuracy, safety, or deployment readiness.
- An agent can achieve 100% selective accuracy simply by auto-handling a single trivially safe message and escalating the remaining 199. High selective accuracy on an extremely narrow cohort is an artifact of conservative risk gating.

---

## 5. Bootstrap Confidence Intervals

Statistical variability is quantified using non-parametric bootstrap resampling:
- **Resamples**: 1,000 iterations
- **Random Seed**: `42`
- **Reported Interval**: 95% Confidence Interval ($2.5^{\text{th}}$ to $97.5^{\text{th}}$ percentiles)
- **Covered Metrics**:
  - Main Intent Accuracy
  - Main Intent Macro F1 (computed across all 8 fixed labels)
  - Action Accuracy
  - Auto-Handle Coverage
  - Selective Accuracy (computed across resamples where auto-handle count > 0)

---

## 6. Strict Data Isolation Invariants

```text
TRAINING PARTITION ONLY:
  - Vectorizer fitting (Phase 6 TF-IDF, Phase 7 Retrieval)
  - Classifier training (Logistic Regression)
  - Retrieval index building (Historical customer dialogues & AppleSupport replies)
  - Confidence and similarity threshold selection

GOLDEN EVALUATION PARTITION:
  - 200 customer messages used strictly as unseen query inputs
  - Never indexed in the retrieval database
  - Never used to fit vectorizers or adjust thresholds
  - Evaluated only after labels are complete and frozen
```

---

## 7. Blinded LLM-as-a-Judge & Human Agreement

To provide qualitative auditing of draft replies and routing decisions, a blinded LLM-as-a-judge system is paired with double-blind human review:

- **Blinding**: Judge sees only customer inquiry, agent action/reasons, draft reply, and sanitized historical snippets. Judge NEVER sees human labels, classifier confidence, or weak labels.
- **Dimensions**: Evaluated across 5 Likert dimensions (1-5), `overall_accept` (`ACCEPT`/`REJECT`), reason codes, and short explanation.
- **Sample**: 40-row stratified human review sheet (`human_judge_agreement_sheet.csv`) initialized with blank scores and `human_review_status = 'NEEDS_HUMAN_REVIEW'`.
- **Inter-Rater Reliability**: Calculated using exact agreement rate, quadratic/linear weighted Cohen's Kappa ($\kappa$) for 1-5 scales, and standard Cohen's Kappa for binary acceptance.
- **Completion Rule**: Agreement is reported only after all 40 human reviews are completed.
