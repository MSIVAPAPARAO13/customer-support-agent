# Golden Evaluation Set Protocol & Governance (Phase 4)

This protocol establishes the scientific methodology, data governance rules, and quality assurance framework for the 200-row human golden evaluation set.

---

## 1. Objective of the Golden Evaluation Benchmark

In conversational AI and NLP systems, automated evaluation metrics (e.g. perplexity, BLEU, rough embedding cosine similarity) can be deceptive.
The **Golden Set** provides an uncontaminated, human-audited ground truth against which:
1. Intent classification precision, recall, and macro-F1 will be measured.
2. Escalation routing safety (identifying security and billing crises) will be verified.
3. Downstream automated response generation will be benchmarked.

---

## 2. Strict Dataset Isolation Rules

> [!CAUTION]
> **Cardinal Rule of ML Evaluation**:
> - The 200 golden evaluation rows are sampled **exclusively from `applesupport_test.csv`**.
> - **Zero Leakage Mandate**: These rows must **NEVER** be included in `applesupport_train.csv`, `applesupport_validation.csv`, fine-tuning datasets, prompt few-shot libraries, or vector retrieval indices.
> - Inclusion of test rows in a retrieval index or training pipeline constitutes **Evaluation Contamination**, rendering all performance claims invalid.

---

## 3. Double-Blind Human Annotation Architecture

To ensure objectivity and eliminate cognitive anchoring:

```text
               Master Test Partition (applesupport_test.csv)
                                     │
                 [Deterministic Sampling: Seed 42]
                                     │
                         golden_set_200.csv (200 rows)
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
  annotator_a_blind.csv                               annotator_b_blind.csv
  (Randomized Seed 101)                               (Randomized Seed 202)
           │                                                   │
  [Human Annotator A]                                 [Human Annotator B]
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
                      scripts/reconcile_golden_labels.py
                                     │
              ┌──────────────────────┴──────────────────────┐
              ▼                                             ▼
  Inter-Annotator Agreement                      adjudication_sheet.csv
  (Raw % & Cohen's Kappa)                    (Human Adjudication of Disagreements)
```

### Omission of Confounding Context
Human annotators receive **blind annotation sheets**:
1. **No Historical Apple Replies**: Knowing how an Apple agent responded biases human annotators to label the agent's interpretation rather than the customer's actual input.
2. **No Model Predictions or Weak Buckets**: Omitting algorithmic hints eliminates automation bias.
3. **Different Presentation Order**: Annotator A (seed 101) and Annotator B (seed 202) inspect the 200 messages in distinct sequences, eliminating order effects.

---

## 4. Inter-Annotator Agreement & Reconciliation

Upon completion of both blind sheets, `scripts/reconcile_golden_labels.py` evaluates consistency using **Cohen's Kappa ($\kappa$)**:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

- **$p_o$ (Observed Agreement)**: Proportion of items where both annotators assigned the identical label.
- **$p_e$ (Hypothetical Chance Agreement)**: Probability of agreement occurring by random chance based on each annotator's marginal class distribution.

### Kappa Interpretation Standard (Landis & Koch, 1977)
- **$\kappa \ge 0.80$**: Almost Perfect Agreement.
- **$0.60 \le \kappa < 0.80$**: Substantial Agreement (acceptable for support intent taxonomies).
- **$0.40 \le \kappa < 0.60$**: Moderate Agreement (indicates ambiguous boundary guidelines requiring revision).
- **$\kappa < 0.40$**: Poor Agreement (taxonomy failure; redesign required).

---

## 5. Adjudication Workflow

1. Where Annotator A and Annotator B **agree**, the agreed label automatically becomes the proposed consensus.
2. Where annotators **disagree** on intent or action:
   - The row is populated into `data/golden/adjudication_sheet.csv`.
   - A third human adjudicator reviews the message and boundary guidelines, recording `final_intent`, `final_action`, and `adjudication_notes`.
3. All fields remain blank until human annotators and adjudicators physically record their decisions.
