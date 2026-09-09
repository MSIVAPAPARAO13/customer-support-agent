# Intent Classification Baseline Methodology & Integrity Guide

## 1. Introduction: What is a Baseline in Machine Learning?

In applied machine learning and Natural Language Processing (NLP), a **baseline** is the simplest possible benchmark model or decision rule created before training sophisticated algorithms. Baselines serve as the **performance floor** for a project.

Without baselines, evaluation metrics are meaningless. For instance, if an advanced neural network achieves **72% accuracy**, is that impressive or terrible?
- If the dataset has 10 classes evenly balanced (10% each), 72% is remarkably good.
- If one dominant class represents **70%** of all data, a trivial model that ignores the text entirely and always predicts that dominant class achieves **70% accuracy**. In that case, a 72% neural network is barely outperforming random chance!

In Phase 5 of our AppleSupport dialogue system, we construct two transparent baseline models:
1. **Baseline 1 (`majority_weak_intent`)**: A trivial, text-agnostic mode predictor.
2. **Baseline 2 (`keyword_rule_classifier`)**: A deterministic keyword-heuristic classifier.

---

## 2. Baseline 1: The Majority-Class Classifier (`majority_weak_intent`)

### How It Works
1. Inspects the weak-labelled training split (`applesupport_train_weak_labels.csv`).
2. Calculates the frequency of all eight intent categories.
3. Finds the most common label (the mathematical mode).
4. Predicts this identical label for **every customer message**, regardless of what the customer actually wrote.

### Our Training Data Weak-Label Distribution
In our 82,077-row AppleSupport weak training set:
- **68.36% is the proportion of training rows assigned `other_or_unclear` by the weak-label rules. It is a pseudo-label class distribution, not a real accuracy score or a valid performance floor.**
- **`other_or_unclear` is a fallback bucket created when keyword rules fail. It is not a clean semantic category, and its sampled training examples may contain many different underlying customer issues.**
- Therefore, the majority-intent baseline predicts `other_or_unclear` for every single message based purely on this pseudo-label mode.

### Why This Trivial Strategy is Included
- **Pipeline Sanity Check**: It confirms that data loading, tensor/vector pipelines, and prediction exports function correctly.
- **Exposes Heuristic Imbalance**: It exposes how heavily skewed rule-based pseudo-labels are toward conversational greetings, follow-ups, and uncaptured complaints (`other_or_unclear`).
- **Circularity Reminder**: A model can agree highly with weak labels while merely reproducing the same keyword-rule biases. Final claims require the completed, human-reviewed golden set.

---

## 3. Baseline 2: The Keyword-Rule Classifier (`keyword_rule_classifier`)

### How It Works
The keyword classifier uses domain-informed regular expressions and vocabulary dictionaries to inspect the customer's cleaned text. It executes prioritized pattern matching:
1. **Priority 1 (Security & Account Access)**: Matches explicit security tokens (`Apple ID`, `passcode`, `two-factor`, `hacked`, `phishing`).
2. **Priority 2 (Financial & Purchases)**: Matches monetary transactions (`refund`, `subscription`, `overcharged`, `receipt`).
3. **Priority 3 (Hardware Repair & Logistics)**: Matches service terms (`Genius Bar`, `AppleCare`, `warranty`, `repair cost`, `shipping`).
4. **Priority 4 (Connectivity)**: Matches network hardware (`Wi-Fi`, `Bluetooth`, `cellular`, `LTE`, `AirDrop`).
5. **Priority 5 (Software Updates)**: Matches OS release terms (`iOS 11`, `update`, `upgrade`, `install failed`).
6. **Priority 6 (Apps & Cloud)**: Matches native services (`iCloud`, `App Store`, `Apple Music`, `iMessage`, `FaceTime`).
7. **Priority 7 (Physical Device)**: Matches components (`battery`, `overheating`, `cracked screen`, `charging`).
8. **Fallback**: If no rule triggers, defaults to `other_or_unclear`.

### What Keyword Rules Can Understand (Strengths)
- **Explicit Technical Vocabulary**: Easily catches unmistakable terms like `Genius Bar`, `Bluetooth`, or `AirDrop`.
- **Zero Compute Cost**: Runs instantaneously with minimal memory footprint and zero GPU requirements.
- **100% Explainability**: Every prediction links directly to an audited rule name (e.g., `rule_apple_id_credential_lock`), making debugging transparent.

### What Keyword Rules Cannot Understand (Weaknesses & Blind Spots)
1. **Polysemy & Word-Sense Collisions**: The word *"charge"* can mean electrical energy (*"won't charge"*) or financial cost (*"charged twice"*). While our rules disambiguate this via battery context (`is_battery_or_power_context`), complex sentences confuse simple rules.
2. **Implicit Context & Ellipsis**: In conversational dialogues, customers often omit key nouns:
   - *"I upgraded yesterday and now it won't connect."* (Needs to know whether "it" is Wi-Fi, Bluetooth, or CarPlay).
   - *"Tried resetting settings, restarting, all that."* (No explicit hardware or OS keyword mentioned).
3. **Sarcasm and Sentiment Inversion**:
   - *"Thanks Apple for another flawless update that completely bricked my phone!"*
   - Keyword rules detect *"update"* and classify it into software update, missing the severe hardware failure.
4. **Spelling Variations & Slang**: Tweets contain informal language, typos (*"cant conect"*, *"loosing battery"*), and emojis that rigid regex patterns overlook.

---

## 4. Weak Labels ("Silver Standard") vs. Human Ground Truth ("Gold Standard")

Understanding the distinction between weak labels and ground truth is paramount in real-world ML engineering:

| Dimension | Weak / Silver Labels (`applesupport_train_weak_labels.csv`) | Human Ground Truth (`golden_set_200.csv`) |
| :--- | :--- | :--- |
| **Origin** | Algorithmic heuristics, regex rules, keyword dictionaries. | Two independent human annotators + expert adjudication. |
| **Scale** | Massive (82,077 train rows, 10,357 validation rows). | Targeted, high-quality benchmark (200 protected rows). |
| **Noise Level** | Moderate to high (~15%–30% noise / misclassifications). | Near zero; validated against standardized annotation guidelines. |
| **Edge Cases** | Struggles with sarcasm, multi-intent queries, and indirect speech. | Explicitly adjudicates multi-intent tradeoffs and edge cases. |
| **Appropriate Use** | **Training and pre-training experiments ONLY**. | **Final benchmarking, headline accuracy, and deployment gating**. |

---

## 5. Why Evaluating Against Weak Labels is Misleading (The Circularity Trap)

It is tempting for junior practitioners to evaluate their machine learning models on the weak-labelled dataset because it is large and immediately available. **This is a severe methodological mistake known as Circular Validation.**

### The Circularity Fallacy
1. If a machine learning model is trained on labels produced by keyword rules,
2. And that model is evaluated against those same keyword-generated labels,
3. A model that perfectly memorizes the keyword dictionary will achieve **100% accuracy**.

However, that 100% score is an illusion! The model did not learn to understand customer support dialogues; it simply learned to replicate the developer's heuristic biases and flaws. If the keyword rule misclassified 3,000 tweets, the model is rewarded for making those exact same 3,000 mistakes.

---

## 6. Why Final Benchmarks Must Strictly Use the Human-Reviewed Golden Set

To avoid circularity, **the human golden evaluation set is strictly isolated**:
1. **Zero Data Leakage**: Golden rows are sampled exclusively from the protected test split and never appear in the training or validation sets.
2. **True Generalization Measure**: Because the golden set is annotated by humans, it evaluates whether an ML model can generalize beyond simple keywords to interpret customer meaning.
3. **Safety Enforcement**: `scripts/evaluate_completed_golden.py` contains hardcoded safety gates that refuse to execute until all 200 human golden adjudications are complete (`final_status == 'DONE'`).

---

## 7. Operational Action & Safety Routing

In customer support automation, predicting the intent is only half the battle. The system must also decide **how to act**:
- `direct_troubleshoot`: Safe for automated bot resolution (e.g. self-service links for Wi-Fi resets).
- `request_device_or_os_details`: Safe gathering of additional context.
- `escalate_to_human_agent`: **Critical human safety handoff** for account lockouts, security breaches, or billing disputes.

### Key Operational Metrics
When golden evaluation is unlocked, the evaluation suite computes:
- **Action Accuracy**: Did the model pick the correct support routing path?
- **Unsafe Auto-Handle Count**: The number of times the model attempted to automate a query that required human escalation. In production, this metric must be as close to zero as possible to prevent security breaches and customer churn.
- **Auto-Handle Coverage**: What percentage of total customer support volume can be safely resolved without human intervention.
- **Selective Accuracy**: The accuracy achieved specifically on the automated cohort.
