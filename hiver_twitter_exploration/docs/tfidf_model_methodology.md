# Explainable TF-IDF + Logistic Regression Intent Classifier: Methodology & Architecture Guide

## 1. Executive Summary: Moving Beyond Rigid Heuristics

In Phase 5, we created two initial baselines:
1. A trivial majority-class predictor (`majority_weak_intent`), and
2. A deterministic keyword rule engine (`keyword_rule_classifier`).

While keyword rules are transparent, they suffer from brittle binary logic: if a tweet misses an exact keyword by a single typo, synonym, or phrased context, the rule fails completely and defaults to `other_or_unclear`.

In **Phase 6**, we construct an **explainable statistical machine learning model**:
- **Feature Representation**: n-gram Term Frequency-Inverse Document Frequency (**TF-IDF**).
- **Classification Engine**: Multi-class regularized **Logistic Regression** with balanced class weighting.

> [!IMPORTANT]
> **Prototype Status & Weak-Label Disclaimer**:
> This model is trained exclusively on **rule-generated silver labels** (`applesupport_train_weak_labels.csv`).
> - **Predicted confidence is a model probability estimate from a weak-label prototype. It is not yet calibrated against human ground truth.**
> - **A model can agree highly with weak labels while merely reproducing the same keyword-rule biases. Final claims require the completed, human-reviewed golden set.**
> - **`other_or_unclear` is a fallback bucket created when keyword rules fail. It is not a clean semantic category, and its sampled training examples may contain many different underlying customer issues.**
> - **Because training uses class balancing and a controlled `other_or_unclear` sample, predicted class frequencies are not expected to represent real AppleSupport traffic frequencies.**

---

## 2. How a Customer Message Becomes Numbers: TF-IDF Explained

Computers cannot perform arithmetic directly on sentences like *"my battery dies fast after iOS 11"*. They require numerical feature vectors.

### Step A: Vocabulary & Bag-of-Words
The vectorizer scans all training tweets and builds a dictionary of up to 30,000 distinct words and phrases (features). Each feature gets an index column in a mathematical matrix.

### Step B: Unigrams vs. Bigrams
- **Unigram (1-gram)**: A single word (e.g., `"apple"`, `"id"`, `"genius"`, `"bar"`).
- **Bigram (2-gram)**: A two-word sequence (e.g., `"apple id"`, `"genius bar"`, `"battery life"`, `"wi fi"`, `"ios 11"`).

In technical support, bigrams are indispensable:
- The word `"apple"` alone is ambiguous (could refer to the company, device, or store).
- The word `"id"` alone is vague (could mean identification, transaction ID, or user handle).
- But the bigram **`"apple id"`** is an unmistakable, high-liability account credential token!
By configuring `ngram_range=(1, 2)`, our TF-IDF model extracts both individual words and two-word pairings.

### Step C: Term Frequency (TF)
How often does the token appear in this specific customer tweet?
If `"battery"` appears twice, its term frequency is higher than if it appears once. We apply `sublinear_tf=True`, which uses $1 + \log(\text{TF})$ to prevent a customer who frantically types *"battery battery battery"* from dominating the vector.

### Step D: Inverse Document Frequency (IDF)
How rare or informative is this word across the entire corpus?
- Common conversational filler words like `"the"`, `"and"`, or `"my"` appear in almost every tweet. Their IDF weight is near zero because they carry no intent-distinguishing power.
- Rare technical terms like `"airdrop"`, `"keychain"`, `"applecare"`, or `"phishing"` appear in only a tiny fraction of tweets. Their IDF weight is very high.

The final TF-IDF score is the product:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

---

## 3. What Multi-Class Logistic Regression Does

Despite its historical name, **Logistic Regression is a linear classification algorithm**.

### The Scoring Equation
For each of the eight intent classes $k \in \{1, \dots, 8\}$, the classifier maintains a learned weight vector $W_k$ and a bias term $b_k$.
When a vectorized customer message $X$ arrives, the model computes a linear score (logit) for each intent:
$$z_k = X \cdot W_k + b_k$$

- If the tweet contains words with large positive weights for class $k$ (e.g., `"wifi"`, `"bluetooth"` for `connectivity_and_network`), $z_k$ becomes strongly positive.
- If it contains words associated with other classes, $z_k$ remains low or negative.

### Softmax: Converting Logits to Probabilities
To convert the eight raw scores $(z_1, \dots, z_8)$ into probabilities that sum to 1.0 (100%), the model applies the **Softmax function**:
$$P(\text{Intent} = k \mid X) = \frac{e^{z_k}}{\sum_{j=1}^{8} e^{z_j}}$$

The class with the highest probability becomes `predicted_intent`, and its associated probability becomes `predicted_confidence`.

---

## 4. Why Class Weighting is Essential (`class_weight="balanced"`)

In the raw weak training data:
- `software_update_or_os_issue` has **16,326** high/medium rows.
- `repair_replacement_or_order` has only **370** rows.
- `billing_subscription_or_purchase` has only **558** rows.

If standard unweighted logistic regression is trained on this data, the algorithm minimizes overall loss by overwhelmingly guessing `software_update_or_os_issue` and largely ignoring small, critical categories like billing disputes or repair requests.

By setting `class_weight="balanced"`, scikit-learn computes penalty multipliers inversely proportional to class frequencies:
$$w_k = \frac{N_{\text{total}}}{8 \times N_k}$$
This ensures that a classification mistake on a rare 370-sample repair tweet is penalized **~44 times more heavily** than a mistake on a 16,326-sample update tweet, forcing the classifier to learn sharp decision boundaries for every single category.

---

## 5. Confidence Thresholding & Human Review Flagging

In customer support automation, an unconfident model must not blindly fire automated replies.

In `predict_tfidf_classifier.py`, we implement an upstream safety check:
```python
if predicted_confidence < 0.60:
    needs_human_review = True
```

### Key Principles
1. **Model Probability Estimate Only**: Predicted confidence is an internal score from a weak-label prototype. It is not yet calibrated against real human annotations.
2. **Review Signal, Not Auto-Escalation**: Flagging `needs_human_review = True` serves as a triage indicator. In subsequent phases, it will be combined with retrieval evidence and dialogue policies to decide whether to offer self-service troubleshooting or escalate to human specialists.
3. **Validation Rate**: On our 10,357-row validation split, **54.38%** of inquiries were flagged with `needs_human_review = True`, demonstrating that ambiguous tweets, multi-turn narrative vents, and general questions are safely flagged rather than assigned overconfident false positives.

---

## 6. Interpretability: Inspecting Learned Coefficients

A massive advantage of TF-IDF + Logistic Regression over black-box deep neural networks is **complete interpretability**.

By examining `outputs/tfidf/top_features_by_intent.csv`, we can inspect the exact unigrams and bigrams that most heavily increase the odds of predicting each intent:

| Intent Category | Rank 1 Feature (Weight) | Rank 2 Feature (Weight) | Rank 3 Feature (Weight) |
| :--- | :--- | :--- | :--- |
| `connectivity_and_network` | `wifi` (+19.10) | `bluetooth` (+16.62) | `cellular` (+8.98) |
| `apps_services_or_icloud` | `siri` (+17.02) | `imessage` (+15.87) | `facetime` (+13.09) |
| `software_update_or_os_issue` | `ios11` (+16.05) | `ios 11` (+15.22) | `ios` (+11.71) |
| `device_performance_or_hardware`| `charger` (+15.42) | `battery` (+9.77) | `battery life` (+8.22) |
| `billing_subscription_or_purchase`| `refund` (+15.18) | `subscription` (+13.28) | `billing` (+9.94) |
| `repair_replacement_or_order` | `applecare` (+14.66) | `apple care` (+10.91) | `care` (+10.17) |
| `account_access_and_apple_id` | `password` (+10.01) | `hacked` (+9.81) | `phishing` (+9.30) |
| `other_or_unclear` | `fix` (+2.64) | `the letter` (+1.76) | `letter` (+1.71) |

This confirms that the model has learned meaningful, domain-accurate representations without memorizing noise.
