# Historical Dialogue Retrieval: Methodology & Grounding Guide

## 1. Executive Summary: Why Dialogue Retrieval Matters

In customer support automation, generating responses using ungrounded language models often produces **hallucinations**—invented refund policies, fictional technical advice, or promised compensation that violates company protocols.

To prevent this, **Phase 7** introduces an **in-memory dialogue retrieval engine**:
- Instead of generating text from scratch, the system first searches real historical AppleSupport interactions where a customer asked a similar question and an official support agent answered.
- These historical customer $\rightarrow$ brand reply pairs provide **empirical precedent** for downstream response drafting.

```text
Incoming customer message
        ↓
Search train-only historical AppleSupport messages
        ↓
Retrieve similar inquiry + actual historical AppleSupport reply
        ↓
Evidence for the future reply drafter
        ↓
Low similarity = weaker evidence = later escalation signal
```

> [!IMPORTANT]
> **Required Epistemic Disclaimer on Grounding & Accuracy**:
> **Retrieval has been verified as train-only historical search. It has not yet been proven that retrieved reply patterns are relevant, safe, or beneficial. The 30-row human review sheet is required before making that claim.**
>
> **A high TF-IDF cosine score means the new message and historical message share important words or phrases. It does not prove the historical reply is relevant, factually applicable, safe, or appropriate for the new customer.**
>
> A vague complaint with nearly identical profanity can receive similarity `1.0` but still provide poor grounding.
> **High similarity reflects stronger lexical similarity only.** High lexical similarity (`similarity >= 0.70`) does **NOT** mean safe grounding or safe auto-handling.

---

## 2. Core Concepts Explained in Beginner-Friendly Language

### What is a Vector?
A vector is an ordered list of numbers representing a text document in high-dimensional mathematical space. With a vocabulary of 50,000 distinct words and phrases (features), every customer message is represented as a point in 50,000-dimensional space, where each coordinate reflects the prominence of a word or bigram.

### What is TF-IDF Retrieval?
**Term Frequency-Inverse Document Frequency (TF-IDF)** converts text into numbers by balancing two forces:
1. **Term Frequency (TF)**: Words repeated within a customer inquiry receive higher weight.
2. **Inverse Document Frequency (IDF)**: Common conversational filler words (`"the"`, `"help"`, `"please"`) receive near-zero weight, whereas distinctive technical tokens (`"airdrop"`, `"itunes billing"`, `"applecare"`, `"overheating"`) receive large weights.

When a new customer inquiry arrives, it is transformed using the exact vocabulary and IDF weights learned on historical training tweets.

### What is Cosine Similarity?
Cosine similarity measures the angle between two multi-dimensional vectors, regardless of sentence length:
$$\text{Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$
- **Score = 1.0**: The two customer messages share identical vocabulary in identical proportions.
- **Score = 0.0**: The two messages share zero vocabulary in common.

Because our TF-IDF vectorizer explicitly enforces L2-normalization (`norm="l2"`), the dot product of two vectors is **mathematically identical to cosine similarity**.

---

## 3. Why High Similarity Does Not Guarantee Identical Meaning

A cosine similarity score near 1.0 indicates **stronger lexical similarity only**. It does **NOT** guarantee semantic or factual equivalence:
- Query A: *"Can I cancel my Apple Music subscription without losing my playlists?"*
- Query B: *"I cancelled my Apple Music subscription and lost all my playlists!"*
- Both queries share almost identical vocabulary (`"cancel"`, `"apple music"`, `"subscription"`, `"playlists"`), resulting in a high similarity score ($> 0.85$).
- However, Query A is a preventative inquiry asking for safety steps, while Query B is a post-incident data loss dispute requiring data recovery!

Similarly, two customers venting with identical profanity and vague frustration (*"apple sucks your product is broken fix this"*) will match with similarity near 1.0, yet the historical agent reply may have been tailored to a hardware defect that is completely unrelated to the new customer's software issue.

---

## 4. The Real-World Evidence Interpretation Example

Consider this realistic support interaction:

```text
New customer:
“My iPhone battery drains after iOS 11.”

Historical customer (Retrieved):
“My battery life became terrible after updating to iOS 11.”

Historical AppleSupport reply:
“Please tell us your device model and iOS version through secure support.”
```

### Correct Interpretation:
This retrieved interaction is **evidence that AppleSupport historically requested diagnostic details** (device model and iOS sub-version) to isolate the regression.

### Incorrect Interpretation:
Assuming the historical reply is proof that the new customer has the same device, the same iOS version, or the same resolution. Historical replies are **evidence of past communication patterns**, never factual proof for the current customer.

---

## 5. Unique Customer Inquiry Indexing (Design Improvement)

In raw dialogue archives, a single customer inquiry can occasionally receive multiple replies from brand agents (e.g., split tweets or follow-ups). In our training set, **14 customer tweets received multiple AppleSupport replies**.

To ensure that the same customer inquiry does not occupy multiple ranks in the top-k results:
- **One searchable document = one unique `customer_tweet_id`** (82,063 unique documents indexed).
- All unique cleaned brand replies are aggregated into structured fields (`brand_reply_clean`, `brand_replies_list`).
- All associated brand tweet IDs are preserved (`brand_tweet_id`, `brand_tweet_ids_list`).
- Top-k results are guaranteed to contain distinct historical customer inquiries.

---

## 6. Conservative Evidence Threshold (0.30) & Lexical Similarity Bands

In Phase 7, the default evidence threshold is calibrated to a conservative **`0.30`** (up from 0.15). Every retrieval result is categorized into a standardized lexical similarity band:

| Lexical Similarity Band | Score Range | Operational Meaning | Safety Directive |
| :--- | :--- | :--- | :--- |
| `high_lexical_similarity` | $\ge 0.70$ | Stronger lexical similarity only | Relevance unverified; needs human review |
| `moderate_lexical_evidence` | $0.50 \le \text{sim} < 0.70$ | Substantial vocabulary overlap | Reference pattern only |
| `weak_lexical_evidence` | $0.30 \le \text{sim} < 0.50$ | Partial vocabulary overlap | Candidate for human review |
| `insufficient_lexical_evidence` | $< 0.30$ | Minimal or no vocabulary match | **Escalation trigger: weak precedent** |

### Unknown Vocabulary Safeguard
If a customer inquiry contains no vocabulary words recognized by the training TF-IDF dictionary:
- `similarity_score = 0.0`
- `lexical_similarity_band = "insufficient_lexical_evidence"`
- `has_sufficient_historical_evidence = False`
- `retrieval_warning = "no_known_vocabulary"`

---

## 7. Cross-Conversation Repeated Wording vs. Data Leakage

In `outputs/retrieval/train_validation_text_overlap_report.md`, we analyze exact normalized customer text matches across partitions:
- **66 out of 10,357 validation inquiries (0.64%)** share exact normalized wording with historical training inquiries.
- This is **NOT data leakage**; all conversation trees are strictly isolated by `conversation_root_or_group_id` and `customer_tweet_id`.
- It occurs because real users independently post identical short phrases (*"iphone 7 ios 11"*, *"what is the new update?"*).
- However, our strict diagnostic confirms that when identical wording candidates are excluded, retrieval similarity drops, reminding us that offline retrieval on common phrasing must not be conflated with performance on novel customer language.

---

## 8. Evidence Sanitization

Before historical pairs are output as evidence for future response drafting:
- All raw Twitter handles (`@AppleSupport`, `@115858`, user handles) are replaced with `[USER]`.
- All URLs are standardized as `[URL]`.
- Only cleaned, sanitized text is used; raw tweets are never passed into future prompts.
