# Phase 8: Safe, Historically Grounded Reply Drafting Methodology

## Executive Summary

Phase 8 builds the **reply-drafting component** of the AppleSupport inquiry pipeline. It bridges intent classification (Phase 6) and dialogue retrieval (Phase 7) with deterministic safety constraints to produce cautious, verifiable draft replies.

```text
Customer message
+ predicted intent and confidence (Phase 6 TF-IDF classifier)
+ retrieved historical AppleSupport response patterns (Phase 7 train-only index)
+ deterministic safety restrictions (Phase 8 reply_safety)
        ↓
Safe draft reply
+ visible evidence tweet IDs
+ safety flags and grounding notes
```

> [!IMPORTANT]
> **Fundamental Safety Wording**:
> **Constrained templates reduce unsupported claims and make drafts easier to audit. They do not guarantee that every draft is relevant or safe.**
> 
> Furthermore:
> **A high TF-IDF cosine score means the new message and historical message share important words or phrases. It does not prove the historical reply is relevant, factually applicable, safe, or appropriate for the new customer.**

---

## 1. Why Constrained Templates Make the System Safer

In production customer support, deploying unconstrained generative models directly to customer-facing channels poses critical risks:
- **Hallucination of Policy & Commitments**: Models can fabricate promises (e.g., *"Your refund has been approved"* or *"We will replace your iPhone for free"*), creating legal and financial liability.
- **Unauthorized Data Collection**: Models may prompt customers for sensitive credentials (passwords, two-factor codes, credit card numbers, PINs).
- **Inconsistent Support Paths**: Generative models may offer conflicting technical advice across similar interactions.

By using **constrained, intent-conditioned templates**:
1. Every word and troubleshooting suggestion is human-reviewed and bounded.
2. The system enforces strict privacy guards: it never requests sensitive personal data.
3. Every draft can be deterministically reproduced and audited.
4. It sets an auditable baseline against which future LLM generators or policies can be objectively measured.

---

## 2. Grounding in Support Protocols, Not Customer Facts

A common pitfall in retrieval-augmented generation (RAG) is assuming that a retrieved past reply represents ground-truth facts about the *new* customer. In AppleSupport interactions:
- A past customer had a specific device (e.g., iPhone 7 running iOS 11.1).
- Their issue had a specific resolution (e.g., an iOS 11.1.1 patch).
- The *new* customer may be describing a superficially similar symptom on an entirely different device or OS version.

### Safe Influence:
Retrieved historical replies are allowed to influence **general support protocol patterns only**, such as:
- *“Have your device model and software version ready when contacting Apple Support through a secure channel.”*
- *“As a general initial step, restarting your device can help resolve temporary issues.”*

### Prohibited Claims:
The drafting engine is strictly prohibited from inferring or stating:
- ❌ *“This update will fix the issue.”*
- ❌ *“Your battery needs replacement.”*
- ❌ *“Your refund is approved.”*
- ❌ *“Your device is eligible for repair.”*
- ❌ *“We can access your account.”*

> [!NOTE]
> No matches to the defined prohibited phrases were found by the automated regex audit. Regex checks can detect selected risky phrases, but they cannot prove every generated draft is relevant, accurate, or safe. Human review remains necessary.

Historical evidence is recorded with provenance:
- `evidence_customer_tweet_ids`
- `evidence_brand_tweet_ids`
- `best_similarity_score`
- `lexical_similarity_band`
- `grounding_note`

All historical evidence is sanitized: URLs $\rightarrow$ `[URL]`, Twitter handles $\rightarrow$ `[USER]`.

---

## 3. Strict Thresholds and Draft Modes

To prevent weak retrieval signals from injecting misleading technical suggestions, Phase 8 adopts a conservative multi-tiered threshold:

```text
restricted_draft = True
→ draft_mode = restricted_safety

restricted_draft = False and best_similarity_score < 0.50
→ draft_mode = conservative_no_evidence

restricted_draft = False and best_similarity_score >= 0.50
→ draft_mode = template_with_historical_pattern
```

### Why 0.30–0.49 Does Not Adapt Drafts:
A score between `0.30` and `0.49` represents **weak lexical evidence**. While it satisfies basic retrieval relevance for audit tracking (`has_sufficient_historical_evidence = True`), it carries too high a risk of lexical false friends (e.g., sharing common words like *"phone"*, *"working"*, *"update"* without matching technical context). Therefore, scores $< 0.50$ are barred from altering draft text and default strictly to `conservative_no_evidence`.

### The Three Standard Draft Modes:

| Draft Mode | Condition | Behavior |
| :--- | :--- | :--- |
| **`restricted_safety`** | Safety flags detected (`restricted_draft = True`) | Overrides all templates with a brief, neutral secure handoff. Zero technical troubleshooting. |
| **`conservative_no_evidence`** | No safety flags, similarity $< 0.50$ | Uses cautious base intent template. Directs to official secure channel without historical pattern adaptation. |
| **`template_with_historical_pattern`** | No safety flags, similarity $\ge 0.50$ | Synthesizes base intent template with the general support pattern observed in historical replies (e.g., preparing OS version). |

---

## 4. Risk-Aware Restrictions: High-Risk Domains

Customer support on public social media (Twitter / X) encounters severe edge cases requiring immediate policy intervention. `src/reply_safety.py` scans for trigger keywords across six high-risk domains:

1. **Account Compromise**: Mentions of hacking, unauthorized account access, or security breaches.
2. **Password & Credential Recovery**: Requests for password resets, two-factor authentication codes, security questions, or activation locks.
3. **Unauthorized Payment & Fraud**: Reports of unauthorized credit card charges, scams, or billing fraud.
4. **Refund & Payment Disputes**: Demands for refunds, chargebacks, or transaction disputes.
5. **Legal Threats & Abuse**: Demands to sue, threats of litigation, contacting attorneys, or regulatory complaints.
6. **Self-Harm & Immediate Danger**: Language indicating self-harm, suicide, or personal crisis.

### Policy Response:
For any detected safety flag:
- `restricted_draft = True`
- `draft_mode = "restricted_safety"`
- Generates a brief, neutral handoff to official secure channels.
- **Never provides medical, psychiatric, or legal advice.**
- For self-harm or immediate distress, immediately provides standard emergency helpline handoffs and flags for human crisis routing.
- **Never asks for credentials or private information.**

---

## 5. Epistemic Humility: Pleasant Wording $\neq$ Correct Support Resolution

A fluent, polite draft can easily create the illusion of resolution. However:
- A polite draft telling a customer to restart their phone when their account has been breached is dangerous.
- A draft suggesting an OS update for a customer experiencing hardware damage wastes customer time and erodes brand trust.
- TF-IDF similarity measures lexical overlap; it cannot verify whether the user's specific symptom matches the historical customer's root cause.

Therefore, **Phase 8 makes zero claims of reply quality, acceptance rate, or customer satisfaction.** Quality cannot be proven through automated string matching; it requires human review on the structured review sheet ([`outputs/replies/reply_human_review_sheet.csv`](file:///c:/Users/msiva/Videos/HIVER/hiver_twitter_exploration/outputs/replies/reply_human_review_sheet.csv)).

---

## 6. Separation of Concerns: Drafting vs. Decision Policy (Phase 9)

A fundamental architectural principle of this system is:
> **Drafting a reply is NOT a decision to send it.**

Phase 8 creates candidate drafts and records signals (`predicted_confidence`, `best_similarity_score`, `needs_human_review`, `restricted_draft`, `safety_flags`).

Phase 9 will introduce the **Decision Policy Router**, which will evaluate these signals against strict risk thresholds to decide:
- **`AUTO_HANDLE`**: Low-risk, high-confidence, high-similarity inquiries that are safe to send automatically.
- **`HUMAN_REVIEW_RECOMMENDED`**: Inquiries with moderate similarity or borderline confidence where human oversight is advised.
- **`ESCALATE_TO_SPECIALIST`**: High-risk safety cases, low-confidence predictions, or out-of-vocabulary inquiries requiring immediate human intervention.

---

## 7. Strict Data Protection & Integrity Rules

```text
Historical Retrieval Index:
- Exclusively data/processed/applesupport_train.csv (82,063 unique inquiries).

Development Drafts:
- Exclusively data/processed/applesupport_validation.csv (10,357 inquiries).

Protected Data:
- applesupport_test.csv: NEVER accessed.
- golden_set_200.csv: NEVER accessed.
- annotator_a_blind.csv / annotator_b_blind.csv: NEVER modified.
```
