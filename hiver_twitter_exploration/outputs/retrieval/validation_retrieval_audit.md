# Historical AppleSupport Dialogue Retrieval Audit Report (Phase 7 - Improved)

> [!IMPORTANT]
> **Required Epistemic Disclaimer on Grounding & Accuracy**:
> **Retrieval has been verified as train-only historical search. It has not yet been proven that retrieved reply patterns are relevant, safe, or beneficial. The 30-row human review sheet is required before making that claim.**
>
> **A high TF-IDF cosine score means the new message and historical message share important words or phrases. It does not prove the historical reply is relevant, factually applicable, safe, or appropriate for the new customer.**
> A vague complaint with nearly identical profanity can receive similarity 1.0 but still provide poor grounding.
> High lexical similarity (`similarity >= 0.70`) reflects **stronger lexical similarity only**, NOT safe grounding or safe auto-handling.

## 1. Indexing & Querying Provenance

- **Unique Customer Inquiries Indexed**: 82,063 (`One searchable document = one unique customer_tweet_id`)
- **Source Interaction Pairs Processed**: 82,077 (`data/processed/applesupport_train.csv`)
- **Multi-Reply Customer Inquiries**: 14 rows (unique brand replies aggregated into structured fields)
- **Source File SHA-256**: `ade17dd3b359c08157773e4290f6fa5dc8dd19aa152d5d14417faa389326ad19`
- **Validation Queries Evaluated**: 10,357 external queries (`data/processed/applesupport_validation.csv`)
- **Audit Sample Selected**: Exactly 30 validation queries (seed `42`), producing 90 candidate pairs (top-k = 3)
- **Evidence Similarity Threshold**: 0.30 (conservative)

## 2. Partition Isolation & Anti-Leakage Verification

- **Query Tweet ID Overlap with Index**: 0 records
- **Conversation Group Overlap with Index**: 0 records
- **Isolation Status**: CONFIRMED ZERO LEAKAGE
- **Test & Golden Status**: Strictly 0 rows from `applesupport_test.csv` or `golden_set_200.csv` were loaded or queried.

## 3. Lexical Similarity Score Distribution (Full Validation Set)

- **Mean Best Similarity Score**: 0.3906
- **Median Best Similarity Score**: 0.3480
- **Proportion Below Evidence Threshold (< 0.30)**: 31.63% (3,276 / 10,357 queries)
- **Proportion with Sufficient Evidence (>= 0.30)**: 68.37% (7,081 / 10,357 queries)

### Lexical Similarity Band Breakdown

| Lexical Similarity Band | Query Count | % of Validation | Operational Interpretation |
| :--- | :--- | :--- | :--- |
| `high_lexical_similarity` (>= 0.70) | 583 | 5.63% | Stronger lexical similarity only; relevance unverified |
| `moderate_lexical_evidence` (0.50 - 0.70) | 1,193 | 11.52% | Substantial vocabulary overlap |
| `weak_lexical_evidence` (0.30 - 0.50) | 5,304 | 51.21% | Partial vocabulary overlap |
| `insufficient_lexical_evidence` (< 0.30) | 3,277 | 31.64% | **Escalation trigger: weak precedent** |

## 4. Five De-Identified High Lexical Similarity Examples (Relevance Unverified)

> [!WARNING]
> **Relevance Unverified**: High cosine similarity indicates lexical n-gram overlap. It does NOT guarantee that the historical reply is appropriate or safe for the new customer.

### Example 1 (Similarity Score: 1.0000 | Band: `high_lexical_similarity`)

- **Validation Query (Tweet `2278811`):**
  > "I️ I️ I️ I️ I️ I️ I️ I️ get your shit together"
- **Retrieved Historical Inquirer (Tweet `1915997`):**
  > "GET YOUR SHIT TOGETHER"
- **Actual AppleSupport Historical Reply (Tweet `1915996`):**
  > "Thanks for reaching out. We'd like to know more to assist. Send us a DM using the link below and we'll continue. [URL]"
- **Audit Status**: `needs_human_relevance_review = True`

### Example 2 (Similarity Score: 0.9488 | Band: `high_lexical_similarity`)

- **Validation Query (Tweet `2013657`):**
  > "fucking suckssss ive had the 7 plus for 3 months and its already acting up"
- **Retrieved Historical Inquirer (Tweet `2013673`):**
  > "fucking sucks ive had the 7 plus for 3 months and its already acting up"
- **Actual AppleSupport Historical Reply (Tweet `2013672`):**
  > "We'd like to know more about the issue you're experiencing. DM us the specifics, including which iOS version you’re using. [URL]"
- **Audit Status**: `needs_human_relevance_review = True`

### Example 3 (Similarity Score: 0.9014 | Band: `high_lexical_similarity`)

- **Validation Query (Tweet `1757772`):**
  > "Errrr [USER] what’s going on here [URL]"
- **Retrieved Historical Inquirer (Tweet `41612`):**
  > "What's going on here? [URL]"
- **Actual AppleSupport Historical Reply (Tweet `41611`):**
  > "Let's find out. What do you see if you click on the battery? Reply in DM to get started. [URL]"
- **Audit Status**: `needs_human_relevance_review = True`

### Example 4 (Similarity Score: 0.8210 | Band: `high_lexical_similarity`)

- **Validation Query (Tweet `1517054`):**
  > "I️ 👈🏽 this shit gotta go [USER]"
- **Retrieved Historical Inquirer (Tweet `2310607`):**
  > "This “ I️ I️ I️ I️ “ shit gotta go [USER]"
- **Actual AppleSupport Historical Reply (Tweet `2310606`):**
  > "We'd be happy to help. Have you already updated to 11.1.1? If not, backup your device and go ahead and update."
- **Audit Status**: `needs_human_relevance_review = True`

### Example 5 (Similarity Score: 0.8171 | Band: `high_lexical_similarity`)

- **Validation Query (Tweet `1687734`):**
  > "please fix this ASAP! #Apple [URL]"
- **Retrieved Historical Inquirer (Tweet `2116262`):**
  > "please fix this ASAP"
- **Actual AppleSupport Historical Reply (Tweet `2116261`):**
  > "Here’s what you can do to work around the issue until it’s fixed in a future software update: [URL]"
- **Audit Status**: `needs_human_relevance_review = True`

## 5. Five De-Identified Insufficient / Weak Lexical Evidence Examples (Escalation Signals)

### Example 1 (Similarity Score: 0.2148 | Band: `insufficient_lexical_evidence`)

- **Validation Query (Tweet `429596`):**
  > "If I'm in a fullscreen app and click a Mail notification, Mail goes fullscreen and splits the fullscreen app. Can I stop this?"
- **Retrieved Nearest Inquirer (Tweet `1674035`):**
  > "Y’all are on the cusp of world domination but can’t stop this——> I️ [USER]"
- **Actual AppleSupport Historical Reply (Tweet `1674034`):**
  > "Here’s what you can do to work around the issue until it’s fixed in a future software update: [URL]"
- **Audit Status**: Insufficient/weak evidence (`has_sufficient_historical_evidence = False`) | Warning: `Best similarity score (0.2148) below 0.30 threshold; weak or insufficient lexical evidence.`

### Example 2 (Similarity Score: 0.2450 | Band: `insufficient_lexical_evidence`)

- **Validation Query (Tweet `1165726`):**
  > "Messages and any other apps that allow the use of messaging"
- **Retrieved Nearest Inquirer (Tweet `1080379`):**
  > "I haven’t noticed it with any other apps (yet)."
- **Actual AppleSupport Historical Reply (Tweet `1080381`):**
  > "Let's troubleshoot the app itself with these steps mentioned here: [URL]"
- **Audit Status**: Insufficient/weak evidence (`has_sufficient_historical_evidence = False`) | Warning: `Best similarity score (0.2450) below 0.30 threshold; weak or insufficient lexical evidence.`

### Example 3 (Similarity Score: 0.2453 | Band: `insufficient_lexical_evidence`)

- **Validation Query (Tweet `2676470`):**
  > "atau klo HP error,Listrik mati,remote AC gak fungsi,simcard gak ada sinyal, mau ngehack facebook mantan bisa juga hubungi IT... 😂😂😂😂 (END)"
- **Retrieved Nearest Inquirer (Tweet `908231`):**
  > "A normal hp laptop"
- **Actual AppleSupport Historical Reply (Tweet `908232`):**
  > "Thanks for confirming. Please follow the steps in this article and let us know if it helps: [URL]"
- **Audit Status**: Insufficient/weak evidence (`has_sufficient_historical_evidence = False`) | Warning: `Best similarity score (0.2453) below 0.30 threshold; weak or insufficient lexical evidence.`

### Example 4 (Similarity Score: 0.2456 | Band: `insufficient_lexical_evidence`)

- **Validation Query (Tweet `515868`):**
  > "I worry this is bogus and if so you should be aware this is happening as the page requesting to do the download looked like the Apple webpage and it was being prompted by your “AppleCare” complete with icon."
- **Retrieved Nearest Inquirer (Tweet `1132149`):**
  > "Presuming this is bogus, yes? [URL]"
- **Actual AppleSupport Historical Reply (Tweet `1132148`):**
  > "This does appear to be a phishing attempt rather than a genuine warning. You can learn more here: [URL]"
- **Audit Status**: Insufficient/weak evidence (`has_sufficient_historical_evidence = False`) | Warning: `Best similarity score (0.2456) below 0.30 threshold; weak or insufficient lexical evidence.`

### Example 5 (Similarity Score: 0.2483 | Band: `insufficient_lexical_evidence`)

- **Validation Query (Tweet `14212`):**
  > "IOS 11 has been a nightmare since I downloaded it. Battery SOC is all over the board and slower than ever before"
- **Retrieved Nearest Inquirer (Tweet `1848938`):**
  > "touch id on my iPhone se is way much slower and inaccurate than ever before."
- **Actual AppleSupport Historical Reply (Tweet `1848937`):**
  > "Touch ID is an important part of the iPhone experience; we'll do all we can to help. Connect with us in DM to get started: [URL]"
- **Audit Status**: Insufficient/weak evidence (`has_sufficient_historical_evidence = False`) | Warning: `Best similarity score (0.2483) below 0.30 threshold; weak or insufficient lexical evidence.`

## 6. Cross-Conversation Repeated Wording Limitation

- A detailed audit is maintained in `train_validation_text_overlap_report.md`.
- Independent customers frequently post identical short support inquiries across separate conversation trees.
- While conversation isolation is preserved, repeated wording inflates lexical similarity on common phrases.
- For novel phrasing in production, retrieval evidence will naturally fall back to lower similarity bands, triggering necessary human review.

---
*Audit report generated by scripts/audit_retrieval.py.*