# Phase 8 Reply Drafting Safety & Distribution Audit

## Executive Summary & Integrity Principles

In **Phase 8**, the AppleSupport inquiry pipeline was augmented with a safe, grounded reply-drafting engine. The engine operates on predictions from the Phase 6 TF-IDF classifier and historical support patterns retrieved from the Phase 7 train-only deduplicated retrieval index.

> [!IMPORTANT]
> **Core Safety & Grounding Principles**:
> 1. **No Hallucination Claims**: Constrained templates reduce unsupported claims and make drafts easier to audit. They do not guarantee that every draft is relevant or safe.
> 2. **Support Protocols vs. Facts**: Retrieved historical AppleSupport replies represent past interaction patterns (such as requesting OS version or advising a restart). They are **never** treated as customer-specific facts, promises, or diagnoses.
> 3. **Stricter Adaptation Threshold (0.50)**: Inquiries with similarity score $< 0.50$ default strictly to `conservative_no_evidence`. Weak lexical similarity (0.30–0.49) is recorded in metadata but is barred from altering drafts.
> 4. **No Automated Quality or Satisfaction Claims**: Zero reply acceptance, satisfaction, or accuracy claims are made prior to human or LLM-judge review. All 30 review rows in `reply_human_review_sheet.csv` are initialized to `NEEDS_HUMAN_REVIEW`.
> 5. **Strict Train-Only Isolation**: All retrieval vectors and classifiers were trained exclusively on `applesupport_train.csv`. Reply drafts were generated exclusively on `applesupport_validation.csv`. Zero rows of `applesupport_test.csv` or `golden_set_200.csv` were loaded.

---

## 1. Overall Population Statistics (`applesupport_validation.csv`)

- **Total Customer Inquiries Evaluated**: **10,357**
- **Restricted Safety Drafts (`restricted_draft = True`)**: **74 (0.71%)**
- **Human Classification Review Flags (`needs_human_review = True`)**: **5,634 (54.40%)**
- **Average Best Lexical Similarity Score**: **0.3906**

### Draft Mode Distribution

| Draft Mode | Count | Proportion | Operational Meaning |
| :--- | :---: | :---: | :--- |
| `template_with_historical_pattern` | 1,770 | 17.09% | Similarity $\ge 0.50$; synthesizes intent base template with safe historical protocol pattern |
| `conservative_no_evidence` | 8,513 | 82.20% | Similarity $< 0.50$; uses cautious base intent template without historical adaptation |
| `restricted_safety` | 74 | 0.71% | Safety flags detected; brief, neutral secure handoff without troubleshooting |

### Lexical Similarity Band Breakdown

| Similarity Band | Score Range | Count | Proportion | Influence on Draft |
| :--- | :---: | :---: | :---: | :--- |
| `insufficient_lexical_evidence` | < 0.30 | 3,277 | 31.64% | Conservative base template; no historical pattern |
| `weak_lexical_evidence` | 0.30 – 0.49 | 5,304 | 51.21% | Conservative base template; barred from altering drafts |
| `moderate_lexical_evidence` | 0.50 – 0.69 | 1,193 | 11.52% | Eligible for historical pattern adaptation |
| `high_lexical_similarity` | >= 0.70 | 583 | 5.63% | Eligible for historical pattern adaptation |

### Safety Flag Detection Breakdown

| Safety Trigger Category | Occurrences | Strict Handling Action |
| :--- | :---: | :--- |
| `unauthorized_payment_or_fraud` | 31 | Directs to official secure channel; zero credential/payment info requested |
| `password_or_recovery` | 29 | Directs to official secure channel; zero credential/payment info requested |
| `account_compromise` | 9 | Directs to official secure channel; zero credential/payment info requested |
| `threats_abuse_legal` | 3 | Directs to official secure channel; zero credential/payment info requested |
| `self_harm_or_danger` | 2 | Directs to official secure channel; zero credential/payment info requested |

---

## 2. Exemplary De-Identified Draft Cases

### A. Five Safe Drafts with Historical Patterns (`template_with_historical_pattern`)

#### Example A1: Tweet `725`
- **Customer Inquiry**: "I️ upgraded. I️t didn’t work."
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.5943`)
- **Best Similarity Score**: `0.8431` (`high_lexical_similarity`)
- **Historical Evidence Customer Tweets**: `1584670; 1706173; 1157354`
- **Historical Evidence Brand Replies**: `1584672; 1706175; 1157353`
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly. Please connect with Apple Support through an official secure channel so an advisor can review the specifics with you."
- **Grounding Note**: *Historical replies for similar training messages directed customers to secure support channels for investigation.*

#### Example A2: Tweet `1757`
- **Customer Inquiry**: "I️ just updated it and it’s still coming up [URL]"
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.5394`)
- **Best Similarity Score**: `0.5628` (`moderate_lexical_evidence`)
- **Historical Evidence Customer Tweets**: `1617107; 1473325; 2287437`
- **Historical Evidence Brand Replies**: `1617106; 1473324; 2287439`
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly. To help troubleshoot, have your device model and current software version ready when contacting official Apple Support through a secure channel."
- **Grounding Note**: *Historical replies for similar training messages commonly requested device model and software version. This draft incorporates that general support protocol without asserting a customer-specific diagnosis.*

#### Example A3: Tweet `2654`
- **Customer Inquiry**: "newest version of iOS and iPhone 7"
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.5362`)
- **Best Similarity Score**: `0.5377` (`moderate_lexical_evidence`)
- **Historical Evidence Customer Tweets**: `1851587; 2687716; 1679928`
- **Historical Evidence Brand Replies**: `1852462; 2687718; 1679927`
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly. Please connect with Apple Support through an official secure channel so an advisor can review the specifics with you."
- **Grounding Note**: *Historical replies for similar training messages directed customers to secure support channels for investigation.*

#### Example A4: Tweet `4870`
- **Customer Inquiry**: "iPhone 6 and yes it’s got the latest software update"
- **Predicted Intent**: `software_update_or_os_issue` (Confidence: `0.7981`)
- **Best Similarity Score**: `0.5010` (`moderate_lexical_evidence`)
- **Historical Evidence Customer Tweets**: `1537601; 1001163; 2301462`
- **Historical Evidence Brand Replies**: `1537603; 1001161; 2301465`
- **Draft Reply**: > "We understand you are experiencing an issue related to software or updating. We recommend reviewing Apple’s official support resources for software updates. For troubleshooting tailored to your setup, please reach out through an official secure channel. Please connect with Apple Support through an official secure channel so an advisor can review the specifics with you."
- **Grounding Note**: *Historical replies for similar training messages directed customers to secure support channels for investigation.*

#### Example A5: Tweet `4896`
- **Customer Inquiry**: "please make this stop!!! [URL]"
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.7428`)
- **Best Similarity Score**: `0.9096` (`high_lexical_similarity`)
- **Historical Evidence Customer Tweets**: `1712979; 1715207; 2093706`
- **Historical Evidence Brand Replies**: `1712978; 1715206; 2093705`
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly. Please connect with Apple Support through an official secure channel so an advisor can review the specifics with you."
- **Grounding Note**: *Historical replies for similar training messages directed customers to secure support channels for investigation.*

### B. Five Conservative No-Evidence Drafts (`conservative_no_evidence`)

#### Example B1: Tweet `730`
- **Customer Inquiry**: "Hello, internet. Can someone explain why this symbol keeps appearing on my phone and when I️ try to type the letter I️? Also [USER] [URL]"
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.6700`)
- **Best Similarity Score**: `0.3411` (`weak_lexical_evidence`)
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly."
- **Grounding Note**: *Best lexical similarity (0.3411) is below the 0.50 adaptation threshold. Draft uses conservative base template without historical troubleshooting guidance.*

#### Example B2: Tweet `1759`
- **Customer Inquiry**: "Why is “I️” keep changing to this and how do I️ stop it🤦🏽‍♀️ [USER] [USER] #anybody [URL]"
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.7855`)
- **Best Similarity Score**: `0.4895` (`weak_lexical_evidence`)
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly."
- **Grounding Note**: *Best lexical similarity (0.4895) is below the 0.50 adaptation threshold. Draft uses conservative base template without historical troubleshooting guidance.*

#### Example B3: Tweet `1781`
- **Customer Inquiry**: "why can’t I change ringer volume with the buttons? Whose dumb idea was it to change that and how do they still have a job?"
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.4178`)
- **Best Similarity Score**: `0.2430` (`insufficient_lexical_evidence`)
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly."
- **Grounding Note**: *Best lexical similarity (0.2430) is below the 0.50 adaptation threshold. Draft uses conservative base template without historical troubleshooting guidance.*

#### Example B4: Tweet `1785`
- **Customer Inquiry**: "epıl bana yardımcı olur musun telefon internet konusunda çok yakıyo napcaz ?"
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.4493`)
- **Best Similarity Score**: `0.4519` (`weak_lexical_evidence`)
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly."
- **Grounding Note**: *Best lexical similarity (0.4519) is below the 0.50 adaptation threshold. Draft uses conservative base template without historical troubleshooting guidance.*

#### Example B5: Tweet `2620`
- **Customer Inquiry**: "watchOs4 made my watch pointless Browsing music on my phone via the watch was 80% reason for buying it now it’s useless."
- **Predicted Intent**: `other_or_unclear` (Confidence: `0.5006`)
- **Best Similarity Score**: `0.2157` (`insufficient_lexical_evidence`)
- **Draft Reply**: > "Thank you for reaching out. To assist you safely, please provide more details about what you are experiencing, or contact Apple Support through an official secure channel. Please do not share private or account details publicly."
- **Grounding Note**: *Best lexical similarity (0.2157) is below the 0.50 adaptation threshold. Draft uses conservative base template without historical troubleshooting guidance.*

### C. Restricted Safety Drafts (`restricted_safety`)

#### Example C1: Tweet `84259`
- **Customer Inquiry**: "I’m hey [USER] do i have a problem here? So much for never getting hacked... [URL]"
- **Triggered Safety Flags**: `account_compromise`
- **Draft Reply**: > "For your account security, please use Apple’s official account-recovery or support channels immediately. Do not share passwords, verification codes, or personal details publicly."
- **Grounding Note**: *Safety restriction triggered: account compromise terms detected. Draft enforces strict credential security and directs to official recovery channels.*

#### Example C2: Tweet `97006`
- **Customer Inquiry**: "Looks a bit fishy. Scam? [USER] [USER] [URL]"
- **Triggered Safety Flags**: `unauthorized_payment_or_fraud`
- **Draft Reply**: > "For payment, purchase history, or refund concerns, please use Apple’s secure support or purchase-history options. Do not share payment details or card numbers publicly."
- **Grounding Note**: *Safety restriction triggered: billing/fraud/dispute terms detected. Draft enforces payment privacy and directs to secure purchase support without asserting transaction status.*

#### Example C3: Tweet `118047`
- **Customer Inquiry**: "How long do I have to wait after "invalid verification code entered too many times"? I can't update/download any apps"
- **Triggered Safety Flags**: `password_or_recovery`
- **Draft Reply**: > "For account access or password concerns, please use Apple’s official account-recovery or secure support channels. Never share passwords, verification codes, or personal credentials publicly."
- **Grounding Note**: *Safety restriction triggered: password/recovery terms detected. Draft directs to secure account recovery without requesting private credentials.*

---

## 3. Known Limitations & Technical Constraints

1. **Template Generalization vs. Specificity**: While constrained templates completely eliminate factual hallucination, they cannot resolve complex multi-turn edge cases on their own.
2. **Lexical Retrieval Boundaries**: TF-IDF cosine similarity measures token and n-gram overlap. It does not measure semantic entailment or verify if the historical diagnosis matches the user's specific symptom.
3. **Absence of Real-Time Account Access**: The system cannot query live order status, activation locks, or iCloud account states.
4. **Need for Phase 9 Decision Routing**: Drafting a reply is distinct from sending it. Phase 9 must implement a rigorous auto-handle vs. human escalation decision policy based on confidence, similarity, and safety restrictions.
5. **Regex Safety Audit Scope**: No matches to the defined prohibited phrases were found by the automated regex audit. Regex checks can detect selected risky phrases, but they cannot prove every generated draft is relevant, accurate, or safe. Human review remains necessary.

---

## 4. Required Epistemic Statement

> **Retrieval has been verified as train-only historical search. It has not yet been proven that retrieved reply patterns are relevant, safe, or beneficial. The 30-row human review sheet (`reply_human_review_sheet.csv`) is required before making that claim.**

