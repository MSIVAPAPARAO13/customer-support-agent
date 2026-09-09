# Cross-Conversation Repeated Wording & Strict Retrieval Audit

> [!IMPORTANT]
> **Epistemic Distiction: Cross-Conversation Repeated Wording vs. Data Leakage**:
> The text overlap documented here is **cross-conversation repeated wording**, NOT data leakage.
> Every validation conversation has a distinct `conversation_root_or_group_id` from the training set.
> However, real customers frequently post identical short inquiries (e.g. *'my battery is draining fast'*), which inflates lexical retrieval scores if unanalyzed.

## 1. Executive Summary: Overlap Statistics

- **Total Validation Inquiries Audited**: 10,357
- **Validation Inquiries with Exact Normalized Match in Train**: 66 (0.64%)
- **Validation Inquiries with Novel Phrasing**: 10,291 (99.36%)

## 2. Why Conversation-Group Isolation Does Not Prevent Repeated Wording

When we constructed our 80/10/10 dataset split in Phase 2, we enforced **strict conversation-group isolation**: all tweets belonging to the same root tree were kept together in either train, validation, or test.

However, independent customers around the world who experience common issues (such as iOS 11 battery drain, locked Apple IDs, or cracked screens) frequently compose tweets using identical or near-identical syntax. For instance, hundreds of distinct users tweeted the exact four words: *'iphone 7 ios 11'* or *'my battery is terrible'* in completely separate support threads.

This is natural language reuse, but it introduces an important evaluation caveat: offline retrieval evaluations will look exceptionally strong on these repeated inquiries, while providing less guidance on truly novel phrasing.

## 3. Five De-Identified Examples of Cross-Conversation Repeated Wording

### Example 1: "yes i am"
- **Validation Sample**: Tweet `708647` (Conversation Group `708645`)
- **Training Match**: Tweet `150015` (Conversation Group `150016`)
- **Corpus Frequency**: Appears 14 time(s) in Train, 3 time(s) in Validation
- **Status**: Independent conversations with identical normalized wording.

### Example 2: "yes it does"
- **Validation Sample**: Tweet `455050` (Conversation Group `455048`)
- **Training Match**: Tweet `302650` (Conversation Group `302655`)
- **Corpus Frequency**: Appears 6 time(s) in Train, 2 time(s) in Validation
- **Status**: Independent conversations with identical normalized wording.

### Example 3: "fix this shit [user]"
- **Validation Sample**: Tweet `2072151` (Conversation Group `2072151`)
- **Training Match**: Tweet `2177354` (Conversation Group `2177354`)
- **Corpus Frequency**: Appears 2 time(s) in Train, 2 time(s) in Validation
- **Status**: Independent conversations with identical normalized wording.

### Example 4: "iphone 7 plus"
- **Validation Sample**: Tweet `263453` (Conversation Group `263454`)
- **Training Match**: Tweet `63202` (Conversation Group `63200`)
- **Corpus Frequency**: Appears 20 time(s) in Train, 1 time(s) in Validation
- **Status**: Independent conversations with identical normalized wording.

### Example 5: "fix this i️"
- **Validation Sample**: Tweet `2284098` (Conversation Group `2284098`)
- **Training Match**: Tweet `50068` (Conversation Group `50068`)
- **Corpus Frequency**: Appears 12 time(s) in Train, 1 time(s) in Validation
- **Status**: Independent conversations with identical normalized wording.

## 4. Normal Retrieval vs. Strict Text-Exclusion Retrieval Comparison

To understand how much offline retrieval relies on exact query duplicates, we ran a **Strict Retrieval Diagnostic** where candidates sharing identical normalized wording with the query were excluded (in addition to standard tweet ID and conversation-group exclusion).

| Metric | Normal Retrieval | Strict Diagnostic Retrieval | Delta / Observation |
| :--- | :--- | :--- | :--- |
| **Mean Best Similarity** | 0.3906 | 0.3898 | -0.0008 |
| **Median Best Similarity** | 0.3480 | 0.3479 | -0.0001 |
| **Proportion Below 0.30 Threshold** | 31.63% | 31.68% | +0.05% |
| **Proportion with High Similarity (>= 0.70)** | 5.63% | 5.51% | -0.12% |

### Lexical Similarity Band Breakdown Comparison

| Lexical Similarity Band | Normal Retrieval Count (%) | Strict Diagnostic Count (%) |
| :--- | :--- | :--- |
| `high_lexical_similarity` | 583 (5.63%) | 571 (5.51%) |
| `moderate_lexical_evidence` | 1,193 (11.52%) | 1,199 (11.58%) |
| `weak_lexical_evidence` | 5,304 (51.21%) | 5,305 (51.22%) |
| `insufficient_lexical_evidence` | 3,277 (31.64%) | 3,282 (31.69%) |

## 5. Architectural Implications for Production

1. **Escalation Gating**: Queries dropping into `insufficient_lexical_evidence` (< 0.30) or `weak_lexical_evidence` (0.30 - 0.50) must not trigger autonomous reply drafting.
2. **Evidence Sanitization**: All candidate pairs are sanitized (@mentions -> `[USER]`, URLs -> `[URL]`) before being provided to downstream prompt builders.
3. **Epistemic Humility**: Retrieval provides lexical evidence of past communication patterns, never factual ground truth for the new customer.

---
*Report generated by scripts/audit_retrieval_text_overlap.py.*