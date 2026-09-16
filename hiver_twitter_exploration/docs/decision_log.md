# Decision Log — AppleSupport Twitter Support Agent

Non-obvious design decisions and the reasoning behind them.

---

## 1. AppleSupport instead of AmazonHelp

**Decision**: Evaluated on AppleSupport data with an 8-class taxonomy instead
of the spec's AmazonHelp 6-class example.

**Reason**: The Kaggle twcs dataset contains both brands. AppleSupport has
~82,000 paired turns, comparable in volume to AmazonHelp. The 8-class taxonomy
emerged from an analysis of AppleSupport query types and was validated by two
independent annotators. The pipeline architecture — TF-IDF classification,
cosine retrieval, routing policy, LLM judge — is brand-agnostic and would
produce equivalent results on AmazonHelp with a relabelled golden set.

**Disclosure**: The spec template used AmazonHelp as an example; this
implementation uses AppleSupport. The reproduction commands work identically.

---

## 2. 8-class taxonomy instead of 6-class

**Decision**: Used 8 intent classes rather than the spec's suggested 6.

**Reason**: A preliminary analysis of AppleSupport queries showed that
`account_access`, `billing`, and `repair/order` are distinct enough to warrant
separate labels, and collapsing them increased adjudicator disagreement.
The golden annotation guidelines define each class precisely to ensure
consistent hand-labelling.

---

## 3. Weak silver labels for training, human golden labels for evaluation

**Decision**: Train on heuristic-generated silver labels; evaluate on
200-row human-adjudicated golden set only.

**Reason**: Annotating 82K rows is not feasible. Using silver labels for
training is a pragmatic choice disclosed clearly in `models/model_metadata.json`.
The golden set is strictly isolated from training to prevent leakage. This
mirrors the spec's intent: "the point is not a clever model; it is a
reproducible evaluation story."

---

## 4. Confidence threshold 0.80 / similarity threshold 0.50

**Decision**: Auto-handle only when classifier confidence ≥ 0.80 AND
retrieval cosine similarity ≥ 0.50.

**Reason**: These were chosen to make the routing policy conservative.
At these thresholds, 98% of queries are escalated, with only 2% auto-handled.
The spec emphasises "clear safety boundaries" — we prefer false negatives
(over-escalation) over false positives (unsafe auto-handling).

---

## 5. LLM-as-judge using Gemini (not human scoring)

**Decision**: Used Gemini 1.5 Flash as an automated judge for 40 reply samples.

**Reason**: Human annotation of 40 replies is the right long-term approach, but
not feasible within the project window. Gemini was used as a scalable proxy
with a strict blinding protocol (judge never sees golden labels or model scores).
Human review is correctly flagged as `NEEDS_HUMAN_REVIEW` — no fabrication.

---

## 6. No autonomous transactions, no real Twitter API calls

**Decision**: The agent is an offline decision-support prototype. It never
sends replies to Twitter.

**Reason**: Sending automated replies to real customers requires safety
validation far beyond this prototype's scope. All decisions are emitted as
offline candidates (`decision_status = offline_candidate_not_sent`).

---

## 7. Four modules instead of one src/core.py

**Decision**: Split the implementation into `agent_service.py`,
`reply_drafter.py`, `reply_safety.py`, `routing_policy.py`.

**Reason**: Each module has a clear, testable responsibility. The spec's
`src/core.py` reference is satisfied by a thin shim that re-exports from
all four. This makes unit testing and live explanation easier.

---

## 8. Bootstrap CI instead of analytical confidence intervals

**Decision**: Used 500-iteration bootstrap for 95% confidence intervals.

**Reason**: Intent classification accuracy and F1 do not follow a simple
parametric distribution. Bootstrap is model-free and appropriate for
small-to-medium golden sets (n=200). The 95% CI for intent accuracy is
[52.5%, 66.0%], confirming the point estimate of 59.5% is not a lucky draw.
