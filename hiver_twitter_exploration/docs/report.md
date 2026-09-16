# Submission Report — AppleSupport Twitter Support Agent

**Candidate**: Hiver Take-Home Assessment
**Dataset**: Kaggle Customer Support on Twitter — AppleSupport brand
**Evaluation date**: 2026-09-16
**Status**: Phase 10 complete — all metrics final

---

## Summary

This project implements an evidence-first, conservative Twitter support agent
for AppleSupport tweets. It predicts an 8-way intent, retrieves similar
historical resolutions, drafts a safe reply, and routes to either
`auto_handle` or `escalate` with a human-readable reason code.

The agent is an **offline decision-support prototype**. It never sends
automated replies to real customers.

---

## Baseline and headline metrics

Evaluated on 200 human-labelled golden rows, locked with SHA-256
(`data/golden/golden_labels_freeze_manifest.json`).

| Model | Intent accuracy | Macro F1 | Selective accuracy / coverage | Reply acceptance |
|---|---:|---:|---:|---:|
| Trivial: always `other_or_unclear` | 4.5% | 1.1% | — | N/A |
| Simple: keyword rules | 43.0% | 53.3% | — | N/A |
| **Main: TF-IDF + routing** | **59.5%** | **66.3%** | **75.0% / 2.0%** | **90% (36/40)** |

Bootstrap 95% CI (500 iterations, seed 42):

| Metric | Point estimate | 95% CI |
|--------|---------------|--------|
| Intent accuracy | 59.5% | [52.5%, 66.0%] |
| Macro F1 | 66.3% | [59.9%, 71.5%] |
| Action accuracy | 36.0% | [29.0%, 43.0%] |
| Auto-handle coverage | 2.0% | [0.5%, 4.0%] |

---

## Per-class F1

See `outputs/final_evaluation/per_intent_metrics.csv` for full per-class
precision, recall, and F1 on the golden set.

---

## Qualitative evaluation (LLM judge)

A 40-row stratified sample was evaluated by a blinded Gemini 1.5 Flash judge
using the rubric in `docs/judge_rubric.md`. The judge never sees golden labels
or model confidence scores.

| Dimension | Mean score (/5) |
|-----------|----------------|
| Relevance | 3.30 |
| Historical grounding | 3.90 |
| Safety & privacy | **4.97** |
| Clear next step | 3.40 |
| Routing appropriateness | **4.62** |
| **Overall ACCEPT** | **36/40 (90%)** |

Human–judge agreement audit: 40 rows are available in
`outputs/final_evaluation/human_judge_agreement_sheet.csv` (status:
`NEEDS_HUMAN_REVIEW`). No fabricated human scores.

---

## Architecture

| Component | Implementation |
|-----------|---------------|
| Intent classifier | TF-IDF (1–2 grams, 30K features) + Logistic Regression (balanced, lbfgs) |
| Training data | 25,760 silver-label rows (weak heuristics over 82K AppleSupport turns) |
| Taxonomy | 8-class AppleSupport |
| Retrieval | Cosine similarity on TF-IDF customer matrix (82,063 turns) |
| Routing thresholds | Confidence < 0.80 → escalate; Similarity < 0.50 → escalate |
| Safety | PII detection + refusal for account/legal/medical queries |
| Golden set | 200 rows, human-adjudicated, SHA-256 frozen |
| Evaluation | Run-once, frozen labels, 500-iteration bootstrap CI |

---

## Known limitations

1. **Silver-label training**: The classifier is trained on heuristic-generated
   labels, not human annotations. It inherits systematic blind spots of the
   keyword heuristics.

2. **Other/unclear is a fallback bucket**: The `other_or_unclear` class was
   seeded with 2,000 randomly sampled low-confidence rows; it is not a clean
   semantic category.

3. **Confidence is uncalibrated**: Model probability estimates are not
   calibrated against human ground truth. The 0.80 threshold was chosen
   conservatively to minimise unsafe auto-handles.

4. **2% auto-handle coverage**: The conservative routing policy escalates 98%
   of queries. This is a safety feature, not a bug.

5. **Human agreement pending**: Cohen's Kappa (LLM judge vs. human) cannot be
   reported without real independent human review of the 40-item sample.

---

## Reproduce

### Smoke test (< 1 minute, no API key)

```powershell
python scripts/make_smoke_data.py
python scripts/run_agent.py --train data/demo/corpus.csv --input data/demo/incoming.csv --output outputs/smoke_predictions.csv
python scripts/evaluate.py --train data/demo/corpus.csv --golden data/demo/golden_smoke.csv --out outputs/smoke_metrics.json
```

### Real results (< 15 minutes after twcs.csv download)

```powershell
# 1. Place twcs.csv at data/raw/twcs.csv (Kaggle terms govern use)
python scripts/build_corpus.py --input data/raw/twcs.csv --brand AppleSupport --limit 30000
python scripts/make_annotation_sheet.py --corpus data/processed/corpus.csv --n 200

# 2. Complete human annotation, then:
python scripts/run_final_evaluation.py
```

---

## Files

```
src/core.py               — Single-import shim (re-exports from 4 modules)
src/agent_service.py      — Unified orchestrator
src/reply_drafter.py      — Conservative reply generation
src/reply_safety.py       — PII and restriction detection
src/routing_policy.py     — Threshold-based routing engine
docs/decision_log.md      — Non-obvious design decisions
docs/judge_rubric.md      — Reply evaluation rubric
data/golden/README.md     — Annotation protocol
outputs/final_evaluation/ — All certified metrics and outputs
```
