# Golden Set — Annotation Protocol

This directory contains the 200-row human-annotated golden evaluation set
for the Hiver AppleSupport Twitter Support Agent.

---

## Files

| File | Purpose |
|------|---------|
| `annotation_guidelines.md` | **Authoritative label rules** — read this first |
| `annotator_a_blind.csv` | First-pass blind annotations (do not modify) |
| `annotator_b_blind.csv` | Second-pass blind annotations (do not modify) |
| `adjudication_sheet.csv` | Final human-endorsed labels (200/200 DONE) |
| `ai_assisted_annotation_review.csv` | AI-generated proposals used as review aid only |
| `human_assisted_adjudication_audit.csv` | Audit trail of adjudication decisions |
| `golden_labels_freeze_manifest.json` | SHA-256 integrity lock — do not alter |
| `golden_set_protocol.md` | Full sampling and double-labelling protocol |
| `golden_isolation_report.md` | Confirms golden IDs are excluded from training |

---

## Taxonomy (8-class AppleSupport)

| Label | Meaning |
|-------|---------|
| `software_update_or_os_issue` | iOS/macOS update failures, crashes, boot loops |
| `device_performance_or_hardware` | Battery, overheating, unresponsive touchscreen |
| `connectivity_and_network` | Wi-Fi, Bluetooth, cellular connectivity |
| `apps_services_or_icloud` | App Store, iCloud sync, Apple Music, Siri |
| `account_access_and_apple_id` | Login failures, locked Apple ID, password resets |
| `billing_subscription_or_purchase` | Charges, refunds, subscription management |
| `repair_replacement_or_order` | Screen repair, device replacement, order status |
| `other_or_unclear` | Does not fit above categories or too ambiguous |

> [!NOTE]
> **Brand note**: The spec template uses AmazonHelp (6-class). This project
> uses AppleSupport (8-class), which is also in the same Kaggle dataset. The
> taxonomy is more fine-grained and internally consistent. The pipeline
> architecture is identical — only the labels differ.

---

## How the golden set was built

1. `scripts/build_corpus.py` built 82,063 AppleSupport pairs from `twcs.csv`
2. `scripts/create_golden_set.py` stratified-sampled 200 rows (seed 42)
3. Golden IDs were isolated from training (see `golden_isolation_report.md`)
4. Two annotators labelled independently (`annotator_a_blind.csv`, `annotator_b_blind.csv`)
5. AI-generated proposals were used for review aid only, never as independent labels
6. A human adjudicator reconciled disagreements into `adjudication_sheet.csv`
7. `scripts/freeze_golden_labels.py` locked labels with SHA-256 (`golden_labels_freeze_manifest.json`)

---

## Integrity rules

- **Do NOT modify** `annotator_a_blind.csv` or `annotator_b_blind.csv`
- **Do NOT modify** `golden_labels_freeze_manifest.json`
- **Do NOT** use AI suggestions as independent human annotations
- All 200 rows must have `status=DONE` before running evaluation

---

## Reproduce evaluation

```powershell
# Verify integrity
python scripts/verify_final_evaluation_inputs.py

# Run inference on golden set
python scripts/run_golden_agent_inference.py

# Compute final metrics
python scripts/run_final_evaluation.py
```

Or use the unified evaluate.py harness:

```powershell
python scripts/evaluate.py \
    --train data/processed/corpus.csv \
    --golden data/golden/adjudication_sheet.csv \
    --out outputs/real_metrics.json
```
