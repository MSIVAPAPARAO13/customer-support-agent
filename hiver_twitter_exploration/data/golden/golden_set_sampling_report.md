# Golden Evaluation Set Sampling Audit Report (Phase 4)

## 1. Sampling Parameters & Provenance

- **Source Dataset**: Strictly `data/processed/applesupport_test.csv` (10,279 rows).
- **Unique Usable Test Candidates**: 10,273 unique customer tweets.
- **Sample Size**: Exactly **200 unique customer tweets** (`golden_001` through `golden_200`).
- **Random Seed**: `42` (deterministic reproducibility).
- **Safety Slice Count**: **17 critical safety/fraud/billing cases**.
- **Unique Tweet ID Integrity**: 200 distinct IDs (0 duplicates).

## 2. Weak Sampling Bucket Distribution

> [!NOTE]
> **Weak Bucket Disclaimer**: These buckets were used solely to enforce topical diversity during sampling. They were **never shown to human annotators** and are **never treated as ground-truth intent labels**.

| Weak Sampling Bucket | Count in Golden Set | % of Golden Set | Sampling Objective |
| :--- | :--- | :--- | :--- |
| `software_update_bucket` | 35 | 17.5% | Broad diversity coverage |
| `device_hardware_bucket` | 35 | 17.5% | Broad diversity coverage |
| `unclear_general_bucket` | 25 | 12.5% | Broad diversity coverage |
| `apps_icloud_bucket` | 25 | 12.5% | Broad diversity coverage |
| `connectivity_network_bucket` | 20 | 10.0% | Broad diversity coverage |
| `repair_order_bucket` | 20 | 10.0% | Broad diversity coverage |
| `account_access_bucket` | 20 | 10.0% | Broad diversity coverage |
| `billing_purchase_bucket` | 20 | 10.0% | Broad diversity coverage |

## 3. High-Priority Safety Slice

Exactly **17 messages** contain high-liability keywords related to account recovery, compromised credentials, unauthorized credit card charges, and payment disputes. These cases evaluate whether the automated agent safely escalates sensitive emergencies to human specialists.

## 4. Double-Blind Protocol Integrity

1. **Historical Brand Replies Omitted**: Annotators see only the customer text, ensuring they evaluate customer intent rather than Apple agent responses.
2. **Model Predictions Omitted**: Zero synthetic, keyword, or model predictions appear in the annotation sheets to prevent anchoring bias.
3. **Independent Shuffling**: Annotator A (seed 101) and Annotator B (seed 202) receive different presentation sequences to prevent order-effect correlation.
