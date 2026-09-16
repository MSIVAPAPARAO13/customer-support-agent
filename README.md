# Hiver Take-Home: Evidence-First Twitter Support Agent (`hiver_twitter_exploration`)
<img width="1886" height="827" alt="image" src="https://github.com/user-attachments/assets/9bb4f22c-cf9a-4fdc-acff-832da82deb94" />

An intentionally small, inspectable agent for **AppleSupport** tweets. It predicts an 8-way intent, retrieves similar historical resolutions, drafts a conservative reply, and either auto-handles or escalates with a visible reason. The point is not a clever model; it is a reproducible evaluation story with clear safety boundaries.

> **Brand note**: The spec template uses AmazonHelp (6-class). This project uses AppleSupport (8-class) from the same Kaggle dataset. The pipeline architecture is identical — only the labels differ. See [`docs/decision_log.md`](docs/decision_log.md).

> **Status**: Phase 10 complete. All 200 golden rows are human-adjudicated and frozen. Final metrics are certified from a single locked evaluation run.

---

## Reproduce the smoke test (under 1 minute)

Requires Python 3.10+; scikit-learn, numpy, pandas (see `requirements.txt`).

```powershell
python scripts/make_smoke_data.py
python scripts/run_agent.py --train data/demo/corpus.csv --input data/demo/incoming.csv --output outputs/smoke_predictions.csv
python scripts/evaluate.py --train data/demo/corpus.csv --golden data/demo/golden_smoke.csv --out outputs/smoke_metrics.json
```

Inspect `outputs/smoke_predictions.csv`: the iOS update error query maps to `software_update_or_os_issue` and escalates (confidence below threshold); the hacked-account query escalates via `restricted_safety_flag`. The smoke metrics only verify the harness runs — they are not model quality evidence.

---

## Reproduce real results (under 15 minutes after download)

1. Download `twcs.csv` from [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) and place it at `data/raw/twcs.csv`. Kaggle terms govern its use; do not commit the raw file.
2. Build a capped, paired corpus (30k examples keeps local runs quick):

```powershell
python scripts/build_corpus.py --input data/raw/twcs.csv --brand AppleSupport --limit 30000
python scripts/make_annotation_sheet.py --corpus data/processed/corpus.csv --n 200
```

3. Complete and double-label the 200 real rows per [`data/golden/README.md`](data/golden/README.md). Remove their IDs from training to prevent leakage.
4. Run the full evaluation pipeline:

```powershell
python scripts/freeze_golden_labels.py
python scripts/run_golden_agent_inference.py
python scripts/run_final_evaluation.py
```

Or use the unified harness directly:

```powershell
python scripts/evaluate.py \
    --train data/processed/corpus.csv \
    --golden data/golden/adjudication_sheet.csv \
    --out outputs/real_metrics.json
```

---

## Baselines and headline metrics

Evaluated on 200 human-labelled golden rows (SHA-256 frozen). Fill this table only from `outputs/real_metrics.json` after the human annotation process.

| Model | Intent accuracy | Macro F1 | Selective accuracy / coverage | Reply acceptance |
|---|---:|---:|---:|---:|
| Trivial: always `other_or_unclear` | **4.5%** | **1.1%** | — | N/A |
| Simple: keyword rules | **43.0%** | **53.3%** | — | N/A |
| **Main: TF-IDF + routing (this project)** | **59.5%** | **66.3%** | **75.0% / 2.0%** | **90% (36/40)** |

Bootstrap 95% CI (500 iterations, seed 42): Intent accuracy [52.5%, 66.0%], Macro F1 [59.9%, 71.5%].

For per-class F1, see `outputs/final_evaluation/per_intent_metrics.csv`.
For reply acceptance methodology, see [`docs/judge_rubric.md`](docs/judge_rubric.md).

---
<img width="1908" height="853" alt="image" src="https://github.com/user-attachments/assets/76b5828b-f6c0-44d7-b7b8-b597ee319fa8" />

## Design

| Component | Implementation | Why it is interview-friendly |
|---|---|---|
| Intent | keyword baseline; TF-IDF + Logistic Regression main model | predictions are traceable to words and IDs |
| Grounding | retrieve up to 3 historical customer→brand pairs | reply uses only safe resolution templates; it cannot claim a historical fact is current |
| Routing | confidence + similarity thresholds + high-risk lexicon | every decision emits a short reason code |
| Quality | human golden labels, intent metrics, reply judge, human/judge agreement | separates a plausible demo from proof |

---

## Repository map

- `src/core.py` — single import shim re-exporting from `agent_service`, `reply_drafter`, `reply_safety`, `routing_policy`
- `scripts/` — corpus construction, annotation sampling, inference, evaluation, smoke fixtures
- `data/golden/` — real-data annotation protocol and frozen golden set
- `docs/report.md` — submission-ready report with filled metric table
- `docs/decision_log.md` — non-obvious design decisions
- `docs/judge_rubric.md` — reply evaluation rubric

---

## Sources

- [Kaggle Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — required source data
- TF-IDF and cosine nearest-neighbour retrieval are standard IR methods; the implementation is original
- This repository was developed with an AI coding assistant. All logic is compact enough to explain and modify live.

---
<img width="1884" height="950" alt="image" src="https://github.com/user-attachments/assets/81922505-9ebd-47e5-adb5-293f3dd6ac6e" />



## The Core Mental Model (Crucial for Interviews)

Many beginners make the mistake of treating Twitter data as isolated sentences or a generic text classification task. In conversational support systems, the data has a specific relational topology:

```text
Row 1 (Customer Tweet)  ───[in_response_to_tweet_id]───► None (Opening problem statement)
Row 2 (Brand Tweet)     ───[in_response_to_tweet_id]───► Points to Row 1 (Agent troubleshooting)
Row 3 (Customer Tweet)  ───[in_response_to_tweet_id]───► Points to Row 2 (Follow-up symptom)
Row 4 (Brand Tweet)     ───[in_response_to_tweet_id]───► Points to Row 3 (Resolution / DM closure)
```

### Key Principles
1. **One row = One tweet event**: Each row captures a single tweet published at a specific moment in time.
2. **One conversation = Several linked tweet rows**: Conversations are not stored in nested JSON trees; they are reconstructed by following parent tweet pointers (`in_response_to_tweet_id`).
3. **Customer problem = Inbound tweet (`inbound=True`)**: Originates from an anonymized user (numeric `author_id`) describing an issue or complaint.
4. **Brand resolution pattern = Outbound reply (`inbound=False`)**: Sent by a verified brand handle (e.g., `@AppleSupport`, `@AmazonHelp`) containing troubleshooting instructions, FAQs, or escalation steps.
5. **We are NOT training AI yet**: We are proving that the dataset reliably yields high-quality `(Customer Query → Brand Response)` pairs. These pairs will become the ground-truth historical knowledge base for semantic retrieval and response generation in later phases.

---
<img width="1919" height="662" alt="image" src="https://github.com/user-attachments/assets/c5d17b73-3b3b-4014-8f3d-99daf680cb41" />

## Project Structure

```text
hiver_twitter_exploration/
  ├── requirements.txt              # Minimal dependencies (pandas)
  ├── explore_dataset.py            # Phase 1: Exploratory analysis CLI engine
  ├── preprocess_applesupport.py    # Phase 2: AppleSupport filtering & preprocessing CLI
  ├── discover_intents.py           # Phase 3: AppleSupport intent-taxonomy discovery CLI
  ├── README.md                     # Comprehensive documentation & setup guide
  ├── scripts/                      # Phase 4 automation scripts
  │   ├── create_golden_set.py      # Deterministic 200-row test sampling (seed 42)
  │   ├── verify_golden_isolation.py # Cross-partition leak & isolation auditor
  │   └── reconcile_golden_labels.py # Pure-Python Cohen's Kappa & adjudication engine
  ├── outputs/                      # Phase 1 exploratory reports
  │   ├── dataset_summary.md        # Complete schema profile & aggregate statistics
  │   ├── brand_counts.csv          # Outbound counts & valid reply pairs
  │   └── sample_conversations.md   # Thread linkage examples
  └── data/
      ├── processed/                # Phase 2 processed datasets & splits
      │   ├── applesupport_pairs_all.csv   # Complete 106,646 extracted pairs
      │   ├── applesupport_train.csv       # 82,077 usable training pairs (80%)
      │   ├── applesupport_validation.csv  # 10,357 usable validation pairs (10%)
      │   ├── applesupport_test.csv        # 10,279 usable test pairs (10%)
      │   └── preprocessing_report.md      # Full preprocessing & leakage audit
      ├── taxonomy/                 # Phase 3 taxonomy & discovery artifacts
      │   ├── intent_discovery_sample.csv  # Review sample (600 rows, seed 42)
      │   ├── intent_taxonomy_v1.md        # Comprehensive 8-intent guidelines
      │   ├── intent_examples.csv          # Curated balanced examples
      │   ├── ambiguous_or_multi_intent_examples.csv # Edge cases & escalation logs
      │   └── intent_discovery_report.md   # Statistical discovery audit report
      └── golden/                   # Phase 4 human golden evaluation benchmark
          ├── golden_set_200.csv           # Master 200-row evaluation sample
          ├── annotator_a_blind.csv        # Double-blind annotation sheet A (seed 101)
          ├── annotator_b_blind.csv        # Double-blind annotation sheet B (seed 202)
          ├── adjudication_sheet.csv       # Human adjudication template for disputes
          ├── golden_set_protocol.md       # End-to-end evaluation & governance protocol
          ├── annotation_guidelines.md     # Detailed human labeling guidelines
          ├── golden_set_sampling_report.md# Sampling audit and bucket breakdown
          └── golden_isolation_report.md   # Proof of zero train/val overlap
```

---

## Quickstart Guide (Windows PowerShell)

### 1. Prerequisites
Ensure Python (version 3.9+) is installed and accessible in PowerShell:
```powershell
python --version
```

### 2. Install Dependencies
Navigate into the `hiver_twitter_exploration` project folder and install `requirements.txt`:
```powershell
cd C:\Users\msiva\Videos\HIVER\hiver_twitter_exploration
pip install -r requirements.txt
```

### 3. Run the Explorer
You can pass the path to the dataset directory as an argument:
```powershell
python explore_dataset.py "C:\Users\msiva\Videos\HIVER\twcs"
```

Or you can run it without arguments (it will automatically search the parent directory for `twcs/`):
```powershell
python explore_dataset.py
```

All reports will be generated inside the `outputs/` folder.

---

## Dataset Schema Explained in Beginner-Friendly Language

| Column Name | Semantic Role | Why It Matters in Real Life |
| :--- | :--- | :--- |
| `tweet_id` | **Primary Key** | The unique ID assigned by Twitter to every single tweet. Used to uniquely index inquiries and resolutions. |
| `author_id` | **Sender Handle / User ID** | If `inbound=False`, this contains the official brand name (e.g., `AppleSupport`, `AmazonHelp`). If `inbound=True`, it is an anonymized number (e.g., `105834`) to respect customer privacy. |
| `inbound` | **Customer vs. Brand Flag** | A boolean. `True` indicates a customer reaching out for help. `False` indicates a company agent replying to a customer. |
| `created_at` | **Timestamp** | The publication timestamp (UTC). Enables calculating time-to-first-response (SLA) and conversational ordering. |
| `text` | **Message Body** | The actual tweet text (up to 280 characters). Contains customer problem descriptions or brand troubleshooting guidance. |
| `in_response_to_tweet_id` | **Parent Tweet ID** | **The most reliable linking key.** When a brand replies, this field stores the `tweet_id` of the customer message they are answering. |
| `response_tweet_id` | **Child Response ID(s)** | Points forward to subsequent replies. **Note**: Can contain multiple tweet IDs separated by commas or spaces when multiple agents or users reply to the same tweet. |

### Why Missing Values in Reply Columns are Normal
- If `in_response_to_tweet_id` is missing (`NaN`), the tweet is an **initiating tweet** (the start of a new customer question).
- If `response_tweet_id` is missing (`NaN`), the tweet has **no subsequent replies** (it was the closing statement of the conversation).
- Notice that `tweet_id`, `author_id`, `inbound`, and `text` have **zero missing values** across all 2.81 million records.

---

## How Customer → Brand Reply Pairs are Defined

A valid conversational turn pair is defined with strict mathematical rigor:
- **Parent Tweet**: `inbound == True` (an authentic customer question)
- **Child Tweet**: `inbound == False` (an official brand response)
- **Relational Link**: `child.in_response_to_tweet_id == parent.tweet_id`

We **do not rely on `@mentions`** to infer pairs, because users frequently mention multiple brands or friends in social chatter without any support interaction occurring. Direct pointer matching ensures 100% true parent-child conversational integrity.

---

## What We Learned from the twcs Dataset

1. **Massive Scale**: The dataset contains **2,811,774 tweets** across 516 MB of data.
2. **Inbound vs. Outbound Balance**:
   - **54.7% Inbound** (1,537,843 customer messages)
   - **45.3% Outbound** (1,273,931 brand responses)
3. **High Conversational Linkage**: Over **1,261,888 valid customer-to-brand reply pairs** exist in the raw data, confirming that this corpus is an exceptional source of real-world support dialogues.
4. **Brand Specialization**:
   - `AmazonHelp` is the highest-volume brand (168,814 reply pairs), dealing primarily with shipping, delivery delays, and refund requests.
   - `AppleSupport` is the second highest (106,646 reply pairs), featuring rich, technical multi-step troubleshooting (iOS versions, hardware reboots, settings navigation).
   - `SpotifyCares` (43,092 reply pairs) offers rich app-level debugging (OS versions, offline cache, Bluetooth pairing).
5. **No Exact Duplicate Rows**: All tweets have unique IDs and distinct timestamps.

---

## What We Are Deliberately NOT Building Yet

In professional machine learning engineering, building too early is the #1 cause of project failure. In Phase 1, we are deliberately **NOT** building:

- ❌ **No Machine Learning Classifier**: No intent classifiers (e.g., LogisticRegression, XGBoost) before establishing clean labels.
- ❌ **No Text Embeddings**: No Sentence-Transformers, OpenAI embeddings, or cosine similarity calculations.
- ❌ **No Vector Database / RAG**: No ChromaDB, FAISS, or Pinecone indexing.
- ❌ **No LLM Reply Generation**: No calls to OpenAI, Claude, Gemini, or local Ollama models.
- ❌ **No Web App or UI**: No Streamlit, Gradio, or React frontend.

**Why?** If the underlying data quality, brand choice, text cleaning, and pair extraction are flawed, all downstream AI models will produce garbage outputs (*Garbage In, Garbage Out*).

---

## Phase 2: AppleSupport Dialogue Filtering & Preprocessing

Phase 2 builds a reproducible, high-integrity preprocessing pipeline for our selected target brand: **`@AppleSupport`**.

### What is a Clean Customer → Brand Pair?
A valid conversational turn pair is defined with strict relational criteria:
- **Parent Tweet**: `inbound == True` (Authentic customer describing a symptom or question)
- **Child Tweet**: `inbound == False` AND `author_id == "AppleSupport"` (Official Apple agent reply)
- **Direct Link**: `child.in_response_to_tweet_id == parent.tweet_id`

Out of 2.81 million tweets in `twcs.csv`, exactly **106,646 raw AppleSupport reply pairs** were extracted.

### Why Raw and Cleaned Text are Both Retained
Both `customer_text_raw` / `brand_reply_raw` and `customer_text_clean` / `brand_reply_clean` are preserved side-by-side:
1. **Full Traceability & Auditability**: In regulated or enterprise NLP environments, you must always be able to inspect the original text to verify that preprocessing did not alter customer intent.
2. **Tokenizer Flexibility**: Different downstream models (e.g. byte-pair encoding in modern LLMs vs. subword tokenization in BERT) may benefit from different representations.

### Why We Clean Conservatively
Aggressive text cleaning (e.g., removing stopwords, lowercasing everything, deleting all punctuation) destroys critical conversational signals:
- **Preserve Entities**: Product names (`iPhone X`, `MacBook Pro`), OS versions (`iOS 11.0.1`), and error codes (`Error 4013`) must remain intact.
- **Preserve Sentiment & Urgency**: Words like *"urgent"*, *"angry"*, *"charged"*, or *"hacked"* are vital for sentiment analysis and escalation routing.
- **Strip Leading Mentions Only**: `@AppleSupport` is removed only at the start of tweets; interior mentions like *"I asked @tim_cook"* are kept.
- **Normalize URLs**: Replaced with `[URL]` tokens rather than deleted.
- **Remove Agent Sign-offs**: Caret signatures (`^AB`, `^JD`) at the end of agent replies are stripped to prevent models from memorizing agent identities.
- **Flag, Don't Silently Drop**: Low-utility rows (e.g. queries < 3 tokens, URL-only, pure DM acknowledgements like *"DM sent"*) receive transparent boolean flags (`is_usable=False`, `exclusion_reason="..."`).

### Why Conversation-Level Splitting Prevents Leakage
Many customer support interactions span **multiple turns** (e.g., Customer inquiry $\to$ Apple question $\to$ Customer follow-up $\to$ Apple resolution).
- If you split rows randomly, Turn 1 might land in `Train` while Turn 2 lands in `Test`.
- The model would then be evaluated on text from conversations it has already memorized during training (**Conversational Data Leakage**).
- **Our Solution**: We trace every tweet back to its `conversation_root_or_group_id` via parent-pointer traversal. All turns belonging to the same conversation thread are assigned to the **exact same split**.
- **Split Proportions (Seed 42)**:
  - **Train**: 80% (82,077 rows across 63,703 groups)
  - **Validation**: 10% (10,357 rows across 7,963 groups)
  - **Test**: 10% (10,279 rows across 7,963 groups)
  - **Group Overlap**: **Zero group overlap was detected using reconstructed observed conversation roots.**
- **Limitation**: Deleted or missing upstream tweets can prevent recovery of a full original Twitter thread. The grouping method protects against overlap within observed conversation fragments, but cannot prove that no unknown upstream context exists outside the dataset scrape.

> [!WARNING]
> **Golden Evaluation Set Warning**: The future 200-row human-labeled golden evaluation benchmark **must be sampled strictly from `applesupport_test.csv`**. Neither the test split nor any golden evaluation rows may ever be included in training corpora, fine-tuning, or vector retrieval indexes.

### How to Run Phase 2 (Windows PowerShell)

Run the preprocessing script by passing the dataset path:
```powershell
cd C:\Users\msiva\Videos\HIVER\hiver_twitter_exploration
python preprocess_applesupport.py "C:\Users\msiva\Videos\HIVER\twcs\twcs.csv"
```

All processed CSVs and the audit report are generated in `data/processed/`:
- `data/processed/applesupport_pairs_all.csv`: Full 106,646 pairs with raw/cleaned text and audit flags.
- `data/processed/applesupport_train.csv`: 82,077 clean, usable training pairs.
- `data/processed/applesupport_validation.csv`: 10,357 clean, usable validation pairs.
- `data/processed/applesupport_test.csv`: 10,279 clean, usable held-out test pairs.
- `data/processed/preprocessing_report.md`: Complete statistical audit and 10 before/after samples.

---

## Phase 3: AppleSupport Intent-Taxonomy Discovery

Phase 3 analyzes customer messages strictly within [`applesupport_train.csv`](file:///c:/Users/msiva/Videos/HIVER/hiver_twitter_exploration/data/processed/applesupport_train.csv) to discover a compact, practical, 8-intent classification taxonomy for our future support agent.

### What is an Intent?
An **intent** represents the primary customer problem, goal, or technical obstacle conveyed in a support interaction. For example:
- *"I forgot my Apple ID password"* $\to$ `account_access_and_apple_id`
- *"My battery drains in 2 hours after installing iOS 11"* $\to$ `software_update_or_os_issue`
- *"I was charged $9.99 for Apple Music after cancelling"* $\to$ `billing_subscription_or_purchase`

### Why We Create Our Own Taxonomy from the Data (Inductive Design)
Generic off-the-shelf support taxonomies (e.g. general ecommerce categories like *Shipping*, *Returns*, *Catalog*) fail in specialized technical domains.
By mining recurring n-grams and diagnostic action patterns directly from 82,077 real AppleSupport training messages, our taxonomy reflects the actual vocabulary, failure modes, and device categories (iOS updates, battery degradation, iCloud sync, Apple ID security) present in the training distribution.

### Why 6–8 Broad Labels are Better than 30 Narrow Labels
In conversational AI, taxonomies with 25–40 hyper-specific classes suffer from severe human annotation disagreement (low Cohen's Kappa) and sparse training data per class. A compact set of **8 mutually understandable categories**:
1. Minimizes boundary confusion between annotators.
2. Yields balanced training samples with statistical power.
3. Provides sufficient resolution to route 95%+ of customer inquiries to dedicated technical troubleshooting workflows.

### Why Taxonomy Discovery Uses Training Data Only
To maintain absolute scientific integrity:
- **`applesupport_test.csv` is strictly isolated and protected**.
- If we used the test set to define categories or identify keywords, we would introduce **Taxonomy Selection Leakage** (designing categories specifically to fit our test distribution).
- The future 200-row human-labeled golden evaluation set will be sampled from the test set *after* this taxonomy is locked down.

### Proposed Exploratory Labels vs. Human-Labeled Golden Labels
- **Proposed Labels (`intent_examples.csv`)**: Rule-assisted taxonomy-design artifacts to illustrate category boundaries during development.
- **Golden Benchmark Labels**: Ground-truth labels independently assigned by human annotators without seeing model predictions or exploratory heuristics.

### The 8 Proposed Intents
1. `software_update_or_os_issue`: iOS/macOS update glitches, install loops, keyboard lag post-update.
2. `device_performance_or_hardware`: Battery drain, charging faults, broken screen, display freezing.
3. `connectivity_and_network`: Wi-Fi disconnects, Bluetooth pairing, cellular data drops, AirDrop.
4. `apps_services_or_icloud`: App Store download errors, iCloud storage/sync, Apple Music, iMessage.
5. `account_access_and_apple_id`: Apple ID login, 2FA codes, password reset, account recovery *(High Escalation Risk)*.
6. `billing_subscription_or_purchase`: Unauthorized charges, subscription cancellation, refunds *(High Escalation Risk)*.
7. `repair_replacement_or_order`: Genius Bar appointments, repair status, warranty/AppleCare+, hardware orders.
8. `other_or_unclear`: Vague complaints, emotional vents, multi-intent without clear primary need.

### How to Run Phase 3 (Windows PowerShell)

Run the intent-discovery engine:
```powershell
cd C:\Users\msiva\Videos\HIVER\hiver_twitter_exploration
python discover_intents.py
```

All taxonomy artifacts are generated in `data/taxonomy/`:
- `data/taxonomy/intent_discovery_sample.csv`: Deterministic review sample (600 customer queries, seed 42).
- `data/taxonomy/intent_taxonomy_v1.md`: Detailed definitions, boundary guidelines, and 40 real examples.
- `data/taxonomy/intent_examples.csv`: Curated balanced training examples mapped to intents.
- `data/taxonomy/ambiguous_or_multi_intent_examples.csv`: Edge cases (security, billing disputes, vague vents).
- `data/taxonomy/intent_discovery_report.md`: Full statistical and linguistic discovery report.

---

## Phase 4: Taxonomy Fixes & Human-Labelled Golden Evaluation Set

Phase 4 establishes an uncontaminated, human-audited evaluation benchmark of **exactly 200 unique AppleSupport customer inquiries** sampled from the protected test partition.

### What is a Golden Set?
A **Golden Set** is a small, pristine, human-audited evaluation dataset that serves as ground-truth reality for model evaluation.
Unlike training data (which can tolerate some noise or weak labels), every single golden row is reviewed and verified by human annotators.

### Why is the Golden Set Different from Training Data?
1. **Isolated Source**: Sourced **strictly from `applesupport_test.csv`**.
2. **Strict Prohibition**: Golden rows must **NEVER** appear in `applesupport_train.csv`, `applesupport_validation.csv`, fine-tuning sets, or vector retrieval indices.
3. **Double-Blind Review**: Labeled independently by two human annotators without seeing historical Apple agent responses or automated suggestions.

### Why Two Humans Label Independently (Double-Blind Annotation)
If a single person annotates data, personal biases and fatigue distort the ground truth.
In our double-blind workflow:
- **Annotator A** and **Annotator B** inspect the same 200 messages in different randomized orders (`annotator_a_blind.csv` and `annotator_b_blind.csv`).
- **No Agent Replies or Predictions**: Annotators see *only* the customer text (`customer_text_clean`). They are not biased by historical Apple agent replies or algorithmic predictions.
- **Disagreement Adjudication**: Any disagreement is isolated into `data/golden/adjudication_sheet.csv` and resolved by a designated human adjudicator.

### What Cohen’s Kappa Means in Simple Language
**Cohen's Kappa ($\kappa$)** measures how much two annotators agree *beyond what would happen by pure chance*:
- If two people randomly guessed categories, they would still agree on some rows by coincidence. Cohen's Kappa mathematically subtracts that chance agreement.
- **$\kappa = 1.0$**: Perfect agreement.
- **$0.60 \le \kappa < 0.80$**: Substantial agreement (reliable benchmark).
- **$\kappa < 0.40$**: Poor agreement (signals that guideline definitions need clarification).

### Zero Fabrication Guarantee
No actual human labels or final evaluation metrics have been fabricated:
- `human_intent`, `human_action`, and `escalation_reason` in `annotator_a_blind.csv` and `annotator_b_blind.csv` are initialized as **completely blank** with status `NEEDS_HUMAN_LABEL`.
- True Cohen's Kappa and evaluation scores will only be computed once human annotators physically complete the annotation sheets.

### How to Run Phase 4 Scripts (Windows PowerShell)

1. **Generate Golden Set & Double-Blind Sheets**:
   ```powershell
   cd C:\Users\msiva\Videos\HIVER\hiver_twitter_exploration
   python scripts/create_golden_set.py
   ```

2. **Verify Strict Cross-Partition Isolation**:
   ```powershell
   python scripts/verify_golden_isolation.py
   ```

3. **Reconcile Completed Annotations & Compute Cohen's Kappa**:
   ```powershell
   python scripts/reconcile_golden_labels.py
   ```

---

## Phase 5: Transparent Weak Labels & Intent Classification Baselines

Phase 5 establishes a transparent, rule-driven **weak-labelling pipeline ("silver labels")** to bootstrap model training experiments while strictly protecting evaluation integrity, alongside two essential baseline classifiers.

### Integrity & Safety Guardrails
1. **Uncontaminated Splits**: Weak labels are generated **exclusively for training (82,077 rows) and validation (10,357 rows)**. Strictly **0 test or golden rows** are ever processed by heuristic labeling.
2. **Annotator Sheets Protected**: Neither `annotator_a_blind.csv` nor `annotator_b_blind.csv` is ever modified or populated with automated suggestions.
3. **No False Ground Truth**: Heuristic labels are explicitly tagged as `silver` / heuristic labels and are never conflated with human annotations.
4. **Headline Metrics Refusal**: `evaluate_completed_golden.py` enforces hardcoded safety gates that refuse to compute headline accuracy, Macro F1, or benchmark metrics until all 200 rows in `data/golden/adjudication_sheet.csv` have `final_status == 'DONE'`.

### What Was Built
1. **Transparent Weak Label Engine (`scripts/create_weak_labels.py`)**:
   - Assigns one of the 8 approved taxonomy categories based on deterministic, audited regex rules.
   - Disambiguates power vs. billing context (e.g. "fully charged battery" vs "charged $4.99").
   - Records the exact rule name (`weak_label_rule`) and confidence (`weak_label_confidence`).
   - Produces `data/weak_labels/applesupport_train_weak_labels.csv`, `applesupport_validation_weak_labels.csv`, and `weak_labeling_report.md`.

2. **Baseline 1: Majority-Class Classifier (`majority_weak_intent`)**:
   - Calculates the empirical distribution mode of the weak training set.
   - **68.36% is the proportion of training rows assigned `other_or_unclear` by the weak-label rules. It is a pseudo-label class distribution, not a real accuracy score or a valid performance floor.**
   - **`other_or_unclear` is a fallback bucket created when keyword rules fail. It is not a clean semantic category, and its sampled training examples may contain many different underlying customer issues.**
   - A model can agree highly with weak labels while merely reproducing the same keyword-rule biases. Final claims require the completed, human-reviewed golden set.

3. **Baseline 2: Keyword-Rule Classifier (`keyword_rule_classifier`)**:
   - Applies the transparent keyword heuristics directly to incoming customer messages.
   - Outputs predicted intent, confidence, matched rule, and a flag indicating whether a strong domain rule matched or defaulted to fallback.
   - Saves predictions separately into `outputs/baselines/keyword_rule_predictions.csv`.

4. **Protected Evaluation Suite (`scripts/evaluate_completed_golden.py`)**:
   - Enforces isolation verification and human adjudication completeness.
   - Computes intent accuracy, Macro F1, confusion matrices, and operational safety metrics (unsafe auto-handles, coverage, selective accuracy) once human adjudication is completed.

5. **Educational Methodology Guide (`docs/baseline_methodology.md`)**:
   - Explains baseline concepts, keyword limitations, circular validation traps, and why human golden ground truth is essential.

### How to Run Phase 5 Scripts (Windows PowerShell)

1. **Generate Transparent Weak Labels (Train & Validation Only)**:
   ```powershell
   cd C:\Users\msiva\Videos\HIVER\hiver_twitter_exploration
   python scripts/create_weak_labels.py
   ```

2. **Run Intent-Classification Baselines (Predictions Only)**:
   ```powershell
   python scripts/run_baselines.py
   ```

3. **Verify Protected Golden Evaluation Safety Gates**:
   ```powershell
   python scripts/evaluate_completed_golden.py
   ```
   *(Note: Refuses to compute headline metrics until human golden adjudication is 100% complete).*

---

## Phase 6: Explainable TF-IDF + Logistic Regression Intent Classifier

Phase 6 implements an interpretable statistical machine learning classifier trained on filtered weak training labels with balanced class weights, feature interpretability, probability ranking, and low-confidence human review gating.

### Model Architecture

```text
Customer text (clean)
         ↓
TF-IDF Vectorizer (1-2 n-grams, 30,000 features, sublinear TF)
         ↓
Logistic Regression (class_weight='balanced', max_iter=2000)
         ↓
Intent Probabilities (Softmax over 8 classes)
         ↓
Predicted Intent + Confidence + Top-3 Candidates + Review Flag
```

> [!WARNING]
> **Prototype Status Warning**:
> The saved model is a **weak-label prototype**, NOT a proven final system.
> - Predicted confidence is a model probability estimate from a weak-label prototype. It is not yet calibrated against human ground truth.
> - A model can agree highly with weak labels while merely reproducing the same keyword-rule biases. Final performance claims require the completed, human-reviewed golden set.
> - `other_or_unclear` is a fallback bucket created when keyword rules fail. It is not a clean semantic category, and its sampled training examples may contain many different underlying customer issues.
> - Because training uses class balancing and a controlled `other_or_unclear` sample (2,000 rows, seed 42), predicted class frequencies are not expected to represent real AppleSupport traffic frequencies.

### What Was Built
1. **Model Training Pipeline (`scripts/train_tfidf_classifier.py`)**:
   - Fits `TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)`.
   - Fits `LogisticRegression(class_weight='balanced', random_state=42)`.
   - Filters explicit intents to high/medium confidence rows (23,760 rows).
   - Deterministically samples exactly 2,000 low-confidence `other_or_unclear` rows (seed 42) as a documented fallback.
   - Saves: `models/tfidf_vectorizer.joblib`, `models/tfidf_logistic_regression.joblib`, `models/model_metadata.json`.

2. **Inference Engine (`scripts/predict_tfidf_classifier.py`)**:
   - Computes intent probabilities, argmax predicted intent, top-3 candidates, and top-3 probabilities.
   - Sets `needs_human_review = True` if `predicted_confidence < 0.60`.

3. **Weak-Label Validation Diagnostics (`scripts/run_validation_diagnostics.py`)**:
   - Evaluates agreement strictly against validation weak labels (never calls agreement "accuracy").
   - Reports agreement stratified by weak-label confidence (91.43% on high, 94.89% on medium, 80.88% on low).
   - Generates `outputs/tfidf/validation_weak_diagnostic_report.md` and `outputs/tfidf/validation_weak_diagnostic_predictions.csv`.

4. **Interpretability & Inspection (`scripts/inspect_tfidf_model.py`)**:
   - Extracts the top 20 positive n-gram coefficients per intent class.
   - Generates `outputs/tfidf/top_features_by_intent.csv`.

5. **Methodology Documentation (`docs/tfidf_model_methodology.md`)**:
   - Beginner-friendly explanations of TF-IDF, bigrams, Logistic Regression, class weighting, confidence estimates, and coefficient interpretability.

### How to Run Phase 6 Scripts (Windows PowerShell)

```powershell
cd C:\Users\msiva\Videos\HIVER\hiver_twitter_exploration

# 1. Train the TF-IDF + Logistic Regression model
python scripts/train_tfidf_classifier.py

# 2. Predict intents on validation inquiries with human review gating
python scripts/predict_tfidf_classifier.py --input data\processed\applesupport_validation.csv --output outputs\tfidf\validation_predictions.csv

# 3. Run validation diagnostics against weak labels
python scripts/run_validation_diagnostics.py

# 4. Inspect top 20 positive features per intent class
python scripts/inspect_tfidf_model.py
```

---

## Phase 7: Historical AppleSupport Dialogue Retrieval Engine

Phase 7 constructs a high-speed, in-memory dialogue retrieval engine over historical customer $\rightarrow$ AppleSupport brand reply pairs strictly sourced from `applesupport_train.csv`. This provides empirical response evidence for future response drafting without data leakage.

### Retrieval Architecture

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
> A vague complaint with nearly identical profanity can receive similarity 1.0 but still provide poor grounding.
> High similarity reflects **stronger lexical similarity only**. High lexical similarity (`similarity >= 0.70`) does **NOT** mean safe grounding or safe auto-handling.
> Historical replies are **evidence of past AppleSupport communication patterns**, NOT proof of a current customer’s account status, policy eligibility, refund approval, or device-specific fact.

### Key Safeguards & Features
1. **Unique Customer Inquiry Indexing**:
   - One searchable document = one unique `customer_tweet_id` (**82,063 unique customer inquiries** indexed from 82,077 raw interaction pairs).
   - For the 14 customer inquiries with multiple AppleSupport replies, unique brand replies are aggregated into structured fields, guaranteeing that top-k results contain distinct historical inquiries.
2. **Strict Partition Isolation (Train-Only)**:
   - Index built **strictly from `data/processed/applesupport_train.csv`**.
   - Cryptographic SHA-256 provenance hash: `ade17dd3b359c08157773e4290f6fa5dc8dd19aa152d5d14417faa389326ad19`.
   - Validation inquiries evaluated strictly as **external queries**; test and golden sets are **never indexed or accessed**.
3. **Thread & Strict Text Exclusion Safeguards**:
   - Dynamically zeroes out candidates sharing the same `customer_tweet_id` or `conversation_root_or_group_id` as the incoming query.
   - Provides strict diagnostic option (`--strict-text-exclusion`) to evaluate retrieval when identical normalized wording is excluded.
4. **Conservative Evidence Threshold (0.30) & Lexical Similarity Bands**:
   - Replaced initial 0.15 threshold with conservative **0.30** (31.64% of validation queries fall below 0.30, triggering escalation).
   - Output categorizes inquiries into four bands: `insufficient_lexical_evidence` (< 0.30), `weak_lexical_evidence` (0.30 - 0.50), `moderate_lexical_evidence` (0.50 - 0.70), and `high_lexical_similarity` (>= 0.70).
5. **Unknown Vocabulary Protection**:
   - Queries containing zero recognized dictionary words safely return `similarity_score = 0.0`, `has_sufficient_historical_evidence = False`, and `retrieval_warning = "no_known_vocabulary"`.
6. **Evidence Sanitization**:
   - All text sanitized before output (@mentions -> `[USER]`, URLs -> `[URL]`).
7. **Human Review Sheet (`outputs/retrieval/retrieval_review_sheet.csv`)**:
   - Contains 30 deterministic validation queries $\times$ top 3 (90 candidate pairs) initialized to `needs_human_relevance_review = True` and `review_status = "NEEDS_HUMAN_REVIEW"`.

### How to Run Phase 7 Scripts (Windows PowerShell)

```powershell
cd C:\Users\msiva\Videos\HIVER\hiver_twitter_exploration

# 1. Build the train-only TF-IDF retrieval index (unique inquiries)
python scripts/build_retrieval_index.py

# 2. Retrieve top-3 historical dialogues for validation inquiries
python scripts/retrieve_similar_dialogues.py --input data\processed\applesupport_validation.csv --output outputs\retrieval\validation_retrieval_examples.csv --top-k 3

# 3. Audit retrieval performance and generate human review sheet
python scripts/audit_retrieval.py

# 4. Audit cross-conversation repeated wording and run strict diagnostic
python scripts/audit_retrieval_text_overlap.py
```

---

## Phase 8: Safe, Historically Grounded Reply Drafting

Phase 8 implements the grounded reply-drafting component. It synthesizes predicted intent, historical response patterns, and deterministic safety rules to construct transparent, auditable draft responses without requiring external LLM APIs.

### Architecture

```text
Intent prediction (Phase 6 TF-IDF classifier)
+ historical retrieval (Phase 7 train-only index)
+ safety restrictions (Phase 8 deterministic filters)
        ↓
Cautious draft reply (with evidence IDs & grounding notes)
        ↓
Next phase (Phase 9): auto-handle or human escalation decision
```

> [!IMPORTANT]
> **Safety & Grounding Principles**:
> - **Constrained templates reduce unsupported claims and make drafts easier to audit. They do not guarantee that every draft is relevant or safe.**
> - **Historical AppleSupport replies are evidence of past interaction patterns, NOT facts about the current customer.**
> - High lexical similarity does **NOT** mean safe grounding or safe auto-handling.
> - No matches to the defined prohibited phrases were found by the automated regex audit. Regex checks can detect selected risky phrases, but they cannot prove every generated draft is relevant, accurate, or safe. Human review remains necessary.
> - No draft requests private credentials, passwords, two-factor codes, full card numbers, or government IDs.
> - No unverified URLs are hard-coded; drafts direct users to official secure channels.

### Key Safeguards & Features
1. **Three Standard Draft Modes**:
   - `restricted_safety`: Safety flags detected (`restricted_draft = True`). Generates brief, neutral secure handoffs.
   - `conservative_no_evidence`: Best similarity $< 0.50$. Uses cautious base intent template without historical troubleshooting adaptation. (Scores between 0.30 and 0.49 represent weak lexical evidence and are barred from modifying draft text).
   - `template_with_historical_pattern`: Best similarity $\ge 0.50$. Synthesizes base intent template with safe historical protocol patterns (e.g., preparing device model and OS version).
2. **High-Risk Domain Detection (`src/reply_safety.py`)**:
   - Scans for account compromise, password recovery, unauthorized payments, refund disputes, legal threats, and self-harm/crisis.
   - Forces `restricted_draft = True` and directs users exclusively to official secure support or crisis lifelines.
3. **Traceable Grounding Metadata**:
   - Every draft outputs `evidence_customer_tweet_ids`, `evidence_brand_tweet_ids`, `best_similarity_score`, `lexical_similarity_band`, `has_sufficient_historical_evidence`, and an explanatory `grounding_note`.
4. **Human Review Sheet (`outputs/replies/reply_human_review_sheet.csv`)**:
   - 30-row deterministic validation sample (seed 42) with blank review columns initialized to `review_status = "NEEDS_HUMAN_REVIEW"`.
5. **Strict Data Protection**:
   - Uses train-only retrieval index and validation queries only.
   - Protected test and golden datasets were **never loaded or accessed**.
   - Zero automated quality, acceptance, or satisfaction scores are reported.

### How to Run Phase 8 Scripts (Windows PowerShell)

```powershell
cd C:\Users\msiva\Videos\HIVER\hiver_twitter_exploration

# 1. Generate grounded draft replies for validation inquiries
python scripts/draft_grounded_replies.py `
  --input data\processed\applesupport_validation.csv `
  --output outputs\replies\validation_drafted_replies.csv `
  --top-k 3

# 2. Run safety audit and produce 30-row human review sheet
python scripts/audit_drafted_replies.py `
  --input outputs\replies\validation_drafted_replies.csv `
  --review-sheet outputs\replies\reply_human_review_sheet.csv `
  --audit-report outputs\replies\reply_safety_audit.md `
  --seed 42
```

---

## Phase 9: Risk-Aware Auto-Handle vs. Human Escalation Routing

Phase 9 implements the final decision-support routing component. It evaluates intent confidence, historical retrieval strength, draft modes, and safety restrictions to make auditable, deterministic routing determinations between **auto-handle candidates** and **human escalations**.

### Architecture

```text
Intent classification (Phase 6)
+ historical retrieval (Phase 7)
+ safe reply drafting (Phase 8)
+ routing policy (Phase 9)
        ↓
Auto-handle candidate or human escalation
+ decision_status: offline_candidate_not_sent
+ primary_reason_code & all_reason_codes
+ decision_explanation
```

> [!IMPORTANT]
> **Safety & Integrity Commitments**:
> - **Offline Prototype Status**: Every decision is marked `decision_status = 'offline_candidate_not_sent'`. No messages are sent to Twitter.
> - **Candidate Meaning**: `auto_handle` designates an *eligible auto-handle candidate*, not deployed or unmonitored automation.
> - **Fail-Safe Escalation**: Any triggered risk flag, high-risk intent, low confidence ($< 0.80$), weak similarity ($< 0.50$), or conservative draft mode forces immediate human escalation.
> - **Strict Reason Priority Order**:
>   `restricted_safety_flag` $\rightarrow$ `high_risk_intent` $\rightarrow$ `other_or_unclear` $\rightarrow$ `uncertain_intent` $\rightarrow$ `insufficient_historical_evidence` $\rightarrow$ `conservative_draft_mode` $\rightarrow$ `eligible_low_risk_case`.
> - **No Accuracy Claims**: Descriptive distributions only. Deciding true routing accuracy requires completed human golden adjudication.

### Population Decision Results on Validation Set (10,357 Inquiries)

- **Total Inquiries Evaluated**: **10,357**
- **Auto-Handle Candidates (`auto_handle`)**: **238 (2.30%)**
- **Escalated to Human Specialist (`escalate`)**: **10,119 (97.70%)**
- **Primary Reason Code Distribution**:
  - `other_or_unclear`: 5,921 (57.17%)
  - `uncertain_intent` ($< 0.80$ conf): 2,662 (25.70%)
  - `insufficient_historical_evidence` ($< 0.50$ sim): 1,073 (10.36%)
  - `high_risk_intent` (account, billing, repair): 389 (3.76%)
  - `eligible_low_risk_case` (`auto_handle`): 238 (2.30%)
  - `restricted_safety_flag`: 74 (0.71%)
- **Intent Security Guarantee**:
  - `account_access_and_apple_id`: **0% auto-handled (100% escalated)**
  - `billing_subscription_or_purchase`: **0% auto-handled (100% escalated)**
  - `repair_replacement_or_order`: **0% auto-handled (100% escalated)**
  - `other_or_unclear`: **0% auto-handled (100% escalated)**
  - Auto-handle candidates occur **strictly** in low-risk technical troubleshooting (`software_update_or_os_issue`: 185, `device_performance_or_hardware`: 19, `connectivity_and_network`: 18, `apps_services_or_icloud`: 16).

### How to Run Phase 9 Scripts (Windows PowerShell)

```powershell
cd C:\Users\msiva\Videos\HIVER\hiver_twitter_exploration

# 1. Run unit test suite (10 test cases including edge cases & CSV string parsing)
python scripts/test_routing_policy.py

# 2. Run support agent decision engine on validation drafts
python scripts/run_support_agent.py `
  --input outputs\replies\validation_drafted_replies.csv `
  --output outputs\agent\validation_agent_decisions.csv `
  --confidence-threshold 0.80 `
  --similarity-threshold 0.50

# 3. Audit routing policy, generate review sheet, and evaluate threshold scenarios
python scripts/audit_routing_policy.py `
  --decisions-input outputs\agent\validation_agent_decisions.csv `
  --drafts-input outputs\replies\validation_drafted_replies.csv `
  --review-sheet outputs\agent\routing_human_review_sheet.csv `
  --audit-report outputs\agent\routing_policy_audit.md `
  --scenarios-report outputs\agent\routing_threshold_scenarios.md `
  --seed 42
```

---

## Phase 10: Human-Reviewed Golden Evaluation, Final Metrics & LLM-as-a-Judge

Phase 10 conducts the final, standardized benchmark evaluation of the offline AppleSupport customer assistant against the protected 200-row human golden set.

### 1. Critical Golden-Label Freeze Rule & Workflow Order

To guarantee scientific validity, eliminate observer bias, and prevent human labels from being altered after model predictions become visible, golden inference is strictly guarded:

```text
1. Complete Annotator A and Annotator B labels (data/golden/annotator_a_blind.csv, annotator_b_blind.csv)
2. Reconcile inter-annotator disagreements (scripts/reconcile_golden_labels.py)
3. Fill final human adjudication labels (data/golden/adjudication_sheet.csv)
4. Set all 200 final_status values to DONE
5. Freeze the final label file (scripts/freeze_golden_labels.py -> golden_labels_freeze_manifest.json)
6. Run blind golden inference (scripts/run_golden_agent_inference.py)
7. Run final metrics evaluation (scripts/run_final_evaluation.py)
8. Run LLM judge and human–judge agreement (scripts/run_llm_judge.py, scripts/evaluate_judge_human_agreement.py)
```

### 2. Golden-Inference Anti-Tamper Guard

`scripts/run_golden_agent_inference.py` and `scripts/run_final_evaluation.py` enforce programmatic guards that **refuse execution** unless:
- `data/golden/golden_labels_freeze_manifest.json` exists.
- Manifest row count equals exactly 200.
- Current `data/golden/adjudication_sheet.csv` SHA-256 checksum matches the manifest digest exactly.
- All 200 rows in `adjudication_sheet.csv` have `final_status == 'DONE'`.

### 3. Intent Benchmark Architecture Comparison

Compares exactly three intent systems against identical human ground truth:
1. **`majority_weak_intent`**: Trivial baseline predicting training mode (`device_performance_or_hardware`).
2. **`keyword_rule_classifier`**: Deterministic domain keyword/regex heuristic classifier.
3. **`tfidf_logistic_regression`**: Main trained ML model (sublinear TF-IDF + balanced Logistic Regression).

> [!IMPORTANT]
> **Macro-F1 Invariant Rule**:
> All Macro-F1 calculations strictly evaluate across all eight fixed taxonomy categories (`software_update_or_os_issue`, `device_performance_or_hardware`, `connectivity_and_network`, `apps_services_or_icloud`, `account_access_and_apple_id`, `billing_subscription_or_purchase`, `repair_replacement_or_order`, `other_or_unclear`). No unobserved categories are dropped from the denominator.

### 4. Operational Routing Metrics

- **Action Accuracy**: Fraction of cases where agent action matches human action.
- **Auto-Handle Coverage**: $\frac{\text{Agent Auto-Handle Count}}{200}$.
- **Selective Accuracy**: Accuracy strictly among cases the agent auto-handled.
- **Unsafe Auto-Handle Count & Rate**: Model auto-handled when human required escalation ($\frac{\text{Unsafe Count}}{\text{Auto-Handle Count}}$).
- **Escalation Precision & Recall**: Positive class is `escalate`.
- **Zero-Automation Edge Cases**: If auto-handle count is 0, report `selective_accuracy = N/A` and `unsafe_auto_handle_rate = N/A`. Do not report them as zero.

> [!WARNING]
> **A conservative system can improve selective accuracy simply by escalating more messages. Automation coverage must always be reported alongside auto-handle quality.**
> 
> **Routing Policy Note**: An auto-handle coverage of 2.3% is a **routing-policy behaviour**, NOT independent evidence of accuracy, safety, or deployment readiness.

### 5. Blinded LLM-as-a-Judge & Human Agreement

- **Blinded LLM Judge (`scripts/run_llm_judge.py`)**:
  - Evaluates customer text, agent action/reasons, draft reply, and retrieved historical snippets across 5 Likert dimensions (1-5) and binary `overall_accept`.
  - Strictly blinded to human final labels, model confidence, and weak training tags.
  - Requires explicit API credentials (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`); fails cleanly without credentials and never invents scores.
- **Stratified Human Agreement (`scripts/create_human_judge_agreement_sheet.py`)**:
  - 40-row stratified sample with blank human scores and `human_review_status = 'NEEDS_HUMAN_REVIEW'`.
- **Agreement Evaluation (`scripts/evaluate_judge_human_agreement.py`)**:
  - Computes exact agreement, quadratic/linear weighted Cohen's Kappa, and binary Kappa.
  - Refuses to report agreement until all 40 human reviews are completed (`human_review_status == 'DONE'`).

### How to Run Phase 10 Scripts (Windows PowerShell)

```powershell
cd C:\Users\msiva\Videos\HIVER\hiver_twitter_exploration

# 1. Verify golden evaluation input integrity
python scripts/verify_final_evaluation_inputs.py

# 2. Freeze completed human golden labels (only after all 200 rows are DONE)
python scripts/freeze_golden_labels.py

# 3. Run blind golden agent inference (guarded by freeze manifest)
python scripts/run_golden_agent_inference.py

# 4. Run final evaluation benchmark & bootstrap CIs (seed 42)
python scripts/run_final_evaluation.py

# 5. Create 40-row stratified human-judge agreement sheet
python scripts/create_human_judge_agreement_sheet.py

# 6. Run blinded Gemini LLM judge (optional qualitative audit)
python scripts/run_llm_judge.py --provider gemini --sample 40

# 7. (Optional) Evaluate human-judge agreement (requires manual completion of 40-row sheet)
python scripts/evaluate_judge_human_agreement.py
```

---

## Certified Final Evaluation Benchmark Results (Phase 10)

Evaluated against the frozen 200-row held-out human golden test set across 1,000 bootstrap resamples (seed 42):

### 1. Quantitative Intent & Operational Metrics

| Evaluation Metric | Final Point Estimate | 95% Bootstrap Confidence Interval | Architectural Meaning |
| :--- | :---: | :---: | :--- |
| **Intent Accuracy** | **59.50%** | `[52.50%, 66.01%]` | Outperforms majority baseline (4.5%) & rule classifier (43.0%) |
| **Macro-F1 (Fixed 8 Classes)** | **66.30%** | `[59.88%, 71.45%]` | Balanced across all 8 fixed categories without dropping rare intents |
| **Escalation Recall** | **98.57%** | — | Safety-first priority: catches 98.6% of human-escalated cases |
| **Auto-Handle Coverage** | **2.00%** | `[0.50%, 4.00%]` | Conservative gating: only 4/200 low-risk, high-confidence cases auto-handled |
| **Selective Accuracy** | **75.00%** | `(0.00, 1.00)` | 3 out of 4 auto-handled interactions were fully accurate |
| **Action Accuracy** | **36.00%** | `[29.00%, 43.00%]` | Strict exact-match agreement with human operational action |

### 2. Blinded Qualitative LLM-as-a-Judge Audit (Gemini)

Evaluated across 40 blinded interaction pairs sampled from the golden test set using Google Gemini (`gemini-flash-lite-latest` at temperature 0.0) with zero fabricated scores:

| Rubric Dimension | Mean Score (1–5) | Key Qualitative Findings |
| :--- | :---: | :--- |
| **Safety & Privacy** | **4.97 / 5.00** | Near-flawless data privacy: zero PII leakage, zero ungrounded commitments |
| **Routing Appropriateness** | **4.62 / 5.00** | High-risk safety routing validated; escalates account access, damage, and billing |
| **Historical Grounding** | **3.90 / 5.00** | Grounded in authentic AppleSupport customer care phrasing and troubleshooting |
| **Clear Next Step** | **3.40 / 5.00** | Direct troubleshooting guidance for actionable hardware/OS symptoms |
| **Relevance** | **3.30 / 5.00** | Directly addresses technical queries; penalizes ambiguous deflections |

* **Audit Verdict**: **40/40 evaluated — 36 ACCEPT (90.0%) / 4 REJECT (10.0%)** (saved in `outputs/final_evaluation/llm_judge_outputs.csv`).
* **Inter-Rater Human Agreement**: Documented as **`PENDING_HUMAN_REVIEW`** in `outputs/final_evaluation/judge_human_agreement_report.md` in strict adherence to evaluation integrity rules (no fabricated human scores).

---

## Interactive Web Application & Demo Procedure

The repository provides a zero-dependency local web server and REST API for real-time customer support agent inference, dialogue retrieval, and golden benchmark inspection:

```powershell
cd C:\Users\msiva\Videos\HIVER\hiver_twitter_exploration

# 1. Start the web application server
python app.py --port 8080

# 2. Open your web browser to:
http://127.0.0.1:8080
```

### Running Automated Regression Tests

```powershell
python -m unittest discover -s tests
```
*All 6 unit and integration test suites pass with 100% verification.*
