# AppleSupport Intent-Taxonomy Discovery Report (Phase 3)

## 1. Executive Summary

- **Source Partition**: Strictly [`data/processed/applesupport_train.csv`](file:///c:/Users/msiva/Videos/HIVER/hiver_twitter_exploration/data/processed/applesupport_train.csv) (82,077 rows).
- **Test Isolation**: `applesupport_test.csv` was **completely untouched and isolated**.
- **Review Sample Created**: 600 unique customer inquiries sampled deterministically with `seed=42`.
- **Final Recommended Intents**: **8 mutually understandable categories**.

> [!IMPORTANT]
> **Taxonomy Status Disclaimer**:
> The proposed intent labels are a **taxonomy-design artifact**, not ground-truth labels. Human annotators must independently apply and validate this taxonomy in the later golden-set phase.

## 2. Quantitative Linguistic Discoveries (Training Split)

### Frequent Technical Unigrams (Excluding Stopwords)
| Word | Occurrences in Train |
| :--- | :--- |
| `phone` | 17,567 |
| `iphone` | 16,193 |
| `ios` | 12,721 |
| `update` | 12,615 |
| `fix` | 11,669 |
| `new` | 7,200 |
| `battery` | 6,795 |
| `since` | 4,713 |
| `app` | 4,172 |
| `time` | 4,002 |
| `updated` | 3,868 |
| `screen` | 3,580 |

### Frequent Customer Bigrams
| Bigram Phrase | Occurrences in Train |
| :--- | :--- |
| `ios 11` | 8,300 |
| `115858 fix` | 1,428 |
| `new update` | 1,421 |
| `battery life` | 1,410 |
| `iphone 6s` | 1,282 |
| `every time` | 1,233 |
| `question mark` | 1,219 |
| `hey 115858` | 1,196 |
| `ios update` | 1,147 |
| `new ios` | 1,075 |
| `iphone plus` | 1,032 |
| `ever since` | 970 |

### Issue Keyword Clusters

> [!NOTE]
> **Exploratory Disclaimer**: Keyword cluster counts below are **exploratory estimates from a reviewed training sample, not final human-labelled class distribution**. They represent lexical pattern occurrences, not ground-truth intent labels.

| Problem Cluster | Customer Queries Matching Keywords | % of Train (Exploratory Match) |
| :--- | :--- | :--- |
| **iOS / Software Update** | 26,728 | 32.56% |
| **Battery / Power / Charge** | 8,241 | 10.04% |
| **Display / Screen / Touch** | 7,559 | 9.21% |
| **Connectivity / Wi-Fi / Bluetooth** | 4,165 | 5.07% |
| **Apple ID / Account / Password** | 2,132 | 2.60% |
| **App Store / iCloud / Services** | 6,997 | 8.52% |
| **Billing / Subscriptions / Refunds** | 2,894 | 3.53% |
| **Store / Repair / Warranty** | 3,903 | 4.76% |

### Top Apple Diagnostic Action Phrases in Replies
| Diagnostic Instruction | Frequency in Brand Replies |
| :--- | :--- |
| `let us know` | 11,471 |
| `send us a dm` | 9,499 |
| `happy to help` | 3,575 |
| `settings > general > about` | 2,580 |
| `which ios version` | 1,585 |
| `take a closer look` | 1,567 |
| `direct message` | 828 |
| `which model` | 390 |
| `reach out via dm` | 204 |
| `what type of device` | 138 |

## 3. Recommended 8-Intent Taxonomy

### 1. `software_update_or_os_issue`
- **Core Purpose**: Issues arising specifically from an iOS, watchOS, or macOS update, installation errors, update freezes, release-day bugs, or OS-level software glitches.
- **Escalation Risk**: No. Usually resolved through standard update troubleshooting (Settings > General > Software Update, or recovery mode via iTunes).

### 2. `device_performance_or_hardware`
- **Core Purpose**: Physical hardware malfunctions, battery health, charging faults, broken screens, overheating, speaker/microphone physical failures, or severe sluggishness.
- **Escalation Risk**: Low to Moderate. Hardware diagnostic steps can be provided, but physical hardware damage requires store referral.

### 3. `connectivity_and_network`
- **Core Purpose**: Issues connecting to Wi-Fi networks, Bluetooth accessories, cellular data / carrier signal, GPS location services, AirDrop, or Personal Hotspot.
- **Escalation Risk**: No. Standard network resets (Reset Network Settings, toggling Airplane Mode, forgetting network) resolve most cases.

### 4. `apps_services_or_icloud`
- **Core Purpose**: Problems with Apple ecosystem applications and cloud services, including App Store downloads, iCloud backup/storage full, Apple Music, iMessage, FaceTime, Photos, or Notes.
- **Escalation Risk**: Low. Clear troubleshooting steps exist (re-signing into iTunes/App Store, checking system status).

### 5. `account_access_and_apple_id`
- **Core Purpose**: Authentication, account security, password recovery, Apple ID locks, two-factor authentication (2FA) verification codes, or compromised/hacked accounts.
- **Escalation Risk**: YES. Highly sensitive. Account recovery and security locks strictly require secure identity verification and human escalation.

### 6. `billing_subscription_or_purchase`
- **Core Purpose**: Monetary transactions, unexpected charges, App Store refund requests, recurring subscription management, in-app purchase receipts, or payment method declines.
- **Escalation Risk**: YES. Involves financial data, refund approvals, and PCI compliance. Automated bots can provide reportaproblem.apple.com links, but disputes require human agents.

### 7. `repair_replacement_or_order`
- **Core Purpose**: Physical hardware repair services, Genius Bar appointments, Apple Store trade-ins, warranty / AppleCare+ coverage status, or new device delivery tracking.
- **Escalation Risk**: Moderate. Appointment booking can be automated via link, but warranty disputes and repair progress tracking require human support.

### 8. `other_or_unclear`
- **Core Purpose**: Messages that lack actionable technical details, vague expressions of frustration, general brand feedback, social media chatter, or multi-issue inquiries without a single discernible focus.
- **Escalation Risk**: YES. Requires a human agent to ask clarifying diagnostic questions to isolate the root problem.

## 4. Ambiguous Boundaries & Failure Analysis

The boundary between `software_update_or_os_issue` and `device_performance_or_hardware` is the most common point of friction. When users complain that battery drain started *immediately* after updating, annotators must adhere to the **Golden Tie-Breaker Rule**: identify whether the update is explicitly blamed as the root trigger.

## 5. Security & Financial Escalation Requirements

- **`account_access_and_apple_id`**: Accounts reporting hacked status, 2FA bypass, or account recovery delays must be escalated immediately to Apple ID Tier-2 security specialists.
- **`billing_subscription_or_purchase`**: Any disputed credit card charges require human verification to comply with financial privacy regulations.

## 6. Suitability for the 200-Row Golden Benchmark

By establishing 8 clean, broad categories instead of 30 narrow, overlapping ones, human annotators will achieve significantly higher inter-annotator agreement (Cohen's Kappa). When we sample the 200-row golden set from `applesupport_test.csv`, each example can be labeled cleanly with low ambiguity.
