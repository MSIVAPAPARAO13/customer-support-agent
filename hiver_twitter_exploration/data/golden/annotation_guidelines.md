# Human Annotation Guidelines (Phase 4: Golden Evaluation Benchmark)

These guidelines govern the double-blind human annotation of the 200 customer support messages in `annotator_a_blind.csv` and `annotator_b_blind.csv`.

---

## 1. Annotation Objectives & Principles

As an annotator, your goal is to evaluate the customer's text **independently and objectively**:
- **Do not guess what Apple agents replied**: Evaluate only what the customer wrote.
- **Select one primary intent** from the 8 standardized categories.
- **Select one operational action** (`auto_handle` vs. `escalate`).
- **Explain escalations**: If selecting `escalate`, provide a short reason in `escalation_reason`.

---

## 2. Intent Label Choices (8 Standardized Classes)

Every row must be assigned exactly one of the following 8 `snake_case` intents:

### 1. `software_update_or_os_issue`
- **Definition**: The primary customer complaint is that installing, updating, or running a specific iOS, macOS, or watchOS version caused the problem or generated an operating system glitch.
- **Inclusion Rules**:
  - Mentions explicit OS updates (`iOS 11`, `iOS 11.1`, `update`, `High Sierra`).
  - Update installation failures, verification errors, or recovery loops during update.
  - OS-level software bugs (e.g. keyboard autotype glitch, UI freezes post-update).
- **Exclusion Rules**:
  - Ongoing battery degradation without attributing it to an update (use `device_performance_or_hardware`).
  - Third-party app-specific bugs unrelated to an OS update (use `apps_services_or_icloud`).
- **De-Identified Examples**:
  1. *"Thank you @AppleSupport I updated my phone and now it is even slower and barely works."*
  2. *"Can you get my iPhone 7plus back on the old iOS please? Battery runs out in half the time since the update."*
- **Common Confusion**: Distinguishing from `device_performance_or_hardware`. Rule: If the user explicitly blames the update as the cause, choose `software_update_or_os_issue`.

---

### 2. `device_performance_or_hardware`
- **Definition**: Physical hardware malfunctions, battery degradation, charging failures, broken screens, overheating, camera/speaker hardware defects, or severe sluggishness without a primary update cause.
- **Inclusion Rules**:
  - Rapid battery drain, battery dying at 20%, phone shutting down randomly.
  - Charging port loose, cable not recognized, phone won't charge.
  - Cracked glass, touchscreen unresponsive, black screen of death.
  - Overheating while idle; speaker crackling or microphone silent.
- **Exclusion Rules**:
  - Inquiring about repair appointments, AppleCare warranty coverage, or replacement costs (use `repair_replacement_or_order`).
  - Bluetooth accessory failing to pair (use `connectivity_and_network`).
- **De-Identified Examples**:
  1. *"I used my phone for 2 minutes and it drains down 8 percent. My battery is ruined."*
  2. *"My screen is completely black and won't turn on even after charging all night."*
- **Common Confusion**: User asks *"My screen is cracked, how much to fix it?"* -> Primary purpose is repair pricing/appointment, so choose `repair_replacement_or_order`.

---

### 3. `connectivity_and_network`
- **Definition**: Inability to connect to Wi-Fi networks, Bluetooth accessories, cellular data / LTE carriers, AirDrop, Personal Hotspot, or GPS navigation.
- **Inclusion Rules**:
  - Wi-Fi disconnecting constantly or greyed out in Settings.
  - Bluetooth will not pair with car stereo, external speaker, or wireless keyboard.
  - "No Service", dropped calls, or cellular data not working.
  - AirDrop failing to discover nearby devices.
- **Exclusion Rules**:
  - Physical damage to external headphones (use `device_performance_or_hardware`).
  - Apple Music streaming errors caused by service outage (use `apps_services_or_icloud`).
- **De-Identified Examples**:
  1. *"Still no reliable Bluetooth connection on my iPhone 8 with my car audio."*
  2. *"My Wi-Fi keeps disconnecting every 5 minutes while all other devices in my house work fine."*
- **Common Confusion**: Cellular carrier billing issues belong in `billing_subscription_or_purchase` (if Apple-billed) or are external to Apple.

---

### 4. `apps_services_or_icloud`
- **Definition**: Issues interacting with Apple native applications and cloud services: App Store downloads, iCloud storage/sync, Apple Music, iMessage, FaceTime, Photos, or Notes.
- **Inclusion Rules**:
  - App Store downloads stuck on "Waiting" or spinning wheel.
  - iCloud storage full warnings despite deleting files; photo library sync stalled.
  - Apple Music playlists disappearing or songs not playing offline.
  - iMessage sending green SMS instead of blue bubbles; FaceTime failing to connect.
- **Exclusion Rules**:
  - Disputing a credit card charge for an App Store purchase (use `billing_subscription_or_purchase`).
  - Forgotten Apple ID password required to log into iCloud (use `account_access_and_apple_id`).
- **De-Identified Examples**:
  1. *"I can’t download songs on Apple Music. Progress icon keeps spinning on both data and Wi-Fi."*
  2. *"My iCloud backup has failed for three days in a row saying there is not enough storage."*
- **Common Confusion**: Distinguishing service bugs from billing disputes. If the customer explicitly demands a refund or queries an unauthorized charge, choose `billing_subscription_or_purchase`.

---

### 5. `account_access_and_apple_id`
- **Definition**: Authentication, account security, password recovery, Apple ID locks, two-factor authentication (2FA) verification codes, or compromised/hacked accounts.
- **Inclusion Rules**:
  - Forgotten Apple ID password, passcode, or security questions.
  - Account locked for security reasons; unable to log into App Store/iCloud.
  - Verification code SMS not arriving on trusted phone number.
  - Account takeover, phishing alerts, or unauthorized recovery email change.
- **Exclusion Rules**:
  - Changing subscription payment method on an already logged-in account (use `billing_subscription_or_purchase`).
  - App Store app crashing after login (use `apps_services_or_icloud`).
- **De-Identified Examples**:
  1. *"I forgot my Apple ID password and the recovery email is an old work account I can't access."*
  2. *"Someone hacked my Apple ID and changed my trusted phone number. Please help!"*
- **Common Confusion**: Account takeover accompanied by fraudulent charges. Rule: High-security alert; choose `account_access_and_apple_id` because account recovery must precede financial remediation.

---

### 6. `billing_subscription_or_purchase`
- **Definition**: Monetary transactions, unexpected charges, App Store refund requests, recurring subscription cancellations, receipts, or payment method declines.
- **Inclusion Rules**:
  - Unrecognized charge from `ITUNES.COM/BILL` on bank statement.
  - Accidental in-app purchase by a child; requesting refund.
  - Canceling Apple Music, iCloud, or third-party recurring subscriptions.
  - Payment method declined or billing address verification errors.
- **Exclusion Rules**:
  - Hardware device shipping or Apple Store order delivery status (use `repair_replacement_or_order`).
  - Inability to download a free app (use `apps_services_or_icloud`).
- **De-Identified Examples**:
  1. *"I was charged $9.99 for Apple Music after I cancelled the free trial last week. I want my money back."*
  2. *"Why does my credit card statement show three separate $4.99 charges from Apple today?"*
- **Common Confusion**: Inquiring about trade-in value or device purchase price belongs in `repair_replacement_or_order`.

---

### 7. `repair_replacement_or_order`
- **Definition**: Physical repair services, Genius Bar appointments, Apple Store trade-ins, warranty / AppleCare+ coverage verification, or hardware order tracking.
- **Inclusion Rules**:
  - Booking or checking a Genius Bar appointment at an Apple Retail Store.
  - Status of a mail-in repair or battery replacement order.
  - AppleCare+ coverage check, deductible questions, or warranty claims.
  - New iPhone / Mac shipping status, tracking number, or delivery delays.
- **Exclusion Rules**:
  - Troubleshooting an issue prior to deciding on repair (use `device_performance_or_hardware`).
  - Digital App Store software returns (use `billing_subscription_or_purchase`).
- **De-Identified Examples**:
  1. *"My iPhone 7 screen is cracked. Can I book a Genius Bar appointment for tomorrow?"*
  2. *"I ordered an iPhone X two weeks ago and the tracking still hasn't updated. Where is my order?"*
- **Common Confusion**: Hardware broken vs. wanting a repair. If the user explicitly asks for service, booking, or cost, choose `repair_replacement_or_order`.

---

### 8. `other_or_unclear`
- **Definition**: Messages that lack actionable technical details, vague expressions of frustration, general brand feedback, social media chatter, or multi-issue inquiries without a single discernible focus.
- **Inclusion Rules**:
  - Vague complaints (*"My phone is acting crazy please help"*).
  - General brand sentiment (*"Apple products have gone downhill"*).
  - Greetings without details (*"Hey AppleSupport are you there?"*).
  - Multi-issue messages where 3+ unrelated topics are mentioned with equal priority.
- **Exclusion Rules**:
  - **Do NOT use simply because a message is angry or emotional.** An angry message with a concrete issue (*"I am furious, my screen is completely dead!"*) belongs in `device_performance_or_hardware`.
- **De-Identified Examples**:
  1. *"Apple is absolute garbage now nothing works properly smh"*
  2. *"Hello can someone please reply to me right now I need assistance"*
- **Common Confusion**: Message mentions taking troubleshooting steps without mentioning the original symptom. Choose `other_or_unclear` because diagnostic clarification is required.

---

## 3. Human Operational Action Labels

Every row must also be assigned an operational routing decision:

```text
auto_handle
escalate
```

### Action Criteria

#### `auto_handle`
Select `auto_handle` when an automated agent can safely provide a high-quality, standardized next step (e.g. self-service URL, diagnostic instructions, knowledge-base article) **without requiring private account credentials, payment lookup, identity verification, or subjective human discretion**.
- *Examples*: Standard iOS update instructions, Wi-Fi network reset steps, battery health check navigation (`Settings > Battery`), Apple Music restart guidance.

#### `escalate`
Select `escalate` when human intervention, authenticated access, or formal escalation is required:
- **Mandatory Escalation Triggers**:
  1. **Security & Identity**: Account takeover, hacked Apple ID, 2FA bypass, lost/stolen device lock.
  2. **Financial Transactions**: Disputed credit card charges, refund requests, billing errors.
  3. **High Urgency / Sentiment**: Threats of legal action, extreme distress, abusive language.
  4. **Multi-Issue Inquiries**: Customer lists multiple unrelated blockers with no clear primary focus.
  5. **Insufficient Context**: Vague inquiries where a human agent must conduct exploratory diagnostic dialogue.

---

## 4. Multi-Intent Handling & Tie-Breaker Rule

1. **Identify Primary Customer Need**:
   - If a customer mentions an update and battery drain (*"My battery drains in 2 hours since updating to iOS 11"*), determine the primary customer objective. The update triggered the symptom $\to$ `software_update_or_os_issue`.
2. **Ambiguous Multi-Intent Fallback**:
   - If a customer raises 2 or more equally critical problems without a single clear focus (*"My phone won't charge, my account is locked, and you billed me twice"*), label as `other_or_unclear` and select `escalate`.
