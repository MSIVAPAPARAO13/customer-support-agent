# AppleSupport Customer Intent Taxonomy (Version 1.0)

## 1. Introduction & Theoretical Foundation

In customer service AI, an **intent** is the primary purpose or underlying technical issue motivating a customer's inquiry. This taxonomy was inductively derived by analyzing **82,077 usable training interactions** from `@AppleSupport`. It defines **8 mutually understandable intents** tailored specifically to Apple's consumer electronics and software ecosystem.

> [!IMPORTANT]
> **The Golden Tie-Breaker Rule**:
> - If a message mentions multiple issues, **label its primary customer need** (the core blocker or explicit question).
> - If multiple issues carry equal weight with no single focus, **label as `other_or_unclear` and flag for human escalation**.

## 2. Taxonomy Overview

> [!NOTE]
> **Prevalence Disclaimer**: All percentages below are **exploratory estimates from a reviewed training sample, not final human-labelled class distribution**.

| Intent Name | Plain-English Summary | Prevalence in Train (Exploratory Estimate) | Human Escalation Risk |
| :--- | :--- | :--- | :--- |
| `software_update_or_os_issue` | Issues arising specifically from an iOS, watchOS, or macOS update... | ~22-26% (Exploratory estimate from reviewed sample) | No. Usually resolved through update diagnostics |
| `device_performance_or_hardware` | Physical hardware malfunctions, battery health, charging faults, ... | ~18-22% (Exploratory estimate from reviewed sample) | Low to Moderate. Hardware diagnostics/repairs |
| `connectivity_and_network` | Issues connecting to Wi-Fi networks, Bluetooth accessories, cellu... | ~5-8% (Exploratory estimate from reviewed sample) | No. Standard network resets |
| `apps_services_or_icloud` | Problems with Apple ecosystem applications and cloud services, in... | ~8-12% (Exploratory estimate from reviewed sample) | Low. Clear troubleshooting steps |
| `account_access_and_apple_id` | Authentication, account security, password recovery, Apple ID loc... | ~3-5% (Exploratory estimate from reviewed sample) | YES. Highly sensitive. Account security |
| `billing_subscription_or_purchase` | Monetary transactions, unexpected charges, App Store refund reque... | ~2-4% (Exploratory estimate from reviewed sample) | YES. Involves financial data / refunds |
| `repair_replacement_or_order` | Physical hardware repair services, Genius Bar appointments, Apple... | ~3-5% (Exploratory estimate from reviewed sample) | Moderate. Appointment booking / warranty |
| `other_or_unclear` | Messages that lack actionable technical details, vague expression... | ~12-16% (Exploratory estimate from reviewed sample) | YES. Requires clarifying questions |

## 3. Comprehensive Intent Definitions & Guidelines

### 1. `software_update_or_os_issue`

**Definition**: Issues arising specifically from an iOS, watchOS, or macOS update, installation errors, update freezes, release-day bugs, or OS-level software glitches.

**What Belongs in This Intent**:
- iOS 11 update failures, installation verification errors
- Bugs introduced post-update (e.g. keyboard letter 'I' glitch)
- Phone frozen on Apple logo or recovery screen during update
- App crashes occurring immediately after updating iOS/macOS
- Operating system navigation and UI stutter

**What Does NOT Belong (Exclusions)**:
- Battery drain complaints where the user simply asks about battery degradation without an update context (assign to device_performance_or_hardware)
- Third-party app-specific bugs unrelated to an OS upgrade (assign to apps_services_or_icloud)
- Forgotten Apple ID passwords (assign to account_access_and_apple_id)

### Explicit Decision Rule: Software Update vs. Hardware/Performance

```text
software_update_or_os_issue:
Use when the main customer complaint is that installing, updating,
or running a specific iOS/macOS/watchOS version caused the problem.

device_performance_or_hardware:
Use when the main customer complaint is battery drain, overheating,
crashing, slowness, restart loops, display problems, or physical
device damage, without a clearly primary update/install cause.
```

#### Real De-Identified Boundary Examples (Distinguishing the Two Intents)

1. **Boundary Case 1** (Tweet `119270`):
   > *"Can you get my iPhone 7plus back on the old iOS please? Battery runs out in half the time, apps now frequently crash."*
   > **Assigned Intent**: `software_update_or_os_issue`
   > **Rationale**: Although battery and crashes are mentioned, the explicit primary customer purpose is rolling back the recent iOS update.

2. **Boundary Case 2** (Tweet `119268`):
   > *"Okay @76099 I used my fucking phone for 2 minutes and it drains it down 8 fucking percent"*
   > **Assigned Intent**: `device_performance_or_hardware`
   > **Rationale**: No update, OS version, or software installation is mentioned; the sole issue is acute battery power loss.

3. **Boundary Case 3** (Tweet `119253`):
   > *"I just updated my phone and suddenly everything takes ages to load wtf @76099 this update sux I hate it fix it bye"*
   > **Assigned Intent**: `software_update_or_os_issue`
   > **Rationale**: The user explicitly links the system slowness to the update ("just updated my phone and suddenly...").

4. **Boundary Case 4** (Tweet `115855`):
   > *"Tried resetting my settings .. restarting my phone .. all that [battery dying]"*
   > **Assigned Intent**: `device_performance_or_hardware`
   > **Rationale**: The core symptom is ongoing battery degradation without attributing it to an OS installation event.

5. **Boundary Case 5** (Tweet `119263`):
   > *"after the 11.0.2 my phone just sucks most of the apps are broken, wifi disconnects frequently"*
   > **Assigned Intent**: `software_update_or_os_issue`
   > **Rationale**: The update "11.0.2" is explicitly identified as the trigger breaking multiple apps and subsystems.

**Requires Human Escalation**: **No. Usually resolved through standard update troubleshooting (Settings > General > Software Update, or recovery mode via iTunes).**

**Real De-Identified Training Examples**:
- *Example 1 (Tweet `697`)*: "The newest update. I️ made sure to download it yesterday."
  - *AppleSupport Response*: "Lets take a closer look into this issue. Select the following link to join us in a DM and we'll go from there. [URL]"
- *Example 2 (Tweet `736`)*: "Thank you @AppleSupport I updated my phone and now it is even slower and barely works. Thank you for ruining my phone.😤"
  - *AppleSupport Response*: "We'd like to help, but we'll need more details. What's happening on your device and which model is it? Do you have iOS 11.1?"
- *Example 3 (Tweet `735`)*: "I have the iPhone 6s Plus and just did the most recent update."
  - *AppleSupport Response*: "To make sure, is iOS 11.1 installed on it currently? Also, any steps tried so far? DM us here: [URL]"
- *Example 4 (Tweet `745`)*: "I need the software update urgently. The battery lasts literally half a day 🙍🏼🙁@AppleSupport"
  - *AppleSupport Response*: "Hi there! What type of device are we working with?"
- *Example 5 (Tweet `756`)*: "MY HOME BUTTON DOESN’T WORK #IOS11 @AppleSupport"
  - *AppleSupport Response*: "Let us help with your Home button. Did this issue start right after iOS 11? Which version of iOS 11 are you running?"

---

### 2. `device_performance_or_hardware`

**Definition**: Physical hardware malfunctions, battery health, charging faults, broken screens, overheating, speaker/microphone physical failures, or severe sluggishness.

**What Belongs in This Intent**:
- Rapid battery drain, sudden battery percentage drops, random shutdowns
- Charging cable not recognized, phone not charging, wireless charge failure
- Cracked glass, unresponsive touchscreen, display lines/flicker
- Overheating device while idle or gaming
- Muffled earpiece, speaker crackling, microphone not picking up voice

**What Does NOT Belong (Exclusions)**:
- Bluetooth headphones not pairing (assign to connectivity_and_network)
- Warranty claims or Genius Bar reservation requests (assign to repair_replacement_or_order)
- Software keyboard bugs (assign to software_update_or_os_issue)

**Confusing Boundary Example**: Customer asks: 'My screen is cracked, can I get it fixed?' -> The customer's primary purpose is getting a repair/service, so assign to repair_replacement_or_order.

**Requires Human Escalation**: **Low to Moderate. Hardware diagnostic steps can be provided, but physical hardware damage requires store referral.**

**Real De-Identified Training Examples**:
- *Example 1 (Tweet `745`)*: "I need the software update urgently. The battery lasts literally half a day 🙍🏼🙁@AppleSupport"
  - *AppleSupport Response*: "Hi there! What type of device are we working with?"
- *Example 2 (Tweet `765`)*: "After update #ios1103 no spotify on my lock screen?@AppleSupport"
  - *AppleSupport Response*: "Thanks for reaching out to us. Are you experiencing the missing app after restating your device?"
- *Example 3 (Tweet `767`)*: "I just need @115858 to do something about the battery life because it sucks ass"
  - *AppleSupport Response*: "We want to help you get your battery life back on track. Please DM and we'll look at it together. [URL]"
- *Example 4 (Tweet `1761`)*: "iOS 11 is killing my battery @115858. Fix it."
  - *AppleSupport Response*: "We're here to help. Which exact iOS version is your device running? This info can be found under Settings > General > About."
- *Example 5 (Tweet `1773`)*: "I just get a white screen and nothing loads. After a short time, it just closes/crashes. Thanks for the reply."
  - *AppleSupport Response*: "Which model do you have and is iOS 11.1 installed on it? Any steps tried so far?"

---

### 3. `connectivity_and_network`

**Definition**: Issues connecting to Wi-Fi networks, Bluetooth accessories, cellular data / carrier signal, GPS location services, AirDrop, or Personal Hotspot.

**What Belongs in This Intent**:
- Wi-Fi disconnects frequently, 'Incorrect Password' for known Wi-Fi
- Bluetooth will not discover or pair with car stereo / external speaker
- No Service / Searching for signal, cellular data dropping
- AirDrop unable to find nearby devices
- Personal Hotspot not connecting to laptop

**What Does NOT Belong (Exclusions)**:
- AirPods physical hardware defect or lost AirPod (assign to device_performance_or_hardware or repair_replacement_or_order)
- iMessage won't send due to Apple ID block (assign to apps_services_or_icloud or account_access_and_apple_id)
- Home router hardware defects (carrier/ISP issue)

**Confusing Boundary Example**: Customer cannot stream music: If due to Wi-Fi dropping, connectivity_and_network; if Apple Music server is down, apps_services_or_icloud.

**Requires Human Escalation**: **No. Standard network resets (Reset Network Settings, toggling Airplane Mode, forgetting network) resolve most cases.**

**Real De-Identified Training Examples**:
- *Example 1 (Tweet `2642`)*: "still no reliable Bluetooth on my iPhone 8+."
  - *AppleSupport Response*: "Thanks for reaching out. DM which Bluetooth device you are trying to connect to so we can help. [URL]"
- *Example 2 (Tweet `4879`)*: "I can’t download songs. Progress icon keeps spinning. On both data + WiFi. Stream works though. Can’t add songs to playlists. [URL]"
  - *AppleSupport Response*: "We want to help. Does restarting help at all? DM us which iOS version you're on and we'll continue support there. [URL]"
- *Example 3 (Tweet `6934`)*: "after the new apple updates: the battery life is really go fast and the WiFi is always on! @AppleSupport #newappleupdates"
  - *AppleSupport Response*: "We'd love to help. Let's work on this together in DM. Start by letting us know what kind of device you are using. [URL]"
- *Example 4 (Tweet `2625`)*: "my 7+ refuses to stay on WiFi calling ever since ios 11 and im seriously so over the customer service -___-"
  - *AppleSupport Response*: "We want to help you. DM us, and we can look into what's going on with Wi-Fi calling on your iPhone. [URL]"
- *Example 5 (Tweet `9840`)*: "mi iPhone no se puede conectar a la red wifi privadas pero si a las free, que hago 😢"
  - *AppleSupport Response*: "We offer support via Twitter in English. Get help in Spanish here: [URL] or join [URL]"

---

### 4. `apps_services_or_icloud`

**Definition**: Problems with Apple ecosystem applications and cloud services, including App Store downloads, iCloud backup/storage full, Apple Music, iMessage, FaceTime, Photos, or Notes.

**What Belongs in This Intent**:
- Apps stuck on 'Waiting' or failing to download from App Store
- iCloud storage full alerts, backup not completing, photo library sync stalled
- Apple Music playlists missing, offline songs not playing
- iMessage green bubbles, FaceTime calls failing to connect
- Notes or Reminders not syncing across devices

**What Does NOT Belong (Exclusions)**:
- Billing or charges for App Store purchases (assign to billing_subscription_or_purchase)
- Forgotten Apple ID password to log into iCloud (assign to account_access_and_apple_id)
- Third-party app developer complaints unrelated to Apple services

**Confusing Boundary Example**: Customer says: 'I paid for iCloud storage but it still says full.' The primary problem is the service storage recognition (apps_services_or_icloud), unless customer explicitly disputes the charge (billing_subscription_or_purchase).

**Requires Human Escalation**: **Low. Clear troubleshooting steps exist (re-signing into iTunes/App Store, checking system status).**

**Real De-Identified Training Examples**:
- *Example 1 (Tweet `2635`)*: "Just updated iOS on iPhone7, now iCloud backup greyed out, cannot be turned on, says “Last Backup Never”"
  - *AppleSupport Response*: "Check out this link: [URL] Send us a DM if it doesn't help. [URL]"
- *Example 2 (Tweet `4890`)*: "whenever I try to see the photos that I’ve just taken my #iphone6 shows me this. I’ve reset it, I’ve restored it... help? [URL]"
  - *AppleSupport Response*: "We know how important it is to be able to view your photos. Are you able to see older photos on your iPhone?"
- *Example 3 (Tweet `8331`)*: "Oh my god i’m so embarrassing @115858 why do I always send voice recordings in iMessage by accident do you know how embarrassing that is?!"
  - *AppleSupport Response*: "Are you accidentally recording audio message or is this happening on its own? Let us know in a DM, we’ll communicate there: [URL]"
- *Example 4 (Tweet `9847`)*: "iMessage & snapchat"
  - *AppleSupport Response*: "Thanks for letting us know. Send us a DM and we'll do everything we can to get your keyboard back on track. [URL]"
- *Example 5 (Tweet `11665`)*: "My backup was stored to my MacBook Pro which battery died 2 months ago."
  - *AppleSupport Response*: "Let's take a look at this together in DM. [URL]"

---

### 5. `account_access_and_apple_id`

**Definition**: Authentication, account security, password recovery, Apple ID locks, two-factor authentication (2FA) verification codes, or compromised/hacked accounts.

**What Belongs in This Intent**:
- Forgotten Apple ID password or security questions
- Account locked for security reasons
- Not receiving two-factor verification SMS or trusted device code
- Account recovery request status or delays
- Suspected unauthorized access, account takeover, or phishing alerts

**What Does NOT Belong (Exclusions)**:
- Billing inquiries for an active Apple ID (assign to billing_subscription_or_purchase)
- Activation Lock on a secondhand device purchased without original credentials (assign to account_access_and_apple_id if credentials known; otherwise escalation)
- General app login issues (e.g. Netflix login failure)

**Confusing Boundary Example**: Customer says: 'Someone charged my Apple ID, I think I was hacked!' High-security alert. Primary is account_access_and_apple_id with immediate billing dispute secondary.

**Requires Human Escalation**: **YES. Highly sensitive. Account recovery and security locks strictly require secure identity verification and human escalation.**

**Real De-Identified Training Examples**:
- *Example 1 (Tweet `1764`)*: "Hello, I need some help regarding the region change on my Apple ID"
  - *AppleSupport Response*: "This article should help with that: [URL]"
- *Example 2 (Tweet `8297`)*: "Hello there i need help with the use of an administartors name and password on my imac"
  - *AppleSupport Response*: "Hi! We're happy to help. Join us in DM with the macOS version installed on your iMac and more details about what's going on. [URL]"
- *Example 3 (Tweet `8437`)*: "suck! Upgrade phone & I lose my Apple ID. Can’t get new id without old & can’t book Genius bar help appointment without ID!!! #stupid"
  - *AppleSupport Response*: "We're happy you reached out so we can find you the info you need. If you forgot your Apple ID, go to the Apple ID account page: [URL] From here, click "Forgot Apple ID or password?" then click "look it up." From here, follow the on screen prompts."
- *Example 4 (Tweet `8551`)*: "ya.lately my screen was disfunction for touch after its been restart. i couldnt enter passcode yet homebutton still figureout my fingerprint"
  - *AppleSupport Response*: "For security, the passcode is always required even if you have Touch ID turned on. Are you still having touch screen issues?"
- *Example 5 (Tweet `8557`)*: "is that safe if we disable passcode for unlock iphone? i mean,is the fingerprint still worked out? i think dat would be enough"
  - *AppleSupport Response*: "We want to make sure we understand the issue. Are you wanting to disable the passcode and still use Touch ID?"

---

### 6. `billing_subscription_or_purchase`

**Definition**: Monetary transactions, unexpected charges, App Store refund requests, recurring subscription management, in-app purchase receipts, or payment method declines.

**What Belongs in This Intent**:
- Unrecognized charge on bank statement from 'ITUNES.COM/BILL'
- Accidental in-app purchase by a child, requesting refund
- Canceling Apple Music, iCloud, or third-party recurring subscriptions
- Payment method declined in App Store or iTunes
- Gift card redemption errors or balance inquiry

**What Does NOT Belong (Exclusions)**:
- Physical Apple Store device purchase shipments or tracking (assign to repair_replacement_or_order)
- Apple ID locked due to security reasons (assign to account_access_and_apple_id)
- Inability to download a free app (assign to apps_services_or_icloud)

**Confusing Boundary Example**: Customer says: 'I was charged for an app that crashed immediately.' Root cause is software crash, but immediate customer goal is getting money back. Primary is billing_subscription_or_purchase.

**Requires Human Escalation**: **YES. Involves financial data, refund approvals, and PCI compliance. Automated bots can provide reportaproblem.apple.com links, but disputes require human agents.**

**Real De-Identified Training Examples**:
- *Example 1 (Tweet `22988`)*: "Trying to download an app to a family member's phone that I purchased as the family group organizer. Being asked to purchase it, but I don't want to be charged again. Suggestions? [URL]"
  - *AppleSupport Response*: "We can help. As long as this app supports Family Sharing, you'll be able to download it by following the steps in our article. Give them a shot, and let us know if this continues: [URL] [URL]"
- *Example 2 (Tweet `23077`)*: "ok not sure why this is happening but I need assistance charged my iPhone X (purchased yesterday) wirelessly and reached for it and this was on the screen, I could probably fry an egg [URL]"
  - *AppleSupport Response*: "We'll be happy to take a look into this with you. Could you please send us a DM with what country you're located in so we can get started? [URL]"
- *Example 3 (Tweet `23100`)*: ", why do I keep getting charged over £20 for anything I buy from iTunes? Bought an album for £4.99 and got charged £20.67. Same thing happened a couple of weeks ago to. Why?! [URL]"
  - *AppleSupport Response*: "Our iTunes Store team would be happy to look into this with you. Reach out to them here: [URL]"
- *Example 4 (Tweet `31436`)*: "trying to cancel an online order, each time I login i'm told I don't have access to the order."
  - *AppleSupport Response*: "Hello. Our Sales Support team will be happy to assist you with this issue here: [URL]"
- *Example 5 (Tweet `32395`)*: "Dear @115858, what you've done to my iPhone 6 w/ IOS 11 should be a crime. Thanks 4 updating 2 useless. Makes me want 2 cancel my X pre-order"
  - *AppleSupport Response*: "We're happy to assist. Which iOS 11 version are you currently running? Go to Settings > General > About for the version."

---

### 7. `repair_replacement_or_order`

**Definition**: Physical hardware repair services, Genius Bar appointments, Apple Store trade-ins, warranty / AppleCare+ coverage status, or new device delivery tracking.

**What Belongs in This Intent**:
- Booking or rescheduling a Genius Bar appointment at a local Apple Store
- Checking status of a device sent in for mail-in repair
- AppleCare+ coverage verification, deductible inquiries, warranty claims
- New iPhone or Mac order tracking, delivery delays from Apple online store
- Cost inquiries for official screen or battery replacement

**What Does NOT Belong (Exclusions)**:
- Software troubleshooting before attempting repair (assign to device_performance_or_hardware or software_update_or_os_issue)
- Third-party non-Apple repair shop complaints
- Digital App Store purchase returns (assign to billing_subscription_or_purchase)

**Confusing Boundary Example**: Customer says: 'My iPhone won't turn on, is it covered by warranty?' Primary goal is warranty service/repair (repair_replacement_or_order).

**Requires Human Escalation**: **Moderate. Appointment booking can be automated via link, but warranty disputes and repair progress tracking require human support.**

**Real De-Identified Training Examples**:
- *Example 1 (Tweet `2638`)*: "...cost for sending in for diagnosing a problem if it's out of warranty?"
  - *AppleSupport Response*: "We can help. Let’s start with the following steps to see if we can save you a trip in for service: [URL]"
- *Example 2 (Tweet `8420`)*: "Chat doesn’t seem to work in Germany. On the order page I cannot leave comments …"
  - *AppleSupport Response*: "If you are unable to access these resources, let's get you connected with our local AppleCare Support team to get you to our Online Support group: [URL]"
- *Example 3 (Tweet `8422`)*: "Dear @115858, I have an online order which I never received - who can I talk to?"
  - *AppleSupport Response*: "We want to help out. Look here to talk with our Online Support: [URL]"
- *Example 4 (Tweet `8437`)*: "suck! Upgrade phone & I lose my Apple ID. Can’t get new id without old & can’t book Genius bar help appointment without ID!!! #stupid"
  - *AppleSupport Response*: "We're happy you reached out so we can find you the info you need. If you forgot your Apple ID, go to the Apple ID account page: [URL] From here, click "Forgot Apple ID or password?" then click "look it up." From here, follow the on screen prompts."
- *Example 5 (Tweet `9195`)*: "I bought an iPhone 8 with AppleCare+. Wanting to return the iPhone 8. What happens with the AppleCare+?"
  - *AppleSupport Response*: "Thanks for contacting us. We'd like to gather some info to help out. Let's meet in DM. [URL]"

---

### 8. `other_or_unclear`

**Definition**: Messages that lack actionable technical details, vague expressions of frustration, general brand feedback, social media chatter, or multi-issue inquiries without a single discernible focus.

> [!IMPORTANT]
> **Strict Usage Rule for `other_or_unclear`**:
> - **Use only when there is insufficient information to identify a primary problem, or when multiple issues are equally important.**
> - **Do NOT use it merely because the message is emotional or angry.** An angry tweet that says *"I am furious, my screen is completely shattered and unresponsive!"* belongs in `device_performance_or_hardware`, NOT `other_or_unclear`.

**What Belongs in This Intent**:
- Vague complaints ('My phone is acting crazy please help')
- General sentiment/brand feedback ('Apple has gone downhill since Steve Jobs')
- Greetings without problem details ('Hey AppleSupport are you there?')
- Multi-issue messages where 3+ unrelated topics are mentioned with equal priority
- Feature requests or non-support questions ('When is the new iPhone coming out?')

**What Does NOT Belong (Exclusions)**:
- A message with slight vagueness but a clear keyword like 'battery' or 'update' (classify into the specific technical category)
- Angry messages that still describe a concrete issue (classify by the issue)

**Confusing Boundary Example**: Customer says: 'I hate this update, my battery sucks, my wifi is dead, and you charged me.' Equal weight across 4 domains with no single primary focus. Default to other_or_unclear and escalate.

**Requires Human Escalation**: **YES. Requires a human agent to ask clarifying diagnostic questions to isolate the root problem.**

**Real De-Identified Training Examples**:
- *Example 1 (Tweet `709`)*: "I️ need answers because it’s annoying 🙃"
  - *AppleSupport Response*: "We'd like to look into this with you. Which model do you have and is iOS 11.1 installed? Any steps tried so far?"
- *Example 2 (Tweet `723`)*: "hello are all the lines closed for tonight #help"
  - *AppleSupport Response*: "What's going on? We're hapy to help if we can. [URL]"
- *Example 3 (Tweet `1755`)*: "Why does my I️ not work ?! @115858 please fix this!!!"
  - *AppleSupport Response*: "We'd like to look into this with you. Please DM us your current iOS version and your iPhone model. [URL]"
- *Example 4 (Tweet `1764`)*: "Hello, I need some help regarding the region change on my Apple ID"
  - *AppleSupport Response*: "This article should help with that: [URL]"
- *Example 5 (Tweet `1770`)*: "why is my home sharing not working and how do i fix it"
  - *AppleSupport Response*: "We're happy to help. Has Home Sharing worked with these devices before?"

---

