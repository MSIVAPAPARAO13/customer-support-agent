# Transparent Weak-Labeling Audit Report (Phase 5)

## 1. Executive Summary & Provenance

- **Train Partition Processed**: 82,077 rows (`applesupport_train_weak_labels.csv`).
- **Validation Partition Processed**: 10,357 rows (`applesupport_validation_weak_labels.csv`).
- **Test & Golden Isolation**: Strictly **0 test or golden rows were processed or labelled**.

> [!IMPORTANT]
> **Explicit Silver-Standard Disclaimer**:
> These weak labels were generated via deterministic keyword heuristics. They **are NOT human ground-truth labels** and must only be used for exploratory model training experiments. All final benchmark evaluations must strictly use the human-adjudicated golden set.

## 2. Weak Intent Distribution (Training Partition)

| Weak Intent Category | Train Row Count | % of Train | Primary Semantic Scope |
| :--- | :--- | :--- | :--- |
| `other_or_unclear` | 56,107 | 68.36% | Derived from keyword heuristics |
| `software_update_or_os_issue` | 16,326 | 19.89% | Derived from keyword heuristics |
| `apps_services_or_icloud` | 3,676 | 4.48% | Derived from keyword heuristics |
| `connectivity_and_network` | 2,457 | 2.99% | Derived from keyword heuristics |
| `device_performance_or_hardware` | 1,952 | 2.38% | Derived from keyword heuristics |
| `account_access_and_apple_id` | 631 | 0.77% | Derived from keyword heuristics |
| `billing_subscription_or_purchase` | 558 | 0.68% | Derived from keyword heuristics |
| `repair_replacement_or_order` | 370 | 0.45% | Derived from keyword heuristics |

## 3. Confidence Distribution

| Confidence Level | Row Count | % of Train | Rationale |
| :--- | :--- | :--- | :--- |
| **LOW** | 58,317 | 71.05% | Explicit keywords vs. general mentions |
| **MEDIUM** | 11,920 | 14.52% | Explicit keywords vs. general mentions |
| **HIGH** | 11,840 | 14.43% | Explicit keywords vs. general mentions |

## 4. Top 15 Triggered Rules

| Rule Name | Trigger Count | % of Train |
| :--- | :--- | :--- |
| `rule_fallback_no_match` | 56,107 | 68.36% |
| `rule_general_os_update` | 9,213 | 11.22% |
| `rule_post_update_regression` | 4,284 | 5.22% |
| `rule_os_version_glitch` | 2,399 | 2.92% |
| `rule_general_ecosystem_mention` | 2,210 | 2.69% |
| `rule_general_wireless_mention` | 1,436 | 1.75% |
| `rule_native_service_mention` | 1,190 | 1.45% |
| `rule_battery_drain_or_degradation` | 1,015 | 1.24% |
| `rule_network_and_cellular_signal` | 668 | 0.81% |
| `rule_hardware_charging_failure` | 609 | 0.74% |
| `rule_explicit_billing_or_refund` | 528 | 0.64% |
| `rule_bluetooth_pairing_failure` | 258 | 0.31% |
| `rule_known_ios11_keyboard_bug` | 240 | 0.29% |
| `rule_warranty_and_applecare` | 218 | 0.27% |
| `rule_os_installation_failure` | 190 | 0.23% |

## 5. De-Identified Examples per Weak Intent (10 per Category)

### Category: `other_or_unclear`

1. (Tweet `697`) "The newest update. I️ made sure to download it yesterday."
   - *Rule*: `rule_fallback_no_match` | *Confidence*: `low`
2. (Tweet `702`) "Tried resetting my settings .. restarting my phone .. all that"
   - *Rule*: `rule_fallback_no_match` | *Confidence*: `low`
3. (Tweet `704`) "This is what it looks like [URL]"
   - *Rule*: `rule_fallback_no_match` | *Confidence*: `low`
4. (Tweet `707`) "I️ have an iPhone 7 Plus and yes I️ do"
   - *Rule*: `rule_fallback_no_match` | *Confidence*: `low`
5. (Tweet `709`) "I️ need answers because it’s annoying 🙃"
   - *Rule*: `rule_fallback_no_match` | *Confidence*: `low`
6. (Tweet `717`) "This is what is happening... [URL]"
   - *Rule*: `rule_fallback_no_match` | *Confidence*: `low`
7. (Tweet `719`) "Tf is wrong with my keyboard @115858"
   - *Rule*: `rule_fallback_no_match` | *Confidence*: `low`
8. (Tweet `721`) "are the call centres closed for the night?"
   - *Rule*: `rule_fallback_no_match` | *Confidence*: `low`
9. (Tweet `723`) "hello are all the lines closed for tonight #help"
   - *Rule*: `rule_fallback_no_match` | *Confidence*: `low`
10. (Tweet `733`) "I’ve got a screenshot saying my #iPhoneX is reserved for the 3rd then an email saying it’s the 18th... what happened?"
   - *Rule*: `rule_fallback_no_match` | *Confidence*: `low`

---

### Category: `software_update_or_os_issue`

1. (Tweet `736`) "Thank you @AppleSupport I updated my phone and now it is even slower and barely works. Thank you for ruining my phone.😤"
   - *Rule*: `rule_post_update_regression` | *Confidence*: `high`
2. (Tweet `735`) "I have the iPhone 6s Plus and just did the most recent update."
   - *Rule*: `rule_post_update_regression` | *Confidence*: `high`
3. (Tweet `745`) "I need the software update urgently. The battery lasts literally half a day 🙍🏼🙁@AppleSupport"
   - *Rule*: `rule_general_os_update` | *Confidence*: `medium`
4. (Tweet `756`) "MY HOME BUTTON DOESN’T WORK #IOS11 @AppleSupport"
   - *Rule*: `rule_general_os_update` | *Confidence*: `medium`
5. (Tweet `765`) "After update #ios1103 no spotify on my lock screen?@AppleSupport"
   - *Rule*: `rule_post_update_regression` | *Confidence*: `high`
6. (Tweet `1761`) "iOS 11 is killing my battery @115858. Fix it."
   - *Rule*: `rule_general_os_update` | *Confidence*: `medium`
7. (Tweet `1772`) "iPhone 6, yes ios11. Checked for updates, none available. I’ve swiped up to close the app several times and I’ve restarted it 2x"
   - *Rule*: `rule_general_os_update` | *Confidence*: `medium`
8. (Tweet `2615`) "Yes, after the update. I Phone 7"
   - *Rule*: `rule_post_update_regression` | *Confidence*: `high`
9. (Tweet `2624`) "It’s been a few days since @115858 made me update my phone’s operating system. Now constantly glitching, hmm 🤔 I’m shocked!"
   - *Rule*: `rule_general_os_update` | *Confidence*: `medium`
10. (Tweet `2631`) "iPhone 7+, iOS 11.1"
   - *Rule*: `rule_general_os_update` | *Confidence*: `medium`

---

### Category: `apps_services_or_icloud`

1. (Tweet `2635`) "Just updated iOS on iPhone7, now iCloud backup greyed out, cannot be turned on, says “Last Backup Never”"
   - *Rule*: `rule_icloud_storage_or_backup` | *Confidence*: `high`
2. (Tweet `8331`) "Oh my god i’m so embarrassing @115858 why do I always send voice recordings in iMessage by accident do you know how embarrassing that is?!"
   - *Rule*: `rule_native_service_mention` | *Confidence*: `medium`
3. (Tweet `9180`) "I bought an iTunes gift card worth £15 a week ago, and the email still hasn’t come into my inbox to tell me the code. Helpppp"
   - *Rule*: `rule_general_ecosystem_mention` | *Confidence*: `low`
4. (Tweet `9847`) "iMessage & snapchat"
   - *Rule*: `rule_native_service_mention` | *Confidence*: `medium`
5. (Tweet `11992`) "your latest iOS update had made my phone slower and apps like iMessage are slow to open and load. What’s up with this?"
   - *Rule*: `rule_native_service_mention` | *Confidence*: `medium`
6. (Tweet `12566`) "It's a shame that Siri can't be used to turn off the timer. Pain to do it with messy hands. Send to the idea file?"
   - *Rule*: `rule_native_service_mention` | *Confidence*: `medium`
7. (Tweet `12572`) "is there a way to opt out of iMessage and FaceTime during initial setup on an iPhone?"
   - *Rule*: `rule_native_service_mention` | *Confidence*: `medium`
8. (Tweet `13197`) "Écran d’iPhone cassé. Impossible à connecter sur iTunes ... des astuces svp ? @AppleSupport"
   - *Rule*: `rule_general_ecosystem_mention` | *Confidence*: `low`
9. (Tweet `14324`) "just purchased iTunes movie on ATV in HD. Stops to buffer and stream every 5 minutes. Very unpleasant experience. Never had issues with @116688 movies. @55 showing 30+ mbps DL🤷🏻‍♂️"
   - *Rule*: `rule_general_ecosystem_mention` | *Confidence*: `low`
10. (Tweet `14337`) "So over @115858 & their awful customer service. Locked out of iCloud for 2 weeks & promised password recovery instructions that never came🖕🏽"
   - *Rule*: `rule_general_ecosystem_mention` | *Confidence*: `low`

---

### Category: `connectivity_and_network`

1. (Tweet `1783`) "if my words even mean a thing to you; I am an iPhone 7 owner and have updated to your latest software and now am having the most dropped calls in history and glitch’s such as apps randomly opening and more ... #iHelp"
   - *Rule*: `rule_network_and_cellular_signal` | *Confidence*: `high`
2. (Tweet `2642`) "still no reliable Bluetooth on my iPhone 8+."
   - *Rule*: `rule_general_wireless_mention` | *Confidence*: `medium`
3. (Tweet `4879`) "I can’t download songs. Progress icon keeps spinning. On both data + WiFi. Stream works though. Can’t add songs to playlists. [URL]"
   - *Rule*: `rule_general_wireless_mention` | *Confidence*: `medium`
4. (Tweet `6934`) "after the new apple updates: the battery life is really go fast and the WiFi is always on! @AppleSupport #newappleupdates"
   - *Rule*: `rule_general_wireless_mention` | *Confidence*: `medium`
5. (Tweet `9840`) "mi iPhone no se puede conectar a la red wifi privadas pero si a las free, que hago 😢"
   - *Rule*: `rule_general_wireless_mention` | *Confidence*: `medium`
6. (Tweet `13933`) "after updating to ios 11.1 I have no service, displays no service. reset networking settings does nothing"
   - *Rule*: `rule_network_and_cellular_signal` | *Confidence*: `high`
7. (Tweet `13949`) "I have updated my WatchOS but cannot stream the radio on LTE (@115911) . Wi Fi (both tethered and direct from watch) works"
   - *Rule*: `rule_network_and_cellular_signal` | *Confidence*: `high`
8. (Tweet `14332`) "I updated my iPhone and now the Bluetooth is constantly on. Why?"
   - *Rule*: `rule_general_wireless_mention` | *Confidence*: `medium`
9. (Tweet `23084`) "I have tried it on different WiFi and cell data. It seems to not work. Then quite some time after I have tired to download it will pop up multiple times to allow it to download then later it downloads out of no where. So it’s strange."
   - *Rule*: `rule_general_wireless_mention` | *Confidence*: `medium`
10. (Tweet `25424`) "Bluetooth connection iPhone X to Infiniti is terrible. Garbled and scratchy cutting in and out >30% of time. Didn't happen W iPhone 6!!"
   - *Rule*: `rule_bluetooth_pairing_failure` | *Confidence*: `high`

---

### Category: `device_performance_or_hardware`

1. (Tweet `767`) "I just need @115858 to do something about the battery life because it sucks ass"
   - *Rule*: `rule_battery_drain_or_degradation` | *Confidence*: `high`
2. (Tweet `1775`) "Battery life just got worst."
   - *Rule*: `rule_battery_drain_or_degradation` | *Confidence*: `high`
3. (Tweet `2616`) "why is my battery life short? I updated to 11.1, my battery is poor. Wife didn’t she likes her battery life"
   - *Rule*: `rule_battery_drain_or_degradation` | *Confidence*: `high`
4. (Tweet `2637`) "Hi, I plugged an adapter straight into the transformer and it still didn't charger. The charger LED does not light up either."
   - *Rule*: `rule_hardware_charging_failure` | *Confidence*: `high`
5. (Tweet `3767`) "I’m no expert in battery life but 40% doesn’t go to 7% in 5minutes"
   - *Rule*: `rule_battery_drain_or_degradation` | *Confidence*: `high`
6. (Tweet `4900`) "I have and my battery life has become even worse"
   - *Rule*: `rule_battery_drain_or_degradation` | *Confidence*: `high`
7. (Tweet `9160`) "I got my phone yesterday and it’s already not charging thanks @115858 🙃"
   - *Rule*: `rule_hardware_charging_failure` | *Confidence*: `high`
8. (Tweet `15323`) "Yo @115858 how come when the IPhone X coming out so soon my phone starts overheating and draining battery when I’m charging it?"
   - *Rule*: `rule_device_overheating` | *Confidence*: `high`
9. (Tweet `18871`) "the screen had green lines on one side and then stopped working altogether. Afraid I can't remember the OS, I expect it was in the latest OS when it broke about 8 weeks ago if hat helps?"
   - *Rule*: `rule_display_or_screen_defect` | *Confidence*: `high`
10. (Tweet `21178`) "please stop making old operating systems extremely slow and buggy when the new phone comes out. I’ll get a new phone eventually but damn it just let me use the one I have now in peace and without freezing every ten seconds"
   - *Rule*: `rule_general_device_instability` | *Confidence*: `medium`

---

### Category: `account_access_and_apple_id`

1. (Tweet `1764`) "Hello, I need some help regarding the region change on my Apple ID"
   - *Rule*: `rule_password_reset_inquiry` | *Confidence*: `high`
2. (Tweet `12586`) "What do you make of this? (Top left corner) I am a US customer, and I think my phone has been hacked. [URL]"
   - *Rule*: `rule_account_security_breach` | *Confidence*: `high`
3. (Tweet `32401`) "Hey @AppleSupport I heard I can change my Apple ID to my @798 email but when I try it says nope. What gives? Spent an hour signing out & in."
   - *Rule*: `rule_password_reset_inquiry` | *Confidence*: `high`
4. (Tweet `32400`) "Yes. I read and did that. I get a message that says I cannot change my Apple ID to an email ending in @798. Which is contrary to the document"
   - *Rule*: `rule_password_reset_inquiry` | *Confidence*: `high`
5. (Tweet `38891`) "how can i reset security questions? i face this error: We don't have sufficient information to reset your security questions."
   - *Rule*: `rule_two_factor_or_verification` | *Confidence*: `high`
6. (Tweet `41556`) "locked out of my iPad. Followed instructions but it hasn’t worked. Please help. Also I want to turn this feature off"
   - *Rule*: `rule_account_lockout` | *Confidence*: `high`
7. (Tweet `41653`) "Hey @AppleSupport either your grammar is going down hill, or someone’s gone #phishing [URL]"
   - *Rule*: `rule_account_security_breach` | *Confidence*: `high`
8. (Tweet `43358`) "locked out of my apple ID bcuz my old phone broke & i had 2 factor authentication on , no one should ever use tht shit @115858"
   - *Rule*: `rule_account_lockout` | *Confidence*: `high`
9. (Tweet `43367`) "Can’t change the Apple ID to @798.com Website stills shows it is not possible. I used this document: [URL]"
   - *Rule*: `rule_password_reset_inquiry` | *Confidence*: `high`
10. (Tweet `44626`) "help I logged out of my Apple ID account and the verification code is sent to a broken phone????"
   - *Rule*: `rule_two_factor_or_verification` | *Confidence*: `high`

---

### Category: `billing_subscription_or_purchase`

1. (Tweet `31482`) "Help!! I need to update the payment method for my iPhone X pre-order. It won’t let me do it online"
   - *Rule*: `rule_explicit_billing_or_refund` | *Confidence*: `high`
2. (Tweet `34703`) "My Q was: How was the delivery date in the cart 2 wks earlier than what’s on my receipt? #ServiceRecovery #SupplyChain #1standlastpreorder"
   - *Rule*: `rule_explicit_billing_or_refund` | *Confidence*: `high`
3. (Tweet `35976`) "You guys over charged me on iTunes @AppleSupport and I want my money back I’m really upset"
   - *Rule*: `rule_unrecognized_card_charge` | *Confidence*: `medium`
4. (Tweet `35979`) "You guys over charged me on music I bought on iTunes and I want my money back I’m mad as hell"
   - *Rule*: `rule_unrecognized_card_charge` | *Confidence*: `medium`
5. (Tweet `39623`) "where is all my music that was in my library? and why do i have to renew my subscription?"
   - *Rule*: `rule_explicit_billing_or_refund` | *Confidence*: `high`
6. (Tweet `47995`) "hi there, I need to inquire something regarding he refund of an item"
   - *Rule*: `rule_explicit_billing_or_refund` | *Confidence*: `high`
7. (Tweet `61323`) "I accidentally bought something is there anyway I can get a refund"
   - *Rule*: `rule_explicit_billing_or_refund` | *Confidence*: `high`
8. (Tweet `67418`) "Why is it that i have been waiting for almost a month for a refund of a returned item i paid cash for in the first place? Where is my money?"
   - *Rule*: `rule_explicit_billing_or_refund` | *Confidence*: `high`
9. (Tweet `87078`) "I completed them but still declined them. All :))"
   - *Rule*: `rule_explicit_billing_or_refund` | *Confidence*: `high`
10. (Tweet `90654`) "excuse me, I'm got a refund confirmation email but nothing happened. Do I have to do anythings?"
   - *Rule*: `rule_explicit_billing_or_refund` | *Confidence*: `high`

---

### Category: `repair_replacement_or_order`

1. (Tweet `2638`) "...cost for sending in for diagnosing a problem if it's out of warranty?"
   - *Rule*: `rule_warranty_and_applecare` | *Confidence*: `high`
2. (Tweet `8437`) "suck! Upgrade phone & I lose my Apple ID. Can’t get new id without old & can’t book Genius bar help appointment without ID!!! #stupid"
   - *Rule*: `rule_genius_bar_appointment` | *Confidence*: `high`
3. (Tweet `9195`) "I bought an iPhone 8 with AppleCare+. Wanting to return the iPhone 8. What happens with the AppleCare+?"
   - *Rule*: `rule_warranty_and_applecare` | *Confidence*: `high`
4. (Tweet `42649`) "For a company that touts Ease of Use and great experiences, @115858 sure makes it hard to get a Genius Bar appointment."
   - *Rule*: `rule_genius_bar_appointment` | *Confidence*: `high`
5. (Tweet `45583`) "#apple #applestore please help got the tip of an aux cable stuck in my #iPhone5 how can I fix this? Will a Genius Bar do it? Thanks!"
   - *Rule*: `rule_genius_bar_appointment` | *Confidence*: `high`
6. (Tweet `53578`) "can I buy Apple care if I got my iPhone from EE and asking as it’s within the time limit to buy?"
   - *Rule*: `rule_warranty_and_applecare` | *Confidence*: `high`
7. (Tweet `61403`) "does apple care cover stolen iPhone?"
   - *Rule*: `rule_warranty_and_applecare` | *Confidence*: `high`
8. (Tweet `84167`) "muy descontento con iPhone 8 Plus y del Apple care no solucionan problemas. No puedo cargar los tonos de llamadas que tenia en el anterior iPhone, respuestas absurdas. La radio calienta muchísimo el terminal,"
   - *Rule*: `rule_warranty_and_applecare` | *Confidence*: `high`
9. (Tweet `90614`) "yet again after another update, my ipod touch will not sync with my laptop and half my artwork is gone. Where can I get help with, and please, don't suggest a genius bar as they were hopeless!"
   - *Rule*: `rule_genius_bar_appointment` | *Confidence*: `high`
10. (Tweet `109050`) "Do I have to buy an iPhone 8 directly from Apple to be eligible for AppleCare+ or can I buy the phone separately and add AC+ afterwards, as long as it's within 60 days of my purchase?"
   - *Rule*: `rule_warranty_and_applecare` | *Confidence*: `high`

---

## 6. Known Rule Collisions, Ambiguities & Weaknesses

1. **Software Update vs. Battery Drain Collision**: When a customer says *'Battery dies fast after iOS 11 update'*, the rule prioritizes `software_update_or_os_issue` because the update is identified as the trigger. However, if the customer simply says *'My battery is dying and I have iOS 11'*, it may misclassify.
2. **Battery Charge vs. Billing Charge Disambiguation**: The rule engine explicitly checks battery power context (`is_battery_or_power_context`). Expressions like *'fully charged'* or *'phone won\'t charge'* are successfully prevented from triggering `billing_subscription_or_purchase`.
3. **Sarcasm and Indirect Phrasing**: Customers expressing frustration through sarcasm (*'Great job Apple, another flawless update'*) are classified into `software_update_or_os_issue` based on keywords, missing the negative sentiment nuances.
