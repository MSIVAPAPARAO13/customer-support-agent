# Real Customer-Support Conversation Samples

This report illustrates the core structural thesis of Phase 1:
> **One row = One tweet**  
> **One conversation = Connected parent-child tweets**  
> **Customer message = Inbound query**  
> **Brand message = Outbound resolution pattern**

---

## 1. Multi-Turn Thread Example (Customer $\leftrightarrow$ Brand)

Demonstrates how a single conversation thread spans multiple individual rows connected by `in_response_to_tweet_id`:

### Turn 1: 👤 Customer (`115712`)
- **Tweet ID**: `3`
- **Timestamp**: `Tue Oct 31 22:08:27 +0000 2017`
- **Message**:
> "@sprintcare I have sent several private messages and no one is responding as usual"

### Turn 2: 🏢 Brand (`sprintcare`)
- **Tweet ID**: `1`
- **Timestamp**: `Tue Oct 31 22:10:47 +0000 2017`
- **Message**:
> "@115712 I understand. I would like to assist you. We would need to get you into a private secured link to further assist."

### Turn 3: 👤 Customer (`115712`)
- **Tweet ID**: `2`
- **Timestamp**: `Tue Oct 31 22:11:45 +0000 2017`
- **Message**:
> "@sprintcare and how do you propose we do that"

---

## 2. Customer $\to$ Brand Reply Pairs (Ground-Truth Training Units)

These paired turns will form the foundation for similarity search, intent classification, and reply drafting in later phases:

### Sample Pair #1 [AppleSupport]
**Customer Query** (Tweet `698` by `115854`):
> "@AppleSupport  https://t.co/NV0yucs0lB"

**Brand Response** (Tweet `696` by `AppleSupport`):
> "@115854 We're here for you. Which version of the iOS are you running? Check from Settings &gt; General &gt; About."

---

### Sample Pair #2 [AppleSupport]
**Customer Query** (Tweet `697` by `115854`):
> "@AppleSupport The newest update. I️ made sure to download it yesterday."

**Brand Response** (Tweet `699` by `AppleSupport`):
> "@115854 Lets take a closer look into this issue. Select the following link to join us in a DM and we'll go from there. https://t.co/GDrqU22YpT"

---

### Sample Pair #3 [AmazonHelp]
**Customer Query** (Tweet `272` by `115770`):
> "amazonのfireTVstickが見れない😢"

**Brand Response** (Tweet `269` by `AmazonHelp`):
> "@115770 こんにちは、アマゾン公式です。Fire TV Stickが見れないというのは、どのような状況でしょうか。一般的なトラブルシューティングを記載したヘルプがございますので、ご参照ください。https://t.co/2pbG55qJ7h ET"

---

### Sample Pair #4 [AmazonHelp]
**Customer Query** (Tweet `271` by `115770`):
> "@AmazonHelp 電話で対応してもらいましたが改良されませんでした。
保証期間も過ぎてるので買い直しになるんでしょうね。"

**Brand Response** (Tweet `273` by `AmazonHelp`):
> "@115770 カスタマーサービスにてお問い合わせ済みとのことで、お手数をおかけいたしました。リプライいただきありがとうございました。ET"

---

### Sample Pair #5 [SpotifyCares]
**Customer Query** (Tweet `850` by `115887`):
> "@SpotifyCares Premium &amp; when i️ have it on shuffle it turns off when the song is done and just plays in order and the repeat lights up but doesn’t repeat"

**Brand Response** (Tweet `848` by `SpotifyCares`):
> "@115887 Hmm. Can you try restarting your device by holding the Sleep/Wake + Volume Down buttons for 10 seconds? Keep us posted /LS"

---

### Sample Pair #6 [SpotifyCares]
**Customer Query** (Tweet `849` by `115887`):
> "@SpotifyCares doesn’t work and i even tried deleting the app"

**Brand Response** (Tweet `851` by `SpotifyCares`):
> "@115887 Could you send us a DM with your account's email address? We'll take a look backstage /CH https://t.co/ldFdZRiNAt"

---

### Sample Pair #7 [Uber_Support]
**Customer Query** (Tweet `771` by `115872`):
> "@115873 you need to correct false charges from a trip in Lexington KY on Saturday. I want my $13 back #lyingdriver #theft"

**Brand Response** (Tweet `768` by `Uber_Support`):
> "@115872 Happy to follow up! Contact us via https://t.co/1xM0TILvAI so we can connect."

---

### Sample Pair #8 [Uber_Support]
**Customer Query** (Tweet `773` by `115874`):
> "@115873 my driver just drove me to the department of air travel instead of the airport.......... rufkm"

**Brand Response** (Tweet `772` by `Uber_Support`):
> "@115874 We're here to help! Send us a note here, https://t.co/WFtPA6RjWt and our team will be in touch."

---

### Sample Pair #9 [Delta]
**Customer Query** (Tweet `611` by `115818`):
> "@DELTA i booked my flight using delta amex card. Checking in now &amp; was being charged for baggage"

**Brand Response** (Tweet `609` by `Delta`):
> "@115818 Glad to check. Pls, DM your confirmation number for assistance.  *QB https://t.co/6iDGBJAc2m"

---

### Sample Pair #10 [Delta]
**Customer Query** (Tweet `792` by `115882`):
> "@Delta why wasn't earlier flight offered when I tried to rebook, not cool at all. Just happened to look at moniter after deplaning."

**Brand Response** (Tweet `790` by `Delta`):
> "@115882 I'm sorry. The earlier flight may not have been available at the time of your scheduled change. *TMT"

---

