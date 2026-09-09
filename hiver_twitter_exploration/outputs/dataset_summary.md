# Twitter Customer Support Dataset Exploration (Phase 1 Summary)

## 1. Dataset Overview & File Verification

- **File Name**: `twcs.csv`
- **Absolute Path**: `C:\Users\msiva\Videos\HIVER\twcs\twcs.csv`
- **File Size**: 492.58 MB (516,508,641 bytes)
- **Total Records (Rows)**: 2,811,774
- **Total Columns**: 7
- **Duplicate Rows (by Tweet ID)**: 0
- **Load & Inspection Time**: 11.71 seconds

## 2. Column Schema & Data Types

| Column Name | Detected Semantic Role | Inferred Type | Missing Values | Missing % |
| :--- | :--- | :--- | :--- | :--- |
| `tweet_id` | **tweet_id** | `str` | 0 | 0.00% |
| `author_id` | **author_id** | `str` | 0 | 0.00% |
| `inbound` | **inbound** | `bool` | 0 | 0.00% |
| `created_at` | **created_at** | `str` | 0 | 0.00% |
| `text` | **text** | `str` | 0 | 0.00% |
| `response_tweet_id` | **response_tweet_id** | `str` | 1,040,629 | 37.01% |
| `in_response_to_tweet_id` | **in_response_to_tweet_id** | `str` | 794,335 | 28.25% |

### Column Explanations (Beginner-Friendly)
1. `tweet_id`: The unique numeric identifier assigned by Twitter to each tweet.
2. `author_id`: The author. For brands, this is their recognizable handle (e.g., `AppleSupport`, `AmazonHelp`). For customers, this is an anonymized numeric ID (e.g., `105834`) to protect user privacy.
3. `inbound`: A boolean flag (`True`/`False`). `True` means the tweet is an incoming customer inquiry. `False` means it is an outgoing brand agent response.
4. `created_at`: The UTC timestamp when the tweet was posted.
5. `text`: The raw 280-character (or 140-character) message content, containing user questions, error descriptions, or agent replies.
6. `response_tweet_id`: The ID(s) of any tweets that responded to this tweet. **Important Note**: This field can contain multiple response IDs separated by commas or spaces.
7. `in_response_to_tweet_id`: The ID of the parent tweet that this tweet replies to. This is the **primary, most reliable key** for connecting a brand reply back to the originating customer problem.

## 3. Data Integrity & Missingness Rationale

Notice that `response_tweet_id` and `in_response_to_tweet_id` have missing values. This is **expected by design**:
- An initial customer question starting a thread has no parent, so `in_response_to_tweet_id` is empty (`NaN`).
- A closing response from an agent often receives no further response from the user, so `response_tweet_id` is empty (`NaN`).
- Crucially, `text`, `tweet_id`, `author_id`, and `inbound` have **zero missing values** across all rows.

## 4. First 5 Rows (Preview)

| tweet_id | author_id | inbound | text (truncated) | in_response_to_tweet_id |
| :--- | :--- | :--- | :--- | :--- |
| 1 | sprintcare | False | @115712 I understand. I would like to assist yo... | 3 |
| 2 | 115712 | True | @sprintcare and how do you propose we do that | 1 |
| 3 | 115712 | True | @sprintcare I have sent several private message... | 4 |
| 4 | sprintcare | False | @115712 Please send us a Private Message so tha... | 5 |
| 5 | 115712 | True | @sprintcare I did. | 6 |

## 5. Dialogue Volume & Reply Pair Aggregates

- **Total Customer Inbound Tweets (`inbound=True`)**: 1,537,843 (54.7%)
- **Total Brand Outbound Tweets (`inbound=False`)**: 1,273,931 (45.3%)
- **Total Valid Customer $\to$ Brand Reply Pairs**: 1,261,888

### Top 15 Brands by Outbound Volume & Valid Reply Pairs

| Brand Handle | Outbound Tweets | Valid Reply Pairs | Coverage % | Avg Customer Words | Avg Brand Words | Short Replies (<4 words) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AmazonHelp` | 169,840 | 168,814 | 99.4% | 19.3 | 19.8 | 5.3% |
| `AppleSupport` | 106,860 | 106,646 | 99.8% | 18.8 | 22.7 | 0.2% |
| `Uber_Support` | 56,270 | 56,160 | 99.8% | 21.1 | 19.2 | 0.0% |
| `SpotifyCares` | 43,265 | 43,092 | 99.6% | 17.9 | 22.1 | 0.1% |
| `Delta` | 42,253 | 42,114 | 99.7% | 19.7 | 17.9 | 3.5% |
| `Tesco` | 38,573 | 38,468 | 99.7% | 20.3 | 25.9 | 0.2% |
| `AmericanAir` | 36,764 | 36,531 | 99.4% | 20.6 | 18.4 | 0.8% |
| `TMobileHelp` | 34,317 | 34,215 | 99.7% | 19.5 | 21.5 | 0.2% |
| `comcastcares` | 33,031 | 32,921 | 99.7% | 18.8 | 24.0 | 0.2% |
| `British_Airways` | 29,361 | 29,290 | 99.8% | 21.9 | 21.7 | 0.2% |
| `SouthwestAir` | 28,977 | 28,828 | 99.5% | 18.9 | 20.4 | 1.0% |
| `VirginTrains` | 27,817 | 27,416 | 98.6% | 19.3 | 15.0 | 2.5% |
| `Ask_Spectrum` | 25,860 | 25,617 | 99.1% | 17.9 | 23.8 | 0.3% |
| `XboxSupport` | 24,557 | 23,235 | 94.6% | 19.3 | 20.0 | 0.2% |
| `sprintcare` | 22,381 | 22,209 | 99.2% | 18.1 | 21.2 | 0.3% |

## 6. Top 3 Candidate Brands for Phase 2

### 1. AppleSupport (Excellent (Recommended Primary Target))
- **Domain**: Consumer Electronics & Software (iOS, macOS, iPhone, Apple ID)
- **Valid Reply Pairs**: 106,646
- **Text Quality**: Avg 18.8 words per customer query, avg 22.7 words per brand reply, only 0.2% short replies.
- **Strategic Value**: Massive volume (>106k clean reply pairs) with highly focused technical dialogue. Customer queries describe concrete software/hardware symptoms (battery drain, update errors, iCloud sync), and agent replies follow disciplined troubleshooting patterns. Low noise, ideal for technical retrieval.

### 2. AmazonHelp (Strong (High-Volume Enterprise Domain))
- **Domain**: E-Commerce, Shipping, Returns & Prime Services
- **Valid Reply Pairs**: 168,814
- **Text Quality**: Avg 19.3 words per customer query, avg 19.8 words per brand reply, only 5.3% short replies.
- **Strategic Value**: Largest dataset volume (>168k valid pairs). Covers structured transaction intents (late packages, refunds, cancellations, digital orders). Extremely realistic for enterprise support workflows, though with a higher frequency of canned DM requests.

### 3. SpotifyCares (Strong (Focused Software/App Troubleshooting))
- **Domain**: Digital Media, Streaming & Audio Apps
- **Valid Reply Pairs**: 43,092
- **Text Quality**: Avg 17.9 words per customer query, avg 22.1 words per brand reply, only 0.1% short replies.
- **Strategic Value**: Substantial volume (>43k pairs) with rich interactive troubleshooting. Discussions feature specific device models, app versions, Bluetooth issues, and playlist caching. Shows higher multi-turn conversational back-and-forth than standard airline or telecom accounts.

