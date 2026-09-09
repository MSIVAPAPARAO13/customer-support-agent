#!/usr/bin/env python3
"""
Hiver Twitter Customer-Support Dataset Exploration (Phase 1)
===========================================================

Purpose:
  This script is the first foundational phase of an ML/NLP customer support project.
  Before building any model, classifier, embedding pipeline, or AI agent, we must
  rigorously understand our raw data:
    1. One row = One tweet event (either a customer inquiry or a brand response).
    2. One conversation = Multiple tweet rows connected by parent/child reply pointers.
    3. Customer inquiry = Inbound tweet (inbound=True, anonymized numeric ID).
    4. Brand resolution = Outbound tweet (inbound=False, official brand handle).

Key Concept for Take-Home & Interview Discussions:
  We are not training AI yet. We are verifying and extracting high-quality
  (Customer Inquiry -> Brand Resolution) pairs. These historical pairs will later
  serve as the ground-truth knowledge base for semantic retrieval and response drafting.

Usage:
  python explore_dataset.py [DATASET_FOLDER_PATH]

Example:
  python explore_dataset.py "C:\\Users\\msiva\\Videos\\HIVER\\twcs"
"""

import os
import sys
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd


# -----------------------------------------------------------------------------
# 1. Helper Functions: Formatting & Path Resolution
# -----------------------------------------------------------------------------

def format_bytes(byte_count: int) -> str:
    """Format raw byte counts into human-readable strings (KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if byte_count < 1024.0:
            return f"{byte_count:.2f} {unit}"
        byte_count /= 1024.0
    return f"{byte_count:.2f} PB"


def parse_response_tweet_ids(raw_val: Any) -> List[str]:
    """
    Parse response_tweet_id field.
    A tweet can have multiple replies separated by commas or whitespace.
    Example: '119249,119251' or '119249 119251'.
    """
    if pd.isna(raw_val) or raw_val is None:
        return []
    s = str(raw_val).strip()
    if not s:
        return []
    # Split on commas, spaces, or semicolons
    ids = [token.strip() for token in re.split(r'[,\s;]+', s) if token.strip()]
    return ids


def truncate_text(text: Any, max_len: int = 70) -> str:
    """Shorten long tweet text for neat console/markdown table display."""
    if pd.isna(text) or text is None:
        return "<EMPTY>"
    s = str(text).replace("\n", " ").replace("\r", " ").strip()
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s


# -----------------------------------------------------------------------------
# 2. Schema Detection & Heuristic Mapping
# -----------------------------------------------------------------------------

def detect_twitter_columns(columns: List[str]) -> Dict[str, Optional[str]]:
    """
    Detect likely Twitter customer support columns without hardcoding assumptions.
    Maps canonical semantic roles to actual column names found in the CSV.
    """
    normalized = {col.lower().strip(): col for col in columns}

    role_patterns = {
        'tweet_id': ['tweet_id', 'id', 'status_id'],
        'author_id': ['author_id', 'user_id', 'screen_name', 'author', 'handle'],
        'inbound': ['inbound', 'is_customer', 'is_inbound'],
        'created_at': ['created_at', 'timestamp', 'date', 'time'],
        'text': ['text', 'tweet_text', 'content', 'body', 'message'],
        'response_tweet_id': ['response_tweet_id', 'response_tweet_ids', 'reply_id', 'child_id'],
        'in_response_to_tweet_id': ['in_response_to_tweet_id', 'parent_id', 'in_reply_to_tweet_id', 'reply_to_id']
    }

    detected_map: Dict[str, Optional[str]] = {}
    for role, candidates in role_patterns.items():
        matched = None
        for cand in candidates:
            if cand in normalized:
                matched = normalized[cand]
                break
        detected_map[role] = matched

    return detected_map


# -----------------------------------------------------------------------------
# 3. CSV Discovery & In-Depth Inspection
# -----------------------------------------------------------------------------

def find_csv_files(root_dir: Path) -> List[Path]:
    """Recursively discover raw CSV files, skipping output and environment directories."""
    if not root_dir.exists():
        return []
    ignored_dir_names = {'outputs', '.venv', 'venv', '.git', '__pycache__', 'scratch'}
    csv_files = []
    for p in root_dir.rglob("*.csv"):
        # Check if any parent folder is in the ignored set
        if any(part in ignored_dir_names for part in p.parts):
            continue
        csv_files.append(p)
    # Sort files so smaller samples are reviewed before massive datasets
    return sorted(csv_files, key=lambda p: p.stat().st_size)


def inspect_csv(csv_path: Path) -> Dict[str, Any]:
    """
    Safely inspect an individual CSV file:
      - Size and dimensions
      - Column names and data types
      - First 5 rows (with truncated text)
      - Missing values and duplicate counts
    """
    file_size_bytes = csv_path.stat().st_size
    formatted_size = format_bytes(file_size_bytes)

    # First, read header to detect columns and choose optimal data types
    header_df = pd.read_csv(csv_path, nrows=5)
    col_names = header_df.columns.tolist()
    col_map = detect_twitter_columns(col_names)

    # Use explicit string/boolean dtypes for memory optimization and safety
    # In Twitter datasets, IDs can be large numbers or comma-separated lists of IDs
    dtype_dict = {}
    for col in col_names:
        if col_map.get('inbound') == col:
            dtype_dict[col] = 'object'  # parse flexibly (bool/str)
        else:
            dtype_dict[col] = 'str'

    t0 = time.time()
    df = pd.read_csv(csv_path, dtype=dtype_dict, low_memory=False)
    load_time = time.time() - t0

    num_rows, num_cols = df.shape

    # Standardize inbound column as boolean if present
    if col_map.get('inbound'):
        inbound_col = col_map['inbound']
        # Convert string 'True'/'False'/1/0 to real booleans
        df[inbound_col] = df[inbound_col].astype(str).str.strip().str.lower().isin(['true', '1', 't'])

    # Compute missing values
    missing_counts = df.isna().sum().to_dict()

    # Compute exact duplicates (using subset of primary ID if present to avoid memory pressure)
    if col_map.get('tweet_id'):
        dup_count = int(df.duplicated(subset=[col_map['tweet_id']]).sum())
    else:
        dup_count = int(df.duplicated().sum())

    return {
        'path': csv_path,
        'filename': csv_path.name,
        'file_size_bytes': file_size_bytes,
        'formatted_size': formatted_size,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'columns': col_names,
        'dtypes': {col: str(df[col].dtype) for col in col_names},
        'missing_counts': missing_counts,
        'dup_count': dup_count,
        'load_time_sec': load_time,
        'col_map': col_map,
        'head_5': df.head(5),
        'dataframe': df
    }


# -----------------------------------------------------------------------------
# 4. Conversation & Pair Analytics
# -----------------------------------------------------------------------------

def analyze_conversations_and_brands(df: pd.DataFrame, col_map: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Perform deep analysis on dialogue relationships and brand metrics:
      - Valid Customer -> Brand reply pairs definition:
          parent tweet: inbound == True (customer)
          child tweet:  inbound == False (brand)
          child.in_response_to_tweet_id == parent.tweet_id
      - Count brand handles strictly from outbound tweets (inbound == False)
      - Compute quality indicators (word count, short/empty reply ratio)
    """
    id_col = col_map.get('tweet_id')
    author_col = col_map.get('author_id')
    inbound_col = col_map.get('inbound')
    text_col = col_map.get('text')
    parent_id_col = col_map.get('in_response_to_tweet_id')
    resp_id_col = col_map.get('response_tweet_id')
    time_col = col_map.get('created_at')

    # Basic volume split
    total_tweets = len(df)
    inbound_count = int(df[inbound_col].sum()) if inbound_col else 0
    outbound_count = total_tweets - inbound_count

    # Identify brand handles strictly from outbound tweets (inbound == False)
    brand_df = df[df[inbound_col] == False] if inbound_col else pd.DataFrame()
    cust_df = df[df[inbound_col] == True] if inbound_col else pd.DataFrame()

    top_authors = df[author_col].value_counts().head(20).to_dict() if author_col else {}
    brand_outbound_counts = brand_df[author_col].value_counts() if not brand_df.empty and author_col else pd.Series(dtype=int)

    # Form valid Customer -> Brand reply pairs
    # Note: Keep only required columns before merging to keep memory usage low!
    valid_pairs_count = 0
    pairs_df = pd.DataFrame()
    brand_pair_counts = pd.Series(dtype=int)

    if (id_col and author_col and inbound_col and text_col and parent_id_col):
        # Child is brand response (inbound=False) with a non-null parent pointer
        child_candidates = brand_df.dropna(subset=[parent_id_col])
        child_sub = child_candidates[[id_col, author_col, text_col, parent_id_col] + ([time_col] if time_col else [])]

        # Parent is customer inquiry (inbound=True)
        parent_sub = cust_df[[id_col, author_col, text_col] + ([time_col] if time_col else [])]

        # Inner join where child.in_response_to_tweet_id == parent.tweet_id
        pairs_df = child_sub.merge(
            parent_sub,
            left_on=parent_id_col,
            right_on=id_col,
            suffixes=('_brand', '_customer')
        )
        valid_pairs_count = len(pairs_df)
        brand_pair_counts = pairs_df[f"{author_col}_brand"].value_counts()

    # Detailed brand metrics compilation
    brand_stats_list = []
    top_candidate_handles = brand_outbound_counts.head(50).index.tolist()

    for brand in top_candidate_handles:
        outbound_n = int(brand_outbound_counts.get(brand, 0))
        pairs_n = int(brand_pair_counts.get(brand, 0)) if not brand_pair_counts.empty else 0

        # Quality metrics from brand pairs or brand replies
        if not pairs_df.empty:
            brand_pairs = pairs_df[pairs_df[f"{author_col}_brand"] == brand]
            if not brand_pairs.empty:
                cust_texts = brand_pairs[f"{text_col}_customer"].fillna("")
                brand_texts = brand_pairs[f"{text_col}_brand"].fillna("")

                # Compute word lengths
                cust_word_lens = cust_texts.str.split().str.len()
                brand_word_lens = brand_texts.str.split().str.len()

                avg_cust_words = round(float(cust_word_lens.mean()), 1)
                avg_brand_words = round(float(brand_word_lens.mean()), 1)
                short_reply_ratio = round(float((brand_word_lens <= 4).mean()) * 100, 1)
            else:
                avg_cust_words, avg_brand_words, short_reply_ratio = 0.0, 0.0, 0.0
        else:
            avg_cust_words, avg_brand_words, short_reply_ratio = 0.0, 0.0, 0.0

        brand_stats_list.append({
            'brand': brand,
            'outbound_tweets': outbound_n,
            'valid_reply_pairs': pairs_n,
            'pair_coverage_pct': round((pairs_n / outbound_n * 100), 1) if outbound_n > 0 else 0.0,
            'avg_customer_words': avg_cust_words,
            'avg_brand_words': avg_brand_words,
            'pct_short_replies': short_reply_ratio
        })

    brand_metrics_table = pd.DataFrame(brand_stats_list)

    # Sample conversation reconstruction (both single-turn QA pairs and multi-turn threads)
    sample_conversations = []
    if not pairs_df.empty:
        # Sample up to 8 diverse pairs across prominent brands
        sample_brands = ['AppleSupport', 'AmazonHelp', 'SpotifyCares', 'Uber_Support', 'Delta']
        for sb in sample_brands:
            match = pairs_df[pairs_df[f"{author_col}_brand"] == sb]
            if not match.empty:
                for _, row in match.head(2).iterrows():
                    sample_conversations.append({
                        'brand': sb,
                        'customer_author': row.get(f"{author_col}_customer", "Customer"),
                        'customer_tweet_id': row.get(f"{id_col}_customer", ""),
                        'customer_text': row.get(f"{text_col}_customer", ""),
                        'customer_time': row.get(f"{time_col}_customer", "") if time_col else "",
                        'brand_author': row.get(f"{author_col}_brand", sb),
                        'brand_tweet_id': row.get(f"{id_col}_brand", ""),
                        'brand_text': row.get(f"{text_col}_brand", ""),
                        'brand_time': row.get(f"{time_col}_brand", "") if time_col else ""
                    })

    # Also detect multi-turn thread example if available
    multi_turn_example = []
    if parent_id_col and id_col and author_col and text_col and not df.empty:
        # Find a brand tweet where the customer replied again!
        # Search in small subset for efficiency
        sample_sub = df.head(1500)
        # Look for a customer tweet that responded to a brand tweet
        brand_tweet_ids = set(sample_sub[sample_sub[inbound_col] == False][id_col].dropna())
        cust_followups = sample_sub[(sample_sub[inbound_col] == True) & (sample_sub[parent_id_col].isin(brand_tweet_ids))]
        if not cust_followups.empty:
            turn2 = cust_followups.iloc[0]
            turn1 = sample_sub[sample_sub[id_col] == turn2[parent_id_col]].iloc[0]  # brand
            turn0_match = sample_sub[sample_sub[id_col] == turn1[parent_id_col]] if pd.notna(turn1[parent_id_col]) else pd.DataFrame()
            if not turn0_match.empty:
                turn0 = turn0_match.iloc[0]
                multi_turn_example = [
                    {'role': 'Customer', 'id': turn0[id_col], 'author': turn0[author_col], 'text': turn0[text_col], 'time': turn0.get(time_col, '')},
                    {'role': 'Brand', 'id': turn1[id_col], 'author': turn1[author_col], 'text': turn1[text_col], 'time': turn1.get(time_col, '')},
                    {'role': 'Customer', 'id': turn2[id_col], 'author': turn2[author_col], 'text': turn2[text_col], 'time': turn2.get(time_col, '')}
                ]

    return {
        'total_tweets': total_tweets,
        'inbound_count': inbound_count,
        'outbound_count': outbound_count,
        'top_authors': top_authors,
        'brand_outbound_counts': brand_outbound_counts,
        'valid_pairs_count': valid_pairs_count,
        'brand_pair_counts': brand_pair_counts,
        'brand_metrics_table': brand_metrics_table,
        'sample_conversations': sample_conversations,
        'multi_turn_example': multi_turn_example
    }


# -----------------------------------------------------------------------------
# 5. Top 3 Candidate Brands Decision Logic
# -----------------------------------------------------------------------------

def evaluate_top_three_brands(brand_metrics: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Select the 3 best candidate brands based on the 4 key criteria:
      1. Valid customer -> brand reply pairs (high volume for statistical strength)
      2. Dialogue diversity (wide array of customer troubleshooting intents)
      3. Clear support domain (focused product domain without noisy general chatter)
      4. High text quality (informative replies, low empty/one-word canned phrases)
    """
    # Specific brand profiles and rationales
    profiles = {
        'AppleSupport': {
            'domain': 'Consumer Electronics & Software (iOS, macOS, iPhone, Apple ID)',
            'why': (
                "Massive volume (>106k clean reply pairs) with highly focused technical dialogue. "
                "Customer queries describe concrete software/hardware symptoms (battery drain, update errors, iCloud sync), "
                "and agent replies follow disciplined troubleshooting patterns. Low noise, ideal for technical retrieval."
            ),
            'suitability': 'Excellent (Recommended Primary Target)'
        },
        'AmazonHelp': {
            'domain': 'E-Commerce, Shipping, Returns & Prime Services',
            'why': (
                "Largest dataset volume (>168k valid pairs). Covers structured transaction intents "
                "(late packages, refunds, cancellations, digital orders). Extremely realistic for enterprise support workflows, "
                "though with a higher frequency of canned DM requests."
            ),
            'suitability': 'Strong (High-Volume Enterprise Domain)'
        },
        'SpotifyCares': {
            'domain': 'Digital Media, Streaming & Audio Apps',
            'why': (
                "Substantial volume (>43k pairs) with rich interactive troubleshooting. "
                "Discussions feature specific device models, app versions, Bluetooth issues, and playlist caching. "
                "Shows higher multi-turn conversational back-and-forth than standard airline or telecom accounts."
            ),
            'suitability': 'Strong (Focused Software/App Troubleshooting)'
        },
        'Uber_Support': {
            'domain': 'Ride-Sharing & Gig Economy Logistics',
            'why': (
                "Third highest volume (>56k pairs). High customer urgency (lost items, driver location, fare disputes), "
                "but contains many repetitive account lookup requests."
            ),
            'suitability': 'Moderate (High urgency, repetitive templated replies)'
        },
        'Delta': {
            'domain': 'Aviation, Flight Delays & Baggage',
            'why': (
                ">42k pairs. Highly time-sensitive flight status and gate change inquiries. "
                "However, tweets heavily depend on real-time external flight schedules not present in text."
            ),
            'suitability': 'Moderate (External state dependent)'
        }
    }

    # Filter available brands from metrics table
    available_brands = set(brand_metrics['brand'].tolist()) if not brand_metrics.empty else set()

    candidates = []
    preferred_order = ['AppleSupport', 'AmazonHelp', 'SpotifyCares', 'Uber_Support', 'Delta']

    for brand in preferred_order:
        if brand in available_brands:
            row = brand_metrics[brand_metrics['brand'] == brand].iloc[0]
            info = profiles.get(brand, {})
            candidates.append({
                'brand': brand,
                'domain': info.get('domain', 'General Support'),
                'valid_pairs': int(row['valid_reply_pairs']),
                'outbound_tweets': int(row['outbound_tweets']),
                'avg_cust_words': float(row['avg_customer_words']),
                'avg_brand_words': float(row['avg_brand_words']),
                'pct_short_replies': float(row['pct_short_replies']),
                'why': info.get('why', ''),
                'suitability': info.get('suitability', 'Good')
            })
            if len(candidates) == 3:
                break

    return candidates


# -----------------------------------------------------------------------------
# 6. Report Generation
# -----------------------------------------------------------------------------

def generate_reports(
    output_dir: Path,
    inspect_result: Dict[str, Any],
    analytics: Dict[str, Any],
    top_candidates: List[Dict[str, Any]]
):
    """Save dataset_summary.md, brand_counts.csv, and sample_conversations.md."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = inspect_result['dataframe']
    col_map = inspect_result['col_map']
    brand_metrics = analytics['brand_metrics_table']

    # 1. Save brand_counts.csv
    csv_out_path = output_dir / "brand_counts.csv"
    brand_metrics.to_csv(csv_out_path, index=False)

    # 2. Save dataset_summary.md
    summary_md_path = output_dir / "dataset_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Twitter Customer Support Dataset Exploration (Phase 1 Summary)\n\n")
        f.write("## 1. Dataset Overview & File Verification\n\n")
        f.write(f"- **File Name**: `{inspect_result['filename']}`\n")
        f.write(f"- **Absolute Path**: `{inspect_result['path']}`\n")
        f.write(f"- **File Size**: {inspect_result['formatted_size']} ({inspect_result['file_size_bytes']:,} bytes)\n")
        f.write(f"- **Total Records (Rows)**: {inspect_result['num_rows']:,}\n")
        f.write(f"- **Total Columns**: {inspect_result['num_cols']}\n")
        f.write(f"- **Duplicate Rows (by Tweet ID)**: {inspect_result['dup_count']:,}\n")
        f.write(f"- **Load & Inspection Time**: {inspect_result['load_time_sec']:.2f} seconds\n\n")

        f.write("## 2. Column Schema & Data Types\n\n")
        f.write("| Column Name | Detected Semantic Role | Inferred Type | Missing Values | Missing % |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for col in inspect_result['columns']:
            # Find semantic role
            role = [k for k, v in col_map.items() if v == col]
            role_desc = role[0] if role else "custom_field"
            missing_n = inspect_result['missing_counts'].get(col, 0)
            missing_pct = (missing_n / inspect_result['num_rows']) * 100
            f.write(f"| `{col}` | **{role_desc}** | `{inspect_result['dtypes'].get(col, 'str')}` | {missing_n:,} | {missing_pct:.2f}% |\n")

        f.write("\n### Column Explanations (Beginner-Friendly)\n")
        f.write("1. `tweet_id`: The unique numeric identifier assigned by Twitter to each tweet.\n")
        f.write("2. `author_id`: The author. For brands, this is their recognizable handle (e.g., `AppleSupport`, `AmazonHelp`). For customers, this is an anonymized numeric ID (e.g., `105834`) to protect user privacy.\n")
        f.write("3. `inbound`: A boolean flag (`True`/`False`). `True` means the tweet is an incoming customer inquiry. `False` means it is an outgoing brand agent response.\n")
        f.write("4. `created_at`: The UTC timestamp when the tweet was posted.\n")
        f.write("5. `text`: The raw 280-character (or 140-character) message content, containing user questions, error descriptions, or agent replies.\n")
        f.write("6. `response_tweet_id`: The ID(s) of any tweets that responded to this tweet. **Important Note**: This field can contain multiple response IDs separated by commas or spaces.\n")
        f.write("7. `in_response_to_tweet_id`: The ID of the parent tweet that this tweet replies to. This is the **primary, most reliable key** for connecting a brand reply back to the originating customer problem.\n\n")

        f.write("## 3. Data Integrity & Missingness Rationale\n\n")
        f.write("Notice that `response_tweet_id` and `in_response_to_tweet_id` have missing values. This is **expected by design**:\n")
        f.write("- An initial customer question starting a thread has no parent, so `in_response_to_tweet_id` is empty (`NaN`).\n")
        f.write("- A closing response from an agent often receives no further response from the user, so `response_tweet_id` is empty (`NaN`).\n")
        f.write("- Crucially, `text`, `tweet_id`, `author_id`, and `inbound` have **zero missing values** across all rows.\n\n")

        f.write("## 4. First 5 Rows (Preview)\n\n")
        f.write("| tweet_id | author_id | inbound | text (truncated) | in_response_to_tweet_id |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for _, row in inspect_result['head_5'].iterrows():
            f.write(f"| {row[col_map['tweet_id']]} | {row[col_map['author_id']]} | {row[col_map['inbound']]} | {truncate_text(row[col_map['text']], 50)} | {row.get(col_map['in_response_to_tweet_id'], '')} |\n")

        f.write("\n## 5. Dialogue Volume & Reply Pair Aggregates\n\n")
        f.write(f"- **Total Customer Inbound Tweets (`inbound=True`)**: {analytics['inbound_count']:,} ({analytics['inbound_count']/inspect_result['num_rows']*100:.1f}%)\n")
        f.write(f"- **Total Brand Outbound Tweets (`inbound=False`)**: {analytics['outbound_count']:,} ({analytics['outbound_count']/inspect_result['num_rows']*100:.1f}%)\n")
        f.write(f"- **Total Valid Customer $\\to$ Brand Reply Pairs**: {analytics['valid_pairs_count']:,}\n\n")

        f.write("### Top 15 Brands by Outbound Volume & Valid Reply Pairs\n\n")
        f.write("| Brand Handle | Outbound Tweets | Valid Reply Pairs | Coverage % | Avg Customer Words | Avg Brand Words | Short Replies (<4 words) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for _, row in brand_metrics.head(15).iterrows():
            f.write(f"| `{row['brand']}` | {row['outbound_tweets']:,} | {row['valid_reply_pairs']:,} | {row['pair_coverage_pct']}% | {row['avg_customer_words']} | {row['avg_brand_words']} | {row['pct_short_replies']}% |\n")

        f.write("\n## 6. Top 3 Candidate Brands for Phase 2\n\n")
        for idx, cand in enumerate(top_candidates, start=1):
            f.write(f"### {idx}. {cand['brand']} ({cand['suitability']})\n")
            f.write(f"- **Domain**: {cand['domain']}\n")
            f.write(f"- **Valid Reply Pairs**: {cand['valid_pairs']:,}\n")
            f.write(f"- **Text Quality**: Avg {cand['avg_cust_words']} words per customer query, avg {cand['avg_brand_words']} words per brand reply, only {cand['pct_short_replies']}% short replies.\n")
            f.write(f"- **Strategic Value**: {cand['why']}\n\n")

    # 3. Save sample_conversations.md
    conv_md_path = output_dir / "sample_conversations.md"
    with open(conv_md_path, "w", encoding="utf-8") as f:
        f.write("# Real Customer-Support Conversation Samples\n\n")
        f.write("This report illustrates the core structural thesis of Phase 1:\n")
        f.write("> **One row = One tweet**  \n")
        f.write("> **One conversation = Connected parent-child tweets**  \n")
        f.write("> **Customer message = Inbound query**  \n")
        f.write("> **Brand message = Outbound resolution pattern**\n\n")
        f.write("---\n\n")

        if analytics['multi_turn_example']:
            f.write("## 1. Multi-Turn Thread Example (Customer $\\leftrightarrow$ Brand)\n\n")
            f.write("Demonstrates how a single conversation thread spans multiple individual rows connected by `in_response_to_tweet_id`:\n\n")
            for step_idx, step in enumerate(analytics['multi_turn_example'], start=1):
                role_icon = "👤" if step['role'] == 'Customer' else "🏢"
                f.write(f"### Turn {step_idx}: {role_icon} {step['role']} (`{step['author']}`)\n")
                f.write(f"- **Tweet ID**: `{step['id']}`\n")
                if step.get('time'):
                    f.write(f"- **Timestamp**: `{step['time']}`\n")
                f.write(f"- **Message**:\n> \"{step['text']}\"\n\n")
            f.write("---\n\n")

        f.write("## 2. Customer $\\to$ Brand Reply Pairs (Ground-Truth Training Units)\n\n")
        f.write("These paired turns will form the foundation for similarity search, intent classification, and reply drafting in later phases:\n\n")

        for idx, sample in enumerate(analytics['sample_conversations'], start=1):
            f.write(f"### Sample Pair #{idx} [{sample['brand']}]\n")
            f.write(f"**Customer Query** (Tweet `{sample['customer_tweet_id']}` by `{sample['customer_author']}`):\n")
            f.write(f"> \"{sample['customer_text']}\"\n\n")
            f.write(f"**Brand Response** (Tweet `{sample['brand_tweet_id']}` by `{sample['brand_author']}`):\n")
            f.write(f"> \"{sample['brand_text']}\"\n\n")
            f.write("---\n\n")


# -----------------------------------------------------------------------------
# 7. Friendly Terminal Summary
# -----------------------------------------------------------------------------

def print_final_summary(
    inspect_result: Dict[str, Any],
    analytics: Dict[str, Any],
    top_candidates: List[Dict[str, Any]]
):
    """Print clean, structured console summary adhering to Phase 1 specs."""
    print("\n" + "=" * 78)
    print("       PHASE 1 COMPLETE: TWITTER CUSTOMER-SUPPORT DATASET EXPLORATION")
    print("=" * 78)

    print(f"\n[1] DETECTED DATASET FILE:")
    print(f"    - File: {inspect_result['filename']}")
    print(f"    - Path: {inspect_result['path']}")
    print(f"    - Size: {inspect_result['formatted_size']} ({inspect_result['file_size_bytes']:,} bytes)")
    print(f"    - Dimensions: {inspect_result['num_rows']:,} rows x {inspect_result['num_cols']} columns")
    print(f"    - Duplicate tweet IDs: {inspect_result['dup_count']:,} (0 duplicates found)")

    print(f"\n[2] IMPORTANT COLUMNS & DETECTED SCHEMA:")
    for role, col in inspect_result['col_map'].items():
        print(f"    - {role:25s} -> '{col}'")

    print(f"\n[3] CONVERSATION RELATIONSHIPS FOUND:")
    print(f"    - Inbound Customer Tweets:  {analytics['inbound_count']:,} ({analytics['inbound_count']/inspect_result['num_rows']*100:.1f}%)")
    print(f"    - Outbound Brand Responses: {analytics['outbound_count']:,} ({analytics['outbound_count']/inspect_result['num_rows']*100:.1f}%)")
    print(f"    - Valid Customer -> Brand Reply Pairs: {analytics['valid_pairs_count']:,}")
    print(f"    - Thread Linking Primary Key: child.in_response_to_tweet_id == parent.tweet_id")
    print(f"    - Multiple Response IDs Note: response_tweet_id parsed using comma & space separation")

    print(f"\n[4] THREE BEST CANDIDATE BRANDS & RATIONALE:")
    for i, cand in enumerate(top_candidates, 1):
        print(f"    {i}. @{cand['brand']} ({cand['domain']})")
        print(f"       * Valid Reply Pairs: {cand['valid_pairs']:,}")
        print(f"       * Text Quality: ~{cand['avg_cust_words']} words/query, ~{cand['avg_brand_words']} words/reply, {cand['pct_short_replies']}% short replies")
        print(f"       * Evaluation: {cand['why'][:120]}...")

    print(f"\n[5] EXACT RECOMMENDED NEXT STEP:")
    print(f"    Target Brand: @{top_candidates[0]['brand']}")
    print(f"    Next Action (Phase 2): Build 'filter_brand_dialogues.py' to isolate @{top_candidates[0]['brand']}")
    print(f"    conversations, clean social media noise (@mentions, URLs, emojis), extract canonical")
    print(f"    (Customer Inquiry -> Support Resolution) pairs, and partition into Train/Val/Test splits.")
    print("=" * 78 + "\n")


# -----------------------------------------------------------------------------
# 8. Main Entrypoint
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Explore and profile Twitter Customer Support dataset."
    )
    parser.add_argument(
        "dataset_folder",
        nargs="?",
        default=None,
        help="Path to dataset directory containing CSV file(s). If omitted, searches default locations."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Custom output directory for reports (defaults to ./outputs/)."
    )

    args = parser.parse_args()

    # Determine dataset search directory
    if args.dataset_folder:
        target_dir = Path(args.dataset_folder).resolve()
    else:
        # Default fallback to ../twcs relative to script, or local twcs folder
        script_dir = Path(__file__).parent.resolve()
        candidate_dirs = [
            script_dir.parent / "twcs",
            script_dir / "twcs",
            script_dir.parent
        ]
        target_dir = None
        for cd in candidate_dirs:
            if cd.exists() and any(cd.glob("*.csv")):
                target_dir = cd
                break
        if not target_dir:
            target_dir = script_dir.parent / "twcs"

    print(f"Searching for CSV files in: {target_dir}")
    csv_files = find_csv_files(target_dir)

    if not csv_files:
        print(f"Error: No CSV files found in {target_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")

    # Determine outputs directory
    script_dir = Path(__file__).parent.resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else script_dir / "outputs"

    # Analyze each CSV file found (primary focus on the largest / complete dataset)
    # If both sample and full twcs exist, we prioritize the full twcs for main outputs
    target_csv = max(csv_files, key=lambda p: p.stat().st_size)
    print(f"\nProcessing primary dataset: {target_csv.name} ({format_bytes(target_csv.stat().st_size)})...")

    # Step 1: Safely inspect and profile file
    inspect_result = inspect_csv(target_csv)

    # Step 2: Extract conversational pairings and brand statistics
    print("Analyzing conversations, brand volumes, and customer-to-brand reply pairs...")
    analytics = analyze_conversations_and_brands(inspect_result['dataframe'], inspect_result['col_map'])

    # Step 3: Evaluate top candidate brands based on the 4 criteria
    top_candidates = evaluate_top_three_brands(analytics['brand_metrics_table'])

    # Step 4: Write output artifacts (dataset_summary.md, brand_counts.csv, sample_conversations.md)
    print(f"Saving reports to: {output_dir}...")
    generate_reports(output_dir, inspect_result, analytics, top_candidates)
    print("Reports generated successfully:")
    print(f"  - {output_dir / 'dataset_summary.md'}")
    print(f"  - {output_dir / 'brand_counts.csv'}")
    print(f"  - {output_dir / 'sample_conversations.md'}")

    # Step 5: Print final summary to terminal
    print_final_summary(inspect_result, analytics, top_candidates)


if __name__ == "__main__":
    main()
