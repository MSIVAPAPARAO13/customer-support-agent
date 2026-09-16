"""
build_corpus.py — Build a capped, paired corpus from twcs.csv.

Usage:
    python scripts/build_corpus.py \\
        --input data/raw/twcs.csv \\
        --brand AppleSupport \\
        --limit 30000 \\
        --output data/processed/corpus.csv

Outputs data/processed/corpus.csv with columns:
    id, customer_text, brand_response, intent

Intent is left as empty string — it is filled by the weak-labelling step
or by human annotation. Use --brand AmazonHelp to build an AmazonHelp corpus.

If --limit is set, randomly samples that many pairs (seed 42) to keep
local runs fast. Disclose any change to this limit in your submitted report.
"""

import argparse
import csv
import os
import random
from pathlib import Path


def load_twcs(path: str):
    """Load twcs.csv into a dict keyed by tweet_id."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["tweet_id"]] = row
    return rows


def build_pairs(rows: dict, brand: str):
    """
    Build (customer_text, brand_response) pairs for a given brand.

    A pair is: an inbound tweet (inbound=True) whose response_tweet_id
    points to an outbound tweet authored by <brand>.
    """
    pairs = []
    for tid, row in rows.items():
        # Only consider inbound (customer) tweets with a known response
        if row.get("inbound", "").strip().lower() not in ("true", "1", "yes"):
            continue
        resp_id = row.get("response_tweet_id", "").strip()
        if not resp_id:
            continue
        # Look up the response
        resp = rows.get(resp_id)
        if resp is None:
            continue
        # Check brand
        author = resp.get("author_id", "").strip()
        if author.lower() != brand.lower():
            continue
        customer_text = row.get("text", "").strip()
        brand_text = resp.get("text", "").strip()
        if not customer_text or not brand_text:
            continue
        pairs.append({
            "id": tid,
            "customer_text": customer_text,
            "brand_response": brand_text,
            "intent": "",
        })
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build paired corpus from twcs.csv")
    parser.add_argument("--input", default="data/raw/twcs.csv", help="Path to twcs.csv")
    parser.add_argument("--brand", default="AppleSupport",
                        help="Brand handle to filter (e.g. AppleSupport, AmazonHelp)")
    parser.add_argument("--limit", type=int, default=30000,
                        help="Max corpus size (default 30000; lower for faster local runs)")
    parser.add_argument("--output", default="data/processed/corpus.csv",
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        print("  Download twcs.csv from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        print("  and place it at data/raw/twcs.csv")
        return 1

    print(f"[build_corpus] Loading {args.input} ...")
    rows = load_twcs(args.input)
    print(f"[build_corpus] Loaded {len(rows):,} tweets total")

    print(f"[build_corpus] Building pairs for brand={args.brand} ...")
    pairs = build_pairs(rows, args.brand)
    print(f"[build_corpus] Found {len(pairs):,} {args.brand} pairs")

    if args.limit and len(pairs) > args.limit:
        rng = random.Random(args.seed)
        pairs = rng.sample(pairs, args.limit)
        print(f"[build_corpus] Sampled {len(pairs):,} pairs (seed={args.seed})")

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "customer_text", "brand_response", "intent"])
        writer.writeheader()
        writer.writerows(pairs)

    print(f"[build_corpus] Written {len(pairs):,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
