"""
make_annotation_sheet.py — Sample N rows from corpus for human annotation.

Usage:
    python scripts/make_annotation_sheet.py \\
        --corpus data/processed/corpus.csv \\
        --n 200 \\
        --output data/golden/annotation_sheet.csv

Outputs a CSV with columns:
    id, customer_text, final_intent, final_action, adjudicator_notes, status

The final_intent, final_action, adjudicator_notes columns are left blank for
annotators to fill in. The status column is set to 'PENDING'.

IMPORTANT: Record the IDs of sampled rows and remove them from any training
split to prevent data leakage.
"""

import argparse
import csv
import os
import random
from pathlib import Path

INTENT_CLASSES = [
    "software_update_or_os_issue",
    "device_performance_or_hardware",
    "connectivity_and_network",
    "apps_services_or_icloud",
    "account_access_and_apple_id",
    "billing_subscription_or_purchase",
    "repair_replacement_or_order",
    "other_or_unclear",
]


def main():
    parser = argparse.ArgumentParser(description="Sample rows for human annotation")
    parser.add_argument("--corpus", default="data/processed/corpus.csv",
                        help="Input corpus CSV (id,customer_text,brand_response,intent)")
    parser.add_argument("--n", type=int, default=200, help="Number of rows to sample")
    parser.add_argument("--output", default="data/golden/annotation_sheet.csv",
                        help="Output annotation sheet CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        print(f"ERROR: corpus not found at {args.corpus}")
        print("  Run build_corpus.py first.")
        return 1

    rows = []
    with open(args.corpus, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"[make_annotation_sheet] Loaded {len(rows):,} corpus rows")

    if len(rows) < args.n:
        print(f"WARNING: corpus has only {len(rows)} rows; sampling all of them.")
        args.n = len(rows)

    rng = random.Random(args.seed)
    sampled = rng.sample(rows, args.n)
    print(f"[make_annotation_sheet] Sampled {len(sampled)} rows (seed={args.seed})")

    os.makedirs(Path(args.output).parent, exist_ok=True)
    fieldnames = ["id", "customer_text", "brand_response_reference",
                  "final_intent", "final_action", "adjudicator_notes", "status"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sampled:
            writer.writerow({
                "id": row["id"],
                "customer_text": row["customer_text"],
                "brand_response_reference": row.get("brand_response", ""),
                "final_intent": "",
                "final_action": "",
                "adjudicator_notes": "",
                "status": "PENDING",
            })

    print(f"[make_annotation_sheet] Written to {args.output}")
    print()
    print("  NEXT STEPS:")
    print("  1. Fill in final_intent from:", ", ".join(INTENT_CLASSES))
    print("  2. Fill in final_action: auto_handle or escalate")
    print("  3. Change status to DONE when a row is complete")
    print("  4. Remove these IDs from training to prevent leakage")
    print()
    print("  Sampled IDs (to exclude from training):")
    for row in sampled[:5]:
        print(f"    {row['id']}")
    if len(sampled) > 5:
        print(f"    ... and {len(sampled) - 5} more (see {args.output})")
    return 0


if __name__ == "__main__":
    exit(main())
