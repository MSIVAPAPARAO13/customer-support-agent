#!/usr/bin/env python3
"""
Intent Classification Baselines (Phase 5)
=========================================

Purpose:
  Implements and executes two essential, transparent baseline models for intent
  classification on Twitter customer support dialogues:
  
  1. Baseline 1: `majority_weak_intent`
     - Reads the weak-labelled training set (`applesupport_train_weak_labels.csv`).
     - Identifies the empirical mode (most frequent intent) of the weak training labels.
     - Predicts that identical intent for all input rows with empirical prior probability.
     - Serves as the minimal performance floor: any learning model (TF-IDF, neural)
       MUST statistically outperform this trivial strategy to demonstrate genuine utility.
       
  2. Baseline 2: `keyword_rule_classifier`
     - Applies deterministic keyword heuristics directly to incoming customer messages.
     - Captures explicit lexicon patterns (e.g., 'Apple ID', 'refund', 'Bluetooth', 'iOS update').
     - Assigns predicted intent, confidence, the exact matched rule, and a flag indicating
       whether a strong domain rule matched or if it defaulted to fallback.
     - Serves as the domain-heuristic benchmark: statistical or ML models must prove they
       provide value beyond simple dictionary search.

Integrity Constraints:
  - Does NOT evaluate against unlabelled test or incomplete golden sets.
  - Generates and persists predictions only in `outputs/baselines/`.
  - Transparently logs matched rules and confidence scores.

Usage:
  # Run both baselines on validation set
  python scripts/run_baselines.py

  # Run on a quick 50-row sample for inspection
  python scripts/run_baselines.py --sample 50
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

# Ensure Windows stdout handles UTF-8 / emojis gracefully
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add repo root and scripts to path for importing weak label rules
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

try:
    from create_weak_labels import assign_weak_label
except ImportError:
    raise ImportError("Unable to import assign_weak_label from scripts/create_weak_labels.py")


def train_majority_baseline(train_weak_path: Path) -> Tuple[str, float]:
    """
    Learns the majority intent class and its empirical proportion from training weak labels.
    """
    if not train_weak_path.exists():
        raise FileNotFoundError(
            f"Training weak labels not found at: {train_weak_path}. "
            "Please run `python scripts/create_weak_labels.py` first."
        )

    print(f"[Baseline 1] Reading training weak labels from {train_weak_path.name}...")
    df_train = pd.read_csv(train_weak_path, usecols=['weak_intent'])
    counts = df_train['weak_intent'].value_counts()
    majority_intent = counts.index[0]
    total = len(df_train)
    proportion = counts.iloc[0] / total

    print(f"[Baseline 1] Training distribution mode: '{majority_intent}' "
          f"({counts.iloc[0]:,} / {total:,} rows = {proportion:.2%})")
    return majority_intent, proportion


def run_majority_baseline(
    input_df: pd.DataFrame,
    majority_intent: str,
    confidence_score: float
) -> pd.DataFrame:
    """
    Predicts the majority intent for every record in input_df.
    Output schema:
      - customer_tweet_id
      - customer_text_clean
      - baseline_name
      - predicted_intent
      - prediction_confidence
    """
    records = []
    baseline_name = "majority_weak_intent"
    conf_str = f"{confidence_score:.4f}"

    for _, row in input_df.iterrows():
        tweet_id = str(row.get('customer_tweet_id', ''))
        text = str(row.get('customer_text_clean', ''))
        records.append({
            'customer_tweet_id': tweet_id,
            'customer_text_clean': text,
            'baseline_name': baseline_name,
            'predicted_intent': majority_intent,
            'prediction_confidence': conf_str
        })

    return pd.DataFrame(records)


def run_keyword_rule_baseline(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies transparent regex/keyword rules directly to customer clean text.
    Output schema:
      - customer_tweet_id
      - customer_text_clean
      - baseline_name
      - predicted_intent
      - prediction_confidence
      - matched_rule
      - no_strong_rule_matched
    """
    records = []
    baseline_name = "keyword_rule_classifier"

    for _, row in input_df.iterrows():
        tweet_id = str(row.get('customer_tweet_id', ''))
        text = str(row.get('customer_text_clean', ''))
        
        intent, rule, conf = assign_weak_label(text)
        
        # A strong rule did not match if it fell back to no match or empty text
        no_strong = (rule in ('rule_fallback_no_match', 'rule_empty_text') or conf == 'low')

        records.append({
            'customer_tweet_id': tweet_id,
            'customer_text_clean': text,
            'baseline_name': baseline_name,
            'predicted_intent': intent,
            'prediction_confidence': conf,
            'matched_rule': rule,
            'no_strong_rule_matched': no_strong
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Run Baseline Intent Classifiers on Dialogue Data.")
    parser.add_argument(
        "--train-weak",
        type=Path,
        default=REPO_ROOT / "data" / "weak_labels" / "applesupport_train_weak_labels.csv",
        help="Path to training weak labels CSV"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "applesupport_validation.csv",
        help="Input CSV to run baseline predictions on (default: validation set)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional row sample size for quick execution/inspection"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "baselines",
        help="Directory to save baseline predictions"
    )

    args = parser.parse_args()

    print("=" * 78)
    print("      PHASE 5: RUN INTENT-CLASSIFICATION BASELINES (PREDICTIONS ONLY)")
    print("=" * 78)

    # 1. Verify train weak labels exist
    majority_intent, majority_prop = train_majority_baseline(args.train_weak)

    # 2. Load input data
    if not args.input.exists():
        # Fallback to weak labels file if processed not found
        fallback_input = REPO_ROOT / "data" / "weak_labels" / "applesupport_validation_weak_labels.csv"
        if fallback_input.exists():
            args.input = fallback_input
        else:
            raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"\nLoading input evaluation targets from: {args.input.name}...")
    df_input = pd.read_csv(args.input, dtype=str)
    
    if args.sample is not None and args.sample > 0 and args.sample < len(df_input):
        print(f"Subsampling to {args.sample} rows as requested...")
        df_input = df_input.head(args.sample)
    else:
        print(f"Processing full input partition: {len(df_input):,} rows...")

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Execute Baseline 1: Majority-intent classifier
    print("\n--- Running Baseline 1: majority_weak_intent ---")
    df_majority = run_majority_baseline(df_input, majority_intent, majority_prop)
    majority_out = args.output_dir / "majority_baseline_predictions.csv"
    df_majority.to_csv(majority_out, index=False)
    print(f"Saved: {majority_out.name} ({len(df_majority):,} predictions)")

    # 4. Execute Baseline 2: Keyword-rule classifier
    print("\n--- Running Baseline 2: keyword_rule_classifier ---")
    df_keyword = run_keyword_rule_baseline(df_input)
    keyword_out = args.output_dir / "keyword_rule_predictions.csv"
    df_keyword.to_csv(keyword_out, index=False)
    print(f"Saved: {keyword_out.name} ({len(df_keyword):,} predictions)")

    # 5. Display sample predictions for audit
    print("\n" + "=" * 78)
    print("SAMPLE PREDICTIONS AUDIT (5 ROWS)")
    print("=" * 78)

    print("\n[Baseline 1: majority_weak_intent Sample]")
    sample_majority = df_majority.head(5)
    for idx, row in sample_majority.iterrows():
        text_preview = (row['customer_text_clean'][:60] + "...") if len(row['customer_text_clean']) > 60 else row['customer_text_clean']
        print(f" {idx+1}. Tweet ID {row['customer_tweet_id']}: '{text_preview}'")
        print(f"    -> Predicted: {row['predicted_intent']} (Conf: {row['prediction_confidence']})")

    print("\n[Baseline 2: keyword_rule_classifier Sample]")
    sample_keyword = df_keyword.head(5)
    for idx, row in sample_keyword.iterrows():
        text_preview = (row['customer_text_clean'][:60] + "...") if len(row['customer_text_clean']) > 60 else row['customer_text_clean']
        print(f" {idx+1}. Tweet ID {row['customer_tweet_id']}: '{text_preview}'")
        print(f"    -> Predicted: {row['predicted_intent']} | Conf: {row['prediction_confidence']}")
        print(f"       Rule: {row['matched_rule']} | No Strong Match: {row['no_strong_rule_matched']}")

    # 6. Integrity and next steps note
    print("\n" + "=" * 78)
    print("INTEGRITY & BENCHMARKING NOTICE:")
    print("  - Predictions have been safely saved to outputs/baselines/.")
    print("  - NO TEST OR GOLDEN EVALUATION METRICS REPORTED.")
    print("  - Final accuracy and F1 require complete human adjudication of the 200-row golden set.")
    print("=" * 78)


if __name__ == "__main__":
    main()
