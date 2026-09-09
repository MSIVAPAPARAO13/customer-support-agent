#!/usr/bin/env python3
"""
Predict Intent with Explainable TF-IDF + Logistic Regression (Phase 6)
======================================================================

Purpose:
  Applies the trained TF-IDF + Logistic Regression model to input customer
  support messages, producing predicted intent, confidence score, top-3 ranked
  candidate intents with probabilities, and a low-confidence human review flag.

Outputs:
  - customer_tweet_id
  - customer_text_clean
  - predicted_intent
  - predicted_confidence
  - top_3_intents
  - top_3_probabilities
  - needs_human_review
  - model_name
  - model_version

Confidence Safeguard:
  Predicted confidence is a model probability estimate from a weak-label prototype.
  It is not yet calibrated against human ground truth.
  If predicted_confidence < 0.60, needs_human_review is set to True.
  This is an upstream review signal; it does NOT make final auto-handle/escalate decisions.

Usage:
  python scripts/predict_tfidf_classifier.py --input data/processed/applesupport_validation.csv --output outputs/tfidf/validation_predictions.csv
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib

# Ensure Windows stdout handles UTF-8 gracefully
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "tfidf_logistic_regression"
MODEL_VERSION = "v1.0-weak-prototype"


def load_model_artifacts(models_dir: Path) -> Tuple[Any, Any]:
    """Loads fitted TfidfVectorizer and LogisticRegression classifier."""
    vec_path = models_dir / "tfidf_vectorizer.joblib"
    cls_path = models_dir / "tfidf_logistic_regression.joblib"

    if not vec_path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found at: {vec_path}. Run training first.")
    if not cls_path.exists():
        raise FileNotFoundError(f"Classifier not found at: {cls_path}. Run training first.")

    print(f"[Model Loader] Loading TF-IDF vectorizer from {vec_path.name}...")
    vectorizer = joblib.load(vec_path)
    print(f"[Model Loader] Loading Logistic Regression classifier from {cls_path.name}...")
    classifier = joblib.load(cls_path)
    return vectorizer, classifier


def run_predictions(
    df_input: pd.DataFrame,
    vectorizer: Any,
    classifier: Any,
    confidence_threshold: float = 0.60
) -> pd.DataFrame:
    """
    Transforms text and generates predictions, top-3 candidates, and human review flags.
    """
    if 'customer_text_clean' not in df_input.columns:
        raise KeyError("Input dataframe must contain 'customer_text_clean' column.")

    texts = df_input['customer_text_clean'].fillna('').astype(str).tolist()
    print(f"[Inference] Vectorizing {len(texts):,} customer inquiries...")
    X = vectorizer.transform(texts)

    print("[Inference] Computing intent class probabilities...")
    proba_matrix = classifier.predict_proba(X)
    classes = list(classifier.classes_)

    results = []
    print(f"[Inference] Formatting top-3 rankings (review threshold < {confidence_threshold:.2f})...")

    for i in range(len(texts)):
        probs = proba_matrix[i]
        # Sort descending by probability
        ranked_indices = np.argsort(probs)[::-1]

        top_idx = ranked_indices[0]
        top_intent = classes[top_idx]
        top_conf = float(probs[top_idx])

        # Top 3 candidates
        top_3_idx = ranked_indices[:3]
        top_3_intents_list = [classes[idx] for idx in top_3_idx]
        top_3_probs_list = [f"{probs[idx]:.4f}" for idx in top_3_idx]

        # Review flag: True if confidence below threshold
        needs_review = bool(top_conf < confidence_threshold)

        tweet_id = df_input.iloc[i].get('customer_tweet_id', '')

        results.append({
            'customer_tweet_id': str(tweet_id),
            'customer_text_clean': texts[i],
            'predicted_intent': top_intent,
            'predicted_confidence': round(top_conf, 4),
            'top_3_intents': "; ".join(top_3_intents_list),
            'top_3_probabilities': "; ".join(top_3_probs_list),
            'needs_human_review': needs_review,
            'model_name': MODEL_NAME,
            'model_version': MODEL_VERSION
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Generate Intent Predictions with TF-IDF Classifier.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV containing customer_tweet_id and customer_text_clean"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV to save predictions"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory containing saved model joblib files"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Confidence threshold below which needs_human_review is set to True (default: 0.60)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional row limit for quick inference testing"
    )

    args = parser.parse_args()

    print("=" * 78)
    print("      PHASE 6: TF-IDF + LOGISTIC REGRESSION PREDICTOR")
    print("=" * 78)

    # 1. Load artifacts
    vectorizer, classifier = load_model_artifacts(args.models_dir)

    # 2. Load input
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"\n[Data Loader] Reading target inputs from: {args.input.name}...")
    df_input = pd.read_csv(args.input, dtype=str)
    print(f"[Data Loader] Loaded {len(df_input):,} rows.")

    if args.sample is not None and args.sample > 0 and args.sample < len(df_input):
        print(f"[Data Loader] Subsampling to {args.sample} rows as requested...")
        df_input = df_input.head(args.sample)

    # 3. Predict
    df_preds = run_predictions(
        df_input=df_input,
        vectorizer=vectorizer,
        classifier=classifier,
        confidence_threshold=args.threshold
    )

    # 4. Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_csv(args.output, index=False)
    print(f"\n[Output] Saved predictions to: {args.output} ({len(df_preds):,} rows)")

    # 5. Summary statistics
    rev_count = df_preds['needs_human_review'].sum()
    rev_pct = rev_count / len(df_preds) if len(df_preds) > 0 else 0
    print(f"[Summary] Flagged for Human Review (conf < {args.threshold:.2f}): {rev_count:,} / {len(df_preds):,} ({rev_pct:.2%})")

    print("\n[Audit Preview: First 3 Predictions]")
    for idx, row in df_preds.head(3).iterrows():
        preview = (row['customer_text_clean'][:60] + "...") if len(row['customer_text_clean']) > 60 else row['customer_text_clean']
        print(f" {idx+1}. Tweet {row['customer_tweet_id']}: '{preview}'")
        print(f"    -> Intent: {row['predicted_intent']} (Conf: {row['predicted_confidence']:.4f}) | Review: {row['needs_human_review']}")
        print(f"       Top 3: {row['top_3_intents']}")
        print(f"       Probs: {row['top_3_probabilities']}")

    print("\n" + "=" * 78)
    print("INFERENCE COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    from typing import Tuple
    main()
