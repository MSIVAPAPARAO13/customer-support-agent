#!/usr/bin/env python3
"""
Inspect Explainable TF-IDF + Logistic Regression Model (Phase 6)
================================================================

Purpose:
  Extracts and exports the top positive n-gram features (words and phrases)
  that most strongly influence the Logistic Regression classifier for each
  of the eight intent categories.

Methodology:
  In linear models, positive coefficients directly reflect how strongly the
  presence of a specific unigram or bigram pushes the log-odds toward that class.
  Inspecting these coefficients provides human-interpretable validation of what
  patterns the model extracted from the weak-labeled training dialogues.

Outputs:
  - outputs/tfidf/top_features_by_intent.csv
    Schema: intent,rank,feature,coefficient

Usage:
  python scripts/inspect_tfidf_model.py
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import joblib

# Ensure Windows stdout handles UTF-8 gracefully
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent


def extract_top_features(
    models_dir: Path,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extracts top positive n-gram coefficients per intent class.
    """
    vec_path = models_dir / "tfidf_vectorizer.joblib"
    cls_path = models_dir / "tfidf_logistic_regression.joblib"

    if not vec_path.exists() or not cls_path.exists():
        raise FileNotFoundError(f"Model files missing from {models_dir}. Train model first.")

    print(f"[Model Loader] Loading TF-IDF vectorizer from {vec_path.name}...")
    vectorizer = joblib.load(vec_path)
    print(f"[Model Loader] Loading Logistic Regression classifier from {cls_path.name}...")
    classifier = joblib.load(cls_path)

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = list(classifier.classes_)
    coef_matrix = classifier.coef_

    print(f"[Inspection] Vocabulary size: {len(feature_names):,} features.")
    print(f"[Inspection] Coefficients matrix shape: {coef_matrix.shape} ({len(classes)} classes).")

    records = []
    for class_idx, intent in enumerate(classes):
        class_coefs = coef_matrix[class_idx]
        # Sort descending by coefficient
        top_indices = np.argsort(class_coefs)[::-1][:top_n]

        for rank, feat_idx in enumerate(top_indices, start=1):
            feat_name = feature_names[feat_idx]
            coef_val = float(class_coefs[feat_idx])
            records.append({
                "intent": intent,
                "rank": rank,
                "feature": feat_name,
                "coefficient": round(coef_val, 4)
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Inspect Top Predictive Features per Intent.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory containing trained model joblib files"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "tfidf" / "top_features_by_intent.csv",
        help="Path to save top features CSV"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top features per intent to extract (default: 20)"
    )

    args = parser.parse_args()

    print("=" * 78)
    print("      PHASE 6: INSPECT TF-IDF LOGISTIC REGRESSION COEFFICIENTS")
    print("=" * 78)

    # Extract features
    df_features = extract_top_features(args.models_dir, top_n=args.top_n)

    # Save to CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(args.output_csv, index=False)
    print(f"\n[Output] Saved top {args.top_n} features per intent to: {args.output_csv.name} ({len(df_features)} total rows)")

    # Print clean terminal audit
    print("\n" + "=" * 78)
    print("TOP PREDICTIVE FEATURES BY INTENT (TOP 5 PREVIEW)")
    print("=" * 78)

    for intent, group in df_features.groupby("intent", sort=False):
        print(f"\nIntent: [{intent}]")
        for _, row in group.head(5).iterrows():
            print(f"  #{row['rank']:<2} '{row['feature']:<25}' -> weight: {row['coefficient']:+.4f}")

    print("\n" + "=" * 78)
    print("INSPECTION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
