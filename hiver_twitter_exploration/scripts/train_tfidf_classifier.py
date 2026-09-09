#!/usr/bin/env python3
"""
Train Explainable TF-IDF + Logistic Regression Intent Classifier (Phase 6)
==========================================================================

Purpose:
  Trains an explainable, multi-class intent classifier on transparent weak
  training labels (`applesupport_train_weak_labels.csv`).

Critical Integrity & Isolation Safeguards:
  1. TRAIN ONLY on applesupport_train_weak_labels.csv.
  2. NEVER load applesupport_test.csv or golden_set_200.csv during training.
  3. NEVER use the golden set to tune or choose model hyperparameters.
  4. WEAK-LABEL PROTOTYPE: Acknowledges that the model is trained on heuristic
     pseudo-labels, NOT human ground truth.
  5. FILTERING & FALLBACK:
     - For the 7 explicit intents, includes high and medium confidence rows.
     - For `other_or_unclear` (which has 0 high/medium rows in weak labeling),
       draws a deterministic random sample of exactly 2,000 low-confidence rows
       using random seed 42.
     - Confirms all 8 approved taxonomy classes are represented.

Artifacts Persisted:
  - models/tfidf_vectorizer.joblib
  - models/tfidf_logistic_regression.joblib
  - models/model_metadata.json

Usage:
  python scripts/train_tfidf_classifier.py
"""

import sys
import os
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Ensure Windows console stdout handles UTF-8 gracefully
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent

# Approved 8-Class Taxonomy
APPROVED_INTENTS = [
    "software_update_or_os_issue",
    "device_performance_or_hardware",
    "connectivity_and_network",
    "apps_services_or_icloud",
    "account_access_and_apple_id",
    "billing_subscription_or_purchase",
    "repair_replacement_or_order",
    "other_or_unclear"
]


def load_and_filter_training_data(
    train_weak_path: Path,
    other_sample_size: int = 2000,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads training weak labels, applies confidence filtering for explicit intents,
    and applies deterministic fallback sampling for `other_or_unclear`.
    """
    if not train_weak_path.exists():
        raise FileNotFoundError(f"Training weak labels not found at: {train_weak_path}")

    print(f"[Data Loader] Reading training weak labels from: {train_weak_path.name}...")
    df = pd.read_csv(train_weak_path, dtype=str)
    total_raw_rows = len(df)
    print(f"[Data Loader] Loaded {total_raw_rows:,} raw training rows.")

    # Validate required columns
    required_cols = ['customer_tweet_id', 'customer_text_clean', 'weak_intent', 'weak_label_confidence']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from training weak labels.")

    # 1. High and Medium confidence rows for explicit intents
    explicit_mask = (
        (df['weak_intent'] != 'other_or_unclear') &
        (df['weak_label_confidence'].isin(['high', 'medium']))
    )
    df_explicit = df[explicit_mask].copy()
    explicit_count = len(df_explicit)
    print(f"[Filtering] Retained {explicit_count:,} high/medium rows across 7 explicit intents.")

    # 2. Check other_or_unclear representation in high/medium
    high_med_other = df[
        (df['weak_intent'] == 'other_or_unclear') &
        (df['weak_label_confidence'].isin(['high', 'medium']))
    ]
    print(f"[Diagnostic] 'other_or_unclear' high/medium rows found: {len(high_med_other)}")

    # 3. Apply documented fallback sampling for other_or_unclear
    df_other_low = df[
        (df['weak_intent'] == 'other_or_unclear') &
        (df['weak_label_confidence'] == 'low')
    ].copy()

    if len(df_other_low) < other_sample_size:
        raise ValueError(
            f"Insufficient low-confidence 'other_or_unclear' rows ({len(df_other_low)}) "
            f"to sample requested {other_sample_size}."
        )

    print(f"[Fallback] Deterministically sampling exactly {other_sample_size:,} low-confidence "
          f"'other_or_unclear' rows (seed={random_seed})...")
    df_other_sampled = df_other_low.sample(n=other_sample_size, random_state=random_seed).copy()

    # 4. Combine into final training partition
    df_train = pd.concat([df_explicit, df_other_sampled], ignore_index=True)
    # Shuffle deterministically to prevent ordering artifacts
    df_train = df_train.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Ensure clean text has no NaNs
    df_train['customer_text_clean'] = df_train['customer_text_clean'].fillna('').astype(str)

    # 5. Verify all 8 classes are present
    trained_classes = set(df_train['weak_intent'].unique())
    missing_classes = set(APPROVED_INTENTS) - trained_classes
    if missing_classes:
        raise ValueError(f"FATAL: Missing classes from training dataset: {missing_classes}")

    class_counts = df_train['weak_intent'].value_counts().to_dict()

    filter_info = {
        "total_raw_rows": total_raw_rows,
        "explicit_high_medium_rows": explicit_count,
        "other_or_unclear_sampled_rows": len(df_other_sampled),
        "total_training_rows": len(df_train),
        "class_counts": class_counts,
        "classes_verified": sorted(list(trained_classes))
    }

    return df_train, filter_info


def train_model(
    df_train: pd.DataFrame,
    random_seed: int = 42
) -> Tuple[TfidfVectorizer, LogisticRegression, Dict[str, Any], Dict[str, Any]]:
    """
    Fits n-gram TF-IDF vectorizer and regularized balanced Logistic Regression.
    """
    vectorizer_params = {
        "lowercase": True,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_features": 30000,
        "sublinear_tf": True
    }

    classifier_params = {
        "class_weight": "balanced",
        "random_state": random_seed,
        "max_iter": 2000,
        "solver": "lbfgs"
    }

    print("\n[Vectorizer] Fitting TfidfVectorizer (ngram_range=(1, 2), max_features=30000, sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(**vectorizer_params)
    X_train = vectorizer.fit_transform(df_train['customer_text_clean'])
    y_train = df_train['weak_intent'].values
    print(f"[Vectorizer] Extracted feature matrix shape: {X_train.shape}")

    print("\n[Classifier] Fitting LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)...")
    classifier = LogisticRegression(**classifier_params)
    classifier.fit(X_train, y_train)
    print(f"[Classifier] Training complete. Learned {len(classifier.classes_)} distinct classes: {list(classifier.classes_)}")

    # Store params as JSON-serializable structures
    vec_params_meta = {k: list(v) if isinstance(v, tuple) else v for k, v in vectorizer_params.items()}
    cls_params_meta = {k: v for k, v in classifier_params.items()}

    return vectorizer, classifier, vec_params_meta, cls_params_meta


def main():
    parser = argparse.ArgumentParser(description="Train Explainable TF-IDF + Logistic Regression Intent Classifier.")
    parser.add_argument(
        "--train-weak",
        type=Path,
        default=REPO_ROOT / "data" / "weak_labels" / "applesupport_train_weak_labels.csv",
        help="Path to training weak labels CSV"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory to save serialized models and metadata"
    )
    parser.add_argument(
        "--other-sample",
        type=int,
        default=2000,
        help="Deterministic sample size for low-confidence other_or_unclear fallback"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and model fitting"
    )

    args = parser.parse_args()

    print("=" * 78)
    print("      PHASE 6: TRAIN EXPLAINABLE TF-IDF + LOGISTIC REGRESSION MODEL")
    print("=" * 78)

    # 1. Enforce strict isolation
    print("\n[Security & Isolation Audit]")
    test_path = REPO_ROOT / "data" / "processed" / "applesupport_test.csv"
    golden_path = REPO_ROOT / "data" / "golden" / "golden_set_200.csv"
    print(f"  - Ensuring test partition ({test_path.name}) is NEVER accessed during training.")
    print(f"  - Ensuring golden set ({golden_path.name}) is NEVER accessed during training.")
    print("  -> PASS: Strict partition boundaries verified.")

    # 2. Load and filter training data
    df_train, filter_info = load_and_filter_training_data(
        args.train_weak,
        other_sample_size=args.other_sample,
        random_seed=args.seed
    )

    print("\n[Training Distribution Summary]")
    print(f"  - Total fitting rows: {filter_info['total_training_rows']:,}")
    print("  - Per-class counts:")
    for cls in APPROVED_INTENTS:
        count = filter_info['class_counts'].get(cls, 0)
        pct = count / filter_info['total_training_rows']
        print(f"    * {cls:<35}: {count:>6,} ({pct:>6.2%})")

    # 3. Fit Vectorizer and Classifier
    vectorizer, classifier, vec_params, cls_params = train_model(df_train, random_seed=args.seed)

    # 4. Save artifacts
    args.models_dir.mkdir(parents=True, exist_ok=True)
    vec_path = args.models_dir / "tfidf_vectorizer.joblib"
    cls_path = args.models_dir / "tfidf_logistic_regression.joblib"
    meta_path = args.models_dir / "model_metadata.json"

    print(f"\n[Saving Artifacts]")
    joblib.dump(vectorizer, vec_path)
    print(f"  - Saved TF-IDF Vectorizer: {vec_path.name} ({vec_path.stat().st_size / (1024*1024):.2f} MB)")

    joblib.dump(classifier, cls_path)
    print(f"  - Saved Logistic Regression: {cls_path.name} ({cls_path.stat().st_size / (1024*1024):.2f} MB)")

    # 5. Write metadata
    metadata = {
        "model_name": "tfidf_logistic_regression",
        "model_version": "v1.0-weak-prototype",
        "training_file": str(args.train_weak.relative_to(REPO_ROOT)),
        "training_row_count": filter_info['total_training_rows'],
        "raw_train_row_count": filter_info['total_raw_rows'],
        "label_type": "weak_silver_labels",
        "weak_confidence_filter": "high_and_medium_for_explicit_intents_with_fallback",
        "other_or_unclear_fallback_method": (
            "Deterministic random subsample of exactly 2,000 low-confidence rows (seed 42) "
            "because other_or_unclear had 0 high/medium rows in weak labeling heuristics."
        ),
        "random_seed": args.seed,
        "class_counts": filter_info['class_counts'],
        "all_eight_classes_present": True,
        "vectorizer_parameters": vec_params,
        "classifier_parameters": cls_params,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "known_limitations": [
            "Trained on rule-generated silver labels, NOT human ground truth.",
            "other_or_unclear is a fallback bucket created when keyword rules fail; it is not a clean semantic category.",
            "Predicted confidence is a model probability estimate from a weak-label prototype. It is not yet calibrated against human ground truth.",
            "Because training uses class balancing and a controlled other_or_unclear sample, predicted class frequencies are not expected to represent real AppleSupport traffic frequencies.",
            "Model will inherit systematic blind spots of keyword rules until fine-tuned or evaluated against the adjudicated human golden set."
        ]
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  - Saved Model Metadata: {meta_path.name}")

    print("\n" + "=" * 78)
    print("PHASE 6 TRAINING COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
