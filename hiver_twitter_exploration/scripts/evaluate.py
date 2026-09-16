"""
evaluate.py — Compute intent accuracy, macro F1, selective accuracy, and coverage.

Usage (smoke run):
    python scripts/evaluate.py \\
        --train data/demo/corpus.csv \\
        --golden data/demo/golden_smoke.csv \\
        --out outputs/smoke_metrics.json

Usage (real results):
    python scripts/evaluate.py \\
        --train data/processed/corpus.csv \\
        --golden data/golden/golden_set.csv \\
        --out outputs/real_metrics.json \\
        --judge-validation data/validation/judge_validation.csv

Only rows with status=DONE in the golden file are evaluated.
Metrics written to --out as JSON. Also prints a summary table.
"""

import argparse
import csv
import json
import os
import sys
import re
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Inline TF-IDF classifier (mirrors run_agent.py — no joblib dependency)
# ---------------------------------------------------------------------------

def _build_tfidf(corpus_rows):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [r["customer_text"] for r in corpus_rows]
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=10000, sublinear_tf=True)
    mat = vec.fit_transform(texts)
    return vec, mat


def _classify_batch(vec, mat, intents, texts):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    q = vec.transform(texts)
    sims = cosine_similarity(q, mat)
    idxs = sims.argmax(axis=1)
    confs = sims.max(axis=1)
    return [(intents[i], float(c)) for i, c in zip(idxs, confs)]


RESTRICT_PATTERNS = [r"\bhack\w*\b", r"\bstolen\b", r"\bfraud\b", r"\bcompromis\w*\b",
                     r"\bunauthori[sz]ed\b", r"\bidentity\b"]
HIGH_RISK_INTENTS = {"account_access_and_apple_id", "billing_subscription_or_purchase", "repair_replacement_or_order"}
CONFIDENCE_THRESHOLD = 0.80
SIMILARITY_THRESHOLD = 0.50


def _is_restricted(text):
    return any(re.search(p, text, re.IGNORECASE) for p in RESTRICT_PATTERNS)


def _route(intent, conf, top_sim, restricted):
    if restricted:
        return "escalate"
    if intent in HIGH_RISK_INTENTS:
        return "escalate"
    if intent == "other_or_unclear":
        return "escalate"
    if conf < CONFIDENCE_THRESHOLD:
        return "escalate"
    if top_sim < SIMILARITY_THRESHOLD:
        return "escalate"
    return "auto_handle"


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def macro_f1(y_true, y_pred):
    labels = sorted(set(y_true))
    f1s = []
    for label in labels:
        tp = sum(t == p == label for t, p in zip(y_true, y_pred))
        fp = sum(p == label and t != label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def bootstrap_ci(y_true, y_pred, metric_fn, n=500, seed=42):
    rng = random.Random(seed)
    n_samples = len(y_true)
    vals = []
    for _ in range(n):
        idxs = [rng.randint(0, n_samples - 1) for _ in range(n_samples)]
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        vals.append(metric_fn(yt, yp))
    vals.sort()
    lo = vals[int(0.025 * n)]
    hi = vals[int(0.975 * n)]
    return round(lo, 4), round(hi, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hiver Agent — Evaluation harness")
    parser.add_argument("--train", required=True, help="Corpus CSV (id,customer_text,brand_response,intent)")
    parser.add_argument("--golden", required=True, help="Golden CSV (id,text,final_intent,final_action,status,...)")
    parser.add_argument("--out", required=True, help="Output path for metrics JSON")
    parser.add_argument("--judge-validation", default=None,
                        help="Optional: judge validation CSV for reply acceptance rate")
    parser.add_argument("--bootstrap-n", type=int, default=500, help="Bootstrap iterations (default 500)")
    args = parser.parse_args()

    # --- Load corpus ---
    corpus_rows = []
    with open(args.train, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            corpus_rows.append(row)
    print(f"[evaluate] Corpus: {len(corpus_rows)} rows")

    intents_corpus = [r["intent"] for r in corpus_rows]
    vec, mat = _build_tfidf(corpus_rows)

    # --- Load golden (only DONE rows) ---
    golden_rows = []
    with open(args.golden, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status", "DONE").upper() == "DONE":
                golden_rows.append(row)
    print(f"[evaluate] Golden rows with status=DONE: {len(golden_rows)}")

    if not golden_rows:
        print("[evaluate] ERROR: No DONE rows found in golden file.")
        sys.exit(1)

    texts = [r.get("text", r.get("customer_text", "")) for r in golden_rows]
    y_true_intent = [r.get("final_intent", r.get("intent", "")) for r in golden_rows]
    y_true_action = [r.get("final_action", r.get("action", "")) for r in golden_rows]

    # Run classification
    preds = _classify_batch(vec, mat, intents_corpus, texts)
    y_pred_intent = [p[0] for p in preds]
    confs = [p[1] for p in preds]
    restricteds = [_is_restricted(t) for t in texts]

    # Compute top similarity (approximate via confidence since we're using 1-NN)
    top_sims = confs  # In 1-NN retrieval, confidence == top similarity
    y_pred_action = [
        _route(intent, conf, sim, restr)
        for intent, conf, sim, restr in zip(y_pred_intent, confs, top_sims, restricteds)
    ]

    # --- Intent metrics ---
    acc = accuracy(y_true_intent, y_pred_intent)
    mf1 = macro_f1(y_true_intent, y_pred_intent)
    ci_acc = bootstrap_ci(y_true_intent, y_pred_intent, accuracy, n=args.bootstrap_n)
    ci_f1 = bootstrap_ci(y_true_intent, y_pred_intent, macro_f1, n=args.bootstrap_n)

    # --- Routing metrics ---
    auto_mask = [a == "auto_handle" for a in y_pred_action]
    auto_count = sum(auto_mask)
    coverage = auto_count / len(golden_rows)
    # Selective accuracy: among auto_handle predictions, how often is intent correct?
    auto_correct = sum(
        yt == yp for yt, yp, is_auto in zip(y_true_intent, y_pred_intent, auto_mask) if is_auto
    )
    sel_acc = (auto_correct / auto_count) if auto_count > 0 else float("nan")

    # Action accuracy
    action_acc = accuracy(y_true_action, y_pred_action)

    # --- Reply acceptance (from judge validation file if provided) ---
    reply_acceptance = None
    if args.judge_validation and os.path.exists(args.judge_validation):
        accept_count = 0
        total_judge = 0
        with open(args.judge_validation, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                verdict = row.get("judge_overall_accept", row.get("verdict", "")).strip().upper()
                if verdict in ("ACCEPT", "REJECT"):
                    total_judge += 1
                    if verdict == "ACCEPT":
                        accept_count += 1
        if total_judge > 0:
            reply_acceptance = round(accept_count / total_judge, 4)
    elif os.path.exists("outputs/final_evaluation/llm_judge_outputs.csv"):
        # Use existing LLM judge output if available
        accept_count = total_judge = 0
        with open("outputs/final_evaluation/llm_judge_outputs.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                verdict = row.get("judge_overall_accept", "").strip().upper()
                if verdict in ("ACCEPT", "REJECT"):
                    total_judge += 1
                    if verdict == "ACCEPT":
                        accept_count += 1
        if total_judge > 0:
            reply_acceptance = round(accept_count / total_judge, 4)

    # --- Per-class F1 ---
    labels = sorted(set(y_true_intent))
    per_class = {}
    for label in labels:
        tp = sum(t == p == label for t, p in zip(y_true_intent, y_pred_intent))
        fp = sum(p == label and t != label for t, p in zip(y_true_intent, y_pred_intent))
        fn = sum(t == label and p != label for t, p in zip(y_true_intent, y_pred_intent))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[label] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                            "support": sum(t == label for t in y_true_intent)}

    # --- Build output ---
    metrics = {
        "metadata": {
            "train_corpus": args.train,
            "golden_file": args.golden,
            "golden_row_count": len(golden_rows),
            "bootstrap_n": args.bootstrap_n,
        },
        "intent_accuracy": round(acc, 4),
        "macro_f1": round(mf1, 4),
        "intent_accuracy_95ci": list(ci_acc),
        "macro_f1_95ci": list(ci_f1),
        "auto_handle_count": auto_count,
        "auto_handle_coverage": round(coverage, 4),
        "selective_accuracy": round(sel_acc, 4) if sel_acc == sel_acc else None,
        "action_accuracy": round(action_acc, 4),
        "reply_acceptance": reply_acceptance,
        "per_class_f1": per_class,
    }

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # --- Print table ---
    print()
    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Intent Accuracy  : {acc:.1%}  95CI [{ci_acc[0]:.1%}, {ci_acc[1]:.1%}]")
    print(f"  Macro F1         : {mf1:.1%}  95CI [{ci_f1[0]:.1%}, {ci_f1[1]:.1%}]")
    print(f"  Auto-handle count: {auto_count}/{len(golden_rows)} ({coverage:.1%} coverage)")
    if auto_count > 0:
        print(f"  Selective Acc    : {sel_acc:.1%}  (accuracy among auto_handle rows)")
    print(f"  Action Accuracy  : {action_acc:.1%}")
    if reply_acceptance is not None:
        print(f"  Reply Acceptance : {reply_acceptance:.1%}")
    print()
    print("  Per-class F1:")
    for label, s in per_class.items():
        print(f"    {label:45s}  F1={s['f1']:.3f}  (n={s['support']})")
    print("=" * 60)
    print(f"\n  Metrics written to: {args.out}")


if __name__ == "__main__":
    main()
