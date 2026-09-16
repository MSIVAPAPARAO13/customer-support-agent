"""
run_agent.py — CLI entry point for the Hiver AppleSupport Agent.

Usage:
    python scripts/run_agent.py \\
        --train data/demo/corpus.csv \\
        --input data/demo/incoming.csv \\
        --output outputs/smoke_predictions.csv

This wraps the full pipeline:
  1. Trains a TF-IDF + cosine-retrieval model on --train corpus.
  2. Classifies each row in --input.
  3. Retrieves up to 3 similar historical resolutions.
  4. Drafts a conservative reply.
  5. Applies the routing policy (escalate / auto_handle).
  6. Writes a prediction CSV to --output.

Works with both the synthetic data/demo/ corpus and the real
data/processed/corpus.csv built from twcs.csv.

No external API keys needed. Only requires: numpy, scipy, scikit-learn, pandas.
"""

import argparse
import csv
import re
import sys
import os
from pathlib import Path

# Allow both installed-package and in-repo imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _build_simple_tfidf(corpus_rows: list, ngram_max: int = 2, max_features: int = 10000):
    """Build a lightweight TF-IDF vectorizer from the training corpus."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [r["customer_text"] for r in corpus_rows]
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, ngram_max),
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(texts)
    return vec, matrix


def _classify(vec, matrix, intents: list, text: str):
    """Return (intent, confidence) using cosine similarity to nearest neighbour."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    q = vec.transform([text])
    sims = cosine_similarity(q, matrix).flatten()
    idx = int(np.argmax(sims))
    return intents[idx], float(sims[idx])


def _retrieve_top_k(vec, matrix, corpus_rows: list, text: str, k: int = 3):
    """Return top-k most similar corpus rows (excluding the query itself)."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    q = vec.transform([text])
    sims = cosine_similarity(q, matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:k]
    return [(corpus_rows[i], float(sims[i])) for i in top_idx]


RESTRICT_PATTERNS = [
    r"\bhack\w*\b", r"\bstolen\b", r"\bfraud\b", r"\bcompromis\w*\b",
    r"\bunauthori[sz]ed\b", r"\bidentity\b",
]


def _is_restricted(text: str) -> bool:
    for pat in RESTRICT_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


HIGH_RISK_INTENTS = {"account_access_and_apple_id", "billing_subscription_or_purchase", "repair_replacement_or_order"}
CONFIDENCE_THRESHOLD = 0.80
SIMILARITY_THRESHOLD = 0.50


def _route(intent: str, confidence: float, top_sim: float, is_restricted: bool):
    """Apply routing policy. Returns (action, reason)."""
    if is_restricted:
        return "escalate", "restricted_safety_flag"
    if intent in HIGH_RISK_INTENTS:
        return "escalate", "high_risk_intent"
    if intent == "other_or_unclear":
        return "escalate", "other_or_unclear"
    if confidence < CONFIDENCE_THRESHOLD:
        return "escalate", "uncertain_intent"
    if top_sim < SIMILARITY_THRESHOLD:
        return "escalate", "insufficient_historical_evidence"
    return "auto_handle", "eligible_low_risk_case"


def _draft_reply(intent: str, top_matches: list) -> str:
    """Draft a conservative reply using the best historical resolution."""
    if not top_matches:
        return "Thank you for contacting Apple Support. A specialist will follow up with you shortly."
    best_row, _ = top_matches[0]
    brand_response = best_row.get("brand_response", "")
    if brand_response:
        return f"[Based on similar past resolution] {brand_response}"
    return "Thank you for contacting Apple Support. We'll look into this and get back to you."


def main():
    parser = argparse.ArgumentParser(description="AppleSupport Agent — CLI inference")
    parser.add_argument("--train", required=True, help="Path to corpus CSV with columns: id,customer_text,brand_response,intent")
    parser.add_argument("--input", required=True, help="Path to incoming queries CSV with columns: id,text")
    parser.add_argument("--output", required=True, help="Output path for predictions CSV")
    parser.add_argument("--top-k", type=int, default=3, help="Number of historical matches to retrieve (default: 3)")
    args = parser.parse_args()

    # --- Load corpus ---
    corpus_rows = []
    with open(args.train, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            corpus_rows.append(row)
    print(f"[run_agent] Loaded {len(corpus_rows)} corpus rows from {args.train}")

    intents = [r["intent"] for r in corpus_rows]

    # --- Build TF-IDF ---
    vec, matrix = _build_simple_tfidf(corpus_rows)
    print(f"[run_agent] TF-IDF matrix: {matrix.shape}")

    # --- Load incoming queries ---
    queries = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
    print(f"[run_agent] Processing {len(queries)} incoming queries...")

    # --- Run inference ---
    os.makedirs(Path(args.output).parent, exist_ok=True)
    fieldnames = [
        "id", "text", "predicted_intent", "confidence",
        "top_similarity", "action", "reason", "draft_reply",
        "retrieved_id_1", "retrieved_sim_1",
        "retrieved_id_2", "retrieved_sim_2",
        "retrieved_id_3", "retrieved_sim_3",
    ]

    results = []
    for q in queries:
        qid = q.get("id", "")
        text = q.get("text", "")

        intent, conf = _classify(vec, matrix, intents, text)
        top_matches = _retrieve_top_k(vec, matrix, corpus_rows, text, k=args.top_k)
        top_sim = top_matches[0][1] if top_matches else 0.0
        restricted = _is_restricted(text)
        action, reason = _route(intent, conf, top_sim, restricted)
        draft = _draft_reply(intent, top_matches if action == "auto_handle" else [])

        row = {
            "id": qid, "text": text,
            "predicted_intent": intent, "confidence": round(conf, 4),
            "top_similarity": round(top_sim, 4),
            "action": action, "reason": reason, "draft_reply": draft,
        }
        for i, (match_row, sim) in enumerate(top_matches[:3], 1):
            row[f"retrieved_id_{i}"] = match_row.get("id", "")
            row[f"retrieved_sim_{i}"] = round(sim, 4)
        for i in range(len(top_matches) + 1, 4):
            row[f"retrieved_id_{i}"] = ""
            row[f"retrieved_sim_{i}"] = ""
        results.append(row)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # --- Print summary ---
    auto = sum(1 for r in results if r["action"] == "auto_handle")
    esc = sum(1 for r in results if r["action"] == "escalate")
    print(f"\n[run_agent] Done. {len(results)} predictions written to {args.output}")
    print(f"  auto_handle : {auto}")
    print(f"  escalate    : {esc}")
    print()
    for r in results:
        print(f"  {r['id']} | {r['action']:11s} | {r['reason']:35s} | {r['predicted_intent']}")


if __name__ == "__main__":
    main()
