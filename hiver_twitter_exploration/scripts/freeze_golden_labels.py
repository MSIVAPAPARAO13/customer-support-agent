#!/usr/bin/env python3
"""
Freeze Golden Labels Manifest Engine (Phase 10)
==============================================

Purpose:
  Freezes the completed, human-adjudicated golden benchmark (`adjudication_sheet.csv`)
  by validating completion of all 200 human labels, computing a cryptographic SHA-256
  digest, and writing `data/golden/golden_labels_freeze_manifest.json`.

Critical Anti-Tamper Rule:
  This freeze must occur BEFORE running golden inference.
  Once frozen, golden inference verifies the SHA-256 checksum against the manifest.
  This guarantees that human golden labels cannot be altered after model predictions
  become visible.

Integrity Requirements:
  - Exactly 200 rows with final_status == 'DONE'.
  - Valid final_intent categories from approved 8-intent taxonomy.
  - Valid final_action ('auto_handle' or 'escalate').
  - Refuses to overwrite existing manifest unless --force is supplied.

Usage:
  python scripts/freeze_golden_labels.py [--force]
"""

import sys
import os
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# Configure console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent

APPROVED_INTENTS = {
    "software_update_or_os_issue",
    "device_performance_or_hardware",
    "connectivity_and_network",
    "apps_services_or_icloud",
    "account_access_and_apple_id",
    "billing_subscription_or_purchase",
    "repair_replacement_or_order",
    "other_or_unclear",
}

APPROVED_ACTIONS = {"auto_handle", "escalate"}


def compute_sha256(filepath: Path) -> str:
    """Computes standard SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def freeze_golden_labels(
    adjudication_path: Path = REPO_ROOT / "data" / "golden" / "adjudication_sheet.csv",
    manifest_path: Path = REPO_ROOT / "data" / "golden" / "golden_labels_freeze_manifest.json",
    force: bool = False,
) -> bool:
    """
    Validates completed human adjudication and writes cryptographic freeze manifest.
    """
    if not adjudication_path.exists():
        raise FileNotFoundError(f"Adjudication sheet not found at: {adjudication_path}")

    if manifest_path.exists() and not force:
        print(f"[FREEZE REFUSED] Freeze manifest already exists at: {manifest_path}.")
        print("To overwrite an existing freeze manifest, you must explicitly supply --force.")
        return False

    df = pd.read_csv(adjudication_path, low_memory=False)

    # 1. Validate exactly 200 rows
    if len(df) != 200:
        print(f"[FREEZE REFUSED] Expected exactly 200 rows, found {len(df)}.")
        return False

    # 2. Validate all 200 rows are DONE
    status_counts = df["final_status"].fillna("MISSING").value_counts().to_dict()
    if status_counts.get("DONE", 0) != 200:
        print(f"[FREEZE REFUSED] Incomplete human adjudication: {status_counts.get('DONE', 0)}/200 rows DONE.")
        print(f"Status distribution: {status_counts}")
        print("Cannot freeze unadjudicated or pending human labels.")
        return False

    # 3. Validate intents and actions
    invalid_intents = [i for i in df["final_intent"] if i not in APPROVED_INTENTS]
    invalid_actions = [a for a in df["final_action"] if a not in APPROVED_ACTIONS]

    if invalid_intents:
        print(f"[FREEZE REFUSED] Found invalid final_intent values: {set(invalid_intents)}")
        return False
    if invalid_actions:
        print(f"[FREEZE REFUSED] Found invalid final_action values: {set(invalid_actions)}")
        return False

    # 4. Compute cryptographic SHA-256
    sha256_hash = compute_sha256(adjudication_path)
    now_utc = datetime.now(timezone.utc).isoformat()

    manifest_data = {
        "adjudication_file": "data/golden/adjudication_sheet.csv",
        "adjudication_sha256": sha256_hash,
        "freeze_timestamp": now_utc,
        "golden_row_count": len(df),
        "final_status_summary": status_counts,
        "intent_distribution": df["final_intent"].value_counts().to_dict(),
        "action_distribution": df["final_action"].value_counts().to_dict(),
        "anti_tamper_rule": "Golden inference and final evaluation verify this SHA-256 checksum before execution.",
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[FREEZE SUCCESSFUL] Golden labels frozen to manifest: {manifest_path}")
    print(f"  SHA-256 Checksum : {sha256_hash}")
    print(f"  Timestamp (UTC)  : {now_utc}")
    print(f"  Row Count        : {len(df)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Freeze Golden Labels Manifest (Phase 10).")
    parser.add_argument("--adjudication-file", type=Path, default=REPO_ROOT / "data" / "golden" / "adjudication_sheet.csv")
    parser.add_argument("--manifest-file", type=Path, default=REPO_ROOT / "data" / "golden" / "golden_labels_freeze_manifest.json")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing freeze manifest.")
    args = parser.parse_args()

    print("=" * 75)
    print("PHASE 10: FREEZE GOLDEN LABELS MANIFEST")
    print("=" * 75)
    print(f"Adjudication File : {args.adjudication_file}")
    print(f"Manifest File     : {args.manifest_file}")
    print(f"Force Overwrite   : {args.force}")
    print("=" * 75)

    success = freeze_golden_labels(args.adjudication_file, args.manifest_file, args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
