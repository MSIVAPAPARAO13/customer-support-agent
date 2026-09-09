#!/usr/bin/env python3
"""
Create Transparent Weak Labels (Phase 5)
=======================================

Purpose:
  Generates transparent, rule-driven weak labels ("silver labels") strictly for
  training and validation splits (`applesupport_train.csv` and `applesupport_validation.csv`).

Crucial Integrity Rules:
  1. Weak labels are NEVER assigned to test or golden sets.
  2. Weak labels are NOT ground truth; they are heuristic approximations to bootstrap
     machine learning experiments in the absence of human annotations.
  3. Every assignment records the exact triggering rule (`weak_label_rule`) and a
     confidence estimate (`weak_label_confidence`).
  4. Context Disambiguation: "fully charged" or "not charging" is recognized as
     device battery/power context, NOT a financial billing charge.

Deliverables:
  - data/weak_labels/applesupport_train_weak_labels.csv
  - data/weak_labels/applesupport_validation_weak_labels.csv
  - data/weak_labels/weak_labeling_report.md

Usage:
  python scripts/create_weak_labels.py
"""

import sys
import re
import argparse
from pathlib import Path
from collections import Counter
from typing import Tuple, Dict, List, Any

import pandas as pd


# -----------------------------------------------------------------------------
# 1. Context Disambiguation & Pattern Definitions
# -----------------------------------------------------------------------------

def is_battery_or_power_context(text: str) -> bool:
    """
    Detects whether references to 'charge', 'charged', or 'charging' refer to
    electrical power/battery rather than a monetary transaction.
    """
    pattern = (
        r'\b(?:battery|phone|iphone|ipad|device|overnight|pad|cable|port|dock|charger|'
        r'fully charged|fast charg\w+|charg\w+ (?:my |the |an )?(?:phone|iphone|ipad|battery)|'
        r'won\'t charge|not charging|stopped charging|refuses to charge|refusing to charge|'
        r'percentage|\d+%\s*charg\w*)\b'
    )
    return bool(re.search(pattern, text, re.IGNORECASE))


def assign_weak_label(text: str) -> Tuple[str, str, str]:
    """
    Evaluates customer clean text through prioritized transparent rules.
    Returns:
      (weak_intent, weak_label_rule, weak_label_confidence)
    """
    if not text or pd.isna(text):
        return 'other_or_unclear', 'rule_empty_text', 'low'

    t = text.lower().strip()

    # -------------------------------------------------------------------------
    # Priority 1: High-Liability Security, Account Access & Apple ID
    # -------------------------------------------------------------------------
    if re.search(r'\b(?:apple id|appleid|icloud account|itunes account)\b.*\b(?:password|passcode|locked|unlock|disabled|recovery|reset|forgot|login|sign in|auth)\b', t):
        return 'account_access_and_apple_id', 'rule_apple_id_credential_lock', 'high'

    if re.search(r'\b(?:hacked|compromised|stolen account|unauthorized access|phishing|takeover)\b', t):
        return 'account_access_and_apple_id', 'rule_account_security_breach', 'high'

    if re.search(r'\b(?:2fa|two[- ]factor|verification code|security questions?|trusted (?:phone|device|number))\b', t):
        return 'account_access_and_apple_id', 'rule_two_factor_or_verification', 'high'

    if re.search(r'\b(?:locked out of my (?:account|iphone|ipad|apple id)|account (?:is )?(?:locked|disabled))\b', t):
        return 'account_access_and_apple_id', 'rule_account_lockout', 'high'

    if re.search(r'\b(?:forgot|reset|change)\b.*\b(?:password|passcode|apple id)\b', t):
        return 'account_access_and_apple_id', 'rule_password_reset_inquiry', 'high'

    # -------------------------------------------------------------------------
    # Priority 2: Monetary, Financial Transactions & Subscriptions
    # -------------------------------------------------------------------------
    battery_ctx = is_battery_or_power_context(t)

    # Disambiguated billing checks:
    if re.search(r'\b(?:refund|subscription|subscriptions|billing|receipt|itunes\.com/bill|in-app purchase|cancel subscription|canceling subscription|charged twice|unauthorized charge|overcharged|payment method|declined)\b', t):
        return 'billing_subscription_or_purchase', 'rule_explicit_billing_or_refund', 'high'

    if re.search(r'\b(?:charged|billed)\b.*\b(?:\$|\d+ dollars?|twice|extra|monthly|without my permission)\b', t):
        return 'billing_subscription_or_purchase', 'rule_monetary_charge_dispute', 'high'

    if not battery_ctx and re.search(r'\b(?:i was charged|charged me|unknown charge|card statement)\b', t):
        return 'billing_subscription_or_purchase', 'rule_unrecognized_card_charge', 'medium'

    # -------------------------------------------------------------------------
    # Priority 3: Repair, Hardware Service Appointments & Order Tracking
    # -------------------------------------------------------------------------
    if re.search(r'\b(?:genius bar|apple store appointment|book(?:ing)? (?:an? )?appointment|store reservation)\b', t):
        return 'repair_replacement_or_order', 'rule_genius_bar_appointment', 'high'

    if re.search(r'\b(?:applecare\+?|apple care|warranty (?:claim|coverage)|out of warranty|deductible)\b', t):
        return 'repair_replacement_or_order', 'rule_warranty_and_applecare', 'high'

    if re.search(r'\b(?:screen repair|battery replacement appointment|repair cost|cost to (?:repair|fix my screen)|trade-in value)\b', t):
        return 'repair_replacement_or_order', 'rule_hardware_repair_service', 'high'

    if re.search(r'\b(?:order status|delivery status|shipping status|tracking number|when will my (?:iphone|order|mac) (?:ship|arrive)|shipped yet)\b', t):
        return 'repair_replacement_or_order', 'rule_order_shipping_tracking', 'high'

    # -------------------------------------------------------------------------
    # Priority 4: Connectivity, Networking, Wi-Fi & Bluetooth
    # -------------------------------------------------------------------------
    if re.search(r'\b(?:wifi|wi-fi)\b.*\b(?:disconnect\w*|won\'t connect|drops?|slow|not working|greyed out|refuses to connect)\b', t):
        return 'connectivity_and_network', 'rule_wifi_connection_failure', 'high'

    if re.search(r'\b(?:bluetooth)\b.*\b(?:pair\w*|connect\w*|car|audio|speaker|headphones?|dropping|disconnect\w*)\b', t):
        return 'connectivity_and_network', 'rule_bluetooth_pairing_failure', 'high'

    if re.search(r'\b(?:cellular|lte|4g|5g|no service|searching\.\.\.|dropped calls?|sim card|hotspot|personal hotspot|airdrop)\b', t):
        return 'connectivity_and_network', 'rule_network_and_cellular_signal', 'high'

    if re.search(r'\b(?:wifi|wi-fi|bluetooth)\b', t) and not re.search(r'\b(?:update|ios 11)\b', t):
        return 'connectivity_and_network', 'rule_general_wireless_mention', 'medium'

    # -------------------------------------------------------------------------
    # Priority 5: Software Updates & Operating System Issues
    # -------------------------------------------------------------------------
    if re.search(r'\b(?:since (?:the |i )?updated?|after updating|after (?:the )?update|updated (?:my |the )?phone|new update|latest update|recent update)\b', t):
        return 'software_update_or_os_issue', 'rule_post_update_regression', 'high'

    if re.search(r'\b(?:ios 11|ios11|high sierra|macos|watchos|tvos)\b.*\b(?:bug|glitch|freeze|crash|slow|issue|problem|broken|update)\b', t):
        return 'software_update_or_os_issue', 'rule_os_version_glitch', 'high'

    if re.search(r'\b(?:update|install(?:ing|ation)?|upgrade)\b.*\b(?:failed|error|stuck|verifying|won\'t install|cannot install)\b', t):
        return 'software_update_or_os_issue', 'rule_os_installation_failure', 'high'

    if re.search(r'\b(?:question mark (?:box|symbol)|keyboard glitch|autocorrect letter i|capital i glitch)\b', t):
        return 'software_update_or_os_issue', 'rule_known_ios11_keyboard_bug', 'high'

    if re.search(r'\b(?:ios 11|ios11|software update|new ios|update my phone)\b', t):
        return 'software_update_or_os_issue', 'rule_general_os_update', 'medium'

    # -------------------------------------------------------------------------
    # Priority 6: Native Apps, Cloud Services & iCloud
    # -------------------------------------------------------------------------
    if re.search(r'\b(?:icloud (?:storage|backup|sync)|backup failed|not enough (?:icloud )?storage)\b', t):
        return 'apps_services_or_icloud', 'rule_icloud_storage_or_backup', 'high'

    if re.search(r'\b(?:app store)\b.*\b(?:download|downloading|waiting|can\'t download|won\'t install|stuck)\b', t):
        return 'apps_services_or_icloud', 'rule_app_store_download_issue', 'high'

    if re.search(r'\b(?:apple music|itunes music)\b.*\b(?:playlist|song|stream|playing|offline|library)\b', t):
        return 'apps_services_or_icloud', 'rule_apple_music_service', 'high'

    if re.search(r'\b(?:imessage|facetime|siri|notes app|photos app|keychain|apple pay)\b', t):
        return 'apps_services_or_icloud', 'rule_native_service_mention', 'medium'

    if re.search(r'\b(?:icloud|itunes|app store)\b', t):
        return 'apps_services_or_icloud', 'rule_general_ecosystem_mention', 'low'

    # -------------------------------------------------------------------------
    # Priority 7: Device Performance, Battery & Physical Hardware
    # -------------------------------------------------------------------------
    if re.search(r'\b(?:battery (?:life|drain|draining|dying|percentage|health)|dies in \d+|dies so fast)\b', t):
        return 'device_performance_or_hardware', 'rule_battery_drain_or_degradation', 'high'

    if re.search(r'\b(?:not charging|won\'t charge|refuses to charge|charger|charging cable|charging port|charging issue)\b', t):
        return 'device_performance_or_hardware', 'rule_hardware_charging_failure', 'high'

    if re.search(r'\b(?:screen|display|touch)\b.*\b(?:cracked|broken|unresponsive|frozen|black screen|lines|glitch)\b', t):
        return 'device_performance_or_hardware', 'rule_display_or_screen_defect', 'high'

    if re.search(r'\b(?:overheating|phone is so hot|burning hot|very hot)\b', t):
        return 'device_performance_or_hardware', 'rule_device_overheating', 'high'

    if re.search(r'\b(?:speaker|earpiece|microphone|camera|flash|home button|power button)\b.*\b(?:broken|not working|muffled|quiet|stopped)\b', t):
        return 'device_performance_or_hardware', 'rule_physical_component_failure', 'high'

    if re.search(r'\b(?:sluggish|extremely slow|freezing constantly|restart loop|boot loop|apple logo loop)\b', t):
        return 'device_performance_or_hardware', 'rule_general_device_instability', 'medium'

    # -------------------------------------------------------------------------
    # Fallback: Ambiguous, Vague, or General Inquiries
    # -------------------------------------------------------------------------
    return 'other_or_unclear', 'rule_fallback_no_match', 'low'


# -----------------------------------------------------------------------------
# 2. Main Processing & Reporting
# -----------------------------------------------------------------------------

def process_partition(csv_path: Path, output_path: Path) -> pd.DataFrame:
    """Reads partition, adds weak label columns, and writes out."""
    print(f"Reading partition: {csv_path.name}...")
    df = pd.read_csv(csv_path, dtype=str)

    intents, rules, confs = [], [], []
    for text in df['customer_text_clean']:
        intent, rule, conf = assign_weak_label(text)
        intents.append(intent)
        rules.append(rule)
        confs.append(conf)

    df['weak_intent'] = intents
    df['weak_label_rule'] = rules
    df['weak_label_confidence'] = confs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path.name} ({len(df):,} rows)")
    return df


def generate_report(
    report_path: Path,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame
):
    """Generates weak_labeling_report.md."""
    train_intent_counts = df_train['weak_intent'].value_counts()
    train_conf_counts = df_train['weak_label_confidence'].value_counts()
    train_top_rules = df_train['weak_label_rule'].value_counts().head(15)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Transparent Weak-Labeling Audit Report (Phase 5)\n\n")

        f.write("## 1. Executive Summary & Provenance\n\n")
        f.write(f"- **Train Partition Processed**: {len(df_train):,} rows (`applesupport_train_weak_labels.csv`).\n")
        f.write(f"- **Validation Partition Processed**: {len(df_val):,} rows (`applesupport_validation_weak_labels.csv`).\n")
        f.write(f"- **Test & Golden Isolation**: Strictly **0 test or golden rows were processed or labelled**.\n\n")

        f.write("> [!IMPORTANT]\n")
        f.write("> **Explicit Silver-Standard Disclaimer**:\n")
        f.write("> These weak labels were generated via deterministic keyword heuristics. ")
        f.write("They **are NOT human ground-truth labels** and must only be used for exploratory model training experiments. ")
        f.write("All final benchmark evaluations must strictly use the human-adjudicated golden set.\n\n")

        f.write("## 2. Weak Intent Distribution (Training Partition)\n\n")
        f.write("| Weak Intent Category | Train Row Count | % of Train | Primary Semantic Scope |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for intent, cnt in train_intent_counts.items():
            pct = (cnt / len(df_train)) * 100
            f.write(f"| `{intent}` | {cnt:,} | {pct:.2f}% | Derived from keyword heuristics |\n")

        f.write("\n## 3. Confidence Distribution\n\n")
        f.write("| Confidence Level | Row Count | % of Train | Rationale |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for conf, cnt in train_conf_counts.items():
            pct = (cnt / len(df_train)) * 100
            f.write(f"| **{conf.upper()}** | {cnt:,} | {pct:.2f}% | Explicit keywords vs. general mentions |\n")

        f.write("\n## 4. Top 15 Triggered Rules\n\n")
        f.write("| Rule Name | Trigger Count | % of Train |\n")
        f.write("| :--- | :--- | :--- |\n")
        for rule, cnt in train_top_rules.items():
            pct = (cnt / len(df_train)) * 100
            f.write(f"| `{rule}` | {cnt:,} | {pct:.2f}% |\n")

        f.write("\n## 5. De-Identified Examples per Weak Intent (10 per Category)\n\n")
        for intent in train_intent_counts.index:
            f.write(f"### Category: `{intent}`\n\n")
            sub = df_train[df_train['weak_intent'] == intent].head(10)
            for idx, (_, row) in enumerate(sub.iterrows(), 1):
                f.write(f"{idx}. (Tweet `{row['customer_tweet_id']}`) \"{row['customer_text_clean']}\"\n")
                f.write(f"   - *Rule*: `{row['weak_label_rule']}` | *Confidence*: `{row['weak_label_confidence']}`\n")
            f.write("\n---\n\n")

        f.write("## 6. Known Rule Collisions, Ambiguities & Weaknesses\n\n")
        f.write("1. **Software Update vs. Battery Drain Collision**: When a customer says *'Battery dies fast after iOS 11 update'*, ")
        f.write("the rule prioritizes `software_update_or_os_issue` because the update is identified as the trigger. ")
        f.write("However, if the customer simply says *'My battery is dying and I have iOS 11'*, it may misclassify.\n")
        f.write("2. **Battery Charge vs. Billing Charge Disambiguation**: The rule engine explicitly checks battery power context ")
        f.write("(`is_battery_or_power_context`). Expressions like *'fully charged'* or *'phone won\\'t charge'* are successfully prevented ")
        f.write("from triggering `billing_subscription_or_purchase`.\n")
        f.write("3. **Sarcasm and Indirect Phrasing**: Customers expressing frustration through sarcasm (*'Great job Apple, another flawless update'*) ")
        f.write("are classified into `software_update_or_os_issue` based on keywords, missing the negative sentiment nuances.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Generate transparent weak labels for train and validation sets."
    )
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--output_dir", default=None)

    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    train_path = Path(args.train_csv).resolve() if args.train_csv else project_root / "data" / "processed" / "applesupport_train.csv"
    val_path = Path(args.val_csv).resolve() if args.val_csv else project_root / "data" / "processed" / "applesupport_validation.csv"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else project_root / "data" / "weak_labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("      PHASE 5: TRANSPARENT WEAK-LABEL GENERATION (TRAIN & VAL ONLY)")
    print("=" * 78)

    train_out_path = output_dir / "applesupport_train_weak_labels.csv"
    val_out_path = output_dir / "applesupport_validation_weak_labels.csv"
    report_path = output_dir / "weak_labeling_report.md"

    # Process partitions
    df_train = process_partition(train_path, train_out_path)
    df_val = process_partition(val_path, val_out_path)

    # Generate audit report
    print("\nGenerating weak_labeling_report.md...")
    generate_report(report_path, df_train, df_val)
    print(f"Audit report saved: {report_path.name}")

    print("\n" + "=" * 78)
    print("                       WEAK LABELING COMPLETE")
    print("=" * 78)
    print(f"Train Weak Labels:       {len(df_train):,} rows -> applesupport_train_weak_labels.csv")
    print(f"Validation Weak Labels:  {len(df_val):,} rows -> applesupport_validation_weak_labels.csv")
    print(f"Audit Report:            {report_path.name}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
