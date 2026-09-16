"""
make_smoke_data.py — Generate synthetic smoke-test fixtures.

Creates:
  data/demo/corpus.csv      — 20 synthetic historical customer→brand pairs
  data/demo/incoming.csv    — 5 incoming customer queries
  data/demo/golden_smoke.csv — 5 golden-labelled rows for smoke evaluation

No external dependencies. Runs in under 1 second.
"""

import csv
import os
import random

random.seed(42)

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "demo")
os.makedirs(DEMO_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Synthetic corpus (historical customer→brand pairs)
# ---------------------------------------------------------------------------
CORPUS_ROWS = [
    # software / OS
    ("c001", "My iPhone is stuck on the update screen and won't boot up.",
     "We're sorry to hear this. Try force-restarting: hold Side + Volume Down until the Apple logo appears. DM us if it persists.", "software_update_or_os_issue"),
    ("c002", "iOS 17 update keeps failing with error -1.",
     "Please try updating via iTunes/Finder on a Mac. If error continues, DM your device model and iOS version.", "software_update_or_os_issue"),
    ("c003", "Software update downloaded but won't install.",
     "Ensure you have at least 5 GB free storage and your battery is above 50%. Let us know if the issue continues.", "software_update_or_os_issue"),
    ("c004", "My phone restarted during update and now shows a black screen.",
     "This can happen due to interrupted updates. Force restart your device. If still unresponsive, visit an Apple Store.", "software_update_or_os_issue"),
    # hardware / performance
    ("c005", "My iPhone battery drains super fast since the last update.",
     "Background app refresh can drain battery. Go to Settings > Battery to see which apps use the most power.", "device_performance_or_hardware"),
    ("c006", "My MacBook Pro fan runs constantly and gets very hot.",
     "Excessive heat can indicate high CPU usage. Open Activity Monitor to identify the responsible process.", "device_performance_or_hardware"),
    ("c007", "Touchscreen not responding in some areas of my iPad.",
     "Please clean the screen and remove any accessories. If the issue persists this may be a hardware issue—visit an Apple Store.", "device_performance_or_hardware"),
    # connectivity
    ("c008", "My iPhone won't connect to Wi-Fi after the latest iOS update.",
     "Try forgetting the network and reconnecting: Settings > Wi-Fi > (i) > Forget This Network, then reconnect.", "connectivity_and_network"),
    ("c009", "Bluetooth keeps disconnecting from my AirPods.",
     "Reset your AirPods by holding the case button for 15 seconds. Then re-pair them in Settings > Bluetooth.", "connectivity_and_network"),
    # apps / iCloud
    ("c010", "My iCloud photos are not syncing across devices.",
     "Make sure iCloud Photos is on for all devices: Settings > [your name] > iCloud > Photos. Check your iCloud storage.", "apps_services_or_icloud"),
    ("c011", "App Store keeps saying my payment was declined.",
     "Please update your payment information in Settings > [your name] > Payment & Shipping. DM if issue continues.", "billing_subscription_or_purchase"),
    ("c012", "I was charged twice for Apple Music subscription.",
     "We're sorry about the double charge. Please DM us your Apple ID email so we can investigate this with our billing team.", "billing_subscription_or_purchase"),
    # account / Apple ID
    ("c013", "I forgot my Apple ID password and can't reset it.",
     "You can reset at iforgot.apple.com. If you still can't access it, DM us and we will guide you through account recovery.", "account_access_and_apple_id"),
    ("c014", "My account got locked after too many failed attempts.",
     "Your account is temporarily locked for security. Wait 24 hours or visit iforgot.apple.com to unlock it now.", "account_access_and_apple_id"),
    # repair / order
    ("c015", "My iPhone screen cracked and I need a replacement.",
     "We can help with screen repairs! Visit apple.com/support/repair or schedule a Genius Bar appointment.", "repair_replacement_or_order"),
    ("c016", "I returned my MacBook 10 days ago but still haven't got my refund.",
     "We apologize for the delay. Please DM your order number and we will escalate your refund request immediately.", "repair_replacement_or_order"),
    # other
    ("c017", "What are your store hours?",
     "Store hours vary by location. Find your nearest Apple Store and its hours at apple.com/retail.", "other_or_unclear"),
    ("c018", "Can I trade in my old iPhone for a new one?",
     "Yes! Visit apple.com/trade-in to see your trade-in value and start the process.", "other_or_unclear"),
    ("c019", "Do you offer student discounts?",
     "Yes, Apple offers education pricing. Visit apple.com/education-discount or visit a store with your student ID.", "other_or_unclear"),
    ("c020", "How do I contact Apple support by phone?",
     "You can reach Apple Support at 1-800-275-2273 or chat with us at getsupport.apple.com.", "other_or_unclear"),
]

corpus_path = os.path.join(DEMO_DIR, "corpus.csv")
with open(corpus_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "customer_text", "brand_response", "intent"])
    writer.writeheader()
    for row in CORPUS_ROWS:
        writer.writerow({"id": row[0], "customer_text": row[1], "brand_response": row[2], "intent": row[3]})
print(f"Written {len(CORPUS_ROWS)} corpus rows to {corpus_path}")


# ---------------------------------------------------------------------------
# 2. Incoming queries
# ---------------------------------------------------------------------------
INCOMING_ROWS = [
    ("i001", "My iPhone update is failing with error code -1. How do I fix it?"),
    ("i002", "My account got hacked and I can't log in to my Apple ID. Someone changed my password."),
    ("i003", "Wi-Fi keeps dropping on my iPhone 14 after iOS 17 update."),
    ("i004", "I was charged twice for my iCloud storage plan this month."),
    ("i005", "My AirPods Pro left earbud is much quieter than the right one."),
]

incoming_path = os.path.join(DEMO_DIR, "incoming.csv")
with open(incoming_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "text"])
    writer.writeheader()
    for row in INCOMING_ROWS:
        writer.writerow({"id": row[0], "text": row[1]})
print(f"Written {len(INCOMING_ROWS)} incoming queries to {incoming_path}")


# ---------------------------------------------------------------------------
# 3. Golden smoke labels (for evaluate.py smoke run)
# ---------------------------------------------------------------------------
GOLDEN_ROWS = [
    ("i001", "My iPhone update is failing with error code -1. How do I fix it?",
     "software_update_or_os_issue", "escalate", "DONE"),
    ("i002", "My account got hacked and I can't log in to my Apple ID. Someone changed my password.",
     "account_access_and_apple_id", "escalate", "DONE"),
    ("i003", "Wi-Fi keeps dropping on my iPhone 14 after iOS 17 update.",
     "connectivity_and_network", "escalate", "DONE"),
    ("i004", "I was charged twice for my iCloud storage plan this month.",
     "billing_subscription_or_purchase", "escalate", "DONE"),
    ("i005", "My AirPods Pro left earbud is much quieter than the right one.",
     "device_performance_or_hardware", "escalate", "DONE"),
]

golden_path = os.path.join(DEMO_DIR, "golden_smoke.csv")
with open(golden_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "text", "final_intent", "final_action", "status"])
    writer.writeheader()
    for row in GOLDEN_ROWS:
        writer.writerow({
            "id": row[0], "text": row[1],
            "final_intent": row[2], "final_action": row[3], "status": row[4],
        })
print(f"Written {len(GOLDEN_ROWS)} golden smoke rows to {golden_path}")

print("\nSmoke data ready in data/demo/")
print("  corpus.csv      —", len(CORPUS_ROWS), "historical pairs")
print("  incoming.csv    —", len(INCOMING_ROWS), "incoming queries")
print("  golden_smoke.csv—", len(GOLDEN_ROWS), "labelled rows")
