#!/usr/bin/env python3
"""
Interactive Customer Support Agent Web Server & REST API
=========================================================

Zero-dependency HTTP server delivering real-time intent classification,
dialogue retrieval, safe reply drafting, and risk routing.
"""

import sys
import os
import json
import mimetypes
import argparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any

import pandas as pd

# Ensure Windows console stdout handles UTF-8 gracefully
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent_service import SupportAgentService

# Pre-defined realistic test samples spanning different categories & risk levels
SAMPLE_INQUIRIES = [
    {
        "id": "battery_drain",
        "title": "Battery Drain After Update",
        "category": "device_performance_or_hardware",
        "expected_action": "auto_handle",
        "text": "Ever since updating my iPhone 7 to iOS 11, the battery drains from 100% to dead in less than two hours. Any advice?",
        "description": "Common low-risk hardware/performance issue with high historical evidence."
    },
    {
        "id": "wifi_issue",
        "title": "Wi-Fi Connection Dropping",
        "category": "connectivity_and_network",
        "expected_action": "auto_handle",
        "text": "@AppleSupport My iPhone keeps dropping the home Wi-Fi connection every 10 minutes while other devices stay connected.",
        "description": "Network connectivity problem matching safe diagnostic troubleshooting protocols."
    },
    {
        "id": "keyboard_glitch",
        "title": "iOS Update Keyboard Lag",
        "category": "software_update_or_os_issue",
        "expected_action": "auto_handle",
        "text": "After the new iOS update, typing has massive lag and the letter 'I' turns into a strange symbol when I type.",
        "description": "Widely reported OS update glitch with extensive historical troubleshooting."
    },
    {
        "id": "icloud_sync",
        "title": "iCloud Photos Not Syncing",
        "category": "apps_services_or_icloud",
        "expected_action": "auto_handle",
        "text": "My photos taken on iPhone are not syncing to iCloud Photo Library or my MacBook. Storage has 50GB free.",
        "description": "Cloud service sync question that can qualify for auto-handling if evidence is strong."
    },
    {
        "id": "account_hacked",
        "title": "Apple ID Compromised",
        "category": "account_access_and_apple_id",
        "expected_action": "escalate",
        "text": "@AppleSupport Help! Someone hacked into my Apple ID account, changed my recovery email, and locked me out of my phone!",
        "description": "CRITICAL SECURITY RISK: Must trigger restricted safety flags and escalate immediately."
    },
    {
        "id": "billing_charge",
        "title": "Unauthorized Subscription Charge",
        "category": "billing_subscription_or_purchase",
        "expected_action": "escalate",
        "text": "I was just charged $14.99 for an Apple Music subscription that I cancelled two months ago. I want an immediate refund!",
        "description": "HIGH FINANCIAL RISK: Involves payment dispute and refund demands, requiring human specialist review."
    },
    {
        "id": "genius_bar",
        "title": "Genius Bar Broken Screen",
        "category": "repair_replacement_or_order",
        "expected_action": "escalate",
        "text": "Dropped my iPhone X on pavement and the OLED screen is completely cracked. How do I book a Genius Bar repair under AppleCare+?",
        "description": "HARDWARE REPAIR: Involves physical service appointments and warranty coverage determination."
    },
    {
        "id": "vague_complaint",
        "title": "Vague Frustrated Vent",
        "category": "other_or_unclear",
        "expected_action": "escalate",
        "text": "Apple is the worst company ever. Everything broke and nothing works anymore. Total disaster.",
        "description": "AMBIGUOUS / UNCLEAR: No specific diagnostic symptom; must safely escalate to human."
    }
]


class AgentRequestHandler(SimpleHTTPRequestHandler):
    """Handles API requests and serves static dashboard files."""

    agent_service: SupportAgentService = None
    static_dir: Path = REPO_ROOT / "static"
    golden_dir: Path = REPO_ROOT / "data" / "golden"

    def do_GET(self):
        url_path = self.path.split("?")[0]

        if url_path == "/" or url_path == "/index.html":
            self.serve_file(self.static_dir / "index.html", "text/html")
        elif url_path.startswith("/static/"):
            rel_path = url_path[len("/static/"):]
            file_path = self.static_dir / rel_path
            self.serve_file(file_path)
        elif url_path == "/api/samples":
            self.send_json_response(SAMPLE_INQUIRIES)
        elif url_path == "/api/golden/status":
            self.handle_golden_status()
        elif url_path == "/api/golden/rows":
            self.handle_golden_rows()
        elif url_path == "/api/stats":
            self.handle_stats()
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        url_path = self.path.split("?")[0]

        if url_path == "/api/process":
            self.handle_process()
        elif url_path == "/api/golden/adjudicate":
            self.handle_golden_adjudicate()
        else:
            self.send_error(404, "Endpoint not found")

    def serve_file(self, file_path: Path, content_type: str = None):
        """Safely serves a local static file."""
        try:
            if not file_path.resolve().is_relative_to(self.static_dir.resolve()):
                self.send_error(403, "Access Denied")
                return
        except (ValueError, AttributeError):
            pass

        if not file_path.exists() or file_path.is_dir():
            self.send_error(404, "File Not Found")
            return

        if not content_type:
            content_type, _ = mimetypes.guess_type(str(file_path))
            content_type = content_type or "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {str(e)}")

    def send_json_response(self, data: Any, status_code: int = 200):
        """Sends a UTF-8 encoded JSON response."""
        try:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            self.send_error(500, f"JSON Serialization Error: {str(e)}")

    def handle_process(self):
        """Processes an incoming customer inquiry through the full pipeline."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            req_data = json.loads(body) if body else {}

            text = req_data.get("text", "").strip()
            top_k = int(req_data.get("top_k", 3))
            conf_thresh = float(req_data.get("confidence_threshold", 0.80))
            sim_thresh = float(req_data.get("similarity_threshold", 0.50))

            if not text:
                self.send_json_response({"error": "Query text cannot be empty."}, status_code=400)
                return

            result = self.agent_service.process_inquiry(
                text=text,
                top_k=top_k,
                confidence_threshold=conf_thresh,
                similarity_threshold=sim_thresh,
            )
            self.send_json_response(result)

        except Exception as e:
            self.send_json_response({"error": f"Inference failed: {str(e)}"}, status_code=500)

    def handle_stats(self):
        """Returns overview model and training stats."""
        stats = {
            "model_name": "TF-IDF + Logistic Regression",
            "taxonomy_intents": self.agent_service.intent_classes,
            "indexed_training_dialogues": self.agent_service.retrieval_total_inquiries,
            "default_thresholds": {
                "confidence_threshold": 0.80,
                "similarity_threshold": 0.50,
                "adaptation_threshold": 0.50,
            }
        }
        self.send_json_response(stats)

    def handle_golden_status(self):
        """Returns the current state of Phase 10 human adjudication."""
        sheet_path = self.golden_dir / "adjudication_sheet.csv"
        manifest_path = self.golden_dir / "golden_labels_freeze_manifest.json"

        if not sheet_path.exists():
            self.send_json_response({"error": "Adjudication sheet not found"}, status_code=404)
            return

        df = pd.read_csv(sheet_path)
        total = len(df)
        done_count = int((df["final_status"] == "DONE").sum()) if "final_status" in df.columns else 0
        pending_count = total - done_count
        is_frozen = manifest_path.exists()

        self.send_json_response({
            "total_rows": total,
            "done_count": done_count,
            "pending_count": pending_count,
            "completion_percentage": round((done_count / total) * 100, 1) if total > 0 else 0,
            "is_frozen": is_frozen,
            "freeze_manifest_file": str(manifest_path.name) if is_frozen else None,
        })

    def handle_golden_rows(self):
        """Returns rows from the adjudication sheet for human review."""
        sheet_path = self.golden_dir / "adjudication_sheet.csv"
        if not sheet_path.exists():
            self.send_json_response({"error": "Adjudication sheet not found"}, status_code=404)
            return

        df = pd.read_csv(sheet_path).fillna("")
        records = df.head(50).to_dict(orient="records")
        self.send_json_response({"rows": records, "total_in_file": len(df)})

    def handle_golden_adjudicate(self):
        """Updates an adjudication row."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            req = json.loads(body)

            golden_id = req.get("golden_id")
            final_intent = req.get("final_intent")
            final_action = req.get("final_action")
            final_reason = req.get("final_escalation_reason", "")

            sheet_path = self.golden_dir / "adjudication_sheet.csv"
            df = pd.read_csv(sheet_path)

            mask = df["golden_id"] == golden_id
            if not mask.any():
                self.send_json_response({"error": f"ID {golden_id} not found"}, status_code=404)
                return

            idx = df[mask].index[0]
            df.at[idx, "final_intent"] = final_intent
            df.at[idx, "final_action"] = final_action
            df.at[idx, "final_escalation_reason"] = final_reason
            df.at[idx, "final_status"] = "DONE"
            df.to_csv(sheet_path, index=False)

            self.send_json_response({"success": True, "golden_id": golden_id})
        except Exception as e:
            self.send_json_response({"error": str(e)}, status_code=500)


def main():
    parser = argparse.ArgumentParser(description="Run Support Agent Web Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    print("=" * 65)
    print("CUSTOMER SUPPORT AGENT WEB APPLICATION")
    print("=" * 65)

    # Initialize agent service once
    agent_service = SupportAgentService()
    AgentRequestHandler.agent_service = agent_service

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, AgentRequestHandler)

    url = f"http://{args.host}:{args.port}"
    print(f"\n[SERVER] Server running successfully at: {url}")
    print("Press Ctrl+C to stop the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
        print("Server shutdown cleanly.")


if __name__ == "__main__":
    main()
