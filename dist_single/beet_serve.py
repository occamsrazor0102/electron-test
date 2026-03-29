#!/usr/bin/env python3
"""BEET HTTP API server for Windows embeddable Python distribution.

Thin launcher that imports the monolith (llm_detector.py) and starts
the HTTP server used by the Electron desktop app.

Usage:
    python.exe beet_serve.py --port 8000 --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

# Import from the co-located monolith
import llm_detector as _det

_VERSION = _det.__version__

# ---------------------------------------------------------------------------
# Mapping helpers (from server.py)
# ---------------------------------------------------------------------------

_DET_P_LLM: dict[str, float] = {
    "GREEN": 0.08,
    "YELLOW": 0.30,
    "AMBER": 0.62,
    "RED": 0.87,
    "UNCERTAIN": 0.50,
    "MIXED": 0.55,
}

_INTERVAL_HALF = 0.12


def _pipeline_to_report(result: dict[str, Any], profile: str = "default") -> dict[str, Any]:
    det = result.get("determination", "UNCERTAIN")
    if det == "REVIEW":
        det = "UNCERTAIN"

    p_llm = _DET_P_LLM.get(det, 0.5)
    certainty = float(result.get("calibrated_confidence") or result.get("confidence") or 0.5)
    half = _INTERVAL_HALF * (1.0 - certainty * 0.5)

    top_signals: list[str] = []
    cd = result.get("channel_details") or {}
    channels: dict[str, Any] = cd.get("channels", {}) if isinstance(cd, dict) else {}
    for ch_name, ch_data in channels.items():
        if isinstance(ch_data, dict) and ch_data.get("severity", "GREEN") not in ("GREEN",):
            top_signals.append(ch_name)
    if not top_signals:
        signal_fields = (
            "preamble_severity",
            "self_similarity_determination",
            "continuation_determination",
        )
        signal_map = {
            "preamble_severity": "preamble",
            "self_similarity_determination": "nssi",
            "continuation_determination": "dna_gpt",
        }
        for field in signal_fields:
            val = result.get(field, "GREEN")
            if val and val not in ("GREEN", None):
                top_signals.append(signal_map.get(field, field))

    layer_results = []
    for ch_name, ch_data in (channels.items() if isinstance(channels, dict) else {}.items()):
        if isinstance(ch_data, dict):
            layer_results.append(
                {
                    "layer": ch_name,
                    "score": float(ch_data.get("score", 0.0)),
                    "contribution": float(ch_data.get("score", 0.0)) * 0.25,
                    "detail": ch_data.get("explanation", ""),
                }
            )

    fusion_result = {
        "determination": det,
        "p_llm": p_llm,
        "confidence_low": max(0.0, p_llm - half),
        "confidence_high": min(1.0, p_llm + half),
        "layer_results": layer_results,
        "top_signals": top_signals[:5],
        "reasoning": result.get("reason", ""),
    }

    return {
        "text_id": result.get("task_id", ""),
        "text_excerpt": "",
        "word_count": int(result.get("word_count", 0)),
        "determination": det,
        "p_llm": p_llm,
        "confidence_low": max(0.0, p_llm - half),
        "confidence_high": min(1.0, p_llm + half),
        "top_signals": top_signals[:3],
        "fusion_result": fusion_result,
        "router_decision": {
            "profile": profile,
            "layers_used": list(channels.keys()) if isinstance(channels, dict) else [],
            "stop_reason": (
                cd.get("triggering_rule", "completed") if isinstance(cd, dict) else "completed"
            ),
        },
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "profile": profile,
        "occupation": result.get("occupation", ""),
    }


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class BeetHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/health":
            self._handle_health()
        elif path == "/config":
            qs = parse_qs(urlparse(self.path).query)
            profile = qs.get("profile", ["default"])[0]
            self._handle_config(profile)
        else:
            self._send_json({"detail": "Not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/analyze":
            self._handle_analyze()
        elif path == "/batch":
            self._handle_batch()
        else:
            self._send_json({"detail": "Not found"}, 404)

    def _handle_health(self) -> None:
        detectors: list[str] = ["preamble", "fingerprint_vocab", "prompt_structure", "nssi"]
        if getattr(_det, "HAS_SEMANTIC", False):
            detectors += ["surprisal_dynamics", "contrastive_lm"]
        if getattr(_det, "HAS_PERPLEXITY", False):
            detectors += ["perturbation", "dna_gpt"]
        self._send_json(
            {
                "status": "ok",
                "version": _VERSION,
                "availableDetectors": detectors,
            }
        )

    def _handle_analyze(self) -> None:
        try:
            payload = self._read_json()
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"detail": f"Invalid JSON: {exc}"}, 400)
            return

        text = payload.get("text", "")
        if not text or not text.strip():
            self._send_json({"detail": "Field 'text' is required and must not be empty."}, 422)
            return

        occupation = payload.get("occupation", "")
        profile = payload.get("profile", "default")

        try:
            result = _det.analyze_prompt(
                text,
                task_id=payload.get("text_id", ""),
                occupation=occupation,
            )
        except Exception as exc:
            self._send_json({"detail": f"Analysis failed: {exc}"}, 500)
            return

        self._send_json(_pipeline_to_report(result, profile))

    def _handle_batch(self) -> None:
        try:
            payload = self._read_json()
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"detail": f"Invalid JSON: {exc}"}, 400)
            return

        submissions = payload.get("submissions", [])
        if not isinstance(submissions, list):
            self._send_json({"detail": "'submissions' must be a list."}, 422)
            return

        reports = []
        for sub in submissions:
            text = (sub.get("text") or "") if isinstance(sub, dict) else ""
            occupation = (sub.get("occupation") or "") if isinstance(sub, dict) else ""
            profile = (sub.get("profile") or "default") if isinstance(sub, dict) else "default"
            if not text.strip():
                continue
            try:
                result = _det.analyze_prompt(text, task_id=sub.get("id", ""), occupation=occupation)
                reports.append(_pipeline_to_report(result, profile))
            except Exception:
                pass

        self._send_json(reports)

    def _handle_config(self, profile: str) -> None:
        profiles = {
            "screening": {
                "label": "Quick Scan",
                "description": "Fast, basic check",
                "run_l3": False,
            },
            "default": {
                "label": "Standard",
                "description": "Recommended for most text",
                "run_l3": False,
            },
            "full": {
                "label": "Deep Analysis",
                "description": "Thorough, takes longer",
                "run_l3": True,
            },
        }
        self._send_json({"profile": profile, "settings": profiles.get(profile, profiles["default"])})


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), BeetHandler)
    print(f"BEET API server listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("BEET API server stopped.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="beet-serve", description="BEET HTTP API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
