"""Capture real local-jev HTTP responses; requires a running, warmed-up server.

This is an integration check, not a model accuracy test. It never supplies
expected actions or substitutes a decision when the server fails.
"""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_client import JevClient, build_request, parse_response  # noqa: E402
from models import load_scenario  # noqa: E402

MODEL = "llm-qwen3.5-4b"


def main() -> int:
    destination = ROOT / "results" / "live"
    destination.mkdir(parents=True, exist_ok=True)
    client = JevClient(model=MODEL, timeout=180)
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "base_url": client.base_url, "requested_model": MODEL}
    for route in ("healthz", "v1/models"):
        response = requests.get(f"{client.base_url}/{route}", timeout=10)
        response.raise_for_status()
        report[route] = response.json()
        print(f"GET /{route}: {response.status_code}", flush=True)
    if report["healthz"].get("default_model") != MODEL:
        raise RuntimeError(f"Expected server default {MODEL}; inspect /healthz")

    def ask(scenario):
        payload = build_request(scenario, MODEL)
        start = time.perf_counter()
        response = requests.post(f"{client.base_url}/v1/systemone", json=payload, timeout=180)
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        body = response.json()
        decision = parse_response(body)
        if decision.model != MODEL:
            raise RuntimeError(f"Unexpected answering model: {decision.model}")
        print(f"{scenario.name}: {decision.action}, danger={decision.danger:.4f}, {elapsed:.3f}s", flush=True)
        return {"request": payload, "response": body, "http_status": response.status_code,
                "request_id": response.headers.get("x-typesafe-request-id"), "elapsed_seconds": elapsed}

    report["direct_choice_score"] = ask(load_scenario(ROOT / "scenarios/wounded_npc.json"))
    (destination / "direct-choice-score.json").write_text(
        json.dumps(report["direct_choice_score"], indent=2), encoding="utf-8")
    report["scenarios"] = {}
    for path in sorted((ROOT / "scenarios").glob("*.json")):
        report["scenarios"][path.name] = ask(load_scenario(path))
        (destination / f"{path.stem}.json").write_text(
            json.dumps(report["scenarios"][path.name], indent=2), encoding="utf-8")
    # Remove differing display names as a possible confound in the personality experiment.
    report["controlled_personalities"] = {
        personality: ask(replace(load_scenario(ROOT / f"scenarios/{personality}_npc.json"),
                                 name="Personality comparison"))
        for personality in ("aggressive", "coward", "diplomatic")}
    (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved real requests and responses in {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
