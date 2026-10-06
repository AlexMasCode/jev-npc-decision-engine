"""HTTP client for amithgc/local-jev's verified System One wire format."""
import math
import os
from typing import Any
from urllib.parse import urlsplit

import requests

from models import ACTIONS, Decision, Scenario

ACTION_CRITERIA = {
    "attack": "The NPC believes direct combat is the best action.",
    "flee": "The NPC believes escaping is safer than fighting.",
    "negotiate": "The NPC attempts to peacefully resolve the encounter.",
    "call_guards": "The NPC seeks assistance from nearby guards; requires guards nearby.",
}
# Score uses level indices, not an arbitrary numeric min/max. Two levels give 0-1 directly.
DANGER_LEVELS = ["Completely safe: no meaningful threat to the NPC's safety.",
                 "Extremely dangerous: imminent serious injury or death for the NPC."]


class JevError(RuntimeError):
    """A connection, API or response contract failure."""


def build_request(scenario: Scenario, model: str = "jev-latest") -> dict[str, Any]:
    return {
        "state": scenario.state(), "model": model,
        "questions": {
            "action": {"type": "choice", "instructions":
                       "Choose the best next action for this NPC in the current encounter. "
                       "Consider the NPC's personality and preferences, health, player behavior, "
                       "reputation, distance and available guards. Personality influences willingness "
                       "to fight or negotiate; it does not guarantee an action.",
                       "criteria": ACTION_CRITERIA},
            "danger": {"type": "score", "instructions":
                       "Estimate the current physical danger to the NPC before it acts. "
                       "Consider health, drawn weapon, hostile reputation, distance and guard support. "
                       "Distinguish willingness to accept risk from the danger itself.",
                       "criteria": DANGER_LEVELS},
        },
    }


def _unit_number(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise JevError(f"Unexpected Jev response: {field} must be a finite number in [0, 1].")
    return float(value)


def _probabilities(value: Any, keys: tuple[str, ...], field: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise JevError(f"Unexpected Jev response: {field} must contain {', '.join(keys)}.")
    parsed = {key: _unit_number(value[key], f"{field}.{key}") for key in keys}
    if not math.isclose(sum(parsed.values()), 1, abs_tol=0.001):
        raise JevError(f"Unexpected Jev response: {field} must sum to 1.")
    return parsed


def parse_response(data: Any) -> Decision:
    try:
        if not isinstance(data, dict):
            raise TypeError("response must be an object")
        model = data["model"]
        if not isinstance(model, str) or not model.strip():
            raise TypeError("model must be a nonempty string")
        action, danger = data["answers"]["action"], data["answers"]["danger"]
        if action["type"] != "choice" or danger["type"] != "score":
            raise ValueError("expected Choice action and Score danger")
        selected = action["choice"]
        if selected not in ACTIONS:
            raise JevError(f"Unsupported action returned by Jev: {selected!r}. Check server compatibility.")
        score = _unit_number(danger["score"], "danger.score")
        probabilities = (_probabilities(action["probabilities"], ACTIONS, "action.probabilities")
                         if "probabilities" in action else None)
        if probabilities and probabilities[selected] < max(probabilities.values()):
            raise ValueError("choice disagrees with action probabilities")
        expected_legend = {str(i): value for i, value in enumerate(DANGER_LEVELS)}
        if "legend" in danger and danger["legend"] != expected_legend:
            raise ValueError("danger legend differs from the requested two-level rubric")
        if "probabilities" in danger:
            levels = _probabilities(danger["probabilities"], ("0", "1"), "danger.probabilities")
            if not math.isclose(score, levels["1"], abs_tol=0.001):
                raise ValueError("score disagrees with danger probabilities")
        return Decision(selected, score, model, probabilities,
                        _unit_number(action["confidence"], "action.confidence") if "confidence" in action else None,
                        _unit_number(danger["confidence"], "danger.confidence") if "confidence" in danger else None)
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError(f"Unexpected Jev response: {exc}. Check the local-jev version and API.") from exc


class JevClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120,
                 model: str | None = None) -> None:
        self.base_url = (base_url if base_url is not None else
                         os.getenv("JEV_BASE_URL", "http://127.0.0.1:8765")).rstrip("/")
        parts = urlsplit(self.base_url)
        if (parts.scheme not in ("http", "https") or not parts.hostname or parts.username
                or parts.password or parts.query or parts.fragment):
            raise ValueError("JEV_BASE_URL must be an HTTP(S) server URL without credentials, query or fragment")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and > 0 seconds")
        self.timeout = timeout
        self.model = model if model is not None else os.getenv("JEV_MODEL", "jev-latest")
        if not self.model.strip():
            raise ValueError("JEV_MODEL must be nonempty")

    def decide(self, scenario: Scenario) -> Decision:
        try:
            response = requests.post(f"{self.base_url}/v1/systemone",
                                     json=build_request(scenario, self.model), timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise JevError(f"Jev request timed out after {self.timeout:g}s. Wait for model loading "
                           "to finish or increase --timeout.") from exc
        except requests.ConnectionError as exc:
            raise JevError(f"Cannot connect to local Jev at {self.base_url}. "
                           "Start local-jev serve first and check JEV_BASE_URL.") from exc
        except requests.HTTPError as exc:
            hints = {401: "Use the default local server without LOCAL_JEV_API_KEY.",
                     404: "Check JEV_MODEL and the server endpoint.",
                     422: "Check server compatibility with the documented request schema.",
                     503: "Inspect the server log for model download, memory or CUDA errors."}
            raise JevError(f"Jev API returned HTTP {response.status_code}. "
                           + hints.get(response.status_code, "Inspect the local-jev server log.")) from exc
        except requests.RequestException as exc:
            raise JevError("Jev HTTP request failed. Check the server URL and network configuration.") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise JevError("Jev returned invalid JSON. Check that the URL points to local-jev.") from exc
        return parse_response(data)
