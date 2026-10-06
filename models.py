"""Validated game state and the application's interpreted decision."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

ACTIONS = ("attack", "flee", "negotiate", "call_guards")
PERSONALITIES = {
    "aggressive": "Willing to attack and accept combat risk, but still values survival.",
    "coward": "Avoids confrontation and prefers escaping when threatened.",
    "diplomatic": "Prefers peaceful negotiation whenever reasonably possible.",
    "neutral": "Balances personal risk, available support and player behavior.",
}


class ScenarioError(ValueError):
    """An unreadable or invalid scenario."""


@dataclass(frozen=True)
class Scenario:
    name: str
    npc_health: int
    player_health: int
    player_weapon_drawn: bool
    player_reputation: int
    guards_nearby: bool
    distance_to_player: float
    npc_personality: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ScenarioError("name must be a nonempty string")
        for field, lower, upper in (("npc_health", 0, 100), ("player_health", 0, 100),
                                    ("player_reputation", -100, 100)):
            value = getattr(self, field)
            if type(value) is not int or not lower <= value <= upper:
                raise ScenarioError(f"{field} must be an integer from {lower} to {upper}")
        for field in ("player_weapon_drawn", "guards_nearby"):
            if type(getattr(self, field)) is not bool:
                raise ScenarioError(f"{field} must be a JSON boolean")
        distance = self.distance_to_player
        try:
            valid_distance = (type(distance) in (int, float)
                              and math.isfinite(distance) and distance >= 0)
        except OverflowError:
            valid_distance = False
        if not valid_distance:
            raise ScenarioError("distance_to_player must be finite and >= 0 (meters)")
        if not isinstance(self.npc_personality, str) or self.npc_personality not in PERSONALITIES:
            raise ScenarioError(f"npc_personality must be one of {', '.join(PERSONALITIES)}")

    def state(self) -> dict[str, Any]:
        return {**asdict(self), "personality_description": PERSONALITIES[self.npc_personality],
                "world_context": "A game encounter. Distance is in meters; health is 0-100; "
                "reputation is -100 (hostile) to 100 (trusted). Guards can assist only if nearby."}


def load_scenario(path: str | Path) -> Scenario:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ScenarioError("scenario must be a JSON object")
        return Scenario(**data)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise ScenarioError(f"Cannot load {path}: {exc}. Check the scenario JSON fields.") from exc


@dataclass(frozen=True)
class Decision:
    action: str
    danger: float
    model: str
    action_probabilities: dict[str, float] | None = None
    action_confidence: float | None = None
    danger_confidence: float | None = None

    @property
    def panic_mode(self) -> bool:
        return self.danger >= 0.9
