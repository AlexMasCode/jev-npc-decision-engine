"""Run one scenario or the bundled personality experiment."""
import argparse
from pathlib import Path
import sys

from jev_client import JevClient, JevError
from models import Decision, Scenario, ScenarioError, load_scenario

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"


def format_result(scenario: Scenario, decision: Decision) -> str:
    lines = [f"Scenario: {scenario.name}", f"NPC personality: {scenario.npc_personality}",
             f"NPC health: {scenario.npc_health}", f"Player health: {scenario.player_health}",
             f"Player reputation: {scenario.player_reputation}",
             f"Player weapon drawn: {'yes' if scenario.player_weapon_drawn else 'no'}",
             f"Guards nearby: {'yes' if scenario.guards_nearby else 'no'}",
             f"Distance: {scenario.distance_to_player:g} meters", "", "Jev decision", "------------",
             f"Model: {decision.model}", f"Action: {decision.action.upper()}",
             f"Danger score: {decision.danger:.4f}"]
    if decision.action_confidence is not None:
        lines.append(f"Choice confidence: {decision.action_confidence:.4f}")
    if decision.danger_confidence is not None:
        lines.append(f"Score confidence: {decision.danger_confidence:.4f}")
    if decision.action_probabilities is not None:
        lines.extend(["", "Action probabilities:"])
        lines.extend(f"{action:12} {probability:.4f}" for action, probability in decision.action_probabilities.items())
    lines.extend(["", "Application state:", "PANIC MODE" if decision.panic_mode else "NORMAL MODE"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NPC Decision Engine powered by local-jev")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("scenario", nargs="?", type=Path, help="path to a scenario JSON file")
    source.add_argument("--all", action="store_true", help="run all bundled scenarios")
    parser.add_argument("--timeout", type=float, default=120, help="HTTP timeout in seconds (default: 120)")
    args = parser.parse_args(argv)
    paths = sorted(SCENARIOS_DIR.glob("*.json")) if args.all else [args.scenario]
    try:
        if not paths:
            raise ScenarioError(f"No scenarios found in {SCENARIOS_DIR}")
        # Validate the whole batch before making any model calls.
        scenarios = [load_scenario(path) for path in paths]
        client = JevClient(timeout=args.timeout)
        print("NPC Decision Engine\n===================\n", flush=True)
        for scenario in scenarios:
            print(format_result(scenario, client.decide(scenario)), end="\n\n", flush=True)
    except (ScenarioError, JevError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
