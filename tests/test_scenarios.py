"""Deterministic checks only. HTTP fixtures do not validate model quality."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from jev_client import DANGER_LEVELS, JevClient, JevError, build_request, parse_response
from main import SCENARIOS_DIR, format_result, main
from models import ACTIONS, Decision, Scenario, ScenarioError, load_scenario


def response_fixture() -> dict:
    """Synthetic response shaped like upstream primitives.py, not a real prediction."""
    return {"model": "test-fixture", "answers": {
        "action": {"type": "choice", "choice": "flee", "confidence": 0.7067,
                   "probabilities": {"attack": 0.08, "flee": 0.78, "negotiate": 0.12, "call_guards": 0.02}},
        "danger": {"type": "score", "score": 0.94, "confidence": 0.88,
                   "legend": {str(i): v for i, v in enumerate(DANGER_LEVELS)},
                   "probabilities": {"0": 0.06, "1": 0.94}}},
        "usage": {"input_tokens": 100, "output_tokens": 2}}


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario(SCENARIOS_DIR / "wounded_npc.json")

    def test_all_bundled_scenarios(self):
        paths = sorted(SCENARIOS_DIR.glob("*.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            self.assertIsInstance(load_scenario(path), Scenario)

    def test_personality_experiment_controls_world(self):
        states = [asdict(load_scenario(SCENARIOS_DIR / f"{p}_npc.json"))
                  for p in ("aggressive", "coward", "diplomatic")]
        for state in states:
            del state["name"], state["npc_personality"]
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[1], states[2])

    def test_invalid_values(self):
        cases = [("npc_health", -1), ("npc_health", 101), ("npc_health", True),
                 ("player_health", 1.5), ("player_reputation", -101),
                 ("player_reputation", 101), ("distance_to_player", -0.1),
                 ("distance_to_player", float("nan")), ("distance_to_player", float("inf")),
                 ("distance_to_player", 10 ** 1000),
                 ("distance_to_player", True), ("player_weapon_drawn", "true"),
                 ("guards_nearby", 1), ("npc_personality", "wizard"),
                 ("npc_personality", []), ("name", " ")]
        for field, value in cases:
            with self.subTest(field=field, value=value), self.assertRaises(ScenarioError):
                Scenario(**{**asdict(self.scenario), field: value})

    def test_boundaries(self):
        Scenario(**{**asdict(self.scenario), "npc_health": 0, "player_health": 100,
                    "player_reputation": -100, "distance_to_player": 0})

    def test_bad_files(self):
        with tempfile.TemporaryDirectory(dir=SCENARIOS_DIR.parent) as folder:
            path = Path(folder) / "bad.json"
            for content in ("{", "[]", "{}", json.dumps({**asdict(self.scenario), "typo": 2})):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ScenarioError):
                    load_scenario(path)
            with self.assertRaises(ScenarioError):
                load_scenario(Path(folder) / "missing.json")

    def test_request_contains_personality_and_both_primitives(self):
        payload = build_request(self.scenario)
        self.assertEqual(payload["state"]["npc_personality"], "coward")
        self.assertIn("personality_description", payload["state"])
        self.assertEqual(payload["questions"]["action"]["type"], "choice")
        self.assertEqual(set(payload["questions"]["action"]["criteria"]), set(ACTIONS))
        self.assertEqual(payload["questions"]["danger"]["criteria"], DANGER_LEVELS)


class ResponseTests(unittest.TestCase):
    def test_documented_shape(self):
        decision = parse_response(response_fixture())
        self.assertEqual(decision.action, "flee")
        self.assertEqual(decision.danger, 0.94)
        self.assertEqual(decision.action_probabilities["flee"], 0.78)
        self.assertTrue(decision.panic_mode)

    def test_all_supported_actions(self):
        for action in ACTIONS:
            data = response_fixture()
            data["answers"]["action"] = {"type": "choice", "choice": action}
            self.assertEqual(parse_response(data).action, action)

    def test_unsupported_action(self):
        data = response_fixture()
        data["answers"]["action"]["choice"] = "dance"
        with self.assertRaisesRegex(JevError, "Unsupported action"):
            parse_response(data)

    def test_absent_optional_values_not_fabricated(self):
        data = response_fixture()
        for answer in data["answers"].values():
            del answer["confidence"], answer["probabilities"]
        decision = parse_response(data)
        self.assertIsNone(decision.action_confidence)
        self.assertIsNone(decision.action_probabilities)

    def test_malformed_shapes(self):
        for data in (None, [], {}, {"model": "x", "answers": []},
                     {"model": "x", "answers": {"action": None, "danger": {}}}):
            with self.subTest(data=data), self.assertRaises(JevError):
                parse_response(data)

    def test_invalid_score(self):
        for value in (-0.1, 1.1, "0.9", True, float("nan"), float("inf"), 10 ** 1000):
            data = response_fixture()
            data["answers"]["danger"]["score"] = value
            with self.subTest(value=value), self.assertRaises(JevError):
                parse_response(data)

    def test_invalid_distributions_and_metadata(self):
        cases = [("action", "probabilities", {"flee": 1}),
                 ("action", "confidence", False), ("action", "type", "score"),
                 ("danger", "legend", {"0": "safe", "1": "moderate", "2": "dangerous"}),
                 ("danger", "score", 0.5),
                 ("action", "probabilities", dict.fromkeys(ACTIONS, 0.1)),
                 ("action", "probabilities", {"attack": 0.9, "flee": 0.1, "negotiate": 0, "call_guards": 0})]
        for answer, field, value in cases:
            data = deepcopy(response_fixture())
            data["answers"][answer][field] = value
            with self.subTest(field=field), self.assertRaises(JevError):
                parse_response(data)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario(SCENARIOS_DIR / "friendly_player.json")

    @patch("jev_client.requests.post")
    def test_endpoint_and_payload(self, post):
        post.return_value.json.return_value = response_fixture()
        JevClient("http://127.0.0.1:8765/", timeout=7).decide(self.scenario)
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:8765/v1/systemone")
        self.assertEqual(post.call_args.kwargs["timeout"], 7)
        self.assertEqual(post.call_args.kwargs["json"], build_request(self.scenario))

    def test_environment_overrides(self):
        with patch.dict("os.environ", {"JEV_BASE_URL": "http://127.0.0.1:9876", "JEV_MODEL": "nli-deberta-large"}):
            client = JevClient()
            self.assertEqual(client.base_url, "http://127.0.0.1:9876")
            self.assertEqual(client.model, "nli-deberta-large")

    def test_invalid_configuration(self):
        for url in ("", "ftp://localhost", "http://u:secret@localhost", "http://localhost?x=1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                JevClient(url)
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                JevClient(timeout=timeout)

    @patch("jev_client.requests.post")
    def test_transport_errors(self, post):
        for exception, message in ((requests.ConnectionError(), "Cannot connect"),
                                   (requests.Timeout(), "timed out"),
                                   (requests.RequestException(), "HTTP request failed")):
            post.side_effect = exception
            with self.subTest(message=message), self.assertRaisesRegex(JevError, message):
                JevClient().decide(self.scenario)

    @patch("jev_client.requests.post")
    def test_http_errors(self, post):
        for status in (401, 404, 422, 503, 500):
            post.return_value.status_code = status
            post.return_value.raise_for_status.side_effect = requests.HTTPError()
            with self.subTest(status=status), self.assertRaisesRegex(JevError, f"HTTP {status}"):
                JevClient().decide(self.scenario)

    @patch("jev_client.requests.post")
    def test_invalid_json_response(self, post):
        post.return_value.json.side_effect = ValueError("bad json")
        with self.assertRaisesRegex(JevError, "invalid JSON"):
            JevClient().decide(self.scenario)


class CliTests(unittest.TestCase):
    def test_panic_threshold(self):
        self.assertTrue(Decision("flee", 0.9, "fixture").panic_mode)
        self.assertFalse(Decision("flee", 0.8999, "fixture").panic_mode)

    def test_render_without_confidence(self):
        text = format_result(load_scenario(SCENARIOS_DIR / "wounded_npc.json"), Decision("flee", 0.9, "fixture"))
        self.assertIn("PANIC MODE", text)
        self.assertIn("Action: FLEE", text)
        self.assertNotIn("confidence", text)
        self.assertNotIn("probabilities", text)

    @patch("main.JevClient")
    def test_all_cli(self, client):
        client.return_value.decide.return_value = parse_response(response_fixture())
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--all"]), 0)
        self.assertEqual(client.return_value.decide.call_count, 6)
        self.assertEqual(output.getvalue().count("Scenario:"), 6)

    @patch("main.JevClient")
    def test_single_cli(self, client):
        client.return_value.decide.return_value = parse_response(response_fixture())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(SCENARIOS_DIR / "wounded_npc.json")]), 0)
        client.return_value.decide.assert_called_once()

    @patch("main.JevClient")
    def test_error_exit(self, client):
        client.return_value.decide.side_effect = JevError("Cannot connect")
        error = io.StringIO()
        with redirect_stderr(error), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--all"]), 1)
        self.assertIn("Cannot connect", error.getvalue())

    @patch("main.JevClient")
    def test_invalid_file_prevents_request(self, client):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["missing.json"]), 1)
        client.assert_not_called()

    def test_conflicting_arguments(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--all", "anything.json"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
