"""
tests/test_log_parser.py — Tests for pipeline/log_parser.py and pipeline/schema.py

Coverage:
  - RawLogLoader: file loading, JSON error handling, directory iteration
  - LogValidator: top-level keys, event field presence, value constraints,
                  enum values, player index bounds, game consistency rules
  - ParsedGame: typed accessors, iter_action_contexts
  - load_logs: directory loading, strict mode, invalid log filtering
  - Round-trip: simulator-generated logs all pass validation
"""

import json
import os
import sys
import copy
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.log_parser import (
    LogValidator,
    RawLogLoader,
    ParsedGame,
    ParsedEvent,
    load_logs,
    validate_log,
)
from pipeline.schema import VALID_CARDS, VALID_ACTION_TYPES

# ---------------------------------------------------------------------------
# Helpers — minimal valid log builder
# ---------------------------------------------------------------------------

def _player_snapshot(idx: int, coins: int = 2, influence: int = 2) -> dict:
    return {
        "name": f"P{idx}",
        "idx": idx,
        "coins": coins,
        "influence_count": influence,
        "revealed_cards": [],
        "is_alive": influence > 0,
    }


def _make_valid_log(n_players: int = 3) -> dict:
    """Construct the smallest structurally-valid log dict."""
    names = [f"P{i}" for i in range(n_players)]
    snap = [_player_snapshot(i) for i in range(n_players)]
    return {
        "game_id": 1234567890,
        "player_names": names,
        "n_players": n_players,
        "events": [
            {
                "event_type": "game_start",
                "turn_number": 0,
                "player_names": names,
                "n_players": n_players,
                "state": snap,
            },
            {
                "event_type": "action",
                "turn_number": 1,
                "actor_idx": 0,
                "actor_name": "P0",
                "action_type": "Income",
                "target_idx": None,
                "target_name": None,
                "claimed_card": None,
                "state": snap,
            },
            {
                "event_type": "action_resolved",
                "turn_number": 1,
                "action_type": "Income",
                "actor_idx": 0,
                "actor_name": "P0",
                "state": snap,
            },
            {
                "event_type": "game_end",
                "turn_number": 5,
                "winner_idx": 0,
                "winner_name": "P0",
                "total_turns": 5,
            },
        ],
    }


# ===========================================================================
# Schema / validator tests
# ===========================================================================

class TestTopLevelValidation:

    def test_valid_log_passes(self):
        result = validate_log(_make_valid_log())
        assert result.is_valid, result.summary()

    def test_missing_game_id(self):
        log = _make_valid_log()
        del log["game_id"]
        result = validate_log(log)
        assert not result.is_valid
        assert any("game_id" in e.message for e in result.errors)

    def test_missing_player_names(self):
        log = _make_valid_log()
        del log["player_names"]
        result = validate_log(log)
        assert not result.is_valid

    def test_missing_n_players(self):
        log = _make_valid_log()
        del log["n_players"]
        result = validate_log(log)
        assert not result.is_valid

    def test_missing_events(self):
        log = _make_valid_log()
        del log["events"]
        result = validate_log(log)
        assert not result.is_valid

    def test_empty_events_list(self):
        log = _make_valid_log()
        log["events"] = []
        result = validate_log(log)
        assert not result.is_valid

    def test_n_players_mismatch(self):
        log = _make_valid_log(n_players=3)
        log["n_players"] = 4   # says 4 but player_names has 3
        result = validate_log(log)
        assert not result.is_valid

    @pytest.mark.parametrize("count", [1, 7, 10])
    def test_invalid_player_count(self, count):
        log = _make_valid_log()
        log["player_names"] = [f"P{i}" for i in range(count)]
        log["n_players"] = count
        result = validate_log(log)
        assert not result.is_valid

    @pytest.mark.parametrize("count", [2, 3, 4, 5, 6])
    def test_valid_player_counts(self, count):
        result = validate_log(_make_valid_log(n_players=count))
        assert result.is_valid, result.summary()

    def test_non_dict_input(self):
        result = validate_log([1, 2, 3])
        assert not result.is_valid


class TestEventValidation:

    def test_unknown_event_type(self):
        log = _make_valid_log()
        log["events"][1]["event_type"] = "super_coup"
        result = validate_log(log)
        assert not result.is_valid

    def test_missing_turn_number(self):
        log = _make_valid_log()
        del log["events"][1]["turn_number"]
        result = validate_log(log)
        assert not result.is_valid

    def test_negative_turn_number(self):
        log = _make_valid_log()
        log["events"][1]["turn_number"] = -1
        result = validate_log(log)
        assert not result.is_valid

    def test_action_missing_actor_idx(self):
        log = _make_valid_log()
        del log["events"][1]["actor_idx"]
        result = validate_log(log)
        assert not result.is_valid

    def test_actor_idx_out_of_range(self):
        log = _make_valid_log(n_players=3)
        log["events"][1]["actor_idx"] = 5  # only 3 players
        result = validate_log(log)
        assert not result.is_valid

    def test_invalid_action_type(self):
        log = _make_valid_log()
        log["events"][1]["action_type"] = "SuperCoup"
        result = validate_log(log)
        assert not result.is_valid

    @pytest.mark.parametrize("action_type", list(VALID_ACTION_TYPES))
    def test_all_valid_action_types_pass(self, action_type):
        log = _make_valid_log()
        log["events"][1]["action_type"] = action_type
        result = validate_log(log)
        assert result.is_valid, result.summary()

    def test_invalid_claimed_card(self):
        log = _make_valid_log()
        log["events"][1]["claimed_card"] = "Wizard"
        result = validate_log(log)
        assert not result.is_valid

    @pytest.mark.parametrize("card", list(VALID_CARDS))
    def test_all_valid_cards_pass(self, card):
        log = _make_valid_log()
        log["events"][1]["claimed_card"] = card
        result = validate_log(log)
        assert result.is_valid, result.summary()

    def test_influence_remaining_out_of_range(self):
        log = _make_valid_log()
        log["events"].insert(2, {
            "event_type": "influence_loss",
            "turn_number": 1,
            "player_idx": 1,
            "player_name": "P1",
            "card_lost": "Duke",
            "influence_remaining": 5,  # invalid
        })
        result = validate_log(log)
        assert not result.is_valid

    def test_influence_remaining_valid(self):
        log = _make_valid_log()
        for ir in [0, 1]:
            log["events"].insert(2, {
                "event_type": "influence_loss",
                "turn_number": 1,
                "player_idx": 1,
                "player_name": "P1",
                "card_lost": "Duke",
                "influence_remaining": ir,
            })
            result = validate_log(log)
            assert result.is_valid, result.summary()
            log["events"].pop(2)  # clean up for next iteration


class TestStateSnapshotValidation:

    def test_wrong_player_count_in_snapshot(self):
        log = _make_valid_log(n_players=3)
        # Add an extra player to the snapshot
        log["events"][0]["state"].append(_player_snapshot(99))
        result = validate_log(log)
        assert not result.is_valid

    def test_negative_coins_in_snapshot(self):
        log = _make_valid_log()
        log["events"][0]["state"][0]["coins"] = -1
        result = validate_log(log)
        assert not result.is_valid

    def test_invalid_revealed_card_in_snapshot(self):
        log = _make_valid_log()
        log["events"][0]["state"][0]["revealed_cards"] = ["Wizard"]
        result = validate_log(log)
        assert not result.is_valid

    def test_missing_snapshot_field(self):
        log = _make_valid_log()
        del log["events"][0]["state"][0]["coins"]
        result = validate_log(log)
        assert not result.is_valid

    def test_high_coins_produces_warning(self):
        log = _make_valid_log()
        log["events"][0]["state"][0]["coins"] = 13
        result = validate_log(log)
        # Should still be valid but with a warning
        assert result.is_valid
        assert len(result.warnings) > 0


class TestGameConsistency:

    def test_first_event_must_be_game_start(self):
        log = _make_valid_log()
        log["events"][0]["event_type"] = "action"
        # patch to satisfy action schema
        log["events"][0].update({
            "actor_idx": 0, "actor_name": "P0",
            "action_type": "Income", "target_idx": None,
            "target_name": None, "claimed_card": None,
        })
        result = validate_log(log)
        assert not result.is_valid

    def test_last_event_must_be_game_end(self):
        log = _make_valid_log()
        log["events"][-1]["event_type"] = "action_resolved"
        result = validate_log(log)
        assert not result.is_valid

    def test_no_action_events_is_invalid(self):
        log = _make_valid_log()
        log["events"] = [
            e for e in log["events"]
            if e["event_type"] not in ("action", "action_resolved")
        ]
        result = validate_log(log)
        assert not result.is_valid

    def test_missing_winner_produces_warning(self):
        log = _make_valid_log()
        log["events"][-1]["winner_name"] = None
        log["events"][-1]["winner_idx"] = None
        result = validate_log(log)
        # Valid but with a warning (turn-limit games are legal)
        assert result.is_valid
        assert any("winner" in w.message.lower() for w in result.warnings)


# ===========================================================================
# RawLogLoader tests
# ===========================================================================

class TestRawLogLoader:

    def test_load_valid_file(self, tmp_path):
        log = _make_valid_log()
        path = tmp_path / "game.json"
        path.write_text(json.dumps(log))
        result, err = RawLogLoader.from_file(str(path))
        assert err is None
        assert result["game_id"] == log["game_id"]

    def test_load_malformed_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        result, err = RawLogLoader.from_file(str(path))
        assert result is None
        assert err is not None
        assert "JSON" in err

    def test_load_missing_file(self):
        result, err = RawLogLoader.from_file("/nonexistent/path/game.json")
        assert result is None
        assert err is not None

    def test_load_directory(self, tmp_path):
        for i in range(5):
            log = _make_valid_log()
            log["game_id"] = i
            (tmp_path / f"game_{i:04d}.json").write_text(json.dumps(log))
        # Add a non-JSON file — should be skipped
        (tmp_path / "notes.txt").write_text("ignore me")

        results = list(RawLogLoader.from_directory(str(tmp_path)))
        assert len(results) == 5
        for path, raw, err in results:
            assert err is None
            assert raw is not None

    def test_load_directory_not_found(self):
        with pytest.raises(FileNotFoundError):
            list(RawLogLoader.from_directory("/nonexistent/dir"))

    def test_load_directory_includes_malformed(self, tmp_path):
        (tmp_path / "good.json").write_text(json.dumps(_make_valid_log()))
        (tmp_path / "bad.json").write_text("{broken")
        results = list(RawLogLoader.from_directory(str(tmp_path)))
        assert len(results) == 2
        errors = [err for _, _, err in results if err is not None]
        assert len(errors) == 1


# ===========================================================================
# ParsedGame tests
# ===========================================================================

class TestParsedGame:

    def _make_parsed(self) -> ParsedGame:
        log = _make_valid_log(n_players=3)
        events = [ParsedEvent.from_dict(e) for e in log["events"]]
        return ParsedGame(
            game_id=log["game_id"],
            player_names=log["player_names"],
            n_players=log["n_players"],
            events=events,
        )

    def test_events_of_type(self):
        game = self._make_parsed()
        actions = game.events_of_type("action")
        assert len(actions) == 1
        assert all(e.type == "action" for e in actions)

    def test_action_events_property(self):
        game = self._make_parsed()
        assert len(game.action_events) == 1

    def test_game_end_event(self):
        game = self._make_parsed()
        end = game.game_end_event
        assert end is not None
        assert end.type == "game_end"

    def test_winner_name(self):
        game = self._make_parsed()
        assert game.winner_name == "P0"

    def test_total_turns(self):
        game = self._make_parsed()
        assert game.total_turns == 5

    def test_len(self):
        game = self._make_parsed()
        assert len(game) == 4  # start + action + resolved + end

    def test_iteration(self):
        game = self._make_parsed()
        types = [e.type for e in game]
        assert types[0] == "game_start"
        assert types[-1] == "game_end"

    def test_iter_action_contexts(self):
        game = self._make_parsed()
        contexts = list(game.iter_action_contexts())
        assert len(contexts) == 1
        action_evt, reactions = contexts[0]
        assert action_evt.type == "action"
        assert any(e.type == "action_resolved" for e in reactions)

    def test_parsed_event_fields(self):
        log = _make_valid_log()
        event = ParsedEvent.from_dict(log["events"][1])
        assert event.type == "action"
        assert event.actor_idx == 0
        assert event.action_type == "Income"
        assert event.turn == 1


# ===========================================================================
# load_logs integration tests
# ===========================================================================

class TestLoadLogs:

    def test_load_from_directory(self, tmp_path):
        for i in range(10):
            log = _make_valid_log(n_players=3)
            log["game_id"] = i
            (tmp_path / f"game_{i:04d}.json").write_text(json.dumps(log))
        games = load_logs(str(tmp_path))
        assert len(games) == 10
        assert all(isinstance(g, ParsedGame) for g in games)

    def test_load_single_file(self, tmp_path):
        log = _make_valid_log()
        path = tmp_path / "game.json"
        path.write_text(json.dumps(log))
        games = load_logs(str(path))
        assert len(games) == 1

    def test_invalid_logs_skipped_by_default(self, tmp_path):
        good = _make_valid_log()
        bad = {"broken": True}   # missing required keys
        (tmp_path / "good.json").write_text(json.dumps(good))
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        games = load_logs(str(tmp_path))
        assert len(games) == 1

    def test_strict_mode_raises_on_invalid(self, tmp_path):
        bad = {"broken": True}
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_logs(str(tmp_path), strict=True)

    def test_load_logs_source_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_logs("/nonexistent/path")

    def test_parsed_games_have_source_path(self, tmp_path):
        log = _make_valid_log()
        path = tmp_path / "game.json"
        path.write_text(json.dumps(log))
        games = load_logs(str(path))
        assert games[0].source_path == str(path)

    def test_malformed_json_skipped(self, tmp_path):
        (tmp_path / "good.json").write_text(json.dumps(_make_valid_log()))
        (tmp_path / "bad.json").write_text("{not json")
        games = load_logs(str(tmp_path))
        assert len(games) == 1


# ===========================================================================
# Round-trip: simulator-generated logs all pass validation
# ===========================================================================

class TestRoundTrip:
    """
    Load all logs generated by the simulator and assert they all pass.
    Requires data/raw_logs/ to exist (generated by running simulator.py).
    """

    RAW_LOGS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "data", "raw_logs"
    )

    def test_simulator_logs_exist(self):
        if not os.path.isdir(self.RAW_LOGS_DIR):
            pytest.skip("data/raw_logs/ not found — run simulator.py first")
        files = [f for f in os.listdir(self.RAW_LOGS_DIR) if f.endswith(".json")]
        assert len(files) > 0, "No .json files in data/raw_logs/"

    def test_all_simulator_logs_are_valid(self):
        if not os.path.isdir(self.RAW_LOGS_DIR):
            pytest.skip("data/raw_logs/ not found — run simulator.py first")

        validator = LogValidator()
        files = sorted(
            f for f in os.listdir(self.RAW_LOGS_DIR) if f.endswith(".json")
        )
        failures = []
        for fname in files:
            path = os.path.join(self.RAW_LOGS_DIR, fname)
            raw, err = RawLogLoader.from_file(path)
            if err:
                failures.append(f"{fname}: load error — {err}")
                continue
            result = validator.validate(raw)
            if not result.is_valid:
                failures.append(f"{fname}: {result.summary()}")

        assert failures == [], (
            f"{len(failures)}/{len(files)} logs failed validation:\n"
            + "\n".join(failures[:10])
        )

    def test_load_logs_returns_all_simulator_logs(self):
        if not os.path.isdir(self.RAW_LOGS_DIR):
            pytest.skip("data/raw_logs/ not found — run simulator.py first")

        n_files = len([
            f for f in os.listdir(self.RAW_LOGS_DIR) if f.endswith(".json")
        ])
        games = load_logs(self.RAW_LOGS_DIR, strict=True)
        assert len(games) == n_files

    def test_parsed_games_have_correct_structure(self):
        if not os.path.isdir(self.RAW_LOGS_DIR):
            pytest.skip("data/raw_logs/ not found — run simulator.py first")

        games = load_logs(self.RAW_LOGS_DIR)
        for game in games:
            assert game.n_players in range(2, 7)
            assert len(game.player_names) == game.n_players
            assert len(game.events) > 0
            assert game.game_end_event is not None
            assert game.total_turns is not None and game.total_turns > 0
            assert len(game.action_events) > 0
