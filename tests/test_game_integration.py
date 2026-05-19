"""
tests/test_game_integration.py

End-to-end integration tests:
  - Games always terminate with a winner
  - Game log contains expected event types
  - Coin and influence invariants hold throughout
  - Simulator generates the right number of logs
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coup.game import Game
from coup.agents.random_agent import RandomAgent, HonestAgent
from coup.cards import Card
from coup.logger import EventType


# =========================================================================
# Helpers
# =========================================================================

def _make_game(n_players=3, seed=0, verbose=False, **agent_kwargs):
    agents = [RandomAgent(**agent_kwargs) for _ in range(n_players)]
    return Game(agents=agents, seed=seed, verbose=verbose)


def _run(n_players=3, seed=0, **agent_kwargs):
    game = _make_game(n_players=n_players, seed=seed, **agent_kwargs)
    return game.play_game(), game.state


# =========================================================================
# Game termination
# =========================================================================

class TestGameTermination:
    @pytest.mark.parametrize("n_players,seed", [
        (2, 0), (2, 1), (3, 0), (3, 5), (4, 42), (5, 99), (6, 7)
    ])
    def test_game_ends_with_a_winner(self, n_players, seed):
        log, state = _run(n_players=n_players, seed=seed)
        events = log["events"]
        end_events = [e for e in events if e["event_type"] == "game_end"]
        assert len(end_events) == 1
        assert end_events[0]["winner_name"] is not None

    @pytest.mark.parametrize("n_players,seed", [
        (2, 10), (3, 20), (4, 30)
    ])
    def test_winner_is_last_alive_player(self, n_players, seed):
        log, state = _run(n_players=n_players, seed=seed)
        assert state.is_game_over
        winner = state.winner
        assert winner is not None
        assert winner.is_alive
        # All others are dead
        for p in state.players:
            if p.idx != winner.idx:
                assert not p.is_alive

    def test_game_always_has_game_start_event(self):
        log, _ = _run()
        starts = [e for e in log["events"] if e["event_type"] == "game_start"]
        assert len(starts) == 1

    def test_game_starts_before_ends(self):
        log, _ = _run()
        events = log["events"]
        types = [e["event_type"] for e in events]
        assert types[0] == "game_start"
        assert types[-1] == "game_end"

    def test_two_player_game_terminates(self):
        for seed in range(20):
            log, state = _run(n_players=2, seed=seed)
            assert state.is_game_over


# =========================================================================
# Log structure
# =========================================================================

class TestLogStructure:
    def test_log_has_required_keys(self):
        log, _ = _run()
        assert "game_id" in log
        assert "player_names" in log
        assert "events" in log

    def test_all_action_events_have_actor_idx(self):
        log, _ = _run()
        for evt in log["events"]:
            if evt["event_type"] == "action":
                assert "actor_idx" in evt
                assert "action_type" in evt

    def test_log_is_json_serialisable(self):
        import json
        log, _ = _run()
        json.dumps(log)

    def test_influence_loss_events_name_the_card(self):
        log, _ = _run(n_players=4, seed=77)
        losses = [e for e in log["events"] if e["event_type"] == "influence_loss"]
        for loss in losses:
            assert loss["card_lost"] in [c.value for c in Card]


# =========================================================================
# Game-state invariants
# =========================================================================

class TestInvariants:
    def test_total_coins_consistent_after_game(self):
        """
        Coins are not created from thin air. Total coins in play may
        change (income/tax/foreign_aid add coins; coup removes them
        from the system), but each player's coins should be >= 0.
        """
        for seed in range(10):
            log, state = _run(n_players=3, seed=seed)
            for p in state.players:
                assert p.coins >= 0

    def test_no_player_exceeds_card_count_limit(self):
        """No player should ever hold more than 2 cards simultaneously."""
        for seed in range(10):
            log, state = _run(n_players=4, seed=seed)
            for p in state.players:
                assert len(p.hand) <= 2

    def test_card_conservation(self):
        """
        Total cards in deck + all hands + all revealed = 15 at end of game.
        """
        for seed in range(10):
            log, state = _run(n_players=3, seed=seed)
            total = len(state.deck)
            for p in state.players:
                total += len(p.hand) + len(p.revealed)
            assert total == 15


# =========================================================================
# Multi-player / agent variety
# =========================================================================

class TestAgentVariety:
    def test_honest_agents_game_terminates(self):
        agents = [HonestAgent() for _ in range(3)]
        game = Game(agents=agents, seed=0)
        log = game.play_game()
        end = [e for e in log["events"] if e["event_type"] == "game_end"]
        assert len(end) == 1

    def test_mixed_agents_game_terminates(self):
        agents = [RandomAgent(seed=i) for i in range(2)] + [HonestAgent(seed=10)]
        game = Game(agents=agents, seed=1)
        log = game.play_game()
        end = [e for e in log["events"] if e["event_type"] == "game_end"]
        assert len(end) == 1

    def test_high_challenge_prob_game_still_terminates(self):
        """Even very aggressive challengers shouldn't break the game loop."""
        for seed in range(5):
            log, state = _run(
                n_players=3, seed=seed,
                challenge_prob=0.9, block_prob=0.9
            )
            assert state.is_game_over

    def test_zero_challenge_prob_game_still_terminates(self):
        """Passive players (never challenge) should still finish."""
        for seed in range(5):
            log, state = _run(
                n_players=3, seed=seed,
                challenge_prob=0.0, block_prob=0.0
            )
            assert state.is_game_over


# =========================================================================
# Simulator smoke test
# =========================================================================

class TestSimulator:
    def test_run_simulation_returns_correct_count(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from simulator import run_simulation
        logs = run_simulation(
            n_games=10, n_players=3, seed=0, show_progress=False
        )
        assert len(logs) == 10

    def test_simulation_logs_are_valid(self):
        from simulator import run_simulation
        import json
        logs = run_simulation(
            n_games=5, n_players=2, seed=1, show_progress=False
        )
        for log in logs:
            json.dumps(log)  # Must be serialisable
            assert "events" in log
            end_evts = [e for e in log["events"] if e["event_type"] == "game_end"]
            assert len(end_evts) == 1
