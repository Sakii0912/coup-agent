"""
evaluation/match.py — Run a set of games between agent configurations.

A "match" pits one agent type (the subject) against a pool of opponents and
collects win rate, game-length, action-distribution, and behaviour statistics.

Match format:
  - n_players seats total
  - Seat 0: the subject agent
  - Seats 1..(n_players-1): opponent agents (all the same type by default)
  - Repeat n_games times, rotating which seat starts each game
  - Win rate is computed as wins / (n_games × n_subject_seats)

This seat-rotation design ensures the result is not inflated by first-mover
advantage (which is small in Coup but nonzero).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from coup.game import Game
from coup.state import GameState
from coup.agents.base import Agent


# ---------------------------------------------------------------------------
# Agent factory type
# ---------------------------------------------------------------------------

# An AgentFactory is a zero-argument callable that returns a fresh Agent.
# Using factories (not instances) ensures each game gets independent agents
# with no shared state across games.
AgentFactory = Callable[[], Agent]


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """
    Statistics from a completed match between two agent configurations.

    Attributes:
        subject_name:    Name of the agent under evaluation.
        opponent_name:   Name of the opponent agent.
        n_games:         Number of games played.
        n_players:       Players per game.
        subject_wins:    Number of games won by the subject.
        opponent_wins:   Number of games won by any opponent.
        draws:           Games with no winner (turn limit reached).
        win_rate:        subject_wins / n_games.
        avg_turns:       Mean turns per game.
        p95_turns:       95th percentile turns (game length spread).
        action_counts:   {action_type: total count by subject agent}.
        elapsed_s:       Wall-clock time for the match.
    """
    subject_name:  str
    opponent_name: str
    n_games:       int
    n_players:     int
    subject_wins:  int           = 0
    opponent_wins: int           = 0
    draws:         int           = 0
    avg_turns:     float         = 0.0
    p95_turns:     float         = 0.0
    action_counts: Dict[str, int] = field(default_factory=dict)
    elapsed_s:     float         = 0.0

    @property
    def win_rate(self) -> float:
        return self.subject_wins / max(self.n_games, 1)

    @property
    def opponent_win_rate(self) -> float:
        return self.opponent_wins / max(self.n_games, 1)

    def summary(self) -> str:
        return (
            f"{self.subject_name:<20} vs {self.opponent_name:<20}  "
            f"win={self.win_rate:5.1%}  "
            f"avg_turns={self.avg_turns:5.1f}  "
            f"({self.n_games} games, {self.elapsed_s:.1f}s)"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject":      self.subject_name,
            "opponent":     self.opponent_name,
            "n_games":      self.n_games,
            "n_players":    self.n_players,
            "subject_wins": self.subject_wins,
            "opponent_wins":self.opponent_wins,
            "draws":        self.draws,
            "win_rate":     round(self.win_rate, 4),
            "avg_turns":    round(self.avg_turns, 2),
            "p95_turns":    round(self.p95_turns, 2),
            "action_counts":self.action_counts,
            "elapsed_s":    round(self.elapsed_s, 2),
        }


# ---------------------------------------------------------------------------
# run_match
# ---------------------------------------------------------------------------

def run_match(
    subject_factory:  AgentFactory,
    opponent_factory: AgentFactory,
    subject_name:     str  = "Subject",
    opponent_name:    str  = "Opponent",
    n_games:          int  = 100,
    n_players:        int  = 4,
    seed:             int  = 0,
    show_progress:    bool = False,
) -> MatchResult:
    """
    Run `n_games` games: subject in seat 0, opponents in seats 1..(n-1).

    Args:
        subject_factory:  Callable returning a fresh subject Agent.
        opponent_factory: Callable returning a fresh opponent Agent.
        subject_name:     Label for the subject in reports.
        opponent_name:    Label for the opponent in reports.
        n_games:          Number of games to play.
        n_players:        Players per game (2–6).
        seed:             Base random seed.
        show_progress:    Print a dot every 10 games.

    Returns:
        MatchResult with win rates and statistics.
    """
    import numpy as np

    result = MatchResult(
        subject_name=subject_name,
        opponent_name=opponent_name,
        n_games=n_games,
        n_players=n_players,
    )

    turn_lengths:  List[int]       = []
    action_counts: Dict[str, int]  = defaultdict(int)
    player_names = [f"P{i}" for i in range(n_players)]
    t0 = time.time()

    for game_idx in range(n_games):
        # Subject always sits in seat 0; opponents fill other seats
        agents = [subject_factory()] + [
            opponent_factory() for _ in range(n_players - 1)
        ]

        game = Game(
            agents=agents,
            player_names=player_names,
            seed=seed + game_idx,
            verbose=False,
        )
        log = game.play_game()

        # ── Parse log ────────────────────────────────────────────────────
        end = next(
            (e for e in reversed(log["events"]) if e["event_type"] == "game_end"),
            None,
        )
        if end:
            turn_lengths.append(end.get("total_turns", 0))
            winner_idx = end.get("winner_idx")
            if winner_idx is None:
                result.draws += 1
            elif winner_idx == 0:
                result.subject_wins += 1
            else:
                result.opponent_wins += 1

        # Count subject (seat 0) action types
        for evt in log["events"]:
            if evt["event_type"] == "action" and evt.get("actor_idx") == 0:
                at = evt.get("action_type", "Unknown")
                action_counts[at] += 1

        if show_progress and (game_idx + 1) % 10 == 0:
            print(".", end="", flush=True)

    if show_progress:
        print()

    # ── Aggregate ────────────────────────────────────────────────────────
    if turn_lengths:
        arr            = np.array(turn_lengths)
        result.avg_turns = float(arr.mean())
        result.p95_turns = float(np.percentile(arr, 95))
    result.action_counts = dict(action_counts)
    result.elapsed_s     = time.time() - t0

    return result
