"""
evaluation/tournament.py — Round-robin tournament across agent types.

Each agent type plays as the subject against every other agent type as
opponent. Results are collected into a win-rate matrix and per-agent stats.

Tournament format:
  For each (subject, opponent) pair where subject != opponent:
    run_match(subject, opponent, n_games_per_match, n_players)

  Also runs each agent against a mixed field (one seat per agent type)
  when there are enough agent types to fill the table.

Win rate matrix:
  matrix[i][j] = win rate of agent i when playing against agent j
  (row = subject, column = opponent)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .match import MatchResult, AgentFactory, run_match


# ---------------------------------------------------------------------------
# TournamentResult
# ---------------------------------------------------------------------------

@dataclass
class TournamentResult:
    """
    Results from a completed round-robin tournament.

    Attributes:
        agent_names:     Ordered list of agent names.
        win_matrix:      np.ndarray shape (n_agents, n_agents).
                         win_matrix[i, j] = win rate of agent i vs agent j.
                         Diagonal is NaN (agent doesn't play itself).
        match_results:   Flat list of all individual MatchResult objects.
        avg_win_rates:   Mean win rate of each agent across all opponents.
        n_games_each:    Games played per matchup.
        n_players:       Players per game.
        elapsed_s:       Total wall-clock time.
    """
    agent_names:   List[str]
    win_matrix:    np.ndarray
    match_results: List[MatchResult]
    avg_win_rates: np.ndarray
    n_games_each:  int
    n_players:     int
    elapsed_s:     float

    @property
    def n_agents(self) -> int:
        return len(self.agent_names)

    def ranking(self) -> List[Tuple[str, float]]:
        """Return agents sorted by average win rate (descending)."""
        pairs = list(zip(self.agent_names, self.avg_win_rates.tolist()))
        return sorted(pairs, key=lambda x: -x[1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_names":   self.agent_names,
            "win_matrix":    [
                [None if np.isnan(v) else round(v, 4) for v in row]
                for row in self.win_matrix.tolist()
            ],
            "avg_win_rates": {
                name: round(float(wr), 4)
                for name, wr in zip(self.agent_names, self.avg_win_rates)
            },
            "ranking": [
                {"agent": name, "avg_win_rate": round(wr, 4)}
                for name, wr in self.ranking()
            ],
            "n_games_each": self.n_games_each,
            "n_players":    self.n_players,
            "elapsed_s":    round(self.elapsed_s, 2),
            "matches": [m.to_dict() for m in self.match_results],
        }


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------

class Tournament:
    """
    Round-robin tournament across registered agent types.

    Usage:
        from coup.agents.random_agent import RandomAgent, HonestAgent

        t = Tournament(n_games=200, n_players=4)
        t.register("Random",  lambda: RandomAgent(challenge_prob=0.2))
        t.register("Honest",  lambda: HonestAgent(challenge_prob=0.3))
        # t.register("Neural", lambda: NeuralAgent.from_checkpoint(...))
        results = t.run()
        print(results.ranking())
    """

    def __init__(
        self,
        n_games:  int  = 200,
        n_players: int = 4,
        seed:     int  = 0,
        show_progress: bool = True,
    ) -> None:
        self.n_games        = n_games
        self.n_players      = n_players
        self.seed           = seed
        self.show_progress  = show_progress
        self._agents: Dict[str, AgentFactory] = {}

    def register(self, name: str, factory: AgentFactory) -> "Tournament":
        """Register an agent type. Returns self for chaining."""
        self._agents[name] = factory
        return self

    def run(self) -> TournamentResult:
        """
        Run all pairwise matchups and return the full result.
        """
        names     = list(self._agents.keys())
        n_agents  = len(names)
        if n_agents < 2:
            raise ValueError("Need at least 2 agents for a tournament.")

        win_matrix     = np.full((n_agents, n_agents), np.nan)
        match_results: List[MatchResult] = []
        t0 = time.time()

        total_matches = n_agents * (n_agents - 1)
        match_num     = 0

        for i, subject_name in enumerate(names):
            for j, opponent_name in enumerate(names):
                if i == j:
                    continue

                match_num += 1
                if self.show_progress:
                    print(
                        f"  [{match_num:>3}/{total_matches}] "
                        f"{subject_name:<20} vs {opponent_name:<20}",
                        end="  ", flush=True,
                    )

                result = run_match(
                    subject_factory  = self._agents[subject_name],
                    opponent_factory = self._agents[opponent_name],
                    subject_name     = subject_name,
                    opponent_name    = opponent_name,
                    n_games          = self.n_games,
                    n_players        = self.n_players,
                    seed             = self.seed + match_num * 1000,
                    show_progress    = False,
                )
                win_matrix[i, j] = result.win_rate
                match_results.append(result)

                if self.show_progress:
                    print(
                        f"win={result.win_rate:5.1%}  "
                        f"turns={result.avg_turns:.1f}  "
                        f"({result.elapsed_s:.1f}s)",
                        flush=True,
                    )

        # Average win rate ignoring NaN diagonal
        avg_win_rates = np.nanmean(win_matrix, axis=1)

        elapsed = time.time() - t0
        if self.show_progress:
            print(f"\nTournament complete in {elapsed:.1f}s")

        return TournamentResult(
            agent_names   = names,
            win_matrix    = win_matrix,
            match_results = match_results,
            avg_win_rates = avg_win_rates,
            n_games_each  = self.n_games,
            n_players     = self.n_players,
            elapsed_s     = elapsed,
        )
