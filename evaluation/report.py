"""
evaluation/report.py — Format and save evaluation reports.

Produces human-readable tables from MatchResult and TournamentResult objects.
Reports are printed to stdout and optionally saved to a .txt file.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

import numpy as np

from .match import MatchResult
from .tournament import TournamentResult


# ---------------------------------------------------------------------------
# Match report
# ---------------------------------------------------------------------------

def match_report(result: MatchResult) -> str:
    """Format a single MatchResult as a readable string."""
    lines = []
    w = 56
    lines.append("=" * w)
    lines.append(f"  Match Report")
    lines.append("=" * w)
    lines.append(f"  Subject:        {result.subject_name}")
    lines.append(f"  Opponent:       {result.opponent_name}")
    lines.append(f"  Games:          {result.n_games}  ({result.n_players} players each)")
    lines.append(f"  Elapsed:        {result.elapsed_s:.1f}s")
    lines.append("-" * w)
    lines.append(f"  Subject wins:   {result.subject_wins:>4}  ({result.win_rate:6.1%})")
    lines.append(f"  Opponent wins:  {result.opponent_wins:>4}  ({result.opponent_win_rate:6.1%})")
    lines.append(f"  Draws:          {result.draws:>4}")
    lines.append("-" * w)
    lines.append(f"  Avg turns/game: {result.avg_turns:.1f}  (p95: {result.p95_turns:.0f})")

    if result.action_counts:
        lines.append("-" * w)
        lines.append(f"  Subject action distribution:")
        total = sum(result.action_counts.values())
        for action, count in sorted(
            result.action_counts.items(), key=lambda x: -x[1]
        ):
            pct = 100 * count / max(total, 1)
            bar = "█" * int(pct / 3)
            lines.append(f"    {action:<20} {count:>6}  {pct:5.1f}%  {bar}")

    lines.append("=" * w)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tournament report
# ---------------------------------------------------------------------------

def tournament_report(result: TournamentResult) -> str:
    """Format a TournamentResult as a readable report with win-rate matrix."""
    lines = []
    w = max(70, 20 + 10 * result.n_agents)

    lines.append("=" * w)
    lines.append(f"  Tournament Report")
    lines.append("=" * w)
    lines.append(f"  Agents:      {result.n_agents}")
    lines.append(f"  Games/match: {result.n_games_each}")
    lines.append(f"  Players:     {result.n_players}")
    lines.append(f"  Elapsed:     {result.elapsed_s:.1f}s")

    # ── Win rate matrix ───────────────────────────────────────────────────
    lines.append("")
    lines.append("  Win Rate Matrix  (row = subject, column = opponent)")
    lines.append("  Values show: how often the ROW agent beats the COLUMN agent")
    lines.append("")

    col_w  = max(10, max(len(n) for n in result.agent_names) + 2)
    header = f"  {'':>{col_w}}" + "".join(
        f"{n:>{col_w}}" for n in result.agent_names
    )
    lines.append(header)
    lines.append("  " + "-" * (col_w * (result.n_agents + 1)))

    for i, row_name in enumerate(result.agent_names):
        row = f"  {row_name:>{col_w}}"
        for j in range(result.n_agents):
            v = result.win_matrix[i, j]
            if np.isnan(v):
                row += f"{'—':>{col_w}}"
            else:
                row += f"{v:>{col_w}.1%}"
        lines.append(row)

    # ── Ranking ───────────────────────────────────────────────────────────
    lines.append("")
    lines.append("  Overall Ranking  (by average win rate across all opponents)")
    lines.append("  " + "-" * 44)
    lines.append(f"  {'Rank':<6} {'Agent':<24} {'Avg Win Rate':>12}")
    lines.append("  " + "-" * 44)

    for rank, (name, wr) in enumerate(result.ranking(), start=1):
        bar    = "█" * int(wr * 20)
        medal  = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        lines.append(f"  {rank:<6} {name:<24} {wr:>11.1%}  {bar}")

    # ── Per-match summary ─────────────────────────────────────────────────
    lines.append("")
    lines.append("  All Matchups")
    lines.append("  " + "-" * 60)
    for m in sorted(result.match_results, key=lambda x: -x.win_rate):
        lines.append(
            f"  {m.subject_name:<20} vs {m.opponent_name:<20}  "
            f"win={m.win_rate:5.1%}  turns={m.avg_turns:.1f}"
        )

    lines.append("=" * w)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_report(text: str, path: str) -> None:
    """Save a formatted report string to a .txt file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"Report saved to {path!r}")


def save_json(result: Any, path: str) -> None:
    """Save a MatchResult or TournamentResult as JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"JSON saved to {path!r}")
