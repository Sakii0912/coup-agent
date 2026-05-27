"""
pipeline/data_quality.py — Data quality report for game log corpora.

Computes and prints a structured report covering:
  - Corpus overview (game count, player counts, turn lengths)
  - Action frequency distribution
  - Win rates by seat position
  - Challenge statistics (rate, success rate per action type)
  - Block statistics (rate, success rate per action type)
  - Bluff detection rate (how often claims are false, inferred from challenge results)
  - Influence loss breakdown by card
  - Dataset-level feature statistics (mean, std, sparsity)

Usage (CLI):
    python -m pipeline.data_quality --log_dir data/raw_logs --output report.txt

Usage (API):
    from pipeline.data_quality import run_quality_report
    report = run_quality_report("data/raw_logs/")
    print(report.summary())
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from .log_parser import ParsedGame, ParsedEvent, load_logs
from .feature_extractor import (
    extract_dataset,
    FeatureConfig,
    ACTIONS,
    CARDS,
)


# ---------------------------------------------------------------------------
# Stats containers
# ---------------------------------------------------------------------------

@dataclass
class CorpusStats:
    """All computed statistics for a game log corpus."""

    # ── Corpus overview ──────────────────────────────────────────────────
    n_games: int = 0
    player_count_dist: Dict[int, int] = field(default_factory=dict)
    turn_lengths: List[int] = field(default_factory=list)

    # ── Actions ──────────────────────────────────────────────────────────
    action_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # ── Wins by seat ─────────────────────────────────────────────────────
    # seat = relative index (0 = first dealer, 1 = second, etc.)
    wins_by_seat: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    games_by_seat: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # ── Challenges ───────────────────────────────────────────────────────
    # per action type: (n_challenged, n_challenger_won)
    challenge_stats: Dict[str, List[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    total_challenges: int = 0

    # ── Blocks ───────────────────────────────────────────────────────────
    # per action type: (n_blocked, n_block_survived)
    block_stats: Dict[str, List[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    total_blocks: int = 0

    # ── Bluff detection ──────────────────────────────────────────────────
    # per claimed card: (n_challenged, n_was_bluffing)
    bluff_stats: Dict[str, List[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )

    # ── Influence loss ───────────────────────────────────────────────────
    influence_loss_by_card: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    # ── Feature stats (filled by analyse_features) ───────────────────────
    feature_mean: Optional[np.ndarray] = None
    feature_std: Optional[np.ndarray] = None
    feature_sparsity: Optional[float] = None   # fraction of near-zero values
    n_samples: int = 0


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

class QualityReport:
    """Holds computed CorpusStats and formats them for display."""

    def __init__(self, stats: CorpusStats, log_dir: str = "") -> None:
        self.stats = stats
        self.log_dir = log_dir

    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        s = self.stats
        lines: List[str] = []
        w = 60

        def h(title: str) -> str:
            return f"\n{'='*w}\n  {title}\n{'='*w}"

        def row(label: str, value: Any, width: int = 28) -> str:
            return f"  {label:<{width}} {value}"

        # ── Overview ─────────────────────────────────────────────────────
        lines.append(h("Corpus Overview"))
        lines.append(row("Games:", f"{s.n_games:,}"))
        lines.append(row("Log directory:", self.log_dir or "—"))

        if s.turn_lengths:
            arr = np.array(s.turn_lengths)
            lines.append(row("Turns per game:", f"mean={arr.mean():.1f}  "
                             f"median={np.median(arr):.0f}  "
                             f"p95={np.percentile(arr,95):.0f}  "
                             f"max={arr.max()}"))

        if s.player_count_dist:
            dist_str = "  ".join(
                f"{k}p:{v}" for k, v in sorted(s.player_count_dist.items())
            )
            lines.append(row("Player counts:", dist_str))

        lines.append(row("Training samples:", f"{s.n_samples:,}"))

        # ── Action frequencies ────────────────────────────────────────────
        lines.append(h("Action Frequencies"))
        total_actions = sum(s.action_counts.values())
        if total_actions > 0:
            lines.append(f"  {'Action':<22} {'Count':>8}  {'%':>6}")
            lines.append(f"  {'-'*38}")
            for action in ACTIONS:
                count = s.action_counts.get(action, 0)
                pct = 100 * count / total_actions
                bar = "█" * int(pct / 2)
                lines.append(f"  {action:<22} {count:>8,}  {pct:>5.1f}%  {bar}")
            lines.append(f"  {'TOTAL':<22} {total_actions:>8,}")

        # ── Win rates by seat ─────────────────────────────────────────────
        lines.append(h("Win Rates by Seat Position"))
        lines.append(f"  {'Seat':<8} {'Wins':>6}  {'Games':>6}  {'Win%':>6}")
        lines.append(f"  {'-'*32}")
        for seat in sorted(s.games_by_seat.keys()):
            n_games_seat = s.games_by_seat[seat]
            n_wins = s.wins_by_seat.get(seat, 0)
            win_pct = 100 * n_wins / n_games_seat if n_games_seat else 0
            lines.append(f"  {seat:<8} {n_wins:>6}  {n_games_seat:>6}  {win_pct:>5.1f}%")

        # ── Challenge stats ───────────────────────────────────────────────
        lines.append(h("Challenge Statistics"))
        lines.append(row("Total challenges:", f"{s.total_challenges:,}"))
        if total_actions > 0:
            lines.append(row("Challenge rate:", f"{100*s.total_challenges/total_actions:.1f}% of actions"))

        if s.total_challenges > 0:
            lines.append(f"\n  {'Action':<22} {'Challenged':>10}  {'Chall. Won':>10}  {'Bluff%':>7}")
            lines.append(f"  {'-'*54}")
            for action in ACTIONS:
                cs = s.challenge_stats.get(action, [0, 0])
                n_ch, n_win = cs[0], cs[1]
                if n_ch == 0:
                    continue
                bluff_pct = 100 * n_win / n_ch
                lines.append(
                    f"  {action:<22} {n_ch:>10,}  {n_win:>10,}  {bluff_pct:>6.1f}%"
                )

        # ── Block stats ───────────────────────────────────────────────────
        lines.append(h("Block Statistics"))
        lines.append(row("Total blocks:", f"{s.total_blocks:,}"))
        if total_actions > 0:
            lines.append(row("Block rate:", f"{100*s.total_blocks/total_actions:.1f}% of actions"))

        if s.total_blocks > 0:
            lines.append(f"\n  {'Action':<22} {'Blocked':>8}  {'Block Held':>10}  {'Hold%':>6}")
            lines.append(f"  {'-'*50}")
            for action in ACTIONS:
                bs = s.block_stats.get(action, [0, 0])
                n_bl, n_held = bs[0], bs[1]
                if n_bl == 0:
                    continue
                hold_pct = 100 * n_held / n_bl
                lines.append(
                    f"  {action:<22} {n_bl:>8,}  {n_held:>10,}  {hold_pct:>5.1f}%"
                )

        # ── Bluff rates ───────────────────────────────────────────────────
        lines.append(h("Bluff Detection (from Challenge Results)"))
        lines.append("  (Bluff% = fraction of challenges where actor was bluffing)")
        total_ch = sum(v[0] for v in s.bluff_stats.values())
        total_bluffing = sum(v[1] for v in s.bluff_stats.values())
        if total_ch > 0:
            lines.append(row("\n  Overall bluff rate:", f"{100*total_bluffing/total_ch:.1f}%"))
            lines.append(f"\n  {'Card Claimed':<22} {'Challenged':>10}  {'Bluffing':>9}  {'Bluff%':>7}")
            lines.append(f"  {'-'*52}")
            for card in CARDS:
                bs = s.bluff_stats.get(card, [0, 0])
                n_ch, n_bluff = bs[0], bs[1]
                if n_ch == 0:
                    continue
                pct = 100 * n_bluff / n_ch
                lines.append(f"  {card:<22} {n_ch:>10,}  {n_bluff:>9,}  {pct:>6.1f}%")

        # ── Influence loss by card ─────────────────────────────────────────
        lines.append(h("Influence Loss by Card"))
        total_losses = sum(s.influence_loss_by_card.values())
        if total_losses > 0:
            lines.append(f"  {'Card':<22} {'Lost':>8}  {'%':>6}")
            lines.append(f"  {'-'*38}")
            for card in CARDS:
                count = s.influence_loss_by_card.get(card, 0)
                pct = 100 * count / total_losses
                lines.append(f"  {card:<22} {count:>8,}  {pct:>5.1f}%")
            lines.append(f"  {'TOTAL':<22} {total_losses:>8,}")

        # ── Feature stats ──────────────────────────────────────────────────
        if s.feature_mean is not None:
            lines.append(h("Feature Statistics"))
            lines.append(row("Feature dim:", len(s.feature_mean)))
            lines.append(row("Samples:", f"{s.n_samples:,}"))
            lines.append(row("Mean (avg across dims):", f"{s.feature_mean.mean():.4f}"))
            lines.append(row("Std  (avg across dims):", f"{s.feature_std.mean():.4f}"))
            lines.append(row("Sparsity (fraction ~0):", f"{s.feature_sparsity:.1%}"))

        lines.append(f"\n{'='*w}\n")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.summary())
        print(f"Report saved to {path!r}")


# ---------------------------------------------------------------------------
# Statistics accumulator
# ---------------------------------------------------------------------------

def _accumulate(games: List[ParsedGame], stats: CorpusStats) -> None:
    """
    Walk every game's events once, accumulating all statistics.
    No assumptions about event ordering beyond what the validator guarantees.
    """
    # Track pending challenge context per game: which action type was being challenged
    for game in games:
        stats.n_games += 1
        stats.player_count_dist[game.n_players] = (
            stats.player_count_dist.get(game.n_players, 0) + 1
        )

        # Record per-seat win/game counts
        for seat in range(game.n_players):
            stats.games_by_seat[seat] += 1

        events = game.events

        # State for tracking within-game context
        pending_action_type: Optional[str] = None   # action under challenge
        pending_claimed_card: Optional[str] = None  # card claimed in pending action
        pending_block_action: Optional[str] = None  # action that was blocked
        in_block_challenge: bool = False

        for i, evt in enumerate(events):

            if evt.type == "action":
                pending_action_type  = evt.action_type
                pending_claimed_card = evt.claimed_card
                pending_block_action = None
                in_block_challenge   = False
                if evt.action_type:
                    stats.action_counts[evt.action_type] += 1

            elif evt.type == "challenge":
                stats.total_challenges += 1
                if pending_action_type:
                    cs = stats.challenge_stats[pending_action_type]
                    cs[0] += 1

            elif evt.type == "block":
                stats.total_blocks += 1
                if pending_action_type:
                    bs = stats.block_stats[pending_action_type]
                    bs[0] += 1
                pending_block_action = pending_action_type
                in_block_challenge   = False

            elif evt.type == "block_challenge":
                in_block_challenge = True
                stats.total_challenges += 1
                # The blocked action type is pending_block_action
                if pending_block_action:
                    cs = stats.challenge_stats[pending_block_action]
                    cs[0] += 1

            elif evt.type == "challenge_result":
                # winner_idx == actor or blocker → they held the card → challenger lost
                # Use name comparison since we have names reliably
                if evt.winner_name and evt.loser_name:
                    # Challenger (loser) called the bluff wrong → actor was honest
                    # Or challenger won → actor was bluffing
                    # We infer: if loser challenged the actor and loser lost → actor was honest
                    # To determine who was the original actor, look up pending context
                    card = pending_claimed_card
                    if card and not in_block_challenge:
                        bc = stats.bluff_stats[card]
                        bc[0] += 1
                        # challenger won challenge_result → loser is the actor → bluff detected
                        # winner_idx is challenger → actor (loser) was bluffing
                        # We can approximate: look at pending_action_type challenge stats
                        if evt.loser_idx == (events[i-2].actor_idx if i >= 2 else None):
                            bc[1] += 1  # actor was bluffing

                    # Update challenge success (challenger wins → actor was bluffing)
                    if pending_action_type and not in_block_challenge:
                        cs = stats.challenge_stats[pending_action_type]
                        # If loser is the original actor: challenger won → bluff detected
                        look_back = max(0, i - 5)
                        actor_of_action = next(
                            (e.actor_idx for e in reversed(events[look_back:i])
                             if e.type == "action"),
                            None
                        )
                        if actor_of_action is not None and evt.loser_idx == actor_of_action:
                            cs[1] += 1   # challenger won

                    # Block held / failed
                    if pending_block_action and in_block_challenge:
                        bs = stats.block_stats[pending_block_action]
                        # If the blocker lost the challenge → block failed → action proceeds
                        look_back = max(0, i - 5)
                        blocker_idx = next(
                            (e.blocker_idx for e in reversed(events[look_back:i])
                             if e.type == "block"),
                            None
                        )
                        if blocker_idx is not None and evt.winner_idx == blocker_idx:
                            bs[1] += 1   # block survived

                    elif pending_block_action and not in_block_challenge:
                        # Block was not challenged → survives implicitly (counted elsewhere)
                        pass

            elif evt.type == "action_resolved":
                # Block held if we had a block but no block challenge
                if pending_block_action and not in_block_challenge:
                    bs = stats.block_stats[pending_block_action]
                    # If we reach action_resolved with a pending block, block failed
                    # (action_resolved means the action went through, so block failed/wasn't there)
                    pass

            elif evt.type == "influence_loss":
                if evt.card_lost:
                    stats.influence_loss_by_card[evt.card_lost] += 1

            elif evt.type == "game_end":
                if evt.total_turns is not None:
                    stats.turn_lengths.append(evt.total_turns)
                if evt.winner_idx is not None:
                    stats.wins_by_seat[evt.winner_idx] += 1


def _analyse_features(
    games: List[ParsedGame], stats: CorpusStats, max_games: int = 2000
) -> None:
    """Extract features from a sample of games to compute feature-level stats."""
    sample = games[:max_games]
    config = FeatureConfig()
    X, _, _, _ = extract_dataset(sample, config=config, show_progress=False)
    stats.n_samples = len(X)
    stats.feature_mean = X.mean(axis=0)
    stats.feature_std  = X.std(axis=0)
    stats.feature_sparsity = float((np.abs(X) < 1e-6).mean())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_quality_report(
    log_dir: str,
    output_path: Optional[str] = None,
    max_feature_games: int = 2000,
    show_progress: bool = True,
) -> QualityReport:
    """
    Run the full data quality report over a corpus of game logs.

    Args:
        log_dir:           Directory containing .json log files (may have subdirs).
        output_path:       If set, write the report to this .txt file.
        max_feature_games: Max games to sample for feature statistics (expensive).
        show_progress:     Print progress messages.

    Returns:
        QualityReport object (call .summary() for formatted text).
    """
    # ── Load all logs ─────────────────────────────────────────────────────
    if show_progress:
        print(f"Loading logs from {log_dir!r}...")

    games: List[ParsedGame] = []
    if os.path.isdir(log_dir):
        for entry in sorted(os.scandir(log_dir), key=lambda e: e.name):
            if entry.is_dir():
                sub = load_logs(entry.path)
                games.extend(sub)
                if show_progress:
                    print(f"  {entry.name:<32} {len(sub):>6} games")
        top = load_logs(log_dir)
        games.extend(top)
    else:
        games = load_logs(log_dir)

    if show_progress:
        print(f"  Total: {len(games):,} games\n")

    # ── Accumulate statistics ─────────────────────────────────────────────
    if show_progress:
        print("Computing statistics...")

    stats = CorpusStats()
    _accumulate(games, stats)

    if show_progress:
        print("Computing feature statistics (may take a moment)...")
    _analyse_features(games, stats, max_games=max_feature_games)

    report = QualityReport(stats, log_dir=log_dir)

    if output_path:
        report.save(output_path)
    elif show_progress:
        print(report.summary())

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Coup data quality report")
    parser.add_argument("--log_dir",  type=str, default="data/raw_logs")
    parser.add_argument("--output",   type=str, default=None,
                        help="Save report to this .txt file (default: print to stdout)")
    parser.add_argument("--max_feat_games", type=int, default=2000)
    args = parser.parse_args()

    run_quality_report(
        log_dir=args.log_dir,
        output_path=args.output,
        max_feature_games=args.max_feat_games,
    )
