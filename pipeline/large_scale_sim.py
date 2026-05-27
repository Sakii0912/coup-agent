"""
pipeline/large_scale_sim.py — Large-scale synthetic data generator.

Produces a diverse corpus of game logs by running batches of games across
multiple agent configurations and player counts. Each batch writes its logs
into a separate subdirectory of the output root so runs are reproducible
and resumable.

Target: ~100k games covering a wide range of challenge/block probabilities,
player counts (2–6), and agent types (Random vs Honest).

Directory layout after a full run:
    data/raw_logs/
        random_2p_low/      game_000000.json …
        random_3p_low/
        random_4p_low/
        …
        honest_4p_high/
        mixed_4p/
        …

Usage (CLI):
    python -m pipeline.large_scale_sim --total 100000 --output_dir data/raw_logs
    python -m pipeline.large_scale_sim --total 10000  --output_dir data/raw_logs --dry_run

Usage (API):
    from pipeline.large_scale_sim import run_large_scale
    run_large_scale(total_games=50000, output_dir="data/raw_logs", seed=42)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coup.game import Game
from coup.agents.base import Agent
from coup.agents.random_agent import RandomAgent, HonestAgent


# ---------------------------------------------------------------------------
# Batch configuration
# ---------------------------------------------------------------------------

@dataclass
class BatchConfig:
    """
    One simulation batch: a specific combination of agent type,
    player count, and behavioural parameters.
    """
    name: str                         # subdirectory name + log label
    agent_class: Type[Agent]
    agent_kwargs: Dict[str, Any]
    n_players: int
    weight: float = 1.0               # relative share of total_games budget

    def __repr__(self) -> str:
        return (
            f"BatchConfig({self.name!r}, "
            f"n_players={self.n_players}, weight={self.weight})"
        )


def default_batch_configs() -> List[BatchConfig]:
    """
    Returns the standard set of 18 batch configurations used for training.

    Design rationale:
      - Three challenge/block probability tiers: low (0.1), medium (0.3), high (0.7)
      - Three player counts: 2, 4, 6  (covers 2-player strategic, mid-range, full table)
      - Two agent types: Random (diverse bluffing) and Honest (no bluffing baseline)
      - One mixed batch at 4p (equal mix of both) for diversity
      - Weights are tuned so 4-player games dominate (most common real-world setup)
    """
    configs: List[BatchConfig] = []

    # Random agent × 3 challenge tiers × 3 player counts
    for n_players, p_weight in [(2, 0.8), (4, 2.0), (6, 1.0)]:
        for tier, (cp, bp) in [("low", (0.1, 0.15)),
                                ("med", (0.3, 0.35)),
                                ("high", (0.7, 0.6))]:
            configs.append(BatchConfig(
                name=f"random_{n_players}p_{tier}",
                agent_class=RandomAgent,
                agent_kwargs={"challenge_prob": cp, "block_prob": bp},
                n_players=n_players,
                weight=p_weight,
            ))

    # Honest agent × 3 player counts (medium challenge only — honest agents
    # behave consistently across tiers)
    for n_players, p_weight in [(2, 0.5), (4, 1.2), (6, 0.7)]:
        configs.append(BatchConfig(
            name=f"honest_{n_players}p",
            agent_class=HonestAgent,
            agent_kwargs={"challenge_prob": 0.4, "block_prob": 0.3},
            n_players=n_players,
            weight=p_weight,
        ))

    # Mixed batch: random + honest seats interleaved
    configs.append(BatchConfig(
        name="mixed_4p",
        agent_class=RandomAgent,       # placeholder; _run_mixed_batch handles mixing
        agent_kwargs={"challenge_prob": 0.3, "block_prob": 0.35},
        n_players=4,
        weight=1.5,
    ))

    return configs


# ---------------------------------------------------------------------------
# Per-batch runner
# ---------------------------------------------------------------------------

def _run_batch(
    config: BatchConfig,
    n_games: int,
    output_dir: str,
    base_seed: int,
    start_idx: int = 0,
    show_progress: bool = True,
) -> Tuple[int, int]:
    """
    Run `n_games` for a single BatchConfig, saving logs to `output_dir`.

    Returns:
        (n_saved, n_failed) counts.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_saved = 0
    n_failed = 0

    player_names = [f"P{i}" for i in range(config.n_players)]
    is_mixed = config.name == "mixed_4p"

    t0 = time.time()
    for i in range(n_games):
        global_idx = start_idx + i
        game_seed = base_seed + global_idx

        if is_mixed:
            agents = _make_mixed_agents(config.n_players, game_seed)
        else:
            agents = [
                config.agent_class(
                    **config.agent_kwargs,
                    seed=game_seed * 100 + seat_idx,
                )
                for seat_idx in range(config.n_players)
            ]

        try:
            game = Game(
                agents=agents,
                player_names=player_names,
                seed=game_seed,
                verbose=False,
            )
            log = game.play_game()

            path = os.path.join(output_dir, f"game_{global_idx:07d}.json")
            with open(path, "w") as f:
                json.dump(log, f)
            n_saved += 1

        except Exception as e:
            n_failed += 1
            if show_progress:
                print(f"    ⚠ game {global_idx} failed: {e}")

        if show_progress and (i + 1) % max(1, n_games // 10) == 0:
            pct = 100 * (i + 1) / n_games
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.001)
            print(
                f"    [{pct:5.1f}%] {i+1:>6}/{n_games}  "
                f"{rate:>6.0f} games/s",
                flush=True,
            )

    return n_saved, n_failed


def _make_mixed_agents(n_players: int, seed: int) -> List[Agent]:
    """
    Build a seat list alternating Random and Honest agents.
    Seeds are offset per seat for independence.
    """
    import random as _rng
    rng = _rng.Random(seed)
    agents: List[Agent] = []
    for i in range(n_players):
        cp = rng.uniform(0.1, 0.7)
        bp = rng.uniform(0.1, 0.6)
        if rng.random() < 0.5:
            agents.append(RandomAgent(challenge_prob=cp, block_prob=bp))
        else:
            agents.append(HonestAgent(challenge_prob=cp, block_prob=bp))
    return agents


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_large_scale(
    total_games: int = 100_000,
    output_dir: str = "data/raw_logs",
    configs: Optional[List[BatchConfig]] = None,
    seed: int = 0,
    show_progress: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run the full large-scale simulation across all batch configurations.

    Games are distributed across configs proportional to their `weight`.
    Each config writes to its own subdirectory under `output_dir`.

    Args:
        total_games:    Target total number of games across all batches.
        output_dir:     Root output directory.
        configs:        List of BatchConfig objects. Defaults to
                        `default_batch_configs()`.
        seed:           Base random seed. Each game uses a unique offset.
        show_progress:  Print progress lines during simulation.
        dry_run:        Print the plan without running any games.

    Returns:
        Summary dict with per-batch counts and totals.
    """
    if configs is None:
        configs = default_batch_configs()

    # ── Allocate games proportionally ────────────────────────────────────
    total_weight = sum(c.weight for c in configs)
    allocations: List[Tuple[BatchConfig, int]] = []
    allocated = 0
    for i, cfg in enumerate(configs):
        if i < len(configs) - 1:
            n = max(1, round(total_games * cfg.weight / total_weight))
        else:
            n = total_games - allocated   # give remainder to last batch
        allocations.append((cfg, n))
        allocated += n

    # ── Print plan ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Large-scale simulation plan")
    print(f"  Total target:  {total_games:,} games")
    print(f"  Output root:   {output_dir}/")
    print(f"  Seed base:     {seed}")
    print(f"  Batches:       {len(configs)}")
    print(f"{'='*60}")
    print(f"{'Batch':<28} {'Players':>7} {'Games':>8}  Subdir")
    print(f"{'-'*60}")
    cumulative = 0
    for cfg, n in allocations:
        subdir = os.path.join(output_dir, cfg.name)
        print(f"  {cfg.name:<26} {cfg.n_players:>7} {n:>8}  {subdir}")
        cumulative += n
    print(f"{'─'*60}")
    print(f"  {'TOTAL':<26} {'':>7} {cumulative:>8}")
    print(f"{'='*60}\n")

    if dry_run:
        print("Dry run — no games simulated.")
        return {"dry_run": True, "plan": [(c.name, n) for c, n in allocations]}

    # ── Run batches ───────────────────────────────────────────────────────
    t_start = time.time()
    summary: Dict[str, Any] = {"batches": {}, "total_saved": 0, "total_failed": 0}
    game_counter = 0

    for cfg, n_games in allocations:
        subdir = os.path.join(output_dir, cfg.name)
        print(f"▶ {cfg.name}  ({n_games:,} games, {cfg.n_players}p)", flush=True)

        n_saved, n_failed = _run_batch(
            config=cfg,
            n_games=n_games,
            output_dir=subdir,
            base_seed=seed,
            start_idx=game_counter,
            show_progress=show_progress,
        )

        summary["batches"][cfg.name] = {
            "n_games": n_games,
            "n_saved": n_saved,
            "n_failed": n_failed,
            "n_players": cfg.n_players,
            "subdir": subdir,
        }
        summary["total_saved"] += n_saved
        summary["total_failed"] += n_failed
        game_counter += n_games

        elapsed_batch = time.time() - t_start
        print(
            f"  ✓ {n_saved:,} saved  "
            f"{n_failed} failed  "
            f"({elapsed_batch:.1f}s elapsed)\n",
            flush=True,
        )

    elapsed = time.time() - t_start
    rate = summary["total_saved"] / max(elapsed, 0.001)
    summary["elapsed_s"] = round(elapsed, 2)
    summary["games_per_second"] = round(rate, 1)

    print(f"{'='*60}")
    print(f"Simulation complete")
    print(f"  Saved:   {summary['total_saved']:,} games")
    print(f"  Failed:  {summary['total_failed']}")
    print(f"  Elapsed: {elapsed:.1f}s  ({rate:.0f} games/s)")
    print(f"{'='*60}\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Large-scale Coup self-play data generator"
    )
    parser.add_argument(
        "--total", type=int, default=100_000,
        help="Total number of games to simulate (default: 100,000)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/raw_logs",
        help="Root output directory for all game logs"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed (default: 0)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print the plan without running any games"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-batch progress output"
    )
    args = parser.parse_args()

    run_large_scale(
        total_games=args.total,
        output_dir=args.output_dir,
        seed=args.seed,
        show_progress=not args.quiet,
        dry_run=args.dry_run,
    )
