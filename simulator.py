"""
simulator.py — Self-play simulator.

Runs N games with configurable agents and collects/saves game logs.
This is the data-generation engine for Phase 2 / Phase 4 training.

Usage (CLI):
    python simulator.py --n_games 1000 --n_players 4 --output_dir data/raw_logs

Usage (API):
    from simulator import run_simulation
    logs = run_simulation(n_games=500, n_players=4, seed=42)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type

from coup.game import Game
from coup.agents.base import Agent
from coup.agents.random_agent import RandomAgent, HonestAgent


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------

def run_simulation(
    n_games: int,
    n_players: int = 4,
    agent_class: Type[Agent] = RandomAgent,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run `n_games` complete Coup games and return all game logs.

    Args:
        n_games:       Number of games to simulate.
        n_players:     Players per game (2–6).
        agent_class:   Agent class to instantiate for every seat.
        agent_kwargs:  Kwargs passed to each agent constructor.
        output_dir:    If set, save individual game logs as JSON files here.
        seed:          Base random seed. Each game uses seed + game_idx so
                       runs are fully reproducible.
        verbose:       Print turn-by-turn detail for each game.
        show_progress: Print a progress bar.

    Returns:
        List of game log dicts (JSON-serialisable).
    """
    if agent_kwargs is None:
        agent_kwargs = {}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    player_names = [f"P{i}" for i in range(n_players)]
    all_logs: List[Dict[str, Any]] = []

    start_time = time.time()

    for game_idx in range(n_games):
        if show_progress and game_idx % max(1, n_games // 20) == 0:
            pct = 100 * game_idx / n_games
            elapsed = time.time() - start_time
            print(f"  [{pct:5.1f}%] game {game_idx:>6}/{n_games}   "
                  f"elapsed {elapsed:.1f}s", flush=True)

        agents = [agent_class(**agent_kwargs) for _ in range(n_players)]
        game_seed = (seed + game_idx) if seed is not None else None

        game = Game(
            agents=agents,
            player_names=player_names,
            seed=game_seed,
            verbose=verbose,
        )
        log = game.play_game()
        all_logs.append(log)

        if output_dir:
            path = os.path.join(output_dir, f"game_{game_idx:06d}.json")
            with open(path, "w") as f:
                json.dump(log, f)

    elapsed = time.time() - start_time
    print(f"  [100.0%] done — {n_games} games in {elapsed:.2f}s "
          f"({n_games / elapsed:.0f} games/s)", flush=True)

    _print_summary(all_logs, n_players)
    return all_logs


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _print_summary(logs: List[Dict[str, Any]], n_players: int) -> None:
    """Print win-rate, average game length, and action-frequency stats."""
    wins: Dict[str, int] = defaultdict(int)
    turn_counts: List[int] = []
    action_counts: Dict[str, int] = defaultdict(int)
    challenge_attempts = 0
    challenge_successes = 0

    for log in logs:
        events = log["events"]

        # Winner and turn count
        end = next(
            (e for e in reversed(events) if e["event_type"] == "game_end"), None
        )
        if end:
            if end.get("winner_name"):
                wins[end["winner_name"]] += 1
            if end.get("total_turns") is not None:
                turn_counts.append(end["total_turns"])

        # Action frequencies
        for evt in events:
            if evt["event_type"] == "action":
                action_counts[evt["action_type"]] += 1
            elif evt["event_type"] == "challenge_result":
                challenge_attempts += 1
                # winner_idx != actor_idx means challenger won
                # (we can't tell without actor_idx, so just count totals)

    n = len(logs)
    avg_turns = sum(turn_counts) / max(len(turn_counts), 1)

    print(f"\n{'='*50}")
    print(f"Simulation summary — {n} games, {n_players} players")
    print(f"{'='*50}")
    print(f"Avg turns / game:  {avg_turns:.1f}")
    print(f"\nWin rates:")
    for name in sorted(wins):
        count = wins[name]
        print(f"  {name}: {count:>5}/{n}  ({100*count/n:5.1f}%)")
    print(f"\nAction frequencies:")
    total_actions = sum(action_counts.values())
    for atype in sorted(action_counts, key=action_counts.get, reverse=True):
        count = action_counts[atype]
        print(f"  {atype:<20} {count:>7}  ({100*count/total_actions:5.1f}%)")
    print("="*50)


# ---------------------------------------------------------------------------
# Mixed-agent simulation (for future use in Phase 4 opponent modelling)
# ---------------------------------------------------------------------------

def run_mixed_simulation(
    n_games: int,
    agent_pool: List[Type[Agent]],
    agent_kwargs_pool: Optional[List[Dict[str, Any]]] = None,
    n_players: int = 4,
    output_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run games where each seat is filled by a randomly chosen agent type
    from `agent_pool`. Useful for generating diverse training data.
    """
    import random as _random
    if agent_kwargs_pool is None:
        agent_kwargs_pool = [{} for _ in agent_pool]

    assert len(agent_pool) == len(agent_kwargs_pool)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rng = _random.Random(seed)
    all_logs: List[Dict[str, Any]] = []
    player_names = [f"P{i}" for i in range(n_players)]

    for game_idx in range(n_games):
        # Each seat independently draws an agent type
        agents = []
        for _ in range(n_players):
            choice_idx = rng.randrange(len(agent_pool))
            agents.append(agent_pool[choice_idx](**agent_kwargs_pool[choice_idx]))

        game_seed = (seed + game_idx) if seed is not None else None
        game = Game(agents=agents, player_names=player_names, seed=game_seed)
        log = game.play_game()
        all_logs.append(log)

        if output_dir:
            path = os.path.join(output_dir, f"game_{game_idx:06d}.json")
            with open(path, "w") as f:
                json.dump(log, f)

    _print_summary(all_logs, n_players)
    return all_logs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coup self-play simulator")
    parser.add_argument("--n_games", type=int, default=1_000)
    parser.add_argument("--n_players", type=int, default=4, choices=[2, 3, 4, 5, 6])
    parser.add_argument("--output_dir", type=str, default="data/raw_logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--challenge_prob", type=float, default=0.2)
    parser.add_argument("--block_prob", type=float, default=0.3)
    parser.add_argument("--agent", type=str, default="random",
                        choices=["random", "honest", "mixed"])
    args = parser.parse_args()

    print(f"Coup simulator — {args.n_games} games × {args.n_players} players")
    print(f"Agent: {args.agent} | seed: {args.seed} | output: {args.output_dir}\n")

    if args.agent == "random":
        run_simulation(
            n_games=args.n_games,
            n_players=args.n_players,
            agent_class=RandomAgent,
            agent_kwargs={"challenge_prob": args.challenge_prob,
                          "block_prob": args.block_prob},
            output_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.agent == "honest":
        run_simulation(
            n_games=args.n_games,
            n_players=args.n_players,
            agent_class=HonestAgent,
            agent_kwargs={"challenge_prob": args.challenge_prob,
                          "block_prob": args.block_prob},
            output_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.agent == "mixed":
        run_mixed_simulation(
            n_games=args.n_games,
            n_players=args.n_players,
            agent_pool=[RandomAgent, HonestAgent],
            agent_kwargs_pool=[
                {"challenge_prob": args.challenge_prob, "block_prob": args.block_prob},
                {"challenge_prob": args.challenge_prob, "block_prob": args.block_prob},
            ],
            output_dir=args.output_dir,
            seed=args.seed,
        )
