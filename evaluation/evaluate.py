"""
evaluation/evaluate.py — CLI for running agent evaluations (sub-goal 4.4).

Examples:
  # Quick match: Random vs Honest (100 games)
  python -m evaluation.evaluate match --subject random --opponent honest --n_games 100

  # Full tournament: all built-in agents (200 games each matchup)
  python -m evaluation.evaluate tournament --n_games 200 --n_players 4

  # Include a trained NeuralAgent
  python -m evaluation.evaluate tournament --neural_ckpt data/models/best_model.pt

  # Include a trained CFR agent
  python -m evaluation.evaluate tournament --cfr_ckpt data/models/cfr/latest.pkl.gz
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coup.agents.random_agent import RandomAgent, HonestAgent
from evaluation.match import run_match
from evaluation.tournament import Tournament
from evaluation.report import match_report, tournament_report, save_report, save_json


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def _random_factory(challenge_prob=0.2, block_prob=0.3):
    return lambda: RandomAgent(challenge_prob=challenge_prob, block_prob=block_prob)


def _honest_factory(challenge_prob=0.35, block_prob=0.3):
    return lambda: HonestAgent(challenge_prob=challenge_prob, block_prob=block_prob)


def _neural_factory(checkpoint_path: str):
    """Load a NeuralAgent from checkpoint. Returns a factory callable."""
    try:
        import torch
        from agents.neural.neural_agent import NeuralAgent
    except ImportError:
        raise RuntimeError("torch not installed — cannot load NeuralAgent")
    return lambda: NeuralAgent.from_checkpoint(checkpoint_path)


def _cfr_factory(checkpoint_path: str, n_players: int):
    """Load a CFRAgent from a strategy table checkpoint. Returns a factory."""
    from agents.cfr.strategy_table import StrategyTable

    table = StrategyTable.load(checkpoint_path)
    print(f"  CFR table loaded: {table.n_infosets:,} infosets, "
          f"{table.n_iterations:,} iterations")

    # Import here to avoid circular issues; CFRAgent wraps the table
    try:
        from agents.cfr.cfr_agent import CFRAgent
        return lambda: CFRAgent(table=table, n_players=n_players)
    except ImportError:
        # CFRAgent not yet implemented — fall back to random
        print("  Warning: CFRAgent not found, falling back to RandomAgent")
        return _random_factory()


def _build_agent_registry(args) -> dict:
    """Build the dict of {name: factory} based on CLI args."""
    agents = {}

    # Always include built-in baselines
    agents["Random"]     = _random_factory()
    agents["Honest"]     = _honest_factory()
    agents["Random-Low"] = _random_factory(challenge_prob=0.1, block_prob=0.15)
    agents["Random-High"]= _random_factory(challenge_prob=0.5, block_prob=0.5)

    # Optional: trained neural agent
    if hasattr(args, "neural_ckpt") and args.neural_ckpt:
        if os.path.exists(args.neural_ckpt):
            try:
                agents["Neural"] = _neural_factory(args.neural_ckpt)
                print(f"  NeuralAgent loaded from {args.neural_ckpt!r}")
            except Exception as e:
                print(f"  Warning: could not load NeuralAgent: {e}")
        else:
            print(f"  Warning: neural checkpoint not found: {args.neural_ckpt!r}")

    # Optional: trained RL neural agent
    if hasattr(args, "rl_ckpt") and args.rl_ckpt:
        if os.path.exists(args.rl_ckpt):
            try:
                agents["Neural-RL"] = _neural_factory(args.rl_ckpt)
                print(f"  Neural-RL loaded from {args.rl_ckpt!r}")
            except Exception as e:
                print(f"  Warning: could not load Neural-RL: {e}")

    # Optional: CFR agent
    if hasattr(args, "cfr_ckpt") and args.cfr_ckpt:
        if os.path.exists(args.cfr_ckpt):
            try:
                agents["CFR"] = _cfr_factory(args.cfr_ckpt, args.n_players)
            except Exception as e:
                print(f"  Warning: could not load CFR agent: {e}")

    return agents


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_match(args):
    """Run a single match between two named agent types."""
    agent_map = {
        "random":      _random_factory(),
        "honest":      _honest_factory(),
        "random-low":  _random_factory(challenge_prob=0.1),
        "random-high": _random_factory(challenge_prob=0.5),
    }

    if args.subject not in agent_map:
        print(f"Unknown subject agent {args.subject!r}. "
              f"Choose from: {list(agent_map.keys())}")
        sys.exit(1)
    if args.opponent not in agent_map:
        print(f"Unknown opponent agent {args.opponent!r}. "
              f"Choose from: {list(agent_map.keys())}")
        sys.exit(1)

    print(f"\nRunning match: {args.subject} vs {args.opponent} "
          f"({args.n_games} games, {args.n_players} players)")

    result = run_match(
        subject_factory  = agent_map[args.subject],
        opponent_factory = agent_map[args.opponent],
        subject_name     = args.subject.capitalize(),
        opponent_name    = args.opponent.capitalize(),
        n_games          = args.n_games,
        n_players        = args.n_players,
        seed             = args.seed,
        show_progress    = True,
    )

    report_text = match_report(result)
    print(report_text)

    if args.output:
        save_report(report_text, args.output)
        save_json(result, args.output.replace(".txt", ".json"))


def cmd_tournament(args):
    """Run a round-robin tournament across all registered agents."""
    print(f"\nBuilding agent registry...")
    agents = _build_agent_registry(args)
    print(f"  {len(agents)} agents: {list(agents.keys())}")

    t = Tournament(
        n_games   = args.n_games,
        n_players = args.n_players,
        seed      = args.seed,
        show_progress = True,
    )
    for name, factory in agents.items():
        t.register(name, factory)

    print(f"\nRunning tournament ({len(agents)} agents, "
          f"{args.n_games} games/match, {args.n_players} players)...")
    results = t.run()

    report_text = tournament_report(results)
    print(report_text)

    if args.output:
        save_report(report_text, args.output)
        save_json(results, args.output.replace(".txt", ".json"))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation.evaluate",
        description="Coup AI — Agent evaluation CLI",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── match ────────────────────────────────────────────────────────────
    p_match = sub.add_parser("match", help="Single match between two agents")
    p_match.add_argument("--subject",   type=str, default="random",
                         help="Subject agent: random|honest|random-low|random-high")
    p_match.add_argument("--opponent",  type=str, default="honest")
    p_match.add_argument("--n_games",   type=int, default=100)
    p_match.add_argument("--n_players", type=int, default=4)
    p_match.add_argument("--seed",      type=int, default=0)
    p_match.add_argument("--output",    type=str, default=None,
                         help="Save report to this .txt file")

    # ── tournament ───────────────────────────────────────────────────────
    p_tourn = sub.add_parser("tournament", help="Round-robin tournament")
    p_tourn.add_argument("--n_games",     type=int, default=200)
    p_tourn.add_argument("--n_players",   type=int, default=4)
    p_tourn.add_argument("--seed",        type=int, default=0)
    p_tourn.add_argument("--neural_ckpt", type=str, default=None,
                         help="Path to NeuralAgent checkpoint (.pt)")
    p_tourn.add_argument("--rl_ckpt",     type=str, default=None,
                         help="Path to RL-fine-tuned NeuralAgent checkpoint (.pt)")
    p_tourn.add_argument("--cfr_ckpt",    type=str, default=None,
                         help="Path to CFR strategy table (.pkl.gz)")
    p_tourn.add_argument("--output",      type=str,
                         default="data/evaluation_report.txt",
                         help="Save report to this .txt file")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    if args.mode == "match":
        cmd_match(args)
    elif args.mode == "tournament":
        cmd_tournament(args)
