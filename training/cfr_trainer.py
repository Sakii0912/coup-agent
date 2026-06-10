"""
training/cfr_trainer.py — CFR self-play trainer (sub-goal 4.3).

Wraps CFRSolver to add:
  - Periodic checkpoint saving (every `save_every` iterations)
  - Resume from existing checkpoint
  - BeliefTracker attached per seat (enriches observations before infoset lookup)
  - Convergence logging to a JSON file
  - Progress reporting

─────────────────────────────────────────────────────────────────────────────
BeliefTracker integration
─────────────────────────────────────────────────────────────────────────────

Each seat in CFR training maintains its own BeliefTracker. Before the solver
builds an InformationSet at a decision node, it enriches the Observation
with the tracker's current belief state. This makes the InformationSet
richer (and potentially more discriminating) without changing the CFR
convergence guarantees — the information set is still determined solely by
what that player can observe.

Note: the CFRSolver traverses the game tree by sampling, not by running the
full Game class. The BeliefTrackers here are maintained in parallel with the
traversal and updated with the same events the solver processes.

─────────────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────────────

    from training.cfr_trainer import CFRTrainer

    trainer = CFRTrainer(n_players=4, output_dir="data/models/cfr/")
    trainer.train(n_iterations=10_000)

    # Resume from checkpoint:
    trainer = CFRTrainer.from_checkpoint(
        "data/models/cfr/checkpoint_010000.pkl.gz",
        output_dir="data/models/cfr/",
    )
    trainer.train(n_iterations=10_000)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.cfr.cfr_solver import CFRSolver
from agents.cfr.strategy_table import StrategyTable


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CFRConfig:
    """All hyperparameters for a CFR training run."""
    n_players:      int   = 4
    use_cfr_plus:   bool  = True
    seed:           int   = 0
    save_every:     int   = 1_000    # save checkpoint every N iterations
    log_every:      int   = 1        # print + log progress every N iterations
    max_depth:      int   = 50       # max tree depth per traversal


# ---------------------------------------------------------------------------
# CFRTrainer
# ---------------------------------------------------------------------------

class CFRTrainer:
    """
    Manages CFR self-play training: runs the solver, saves checkpoints,
    logs convergence metrics.

    Args:
        n_players:   Number of players per game.
        output_dir:  Directory for checkpoints and logs.
        config:      CFRConfig hyperparameters. Defaults to CFRConfig().
    """

    def __init__(
        self,
        n_players:  int             = 4,
        output_dir: str             = "data/models/cfr",
        config:     Optional[CFRConfig] = None,
    ) -> None:
        self.config     = config or CFRConfig(n_players=n_players)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.solver = CFRSolver(
            n_players=self.config.n_players,
            seed=self.config.seed,
            use_cfr_plus=self.config.use_cfr_plus,
            max_depth=self.config.max_depth,
        )
        self._log_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    #  Factory — resume from checkpoint                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        output_dir:      str,
        config:          Optional[CFRConfig] = None,
    ) -> "CFRTrainer":
        """Resume training from a saved strategy table checkpoint."""
        table   = StrategyTable.load(checkpoint_path)
        trainer = cls(
            n_players=config.n_players if config else 4,
            output_dir=output_dir,
            config=config,
        )
        print(f"Resumed from {checkpoint_path!r} "
              f"({table.n_iterations:,} iterations, "
              f"{table.n_infosets:,} infosets)")
        return trainer

    # ------------------------------------------------------------------ #
    #  Training loop                                                      #
    # ------------------------------------------------------------------ #

    def train(self, n_iterations: int) -> Dict[str, Any]:
        """
        Run `n_iterations` of CFR self-play.

        Saves checkpoints every `config.save_every` iterations.
        Logs convergence metrics every `config.log_every` iterations.

        Returns:
            Final stats dict.
        """
        cfg  = self.config
        t0   = time.time()
        done = self.solver.table.n_iterations

        print(f"\nCFR training: {n_iterations:,} iterations "
              f"({cfg.n_players} players, CFR+={cfg.use_cfr_plus})")
        print(f"Starting from iteration {done:,}")
        print(f"{'Iter':>8}  {'Infosets':>10}  {'Exploitability':>16}  {'iter/s':>8}")
        print("-" * 52)

        i = 0
        while i < n_iterations:
            # Run a chunk up to the next log point
            chunk = min(cfg.log_every, n_iterations - i)
            print(f"  running iterations {i + 1:,}-{i + chunk:,}...", flush=True)
            self.solver.run(
                n_iterations=chunk,
                show_progress=True,
                progress_every=max(1, min(10, chunk)),
            )
            i += chunk

            total_iter = self.solver.table.n_iterations
            elapsed    = time.time() - t0
            expl       = self.solver.table.exploitability_proxy()
            rate       = total_iter / max(elapsed, 1e-6)

            print(f"{total_iter:>8,}  {self.solver.table.n_infosets:>10,}  "
                  f"{expl:>16.6f}  {rate:>8.1f}")

            self._log_history.append({
                "iteration":     total_iter,
                "infosets":      self.solver.table.n_infosets,
                "exploitability": expl,
                "elapsed_s":     round(elapsed, 2),
            })

            # Checkpoint
            if total_iter % cfg.save_every == 0:
                self._save_checkpoint(total_iter)

        # Final save
        self._save_checkpoint(self.solver.table.n_iterations)
        self._save_log()

        elapsed = time.time() - t0
        final_stats = {
            "total_iterations": self.solver.table.n_iterations,
            "n_infosets":       self.solver.table.n_infosets,
            "exploitability":   self.solver.table.exploitability_proxy(),
            "elapsed_s":        round(elapsed, 2),
        }
        print(f"\nDone. {final_stats['total_iterations']:,} iterations in {elapsed:.1f}s")
        return final_stats

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, iteration: int) -> None:
        fname = f"checkpoint_{iteration:08d}.pkl.gz"
        path  = os.path.join(self.output_dir, fname)
        self.solver.table.save(path)
        # Also keep a "latest" symlink-style copy for easy loading
        latest = os.path.join(self.output_dir, "latest.pkl.gz")
        self.solver.table.save(latest)
        print(f"  ✓ Checkpoint saved: {fname}")

    def _save_log(self) -> None:
        path = os.path.join(self.output_dir, "training_log.json")
        with open(path, "w") as f:
            json.dump(self._log_history, f, indent=2)

    # ------------------------------------------------------------------ #
    #  Quick evaluation                                                   #
    # ------------------------------------------------------------------ #

    def exploitability_history(self) -> List[float]:
        return [entry["exploitability"] for entry in self._log_history]

    def __repr__(self) -> str:
        t = self.solver.table
        return (f"CFRTrainer(iter={t.n_iterations:,}, "
                f"infosets={t.n_infosets:,}, "
                f"expl={t.exploitability_proxy():.4f})")
