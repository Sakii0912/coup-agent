"""
agents/cfr/strategy_table.py — Regret table and strategy table for CFR.

─────────────────────────────────────────────────────────────────────────────
CFR core concepts
─────────────────────────────────────────────────────────────────────────────

  Regret:           For action a at infoset I in iteration t:
                    r_t(I, a) = counterfactual_value(I→a) - counterfactual_value(I)
                    Positive regret = "I wish I had played a more often."

  Regret matching:  σ_t(I, a) = max(0, R_t(I, a)) / Σ_b max(0, R_t(I, b))
                    where R_t = cumulative regret after t iterations.
                    If all regrets ≤ 0, play uniformly.

  Average strategy: σ̄(I, a) = Σ_t σ_t(I, a) * reach(I, t) / Σ_t reach(I, t)
                    The time-averaged strategy converges to Nash equilibrium.

  CFR+:             Like vanilla CFR but floors cumulative regrets at 0:
                    R_{t+1}(I, a) = max(0, R_t(I, a) + r_t(I, a))
                    Faster practical convergence, same theoretical guarantees.

─────────────────────────────────────────────────────────────────────────────
Storage
─────────────────────────────────────────────────────────────────────────────

  The regret table is a dict: InformationSet → {action_key: float}.
  The strategy sum table is a dict: InformationSet → {action_key: float}.
  Both use defaultdict so unseen infosets return 0 without allocation.

  For production training, these tables can be large (millions of entries).
  We provide save() / load() using pickle with optional compression.
"""

from __future__ import annotations

import os
import pickle
import gzip
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .information_set import InformationSet


# ---------------------------------------------------------------------------
# StrategyTable
# ---------------------------------------------------------------------------

class StrategyTable:
    """
    Stores cumulative regrets and strategy sums for all visited information
    sets. Provides regret-matching (current strategy) and average strategy
    computation.

    Usage:
        table = StrategyTable(use_cfr_plus=True)

        # Get current strategy for regret matching
        strategy = table.current_strategy(infoset, ["Income", "Tax", "Steal:0"])

        # After computing regrets for an iteration:
        table.accumulate_regrets(infoset, {"Income": 0.1, "Tax": -0.3, "Steal:0": 0.2})
        table.accumulate_strategy(infoset, strategy, weight=1.0)
        table.n_iterations += 1

        # Get Nash-approximating average strategy:
        avg = table.average_strategy(infoset, ["Income", "Tax", "Steal:0"])
    """

    def __init__(self, use_cfr_plus: bool = True) -> None:
        self.use_cfr_plus   = use_cfr_plus
        self.n_iterations:  int = 0

        # cumulative regrets per infoset per action
        self._regrets:  Dict[InformationSet, Dict[str, float]] = \
            defaultdict(lambda: defaultdict(float))

        # cumulative strategy sums per infoset per action (for average strategy)
        self._strategy_sums: Dict[InformationSet, Dict[str, float]] = \
            defaultdict(lambda: defaultdict(float))

    # ------------------------------------------------------------------ #
    #  Core strategy computations                                         #
    # ------------------------------------------------------------------ #

    def current_strategy(
        self,
        infoset: InformationSet,
        actions: List[str],
    ) -> Dict[str, float]:
        """
        Regret-matching strategy for the current iteration.
        Used during tree traversal (NOT for making actual decisions).

        Returns a probability distribution over `actions`.
        """
        regrets = self._regrets[infoset]
        pos = {a: max(0.0, regrets.get(a, 0.0)) for a in actions}
        total = sum(pos.values())
        if total > 1e-12:
            return {a: pos[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    def average_strategy(
        self,
        infoset: InformationSet,
        actions: List[str],
    ) -> Dict[str, float]:
        """
        Time-averaged strategy — converges to Nash equilibrium.
        Use this when the agent actually plays (not during training).

        Returns a probability distribution over `actions`.
        If no strategy has been accumulated, falls back to uniform.
        """
        sums = self._strategy_sums[infoset]
        total = sum(sums.get(a, 0.0) for a in actions)
        if total > 1e-12:
            return {a: sums.get(a, 0.0) / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    # ------------------------------------------------------------------ #
    #  Accumulation                                                       #
    # ------------------------------------------------------------------ #

    def accumulate_regrets(
        self,
        infoset: InformationSet,
        action_regrets: Dict[str, float],
    ) -> None:
        """
        Add per-action regret values to the cumulative regret table.
        With CFR+: floors each cumulative regret at 0 after update.
        """
        r = self._regrets[infoset]
        for action, regret in action_regrets.items():
            r[action] = r.get(action, 0.0) + regret
            if self.use_cfr_plus:
                r[action] = max(0.0, r[action])

    def accumulate_strategy(
        self,
        infoset: InformationSet,
        strategy: Dict[str, float],
        weight: float = 1.0,
    ) -> None:
        """
        Add the current strategy (weighted by reach probability) to the
        cumulative strategy sum. This builds the time average.
        """
        s = self._strategy_sums[infoset]
        for action, prob in strategy.items():
            s[action] = s.get(action, 0.0) + weight * prob

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    @property
    def n_infosets(self) -> int:
        """Number of unique information sets seen during training."""
        return max(len(self._regrets), len(self._strategy_sums))

    def exploitability_proxy(self) -> float:
        """
        Rough measure of how far from Nash the average strategy is.
        Computed as mean absolute regret across all infosets.
        Lower = closer to Nash equilibrium.
        """
        if not self._regrets:
            return float("inf")
        total = 0.0
        count = 0
        for regrets in self._regrets.values():
            for r in regrets.values():
                total += abs(r)
                count += 1
        return total / count if count > 0 else 0.0

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str, compress: bool = True) -> None:
        """
        Save the strategy table to disk.
        Uses gzip compression by default (~5-10× smaller than raw pickle).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "n_iterations":   self.n_iterations,
            "use_cfr_plus":   self.use_cfr_plus,
            "regrets":        dict(self._regrets),
            "strategy_sums":  dict(self._strategy_sums),
        }
        if compress:
            with gzip.open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "StrategyTable":
        """Load a strategy table from disk."""
        opener = gzip.open if path.endswith(".gz") else open
        mode   = "rb"
        with opener(path, mode) as f:
            data = pickle.load(f)

        table = cls(use_cfr_plus=data["use_cfr_plus"])
        table.n_iterations = data["n_iterations"]
        table._regrets.update(data["regrets"])
        table._strategy_sums.update(data["strategy_sums"])
        return table

    def __repr__(self) -> str:
        return (
            f"StrategyTable(infosets={self.n_infosets:,}, "
            f"iterations={self.n_iterations:,}, "
            f"cfr_plus={self.use_cfr_plus})"
        )
