"""
belief/constraint_normaliser.py — Global deck constraint propagation (sub-goal 3.3).

─────────────────────────────────────────────────────────────────────────────
The problem with independent per-player updates
─────────────────────────────────────────────────────────────────────────────

After sub-goal 3.2's Bayesian updates, individual rows of the probability
matrix are locally consistent (each player's row sums reasonably given their
influence count), but the COLUMNS can violate global card conservation.

Example violation: 4-player game, 3 Dukes in the deck.
  After P0, P1, P2 all claim Duke in succession, the updater raises each
  of their Duke probabilities. But if:
    P0: Duke prob = 0.85
    P1: Duke prob = 0.85
    P2: Duke prob = 0.85
    P3: Duke prob = 0.40

  The expected total Dukes held = 0.85 + 0.85 + 0.85 + 0.40 = 2.95 ≈ 3.
  That's fine here. But if claims are very strong or the game is late-stage,
  we could easily overshoot 3.0 — meaning the beliefs imply more Dukes than
  actually exist. That's mathematically impossible.

─────────────────────────────────────────────────────────────────────────────
The constraint
─────────────────────────────────────────────────────────────────────────────

For each card c, the sum of expected holdings across all players cannot
exceed available[c] (the number of copies not yet revealed or in our hand):

    Σ_i  E[copies of c held by player i]  ≤  available[c]

We approximate E[copies of c held by i] ≈ probs[i, c] × influence_counts[i]
(treating the Bernoulli probability of "holds ≥1 copy" as a proxy for
expected count). This is an approximation — a player holding 2 cards with
P(Duke) = 0.8 could hold 0, 1, or 2 Dukes — but it's tractable and
converges to the correct marginals under iterative normalisation.

─────────────────────────────────────────────────────────────────────────────
The algorithm: Sinkhorn-style iterative column-then-row scaling
─────────────────────────────────────────────────────────────────────────────

We alternate between two normalisation passes until convergence:

  COLUMN PASS (card conservation):
    For each card c:
      cap = available[c]
      total_expected = Σ_i  probs[i, c]  (where player i is alive)
      if total_expected > cap + ε:
          scale each probs[i, c] by  cap / total_expected

  ROW PASS (influence-count consistency):
    For each player i:
      Keep pinned values (certainties: 0.0 or 1.0) unchanged.
      For uncertain values, clip to [0, 1] and ensure they don't exceed
      the hypergeometric prior (a player can't have more of a card than
      the pool allows).

  Repeat until max change < convergence_tol or max_iterations reached.

The column pass enforces conservation. The row pass prevents overcorrection.
Together they converge quickly (typically 3–5 iterations).

─────────────────────────────────────────────────────────────────────────────
Pinned cells
─────────────────────────────────────────────────────────────────────────────

Some cells are "pinned" — they carry certainty and must not be modified:
  - Observer's own hand: probs[observer_idx, c] ∈ {0.0, 1.0}
  - Dead players: probs[dead_player, :] = 0.0  (all zeros)
  - Bluff-detected zeros: probs[i, c] = 0.0 after a lost challenge
    (player i definitely doesn't hold c — this is hard evidence)

Pinned cells are excluded from scaling. Only free (non-pinned) cells
are adjusted. This preserves hard evidence while correcting soft beliefs.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from .belief_state import (
    BeliefState,
    CARDS,
    N_CARDS,
    CARD_TO_IDX,
    TOTAL_CARDS,
    COPIES_PER_CARD,
    CERTAINTY_THRESHOLD,
    hypergeometric_at_least_one,
    _encode_known_hand,
)
from .belief_updater import _current_pool_size, _hypergeometric_prior


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_ITERATIONS: int = 20
DEFAULT_CONVERGENCE_TOL: float = 1e-6


# ---------------------------------------------------------------------------
# ConstraintNormaliser
# ---------------------------------------------------------------------------

class ConstraintNormaliser:
    """
    Enforces global deck-conservation constraints on a BeliefState.

    After any Bayesian update that may have pushed column sums above the
    available card counts, call `normalise(belief)` to restore consistency.

    The normaliser is stateless — it reads everything it needs from the
    BeliefState. It can be called after every update or batched; both
    are correct. Calling it more often is safer.

    Usage:
        normaliser = ConstraintNormaliser()
        normaliser.normalise(belief_state)   # mutates in place
    """

    def __init__(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        convergence_tol: float = DEFAULT_CONVERGENCE_TOL,
    ) -> None:
        self.max_iterations  = max_iterations
        self.convergence_tol = convergence_tol

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def normalise(self, belief: BeliefState) -> int:
        """
        Run iterative Sinkhorn-style normalisation on `belief` in place.

        Returns:
            Number of iterations run before convergence.
        """
        pin_mask = self._build_pin_mask(belief)
        prior_ceiling = self._build_prior_ceiling(belief)

        for iteration in range(self.max_iterations):
            old_probs = belief.probs.copy()

            self._column_pass(belief, pin_mask)
            self._row_pass(belief, pin_mask, prior_ceiling)

            delta = np.abs(belief.probs - old_probs).max()
            if delta < self.convergence_tol:
                return iteration + 1

        return self.max_iterations

    def column_sums(self, belief: BeliefState) -> np.ndarray:
        """
        For each card, the total expected holdings across all alive players.
        shape: (N_CARDS,). Used for diagnostics and tests.
        """
        sums = np.zeros(N_CARDS, dtype=np.float64)
        for p_idx in range(belief.n_players):
            if belief.influence_counts[p_idx] > 0:
                sums += belief.probs[p_idx]
        return sums

    def conservation_violations(self, belief: BeliefState) -> np.ndarray:
        """
        For each card, the amount by which expected holdings exceed available[c].
        Positive = violation. Zero or negative = satisfied.
        shape: (N_CARDS,).
        """
        return self.column_sums(belief) - belief.available

    def is_consistent(self, belief: BeliefState, tol: float = 1e-6) -> bool:
        """
        Return True iff all conservation constraints are satisfied within `tol`
        and all probabilities are in [0, 1].
        """
        if np.any(belief.probs < -tol) or np.any(belief.probs > 1.0 + tol):
            return False
        if np.any(self.conservation_violations(belief) > tol):
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Pin mask                                                           #
    # ------------------------------------------------------------------ #

    def _build_pin_mask(self, belief: BeliefState) -> np.ndarray:
        """
        Build a boolean mask of shape (n_players, N_CARDS) where True means
        the cell is pinned (must not be modified by normalisation).

        Pinned cells:
          - All cells for dead players (entire row)
          - All cells for the observer's known hand (entire row)
          - Any cell that is exactly 0.0 due to a hard exclusion (bluff detected)
            AND whose column has available[c] == 0 (card fully accounted for)
        """
        n = belief.n_players
        mask = np.zeros((n, N_CARDS), dtype=bool)

        for p_idx in range(n):
            # Dead players — entire row pinned at 0
            if belief.influence_counts[p_idx] == 0:
                mask[p_idx, :] = True
                continue

            # Observer with known hand — entire row pinned
            if p_idx == belief.observer_idx and belief.known_hand is not None:
                mask[p_idx, :] = True
                continue

            # Hard-zero cells: probs[i,c] was set to exactly 0.0 by bluff detection.
            # We pin these to preserve the hard evidence.
            for c_idx in range(N_CARDS):
                if belief.probs[p_idx, c_idx] < CERTAINTY_THRESHOLD:
                    mask[p_idx, c_idx] = True

        return mask

    # ------------------------------------------------------------------ #
    #  Prior ceiling                                                      #
    # ------------------------------------------------------------------ #

    def _build_prior_ceiling(self, belief: BeliefState) -> np.ndarray:
        """
        For each (player, card), the hypergeometric prior probability — used
        as an upper ceiling during the row pass to prevent overcorrection.
        shape: (n_players, N_CARDS).
        """
        ceiling = np.zeros((belief.n_players, N_CARDS), dtype=np.float64)
        pool = _current_pool_size(belief)
        for p_idx in range(belief.n_players):
            inf = belief.influence_counts[p_idx]
            if inf == 0:
                continue
            for c_idx in range(N_CARDS):
                ceiling[p_idx, c_idx] = hypergeometric_at_least_one(
                    pool_size=pool,
                    n_hits=int(belief.available[c_idx]),
                    draw_size=inf,
                )
        return ceiling

    # ------------------------------------------------------------------ #
    #  Sinkhorn passes                                                    #
    # ------------------------------------------------------------------ #

    def _column_pass(self, belief: BeliefState, pin_mask: np.ndarray) -> None:
        """
        Enforce card conservation: for each card c, scale down free cells in
        column c if the total expected holdings exceed available[c].
        """
        for c_idx in range(N_CARDS):
            cap = belief.available[c_idx]

            # Sum of pinned cells in this column (they consume some of the budget)
            pinned_sum = 0.0
            for p_idx in range(belief.n_players):
                if pin_mask[p_idx, c_idx]:
                    pinned_sum += belief.probs[p_idx, c_idx]

            remaining_cap = cap - pinned_sum

            # Sum of free cells
            free_sum = 0.0
            for p_idx in range(belief.n_players):
                if not pin_mask[p_idx, c_idx]:
                    free_sum += belief.probs[p_idx, c_idx]

            # Scale down free cells if they exceed remaining capacity
            if free_sum > remaining_cap + CERTAINTY_THRESHOLD and free_sum > CERTAINTY_THRESHOLD:
                scale = max(0.0, remaining_cap / free_sum)
                for p_idx in range(belief.n_players):
                    if not pin_mask[p_idx, c_idx]:
                        belief.probs[p_idx, c_idx] = float(
                            np.clip(belief.probs[p_idx, c_idx] * scale, 0.0, 1.0)
                        )

    def _row_pass(
        self,
        belief: BeliefState,
        pin_mask: np.ndarray,
        prior_ceiling: np.ndarray,
    ) -> None:
        """
        Enforce per-player consistency: clip free cells to [0, prior_ceiling].

        This prevents the column pass from driving probabilities below 0 or
        above the maximum possible given the pool, and ensures probabilities
        don't accumulate floating-point drift.
        """
        for p_idx in range(belief.n_players):
            inf = belief.influence_counts[p_idx]
            if inf == 0:
                belief.probs[p_idx, :] = 0.0
                continue

            for c_idx in range(N_CARDS):
                if pin_mask[p_idx, c_idx]:
                    continue
                ceiling = prior_ceiling[p_idx, c_idx]
                belief.probs[p_idx, c_idx] = float(
                    np.clip(belief.probs[p_idx, c_idx], 0.0, ceiling)
                )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def normalise(
    belief: BeliefState,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    convergence_tol: float = DEFAULT_CONVERGENCE_TOL,
) -> int:
    """
    Convenience wrapper: normalise a BeliefState in place.
    Returns the number of iterations run.
    """
    return ConstraintNormaliser(
        max_iterations=max_iterations,
        convergence_tol=convergence_tol,
    ).normalise(belief)
