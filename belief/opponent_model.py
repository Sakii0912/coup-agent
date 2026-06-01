"""
belief/opponent_model.py — Per-player bluff rate tracking (sub-goal 3.4).

─────────────────────────────────────────────────────────────────────────────
What this solves
─────────────────────────────────────────────────────────────────────────────

Sub-goal 3.2 uses a flat CLAIM_CREDIBILITY constant (1.5) for every player
when updating beliefs on a card claim. This is a weak approximation:

  - A player who has been caught bluffing twice should have their claims
    treated as nearly worthless — maybe 0.1× credibility.
  - A player who has never been caught bluffing, and has proved their card
    twice, should have their claims treated as strong evidence — maybe 2.5×.

The OpponentModel tracks three statistics per player:

    claims_made:     total character action claims observed
    bluffs_caught:   times challenger won (actor was bluffing)
    honest_proved:   times actor won (held the card, proved it)

From these, it derives a per-player credibility multiplier that replaces
the flat CLAIM_CREDIBILITY when BeliefUpdater processes claims.

─────────────────────────────────────────────────────────────────────────────
Bayesian bluff rate estimation
─────────────────────────────────────────────────────────────────────────────

We model each player's bluff tendency as a Beta distribution:

    bluff_rate_i ~ Beta(α_i, β_i)

Prior: Beta(α₀, β₀) where α₀ = β₀ = PRIOR_STRENGTH (default 2.0).
This encodes prior belief that players bluff ~50% of the time, with moderate
uncertainty. The prior strength controls how quickly observations dominate:
higher = slower learning, lower = faster but noisier.

Posterior update after each challenge result:
  - Bluff caught (challenger won): α_i += 1  (bluff evidence)
  - Honest proved (actor won):     β_i += 1  (honesty evidence)

Posterior mean bluff rate:
    bluff_rate_i = α_i / (α_i + β_i)

─────────────────────────────────────────────────────────────────────────────
From bluff rate to credibility multiplier
─────────────────────────────────────────────────────────────────────────────

A player's credibility for a claim is inversely related to their bluff rate:

    credibility(p) = (1 - bluff_rate(p)) / bluff_rate(p)
                   = β_p / α_p   (posterior mean)

Interpretation:
  bluff_rate = 0.1  →  credibility ≈ 9.0  (very trustworthy)
  bluff_rate = 0.5  →  credibility = 1.0  (neutral / prior)
  bluff_rate = 0.9  →  credibility ≈ 0.11 (nearly ignore their claims)

The multiplier is clamped to [MIN_CREDIBILITY, MAX_CREDIBILITY] to prevent
extreme values from dominating the belief update (a single data point
should not set credibility to 0 or infinity).

─────────────────────────────────────────────────────────────────────────────
Integration with BeliefUpdater
─────────────────────────────────────────────────────────────────────────────

BeliefUpdater accepts an optional `opponent_model` parameter. When provided,
it replaces the flat CLAIM_CREDIBILITY with per-player values on every claim.

    model = OpponentModel(n_players=4)
    updater = BeliefUpdater(n_players=4, opponent_model=model)

    # model updates automatically when challenge_result events are processed
    updater.update(belief, "challenge_result", actor_idx=1, ...)
    updater.update(belief, "action", actor_idx=1, claimed_card=Card.DUKE)
    # ↑ now uses model.credibility(1) instead of flat CLAIM_CREDIBILITY
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from coup.cards import Card


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIOR_STRENGTH:    float = 2.0    # α₀ = β₀: prior counts for bluff/honest
MIN_CREDIBILITY:   float = 0.1    # floor: never fully ignore a claim
MAX_CREDIBILITY:   float = 5.0    # ceiling: never fully trust a claim
DEFAULT_CREDIBILITY: float = 1.5  # used when no observations available


# ---------------------------------------------------------------------------
# PlayerBluffStats
# ---------------------------------------------------------------------------

@dataclass
class PlayerBluffStats:
    """
    Running statistics for one player's bluffing behaviour.

    All counts are floats to allow fractional pseudo-counts from the prior.
    """

    player_idx: int
    alpha: float = PRIOR_STRENGTH   # bluff evidence  (higher = more bluffing)
    beta:  float = PRIOR_STRENGTH   # honest evidence (higher = more honest)

    # Raw observation counts (integer increments, separate from prior)
    claims_total:   int = 0
    bluffs_caught:  int = 0
    honest_proved:  int = 0

    # ── Derived ─────────────────────────────────────────────────────────

    @property
    def bluff_rate(self) -> float:
        """Posterior mean bluff rate: E[bluff_rate] = α / (α + β)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def credibility(self) -> float:
        """
        Credibility multiplier for this player's claims.
        = β / α  (posterior honesty-to-bluff ratio), clamped to valid range.
        """
        raw = self.beta / self.alpha
        return float(np.clip(raw, MIN_CREDIBILITY, MAX_CREDIBILITY))

    @property
    def confidence(self) -> float:
        """
        How confident we are in our estimate of this player's bluff rate.
        = total observations / (total observations + 2 × prior_strength)
        Ranges from 0 (no data) to 1 (many observations).
        """
        total_obs = self.bluffs_caught + self.honest_proved
        return total_obs / (total_obs + 2 * PRIOR_STRENGTH)

    # ── Mutation ────────────────────────────────────────────────────────

    def record_bluff(self) -> None:
        """Called when this player was caught bluffing (challenger won)."""
        self.alpha        += 1.0
        self.bluffs_caught += 1
        self.claims_total  += 1

    def record_honest(self) -> None:
        """Called when this player proved their claim (actor won challenge)."""
        self.beta          += 1.0
        self.honest_proved += 1
        self.claims_total  += 1

    def record_unchallenged_claim(self) -> None:
        """
        Called when a claim is not challenged. The claim was not verified,
        so we record it as a claim but don't update α or β.
        (Weak evidence — the option to challenge was available but unused.)
        """
        self.claims_total += 1

    # ── Serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "player_idx":    self.player_idx,
            "alpha":         self.alpha,
            "beta":          self.beta,
            "bluff_rate":    round(self.bluff_rate, 4),
            "credibility":   round(self.credibility, 4),
            "confidence":    round(self.confidence, 4),
            "claims_total":  self.claims_total,
            "bluffs_caught": self.bluffs_caught,
            "honest_proved": self.honest_proved,
        }

    def __repr__(self) -> str:
        return (
            f"PlayerBluffStats(P{self.player_idx}: "
            f"bluff_rate={self.bluff_rate:.2f}, "
            f"cred={self.credibility:.2f}, "
            f"n={self.claims_total})"
        )


# ---------------------------------------------------------------------------
# OpponentModel
# ---------------------------------------------------------------------------

class OpponentModel:
    """
    Tracks per-player bluff statistics and provides credibility scores for
    use in BeliefUpdater's claim update rule.

    Usage:
        model = OpponentModel(n_players=4, observer_idx=0)
        model.record_challenge_result(actor_idx=1, actor_won=False)
        cred = model.credibility(1)   # low: player 1 was just caught bluffing

    The observer's own index is stored but never updated — we don't track
    our own bluff rate (we know our own hand).
    """

    def __init__(
        self,
        n_players: int,
        observer_idx: int = -1,
        prior_strength: float = PRIOR_STRENGTH,
    ) -> None:
        self.n_players    = n_players
        self.observer_idx = observer_idx
        self._stats: List[PlayerBluffStats] = [
            PlayerBluffStats(
                player_idx=i,
                alpha=prior_strength,
                beta=prior_strength,
            )
            for i in range(n_players)
        ]

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def credibility(self, player_idx: int) -> float:
        """
        Credibility multiplier for player_idx's claims.
        Higher = more trustworthy. Range: [MIN_CREDIBILITY, MAX_CREDIBILITY].
        Returns DEFAULT_CREDIBILITY for the observer themselves.
        """
        if player_idx == self.observer_idx:
            return DEFAULT_CREDIBILITY
        return self._stats[player_idx].credibility

    def bluff_rate(self, player_idx: int) -> float:
        """Posterior mean bluff rate for player_idx. Range [0, 1]."""
        return self._stats[player_idx].bluff_rate

    def confidence(self, player_idx: int) -> float:
        """How confident we are in player_idx's bluff rate. Range [0, 1]."""
        return self._stats[player_idx].confidence

    def stats(self, player_idx: int) -> PlayerBluffStats:
        """Return the full PlayerBluffStats for player_idx."""
        return self._stats[player_idx]

    # ------------------------------------------------------------------ #
    #  Recording events                                                   #
    # ------------------------------------------------------------------ #

    def record_challenge_result(
        self,
        actor_idx: int,
        actor_won: bool,
    ) -> None:
        """
        Update the bluff model after a challenge is resolved.

        Args:
            actor_idx: The player whose claim was challenged.
            actor_won: True = actor held the card (honest); False = bluff.
        """
        if actor_idx == self.observer_idx:
            return
        if actor_won:
            self._stats[actor_idx].record_honest()
        else:
            self._stats[actor_idx].record_bluff()

    def record_unchallenged_claim(self, actor_idx: int) -> None:
        """
        Record that player made a claim that was not challenged.
        Increments claims_total but does not update α or β.
        """
        if actor_idx == self.observer_idx:
            return
        self._stats[actor_idx].record_unchallenged_claim()

    # ------------------------------------------------------------------ #
    #  Credibility multipliers for all players                           #
    # ------------------------------------------------------------------ #

    def all_credibilities(self) -> np.ndarray:
        """
        Return credibility multipliers for all players as a float64 array
        of shape (n_players,).
        """
        return np.array(
            [self.credibility(i) for i in range(self.n_players)],
            dtype=np.float64,
        )

    def credibility_vector(self) -> np.ndarray:
        """
        Normalised credibility scores, useful as a feature for the policy
        network. Shape (n_players,), values in [0, 1].
        Normalised as: (credibility - MIN) / (MAX - MIN).
        """
        raw = self.all_credibilities()
        return (raw - MIN_CREDIBILITY) / (MAX_CREDIBILITY - MIN_CREDIBILITY)

    # ------------------------------------------------------------------ #
    #  Serialisation                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "n_players":    self.n_players,
            "observer_idx": self.observer_idx,
            "players":      [s.to_dict() for s in self._stats],
        }

    def copy(self) -> "OpponentModel":
        """Deep copy — used when snapshotting game state."""
        new = OpponentModel(
            n_players=self.n_players,
            observer_idx=self.observer_idx,
        )
        for i, s in enumerate(self._stats):
            new._stats[i] = PlayerBluffStats(
                player_idx=s.player_idx,
                alpha=s.alpha,
                beta=s.beta,
                claims_total=s.claims_total,
                bluffs_caught=s.bluffs_caught,
                honest_proved=s.honest_proved,
            )
        return new

    # ------------------------------------------------------------------ #
    #  Display                                                            #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        lines = [
            f"{'Player':<10} {'BluffRate':>10} {'Credibility':>12} "
            f"{'Confidence':>11} {'Claims':>7} {'Caught':>7} {'Proved':>7}"
        ]
        lines.append("-" * 66)
        for s in self._stats:
            marker = " ◀ me" if s.player_idx == self.observer_idx else ""
            lines.append(
                f"P{s.player_idx:<9} {s.bluff_rate:>10.3f} {s.credibility:>12.3f} "
                f"{s.confidence:>11.3f} {s.claims_total:>7} "
                f"{s.bluffs_caught:>7} {s.honest_proved:>7}{marker}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"OpponentModel(n={self.n_players}, "
            f"credibilities={[round(self.credibility(i), 2) for i in range(self.n_players)]})"
        )
