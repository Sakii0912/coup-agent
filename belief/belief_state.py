"""
belief/belief_state.py — Prior distribution over hidden cards (Sub-goal 3.1).

This module answers: "Given everything I can observe, how likely is each
opponent to hold each card?"

The answer lives in a (n_players, N_CARDS) probability matrix called `probs`.
  probs[i, c] = P(player i holds at least one copy of card c in their hidden hand)

Sub-goal 3.1 covers only initialisation — computing the prior from:
  - The fixed deck composition (3× each of 5 cards = 15 total)
  - The observer's known hand (their own cards are certain)
  - All publicly revealed cards (face-up influence losses)
  - Per-player influence counts (how many hidden cards each player has left)

Sub-goals 3.2–3.4 will add update rules (Bayesian updates on events) and
bluff modelling. Sub-goal 3.5 wires this into Observation.

─────────────────────────────────────────────────────────────────────────────
Mathematical foundation
─────────────────────────────────────────────────────────────────────────────

The deck starts with exactly 3 copies of each card. At any point:

    available[c] = 3 - known_count[c]

where known_count[c] = (copies in my hand) + (copies revealed face-up).

For each opponent, we model their hidden hand as a draw without replacement
from the available pool. The probability that an opponent with `k` hidden
cards holds at least one copy of card c is computed using the
hypergeometric distribution (drawing k cards from a pool containing
`available[c]` hits and `pool_size - available[c]` misses):

    P(≥1 copy of c | k hidden cards) = 1 - C(pool_size - available[c], k)
                                            ──────────────────────────────
                                                    C(pool_size, k)

where pool_size = total unaccounted cards = 15 - known_total.

This gives the correct marginal probability per card per opponent without
enumerating all possible deals — which would be combinatorially explosive
for 6 players.

─────────────────────────────────────────────────────────────────────────────
Key design decisions
─────────────────────────────────────────────────────────────────────────────

1. The observer's own hand is encoded as certainty (probability 1.0 or 0.0)
   for themselves, not included in the opponent distribution. This keeps the
   matrix uniform in shape but semantically clear.

2. Dead players (influence_count == 0) get a zero row — they hold nothing.

3. The pool_size accounts for all cards not yet revealed and not in the
   observer's own hand. Cards in the deck and opponents' hidden hands are
   all drawn from this same unknown pool.

4. When my_hand is None (e.g. for a spectator or when the observer is
   reconstructing from a log without knowing their own hand), we treat the
   observer's cards as part of the unknown pool.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from coup.cards import Card, DECK_COMPOSITION

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CARDS: List[Card] = list(Card)                         # canonical ordering
N_CARDS: int = len(CARDS)                              # 5
CARD_TO_IDX: Dict[Card, int] = {c: i for i, c in enumerate(CARDS)}
TOTAL_CARDS: int = len(DECK_COMPOSITION)               # 15
COPIES_PER_CARD: int = TOTAL_CARDS // N_CARDS          # 3
CERTAINTY_THRESHOLD: float = 1e-9                      # values within this of 0 or 1 are treated as pinned


# ---------------------------------------------------------------------------
# Combinatorics helper
# ---------------------------------------------------------------------------

def _log_comb(n: int, k: int) -> float:
    """log C(n, k) using gammaln for numerical stability. Returns -inf if undefined."""
    if k < 0 or k > n:
        return float("-inf")
    if k == 0 or k == n:
        return 0.0
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
    )


def hypergeometric_at_least_one(
    pool_size: int,
    n_hits: int,
    draw_size: int,
) -> float:
    """
    P(at least 1 success in `draw_size` draws from a pool of `pool_size`
    items containing `n_hits` successes) — hypergeometric distribution.

    Returns 0.0 if pool_size == 0 or draw_size == 0.
    Returns 1.0 if n_hits >= draw_size (guaranteed at least one).
    """
    if pool_size <= 0 or draw_size <= 0:
        return 0.0
    if n_hits <= 0:
        return 0.0
    if n_hits >= pool_size:
        return 1.0
    if draw_size >= pool_size:
        # Drawing the whole pool — exactly n_hits successes guaranteed
        return 1.0 if n_hits > 0 else 0.0

    # P(0 successes) = C(pool_size - n_hits, draw_size) / C(pool_size, draw_size)
    log_p_zero = _log_comb(pool_size - n_hits, draw_size) - _log_comb(pool_size, draw_size)
    p_zero = math.exp(log_p_zero)
    return max(0.0, min(1.0, 1.0 - p_zero))


# ---------------------------------------------------------------------------
# BeliefState
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    """
    Probability distribution over hidden cards for all players.

    Attributes:
        n_players:         Total number of players in the game.
        observer_idx:      Which player owns this belief state. -1 for spectator.
        probs:             Float64 numpy array of shape (n_players, N_CARDS).
                           probs[i, c] = P(player i holds ≥1 copy of card c).
                           Own row (observer_idx) encodes certainty from known hand.
        influence_counts:  Number of hidden cards each player currently holds.
        revealed_cards:    All publicly revealed (face-up) cards per player.
        known_hand:        Observer's own known cards (None for spectator view).
        available:         Per-card count of copies not yet accounted for.
                           Derived from deck - known_hand - revealed. Used for
                           constraint propagation in sub-goal 3.3.
    """

    n_players: int
    observer_idx: int
    probs: np.ndarray                      # shape (n_players, N_CARDS), float64
    influence_counts: List[int]            # length n_players
    revealed_cards: List[List[Card]]       # [player_idx][card_list]
    known_hand: Optional[List[Card]]       # observer's own hand (None = spectator)
    available: np.ndarray                  # shape (N_CARDS,) — remaining pool counts

    # ------------------------------------------------------------------ #
    #  Factory — the main constructor                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def initialise(
        cls,
        n_players: int,
        influence_counts: List[int],
        revealed_cards: List[List[Card]],
        observer_idx: int = -1,
        known_hand: Optional[List[Card]] = None,
    ) -> "BeliefState":
        """
        Compute the prior BeliefState from publicly observable information.

        Args:
            n_players:        Total players (2–6).
            influence_counts: List of length n_players. influence_counts[i]
                              is the number of hidden cards player i currently
                              holds (0 = dead, 1 = one card left, 2 = full hand).
            revealed_cards:   List of length n_players. revealed_cards[i] is
                              the list of face-up (eliminated) cards for player i.
            observer_idx:     Index of the observing player, or -1 for spectator.
            known_hand:       The observer's own hidden cards (if known). Treated
                              as certain information. Must be consistent with
                              influence_counts[observer_idx] if provided.

        Returns:
            BeliefState with probs, available, and all metadata set.

        Raises:
            ValueError: if arguments are inconsistent with deck composition.
        """
        assert 2 <= n_players <= 6, f"n_players must be 2–6, got {n_players}"
        assert len(influence_counts) == n_players
        assert len(revealed_cards) == n_players

        if known_hand is not None and observer_idx >= 0:
            expected_inf = influence_counts[observer_idx]
            if len(known_hand) != expected_inf:
                raise ValueError(
                    f"known_hand has {len(known_hand)} cards but "
                    f"influence_counts[{observer_idx}] = {expected_inf}"
                )

        # ── Step 1: count all known cards ──────────────────────────────────
        # known = observer's hand + all revealed cards
        known_counts = np.zeros(N_CARDS, dtype=np.float64)

        if known_hand:
            for card in known_hand:
                known_counts[CARD_TO_IDX[card]] += 1.0

        for player_revealed in revealed_cards:
            for card in player_revealed:
                known_counts[CARD_TO_IDX[card]] += 1.0

        # Validate: no card can exceed 3 copies total
        for c_idx, count in enumerate(known_counts):
            if count > COPIES_PER_CARD:
                raise ValueError(
                    f"Card {CARDS[c_idx].value} appears {int(count)} times in "
                    f"known cards, but only {COPIES_PER_CARD} copies exist."
                )

        # ── Step 2: compute available pool ────────────────────────────────
        # available[c] = copies of card c that are in an unknown location
        # (i.e., in the deck or in an opponent's hidden hand)
        available = np.array(
            [COPIES_PER_CARD - known_counts[c] for c in range(N_CARDS)],
            dtype=np.float64,
        )

        # Total unknown cards = total - all known
        total_known = int(known_counts.sum())
        pool_size = TOTAL_CARDS - total_known

        # ── Step 3: build probs matrix ────────────────────────────────────
        probs = np.zeros((n_players, N_CARDS), dtype=np.float64)

        for p_idx in range(n_players):
            inf_count = influence_counts[p_idx]

            if inf_count == 0:
                # Dead player — holds nothing
                probs[p_idx, :] = 0.0
                continue

            if p_idx == observer_idx and known_hand is not None:
                # Own hand — encode with certainty
                probs[p_idx, :] = _encode_known_hand(known_hand)
                continue

            # Opponent with inf_count hidden cards drawn from the pool.
            # We use the hypergeometric marginal for each card independently.
            # Note: this gives the correct *marginal* per card but doesn't
            # capture joint correlations (e.g. "they have Duke AND Captain").
            # Joint modelling is deferred to sub-goal 3.3 constraint propagation.
            for c_idx in range(N_CARDS):
                n_hits = int(available[c_idx])
                probs[p_idx, c_idx] = hypergeometric_at_least_one(
                    pool_size=pool_size,
                    n_hits=n_hits,
                    draw_size=inf_count,
                )

        return cls(
            n_players=n_players,
            observer_idx=observer_idx,
            probs=probs,
            influence_counts=list(influence_counts),
            revealed_cards=[list(r) for r in revealed_cards],
            known_hand=list(known_hand) if known_hand else None,
            available=available,
        )

    # ------------------------------------------------------------------ #
    #  Convenience constructors                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_game_state(
        cls,
        game_state,                  # coup.state.GameState
        observer_idx: int,
    ) -> "BeliefState":
        """
        Build a BeliefState directly from a live GameState (god-view).
        The observer's hand is read from the GameState.

        Used in live self-play where the game engine is running.
        """
        influence_counts = [p.influence_count for p in game_state.players]
        revealed_cards   = [list(p.revealed)   for p in game_state.players]
        known_hand       = list(game_state.players[observer_idx].hand)

        return cls.initialise(
            n_players=len(game_state.players),
            influence_counts=influence_counts,
            revealed_cards=revealed_cards,
            observer_idx=observer_idx,
            known_hand=known_hand,
        )

    @classmethod
    def from_public_snapshot(
        cls,
        snapshot: List[dict],        # list of player public_view() dicts
        observer_idx: int = -1,
        known_hand: Optional[List[Card]] = None,
    ) -> "BeliefState":
        """
        Build a BeliefState from the public state snapshots stored in game logs.
        Used when replaying logs (Phase 2 feature extraction) or in the live
        interface (Phase 5) where the user inputs the observable game state.

        Args:
            snapshot:     List of player dicts with keys: influence_count,
                          revealed_cards (list of card name strings), is_alive.
            observer_idx: Index of the observing player (-1 = spectator).
            known_hand:   Observer's own cards as Card enum values.
        """
        n_players = len(snapshot)
        influence_counts = [p["influence_count"] for p in snapshot]
        revealed_cards = [
            [Card(c) for c in p.get("revealed_cards", [])]
            for p in snapshot
        ]
        return cls.initialise(
            n_players=n_players,
            influence_counts=influence_counts,
            revealed_cards=revealed_cards,
            observer_idx=observer_idx,
            known_hand=known_hand,
        )

    # ------------------------------------------------------------------ #
    #  Accessors                                                          #
    # ------------------------------------------------------------------ #

    def prob(self, player_idx: int, card: Card) -> float:
        """P(player_idx holds ≥1 copy of card)."""
        return float(self.probs[player_idx, CARD_TO_IDX[card]])

    def most_likely_card(self, player_idx: int) -> Optional[Card]:
        """The card most likely to be in player_idx's hand. None if dead."""
        if self.influence_counts[player_idx] == 0:
            return None
        c_idx = int(np.argmax(self.probs[player_idx]))
        return CARDS[c_idx]

    def entropy(self, player_idx: int) -> float:
        """
        Shannon entropy (bits) of the card distribution for player_idx.
        High entropy = high uncertainty. Zero for dead players or known hands.
        Calculated per-card as independent Bernoulli variables.
        """
        p = self.probs[player_idx]
        if np.all(p == 0.0) or np.all((p == 0.0) | (p == 1.0)):
            return 0.0
        # H = -sum(p * log2(p) + (1-p) * log2(1-p)) for each card
        eps = 1e-12
        p_safe = np.clip(p, eps, 1 - eps)
        h = -np.sum(p_safe * np.log2(p_safe) + (1 - p_safe) * np.log2(1 - p_safe))
        return float(max(0.0, h))

    def pool_size(self) -> int:
        """Total cards in the unknown pool (deck + all hidden hands)."""
        known = 0
        if self.known_hand:
            known += len(self.known_hand)
        for rev in self.revealed_cards:
            known += len(rev)
        return TOTAL_CARDS - known

    def opponent_probs(self, exclude_idx: int) -> np.ndarray:
        """
        Returns probs matrix with observer's own row zeroed out.
        Useful when the observer only wants to reason about opponents.
        """
        mask = self.probs.copy()
        mask[exclude_idx, :] = 0.0
        return mask

    # ------------------------------------------------------------------ #
    #  Serialisation                                                      #
    # ------------------------------------------------------------------ #

    def to_vector(self) -> np.ndarray:
        """
        Flatten probs to a 1D float32 vector of length n_players × N_CARDS.
        Used as input to the feature extractor (sub-goal 3.5).
        """
        return self.probs.flatten().astype(np.float32)

    def to_dict(self) -> dict:
        """JSON-serialisable representation for logging and debugging."""
        return {
            "n_players": self.n_players,
            "observer_idx": self.observer_idx,
            "probs": self.probs.tolist(),
            "influence_counts": self.influence_counts,
            "revealed_cards": [
                [c.value for c in rev] for rev in self.revealed_cards
            ],
            "known_hand": (
                [c.value for c in self.known_hand] if self.known_hand else None
            ),
            "available": self.available.tolist(),
        }

    def copy(self) -> "BeliefState":
        """Return a deep copy — used before applying updates."""
        return BeliefState(
            n_players=self.n_players,
            observer_idx=self.observer_idx,
            probs=self.probs.copy(),
            influence_counts=list(self.influence_counts),
            revealed_cards=[list(r) for r in self.revealed_cards],
            known_hand=list(self.known_hand) if self.known_hand else None,
            available=self.available.copy(),
        )

    # ------------------------------------------------------------------ #
    #  Display                                                            #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        rows = []
        for p_idx in range(self.n_players):
            inf = self.influence_counts[p_idx]
            label = f"P{p_idx}"
            if p_idx == self.observer_idx:
                label += "(me)"
            card_strs = "  ".join(
                f"{CARDS[c].value[:3]}:{self.probs[p_idx,c]:.2f}"
                for c in range(N_CARDS)
            )
            rows.append(f"  {label:<10} inf={inf}  [{card_strs}]")
        return "BeliefState(\n" + "\n".join(rows) + "\n)"

    def summary_table(self) -> str:
        """Pretty-print the full probability table."""
        header = f"{'Player':<12}" + "".join(f"{c.value:>12}" for c in CARDS)
        lines = [header, "-" * (12 + 12 * N_CARDS)]
        for p_idx in range(self.n_players):
            label = f"P{p_idx}"
            if p_idx == self.observer_idx:
                label += " (me)"
            row = f"{label:<12}"
            for c_idx in range(N_CARDS):
                p = self.probs[p_idx, c_idx]
                row += f"{p:>12.3f}"
            lines.append(row)
        lines.append("-" * (12 + 12 * N_CARDS))
        lines.append(
            f"{'available':<12}"
            + "".join(f"{int(self.available[c]):>12}" for c in range(N_CARDS))
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_known_hand(hand: List[Card]) -> np.ndarray:
    """
    Encode a fully known hand as a certainty probability vector.
    If a card is in the hand: P = 1.0. If not: P = 0.0.
    Note: holding two copies of the same card still gives P = 1.0
    (we model "holds ≥1 copy", not "holds exactly k copies").
    """
    vec = np.zeros(N_CARDS, dtype=np.float64)
    for card in hand:
        vec[CARD_TO_IDX[card]] = 1.0
    return vec
