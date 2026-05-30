"""
belief/belief_updater.py — Bayesian update rules for BeliefState (sub-goal 3.2).

Every observable game event carries information about hidden cards. This module
defines one update rule per event type and exposes a single dispatch method:

    updater = BeliefUpdater(n_players)
    updater.update(belief_state, event)   # mutates belief_state in place

─────────────────────────────────────────────────────────────────────────────
Update rules (one per observable event type)
─────────────────────────────────────────────────────────────────────────────

ACTION (character claim)
  The actor claims to hold a specific card.

  Likelihood ratio update: if a player claims card C, we scale their
  probability for C upward and scale all other players' probabilities
  for C downward (because the claim, even if possibly a bluff, is weak
  evidence of who holds the card).

  Scale factor controlled by CLAIM_CREDIBILITY (default 1.5):
    posterior ∝ prior × credibility   (for the actor)
    posterior ∝ prior × (1/credibility)  (for all others)

  After scaling, renormalise each row independently to [0, 1] via the
  available pool constraint. Full global constraint propagation happens
  in sub-goal 3.3 (normalise()); this update only applies the local
  Bayesian signal.

CHALLENGE_RESULT — actor wins (held the card)
  The actor proved they held the claimed card → set their probability
  for that card to 1.0 (certainty). Also, since the actor swaps the
  proved card for a fresh draw from the deck, we immediately reset their
  row: the proved card's probability returns to the prior (they may or may
  not draw it again), and all other cards get their prior probabilities too.
  The net effect: the actor's row is recomputed from scratch, conditioned
  on having 1 hidden card (after losing influence is handled separately).

CHALLENGE_RESULT — challenger wins (actor was bluffing)
  The actor did NOT hold the claimed card → set their probability for
  that card to 0.0 (certainty of absence). The actor also loses an
  influence card (handled by INFLUENCE_LOSS), so we just record the
  absence here.

INFLUENCE_LOSS
  A specific card is revealed face-up for a specific player.
  - Move the card from that player's hidden count to their revealed list.
  - Decrement their influence_count.
  - Remove the card from available[].
  - Recompute the player's row using the hypergeometric prior on their
    remaining hidden cards (if any).
  - If influence_count reaches 0, zero the row.

BLOCK (block claim)
  Identical in structure to an ACTION claim: the blocker claims to hold
  a specific card. Apply the same likelihood ratio update as ACTION.

EXCHANGE
  The actor draws cards from the deck and returns some back. Their hand
  composition changes without revealing anything. This increases our
  uncertainty about the actor's hand, so we reset their row to the
  hypergeometric prior for their current influence_count (maximum
  uncertainty given pool constraints).

─────────────────────────────────────────────────────────────────────────────
Design decisions
─────────────────────────────────────────────────────────────────────────────

1.  All updates mutate the BeliefState in place. The caller should call
    belief_state.copy() before update() if they need the old state.

2.  The CLAIM_CREDIBILITY constant controls how strongly a claim shifts
    beliefs. Setting it to 1.0 means claims are ignored. The default 1.5
    is conservative — in a game with ~60% bluff rate, claims are only
    mild evidence. Sub-goal 3.4 (bluff modelling) will replace this flat
    constant with a per-player learned bluff rate.

3.  After every update, _recompute_row() is called for affected players
    to re-anchor probabilities to the hypergeometric prior, then the
    Bayesian signal is applied on top. This prevents probabilities from
    drifting outside [0, 1] through accumulated floating-point errors.

4.  GAME_START, INCOME, FOREIGN_AID, COUP, TAX, STEAL, ASSASSINATE (no
    claim variants), ACTION_RESOLVED: these events carry no card information
    beyond what is already encoded in the state snapshot, so they are
    treated as no-ops here. The state snapshot itself (influence counts,
    coins, revealed cards) is consumed at initialisation time.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from coup.cards import Card
from .belief_state import (
    BeliefState,
    CARDS,
    N_CARDS,
    CARD_TO_IDX,
    COPIES_PER_CARD,
    hypergeometric_at_least_one,
    _encode_known_hand,
)


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# How much a claim shifts beliefs toward the actor.
# 1.0 = no shift. 2.0 = claim doubles actor's probability for that card.
# Sub-goal 3.4 will override this with a per-player learned value.
CLAIM_CREDIBILITY: float = 1.5


# ---------------------------------------------------------------------------
# BeliefUpdater
# ---------------------------------------------------------------------------

class BeliefUpdater:
    """
    Applies Bayesian update rules to a BeliefState for each observable event.

    Usage:
        updater = BeliefUpdater(n_players=4)
        updater.update(belief, event_type, **kwargs)

    All update methods mutate `belief` in place. Call `belief.copy()` first
    if you need to preserve the previous state.
    """

    def __init__(self, n_players: int) -> None:
        self.n_players = n_players

    # ------------------------------------------------------------------ #
    #  Main dispatch                                                      #
    # ------------------------------------------------------------------ #

    def update(
        self,
        belief: BeliefState,
        event_type: str,
        *,
        actor_idx: Optional[int] = None,
        claimed_card: Optional[Card] = None,
        challenger_idx: Optional[int] = None,
        actor_won: Optional[bool] = None,
        player_idx: Optional[int] = None,
        card_lost: Optional[Card] = None,
        blocker_idx: Optional[int] = None,
        blocking_card: Optional[Card] = None,
    ) -> None:
        """
        Dispatch an event to the appropriate update rule.

        Args:
            belief:         BeliefState to update in place.
            event_type:     String matching the logger's event_type values.
            actor_idx:      Player who declared the action / claimed card.
            claimed_card:   Card claimed in the action.
            challenger_idx: Player who challenged (for challenge_result).
            actor_won:      True if actor/blocker won the challenge.
            player_idx:     Player losing influence (for influence_loss).
            card_lost:      Card revealed during influence_loss.
            blocker_idx:    Player blocking (for block events).
            blocking_card:  Card claimed by blocker.
        """
        if event_type == "action" and claimed_card is not None and actor_idx is not None:
            self._update_on_claim(belief, actor_idx, claimed_card)

        elif event_type == "challenge_result":
            self._update_on_challenge_result(
                belief, actor_idx, claimed_card, challenger_idx, actor_won
            )

        elif event_type == "influence_loss":
            if player_idx is not None and card_lost is not None:
                self._update_on_influence_loss(belief, player_idx, card_lost)

        elif event_type == "block" and blocker_idx is not None and blocking_card is not None:
            self._update_on_claim(belief, blocker_idx, blocking_card)

        elif event_type == "exchange" and actor_idx is not None:
            self._update_on_exchange(belief, actor_idx)

        # All other event types carry no new card information — no-op.

    # ------------------------------------------------------------------ #
    #  Update rules                                                       #
    # ------------------------------------------------------------------ #

    def _update_on_claim(
        self,
        belief: BeliefState,
        actor_idx: int,
        claimed_card: Card,
    ) -> None:
        """
        Likelihood ratio update when a player claims to hold a card.

        The claim is weak positive evidence for the actor holding the card,
        and weak negative evidence for all other players (zero-sum card pool).

        After scaling, each affected player's probability is re-anchored to
        [0, 1] using the hypergeometric prior as a ceiling.
        """
        c_idx = CARD_TO_IDX[claimed_card]

        # Skip if actor is the observer (own hand is already certain)
        if actor_idx == belief.observer_idx and belief.known_hand is not None:
            return

        for p_idx in range(self.n_players):
            inf = belief.influence_counts[p_idx]
            if inf == 0:
                continue
            if p_idx == belief.observer_idx and belief.known_hand is not None:
                continue  # Own hand — never update from claims

            current = belief.probs[p_idx, c_idx]

            if p_idx == actor_idx:
                # Repeated claims should keep pushing the actor upward.
                updated = min(current * CLAIM_CREDIBILITY, 1.0)
            else:
                # Scale down — card less likely to be elsewhere
                # Floor at 0 but don't go below a small epsilon
                updated = max(current / CLAIM_CREDIBILITY, 0.0)

            belief.probs[p_idx, c_idx] = float(np.clip(updated, 0.0, 1.0))

    def _update_on_challenge_result(
        self,
        belief: BeliefState,
        actor_idx: Optional[int],
        claimed_card: Optional[Card],
        challenger_idx: Optional[int],
        actor_won: Optional[bool],
    ) -> None:
        """
        Hard update on challenge resolution.

        Actor won (held the card):
          - Set actor's probability for claimed_card = 1.0.
          - Actor immediately swaps the proved card for a fresh draw, so their
            row is reset to the hypergeometric prior (maximum uncertainty again).
          - challenger_idx loses an influence (handled by influence_loss event).

        Challenger won (actor was bluffing):
          - Set actor's probability for claimed_card = 0.0 (certainty of absence).
          - Actor loses an influence (handled by influence_loss event).
        """
        if actor_idx is None or claimed_card is None or actor_won is None:
            return

        c_idx = CARD_TO_IDX[claimed_card]

        if actor_won:
            # Actor proved they held the card — temporarily confirm it
            if actor_idx != belief.observer_idx:
                belief.probs[actor_idx, c_idx] = 1.0

            # Actor swaps the card for a random draw → reset their row to
            # hypergeometric prior (the swap removes the information we just gained).
            # We do this only if actor is not the observer (observer's row stays certain).
            if actor_idx != belief.observer_idx:
                self._recompute_row_prior(belief, actor_idx)

        else:
            # Actor was bluffing — they definitely do NOT hold claimed_card
            if actor_idx != belief.observer_idx:
                belief.probs[actor_idx, c_idx] = 0.0
                # Redistribute the excluded probability mass proportionally
                # among the other cards for this player.
                self._redistribute_after_exclusion(belief, actor_idx, c_idx)

    def _update_on_influence_loss(
        self,
        belief: BeliefState,
        player_idx: int,
        card_lost: Card,
    ) -> None:
        """
        Hard update when a card is revealed face-up.

        - The revealed card moves from hidden to public.
        - available[card] decreases by 1.
        - player's influence_count decreases by 1.
        - Player's row is recomputed from the updated available pool.
        - All other players' rows are recomputed too (pool shrank by 1 card).
        """
        c_idx = CARD_TO_IDX[card_lost]

        # Record the reveal
        belief.revealed_cards[player_idx].append(card_lost)
        belief.influence_counts[player_idx] = max(
            0, belief.influence_counts[player_idx] - 1
        )
        belief.available[c_idx] = max(0.0, belief.available[c_idx] - 1.0)

        # Recompute all rows — the pool shrank, affecting everyone's distribution
        for p_idx in range(self.n_players):
            if p_idx == belief.observer_idx and belief.known_hand is not None:
                # Observer's own hand doesn't change from opponent reveals
                belief.probs[p_idx] = _encode_known_hand(belief.known_hand)
                continue
            self._recompute_row_prior(belief, p_idx)

    def _update_on_exchange(
        self,
        belief: BeliefState,
        actor_idx: int,
    ) -> None:
        """
        Update when a player performs an Ambassador Exchange.

        The actor draws cards from the deck and returns different cards.
        Their hand composition changes, but nothing is revealed. This is a
        maximum-uncertainty event: we know their influence_count stays the
        same but we lose all information we had about their hand.

        Response: reset the actor's row to the hypergeometric prior
        (maximum uncertainty given the current pool).
        """
        if actor_idx == belief.observer_idx and belief.known_hand is not None:
            # Observer's own exchange — they know their new hand exactly.
            # The caller (game engine or log replayer) must update known_hand
            # separately; we just leave the certainty encoding intact.
            return

        self._recompute_row_prior(belief, actor_idx)

    # ------------------------------------------------------------------ #
    #  Row recomputation helpers                                          #
    # ------------------------------------------------------------------ #

    def _recompute_row_prior(self, belief: BeliefState, p_idx: int) -> None:
        """
        Recompute player p_idx's probability row from scratch using the
        hypergeometric prior given current influence_count and available pool.
        Ignores any previously accumulated Bayesian signal for this player.
        """
        inf = belief.influence_counts[p_idx]
        if inf == 0:
            belief.probs[p_idx, :] = 0.0
            return

        pool = _current_pool_size(belief)
        for c_idx in range(N_CARDS):
            n_hits = int(belief.available[c_idx])
            belief.probs[p_idx, c_idx] = hypergeometric_at_least_one(
                pool_size=pool,
                n_hits=n_hits,
                draw_size=inf,
            )

    def _redistribute_after_exclusion(
        self,
        belief: BeliefState,
        p_idx: int,
        excluded_c_idx: int,
    ) -> None:
        """
        After setting probs[p_idx, excluded_c_idx] = 0.0, rescale the
        remaining card probabilities without resurrecting cards that were
        already ruled out by earlier bluff detections.
        """
        row = belief.probs[p_idx].copy()
        row[excluded_c_idx] = 0.0

        remaining_mask = row > 0.0
        total_after = row[remaining_mask].sum()
        if total_after < 1e-9:
            # All cards excluded — shouldn't happen, reset to flat prior
            self._recompute_row_prior(belief, p_idx)
            belief.probs[p_idx, excluded_c_idx] = 0.0
            return

        # Scale only the surviving positive entries, preserving earlier zeros.
        scale = belief.probs[p_idx].sum() / total_after
        row[remaining_mask] = np.clip(row[remaining_mask] * scale, 0.0, 1.0)

        row[excluded_c_idx] = 0.0
        belief.probs[p_idx] = row


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _hypergeometric_prior(belief: BeliefState, p_idx: int, c_idx: int) -> float:
    """Compute the hypergeometric prior for (p_idx, c_idx) from current state."""
    inf = belief.influence_counts[p_idx]
    if inf == 0:
        return 0.0
    pool = _current_pool_size(belief)
    return hypergeometric_at_least_one(
        pool_size=pool,
        n_hits=int(belief.available[c_idx]),
        draw_size=inf,
    )


def _current_pool_size(belief: BeliefState) -> int:
    """
    Total number of cards in the unknown pool (deck + all hidden hands).
    = TOTAL_CARDS - known_hand_size - total_revealed_count
    """
    known = len(belief.known_hand) if belief.known_hand else 0
    revealed = sum(len(r) for r in belief.revealed_cards)
    from .belief_state import TOTAL_CARDS
    return max(0, TOTAL_CARDS - known - revealed)
