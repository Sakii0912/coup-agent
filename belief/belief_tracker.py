
"""
belief/belief_tracker.py — High-level orchestrator for Phase 3 (sub-goal 3.6).

BeliefTracker is the single object that agents and the live interface
(Phase 5) interact with. It owns and coordinates:

  - BeliefState        : probability matrix over hidden cards
  - BeliefUpdater      : Bayesian update rules, auto-normalisation
  - OpponentModel      : per-player bluff rate tracking
  - ConstraintNormaliser: (used internally by BeliefUpdater)

Public interface:

    tracker = BeliefTracker.from_game_start(n_players=4, observer_idx=0,
                                             known_hand=[Card.DUKE, Card.ASSASSIN])
    tracker.process_event(event)       # ParsedEvent from log replay
    tracker.process_raw(event_type, **kwargs)  # direct from game engine

    obs = tracker.enrich_observation(obs)  # attach belief/model to Observation
    feat = tracker.feature_vector(config)  # flat numpy array for policy net

    # Snapshots for MCTS rollouts / counterfactual reasoning
    snapshot = tracker.snapshot()
    tracker.restore(snapshot)

─────────────────────────────────────────────────────────────────────────────
Event routing
─────────────────────────────────────────────────────────────────────────────

ParsedEvent fields map to BeliefUpdater kwargs as follows:

  event.type == "action"
    actor_idx    = event.actor_idx
    claimed_card = Card(event.claimed_card)  if event.claimed_card else None

  event.type == "challenge_result"
    actor_idx      = event.loser_idx if challenger_won else event.winner_idx
    claimed_card   = inferred from preceding action/block event
    challenger_idx = event.winner_idx if challenger_won else event.loser_idx
    actor_won      = (event.winner_idx != challenger_idx)

  event.type == "block"
    blocker_idx  = event.blocker_idx
    blocking_card = Card(event.blocking_card)

  event.type == "influence_loss"
    player_idx = event.player_idx
    card_lost  = Card(event.card_lost)

  event.type == "exchange"
    actor_idx = event.actor_idx  (from preceding action_resolved)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from coup.cards import Card
from .belief_state import BeliefState, CARDS, N_CARDS, CARD_TO_IDX, COPIES_PER_CARD
from .belief_updater import BeliefUpdater
from .constraint_normaliser import ConstraintNormaliser
from .opponent_model import OpponentModel


# ---------------------------------------------------------------------------
# Snapshot (for MCTS / counterfactual use)
# ---------------------------------------------------------------------------

@dataclass
class BeliefSnapshot:
    """Immutable snapshot of tracker state at a point in the game."""
    belief: BeliefState
    model:  OpponentModel
    _pending_claimed_card: Optional[Card]
    _pending_actor_idx:    Optional[int]


# ---------------------------------------------------------------------------
# BeliefTracker
# ---------------------------------------------------------------------------

class BeliefTracker:
    """
    Orchestrates BeliefState + BeliefUpdater + OpponentModel for one game.

    Create via a factory, then feed events in order.
    """

    # ------------------------------------------------------------------ #
    #  Factories                                                          #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        belief:   BeliefState,
        updater:  BeliefUpdater,
        model:    OpponentModel,
    ) -> None:
        self._belief  = belief
        self._updater = updater
        self._model   = model

        # Context carried across events to reconstruct challenge results
        self._pending_claimed_card: Optional[Card] = None
        self._pending_actor_idx:    Optional[int]  = None

    @classmethod
    def from_game_start(
        cls,
        n_players:      int,
        observer_idx:   int,
        known_hand:     Optional[List[Card]] = None,
        influence_counts: Optional[List[int]] = None,
        revealed_cards:   Optional[List[List[Card]]] = None,
        prior_strength: float = 2.0,
        auto_normalise: bool  = True,
    ) -> "BeliefTracker":
        """
        Create a tracker at the start of a game.

        Args:
            n_players:      Total player count (2–6).
            observer_idx:   Seat index of the observing player.
            known_hand:     Observer's starting hand (None = spectator).
            influence_counts: Starting influence per player (default: all 2).
            revealed_cards: Starting revealed cards per player (default: none).
            prior_strength: Beta prior strength for the OpponentModel.
            auto_normalise: Whether BeliefUpdater runs normalisation after
                            every event (recommended True).
        """
        if influence_counts is None:
            influence_counts = [2] * n_players
        if revealed_cards is None:
            revealed_cards = [[] for _ in range(n_players)]

        belief = BeliefState.initialise(
            n_players=n_players,
            influence_counts=influence_counts,
            revealed_cards=revealed_cards,
            observer_idx=observer_idx,
            known_hand=known_hand,
        )
        model   = OpponentModel(n_players=n_players, observer_idx=observer_idx,
                                prior_strength=prior_strength)
        updater = BeliefUpdater(n_players=n_players, auto_normalise=auto_normalise,
                                opponent_model=model)

        # Initial hypergeometric marginals are per-player independent and can
        # violate global deck conservation at larger table sizes. Start from a
        # globally consistent belief when auto-normalisation is enabled.
        if auto_normalise:
            ConstraintNormaliser().normalise(belief)

        return cls(belief=belief, updater=updater, model=model)

    @classmethod
    def from_public_snapshot(
        cls,
        snapshot: List[Dict[str, Any]],
        observer_idx: int = -1,
        known_hand: Optional[List[Card]] = None,
        **kwargs,
    ) -> "BeliefTracker":
        """
        Create a tracker from a public state snapshot (e.g. mid-game log replay).
        """
        n_players = len(snapshot)
        influence_counts = [p["influence_count"] for p in snapshot]
        revealed_cards   = [
            [Card(c) for c in p.get("revealed_cards", [])]
            for p in snapshot
        ]
        return cls.from_game_start(
            n_players=n_players,
            observer_idx=observer_idx,
            known_hand=known_hand,
            influence_counts=influence_counts,
            revealed_cards=revealed_cards,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    #  Event processing                                                   #
    # ------------------------------------------------------------------ #

    def process_event(self, event) -> None:
        """
        Process a single ParsedEvent from a game log.
        Routes to the correct BeliefUpdater kwargs.
        """
        et = event.type

        if et == "action":
            claimed = Card(event.claimed_card) if event.claimed_card else None
            # Track pending claim context for upcoming challenge_result
            self._pending_claimed_card = claimed
            self._pending_actor_idx    = event.actor_idx
            self._updater.update(
                self._belief, "action",
                actor_idx=event.actor_idx,
                claimed_card=claimed,
            )

        elif et == "block":
            blocking = Card(event.blocking_card) if event.blocking_card else None
            # Block also sets pending context (block may be challenged)
            self._pending_claimed_card = blocking
            self._pending_actor_idx    = event.blocker_idx
            self._updater.update(
                self._belief, "block",
                blocker_idx=event.blocker_idx,
                blocking_card=blocking,
            )

        elif et == "challenge_result":
            # Determine who the actor was and whether they won.
            # winner_idx = the player who won the challenge resolution.
            # If the actor (pending_actor_idx) is the winner → actor_won=True.
            actor_idx = self._pending_actor_idx
            actor_won = (event.winner_idx == actor_idx)
            self._updater.update(
                self._belief, "challenge_result",
                actor_idx=actor_idx,
                claimed_card=self._pending_claimed_card,
                challenger_idx=event.loser_idx if actor_won else event.winner_idx,
                actor_won=actor_won,
            )

        elif et == "influence_loss":
            card = Card(event.card_lost) if event.card_lost else None
            if card is not None:
                self._updater.update(
                    self._belief, "influence_loss",
                    player_idx=event.player_idx,
                    card_lost=card,
                )

        elif et == "action_resolved":
            # If the resolved action was an Exchange, reset actor's row.
            if event.action_type == "Exchange" and event.actor_idx is not None:
                self._updater.update(
                    self._belief, "exchange",
                    actor_idx=event.actor_idx,
                )
            # Clear pending context — action completed without challenge
            self._pending_claimed_card = None
            self._pending_actor_idx    = None

        # All other events (game_start, game_end, income, coup …) are no-ops.

    def process_raw(self, event_type: str, **kwargs) -> None:
        """
        Process an event specified directly as kwargs (game engine path).
        Kwargs are forwarded verbatim to BeliefUpdater.update().
        """
        self._updater.update(self._belief, event_type, **kwargs)

    def replay_game(self, game) -> None:
        """
        Replay all events in a ParsedGame through the tracker.
        Useful for reconstructing belief state at game end for analysis.
        """
        for event in game.events:
            self.process_event(event)

    # ------------------------------------------------------------------ #
    #  Read-only accessors                                                #
    # ------------------------------------------------------------------ #

    @property
    def belief(self) -> BeliefState:
        return self._belief

    @property
    def model(self) -> OpponentModel:
        return self._model

    def prob(self, player_idx: int, card: Card) -> float:
        """P(player holds ≥1 copy of card)."""
        return self._belief.prob(player_idx, card)

    def credibility(self, player_idx: int) -> float:
        """Credibility multiplier for player_idx's claims."""
        return self._model.credibility(player_idx)

    def entropy(self, player_idx: int) -> float:
        """Shannon entropy of player_idx's card distribution."""
        return self._belief.entropy(player_idx)

    def most_likely_card(self, player_idx: int) -> Optional[Card]:
        """Most probable card in player_idx's hidden hand."""
        return self._belief.most_likely_card(player_idx)

    def is_consistent(self) -> bool:
        """True iff all deck conservation constraints are satisfied."""
        return ConstraintNormaliser().is_consistent(self._belief)

    # ------------------------------------------------------------------ #
    #  Integration with Observation and feature extraction               #
    # ------------------------------------------------------------------ #

    def enrich_observation(self, obs) -> Any:
        """
        Attach belief_state and opponent_model to an Observation object.
        Returns the same Observation with the fields populated (mutates in place).
        """
        obs.belief_state   = self._belief
        obs.opponent_model = self._model
        return obs

    def feature_vector(self, config=None) -> np.ndarray:
        """
        Return the belief portion of the feature vector as a flat float32 array.
        Concatenates belief probs + credibility scores.
        Shape: (n_players × N_CARDS + n_players,) = (n_players × 6,).
        """
        probs = self._belief.probs.flatten().astype(np.float32)
        creds = self._model.all_credibilities().astype(np.float32)
        return np.concatenate([probs, creds])

    # ------------------------------------------------------------------ #
    #  Snapshot / restore (for MCTS / counterfactual reasoning)          #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> BeliefSnapshot:
        """Return an immutable copy of the current tracker state."""
        return BeliefSnapshot(
            belief=self._belief.copy(),
            model=self._model.copy(),
            _pending_claimed_card=self._pending_claimed_card,
            _pending_actor_idx=self._pending_actor_idx,
        )

    def restore(self, snap: BeliefSnapshot) -> None:
        """Restore tracker state from a snapshot."""
        self._belief  = snap.belief.copy()
        self._model   = snap.model.copy()
        self._updater = BeliefUpdater(
            n_players=self._belief.n_players,
            auto_normalise=self._updater.auto_normalise,
            opponent_model=self._model,
        )
        self._pending_claimed_card = snap._pending_claimed_card
        self._pending_actor_idx    = snap._pending_actor_idx

    # ------------------------------------------------------------------ #
    #  Display                                                            #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        lines = ["BeliefTracker summary", "=" * 60]
        lines.append(self._belief.summary_table())
        lines.append("")
        lines.append(self._model.summary())
        lines.append(f"\nConsistent: {self.is_consistent()}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BeliefTracker(n={self._belief.n_players}, "
            f"observer={self._belief.observer_idx}, "
            f"consistent={self.is_consistent()})"
        )
