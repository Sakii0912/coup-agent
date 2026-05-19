"""
logger.py — Structured game-event logging.

Every observable event in a game is recorded as a LogEvent.
The full game log (list of events + metadata) is JSON-serialisable
and forms the raw training data for later phases.

Event types:
  GAME_START       → initial state snapshot
  ACTION           → active player declares an action
  CHALLENGE        → a player challenges a claimed action
  CHALLENGE_RESULT → who won the challenge and who lost influence
  BLOCK            → a player claims to block an action
  BLOCK_CHALLENGE  → a player challenges a block claim
  INFLUENCE_LOSS   → a player reveals and loses a card
  ACTION_RESOLVED  → action effect applied successfully
  GAME_END         → winner declared
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .cards import Card
from .actions import Action, Block
from .state import GameState, PlayerState


class EventType(Enum):
    GAME_START = "game_start"
    ACTION = "action"
    CHALLENGE = "challenge"
    CHALLENGE_RESULT = "challenge_result"
    BLOCK = "block"
    BLOCK_CHALLENGE = "block_challenge"
    INFLUENCE_LOSS = "influence_loss"
    ACTION_RESOLVED = "action_resolved"
    GAME_END = "game_end"


@dataclass
class LogEvent:
    event_type: EventType
    turn_number: int
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "turn_number": self.turn_number,
            **self.data,
        }


class GameLogger:
    """Collects LogEvents throughout a game and serialises them."""

    def __init__(self, player_names: List[str]):
        self.player_names = player_names
        self.events: List[LogEvent] = []
        self.game_id = int(time.time() * 1_000)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _public_snapshot(self, state: GameState) -> List[Dict[str, Any]]:
        """Snapshot of publicly visible state (no hidden cards)."""
        return [p.public_view() for p in state.players]

    def _evt(
        self, event_type: EventType, turn: int, **kwargs: Any
    ) -> None:
        self.events.append(
            LogEvent(event_type=event_type, turn_number=turn, data=kwargs)
        )

    # ------------------------------------------------------------------ #
    #  Logging calls (one per observable game event)                       #
    # ------------------------------------------------------------------ #

    def log_game_start(self, state: GameState) -> None:
        self._evt(
            EventType.GAME_START,
            turn=0,
            player_names=self.player_names,
            n_players=len(state.players),
            state=self._public_snapshot(state),
        )

    def log_action(self, action: Action, state: GameState) -> None:
        self._evt(
            EventType.ACTION,
            turn=state.turn_number,
            actor_idx=action.actor_idx,
            actor_name=state.players[action.actor_idx].name,
            action_type=action.type.value,
            target_idx=action.target_idx,
            target_name=(
                state.players[action.target_idx].name
                if action.target_idx is not None else None
            ),
            claimed_card=(
                action.claimed_card.value if action.claimed_card else None
            ),
            state=self._public_snapshot(state),
        )

    def log_challenge(
        self, challenger_idx: int, action: Action, state: GameState
    ) -> None:
        self._evt(
            EventType.CHALLENGE,
            turn=state.turn_number,
            challenger_idx=challenger_idx,
            challenger_name=state.players[challenger_idx].name,
            challenged_action=action.type.value,
            challenged_card=(
                action.claimed_card.value if action.claimed_card else None
            ),
            actor_idx=action.actor_idx,
            actor_name=state.players[action.actor_idx].name,
        )

    def log_challenge_result(
        self, winner_idx: int, loser_idx: int, state: GameState
    ) -> None:
        self._evt(
            EventType.CHALLENGE_RESULT,
            turn=state.turn_number,
            winner_idx=winner_idx,
            winner_name=state.players[winner_idx].name,
            loser_idx=loser_idx,
            loser_name=state.players[loser_idx].name,
            state=self._public_snapshot(state),
        )

    def log_block(self, block: Block, state: GameState) -> None:
        self._evt(
            EventType.BLOCK,
            turn=state.turn_number,
            blocker_idx=block.blocker_idx,
            blocker_name=state.players[block.blocker_idx].name,
            blocking_card=block.blocking_card.value,
            blocked_action=block.blocked_action.type.value,
        )

    def log_block_challenge(
        self, challenger_idx: int, block: Block, state: GameState
    ) -> None:
        self._evt(
            EventType.BLOCK_CHALLENGE,
            turn=state.turn_number,
            challenger_idx=challenger_idx,
            challenger_name=state.players[challenger_idx].name,
            challenged_card=block.blocking_card.value,
            blocker_idx=block.blocker_idx,
            blocker_name=state.players[block.blocker_idx].name,
        )

    def log_influence_loss(
        self, player_idx: int, card: Card, state: GameState
    ) -> None:
        self._evt(
            EventType.INFLUENCE_LOSS,
            turn=state.turn_number,
            player_idx=player_idx,
            player_name=state.players[player_idx].name,
            card_lost=card.value,
            # influence_count is logged *before* the card is removed
            influence_remaining=state.players[player_idx].influence_count - 1,
        )

    def log_action_resolved(self, action: Action, state: GameState) -> None:
        self._evt(
            EventType.ACTION_RESOLVED,
            turn=state.turn_number,
            action_type=action.type.value,
            actor_idx=action.actor_idx,
            actor_name=state.players[action.actor_idx].name,
            state=self._public_snapshot(state),
        )

    def log_game_end(
        self, winner: Optional[PlayerState], turn_number: int
    ) -> None:
        self._evt(
            EventType.GAME_END,
            turn=turn_number,
            winner_idx=winner.idx if winner else None,
            winner_name=winner.name if winner else None,
            total_turns=turn_number,
        )

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def get_log(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "player_names": self.player_names,
            "n_players": len(self.player_names),
            "events": [e.to_dict() for e in self.events],
        }

    def save(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.get_log(), f, indent=2)
