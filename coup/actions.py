"""
actions.py — Action / Block data classes and legal-action generation.

Key rule summary:
  - 10+ coins → MUST coup (only coup actions are returned)
  - COUP      → costs 7 coins, requires a live target
  - ASSASSINATE → costs 3 coins (deducted when declared), requires a live target
  - STEAL     → requires a live target that has ≥ 1 coin
  - TAX / EXCHANGE → no cost, no target, always claimable (bluffing allowed)
  - INCOME / FOREIGN_AID → no cost, no target, always available
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .cards import (
    ACTION_CLAIMS,
    BLOCK_MAP,
    BLOCKABLE_ACTIONS,
    CHALLENGEABLE_ACTIONS,
    TARGETED_ACTIONS,
    ActionType,
    Card,
)
from .state import GameState


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    """An action declared by the active player on their turn."""

    type: ActionType
    actor_idx: int
    target_idx: Optional[int] = None   # Required for Coup / Assassinate / Steal

    # ---- Derived properties -----------------------------------------------

    @property
    def claimed_card(self) -> Optional[Card]:
        """The card this action implicitly claims (None for general actions)."""
        return ACTION_CLAIMS.get(self.type)

    @property
    def is_challengeable(self) -> bool:
        return self.type in CHALLENGEABLE_ACTIONS

    @property
    def is_blockable(self) -> bool:
        return self.type in BLOCKABLE_ACTIONS

    @property
    def blocking_cards(self) -> List[Card]:
        """Cards that can legally be used to block this action."""
        return BLOCK_MAP.get(self.type, [])

    @property
    def requires_target(self) -> bool:
        return self.type in TARGETED_ACTIONS

    def __repr__(self) -> str:
        parts = [f"Action({self.type.value}", f"actor={self.actor_idx}"]
        if self.target_idx is not None:
            parts.append(f"target={self.target_idx}")
        if self.claimed_card:
            parts.append(f"claims={self.claimed_card.value}")
        return ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Block:
    """A block claim made in response to an action."""

    blocker_idx: int
    blocking_card: Card      # The card the blocker claims to hold
    blocked_action: Action

    def __repr__(self) -> str:
        return (
            f"Block(blocker={self.blocker_idx}, "
            f"card={self.blocking_card.value}, "
            f"blocks={self.blocked_action.type.value})"
        )


# ---------------------------------------------------------------------------
# Legal action generator
# ---------------------------------------------------------------------------

def get_legal_actions(state: GameState, player_idx: int) -> List[Action]:
    """
    Return all legal actions available to `player_idx` on their turn.

    Enforces:
      - Must-coup rule (≥ 10 coins → only Coup is legal)
      - Coin requirements (Coup ≥ 7, Assassinate ≥ 3)
      - Steal only targets players with ≥ 1 coin
      - All targeted actions exclude dead players
    """
    player = state.players[player_idx]
    assert player.is_alive, "Dead players cannot take actions"

    alive_others = [p for p in state.alive_players if p.idx != player_idx]

    # ── Must coup at 10+ coins ─────────────────────────────────────────────
    if player.coins >= 10:
        return [
            Action(type=ActionType.COUP, actor_idx=player_idx, target_idx=t.idx)
            for t in alive_others
        ]

    legal: List[Action] = []

    # ── General actions ────────────────────────────────────────────────────
    legal.append(Action(type=ActionType.INCOME, actor_idx=player_idx))
    legal.append(Action(type=ActionType.FOREIGN_AID, actor_idx=player_idx))

    # Coup — costs 7 coins, needs a target
    if player.coins >= 7:
        for t in alive_others:
            legal.append(
                Action(type=ActionType.COUP, actor_idx=player_idx, target_idx=t.idx)
            )

    # ── Character actions (anyone can claim any card) ──────────────────────
    # Tax (Duke) — no cost, no target
    legal.append(Action(type=ActionType.TAX, actor_idx=player_idx))

    # Assassinate (Assassin) — costs 3 coins, needs a target
    if player.coins >= 3:
        for t in alive_others:
            legal.append(
                Action(
                    type=ActionType.ASSASSINATE,
                    actor_idx=player_idx,
                    target_idx=t.idx,
                )
            )

    # Steal (Captain) — needs a target with ≥ 1 coin
    for t in alive_others:
        if t.coins > 0:
            legal.append(
                Action(
                    type=ActionType.STEAL,
                    actor_idx=player_idx,
                    target_idx=t.idx,
                )
            )

    # Exchange (Ambassador) — no cost, no target
    legal.append(Action(type=ActionType.EXCHANGE, actor_idx=player_idx))

    return legal


# ---------------------------------------------------------------------------
# Legal block generator
# ---------------------------------------------------------------------------

def get_legal_blocks(action: Action, blocker_idx: int) -> List[Block]:
    """
    Return all legal block claims `blocker_idx` can make against `action`.
    Returns empty list if the action is not blockable.
    """
    if not action.is_blockable:
        return []
    return [
        Block(
            blocker_idx=blocker_idx,
            blocking_card=card,
            blocked_action=action,
        )
        for card in action.blocking_cards
    ]
