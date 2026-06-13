"""
live/state_input.py — Data structures for live game state input.

This module defines the data structures that represent what a human player
can observe at the table, and provides parsers that convert CLI input into
those structures. It is deliberately decoupled from the CLI itself so that
alternative frontends (web, GUI) can reuse the same structures.

─────────────────────────────────────────────────────────────────────────────
What the human inputs
─────────────────────────────────────────────────────────────────────────────

At game start:
  - Number of players (2–6)
  - Player names / seat labels
  - Their own hand (2 cards)

Each turn:
  - Which player is active
  - What action was declared (and claimed card if character action)
  - Who the target is (if applicable)
  - Whether a challenge was issued and who won
  - Whether a block was issued (and claimed card)
  - Whether the block was challenged and who won
  - Which cards were revealed (influence losses)
  - What coins changed hands

At decision time:
  - Which decision the human is being asked to make
    (challenge? block? which card to lose?)

─────────────────────────────────────────────────────────────────────────────
Design principle
─────────────────────────────────────────────────────────────────────────────

All inputs are validated on entry. An invalid input raises InputError with
a clear message, which the CLI catches and re-prompts. This keeps validation
logic here and display logic in cli.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from coup.cards import Card, ActionType


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InputError(ValueError):
    """Raised when user input cannot be parsed into a valid game state."""
    pass


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Case-insensitive card name lookup
_CARD_ALIASES: Dict[str, Card] = {
    c.value.lower(): c for c in Card
}
# Allow short names
_CARD_ALIASES.update({
    "d": Card.DUKE, "duke": Card.DUKE,
    "a": Card.ASSASSIN, "assassin": Card.ASSASSIN,
    "c": Card.CONTESSA, "contessa": Card.CONTESSA,
    "ca": Card.CAPTAIN, "cap": Card.CAPTAIN, "captain": Card.CAPTAIN,
    "am": Card.AMBASSADOR, "amb": Card.AMBASSADOR, "ambassador": Card.AMBASSADOR,
})

_ACTION_ALIASES: Dict[str, ActionType] = {
    a.value.lower(): a for a in ActionType
}
_ACTION_ALIASES.update({
    "income": ActionType.INCOME, "i": ActionType.INCOME,
    "foreign aid": ActionType.FOREIGN_AID, "fa": ActionType.FOREIGN_AID, "foreignaid": ActionType.FOREIGN_AID,
    "coup": ActionType.COUP,
    "tax": ActionType.TAX, "t": ActionType.TAX,
    "assassinate": ActionType.ASSASSINATE, "ass": ActionType.ASSASSINATE, "assassin": ActionType.ASSASSINATE,
    "steal": ActionType.STEAL, "s": ActionType.STEAL,
    "exchange": ActionType.EXCHANGE, "ex": ActionType.EXCHANGE, "amb": ActionType.EXCHANGE,
})


def parse_card(text: str) -> Card:
    """Parse a card name from user input. Case-insensitive, short names allowed."""
    key = text.strip().lower()
    if key in _CARD_ALIASES:
        return _CARD_ALIASES[key]
    raise InputError(
        f"Unknown card {text!r}. Valid: Duke, Assassin, Contessa, Captain, Ambassador"
    )


def parse_cards(text: str) -> List[Card]:
    """Parse a comma- or space-separated list of card names."""
    parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    if not parts:
        raise InputError("No cards provided.")
    return [parse_card(p) for p in parts]


def parse_action(text: str) -> ActionType:
    """Parse an action type from user input."""
    key = text.strip().lower()
    if key in _ACTION_ALIASES:
        return _ACTION_ALIASES[key]
    raise InputError(
        f"Unknown action {text!r}. Valid: Income, ForeignAid, Coup, Tax, "
        f"Assassinate, Steal, Exchange"
    )


def parse_player_idx(text: str, n_players: int) -> int:
    """Parse a player index from user input (0-based or name)."""
    text = text.strip()
    try:
        idx = int(text)
        if 0 <= idx < n_players:
            return idx
        raise InputError(f"Player index {idx} out of range [0, {n_players-1}]")
    except ValueError:
        raise InputError(f"Expected a player number, got {text!r}")


def parse_yes_no(text: str) -> bool:
    """Parse yes/no from user input."""
    key = text.strip().lower()
    if key in ("y", "yes", "1", "true"):
        return True
    if key in ("n", "no", "0", "false"):
        return False
    raise InputError(f"Expected yes/no, got {text!r}")


def parse_coins(text: str) -> int:
    """Parse a non-negative coin count."""
    text = text.strip()
    try:
        coins = int(text)
        if coins < 0:
            raise InputError(f"Coins cannot be negative, got {coins}")
        return coins
    except ValueError:
        raise InputError(f"Expected a number, got {text!r}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OpponentState:
    """Publicly observable state of one opponent."""
    name:             str
    idx:              int
    coins:            int
    influence_count:  int          # hidden cards remaining
    revealed_cards:   List[Card]   # face-up (eliminated) cards
    is_alive:         bool

    def to_public_dict(self) -> dict:
        return {
            "name":           self.name,
            "idx":            self.idx,
            "coins":          self.coins,
            "influence_count":self.influence_count,
            "revealed_cards": [c.value for c in self.revealed_cards],
            "is_alive":       self.is_alive,
        }


@dataclass
class MyState:
    """The human player's own state (full information)."""
    idx:        int
    name:       str
    hand:       List[Card]         # hidden cards (known only to the player)
    coins:      int
    revealed:   List[Card]         # face-up eliminated cards


@dataclass
class LiveGameState:
    """
    Complete observable state of the game at a decision point.

    This is the primary input to the recommender. The human constructs
    this by answering prompts in the CLI.
    """
    n_players:        int
    my_state:         MyState
    opponents:        List[OpponentState]    # ordered by seat index
    current_turn:     int                    # whose turn it is (absolute seat idx)
    turn_number:      int                    # how many turns have elapsed

    # ── Within-turn context ─────────────────────────────────────────────
    pending_action:   Optional[ActionType]   = None   # action just declared
    pending_actor:    Optional[int]          = None   # who declared it
    pending_target:   Optional[int]          = None   # target (if any)
    pending_claimed:  Optional[Card]         = None   # card claimed (if character action)

    # ── Decision type ───────────────────────────────────────────────────
    # What decision is the human being asked to make right now?
    decision:         str = "action"   # "action" | "challenge" | "block" | "challenge_block" | "lose"

    def all_player_public_views(self) -> List[dict]:
        """Return ordered list of all players' public views (for BeliefTracker)."""
        views: List[dict] = [None] * self.n_players
        # Own view
        views[self.my_state.idx] = {
            "name":           self.my_state.name,
            "idx":            self.my_state.idx,
            "coins":          self.my_state.coins,
            "influence_count":len(self.my_state.hand),
            "revealed_cards": [c.value for c in self.my_state.revealed],
            "is_alive":       len(self.my_state.hand) > 0,
        }
        # Opponents
        for opp in self.opponents:
            views[opp.idx] = opp.to_public_dict()
        return views


@dataclass
class TurnEvent:
    """
    One fully-resolved turn's worth of observable events.
    Stored in the session history so the BeliefTracker can be
    incrementally updated rather than rebuilt from scratch each turn.
    """
    turn_number:    int
    actor_idx:      int
    action_type:    ActionType
    claimed_card:   Optional[Card]   = None
    target_idx:     Optional[int]    = None
    challenged:     bool             = False
    challenger_idx: Optional[int]    = None
    actor_won_challenge: Optional[bool] = None   # None if not challenged
    blocked:        bool             = False
    blocker_idx:    Optional[int]    = None
    blocking_card:  Optional[Card]   = None
    block_challenged: bool           = False
    blocker_won:    Optional[bool]   = None
    influence_losses: List[Tuple[int, Card]] = field(default_factory=list)
    # (player_idx, card_lost)
    action_resolved: bool            = True      # False if blocked/failed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_game_state(state: LiveGameState) -> None:
    """
    Run consistency checks on a LiveGameState.
    Raises InputError if anything is inconsistent.
    """
    # Total known cards must not exceed deck size
    from coup.cards import DECK_COMPOSITION
    from collections import Counter

    all_known: List[Card] = list(state.my_state.hand) + list(state.my_state.revealed)
    for opp in state.opponents:
        all_known.extend(opp.revealed_cards)

    counts = Counter(c.value for c in all_known)
    for card_name, count in counts.items():
        if count > 3:
            raise InputError(
                f"{card_name} appears {count} times in known cards, "
                f"but only 3 copies exist in the deck."
            )

    # Influence counts must match reality
    my_influence = len(state.my_state.hand)
    if my_influence < 0 or my_influence > 2:
        raise InputError(f"Invalid influence count: {my_influence}")

    for opp in state.opponents:
        if opp.influence_count < 0 or opp.influence_count > 2:
            raise InputError(
                f"Player {opp.name} has invalid influence count: {opp.influence_count}"
            )
        if opp.influence_count == 0 and opp.is_alive:
            raise InputError(
                f"Player {opp.name} is marked alive but has 0 influence."
            )

    # Seat indices must be unique and cover [0, n_players)
    all_idxs = [state.my_state.idx] + [o.idx for o in state.opponents]
    if sorted(all_idxs) != list(range(state.n_players)):
        raise InputError(
            f"Player seat indices {all_idxs} are not a valid assignment "
            f"for {state.n_players} players."
        )
