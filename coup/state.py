"""
state.py — Core data structures: PlayerState, GameState, Observation.

Design principle: GameState is the "god view" (includes hidden info).
Observation is what a specific player can see — used as agent input.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .cards import Card, DECK_COMPOSITION


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    """Mutable state for a single player."""

    name: str
    idx: int
    hand: List[Card]       # Face-down cards — private, hidden from others
    revealed: List[Card]   # Face-up eliminated cards — public info
    coins: int = 2

    # ---- Properties -------------------------------------------------------

    @property
    def influence_count(self) -> int:
        """Number of remaining (face-down) influence cards."""
        return len(self.hand)

    @property
    def is_alive(self) -> bool:
        return len(self.hand) > 0

    # ---- Mutators ---------------------------------------------------------

    def has_card(self, card: Card) -> bool:
        return card in self.hand

    def lose_influence(self, card_idx: int) -> Card:
        """
        Flip card at `card_idx` face-up (player loses that influence).
        Returns the card that was revealed.
        """
        assert 0 <= card_idx < len(self.hand), (
            f"Invalid card index {card_idx} for hand of size {len(self.hand)}"
        )
        card = self.hand.pop(card_idx)
        self.revealed.append(card)
        return card

    def swap_card(self, card: Card, deck: List[Card]) -> Card:
        """
        After winning a challenge: return `card` to the deck, draw a new one.
        Deck is shuffled after the swap. Returns the newly drawn card.
        """
        assert card in self.hand, f"{card.value} not in {self.name}'s hand"
        assert len(deck) > 0, "Deck is empty — cannot swap"
        self.hand.remove(card)
        deck.append(card)
        random.shuffle(deck)
        new_card = deck.pop()
        self.hand.append(new_card)
        return new_card

    # ---- Serialisation ----------------------------------------------------

    def public_view(self) -> Dict[str, Any]:
        """What other players can observe (no hand content)."""
        return {
            "name": self.name,
            "idx": self.idx,
            "coins": self.coins,
            "influence_count": self.influence_count,
            "revealed_cards": [c.value for c in self.revealed],
            "is_alive": self.is_alive,
        }

    def __repr__(self) -> str:
        hand_str = [c.value for c in self.hand]
        rev_str = [c.value for c in self.revealed]
        return (
            f"PlayerState(name={self.name!r}, coins={self.coins}, "
            f"hand={hand_str}, revealed={rev_str})"
        )


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """
    Complete (god-view) game state. Holds all hidden and public information.
    Agents never receive this directly — they receive Observation objects.
    """

    players: List[PlayerState]
    deck: List[Card]
    current_player_idx: int = 0
    turn_number: int = 0

    # ---- Factory ----------------------------------------------------------

    @classmethod
    def new_game(
        cls, player_names: List[str], seed: Optional[int] = None
    ) -> "GameState":
        """Shuffle deck, deal 2 cards and 2 coins to each player."""
        assert 2 <= len(player_names) <= 6, "Coup supports 2–6 players"
        if seed is not None:
            random.seed(seed)

        deck = DECK_COMPOSITION.copy()
        random.shuffle(deck)

        players: List[PlayerState] = []
        for idx, name in enumerate(player_names):
            hand = [deck.pop(), deck.pop()]
            players.append(
                PlayerState(name=name, idx=idx, hand=hand, revealed=[], coins=2)
            )

        return cls(players=players, deck=deck, current_player_idx=0, turn_number=0)

    # ---- Derived properties -----------------------------------------------

    @property
    def alive_players(self) -> List[PlayerState]:
        return [p for p in self.players if p.is_alive]

    @property
    def alive_player_indices(self) -> List[int]:
        return [p.idx for p in self.players if p.is_alive]

    @property
    def is_game_over(self) -> bool:
        return len(self.alive_players) <= 1

    @property
    def winner(self) -> Optional[PlayerState]:
        if not self.is_game_over:
            return None
        alive = self.alive_players
        return alive[0] if alive else None

    def current_player(self) -> PlayerState:
        return self.players[self.current_player_idx]

    # ---- Mutation ---------------------------------------------------------

    def advance_turn(self) -> None:
        """Advance current_player_idx to the next alive player."""
        self.turn_number += 1
        n = len(self.players)
        idx = (self.current_player_idx + 1) % n
        visited = 0
        while not self.players[idx].is_alive and visited < n:
            idx = (idx + 1) % n
            visited += 1
        self.current_player_idx = idx

    # ---- Observation ------------------------------------------------------

    def get_observation(self, player_idx: int) -> "Observation":
        """
        Build the Observation for player `player_idx`.
        The player sees their own hand; other players' hands are hidden.
        """
        me = self.players[player_idx]
        others_public = [
            p.public_view() for p in self.players if p.idx != player_idx
        ]
        return Observation(
            my_idx=player_idx,
            my_hand=me.hand.copy(),
            my_coins=me.coins,
            my_revealed=me.revealed.copy(),
            others=others_public,
            deck_size=len(self.deck),
            current_player_idx=self.current_player_idx,
            turn_number=self.turn_number,
        )

    def __repr__(self) -> str:
        lines = [
            f"GameState(turn={self.turn_number}, "
            f"current={self.players[self.current_player_idx].name})"
        ]
        for p in self.players:
            lines.append(f"  {p}")
        lines.append(f"  deck_size={len(self.deck)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Observation (agent input)
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    What a specific player can observe at a point in the game.
    This is the primary input structure passed to every agent decision method.

    Phase 1 fields: own hand, coins, revealed cards, other players' public
    views, deck size, current player index, turn number.

    Phase 3 fields (optional, None if belief tracker not active):
      belief_state:   BeliefState — probability distribution over each
                      opponent's hidden cards. None until Phase 3 is wired in.
      opponent_model: OpponentModel — per-player bluff rate estimates.
                      None until Phase 3 is wired in.
    """

    my_idx: int
    my_hand: List[Card]              # Own (private) cards
    my_coins: int
    my_revealed: List[Card]          # Own revealed (lost) cards
    others: List[Dict[str, Any]]     # Public info about each other player
    deck_size: int
    current_player_idx: int
    turn_number: int

    # Phase 3 — belief tracker (Optional so Phase 1/2 code is unchanged)
    belief_state: Optional[Any] = None    # belief.BeliefState | None
    opponent_model: Optional[Any] = None  # belief.OpponentModel | None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "my_idx": self.my_idx,
            "my_hand": [c.value for c in self.my_hand],
            "my_coins": self.my_coins,
            "my_revealed": [c.value for c in self.my_revealed],
            "others": self.others,
            "deck_size": self.deck_size,
            "current_player_idx": self.current_player_idx,
            "turn_number": self.turn_number,
        }
        if self.belief_state is not None:
            d["belief_state"] = self.belief_state.to_dict()
        if self.opponent_model is not None:
            d["opponent_model"] = self.opponent_model.to_dict()
        return d

    def __repr__(self) -> str:
        hand = [c.value for c in self.my_hand]
        has_belief = self.belief_state is not None
        return (
            f"Observation(player={self.my_idx}, hand={hand}, "
            f"coins={self.my_coins}, turn={self.turn_number}, "
            f"belief={'yes' if has_belief else 'no'})"
        )
