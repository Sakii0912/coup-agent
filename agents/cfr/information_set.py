"""
agents/cfr/information_set.py — Information set abstraction for Coup CFR.

─────────────────────────────────────────────────────────────────────────────
What is an information set?
─────────────────────────────────────────────────────────────────────────────

In an imperfect-information game, a player's "information set" is the set of
all game states that are indistinguishable to them given their observations.
Two game states are in the same information set if, from the player's
perspective, they look identical.

In Coup, two states are indistinguishable if they share:
  - The player's own hand
  - All publicly observable info (coins, influence counts, revealed cards)
  - The same action history (what has been claimed/done)
  - The same pending decision context (what card is being claimed right now)

The strategy table maps information sets → action probability distributions.
Crucially, the key must be HASHABLE (for dict lookup) and COMPACT (to keep
the table size manageable).

─────────────────────────────────────────────────────────────────────────────
Abstraction
─────────────────────────────────────────────────────────────────────────────

Without abstraction, the information set space is exponentially large.
We apply these abstractions to make tabular CFR tractable:

  Coin bucketing:   0–2 → 0,  3–5 → 1,  6–8 → 2,  9+ → 3
                    Reduces 13 coin values to 4 buckets.

  History length:   Only the last HISTORY_LENGTH action types are kept.
                    Earlier history has diminishing strategic relevance.

  Opponent sorting: Opponents are identified by their (influence, coins_bucket,
                    revealed) tuple, sorted canonically. This means the infoset
                    is the same regardless of which seat an opponent sits in —
                    only their observable state matters.

  Decision types:   Six distinct decision types share the same table structure
                    but are distinguished by a string field, so the strategy
                    for "should I challenge?" is separate from "which action?".

─────────────────────────────────────────────────────────────────────────────
Action encoding
─────────────────────────────────────────────────────────────────────────────

All actions across all decision types are encoded as strings for uniform
table lookup:

  Main actions:      "Income", "ForeignAid", "Tax", "Exchange",
                     "Coup:0", "Assassinate:1", "Steal:2"  (relative target slot)
  Challenge:         "Challenge", "Pass"
  Block:             "Block:Duke", "Block:Contessa", "Pass"
  Block challenge:   "Challenge", "Pass"
  Lose influence:    "Lose:Duke", "Lose:Captain"
  Exchange keep:     "Keep:Duke,Captain"  (sorted card names)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dataclasses import dataclass
from typing import List, Optional, Tuple

from coup.cards import Card, ActionType
from coup.state import Observation
from coup.actions import Action, Block, get_legal_actions, get_legal_blocks


# ---------------------------------------------------------------------------
# Abstraction parameters
# ---------------------------------------------------------------------------

HISTORY_LENGTH: int = 4      # last N action types kept in infoset
COIN_THRESHOLDS = (3, 6, 9)  # bucket boundaries: [0-2], [3-5], [6-8], [9+]


def bucket_coins(coins: int) -> int:
    """Map coin count to a bucket index 0–3."""
    for i, t in enumerate(COIN_THRESHOLDS):
        if coins < t:
            return i
    return len(COIN_THRESHOLDS)


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------

class DecisionType:
    MAIN_ACTION      = "action"
    CHALLENGE_ACTION = "challenge_action"
    BLOCK            = "block"
    CHALLENGE_BLOCK  = "challenge_block"
    LOSE_INFLUENCE   = "lose_influence"
    EXCHANGE         = "exchange"


# ---------------------------------------------------------------------------
# InformationSet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InformationSet:
    """
    Hashable, compact key representing a player's information at a decision
    point. Two game states with identical InformationSets receive the same
    strategy from the CFR table.

    Fields:
        hand:           Sorted tuple of own card names. ("Captain", "Duke")
        my_coins_b:     Own coin bucket (0–3).
        my_influence:   Own remaining hidden cards (1 or 2).
        opponents:      Sorted tuple of (influence, coins_bucket,
                        revealed_sorted) for each alive opponent.
        history:        Last HISTORY_LENGTH action type strings.
        decision_type:  Which of the six decision points this covers.
        pending:        Card claimed / action type in the pending context
                        (e.g. "Duke" when deciding whether to challenge a Tax).
        n_alive:        Number of alive players (affects strategic context).
    """
    hand:          Tuple[str, ...]
    my_coins_b:    int
    my_influence:  int
    opponents:     Tuple[Tuple[int, int, Tuple[str, ...]], ...]
    history:       Tuple[str, ...]
    decision_type: str
    pending:       str
    n_alive:       int

    # ------------------------------------------------------------------ #
    #  Factories (one per decision type)                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def for_main_action(
        cls,
        obs: Observation,
        history: List[str],
    ) -> "InformationSet":
        return cls._build(obs, history, DecisionType.MAIN_ACTION, "")

    @classmethod
    def for_challenge_action(
        cls,
        obs: Observation,
        history: List[str],
        claimed_card: Optional[str],
    ) -> "InformationSet":
        return cls._build(obs, history, DecisionType.CHALLENGE_ACTION,
                          claimed_card or "")

    @classmethod
    def for_block(
        cls,
        obs: Observation,
        history: List[str],
        incoming_action_type: str,
    ) -> "InformationSet":
        return cls._build(obs, history, DecisionType.BLOCK, incoming_action_type)

    @classmethod
    def for_challenge_block(
        cls,
        obs: Observation,
        history: List[str],
        blocking_card: str,
    ) -> "InformationSet":
        return cls._build(obs, history, DecisionType.CHALLENGE_BLOCK, blocking_card)

    @classmethod
    def for_lose_influence(
        cls,
        obs: Observation,
        history: List[str],
    ) -> "InformationSet":
        return cls._build(obs, history, DecisionType.LOSE_INFLUENCE, "")

    @classmethod
    def for_exchange(
        cls,
        obs: Observation,
        history: List[str],
    ) -> "InformationSet":
        return cls._build(obs, history, DecisionType.EXCHANGE, "")

    # ------------------------------------------------------------------ #
    #  Internal builder                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def _build(
        cls,
        obs: Observation,
        history: List[str],
        decision_type: str,
        pending: str,
    ) -> "InformationSet":
        hand = tuple(sorted(c.value for c in obs.my_hand))
        my_coins_b = bucket_coins(obs.my_coins)
        my_influence = len(obs.my_hand)

        # Opponents: alive only, sorted canonically so seat order doesn't matter
        opp_tuples = []
        for p in obs.others:
            if p["is_alive"]:
                opp_tuples.append((
                    p["influence_count"],
                    bucket_coins(p["coins"]),
                    tuple(sorted(p.get("revealed_cards", []))),
                ))
        opponents = tuple(sorted(opp_tuples))

        trimmed_history = tuple(history[-HISTORY_LENGTH:])
        n_alive = sum(1 for p in obs.others if p["is_alive"]) + (
            1 if obs.my_hand else 0
        )

        return cls(
            hand=hand,
            my_coins_b=my_coins_b,
            my_influence=my_influence,
            opponents=opponents,
            history=trimmed_history,
            decision_type=decision_type,
            pending=pending,
            n_alive=n_alive,
        )


# ---------------------------------------------------------------------------
# Action key encoding
# ---------------------------------------------------------------------------

def encode_main_action(action: Action, obs: Observation) -> str:
    """
    Encode a main Action as a canonical string key.

    Targeted actions use a relative slot (0 = next alive player clockwise,
    1 = one after that, etc.) rather than absolute player indices, so the
    strategy generalises across seats.
    """
    if action.target_idx is None:
        return action.type.value

    # Convert absolute target_idx → relative slot
    alive_others = [p for p in obs.others if p["is_alive"]]
    alive_others_sorted = sorted(alive_others, key=lambda p: (
        (p["idx"] - obs.my_idx) % (len(obs.others) + 1)
    ))
    for slot, p in enumerate(alive_others_sorted):
        if p["idx"] == action.target_idx:
            return f"{action.type.value}:{slot}"

    return f"{action.type.value}:0"   # fallback


def encode_block(block: Optional[Block]) -> str:
    """Encode a block choice (or None for Pass) as a string key."""
    if block is None:
        return "Pass"
    return f"Block:{block.blocking_card.value}"


def encode_lose_influence(card: Card) -> str:
    return f"Lose:{card.value}"


def encode_exchange_keep(kept_cards: List[Card]) -> str:
    return "Keep:" + ",".join(sorted(c.value for c in kept_cards))


# ---------------------------------------------------------------------------
# Legal action keys for each decision type
# ---------------------------------------------------------------------------

def legal_main_action_keys(obs: Observation, legal_actions: List[Action]) -> List[str]:
    return [encode_main_action(a, obs) for a in legal_actions]


def legal_challenge_keys() -> List[str]:
    return ["Challenge", "Pass"]


def legal_block_keys(legal_blocks: List[Block]) -> List[str]:
    return ["Pass"] + [encode_block(b) for b in legal_blocks]


def legal_lose_influence_keys(obs: Observation) -> List[str]:
    return [encode_lose_influence(c) for c in obs.my_hand]


def legal_exchange_keys(options: List[Card], n_keep: int) -> List[str]:
    """Generate all valid keep combinations for an Ambassador exchange."""
    from itertools import combinations
    results = []
    for indices in combinations(range(len(options)), n_keep):
        kept = [options[i] for i in indices]
        results.append(encode_exchange_keep(kept))
    return results


def decode_main_action(
    key: str,
    obs: Observation,
    legal_actions: List[Action],
) -> Optional[Action]:
    """Reverse-map a string key back to an Action object."""
    for action in legal_actions:
        if encode_main_action(action, obs) == key:
            return action
    return None


def decode_block(
    key: str,
    legal_blocks: List[Block],
) -> Optional[Block]:
    """Reverse-map a string key back to a Block or None."""
    if key == "Pass":
        return None
    for block in legal_blocks:
        if encode_block(block) == key:
            return block
    return None


def decode_lose_influence(key: str, obs: Observation) -> int:
    """Reverse-map a Lose:CardName key to a hand index."""
    card_name = key.replace("Lose:", "")
    for i, card in enumerate(obs.my_hand):
        if card.value == card_name:
            return i
    return 0   # fallback


def decode_exchange_keep(key: str, options: List[Card], n_keep: int) -> List[int]:
    """Reverse-map a Keep:... key to a list of indices into options."""
    card_names = set(key.replace("Keep:", "").split(","))
    indices = []
    used = set()
    for i, card in enumerate(options):
        if card.value in card_names and i not in used:
            indices.append(i)
            used.add(i)
            if len(indices) == n_keep:
                break
    # Fallback: fill with first available if decoding fails
    if len(indices) < n_keep:
        from itertools import combinations
        indices = list(list(combinations(range(len(options)), n_keep))[0])
    return indices[:n_keep]
