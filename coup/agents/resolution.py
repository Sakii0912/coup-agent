"""
resolution.py — Core game-resolution logic: challenges and action effects.

Challenge resolution rules:
  - Actor HAS the claimed card → actor wins:
      challenger loses 1 influence, actor swaps their card for a fresh draw
  - Actor does NOT have the card → challenger wins:
      actor loses 1 influence, action/block fails

Action effect rules:
  - Assassinate coin cost (3) is deducted BEFORE challenge/block, non-refundable.
  - Coup coin cost (7) is deducted HERE in apply_action_effect.
  - Exchange: draw 2 cards, keep same number as current hand, return rest.
"""

from __future__ import annotations

import random
from typing import Callable, List

from .cards import ActionType, Card
from .actions import Action
from .state import GameState


# Type aliases for callbacks passed in from Game
LoseInfluenceFn = Callable[[int], int]          # player_idx → card_index_to_reveal
ExchangeFn = Callable[[int, List[Card], int], List[int]]  # player_idx, options, n_keep → keep indices


# ---------------------------------------------------------------------------
# Challenge
# ---------------------------------------------------------------------------

def resolve_challenge(
    state: GameState,
    challenger_idx: int,
    actor_idx: int,
    claimed_card: Card,
    lose_influence_callback: LoseInfluenceFn,
) -> bool:
    """
    Resolve a challenge to a claimed card.

    Args:
        state:                   Current game state (mutated in place).
        challenger_idx:          Player who issued the challenge.
        actor_idx:               Player whose claim is being challenged.
        claimed_card:            The card that is being disputed.
        lose_influence_callback: Called with a player_idx, returns the index
                                 of the card in that player's hand to reveal.

    Returns:
        True  → actor wins (had the card); challenger loses influence.
        False → challenger wins (actor was bluffing); actor loses influence.
    """
    actor = state.players[actor_idx]
    challenger = state.players[challenger_idx]

    if actor.has_card(claimed_card):
        # ── Actor wins ─────────────────────────────────────────────────────
        # 1. Swap the proved card for a fresh draw (keeps hand size, removes info)
        actor.swap_card(claimed_card, state.deck)
        # 2. Challenger loses influence
        card_idx = lose_influence_callback(challenger_idx)
        challenger.lose_influence(card_idx)
        return True

    else:
        # ── Challenger wins ────────────────────────────────────────────────
        card_idx = lose_influence_callback(actor_idx)
        actor.lose_influence(card_idx)
        return False


# ---------------------------------------------------------------------------
# Action effects
# ---------------------------------------------------------------------------

def apply_action_effect(
    state: GameState,
    action: Action,
    lose_influence_callback: LoseInfluenceFn,
    exchange_callback: ExchangeFn,
) -> None:
    """
    Apply the mechanical effect of `action` to the game state.

    This is only called after all challenges and blocks have been resolved
    and the action has survived. State is mutated in place.

    Notes:
      - Assassinate: 3-coin cost was deducted when the action was *declared*
        (in game.py). Here we only handle the influence loss on the target.
      - Coup: 7-coin cost deducted here (not pre-deducted).
    """
    actor = state.players[action.actor_idx]

    if action.type == ActionType.INCOME:
        actor.coins += 1

    elif action.type == ActionType.FOREIGN_AID:
        actor.coins += 2

    elif action.type == ActionType.COUP:
        assert action.target_idx is not None, "Coup requires a target"
        actor.coins -= 7
        target = state.players[action.target_idx]
        card_idx = lose_influence_callback(action.target_idx)
        target.lose_influence(card_idx)

    elif action.type == ActionType.TAX:
        actor.coins += 3

    elif action.type == ActionType.ASSASSINATE:
        # Coin cost already deducted in game.py before challenge resolution
        assert action.target_idx is not None, "Assassinate requires a target"
        target = state.players[action.target_idx]
        # Target may have been killed elsewhere (very rare edge case in >2-player games)
        if target.is_alive:
            card_idx = lose_influence_callback(action.target_idx)
            target.lose_influence(card_idx)

    elif action.type == ActionType.STEAL:
        assert action.target_idx is not None, "Steal requires a target"
        target = state.players[action.target_idx]
        stolen = min(2, target.coins)
        actor.coins += stolen
        target.coins -= stolen

    elif action.type == ActionType.EXCHANGE:
        _resolve_exchange(state, action.actor_idx, exchange_callback)

    else:
        raise ValueError(f"Unknown action type: {action.type}")


# ---------------------------------------------------------------------------
# Exchange helper
# ---------------------------------------------------------------------------

def _resolve_exchange(
    state: GameState,
    actor_idx: int,
    exchange_callback: ExchangeFn,
) -> None:
    """
    Ambassador Exchange: draw 2 cards from deck, combine with hand,
    let actor choose which cards to keep (same number as hand size),
    return unchosen cards to the deck and reshuffle.
    """
    actor = state.players[actor_idx]
    n_keep = actor.influence_count          # Keep the same number of cards
    n_draw = min(2, len(state.deck))        # Draw up to 2 (deck may be small)

    drawn = [state.deck.pop() for _ in range(n_draw)]
    all_options = actor.hand + drawn        # Everything available to choose from

    keep_indices = exchange_callback(actor_idx, all_options, n_keep)

    assert len(keep_indices) == n_keep, (
        f"Exchange must keep exactly {n_keep} cards, got {len(keep_indices)}"
    )
    assert len(set(keep_indices)) == len(keep_indices), (
        "Exchange keep_indices must not contain duplicates"
    )
    assert all(0 <= i < len(all_options) for i in keep_indices), (
        "Exchange keep_indices out of range"
    )

    kept = [all_options[i] for i in keep_indices]
    returned = [c for i, c in enumerate(all_options) if i not in keep_indices]

    actor.hand = kept
    state.deck.extend(returned)
    random.shuffle(state.deck)
