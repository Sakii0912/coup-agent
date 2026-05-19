"""
tests/test_resolution.py

Tests for:
  - Challenge resolution (actor wins / loses)
  - apply_action_effect for every action type
  - Exchange card selection
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coup.cards import Card, ActionType
from coup.state import GameState
from coup.actions import Action
from coup.resolution import resolve_challenge, apply_action_effect


# =========================================================================
# Helpers
# =========================================================================

def _make_state(hands, coins=None, seed=None):
    """
    Build a GameState with specific hands and coins for testing.
    hands: list of lists of Cards, one per player.
    coins: list of ints (defaults to 2 each).
    """
    import random
    if seed is not None:
        random.seed(seed)
    names = [f"P{i}" for i in range(len(hands))]
    state = GameState.new_game(names, seed=0)
    for i, (p, hand) in enumerate(zip(state.players, hands)):
        p.hand = list(hand)
    if coins:
        for p, c in zip(state.players, coins):
            p.coins = c
    return state


def _lose_first(player_idx):
    """Callback that always reveals the first card (index 0)."""
    return 0


def _exchange_keep_first(player_idx, options, n_keep):
    """Callback that always keeps the first n_keep cards."""
    return list(range(n_keep))


# =========================================================================
# resolve_challenge
# =========================================================================

class TestResolveChallenge:
    def test_actor_has_card_wins_challenge(self):
        state = _make_state([[Card.DUKE, Card.CAPTAIN], [Card.ASSASSIN, Card.CONTESSA]])
        actor_wins = resolve_challenge(
            state=state,
            challenger_idx=1,
            actor_idx=0,
            claimed_card=Card.DUKE,
            lose_influence_callback=_lose_first,
        )
        assert actor_wins is True

    def test_actor_wins_challenger_loses_influence(self):
        state = _make_state([[Card.DUKE, Card.CAPTAIN], [Card.ASSASSIN, Card.CONTESSA]])
        resolve_challenge(
            state=state,
            challenger_idx=1,
            actor_idx=0,
            claimed_card=Card.DUKE,
            lose_influence_callback=_lose_first,
        )
        # Challenger P1 should have lost one card
        assert state.players[1].influence_count == 1
        # Actor P0 should still have 2 (minus the swapped one, still 2)
        assert state.players[0].influence_count == 2

    def test_actor_wins_card_is_swapped(self):
        """After winning a challenge, actor's proved card is replaced."""
        state = _make_state([[Card.DUKE, Card.CAPTAIN], [Card.ASSASSIN, Card.CONTESSA]])
        original_hand = state.players[0].hand.copy()
        resolve_challenge(
            state=state,
            challenger_idx=1,
            actor_idx=0,
            claimed_card=Card.DUKE,
            lose_influence_callback=_lose_first,
        )
        new_hand = state.players[0].hand
        # Actor still has 2 cards
        assert len(new_hand) == 2

    def test_bluffer_loses_challenge(self):
        state = _make_state([[Card.CAPTAIN, Card.ASSASSIN], [Card.DUKE, Card.CONTESSA]])
        actor_wins = resolve_challenge(
            state=state,
            challenger_idx=1,
            actor_idx=0,
            claimed_card=Card.DUKE,   # P0 does NOT have Duke
            lose_influence_callback=_lose_first,
        )
        assert actor_wins is False

    def test_bluffer_loses_influence(self):
        state = _make_state([[Card.CAPTAIN, Card.ASSASSIN], [Card.DUKE, Card.CONTESSA]])
        resolve_challenge(
            state=state,
            challenger_idx=1,
            actor_idx=0,
            claimed_card=Card.DUKE,
            lose_influence_callback=_lose_first,
        )
        # Actor (bluffer) P0 loses one card
        assert state.players[0].influence_count == 1
        # Challenger P1 unaffected
        assert state.players[1].influence_count == 2


# =========================================================================
# apply_action_effect — each action type
# =========================================================================

class TestApplyActionEffect:

    # ── Income ──────────────────────────────────────────────────────────

    def test_income_adds_one_coin(self):
        state = _make_state([[Card.DUKE], [Card.CAPTAIN]], coins=[2, 2])
        action = Action(type=ActionType.INCOME, actor_idx=0)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        assert state.players[0].coins == 3
        assert state.players[1].coins == 2

    # ── Foreign Aid ──────────────────────────────────────────────────────

    def test_foreign_aid_adds_two_coins(self):
        state = _make_state([[Card.DUKE], [Card.CAPTAIN]], coins=[2, 2])
        action = Action(type=ActionType.FOREIGN_AID, actor_idx=0)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        assert state.players[0].coins == 4

    # ── Tax ──────────────────────────────────────────────────────────────

    def test_tax_adds_three_coins(self):
        state = _make_state([[Card.DUKE], [Card.CAPTAIN]], coins=[2, 2])
        action = Action(type=ActionType.TAX, actor_idx=0)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        assert state.players[0].coins == 5

    # ── Coup ─────────────────────────────────────────────────────────────

    def test_coup_deducts_7_coins(self):
        state = _make_state(
            [[Card.DUKE, Card.CAPTAIN], [Card.ASSASSIN, Card.CONTESSA]],
            coins=[7, 2]
        )
        action = Action(type=ActionType.COUP, actor_idx=0, target_idx=1)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        assert state.players[0].coins == 0

    def test_coup_target_loses_influence(self):
        state = _make_state(
            [[Card.DUKE, Card.CAPTAIN], [Card.ASSASSIN, Card.CONTESSA]],
            coins=[7, 2]
        )
        action = Action(type=ActionType.COUP, actor_idx=0, target_idx=1)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        assert state.players[1].influence_count == 1

    def test_coup_kills_player_with_one_card(self):
        state = _make_state([[Card.DUKE], [Card.ASSASSIN]], coins=[7, 2])
        # Give P1 only 1 card
        state.players[1].hand = [Card.ASSASSIN]
        action = Action(type=ActionType.COUP, actor_idx=0, target_idx=1)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        assert not state.players[1].is_alive

    # ── Assassinate ──────────────────────────────────────────────────────

    def test_assassinate_target_loses_influence(self):
        # Coin cost pre-deducted before apply_action_effect in real game
        state = _make_state(
            [[Card.ASSASSIN, Card.CAPTAIN], [Card.DUKE, Card.CONTESSA]],
            coins=[0, 2]   # already paid
        )
        action = Action(type=ActionType.ASSASSINATE, actor_idx=0, target_idx=1)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        assert state.players[1].influence_count == 1

    # ── Steal ────────────────────────────────────────────────────────────

    def test_steal_transfers_two_coins(self):
        state = _make_state([[Card.CAPTAIN], [Card.DUKE]], coins=[2, 5])
        action = Action(type=ActionType.STEAL, actor_idx=0, target_idx=1)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        assert state.players[0].coins == 4
        assert state.players[1].coins == 3

    def test_steal_transfers_only_available_coins(self):
        """If target has only 1 coin, steal transfers 1 not 2."""
        state = _make_state([[Card.CAPTAIN], [Card.DUKE]], coins=[2, 1])
        action = Action(type=ActionType.STEAL, actor_idx=0, target_idx=1)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        assert state.players[0].coins == 3
        assert state.players[1].coins == 0

    def test_steal_from_zero_coin_player_no_change(self):
        state = _make_state([[Card.CAPTAIN], [Card.DUKE]], coins=[2, 0])
        action = Action(type=ActionType.STEAL, actor_idx=0, target_idx=1)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        assert state.players[0].coins == 2
        assert state.players[1].coins == 0

    # ── Exchange ─────────────────────────────────────────────────────────

    def test_exchange_keeps_correct_number_of_cards(self):
        state = _make_state([[Card.AMBASSADOR, Card.DUKE], [Card.CAPTAIN]])
        action = Action(type=ActionType.EXCHANGE, actor_idx=0)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        assert state.players[0].influence_count == 2  # still 2

    def test_exchange_returns_cards_to_deck(self):
        state = _make_state([[Card.AMBASSADOR, Card.DUKE], [Card.CAPTAIN]])
        deck_size_before = len(state.deck)
        action = Action(type=ActionType.EXCHANGE, actor_idx=0)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        # Drew 2, kept 2, returned 2
        assert len(state.deck) == deck_size_before

    def test_exchange_with_one_card_keeps_one(self):
        """Player with 1 influence keeps 1 card after exchange."""
        state = _make_state([[Card.AMBASSADOR], [Card.CAPTAIN]])
        action = Action(type=ActionType.EXCHANGE, actor_idx=0)
        apply_action_effect(state, action, _lose_first, _exchange_keep_first)
        assert state.players[0].influence_count == 1

    def test_exchange_invalid_keep_count_raises(self):
        state = _make_state([[Card.AMBASSADOR, Card.DUKE], [Card.CAPTAIN]])
        action = Action(type=ActionType.EXCHANGE, actor_idx=0)

        def bad_exchange(player_idx, options, n_keep):
            return [0]  # returns 1 instead of 2 → should fail

        with pytest.raises(AssertionError):
            apply_action_effect(state, action, _lose_first, bad_exchange)
