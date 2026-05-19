"""
tests/test_cards_and_state.py

Tests for:
  - Deck composition
  - GameState.new_game() initialisation
  - PlayerState mutations (lose_influence, swap_card)
  - Observation public/private split
  - Legal action generation rules
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coup.cards import (
    Card, ActionType, DECK_COMPOSITION,
    ACTION_CLAIMS, BLOCK_MAP, DECK_COMPOSITION,
)
from coup.state import PlayerState, GameState, Observation
from coup.actions import Action, Block, get_legal_actions, get_legal_blocks


# =========================================================================
# Deck
# =========================================================================

class TestDeck:
    def test_deck_has_15_cards(self):
        assert len(DECK_COMPOSITION) == 15

    def test_deck_has_3_of_each_card(self):
        for card in Card:
            assert DECK_COMPOSITION.count(card) == 3


# =========================================================================
# GameState initialisation
# =========================================================================

class TestGameStateInit:
    def test_new_game_two_players(self):
        state = GameState.new_game(["Alice", "Bob"], seed=0)
        assert len(state.players) == 2
        assert state.turn_number == 0
        assert state.current_player_idx == 0

    def test_each_player_gets_two_cards(self):
        state = GameState.new_game(["A", "B", "C"], seed=1)
        for p in state.players:
            assert len(p.hand) == 2

    def test_each_player_starts_with_two_coins(self):
        state = GameState.new_game(["A", "B"], seed=2)
        for p in state.players:
            assert p.coins == 2

    def test_deck_has_remaining_cards_after_deal(self):
        # 15 total - (n_players * 2) dealt
        for n in range(2, 7):
            names = [f"P{i}" for i in range(n)]
            state = GameState.new_game(names, seed=3)
            assert len(state.deck) == 15 - n * 2

    def test_all_dealt_cards_unique_within_composition_limit(self):
        """No card type exceeds 3 copies across all hands + deck."""
        state = GameState.new_game(["A", "B", "C", "D"], seed=4)
        all_cards = state.deck.copy()
        for p in state.players:
            all_cards.extend(p.hand)
        for card in Card:
            assert all_cards.count(card) == 3

    def test_seeded_games_are_reproducible(self):
        s1 = GameState.new_game(["A", "B"], seed=99)
        s2 = GameState.new_game(["A", "B"], seed=99)
        assert s1.players[0].hand == s2.players[0].hand

    def test_different_seeds_produce_different_deals(self):
        s1 = GameState.new_game(["A", "B"], seed=1)
        s2 = GameState.new_game(["A", "B"], seed=2)
        # Not guaranteed but overwhelmingly likely
        all_same = all(
            s1.players[i].hand == s2.players[i].hand
            for i in range(2)
        )
        assert not all_same

    def test_too_few_players_raises(self):
        with pytest.raises(AssertionError):
            GameState.new_game(["Solo"])

    def test_too_many_players_raises(self):
        with pytest.raises(AssertionError):
            GameState.new_game([f"P{i}" for i in range(7)])


# =========================================================================
# PlayerState mutations
# =========================================================================

class TestPlayerState:
    def _make_player(self, hand=None):
        hand = hand or [Card.DUKE, Card.ASSASSIN]
        return PlayerState(name="Test", idx=0, hand=hand.copy(), revealed=[])

    def test_influence_count_equals_hand_size(self):
        p = self._make_player([Card.DUKE, Card.CAPTAIN])
        assert p.influence_count == 2

    def test_is_alive_with_cards(self):
        p = self._make_player([Card.DUKE])
        assert p.is_alive

    def test_is_dead_with_no_cards(self):
        p = self._make_player([Card.DUKE])
        p.lose_influence(0)
        assert not p.is_alive

    def test_lose_influence_moves_card_to_revealed(self):
        p = self._make_player([Card.DUKE, Card.ASSASSIN])
        removed = p.lose_influence(0)
        assert removed == Card.DUKE
        assert Card.DUKE in p.revealed
        assert Card.DUKE not in p.hand
        assert len(p.hand) == 1

    def test_lose_influence_invalid_index_raises(self):
        p = self._make_player([Card.DUKE])
        with pytest.raises(AssertionError):
            p.lose_influence(5)

    def test_swap_card_replaces_in_hand(self):
        """
        swap_card removes the proved card and adds exactly one new card drawn
        from the deck. Hand size must stay the same.
        Note: we cannot assert the DUKE is gone — after returning it to the
        deck and reshuffling, it may be drawn straight back. Instead we verify
        hand size stays constant and the new card is in the hand.
        """
        deck = [Card.CONTESSA, Card.CONTESSA, Card.CONTESSA]
        p = self._make_player([Card.DUKE, Card.ASSASSIN])
        hand_size_before = len(p.hand)
        new_card = p.swap_card(Card.DUKE, deck)
        assert len(p.hand) == hand_size_before   # hand size unchanged
        assert new_card in p.hand                # drawn card was added

    def test_swap_card_returns_old_card_to_deck(self):
        # Deck only contains CONTESSA; DUKE is added then drawn out by the draw
        # After swap: DUKE goes in, CONTESSA (or DUKE if reshuffled) comes out.
        # What we verify: deck size stays the same (one in, one out).
        deck = [Card.CONTESSA]
        p = self._make_player([Card.DUKE])
        deck_size_before = len(deck)
        p.swap_card(Card.DUKE, deck)
        # Net deck size must be unchanged (DUKE added, one card drawn)
        assert len(deck) == deck_size_before

    def test_has_card_true(self):
        p = self._make_player([Card.DUKE, Card.CAPTAIN])
        assert p.has_card(Card.DUKE)

    def test_has_card_false(self):
        p = self._make_player([Card.DUKE])
        assert not p.has_card(Card.CONTESSA)

    def test_public_view_hides_hand(self):
        p = self._make_player([Card.DUKE, Card.ASSASSIN])
        view = p.public_view()
        assert "hand" not in view
        assert view["influence_count"] == 2
        assert view["coins"] == 2


# =========================================================================
# Observation
# =========================================================================

class TestObservation:
    def test_observation_shows_own_hand(self):
        state = GameState.new_game(["Alice", "Bob"], seed=5)
        obs = state.get_observation(0)
        assert obs.my_hand == state.players[0].hand
        assert obs.my_idx == 0

    def test_observation_hides_opponent_hand(self):
        state = GameState.new_game(["Alice", "Bob"], seed=5)
        obs = state.get_observation(0)
        for other in obs.others:
            assert "hand" not in other

    def test_observation_to_dict_is_serialisable(self):
        import json
        state = GameState.new_game(["Alice", "Bob"], seed=6)
        obs = state.get_observation(0)
        d = obs.to_dict()
        # Should be JSON-serialisable without error
        json.dumps(d)

    def test_advance_turn_skips_dead_players(self):
        state = GameState.new_game(["A", "B", "C"], seed=7)
        # Kill player 1
        state.players[1].hand.clear()
        state.current_player_idx = 0
        state.advance_turn()
        assert state.current_player_idx == 2

    def test_game_over_with_one_alive(self):
        state = GameState.new_game(["A", "B"], seed=8)
        state.players[1].hand.clear()
        assert state.is_game_over
        assert state.winner == state.players[0]

    def test_game_not_over_with_multiple_alive(self):
        state = GameState.new_game(["A", "B"], seed=8)
        assert not state.is_game_over


# =========================================================================
# Legal action generation
# =========================================================================

class TestLegalActions:
    def _state(self, n=3, seed=10):
        return GameState.new_game([f"P{i}" for i in range(n)], seed=seed)

    def test_income_always_available(self):
        state = self._state()
        actions = get_legal_actions(state, 0)
        types = [a.type for a in actions]
        assert ActionType.INCOME in types

    def test_foreign_aid_always_available(self):
        state = self._state()
        actions = get_legal_actions(state, 0)
        types = [a.type for a in actions]
        assert ActionType.FOREIGN_AID in types

    def test_tax_always_available(self):
        state = self._state()
        actions = get_legal_actions(state, 0)
        types = [a.type for a in actions]
        assert ActionType.TAX in types

    def test_exchange_always_available(self):
        state = self._state()
        actions = get_legal_actions(state, 0)
        types = [a.type for a in actions]
        assert ActionType.EXCHANGE in types

    def test_coup_not_available_below_7_coins(self):
        state = self._state()
        state.players[0].coins = 6
        actions = get_legal_actions(state, 0)
        types = [a.type for a in actions]
        assert ActionType.COUP not in types

    def test_coup_available_at_7_coins(self):
        state = self._state()
        state.players[0].coins = 7
        actions = get_legal_actions(state, 0)
        types = [a.type for a in actions]
        assert ActionType.COUP in types

    def test_must_coup_at_10_coins(self):
        state = self._state()
        state.players[0].coins = 10
        actions = get_legal_actions(state, 0)
        # Only Coup actions should be legal
        assert all(a.type == ActionType.COUP for a in actions)

    def test_assassinate_not_available_below_3_coins(self):
        state = self._state()
        state.players[0].coins = 2
        actions = get_legal_actions(state, 0)
        types = [a.type for a in actions]
        assert ActionType.ASSASSINATE not in types

    def test_assassinate_available_at_3_coins(self):
        state = self._state()
        state.players[0].coins = 3
        actions = get_legal_actions(state, 0)
        types = [a.type for a in actions]
        assert ActionType.ASSASSINATE in types

    def test_steal_not_available_against_zero_coin_players(self):
        state = self._state(n=2)
        state.players[1].coins = 0
        actions = get_legal_actions(state, 0)
        types = [a.type for a in actions]
        assert ActionType.STEAL not in types

    def test_steal_available_against_player_with_coins(self):
        state = self._state(n=2)
        state.players[1].coins = 1
        actions = get_legal_actions(state, 0)
        types = [a.type for a in actions]
        assert ActionType.STEAL in types

    def test_targeted_actions_have_correct_targets(self):
        state = self._state(n=3)
        state.players[0].coins = 7
        actions = get_legal_actions(state, 0)
        coup_targets = {a.target_idx for a in actions if a.type == ActionType.COUP}
        # Should target P1 and P2, not P0 (self)
        assert 0 not in coup_targets
        assert 1 in coup_targets
        assert 2 in coup_targets

    def test_dead_player_not_a_legal_target(self):
        state = self._state(n=3)
        state.players[0].coins = 7
        state.players[2].hand.clear()  # Kill P2
        actions = get_legal_actions(state, 0)
        coup_targets = {a.target_idx for a in actions if a.type == ActionType.COUP}
        assert 2 not in coup_targets

    def test_dead_player_cannot_act(self):
        state = self._state(n=2)
        state.players[0].hand.clear()
        with pytest.raises(AssertionError):
            get_legal_actions(state, 0)


# =========================================================================
# Legal block generation
# =========================================================================

class TestLegalBlocks:
    def test_foreign_aid_blockable_by_duke(self):
        action = Action(type=ActionType.FOREIGN_AID, actor_idx=0)
        blocks = get_legal_blocks(action, blocker_idx=1)
        cards = [b.blocking_card for b in blocks]
        assert Card.DUKE in cards

    def test_assassinate_blockable_by_contessa(self):
        action = Action(type=ActionType.ASSASSINATE, actor_idx=0, target_idx=1)
        blocks = get_legal_blocks(action, blocker_idx=1)
        cards = [b.blocking_card for b in blocks]
        assert Card.CONTESSA in cards

    def test_steal_blockable_by_captain_and_ambassador(self):
        action = Action(type=ActionType.STEAL, actor_idx=0, target_idx=1)
        blocks = get_legal_blocks(action, blocker_idx=1)
        cards = [b.blocking_card for b in blocks]
        assert Card.CAPTAIN in cards
        assert Card.AMBASSADOR in cards

    def test_income_not_blockable(self):
        action = Action(type=ActionType.INCOME, actor_idx=0)
        blocks = get_legal_blocks(action, blocker_idx=1)
        assert blocks == []

    def test_coup_not_blockable(self):
        action = Action(type=ActionType.COUP, actor_idx=0, target_idx=1)
        blocks = get_legal_blocks(action, blocker_idx=1)
        assert blocks == []

    def test_tax_not_blockable(self):
        action = Action(type=ActionType.TAX, actor_idx=0)
        blocks = get_legal_blocks(action, blocker_idx=1)
        assert blocks == []
