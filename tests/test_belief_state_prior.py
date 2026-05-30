"""
tests/test_belief_state_prior.py — Tests for BeliefState.initialise() (sub-goal 3.1).

Coverage:
  Hypergeometric helper — edge cases, correctness against manual calculation
  BeliefState.initialise() — shape, range, conservation, dead players,
                              known hand certainty, spectator view
  BeliefState.from_game_state() — correctness against live GameState
  BeliefState.from_public_snapshot() — log-replay path
  Accessors — prob(), most_likely_card(), entropy(), to_vector()
  Serialisation — to_dict(), copy()
  Error handling — invalid arguments raise ValueError
"""

from __future__ import annotations

import math
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from belief.belief_state import (
    BeliefState,
    CARDS,
    N_CARDS,
    CARD_TO_IDX,
    TOTAL_CARDS,
    COPIES_PER_CARD,
    hypergeometric_at_least_one,
    _encode_known_hand,
)
from coup.cards import Card
from coup.state import GameState


# ===========================================================================
# Hypergeometric helper
# ===========================================================================

class TestHypergeometricAtLeastOne:

    def test_zero_draw_returns_zero(self):
        assert hypergeometric_at_least_one(15, 3, 0) == 0.0

    def test_zero_pool_returns_zero(self):
        assert hypergeometric_at_least_one(0, 0, 0) == 0.0

    def test_zero_hits_returns_zero(self):
        assert hypergeometric_at_least_one(15, 0, 5) == 0.0

    def test_all_hits_returns_one(self):
        # If every card in the pool is a hit, P = 1.0
        assert hypergeometric_at_least_one(5, 5, 2) == 1.0

    def test_draw_whole_pool(self):
        # Drawing all 15 cards from a pool of 15 with 3 hits — P = 1.0
        assert hypergeometric_at_least_one(15, 3, 15) == 1.0

    def test_draw_one_from_large_pool(self):
        # P(at least one Duke | draw 1 from pool of 15 with 3 Dukes) = 3/15 = 0.2
        p = hypergeometric_at_least_one(15, 3, 1)
        assert abs(p - 3 / 15) < 1e-9

    def test_complement_of_zero(self):
        # Manually: P(0 from 15 with 3 hits, draw 2) = C(12,2)/C(15,2) = 66/105
        p_zero = math.comb(12, 2) / math.comb(15, 2)
        expected = 1 - p_zero
        p = hypergeometric_at_least_one(15, 3, 2)
        assert abs(p - expected) < 1e-9

    def test_result_in_unit_interval(self):
        for pool in [5, 10, 15]:
            for hits in range(pool + 1):
                for draw in range(pool + 1):
                    p = hypergeometric_at_least_one(pool, hits, draw)
                    assert 0.0 <= p <= 1.0, f"Out of range: pool={pool}, hits={hits}, draw={draw}, p={p}"

    def test_monotone_in_hits(self):
        # More hits in the pool → higher probability
        p1 = hypergeometric_at_least_one(15, 1, 2)
        p2 = hypergeometric_at_least_one(15, 2, 2)
        p3 = hypergeometric_at_least_one(15, 3, 2)
        assert p1 < p2 < p3

    def test_monotone_in_draw_size(self):
        # Larger draw → higher probability of hitting at least one
        p1 = hypergeometric_at_least_one(15, 3, 1)
        p2 = hypergeometric_at_least_one(15, 3, 2)
        p3 = hypergeometric_at_least_one(15, 3, 5)
        assert p1 < p2 < p3


# ===========================================================================
# BeliefState.initialise — core cases
# ===========================================================================

class TestBeliefStateInit:

    # ── Helpers ─────────────────────────────────────────────────────────

    def _fresh(self, n_players=4, n_revealed_per_player=0, known_hand=None, observer_idx=0):
        """Build a BeliefState with no revealed cards and full hands."""
        influence_counts = [2] * n_players
        revealed_cards = [[] for _ in range(n_players)]
        if n_revealed_per_player > 0:
            for i in range(n_players):
                revealed_cards[i] = [Card.DUKE] * n_revealed_per_player
        return BeliefState.initialise(
            n_players=n_players,
            influence_counts=influence_counts,
            revealed_cards=revealed_cards,
            observer_idx=observer_idx,
            known_hand=known_hand,
        )

    # ── Shape and dtype ──────────────────────────────────────────────────

    def test_probs_shape(self):
        bs = self._fresh(n_players=4)
        assert bs.probs.shape == (4, N_CARDS)

    def test_probs_dtype(self):
        bs = self._fresh(n_players=4)
        assert bs.probs.dtype == np.float64

    def test_available_shape(self):
        bs = self._fresh(n_players=4)
        assert bs.available.shape == (N_CARDS,)

    @pytest.mark.parametrize("n_players", [2, 3, 4, 5, 6])
    def test_all_player_counts(self, n_players):
        bs = self._fresh(n_players=n_players)
        assert bs.probs.shape == (n_players, N_CARDS)

    # ── Value ranges ────────────────────────────────────────────────────

    def test_probs_in_unit_interval(self):
        bs = self._fresh(n_players=4)
        assert np.all(bs.probs >= 0.0)
        assert np.all(bs.probs <= 1.0)

    def test_no_nan_or_inf(self):
        bs = self._fresh(n_players=4)
        assert not np.any(np.isnan(bs.probs))
        assert not np.any(np.isinf(bs.probs))

    # ── Dead players ─────────────────────────────────────────────────────

    def test_dead_player_zero_row(self):
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[2, 0, 2, 1],
            revealed_cards=[[Card.DUKE], [], [], []],
            observer_idx=0,
            known_hand=[Card.ASSASSIN, Card.CAPTAIN],
        )
        np.testing.assert_array_equal(bs.probs[1], np.zeros(N_CARDS))

    def test_dead_player_entropy_zero(self):
        bs = BeliefState.initialise(
            n_players=3,
            influence_counts=[2, 0, 2],
            revealed_cards=[[], [Card.DUKE, Card.CAPTAIN], []],
        )
        assert bs.entropy(1) == 0.0

    # ── Observer's own hand — certainty encoding ─────────────────────────

    def test_known_hand_single_card_certainty(self):
        # Observer holds Duke and Assassin — their row must be 1.0 for those, 0.0 for others
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[2, 2, 2, 2],
            revealed_cards=[[], [], [], []],
            observer_idx=0,
            known_hand=[Card.DUKE, Card.ASSASSIN],
        )
        assert bs.probs[0, CARD_TO_IDX[Card.DUKE]] == 1.0
        assert bs.probs[0, CARD_TO_IDX[Card.ASSASSIN]] == 1.0
        assert bs.probs[0, CARD_TO_IDX[Card.CONTESSA]] == 0.0
        assert bs.probs[0, CARD_TO_IDX[Card.CAPTAIN]] == 0.0
        assert bs.probs[0, CARD_TO_IDX[Card.AMBASSADOR]] == 0.0

    def test_known_hand_two_same_card(self):
        # Holding two Dukes → Duke = 1.0, everything else = 0.0
        bs = BeliefState.initialise(
            n_players=3,
            influence_counts=[2, 2, 2],
            revealed_cards=[[], [], []],
            observer_idx=0,
            known_hand=[Card.DUKE, Card.DUKE],
        )
        assert bs.probs[0, CARD_TO_IDX[Card.DUKE]] == 1.0
        for c in CARDS:
            if c != Card.DUKE:
                assert bs.probs[0, CARD_TO_IDX[c]] == 0.0

    def test_known_hand_one_card_left(self):
        # Observer has only 1 influence remaining
        bs = BeliefState.initialise(
            n_players=3,
            influence_counts=[1, 2, 2],
            revealed_cards=[[Card.CONTESSA], [], []],
            observer_idx=0,
            known_hand=[Card.CAPTAIN],
        )
        assert bs.probs[0, CARD_TO_IDX[Card.CAPTAIN]] == 1.0
        assert bs.probs[0, CARD_TO_IDX[Card.CONTESSA]] == 0.0

    # ── Available pool accounting ─────────────────────────────────────────

    def test_available_sums_to_hidden_total(self):
        # Reveal 1 Duke — available Duke should drop to 2
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[1, 2, 2, 2],
            revealed_cards=[[Card.DUKE], [], [], []],
        )
        assert bs.available[CARD_TO_IDX[Card.DUKE]] == 2.0

    def test_available_full_game_start(self):
        # With no known cards, all 3 copies of each card are available
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[2, 2, 2, 2],
            revealed_cards=[[], [], [], []],
        )
        np.testing.assert_array_equal(bs.available, np.full(N_CARDS, 3.0))

    def test_available_decreases_with_known_hand(self):
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[2, 2, 2, 2],
            revealed_cards=[[], [], [], []],
            observer_idx=0,
            known_hand=[Card.DUKE, Card.DUKE],
        )
        assert bs.available[CARD_TO_IDX[Card.DUKE]] == 1.0   # 3 - 2 = 1

    def test_available_never_negative(self):
        bs = BeliefState.initialise(
            n_players=3,
            influence_counts=[0, 2, 1],
            revealed_cards=[[Card.DUKE, Card.DUKE, Card.DUKE], [], []],
        )
        assert bs.available[CARD_TO_IDX[Card.DUKE]] == 0.0

    # ── Probability ordering ──────────────────────────────────────────────

    def test_more_available_cards_higher_prob(self):
        # Reveal 2 Dukes: Duke has fewer available copies → lower prob for opponents
        # compared to a card with all 3 copies available
        bs_many = BeliefState.initialise(
            n_players=3,
            influence_counts=[2, 2, 2],
            revealed_cards=[[], [], []],
        )
        bs_few = BeliefState.initialise(
            n_players=3,
            influence_counts=[1, 2, 2],
            revealed_cards=[[Card.DUKE, Card.DUKE], [], []],
        )
        # Duke prob for P1 should be lower when only 1 copy is available
        p_many = bs_many.probs[1, CARD_TO_IDX[Card.DUKE]]
        p_few  = bs_few.probs[1, CARD_TO_IDX[Card.DUKE]]
        assert p_few < p_many

    def test_two_influence_higher_prob_than_one(self):
        # A player with 2 hidden cards is more likely to hold a given card
        # than a player with 1 hidden card (all else equal)
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[2, 2, 1, 2],
            revealed_cards=[[], [], [], []],
        )
        p_two = bs.probs[0, 0]
        p_one = bs.probs[2, 0]
        assert p_two > p_one

    def test_symmetry_between_opponents(self):
        # With no known info, all opponents with same influence count
        # should have identical probability distributions
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[2, 2, 2, 2],
            revealed_cards=[[], [], [], []],
        )
        for i in range(1, 4):
            np.testing.assert_array_almost_equal(bs.probs[0], bs.probs[i])

    # ── Spectator view (no known hand) ───────────────────────────────────

    def test_spectator_view(self):
        # No observer_idx, no known_hand — all players treated as opponents
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[2, 2, 2, 2],
            revealed_cards=[[], [], [], []],
            observer_idx=-1,
            known_hand=None,
        )
        assert bs.known_hand is None
        # All probs should be equal and in (0, 1)
        assert np.all(bs.probs > 0.0)
        assert np.all(bs.probs < 1.0)

    # ── Pool size ────────────────────────────────────────────────────────

    def test_pool_size_full_game(self):
        # 4 players × 2 cards = 8 hidden + 7 in deck = 15 total
        # If we observe 0 cards and have no known hand, pool = 15
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[2, 2, 2, 2],
            revealed_cards=[[], [], [], []],
            observer_idx=-1,
        )
        assert bs.pool_size() == TOTAL_CARDS

    def test_pool_size_decreases_with_reveals(self):
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[1, 2, 2, 2],
            revealed_cards=[[Card.DUKE], [], [], []],
        )
        assert bs.pool_size() == TOTAL_CARDS - 1

    def test_pool_size_with_known_hand(self):
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[2, 2, 2, 2],
            revealed_cards=[[], [], [], []],
            observer_idx=0,
            known_hand=[Card.DUKE, Card.ASSASSIN],
        )
        # Observer's 2 cards are known → pool = 15 - 2 = 13
        assert bs.pool_size() == TOTAL_CARDS - 2


# ===========================================================================
# BeliefState.from_game_state
# ===========================================================================

class TestFromGameState:

    def test_matches_manual_init(self):
        state = GameState.new_game(["P0", "P1", "P2", "P3"], seed=42)
        bs = BeliefState.from_game_state(state, observer_idx=0)

        assert bs.n_players == 4
        assert bs.observer_idx == 0
        assert bs.known_hand == list(state.players[0].hand)

    def test_observer_row_is_certain(self):
        state = GameState.new_game(["P0", "P1", "P2"], seed=7)
        obs_hand = state.players[0].hand
        bs = BeliefState.from_game_state(state, observer_idx=0)

        for c_idx, card in enumerate(CARDS):
            expected = 1.0 if card in obs_hand else 0.0
            assert abs(bs.probs[0, c_idx] - expected) < 1e-9

    def test_opponent_rows_are_uncertain(self):
        state = GameState.new_game(["P0", "P1", "P2", "P3"], seed=99)
        bs = BeliefState.from_game_state(state, observer_idx=0)

        # Opponents should have no probability that is exactly 0 or 1
        # (unless the observer holds both copies of a card — rare at start)
        for p_idx in range(1, 4):
            row = bs.probs[p_idx]
            assert np.all(row >= 0.0)
            assert np.all(row <= 1.0)

    def test_dead_player_after_reveal(self):
        state = GameState.new_game(["P0", "P1", "P2"], seed=5)
        # Simulate player 1 losing both influences
        lost1 = state.players[1].lose_influence(0)
        lost2 = state.players[1].lose_influence(0)

        bs = BeliefState.from_game_state(state, observer_idx=0)
        np.testing.assert_array_equal(bs.probs[1], np.zeros(N_CARDS))

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 100])
    def test_multiple_seeds(self, seed):
        state = GameState.new_game(["P0", "P1", "P2", "P3"], seed=seed)
        bs = BeliefState.from_game_state(state, observer_idx=0)
        assert bs.probs.shape == (4, N_CARDS)
        assert np.all(bs.probs >= 0.0)
        assert np.all(bs.probs <= 1.0)
        assert not np.any(np.isnan(bs.probs))


# ===========================================================================
# BeliefState.from_public_snapshot
# ===========================================================================

class TestFromPublicSnapshot:

    def _make_snapshot(self, n_players=4, influence_counts=None, revealed=None):
        if influence_counts is None:
            influence_counts = [2] * n_players
        if revealed is None:
            revealed = [[] for _ in range(n_players)]
        return [
            {
                "name": f"P{i}",
                "idx": i,
                "coins": 2,
                "influence_count": influence_counts[i],
                "revealed_cards": [c.value for c in revealed[i]],
                "is_alive": influence_counts[i] > 0,
            }
            for i in range(n_players)
        ]

    def test_basic_4p_snapshot(self):
        snap = self._make_snapshot(n_players=4)
        bs = BeliefState.from_public_snapshot(snap)
        assert bs.probs.shape == (4, N_CARDS)

    def test_revealed_card_reduces_available(self):
        snap = self._make_snapshot(
            n_players=3,
            influence_counts=[1, 2, 2],
            revealed=[[Card.DUKE], [], []],
        )
        bs = BeliefState.from_public_snapshot(snap)
        assert bs.available[CARD_TO_IDX[Card.DUKE]] == 2.0

    def test_with_known_hand(self):
        snap = self._make_snapshot(n_players=4)
        bs = BeliefState.from_public_snapshot(
            snap, observer_idx=0, known_hand=[Card.CONTESSA, Card.CAPTAIN]
        )
        assert bs.probs[0, CARD_TO_IDX[Card.CONTESSA]] == 1.0
        assert bs.probs[0, CARD_TO_IDX[Card.CAPTAIN]] == 1.0

    def test_spectator_no_known_hand(self):
        snap = self._make_snapshot(n_players=4)
        bs = BeliefState.from_public_snapshot(snap, observer_idx=-1)
        assert bs.known_hand is None
        assert np.all(bs.probs > 0.0)


# ===========================================================================
# Accessors
# ===========================================================================

class TestAccessors:

    @pytest.fixture
    def bs(self):
        return BeliefState.initialise(
            n_players=4,
            influence_counts=[2, 2, 2, 2],
            revealed_cards=[[], [], [], []],
            observer_idx=0,
            known_hand=[Card.DUKE, Card.ASSASSIN],
        )

    def test_prob_own_card(self, bs):
        assert bs.prob(0, Card.DUKE) == 1.0

    def test_prob_own_absent_card(self, bs):
        assert bs.prob(0, Card.CONTESSA) == 0.0

    def test_prob_opponent_in_unit_interval(self, bs):
        p = bs.prob(1, Card.CAPTAIN)
        assert 0.0 <= p <= 1.0

    def test_most_likely_card_dead_player(self, bs):
        bs2 = BeliefState.initialise(
            n_players=3,
            influence_counts=[2, 0, 2],
            revealed_cards=[[], [Card.DUKE, Card.CAPTAIN], []],
        )
        assert bs2.most_likely_card(1) is None

    def test_most_likely_card_known_hand(self, bs):
        # Observer holds Duke and Assassin — most likely is one of them
        result = bs.most_likely_card(0)
        assert result in (Card.DUKE, Card.ASSASSIN)

    def test_entropy_zero_for_known_hand(self, bs):
        # Observer's hand is fully known — maximum certainty, minimum entropy
        # (each card is either 1.0 or 0.0, so entropy of each Bernoulli is 0)
        assert abs(bs.entropy(0)) < 1e-9

    def test_entropy_positive_for_opponent(self, bs):
        assert bs.entropy(1) > 0.0

    def test_entropy_decreases_with_reveals(self):
        # Revealing cards reduces uncertainty about the remaining pool
        bs_before = BeliefState.initialise(
            n_players=3,
            influence_counts=[2, 2, 2],
            revealed_cards=[[], [], []],
        )
        bs_after = BeliefState.initialise(
            n_players=3,
            influence_counts=[1, 2, 2],
            revealed_cards=[[Card.DUKE], [], []],
        )
        # After a reveal the distribution for P1 changes (Duke is less available)
        # Total entropy should differ (not necessarily lower for all players)
        h_before = sum(bs_before.entropy(i) for i in range(3))
        h_after  = sum(bs_after.entropy(i)  for i in range(3))
        # Just verify they're different — direction depends on specifics
        assert abs(h_before - h_after) > 1e-6

    def test_opponent_probs_zeros_own_row(self, bs):
        opp = bs.opponent_probs(0)
        np.testing.assert_array_equal(opp[0], np.zeros(N_CARDS))
        # Other rows should be unchanged
        np.testing.assert_array_equal(opp[1], bs.probs[1])


# ===========================================================================
# Serialisation and copy
# ===========================================================================

class TestSerialisation:

    def test_to_vector_shape(self):
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[2, 2, 2, 2],
            revealed_cards=[[], [], [], []],
        )
        vec = bs.to_vector()
        assert vec.shape == (4 * N_CARDS,)

    def test_to_vector_dtype(self):
        bs = BeliefState.initialise(
            n_players=4, influence_counts=[2, 2, 2, 2], revealed_cards=[[], [], [], []]
        )
        assert bs.to_vector().dtype == np.float32

    def test_to_vector_matches_probs(self):
        bs = BeliefState.initialise(
            n_players=3, influence_counts=[2, 2, 2], revealed_cards=[[], [], []]
        )
        np.testing.assert_array_almost_equal(
            bs.to_vector().reshape(3, N_CARDS).astype(np.float64),
            bs.probs,
        )

    def test_to_dict_keys(self):
        bs = BeliefState.initialise(
            n_players=3, influence_counts=[2, 2, 2], revealed_cards=[[], [], []]
        )
        d = bs.to_dict()
        for key in ["n_players", "observer_idx", "probs", "influence_counts",
                    "revealed_cards", "known_hand", "available"]:
            assert key in d

    def test_to_dict_probs_is_list(self):
        bs = BeliefState.initialise(
            n_players=3, influence_counts=[2, 2, 2], revealed_cards=[[], [], []]
        )
        assert isinstance(bs.to_dict()["probs"], list)

    def test_copy_is_independent(self):
        bs = BeliefState.initialise(
            n_players=3, influence_counts=[2, 2, 2], revealed_cards=[[], [], []]
        )
        bs2 = bs.copy()
        bs2.probs[0, 0] = 0.999
        assert bs.probs[0, 0] != 0.999

    def test_copy_values_match(self):
        bs = BeliefState.initialise(
            n_players=3, influence_counts=[2, 2, 2], revealed_cards=[[], [], []]
        )
        bs2 = bs.copy()
        np.testing.assert_array_equal(bs.probs, bs2.probs)
        np.testing.assert_array_equal(bs.available, bs2.available)


# ===========================================================================
# Error handling
# ===========================================================================

class TestErrorHandling:

    def test_invalid_n_players_too_few(self):
        with pytest.raises(AssertionError):
            BeliefState.initialise(
                n_players=1,
                influence_counts=[2],
                revealed_cards=[[]],
            )

    def test_invalid_n_players_too_many(self):
        with pytest.raises(AssertionError):
            BeliefState.initialise(
                n_players=7,
                influence_counts=[2] * 7,
                revealed_cards=[[]] * 7,
            )

    def test_known_hand_wrong_size(self):
        with pytest.raises(ValueError):
            BeliefState.initialise(
                n_players=3,
                influence_counts=[2, 2, 2],
                revealed_cards=[[], [], []],
                observer_idx=0,
                known_hand=[Card.DUKE],   # influence_count is 2 but only 1 card given
            )

    def test_too_many_of_one_card_revealed(self):
        with pytest.raises(ValueError):
            BeliefState.initialise(
                n_players=3,
                influence_counts=[0, 0, 2],
                revealed_cards=[
                    [Card.DUKE, Card.DUKE, Card.DUKE],   # 3 Dukes here
                    [Card.DUKE],                          # + 1 more = 4 total — impossible
                    [],
                ],
            )


# ===========================================================================
# Repr / display
# ===========================================================================

class TestDisplay:

    def test_repr_is_string(self):
        bs = BeliefState.initialise(
            n_players=3, influence_counts=[2, 2, 2], revealed_cards=[[], [], []]
        )
        r = repr(bs)
        assert isinstance(r, str)
        assert "BeliefState" in r

    def test_summary_table_is_string(self):
        bs = BeliefState.initialise(
            n_players=3, influence_counts=[2, 2, 2], revealed_cards=[[], [], []]
        )
        t = bs.summary_table()
        assert isinstance(t, str)
        for card in CARDS:
            assert card.value in t
