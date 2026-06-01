"""
tests/test_constraint_normaliser.py — Tests for ConstraintNormaliser (sub-goal 3.3).

Coverage:
  Core invariants     — probs stay in [0,1]; column sums ≤ available[c]
  Column pass         — overshoot detection and scaling
  Row pass            — prior ceiling enforcement
  Pin mask            — observer row, dead player row, hard-zero cells untouched
  Convergence         — returns iteration count; converges within max_iterations
  Integration with    — auto_normalise=True in BeliefUpdater keeps state
  BeliefUpdater         consistent after every event
  Stress tests        — adversarial belief states with extreme values
"""

from __future__ import annotations

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
    COPIES_PER_CARD,
    TOTAL_CARDS,
    hypergeometric_at_least_one,
)
from belief.belief_updater import BeliefUpdater
from belief.constraint_normaliser import (
    ConstraintNormaliser,
    normalise,
    DEFAULT_MAX_ITERATIONS,
)
from coup.cards import Card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh(n_players=4, observer_idx=-1, known_hand=None,
           influence_counts=None, revealed_cards=None):
    if influence_counts is None:
        influence_counts = [2] * n_players
    if revealed_cards is None:
        revealed_cards = [[] for _ in range(n_players)]
    return BeliefState.initialise(
        n_players=n_players,
        influence_counts=influence_counts,
        revealed_cards=revealed_cards,
        observer_idx=observer_idx,
        known_hand=known_hand,
    )


def _overshot(n_players=4, card=Card.DUKE, overvalue=0.99):
    """
    Build a BeliefState with artificially inflated Duke probabilities so
    the column sum far exceeds available[Duke] = 3.
    """
    bs = _fresh(n_players=n_players)
    c_idx = CARD_TO_IDX[card]
    for p_idx in range(n_players):
        bs.probs[p_idx, c_idx] = overvalue
    return bs


# ===========================================================================
# Core invariants after normalisation
# ===========================================================================

class TestCoreInvariants:

    def test_probs_non_negative_after_normalise(self):
        bs = _overshot()
        normalise(bs)
        assert np.all(bs.probs >= 0.0)

    def test_probs_at_most_one_after_normalise(self):
        bs = _overshot()
        normalise(bs)
        assert np.all(bs.probs <= 1.0 + 1e-9)

    def test_no_nan_after_normalise(self):
        bs = _overshot()
        normalise(bs)
        assert not np.any(np.isnan(bs.probs))

    def test_no_inf_after_normalise(self):
        bs = _overshot()
        normalise(bs)
        assert not np.any(np.isinf(bs.probs))

    def test_column_sums_at_most_available_after_normalise(self):
        bs = _overshot(n_players=4, overvalue=0.99)
        normaliser = ConstraintNormaliser()
        normaliser.normalise(bs)
        violations = normaliser.conservation_violations(bs)
        assert np.all(violations <= 1e-6), \
            f"Conservation violated: {violations}"

    def test_is_consistent_after_normalise(self):
        bs = _overshot()
        normaliser = ConstraintNormaliser()
        normaliser.normalise(bs)
        assert normaliser.is_consistent(bs)

    @pytest.mark.parametrize("n_players", [2, 3, 4, 5, 6])
    def test_all_player_counts_consistent_after_normalise(self, n_players):
        bs = _overshot(n_players=n_players)
        normaliser = ConstraintNormaliser()
        normaliser.normalise(bs)
        assert normaliser.is_consistent(bs)

    def test_already_consistent_state_unchanged(self):
        """A fresh state from initialise() is already consistent — normalise should not change it."""
        bs = _fresh()
        probs_before = bs.probs.copy()
        normalise(bs)
        np.testing.assert_array_almost_equal(bs.probs, probs_before, decimal=6)


# ===========================================================================
# Column pass — card conservation
# ===========================================================================

class TestColumnPass:

    def test_column_sum_reduced_to_available(self):
        """If Duke column sums to 3.96, normalise should bring it to ≤ 3.0."""
        bs = _fresh(n_players=4, observer_idx=-1)
        c_idx = CARD_TO_IDX[Card.DUKE]
        for p_idx in range(4):
            bs.probs[p_idx, c_idx] = 0.99   # sum = 3.96 > 3.0

        normaliser = ConstraintNormaliser()
        normaliser.normalise(bs)
        col_sum = normaliser.column_sums(bs)[c_idx]
        assert col_sum <= 3.0 + 1e-6

    def test_cards_not_in_violation_unchanged(self):
        """Cards whose column sums are within bounds should not be touched."""
        bs = _fresh(n_players=4, observer_idx=-1)
        c_duke = CARD_TO_IDX[Card.DUKE]
        c_cap  = CARD_TO_IDX[Card.CAPTAIN]

        # Inflate Duke only
        for p_idx in range(4):
            bs.probs[p_idx, c_duke] = 0.99

        captain_col_before = bs.probs[:, c_cap].copy()
        normalise(bs)
        # Captain column should be essentially unchanged
        np.testing.assert_array_almost_equal(
            bs.probs[:, c_cap], captain_col_before, decimal=5
        )

    def test_all_five_cards_conserved(self):
        """Inflate all 5 cards and verify all columns are brought into range."""
        bs = _fresh(n_players=4, observer_idx=-1)
        bs.probs[:, :] = 0.99  # all columns over budget

        normaliser = ConstraintNormaliser()
        normaliser.normalise(bs)
        violations = normaliser.conservation_violations(bs)
        assert np.all(violations <= 1e-6)

    def test_zero_available_card_zeroed(self):
        """If all copies of a card are revealed, available=0, column must be 0."""
        bs = BeliefState.initialise(
            n_players=3,
            influence_counts=[0, 0, 2],
            revealed_cards=[[Card.DUKE], [Card.DUKE, Card.DUKE], []],
            observer_idx=-1,
        )
        # Artificially inflate Duke prob for P2
        bs.probs[2, CARD_TO_IDX[Card.DUKE]] = 0.5
        normalise(bs)
        assert bs.probs[2, CARD_TO_IDX[Card.DUKE]] <= 1e-6


# ===========================================================================
# Row pass — prior ceiling enforcement
# ===========================================================================

class TestRowPass:

    def test_row_values_do_not_exceed_prior_ceiling(self):
        """
        After normalisation, no free cell should exceed its hypergeometric prior.
        A player can't be more likely to hold a card than the prior allows.
        """
        bs = _fresh(n_players=4, observer_idx=-1)
        # Manually set all values to 1.0 (many will exceed the prior ceiling)
        bs.probs[:, :] = 1.0

        normaliser = ConstraintNormaliser()
        ceiling = normaliser._build_prior_ceiling(bs)
        normaliser.normalise(bs)

        for p_idx in range(4):
            for c_idx in range(N_CARDS):
                assert bs.probs[p_idx, c_idx] <= ceiling[p_idx, c_idx] + 1e-6, \
                    f"P{p_idx} card {c_idx}: {bs.probs[p_idx,c_idx]:.4f} > ceiling {ceiling[p_idx,c_idx]:.4f}"

    def test_dead_player_row_stays_zero(self):
        bs = _fresh(
            n_players=4,
            influence_counts=[2, 0, 2, 2],
            revealed_cards=[[], [Card.DUKE, Card.CAPTAIN], [], []],
        )
        # Inflate to create violations
        bs.probs[:, :] = 0.9
        normalise(bs)
        np.testing.assert_array_equal(bs.probs[1], np.zeros(N_CARDS))


# ===========================================================================
# Pin mask — hard cells must not be modified
# ===========================================================================

class TestPinMask:

    def test_observer_known_hand_row_untouched(self):
        """Observer's certainty row must survive normalisation unchanged."""
        bs = _fresh(
            n_players=4,
            observer_idx=0,
            known_hand=[Card.DUKE, Card.ASSASSIN],
        )
        bs.probs[:, :] = 0.9   # inflate to trigger column pass
        # Restore own row to certainty values
        bs.probs[0, CARD_TO_IDX[Card.DUKE]]     = 1.0
        bs.probs[0, CARD_TO_IDX[Card.ASSASSIN]] = 1.0
        bs.probs[0, CARD_TO_IDX[Card.CONTESSA]]  = 0.0
        bs.probs[0, CARD_TO_IDX[Card.CAPTAIN]]   = 0.0
        bs.probs[0, CARD_TO_IDX[Card.AMBASSADOR]] = 0.0

        own_row_before = bs.probs[0].copy()
        normalise(bs)
        np.testing.assert_array_equal(bs.probs[0], own_row_before)

    def test_hard_zero_cell_stays_zero(self):
        """
        A cell set to exactly 0.0 by bluff detection (hard evidence that player
        doesn't hold card) must not be raised above 0 by normalisation.
        """
        bs = _fresh(n_players=4, observer_idx=-1)
        c_idx = CARD_TO_IDX[Card.DUKE]
        # Simulate bluff detection: P1 proved not to hold Duke
        bs.probs[1, c_idx] = 0.0

        # Inflate other cards to trigger column scaling (which could try to scale up)
        for p_idx in [0, 2, 3]:
            bs.probs[p_idx, c_idx] = 0.5

        normalise(bs)
        assert bs.probs[1, c_idx] == 0.0, \
            f"Hard-zero cell was raised to {bs.probs[1, c_idx]}"

    def test_dead_player_all_cells_pinned(self):
        bs = _fresh(
            n_players=4,
            influence_counts=[2, 0, 2, 2],
            revealed_cards=[[], [Card.DUKE, Card.CAPTAIN], [], []],
        )
        # Manually set dead player's probs to non-zero (simulate corruption)
        bs.probs[1, :] = 0.5
        normalise(bs)
        np.testing.assert_array_equal(bs.probs[1], np.zeros(N_CARDS))

    def test_pinned_cells_not_scaled_in_column_pass(self):
        """
        Pinned cells in a column should not reduce the available budget for
        free cells when they're at zero.
        """
        bs = _fresh(n_players=4, observer_idx=-1)
        c_idx = CARD_TO_IDX[Card.CAPTAIN]
        # P0 has hard-zero for Captain (bluff detected)
        bs.probs[0, c_idx] = 0.0
        # Others have moderate probability
        bs.probs[1, c_idx] = 0.7
        bs.probs[2, c_idx] = 0.7
        bs.probs[3, c_idx] = 0.7   # sum of free = 2.1, cap = 3 → no violation

        normaliser = ConstraintNormaliser()
        normaliser.normalise(bs)
        # P0's zero should be preserved
        assert bs.probs[0, c_idx] == 0.0
        # Others' values should remain reasonable
        assert bs.probs[1, c_idx] > 0.0


# ===========================================================================
# Convergence
# ===========================================================================

class TestConvergence:

    def test_returns_iteration_count(self):
        bs = _overshot()
        normaliser = ConstraintNormaliser()
        n_iter = normaliser.normalise(bs)
        assert isinstance(n_iter, int)
        assert 1 <= n_iter <= DEFAULT_MAX_ITERATIONS

    def test_converges_within_max_iterations(self):
        """Even heavily violated states should converge within the limit."""
        bs = _overshot(overvalue=1.0)
        normaliser = ConstraintNormaliser(max_iterations=50)
        n_iter = normaliser.normalise(bs)
        assert n_iter < 50
        assert normaliser.is_consistent(bs)

    def test_fresh_state_converges_in_one_iteration(self):
        """A fresh hypergeometric prior is already consistent — one pass suffices."""
        bs = _fresh()
        normaliser = ConstraintNormaliser()
        n_iter = normaliser.normalise(bs)
        assert n_iter == 1

    def test_custom_convergence_tolerance(self):
        bs = _overshot()
        normaliser_tight = ConstraintNormaliser(convergence_tol=1e-10)
        normaliser_loose = ConstraintNormaliser(convergence_tol=1e-3)
        n_tight = normaliser_tight.normalise(bs.copy())
        n_loose = normaliser_loose.normalise(bs)
        # Tight tolerance should require more iterations
        assert n_tight >= n_loose


# ===========================================================================
# conservation_violations and column_sums diagnostics
# ===========================================================================

class TestDiagnostics:

    def test_column_sums_shape(self):
        bs = _fresh()
        normaliser = ConstraintNormaliser()
        sums = normaliser.column_sums(bs)
        assert sums.shape == (N_CARDS,)

    def test_column_sums_non_negative(self):
        bs = _fresh()
        normaliser = ConstraintNormaliser()
        assert np.all(normaliser.column_sums(bs) >= 0.0)

    def test_conservation_violations_zero_after_normalise(self):
        bs = _overshot()
        normaliser = ConstraintNormaliser()
        normaliser.normalise(bs)
        violations = normaliser.conservation_violations(bs)
        assert np.all(violations <= 1e-6)

    def test_conservation_violations_positive_before_normalise(self):
        """An overshot state should have positive violations before normalisation."""
        bs = _overshot(overvalue=0.99)
        normaliser = ConstraintNormaliser()
        violations = normaliser.conservation_violations(bs)
        assert np.any(violations > 0.01)

    def test_is_consistent_true_for_fresh_state(self):
        bs = _fresh()
        assert ConstraintNormaliser().is_consistent(bs)

    def test_is_consistent_false_for_overshot_state(self):
        bs = _overshot(overvalue=0.99)
        assert not ConstraintNormaliser().is_consistent(bs)


# ===========================================================================
# Integration: BeliefUpdater with auto_normalise=True
# ===========================================================================

class TestIntegrationWithUpdater:

    def _run_and_check(self, n_players, events):
        """Run a sequence of events and verify consistency after each one."""
        bs = _fresh(n_players=n_players, observer_idx=-1)
        upd = BeliefUpdater(n_players=n_players, auto_normalise=True)
        normaliser = ConstraintNormaliser()

        for event_kwargs in events:
            upd.update(bs, **event_kwargs)
            assert normaliser.is_consistent(bs), \
                f"Inconsistent after {event_kwargs['event_type']}: " \
                f"{normaliser.conservation_violations(bs)}"
            assert np.all(bs.probs >= 0.0)
            assert np.all(bs.probs <= 1.0 + 1e-9)

    def test_consistent_after_multiple_claims(self):
        self._run_and_check(4, [
            {"event_type": "action", "actor_idx": 0, "claimed_card": Card.DUKE},
            {"event_type": "action", "actor_idx": 1, "claimed_card": Card.DUKE},
            {"event_type": "action", "actor_idx": 2, "claimed_card": Card.DUKE},
            {"event_type": "action", "actor_idx": 3, "claimed_card": Card.DUKE},
        ])

    def test_consistent_after_influence_losses(self):
        self._run_and_check(4, [
            {"event_type": "influence_loss", "player_idx": 0, "card_lost": Card.DUKE},
            {"event_type": "influence_loss", "player_idx": 1, "card_lost": Card.CAPTAIN},
            {"event_type": "influence_loss", "player_idx": 0, "card_lost": Card.ASSASSIN},
        ])

    def test_consistent_after_bluff_detected(self):
        self._run_and_check(4, [
            {"event_type": "action",           "actor_idx": 1, "claimed_card": Card.DUKE},
            {"event_type": "challenge_result", "actor_idx": 1, "claimed_card": Card.DUKE,
             "challenger_idx": 2, "actor_won": False},
            {"event_type": "influence_loss",   "player_idx": 1, "card_lost": Card.CAPTAIN},
        ])

    def test_consistent_after_exchange(self):
        self._run_and_check(4, [
            {"event_type": "action",   "actor_idx": 2, "claimed_card": Card.AMBASSADOR},
            {"event_type": "exchange", "actor_idx": 2},
        ])

    def test_consistent_after_long_game(self):
        """Simulate a full short game sequence and verify consistency throughout."""
        import random
        rng = random.Random(42)
        n = 4
        bs = _fresh(n_players=n, observer_idx=-1)
        upd = BeliefUpdater(n_players=n, auto_normalise=True)
        normaliser = ConstraintNormaliser()

        event_types = [
            {"event_type": "action", "actor_idx": 0, "claimed_card": Card.DUKE},
            {"event_type": "action", "actor_idx": 1, "claimed_card": Card.CAPTAIN},
            {"event_type": "block",  "blocker_idx": 2, "blocking_card": Card.DUKE},
            {"event_type": "challenge_result", "actor_idx": 0, "claimed_card": Card.DUKE,
             "challenger_idx": 3, "actor_won": True},
            {"event_type": "influence_loss", "player_idx": 3, "card_lost": Card.CONTESSA},
            {"event_type": "action", "actor_idx": 2, "claimed_card": Card.AMBASSADOR},
            {"event_type": "exchange", "actor_idx": 2},
            {"event_type": "action", "actor_idx": 1, "claimed_card": Card.ASSASSIN},
            {"event_type": "challenge_result", "actor_idx": 1, "claimed_card": Card.ASSASSIN,
             "challenger_idx": 0, "actor_won": False},
            {"event_type": "influence_loss", "player_idx": 1, "card_lost": Card.DUKE},
        ]

        for evt in event_types:
            upd.update(bs, **evt)
            assert normaliser.is_consistent(bs), \
                f"Violated after {evt['event_type']}"


# ===========================================================================
# Stress tests — adversarial belief states
# ===========================================================================

class TestStress:

    def test_all_probs_set_to_one(self):
        """All probs = 1.0 is the worst possible violation — should still recover."""
        bs = _fresh(n_players=6, observer_idx=-1)
        bs.probs[:, :] = 1.0
        normaliser = ConstraintNormaliser(max_iterations=50)
        normaliser.normalise(bs)
        assert normaliser.is_consistent(bs)
        assert np.all(bs.probs >= 0.0)

    def test_single_player_inflated(self):
        bs = _fresh(n_players=4, observer_idx=-1)
        bs.probs[0, :] = 1.0   # P0 thinks they hold every card
        normalise(bs)
        assert np.all(bs.probs >= 0.0)
        assert np.all(bs.probs <= 1.0 + 1e-9)
        assert ConstraintNormaliser().is_consistent(bs)

    def test_two_player_game_stress(self):
        bs = _fresh(n_players=2, observer_idx=-1)
        bs.probs[:, :] = 0.99
        normalise(bs)
        assert ConstraintNormaliser().is_consistent(bs)

    def test_late_game_two_players_one_card_each(self):
        """
        Late game: 2 players left, each with 1 influence, several cards revealed.
        Constraint should still be satisfied after normalise.
        """
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[0, 1, 0, 1],
            revealed_cards=[
                [Card.DUKE, Card.CAPTAIN],
                [Card.CONTESSA],
                [Card.DUKE, Card.ASSASSIN],
                [Card.AMBASSADOR],
            ],
            observer_idx=-1,
        )
        # Manually inflate remaining players
        bs.probs[1, :] = 0.8
        bs.probs[3, :] = 0.8
        normalise(bs)
        assert ConstraintNormaliser().is_consistent(bs)

    def test_normalise_idempotent(self):
        """Calling normalise twice should produce the same result as calling it once."""
        bs = _overshot()
        normalise(bs)
        probs_after_first = bs.probs.copy()
        normalise(bs)
        np.testing.assert_array_almost_equal(bs.probs, probs_after_first, decimal=8)
