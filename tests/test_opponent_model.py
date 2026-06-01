"""
tests/test_opponent_model.py — Tests for OpponentModel (sub-goal 3.4).

Coverage:
  PlayerBluffStats  — prior values, record_bluff/honest/unchallenged,
                      credibility formula, confidence, repr
  OpponentModel     — initialisation, credibility(), bluff_rate(),
                      record_challenge_result, all_credibilities,
                      credibility_vector, copy, summary
  Integration       — BeliefUpdater with opponent_model shifts beliefs
                      differently for known bluffers vs honest players
  Behavioural       — many bluffs → low credibility; many honest → high
                      credibility; prior at game start is neutral
"""

from __future__ import annotations

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from belief.opponent_model import (
    OpponentModel,
    PlayerBluffStats,
    PRIOR_STRENGTH,
    MIN_CREDIBILITY,
    MAX_CREDIBILITY,
    DEFAULT_CREDIBILITY,
)
from belief.belief_updater import BeliefUpdater, CLAIM_CREDIBILITY
from belief.belief_state import BeliefState, CARDS, N_CARDS, CARD_TO_IDX
from belief.constraint_normaliser import ConstraintNormaliser
from coup.cards import Card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_belief(n_players=4, observer_idx=-1):
    return BeliefState.initialise(
        n_players=n_players,
        influence_counts=[2] * n_players,
        revealed_cards=[[] for _ in range(n_players)],
        observer_idx=observer_idx,
    )


def _model(n_players=4, observer_idx=-1):
    return OpponentModel(n_players=n_players, observer_idx=observer_idx)


# ===========================================================================
# PlayerBluffStats
# ===========================================================================

class TestPlayerBluffStats:

    def test_prior_alpha_beta(self):
        s = PlayerBluffStats(player_idx=0)
        assert s.alpha == PRIOR_STRENGTH
        assert s.beta  == PRIOR_STRENGTH

    def test_initial_bluff_rate_is_half(self):
        s = PlayerBluffStats(player_idx=0)
        assert abs(s.bluff_rate - 0.5) < 1e-9

    def test_initial_credibility_is_one(self):
        """With α = β, β/α = 1.0 — but clamped to DEFAULT_CREDIBILITY."""
        s = PlayerBluffStats(player_idx=0)
        assert abs(s.credibility - 1.0) < 1e-9

    def test_initial_confidence_is_zero(self):
        """No observations → confidence = 0."""
        s = PlayerBluffStats(player_idx=0)
        assert s.confidence == 0.0

    def test_record_bluff_raises_alpha(self):
        s = PlayerBluffStats(player_idx=0)
        s.record_bluff()
        assert s.alpha == PRIOR_STRENGTH + 1.0
        assert s.bluffs_caught == 1

    def test_record_honest_raises_beta(self):
        s = PlayerBluffStats(player_idx=0)
        s.record_honest()
        assert s.beta == PRIOR_STRENGTH + 1.0
        assert s.honest_proved == 1

    def test_record_unchallenged_increments_claim_only(self):
        s = PlayerBluffStats(player_idx=0)
        s.record_unchallenged_claim()
        assert s.claims_total == 1
        assert s.alpha == PRIOR_STRENGTH  # unchanged
        assert s.beta  == PRIOR_STRENGTH  # unchanged

    def test_bluff_rate_rises_after_bluffs(self):
        s = PlayerBluffStats(player_idx=0)
        for _ in range(10):
            s.record_bluff()
        assert s.bluff_rate > 0.7

    def test_bluff_rate_falls_after_honest(self):
        s = PlayerBluffStats(player_idx=0)
        for _ in range(10):
            s.record_honest()
        assert s.bluff_rate < 0.3

    def test_credibility_low_for_frequent_bluffer(self):
        s = PlayerBluffStats(player_idx=0)
        for _ in range(10):
            s.record_bluff()
        assert s.credibility < 1.0

    def test_credibility_high_for_honest_player(self):
        s = PlayerBluffStats(player_idx=0)
        for _ in range(10):
            s.record_honest()
        assert s.credibility > 1.0

    def test_credibility_clamped_to_min(self):
        s = PlayerBluffStats(player_idx=0)
        for _ in range(100):
            s.record_bluff()
        assert s.credibility >= MIN_CREDIBILITY

    def test_credibility_clamped_to_max(self):
        s = PlayerBluffStats(player_idx=0)
        for _ in range(100):
            s.record_honest()
        assert s.credibility <= MAX_CREDIBILITY

    def test_confidence_grows_with_observations(self):
        s = PlayerBluffStats(player_idx=0)
        c0 = s.confidence
        s.record_bluff()
        c1 = s.confidence
        s.record_honest()
        c2 = s.confidence
        assert c0 < c1 < c2

    def test_confidence_approaches_one_with_many_obs(self):
        s = PlayerBluffStats(player_idx=0)
        for _ in range(100):
            s.record_bluff()
        assert s.confidence > 0.95

    def test_to_dict_has_all_keys(self):
        s = PlayerBluffStats(player_idx=1)
        d = s.to_dict()
        for key in ["player_idx", "alpha", "beta", "bluff_rate",
                    "credibility", "confidence", "claims_total",
                    "bluffs_caught", "honest_proved"]:
            assert key in d

    def test_repr_is_string(self):
        s = PlayerBluffStats(player_idx=2)
        assert "PlayerBluffStats" in repr(s)


# ===========================================================================
# OpponentModel — initialisation and basic properties
# ===========================================================================

class TestOpponentModelInit:

    def test_n_players_stored(self):
        m = _model(n_players=5)
        assert m.n_players == 5

    def test_observer_idx_stored(self):
        m = _model(observer_idx=2)
        assert m.observer_idx == 2

    def test_initial_credibility_is_one_for_all(self):
        m = _model(n_players=4)
        for i in range(4):
            assert abs(m.credibility(i) - 1.0) < 1e-9

    def test_initial_bluff_rate_is_half(self):
        m = _model(n_players=4)
        for i in range(4):
            assert abs(m.bluff_rate(i) - 0.5) < 1e-9

    def test_observer_returns_default_credibility(self):
        m = _model(observer_idx=0)
        assert m.credibility(0) == DEFAULT_CREDIBILITY

    def test_all_credibilities_shape(self):
        m = _model(n_players=4)
        creds = m.all_credibilities()
        assert creds.shape == (4,)
        assert np.all(creds >= MIN_CREDIBILITY)
        assert np.all(creds <= MAX_CREDIBILITY)

    def test_credibility_vector_in_unit_interval(self):
        m = _model(n_players=4)
        v = m.credibility_vector()
        assert v.shape == (4,)
        assert np.all(v >= 0.0)
        assert np.all(v <= 1.0)


# ===========================================================================
# OpponentModel — recording events
# ===========================================================================

class TestOpponentModelRecording:

    def test_record_bluff_lowers_credibility(self):
        m = _model(n_players=4)
        initial = m.credibility(1)
        m.record_challenge_result(actor_idx=1, actor_won=False)
        assert m.credibility(1) < initial

    def test_record_honest_raises_credibility(self):
        m = _model(n_players=4)
        initial = m.credibility(1)
        m.record_challenge_result(actor_idx=1, actor_won=True)
        assert m.credibility(1) > initial

    def test_observer_not_updated(self):
        m = _model(n_players=4, observer_idx=0)
        initial_cred = m.credibility(0)
        m.record_challenge_result(actor_idx=0, actor_won=False)
        assert m.credibility(0) == initial_cred

    def test_other_players_unaffected(self):
        m = _model(n_players=4)
        cred_p2_before = m.credibility(2)
        m.record_challenge_result(actor_idx=1, actor_won=False)
        assert abs(m.credibility(2) - cred_p2_before) < 1e-9

    def test_many_bluffs_drive_credibility_to_min(self):
        m = _model(n_players=4)
        for _ in range(50):
            m.record_challenge_result(actor_idx=1, actor_won=False)
        assert m.credibility(1) == MIN_CREDIBILITY

    def test_many_honest_drive_credibility_to_max(self):
        m = _model(n_players=4)
        for _ in range(50):
            m.record_challenge_result(actor_idx=1, actor_won=True)
        assert m.credibility(1) == MAX_CREDIBILITY

    def test_mixed_record_settles_near_truth(self):
        """70% bluff rate → credibility should be below 1.0 (more bluffing than honest)."""
        m = _model(n_players=4)
        for _ in range(7):
            m.record_challenge_result(actor_idx=2, actor_won=False)
        for _ in range(3):
            m.record_challenge_result(actor_idx=2, actor_won=True)
        assert m.credibility(2) < 1.0

    def test_record_unchallenged_claim(self):
        m = _model(n_players=4)
        initial_alpha = m.stats(1).alpha
        m.record_unchallenged_claim(actor_idx=1)
        assert m.stats(1).claims_total == 1
        assert m.stats(1).alpha == initial_alpha  # not updated


# ===========================================================================
# OpponentModel — copy and serialisation
# ===========================================================================

class TestOpponentModelCopyAndSerial:

    def test_copy_is_independent(self):
        m = _model(n_players=4)
        m.record_challenge_result(actor_idx=1, actor_won=False)
        m2 = m.copy()
        m2.record_challenge_result(actor_idx=1, actor_won=True)
        # Original should not be affected
        assert m.stats(1).honest_proved == 0
        assert m2.stats(1).honest_proved == 1

    def test_copy_values_match(self):
        m = _model(n_players=4)
        m.record_challenge_result(actor_idx=2, actor_won=False)
        m2 = m.copy()
        assert m2.credibility(2) == m.credibility(2)
        assert m2.bluff_rate(2) == m.bluff_rate(2)

    def test_to_dict_has_required_keys(self):
        m = _model(n_players=3)
        d = m.to_dict()
        assert "n_players"    in d
        assert "observer_idx" in d
        assert "players"      in d
        assert len(d["players"]) == 3

    def test_summary_is_string(self):
        m = _model(n_players=3)
        s = m.summary()
        assert isinstance(s, str)
        assert "BluffRate" in s
        assert "Credibility" in s

    def test_repr_is_string(self):
        m = _model(n_players=3)
        assert "OpponentModel" in repr(m)


# ===========================================================================
# Integration: BeliefUpdater with OpponentModel
# ===========================================================================

class TestBeliefUpdaterIntegration:

    def test_no_model_uses_flat_credibility(self):
        """Without an opponent model, CLAIM_CREDIBILITY is used for all players."""
        bs1 = _fresh_belief(observer_idx=-1)
        bs2 = _fresh_belief(observer_idx=-1)
        upd_no_model = BeliefUpdater(n_players=4, opponent_model=None)
        upd_with_flat = BeliefUpdater(n_players=4, opponent_model=None)

        upd_no_model.update(bs1,  "action", actor_idx=1, claimed_card=Card.DUKE)
        upd_with_flat.update(bs2, "action", actor_idx=1, claimed_card=Card.DUKE)

        np.testing.assert_array_almost_equal(bs1.probs, bs2.probs)

    def test_model_updates_on_challenge_result(self):
        """BeliefUpdater should call model.record_challenge_result automatically."""
        m = _model(n_players=4)
        upd = BeliefUpdater(n_players=4, opponent_model=m)
        bs = _fresh_belief(observer_idx=-1)

        initial_cred = m.credibility(1)
        upd.update(bs, "challenge_result",
                   actor_idx=1, claimed_card=Card.DUKE,
                   challenger_idx=2, actor_won=False)
        assert m.credibility(1) < initial_cred

    def test_known_bluffer_has_weaker_claim_effect(self):
        """
        After catching P1 bluffing, P1's Duke claim should shift beliefs
        less than an uncaught player's claim.
        """
        c_idx = CARD_TO_IDX[Card.DUKE]

        # Setup: two fresh states and one model
        bs_honest = _fresh_belief(observer_idx=-1)
        bs_bluffer = _fresh_belief(observer_idx=-1)

        m = _model(n_players=4)
        # Record 5 bluffs for P1
        for _ in range(5):
            m.record_challenge_result(actor_idx=1, actor_won=False)

        upd_honest  = BeliefUpdater(n_players=4, opponent_model=None)
        upd_bluffer = BeliefUpdater(n_players=4, opponent_model=m)

        # Both P1s claim Duke
        upd_honest.update(bs_honest,   "action", actor_idx=1, claimed_card=Card.DUKE)
        upd_bluffer.update(bs_bluffer, "action", actor_idx=1, claimed_card=Card.DUKE)

        # Bluffer's Duke prob should be lower (claim was less credible)
        assert bs_bluffer.probs[1, c_idx] <= bs_honest.probs[1, c_idx]

    def test_proven_honest_has_stronger_claim_effect(self):
        """
        After proving honesty 5 times, P1's Duke claim should raise their
        Duke prob more than an uncaught player's claim.
        """
        c_idx = CARD_TO_IDX[Card.DUKE]

        bs_neutral = _fresh_belief(observer_idx=-1)
        bs_honest  = _fresh_belief(observer_idx=-1)

        m = _model(n_players=4)
        for _ in range(5):
            m.record_challenge_result(actor_idx=1, actor_won=True)

        upd_neutral = BeliefUpdater(n_players=4, opponent_model=None)
        upd_honest  = BeliefUpdater(n_players=4, opponent_model=m)

        upd_neutral.update(bs_neutral, "action", actor_idx=1, claimed_card=Card.DUKE)
        upd_honest.update(bs_honest,   "action", actor_idx=1, claimed_card=Card.DUKE)

        assert bs_honest.probs[1, c_idx] >= bs_neutral.probs[1, c_idx]

    def test_beliefs_stay_valid_with_model(self):
        """All probs stay in [0,1] and deck constraints hold when model is active."""
        m = _model(n_players=4)
        upd = BeliefUpdater(n_players=4, opponent_model=m)
        bs = _fresh_belief(observer_idx=-1)
        normaliser = ConstraintNormaliser()

        events = [
            {"event_type": "action",           "actor_idx": 0, "claimed_card": Card.DUKE},
            {"event_type": "challenge_result",  "actor_idx": 0, "claimed_card": Card.DUKE,
             "challenger_idx": 1, "actor_won": False},
            {"event_type": "influence_loss",    "player_idx": 0, "card_lost": Card.CAPTAIN},
            {"event_type": "action",            "actor_idx": 0, "claimed_card": Card.DUKE},
            {"event_type": "block",             "blocker_idx": 2, "blocking_card": Card.DUKE},
            {"event_type": "action",            "actor_idx": 1, "claimed_card": Card.CAPTAIN},
            {"event_type": "challenge_result",  "actor_idx": 1, "claimed_card": Card.CAPTAIN,
             "challenger_idx": 3, "actor_won": True},
        ]
        for evt in events:
            upd.update(bs, **evt)
            assert np.all(bs.probs >= 0.0)
            assert np.all(bs.probs <= 1.0 + 1e-9)
            assert not np.any(np.isnan(bs.probs))
            assert normaliser.is_consistent(bs)

    def test_model_credibility_monotone_with_bluff_count(self):
        """More bluffs → strictly lower credibility (until clamped at MIN)."""
        creds = []
        for n_bluffs in range(0, 15):
            m = _model(n_players=4)
            for _ in range(n_bluffs):
                m.record_challenge_result(actor_idx=1, actor_won=False)
            creds.append(m.credibility(1))

        # Should be non-increasing
        for i in range(len(creds) - 1):
            assert creds[i] >= creds[i + 1] - 1e-9

    def test_model_credibility_monotone_with_honest_count(self):
        """More honest proofs → non-decreasing credibility (until clamped at MAX)."""
        creds = []
        for n_honest in range(0, 15):
            m = _model(n_players=4)
            for _ in range(n_honest):
                m.record_challenge_result(actor_idx=2, actor_won=True)
            creds.append(m.credibility(2))

        for i in range(len(creds) - 1):
            assert creds[i] <= creds[i + 1] + 1e-9


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_single_bluff_then_many_honest(self):
        """One early bluff shouldn't permanently taint a player with many proofs."""
        m = _model(n_players=4)
        m.record_challenge_result(actor_idx=1, actor_won=False)  # one bluff
        for _ in range(20):
            m.record_challenge_result(actor_idx=1, actor_won=True)  # many honest
        # Should recover to high credibility
        assert m.credibility(1) > 1.5

    def test_2_player_game(self):
        m = OpponentModel(n_players=2, observer_idx=0)
        m.record_challenge_result(actor_idx=1, actor_won=False)
        assert m.credibility(1) < 1.0

    def test_6_player_game(self):
        m = OpponentModel(n_players=6, observer_idx=3)
        creds = m.all_credibilities()
        assert creds.shape == (6,)

    def test_stats_accessor(self):
        m = _model(n_players=4)
        m.record_challenge_result(actor_idx=2, actor_won=False)
        s = m.stats(2)
        assert isinstance(s, PlayerBluffStats)
        assert s.bluffs_caught == 1
