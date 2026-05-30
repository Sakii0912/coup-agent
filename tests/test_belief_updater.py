"""
tests/test_belief_updater.py — Tests for BeliefUpdater (sub-goal 3.2).

Coverage:
  Dispatch       — unknown event types are no-ops; missing kwargs are no-ops
  Claim update   — actor prob rises, others fall; own hand unchanged;
                   dead player unchanged; probs stay in [0, 1]
  Block update   — identical mechanics to claim update
  Challenge win  — actor row reset to prior after card swap
  Challenge loss — actor's card prob set to 0; remaining probs redistributed
  Influence loss — card removed from available; influence_count decremented;
                   all rows recomputed; dead player zeroed
  Exchange       — actor row reset to prior; observer's row untouched
  Sequence       — multi-event sequences leave state consistent
"""

from __future__ import annotations

import sys
import os
import copy
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
from belief.belief_updater import BeliefUpdater, CLAIM_CREDIBILITY, _current_pool_size
from coup.cards import Card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh(n_players=4, observer_idx=0, known_hand=None, influence_counts=None,
           revealed_cards=None):
    """Create a clean BeliefState for testing."""
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


def _updater(n_players=4):
    return BeliefUpdater(n_players=n_players)


# ===========================================================================
# Dispatch — no-ops
# ===========================================================================

class TestDispatchNoOps:

    def test_unknown_event_type_is_noop(self):
        bs = _fresh()
        before = bs.probs.copy()
        _updater().update(bs, "income")
        np.testing.assert_array_equal(bs.probs, before)

    def test_action_without_claimed_card_is_noop(self):
        bs = _fresh()
        before = bs.probs.copy()
        _updater().update(bs, "action", actor_idx=1, claimed_card=None)
        np.testing.assert_array_equal(bs.probs, before)

    def test_challenge_result_without_card_is_noop(self):
        bs = _fresh()
        before = bs.probs.copy()
        _updater().update(bs, "challenge_result", actor_idx=1, claimed_card=None, actor_won=True)
        np.testing.assert_array_equal(bs.probs, before)

    def test_influence_loss_without_player_is_noop(self):
        bs = _fresh()
        before = bs.probs.copy()
        _updater().update(bs, "influence_loss", player_idx=None, card_lost=Card.DUKE)
        np.testing.assert_array_equal(bs.probs, before)

    def test_game_start_is_noop(self):
        bs = _fresh()
        before = bs.probs.copy()
        _updater().update(bs, "game_start")
        np.testing.assert_array_equal(bs.probs, before)

    def test_action_resolved_is_noop(self):
        bs = _fresh()
        before = bs.probs.copy()
        _updater().update(bs, "action_resolved", actor_idx=1)
        np.testing.assert_array_equal(bs.probs, before)


# ===========================================================================
# Claim update (action)
# ===========================================================================

class TestClaimUpdate:

    def test_actor_prob_increases_for_claimed_card(self):
        bs = _fresh(observer_idx=-1)
        c_idx = CARD_TO_IDX[Card.DUKE]
        before = bs.probs[1, c_idx]
        _updater().update(bs, "action", actor_idx=1, claimed_card=Card.DUKE)
        assert bs.probs[1, c_idx] >= before

    def test_other_players_prob_decreases_for_claimed_card(self):
        bs = _fresh(observer_idx=-1)
        c_idx = CARD_TO_IDX[Card.DUKE]
        before_p2 = bs.probs[2, c_idx]
        before_p3 = bs.probs[3, c_idx]
        _updater().update(bs, "action", actor_idx=1, claimed_card=Card.DUKE)
        assert bs.probs[2, c_idx] <= before_p2
        assert bs.probs[3, c_idx] <= before_p3

    def test_probs_stay_in_unit_interval_after_claim(self):
        bs = _fresh(observer_idx=-1)
        for _ in range(5):
            _updater().update(bs, "action", actor_idx=1, claimed_card=Card.DUKE)
        assert np.all(bs.probs >= 0.0)
        assert np.all(bs.probs <= 1.0)

    def test_dead_player_unaffected_by_claim(self):
        bs = _fresh(
            influence_counts=[2, 0, 2, 2],
            revealed_cards=[[], [Card.DUKE, Card.CAPTAIN], [], []],
            observer_idx=-1,
        )
        before_dead = bs.probs[1].copy()
        _updater().update(bs, "action", actor_idx=0, claimed_card=Card.DUKE)
        np.testing.assert_array_equal(bs.probs[1], before_dead)

    def test_known_hand_row_untouched_by_claim(self):
        bs = _fresh(observer_idx=0, known_hand=[Card.DUKE, Card.ASSASSIN])
        before_own = bs.probs[0].copy()
        _updater().update(bs, "action", actor_idx=1, claimed_card=Card.DUKE)
        np.testing.assert_array_equal(bs.probs[0], before_own)

    def test_no_nan_after_claim(self):
        bs = _fresh(observer_idx=-1)
        _updater().update(bs, "action", actor_idx=2, claimed_card=Card.CAPTAIN)
        assert not np.any(np.isnan(bs.probs))

    def test_unclaimed_cards_unaffected(self):
        bs = _fresh(observer_idx=-1)
        c_duke = CARD_TO_IDX[Card.DUKE]
        c_assn = CARD_TO_IDX[Card.ASSASSIN]
        before_assn = bs.probs[:, c_assn].copy()
        _updater().update(bs, "action", actor_idx=1, claimed_card=Card.DUKE)
        # Assassin column should be unchanged (claim was for Duke only)
        np.testing.assert_array_almost_equal(bs.probs[:, c_assn], before_assn)

    def test_actor_claims_own_card_is_noop(self):
        """Observer claiming their own known card should not change anything."""
        bs = _fresh(observer_idx=0, known_hand=[Card.DUKE, Card.ASSASSIN])
        before = bs.probs.copy()
        _updater().update(bs, "action", actor_idx=0, claimed_card=Card.DUKE)
        np.testing.assert_array_equal(bs.probs, before)

    def test_multiple_claims_converge_toward_actor(self):
        """Repeated Duke claims by P1 should raise P1's Duke prob toward 1.0."""
        bs = _fresh(observer_idx=-1)
        c_idx = CARD_TO_IDX[Card.DUKE]
        for _ in range(20):
            _updater().update(bs, "action", actor_idx=1, claimed_card=Card.DUKE)
        # After many claims, actor's prob should be noticeably higher than initial
        assert bs.probs[1, c_idx] > 0.5


# ===========================================================================
# Block update
# ===========================================================================

class TestBlockUpdate:

    def test_blocker_prob_increases_for_blocking_card(self):
        bs = _fresh(observer_idx=-1)
        c_idx = CARD_TO_IDX[Card.CONTESSA]
        before = bs.probs[2, c_idx]
        _updater().update(bs, "block", blocker_idx=2, blocking_card=Card.CONTESSA)
        assert bs.probs[2, c_idx] >= before

    def test_non_blocker_prob_decreases_for_blocking_card(self):
        bs = _fresh(observer_idx=-1)
        c_idx = CARD_TO_IDX[Card.CONTESSA]
        before_p0 = bs.probs[0, c_idx]
        _updater().update(bs, "block", blocker_idx=2, blocking_card=Card.CONTESSA)
        assert bs.probs[0, c_idx] <= before_p0

    def test_block_update_symmetric_to_claim_update(self):
        """Block and action claim with same card/player should have identical effect."""
        bs_block  = _fresh(observer_idx=-1)
        bs_action = _fresh(observer_idx=-1)
        _updater().update(bs_block,  "block",  blocker_idx=1, blocking_card=Card.DUKE)
        _updater().update(bs_action, "action", actor_idx=1,   claimed_card=Card.DUKE)
        np.testing.assert_array_almost_equal(bs_block.probs, bs_action.probs)


# ===========================================================================
# Challenge result — actor won
# ===========================================================================

class TestChallengeResultActorWon:

    def test_actor_row_reset_to_prior_after_win(self):
        """
        After winning a challenge, actor swaps their card, so we reset their
        row to the hypergeometric prior (maximum uncertainty again).
        """
        bs = _fresh(observer_idx=-1)
        # Build up some claim-based probability first
        _updater().update(bs, "action", actor_idx=1, claimed_card=Card.DUKE)
        _updater().update(bs, "challenge_result",
                          actor_idx=1, claimed_card=Card.DUKE,
                          challenger_idx=2, actor_won=True)

        # After win + swap, actor's row should match hypergeometric prior
        pool = _current_pool_size(bs)
        for c_idx in range(N_CARDS):
            expected = hypergeometric_at_least_one(
                pool_size=pool,
                n_hits=int(bs.available[c_idx]),
                draw_size=bs.influence_counts[1],
            )
            assert abs(bs.probs[1, c_idx] - expected) < 1e-9

    def test_probs_in_unit_interval_after_actor_win(self):
        bs = _fresh(observer_idx=-1)
        _updater().update(bs, "challenge_result",
                          actor_idx=1, claimed_card=Card.CAPTAIN,
                          challenger_idx=0, actor_won=True)
        assert np.all(bs.probs >= 0.0)
        assert np.all(bs.probs <= 1.0)

    def test_observer_row_untouched_when_actor_wins(self):
        bs = _fresh(observer_idx=0, known_hand=[Card.DUKE, Card.ASSASSIN])
        before_own = bs.probs[0].copy()
        _updater().update(bs, "challenge_result",
                          actor_idx=1, claimed_card=Card.CAPTAIN,
                          challenger_idx=0, actor_won=True)
        np.testing.assert_array_equal(bs.probs[0], before_own)

    def test_no_nan_after_actor_win(self):
        bs = _fresh(observer_idx=-1)
        _updater().update(bs, "challenge_result",
                          actor_idx=2, claimed_card=Card.TAX if hasattr(Card, "TAX") else Card.DUKE,
                          challenger_idx=1, actor_won=True)
        assert not np.any(np.isnan(bs.probs))


# ===========================================================================
# Challenge result — challenger won (bluff detected)
# ===========================================================================

class TestChallengeResultChallengerWon:

    def test_bluffed_card_prob_set_to_zero(self):
        bs = _fresh(observer_idx=-1)
        c_idx = CARD_TO_IDX[Card.DUKE]
        _updater().update(bs, "challenge_result",
                          actor_idx=1, claimed_card=Card.DUKE,
                          challenger_idx=2, actor_won=False)
        assert bs.probs[1, c_idx] == 0.0

    def test_other_cards_redistributed(self):
        """After bluff, remaining card probabilities should still be positive."""
        bs = _fresh(observer_idx=-1)
        _updater().update(bs, "challenge_result",
                          actor_idx=1, claimed_card=Card.DUKE,
                          challenger_idx=2, actor_won=False)
        c_idx = CARD_TO_IDX[Card.DUKE]
        # Other cards should still have non-zero prob (player holds something)
        other_cards = [i for i in range(N_CARDS) if i != c_idx]
        assert bs.probs[1, other_cards].sum() > 0.0

    def test_probs_in_unit_interval_after_bluff(self):
        bs = _fresh(observer_idx=-1)
        _updater().update(bs, "challenge_result",
                          actor_idx=1, claimed_card=Card.ASSASSIN,
                          challenger_idx=0, actor_won=False)
        assert np.all(bs.probs >= 0.0)
        assert np.all(bs.probs <= 1.0)

    def test_observer_row_untouched_when_challenger_wins(self):
        bs = _fresh(observer_idx=0, known_hand=[Card.DUKE, Card.ASSASSIN])
        before_own = bs.probs[0].copy()
        _updater().update(bs, "challenge_result",
                          actor_idx=1, claimed_card=Card.CAPTAIN,
                          challenger_idx=0, actor_won=False)
        np.testing.assert_array_equal(bs.probs[0], before_own)

    def test_no_nan_after_bluff_detected(self):
        bs = _fresh(observer_idx=-1)
        _updater().update(bs, "challenge_result",
                          actor_idx=3, claimed_card=Card.AMBASSADOR,
                          challenger_idx=1, actor_won=False)
        assert not np.any(np.isnan(bs.probs))

    def test_multiple_bluffs_accumulate(self):
        """Detecting two bluffs on the same player drops two card probs to 0."""
        bs = _fresh(observer_idx=-1)
        _updater().update(bs, "challenge_result",
                          actor_idx=1, claimed_card=Card.DUKE,
                          challenger_idx=2, actor_won=False)
        _updater().update(bs, "challenge_result",
                          actor_idx=1, claimed_card=Card.ASSASSIN,
                          challenger_idx=2, actor_won=False)
        assert bs.probs[1, CARD_TO_IDX[Card.DUKE]] == 0.0
        assert bs.probs[1, CARD_TO_IDX[Card.ASSASSIN]] == 0.0


# ===========================================================================
# Influence loss
# ===========================================================================

class TestInfluenceLoss:

    def test_influence_count_decremented(self):
        bs = _fresh()
        _updater().update(bs, "influence_loss", player_idx=1, card_lost=Card.DUKE)
        assert bs.influence_counts[1] == 1

    def test_available_decremented(self):
        bs = _fresh()
        c_idx = CARD_TO_IDX[Card.DUKE]
        before_avail = bs.available[c_idx]
        _updater().update(bs, "influence_loss", player_idx=1, card_lost=Card.DUKE)
        assert bs.available[c_idx] == before_avail - 1

    def test_revealed_cards_updated(self):
        bs = _fresh()
        _updater().update(bs, "influence_loss", player_idx=2, card_lost=Card.CAPTAIN)
        assert Card.CAPTAIN in bs.revealed_cards[2]

    def test_dead_player_zeroed_after_second_loss(self):
        bs = _fresh(influence_counts=[2, 2, 2, 2], revealed_cards=[[], [], [], []])
        _updater().update(bs, "influence_loss", player_idx=1, card_lost=Card.DUKE)
        _updater().update(bs, "influence_loss", player_idx=1, card_lost=Card.ASSASSIN)
        assert bs.influence_counts[1] == 0
        np.testing.assert_array_equal(bs.probs[1], np.zeros(N_CARDS))

    def test_probs_in_unit_interval_after_loss(self):
        bs = _fresh()
        _updater().update(bs, "influence_loss", player_idx=2, card_lost=Card.CONTESSA)
        assert np.all(bs.probs >= 0.0)
        assert np.all(bs.probs <= 1.0)

    def test_no_nan_after_loss(self):
        bs = _fresh()
        _updater().update(bs, "influence_loss", player_idx=0, card_lost=Card.AMBASSADOR)
        assert not np.any(np.isnan(bs.probs))

    def test_observer_row_stays_certain_after_opponents_loss(self):
        bs = _fresh(observer_idx=0, known_hand=[Card.DUKE, Card.CAPTAIN])
        before_own = bs.probs[0].copy()
        _updater().update(bs, "influence_loss", player_idx=1, card_lost=Card.ASSASSIN)
        np.testing.assert_array_equal(bs.probs[0], before_own)

    def test_all_rows_recomputed_after_loss(self):
        """Pool shrinks by 1 — all opponent rows should change."""
        bs = _fresh(observer_idx=-1)
        probs_before = bs.probs.copy()
        _updater().update(bs, "influence_loss", player_idx=1, card_lost=Card.DUKE)
        # At least some rows should have changed (pool is smaller)
        # We check all alive non-revealed rows
        changed = False
        for p_idx in range(4):
            if p_idx != 1 and bs.influence_counts[p_idx] > 0:
                if not np.allclose(bs.probs[p_idx], probs_before[p_idx]):
                    changed = True
        assert changed

    def test_revealed_card_reduces_available_but_not_below_zero(self):
        """Reveal all 3 Dukes — available should be exactly 0, not negative."""
        bs = BeliefState.initialise(
            n_players=4,
            influence_counts=[0, 0, 0, 2],
            revealed_cards=[[Card.DUKE], [Card.DUKE], [Card.DUKE], []],
        )
        # Duke available is already 0 — a further loss should not go negative
        upd = _updater()
        upd.update(bs, "influence_loss", player_idx=3, card_lost=Card.CAPTAIN)
        assert bs.available[CARD_TO_IDX[Card.DUKE]] >= 0.0


# ===========================================================================
# Exchange
# ===========================================================================

class TestExchange:

    def test_actor_row_reset_to_prior(self):
        bs = _fresh(observer_idx=-1)
        # First bias the actor's row with a claim
        _updater().update(bs, "action", actor_idx=1, claimed_card=Card.AMBASSADOR)
        # Now exchange
        _updater().update(bs, "exchange", actor_idx=1)
        # Row should now equal hypergeometric prior
        pool = _current_pool_size(bs)
        for c_idx in range(N_CARDS):
            expected = hypergeometric_at_least_one(
                pool_size=pool,
                n_hits=int(bs.available[c_idx]),
                draw_size=bs.influence_counts[1],
            )
            assert abs(bs.probs[1, c_idx] - expected) < 1e-9

    def test_other_rows_unchanged_by_exchange(self):
        bs = _fresh(observer_idx=-1)
        probs_p0_before = bs.probs[0].copy()
        probs_p2_before = bs.probs[2].copy()
        _updater().update(bs, "exchange", actor_idx=1)
        np.testing.assert_array_equal(bs.probs[0], probs_p0_before)
        np.testing.assert_array_equal(bs.probs[2], probs_p2_before)

    def test_observer_row_untouched_by_exchange(self):
        """Observer performing their own exchange: row stays as known hand."""
        bs = _fresh(observer_idx=0, known_hand=[Card.DUKE, Card.ASSASSIN])
        before_own = bs.probs[0].copy()
        _updater().update(bs, "exchange", actor_idx=0)
        np.testing.assert_array_equal(bs.probs[0], before_own)

    def test_probs_valid_after_exchange(self):
        bs = _fresh(observer_idx=-1)
        _updater().update(bs, "exchange", actor_idx=2)
        assert np.all(bs.probs >= 0.0)
        assert np.all(bs.probs <= 1.0)
        assert not np.any(np.isnan(bs.probs))

    def test_exchange_increases_uncertainty(self):
        """After a series of claims, an exchange should increase entropy."""
        bs = _fresh(observer_idx=-1)
        for _ in range(5):
            _updater().update(bs, "action", actor_idx=1, claimed_card=Card.AMBASSADOR)
        entropy_before = bs.entropy(1)
        _updater().update(bs, "exchange", actor_idx=1)
        entropy_after = bs.entropy(1)
        assert entropy_after >= entropy_before - 1e-9


# ===========================================================================
# Multi-event sequences
# ===========================================================================

class TestSequences:

    def test_claim_then_bluff_detected(self):
        """Actor claims Duke, then loses challenge — Duke prob should end at 0."""
        bs = _fresh(observer_idx=-1)
        upd = _updater()
        upd.update(bs, "action", actor_idx=1, claimed_card=Card.DUKE)
        upd.update(bs, "challenge_result",
                   actor_idx=1, claimed_card=Card.DUKE,
                   challenger_idx=2, actor_won=False)
        assert bs.probs[1, CARD_TO_IDX[Card.DUKE]] == 0.0

    def test_influence_loss_then_death(self):
        """Two influence losses kill a player — their row ends at zero."""
        bs = _fresh(observer_idx=-1)
        upd = _updater()
        upd.update(bs, "influence_loss", player_idx=3, card_lost=Card.CONTESSA)
        upd.update(bs, "influence_loss", player_idx=3, card_lost=Card.DUKE)
        np.testing.assert_array_equal(bs.probs[3], np.zeros(N_CARDS))
        assert bs.influence_counts[3] == 0

    def test_probs_stay_valid_through_long_sequence(self):
        """Run 20 mixed events and verify invariants hold throughout."""
        import random
        rng = random.Random(42)
        bs = _fresh(observer_idx=-1)
        upd = _updater()

        event_types = ["action", "block", "exchange"]
        cards = list(Card)

        for _ in range(20):
            et = rng.choice(event_types)
            actor = rng.randint(0, 3)
            card = rng.choice(cards)

            if et == "action":
                upd.update(bs, "action", actor_idx=actor, claimed_card=card)
            elif et == "block":
                upd.update(bs, "block", blocker_idx=actor, blocking_card=card)
            elif et == "exchange":
                upd.update(bs, "exchange", actor_idx=actor)

            assert np.all(bs.probs >= 0.0), "Negative probability after event"
            assert np.all(bs.probs <= 1.0), "Probability > 1 after event"
            assert not np.any(np.isnan(bs.probs)), "NaN after event"

    def test_full_game_sequence(self):
        """
        Simulate a plausible 3-player game sequence and verify state consistency
        throughout: P0 claims Duke (Tax), P1 challenges and loses,
        P1 loses Assassin, P2 claims Contessa (block), P0 claims Ambassador (Exchange),
        P2 loses Contessa, P2 dies.
        """
        bs = BeliefState.initialise(
            n_players=3,
            influence_counts=[2, 2, 2],
            revealed_cards=[[], [], []],
            observer_idx=0,
            known_hand=[Card.DUKE, Card.CAPTAIN],
        )
        upd = BeliefUpdater(n_players=3)

        # Turn 1: P0 claims Duke (Tax)
        upd.update(bs, "action", actor_idx=0, claimed_card=Card.DUKE)
        assert np.all(bs.probs >= 0.0) and np.all(bs.probs <= 1.0)

        # P1 challenges — P0 wins (they have Duke)
        upd.update(bs, "challenge_result",
                   actor_idx=0, claimed_card=Card.DUKE,
                   challenger_idx=1, actor_won=True)
        assert np.all(bs.probs >= 0.0)

        # P1 loses influence (Assassin)
        upd.update(bs, "influence_loss", player_idx=1, card_lost=Card.ASSASSIN)
        assert bs.influence_counts[1] == 1
        assert Card.ASSASSIN in bs.revealed_cards[1]

        # Turn 2: P2 claims Contessa (block Foreign Aid)
        upd.update(bs, "block", blocker_idx=2, blocking_card=Card.CONTESSA)
        assert np.all(bs.probs >= 0.0)

        # Turn 3: P0 claims Ambassador (Exchange)
        upd.update(bs, "action", actor_idx=0, claimed_card=Card.AMBASSADOR)
        upd.update(bs, "exchange", actor_idx=0)

        # Turn 4: P2 loses Contessa
        upd.update(bs, "influence_loss", player_idx=2, card_lost=Card.CONTESSA)
        # Turn 4b: P2 loses remaining card
        upd.update(bs, "influence_loss", player_idx=2, card_lost=Card.CAPTAIN)
        assert bs.influence_counts[2] == 0
        np.testing.assert_array_equal(bs.probs[2], np.zeros(N_CARDS))

        # Final invariants
        assert np.all(bs.probs >= 0.0)
        assert np.all(bs.probs <= 1.0)
        assert not np.any(np.isnan(bs.probs))
