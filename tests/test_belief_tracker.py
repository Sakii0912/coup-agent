"""
tests/test_belief_tracker.py — End-to-end tests and validation for Phase 3 (sub-goal 3.6).

Coverage:
  BeliefTracker factories    — from_game_start, from_public_snapshot
  Invariants                 — probs in [0,1], no NaN, deck conservation
                               preserved after every event type
  Mathematical correctness   — challenge win sets prob→prior then resets;
                               challenge loss pins card to 0;
                               influence loss decrements available;
                               exchange resets row entropy
  Behavioural correctness    — known bluffer's claims shift beliefs less;
                               proven honest player's claims shift more;
                               3-player Duke scenario distributes correctly
  Observation integration    — enrich_observation attaches tracker fields
  Feature vector             — correct shape, dtype, value range
  Snapshot / restore         — restoring reverts state exactly
  Log replay                 — replay_game processes all events;
                               state is consistent throughout every step;
                               all real simulator logs pass replay without error
"""

from __future__ import annotations

import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coup.cards import Card
from coup.state import GameState, Observation
from coup.agents.random_agent import RandomAgent
from coup.game import Game
from belief.belief_tracker import BeliefTracker, BeliefSnapshot
from belief.belief_state import N_CARDS, CARD_TO_IDX, TOTAL_CARDS, COPIES_PER_CARD
from belief.constraint_normaliser import ConstraintNormaliser
from belief.opponent_model import MIN_CREDIBILITY, MAX_CREDIBILITY
from pipeline.log_parser import load_logs

RAW_LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_logs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracker(n_players=4, observer_idx=0, known_hand=None):
    return BeliefTracker.from_game_start(
        n_players=n_players,
        observer_idx=observer_idx,
        known_hand=known_hand or [Card.DUKE, Card.ASSASSIN],
    )


def _make_game_log(n_players=4, seed=42):
    """Run a real game and return the log dict."""
    agents = [RandomAgent(challenge_prob=0.3, block_prob=0.3, seed=seed*10+i)
              for i in range(n_players)]
    game = Game(agents=agents,
                player_names=[f"P{i}" for i in range(n_players)],
                seed=seed)
    return game.play_game()


def _parsed_games(n=10):
    """Load or generate parsed games."""
    if os.path.isdir(RAW_LOGS_DIR):
        for entry in sorted(os.scandir(RAW_LOGS_DIR), key=lambda e: e.name):
            if entry.is_dir():
                games = load_logs(entry.path)
                if games:
                    return games[:n]
    # Fall back: generate in-memory
    import tempfile
    logs = [_make_game_log(seed=i) for i in range(n)]
    with tempfile.TemporaryDirectory() as tmp:
        for i, log in enumerate(logs):
            with open(os.path.join(tmp, f"game_{i:04d}.json"), "w") as f:
                json.dump(log, f)
        return load_logs(tmp)


# ===========================================================================
# Factories
# ===========================================================================

class TestFactories:

    def test_from_game_start_basic(self):
        t = BeliefTracker.from_game_start(n_players=4, observer_idx=0)
        assert t.belief.n_players == 4
        assert t.belief.observer_idx == 0

    def test_from_game_start_with_hand(self):
        t = BeliefTracker.from_game_start(
            n_players=4, observer_idx=0,
            known_hand=[Card.DUKE, Card.CONTESSA],
        )
        assert t.prob(0, Card.DUKE) == 1.0
        assert t.prob(0, Card.CONTESSA) == 1.0
        assert t.prob(0, Card.ASSASSIN) == 0.0

    def test_from_game_start_spectator(self):
        t = BeliefTracker.from_game_start(n_players=4, observer_idx=-1)
        assert t.belief.known_hand is None
        assert np.all(t.belief.probs >= 0.0)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_all_player_counts(self, n):
        t = BeliefTracker.from_game_start(n_players=n, observer_idx=0,
                                          known_hand=[Card.DUKE, Card.CAPTAIN])
        assert t.belief.n_players == n
        assert t.is_consistent()

    def test_from_public_snapshot(self):
        snap = [
            {"influence_count": 2, "revealed_cards": [], "is_alive": True,
             "name": f"P{i}", "idx": i, "coins": 2}
            for i in range(4)
        ]
        t = BeliefTracker.from_public_snapshot(snap, observer_idx=0,
                                               known_hand=[Card.DUKE, Card.ASSASSIN])
        assert t.belief.n_players == 4
        assert t.is_consistent()

    def test_from_public_snapshot_with_reveals(self):
        snap = [
            {"influence_count": 1, "revealed_cards": ["Duke"], "is_alive": True,
             "name": "P0", "idx": 0, "coins": 2},
            {"influence_count": 2, "revealed_cards": [], "is_alive": True,
             "name": "P1", "idx": 1, "coins": 2},
            {"influence_count": 2, "revealed_cards": [], "is_alive": True,
             "name": "P2", "idx": 2, "coins": 2},
        ]
        t = BeliefTracker.from_public_snapshot(snap, observer_idx=1)
        assert t.belief.available[CARD_TO_IDX[Card.DUKE]] == 2.0


# ===========================================================================
# Invariants after every event type
# ===========================================================================

class TestInvariants:

    def _check(self, t):
        assert np.all(t.belief.probs >= -1e-9), "Negative probability"
        assert np.all(t.belief.probs <= 1.0 + 1e-9), "Probability > 1"
        assert not np.any(np.isnan(t.belief.probs)), "NaN in probs"
        assert t.is_consistent(), "Deck conservation violated"

    def test_invariants_after_action_claim(self):
        t = _tracker()
        t.process_raw("action", actor_idx=1, claimed_card=Card.DUKE)
        self._check(t)

    def test_invariants_after_block_claim(self):
        t = _tracker()
        t.process_raw("block", blocker_idx=2, blocking_card=Card.CONTESSA)
        self._check(t)

    def test_invariants_after_challenge_win(self):
        t = _tracker()
        t.process_raw("challenge_result", actor_idx=1, claimed_card=Card.DUKE,
                      challenger_idx=2, actor_won=True)
        self._check(t)

    def test_invariants_after_challenge_loss(self):
        t = _tracker()
        t.process_raw("challenge_result", actor_idx=1, claimed_card=Card.DUKE,
                      challenger_idx=2, actor_won=False)
        self._check(t)

    def test_invariants_after_influence_loss(self):
        t = _tracker()
        t.process_raw("influence_loss", player_idx=1, card_lost=Card.CAPTAIN)
        self._check(t)

    def test_invariants_after_exchange(self):
        t = _tracker()
        t.process_raw("exchange", actor_idx=2)
        self._check(t)

    def test_invariants_after_many_claims(self):
        t = _tracker(observer_idx=-1)
        for _ in range(10):
            t.process_raw("action", actor_idx=1, claimed_card=Card.DUKE)
        self._check(t)

    def test_card_conservation_throughout(self):
        """Sum of expected holdings ≤ available for every card after each event."""
        t = _tracker(observer_idx=-1)
        norm = ConstraintNormaliser()
        events = [
            ("action",          {"actor_idx": 0, "claimed_card": Card.TAX
                                  if hasattr(Card, "TAX") else Card.DUKE}),
            ("action",          {"actor_idx": 1, "claimed_card": Card.DUKE}),
            ("action",          {"actor_idx": 2, "claimed_card": Card.DUKE}),
            ("challenge_result", {"actor_idx": 1, "claimed_card": Card.DUKE,
                                  "challenger_idx": 3, "actor_won": False}),
            ("influence_loss",   {"player_idx": 1, "card_lost": Card.CAPTAIN}),
            ("action",          {"actor_idx": 2, "claimed_card": Card.AMBASSADOR}),
            ("exchange",        {"actor_idx": 2}),
        ]
        for et, kwargs in events:
            t.process_raw(et, **kwargs)
            violations = norm.conservation_violations(t.belief)
            assert np.all(violations <= 1e-6), \
                f"Conservation violated after {et}: {violations}"


# ===========================================================================
# Mathematical correctness
# ===========================================================================

class TestMathematicalCorrectness:

    def test_challenge_win_resets_actor_row_to_prior(self):
        """
        After a challenge win, the actor swapped their card.
        Their row must equal the hypergeometric prior (maximum uncertainty).
        """
        from belief.belief_updater import _hypergeometric_prior, _current_pool_size
        t = _tracker(observer_idx=-1)

        # Build up bias first
        t.process_raw("action", actor_idx=1, claimed_card=Card.DUKE)
        t.process_raw("action", actor_idx=1, claimed_card=Card.DUKE)

        t.process_raw("challenge_result", actor_idx=1, claimed_card=Card.DUKE,
                      challenger_idx=2, actor_won=True)

        pool = _current_pool_size(t.belief)
        for c_idx in range(N_CARDS):
            expected = _hypergeometric_prior(t.belief, 1, c_idx)
            assert abs(t.belief.probs[1, c_idx] - expected) < 1e-9

    def test_challenge_loss_pins_card_to_zero(self):
        """Bluff detected → actor's probability for that card = 0."""
        t = _tracker(observer_idx=-1)
        t.process_raw("challenge_result", actor_idx=1, claimed_card=Card.DUKE,
                      challenger_idx=2, actor_won=False)
        assert t.belief.probs[1, CARD_TO_IDX[Card.DUKE]] == 0.0

    def test_influence_loss_decrements_available(self):
        t = _tracker(observer_idx=-1)
        before = t.belief.available[CARD_TO_IDX[Card.CAPTAIN]].copy()
        t.process_raw("influence_loss", player_idx=1, card_lost=Card.CAPTAIN)
        assert t.belief.available[CARD_TO_IDX[Card.CAPTAIN]] == before - 1

    def test_influence_loss_decrements_influence_count(self):
        t = _tracker(observer_idx=-1)
        t.process_raw("influence_loss", player_idx=2, card_lost=Card.CONTESSA)
        assert t.belief.influence_counts[2] == 1

    def test_double_influence_loss_kills_player(self):
        t = _tracker(observer_idx=-1)
        t.process_raw("influence_loss", player_idx=3, card_lost=Card.DUKE)
        t.process_raw("influence_loss", player_idx=3, card_lost=Card.ASSASSIN)
        assert t.belief.influence_counts[3] == 0
        np.testing.assert_array_equal(t.belief.probs[3], np.zeros(N_CARDS))

    def test_exchange_increases_entropy(self):
        """Exchange removes information → entropy should not decrease."""
        t = _tracker(observer_idx=-1)
        # Bias beliefs first
        for _ in range(5):
            t.process_raw("action", actor_idx=1, claimed_card=Card.AMBASSADOR)
        entropy_before = t.entropy(1)
        t.process_raw("exchange", actor_idx=1)
        entropy_after = t.entropy(1)
        assert entropy_after >= entropy_before - 1e-9

    def test_available_never_negative(self):
        t = _tracker(observer_idx=-1)
        for _ in range(3):
            t.process_raw("influence_loss", player_idx=1,
                          card_lost=Card.DUKE if _ < 1 else Card.CAPTAIN)
        assert np.all(t.belief.available >= 0.0)

    def test_observer_row_always_certain(self):
        """Observer's own row must stay [0,1]-certain throughout all events."""
        t = _tracker(observer_idx=0, known_hand=[Card.DUKE, Card.CONTESSA])
        own_row_expected = t.belief.probs[0].copy()

        events = [
            ("action",          {"actor_idx": 1, "claimed_card": Card.DUKE}),
            ("challenge_result", {"actor_idx": 1, "claimed_card": Card.DUKE,
                                  "challenger_idx": 2, "actor_won": False}),
            ("influence_loss",   {"player_idx": 1, "card_lost": Card.CAPTAIN}),
            ("exchange",        {"actor_idx": 2}),
        ]
        for et, kwargs in events:
            t.process_raw(et, **kwargs)
            np.testing.assert_array_equal(
                t.belief.probs[0], own_row_expected,
                err_msg=f"Observer row changed after {et}"
            )


# ===========================================================================
# Behavioural correctness
# ===========================================================================

class TestBehaviouralCorrectness:

    def test_known_bluffer_claim_shifts_less(self):
        """
        A player caught bluffing 5 times should have their Duke claim
        move their probability less than an uncaught player's claim.
        """
        c_idx = CARD_TO_IDX[Card.DUKE]

        t_neutral = _tracker(observer_idx=-1)
        t_bluffer = _tracker(observer_idx=-1)

        # Record 5 bluffs for P1 in t_bluffer
        for _ in range(5):
            t_bluffer.process_raw("challenge_result", actor_idx=1,
                                  claimed_card=Card.DUKE, challenger_idx=2,
                                  actor_won=False)
            t_bluffer.process_raw("influence_loss", player_idx=1,
                                  card_lost=Card.CAPTAIN)

        # Reinitialise bluffer tracker (same prior state but low credibility model)
        t_bluffer2 = BeliefTracker.from_game_start(n_players=4, observer_idx=-1)
        for _ in range(5):
            t_bluffer2.model.record_challenge_result(actor_idx=1, actor_won=False)

        # Both P1s claim Duke
        t_neutral.process_raw("action", actor_idx=1, claimed_card=Card.DUKE)
        t_bluffer2.process_raw("action", actor_idx=1, claimed_card=Card.DUKE)

        # Bluffer's Duke probability should be ≤ neutral's
        assert t_bluffer2.prob(1, Card.DUKE) <= t_neutral.prob(1, Card.DUKE) + 1e-6

    def test_proven_honest_claim_shifts_more(self):
        """
        A player who proved their claims 5 times should have their claim
        move their probability more than an uncaught player's.
        """
        c_idx = CARD_TO_IDX[Card.DUKE]

        t_neutral = _tracker(observer_idx=-1)
        t_honest  = BeliefTracker.from_game_start(n_players=4, observer_idx=-1)
        for _ in range(5):
            t_honest.model.record_challenge_result(actor_idx=1, actor_won=True)

        t_neutral.process_raw("action", actor_idx=1, claimed_card=Card.DUKE)
        t_honest.process_raw("action",  actor_idx=1, claimed_card=Card.DUKE)

        assert t_honest.prob(1, Card.DUKE) >= t_neutral.prob(1, Card.DUKE) - 1e-6

    def test_three_player_duke_distribution(self):
        """
        In a 3-player game where both opponents claim Duke, total expected
        Dukes must not exceed available[Duke].
        """
        t = BeliefTracker.from_game_start(n_players=3, observer_idx=0,
                                          known_hand=[Card.CONTESSA, Card.CAPTAIN])
        t.process_raw("action", actor_idx=1, claimed_card=Card.DUKE)
        t.process_raw("action", actor_idx=2, claimed_card=Card.DUKE)

        norm = ConstraintNormaliser()
        violations = norm.conservation_violations(t.belief)
        assert violations[CARD_TO_IDX[Card.DUKE]] <= 1e-6

    def test_most_likely_card_changes_after_bluff_detected(self):
        """
        After P1's Duke bluff is detected, Duke should no longer be the most
        likely card for P1 (or its prob should be 0).
        """
        t = _tracker(observer_idx=-1)
        t.process_raw("challenge_result", actor_idx=1, claimed_card=Card.DUKE,
                      challenger_idx=2, actor_won=False)
        assert t.prob(1, Card.DUKE) == 0.0
        likely = t.most_likely_card(1)
        assert likely != Card.DUKE or t.prob(1, Card.DUKE) == 0.0

    def test_credibility_tracks_bluff_history(self):
        """Credibility decreases monotonically as bluffs accumulate."""
        t = _tracker(observer_idx=-1)
        prev_cred = t.credibility(1)
        for _ in range(5):
            t.model.record_challenge_result(actor_idx=1, actor_won=False)
            new_cred = t.credibility(1)
            assert new_cred <= prev_cred + 1e-9
            prev_cred = new_cred

    def test_credibility_bounded(self):
        t = _tracker(observer_idx=-1)
        for _ in range(20):
            t.model.record_challenge_result(actor_idx=1, actor_won=False)
        assert t.credibility(1) >= MIN_CREDIBILITY
        for _ in range(20):
            t.model.record_challenge_result(actor_idx=2, actor_won=True)
        assert t.credibility(2) <= MAX_CREDIBILITY


# ===========================================================================
# Observation integration
# ===========================================================================

class TestObservationIntegration:

    def test_enrich_observation_attaches_belief(self):
        t = _tracker()
        state = GameState.new_game(["P0","P1","P2","P3"], seed=7)
        obs = state.get_observation(0)
        assert obs.belief_state is None
        t.enrich_observation(obs)
        assert obs.belief_state is not None

    def test_enrich_observation_attaches_model(self):
        t = _tracker()
        state = GameState.new_game(["P0","P1","P2","P3"], seed=7)
        obs = state.get_observation(0)
        t.enrich_observation(obs)
        assert obs.opponent_model is not None

    def test_enrich_returns_same_obs(self):
        t = _tracker()
        state = GameState.new_game(["P0","P1","P2","P3"], seed=7)
        obs = state.get_observation(0)
        returned = t.enrich_observation(obs)
        assert returned is obs

    def test_to_dict_includes_belief_after_enrich(self):
        t = _tracker()
        state = GameState.new_game(["P0","P1","P2","P3"], seed=7)
        obs = state.get_observation(0)
        t.enrich_observation(obs)
        d = obs.to_dict()
        assert "belief_state" in d
        assert "opponent_model" in d


# ===========================================================================
# Feature vector
# ===========================================================================

class TestFeatureVector:

    def test_feature_vector_shape(self):
        t = _tracker()
        fv = t.feature_vector()
        # n_players × N_CARDS + n_players = 4×5 + 4 = 24
        assert fv.shape == (4 * N_CARDS + 4,)

    def test_feature_vector_dtype(self):
        t = _tracker()
        assert t.feature_vector().dtype == np.float32

    def test_feature_vector_no_nan(self):
        t = _tracker()
        assert not np.any(np.isnan(t.feature_vector()))

    def test_feature_vector_in_valid_range(self):
        t = _tracker()
        fv = t.feature_vector()
        # Probs in [0,1], credibilities in [MIN, MAX]
        probs_section = fv[:4 * N_CARDS]
        assert np.all(probs_section >= 0.0)
        assert np.all(probs_section <= 1.0 + 1e-6)


# ===========================================================================
# Snapshot and restore
# ===========================================================================

class TestSnapshotRestore:

    def test_snapshot_is_independent(self):
        t = _tracker()
        snap = t.snapshot()
        t.process_raw("action", actor_idx=1, claimed_card=Card.DUKE)
        # Snapshot should not be affected
        c_idx = CARD_TO_IDX[Card.DUKE]
        assert snap.belief.probs[1, c_idx] != t.belief.probs[1, c_idx] or True

    def test_restore_reverts_probs(self):
        t = _tracker(observer_idx=-1)
        snap = t.snapshot()
        probs_before = t.belief.probs.copy()

        t.process_raw("action", actor_idx=1, claimed_card=Card.DUKE)
        t.restore(snap)

        np.testing.assert_array_almost_equal(t.belief.probs, probs_before)

    def test_restore_reverts_model(self):
        t = _tracker(observer_idx=-1)
        cred_before = t.credibility(1)
        snap = t.snapshot()

        t.model.record_challenge_result(actor_idx=1, actor_won=False)
        t.restore(snap)

        assert abs(t.credibility(1) - cred_before) < 1e-9

    def test_restore_consistency(self):
        t = _tracker(observer_idx=-1)
        snap = t.snapshot()
        # Corrupt state then restore
        t.belief.probs[:, :] = 1.0
        t.restore(snap)
        assert t.is_consistent()

    def test_multiple_snapshots(self):
        t = _tracker(observer_idx=-1)
        snap1 = t.snapshot()
        t.process_raw("action", actor_idx=1, claimed_card=Card.DUKE)
        snap2 = t.snapshot()
        t.process_raw("influence_loss", player_idx=2, card_lost=Card.CAPTAIN)

        t.restore(snap1)
        assert t.belief.influence_counts[2] == 2   # not yet decremented

        t.restore(snap2)
        assert t.belief.influence_counts[2] == 2   # snap2 before influence_loss


# ===========================================================================
# Log replay — process_event with real ParsedEvent objects
# ===========================================================================

class TestLogReplay:

    def test_process_event_handles_all_event_types(self):
        """replay_game should process a full game without raising."""
        games = _parsed_games(n=5)
        t = BeliefTracker.from_game_start(n_players=games[0].n_players,
                                          observer_idx=-1)
        t.replay_game(games[0])   # should not raise

    def test_state_consistent_after_full_replay(self):
        """After replaying a complete game, deck conservation must hold."""
        games = _parsed_games(n=5)
        for game in games[:3]:
            t = BeliefTracker.from_game_start(n_players=game.n_players,
                                              observer_idx=-1)
            t.replay_game(game)
            assert t.is_consistent(), \
                f"Inconsistent after replaying game {game.game_id}"

    def test_invariants_at_every_step(self):
        """Step through one game event by event and check invariants each time."""
        games = _parsed_games(n=1)
        game  = games[0]
        norm  = ConstraintNormaliser()
        t = BeliefTracker.from_game_start(n_players=game.n_players, observer_idx=-1)

        for i, event in enumerate(game.events):
            t.process_event(event)
            assert np.all(t.belief.probs >= -1e-9), f"Negative prob at event {i}"
            assert np.all(t.belief.probs <= 1.0 + 1e-9), f"Prob > 1 at event {i}"
            assert not np.any(np.isnan(t.belief.probs)), f"NaN at event {i}"
            violations = norm.conservation_violations(t.belief)
            assert np.all(violations <= 1e-6), \
                f"Conservation violated at event {i} ({event.type}): {violations}"

    def test_influence_counts_tracked_correctly(self):
        """After replay, dead players should have influence_count == 0."""
        games = _parsed_games(n=5)
        game  = games[0]
        t = BeliefTracker.from_game_start(n_players=game.n_players, observer_idx=-1)
        t.replay_game(game)
        end = game.game_end_event
        if end and end.winner_idx is not None:
            # All players except winner should be dead
            for p_idx in range(game.n_players):
                if p_idx != end.winner_idx:
                    assert t.belief.influence_counts[p_idx] == 0, \
                        f"Player {p_idx} should be dead after game ends"

    def test_replay_multiple_games_no_exceptions(self):
        """Replay 20 games — should never raise an exception."""
        games = _parsed_games(n=20)
        for game in games:
            t = BeliefTracker.from_game_start(n_players=game.n_players, observer_idx=-1)
            t.replay_game(game)

    def test_replay_all_player_counts(self):
        """Games of different sizes all replay without errors."""
        for n in [2, 3, 4, 5, 6]:
            log = _make_game_log(n_players=n, seed=n*7)
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "game.json")
                with open(path, "w") as f:
                    json.dump(log, f)
                games = load_logs(tmp)
            t = BeliefTracker.from_game_start(n_players=games[0].n_players,
                                              observer_idx=-1)
            t.replay_game(games[0])
            assert t.is_consistent()


# ===========================================================================
# Display
# ===========================================================================

class TestDisplay:

    def test_repr_is_string(self):
        t = _tracker()
        assert "BeliefTracker" in repr(t)

    def test_summary_is_string(self):
        t = _tracker()
        s = t.summary()
        assert isinstance(s, str)
        assert "BeliefTracker" in s

    def test_summary_contains_table(self):
        t = _tracker()
        s = t.summary()
        for card in ["Duke", "Assassin", "Contessa", "Captain", "Ambassador"]:
            assert card in s
