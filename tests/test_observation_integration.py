"""
tests/test_observation_integration.py — Tests for sub-goal 3.5.

Coverage:
  Observation dataclass  — belief_state and opponent_model fields default to
                           None; to_dict includes them when set; repr shows
                           belief presence; Phase 1/2 code paths unchanged
  FeatureConfig          — new fields default to False; feature_dim correct for
                           all combinations; belief_dims / opp_model_dims correct
  extract_features       — Phase 2 config unchanged (104 dims); Phase 3 config
                           produces 134 dims; belief section populated correctly;
                           credibility section populated correctly
  features_from_observation — shape matches config; belief/cred sections correct;
                              None belief state produces zeros in belief section
  Backward compatibility — all Phase 1/2 tests still pass with new Observation
"""

from __future__ import annotations

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coup.state import Observation, GameState
from coup.cards import Card
from belief.belief_state import BeliefState, N_CARDS, CARD_TO_IDX
from belief.opponent_model import OpponentModel, MIN_CREDIBILITY, MAX_CREDIBILITY
from pipeline.feature_extractor import (
    FeatureConfig,
    extract_features,
    features_from_observation,
    N_ACTIONS,
    HISTORY_LENGTH,
)
from pipeline.log_parser import load_logs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RAW_LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_logs")

def _obs(n_players=4, with_belief=False, with_opp_model=False):
    """Build a minimal Observation for testing."""
    hand = [Card.DUKE, Card.ASSASSIN]
    revealed: list = []
    others = [
        {"name": f"P{i}", "idx": i, "coins": 2, "influence_count": 2,
         "revealed_cards": [], "is_alive": True}
        for i in range(1, n_players)
    ]
    belief = None
    opp_model = None

    if with_belief:
        belief = BeliefState.initialise(
            n_players=n_players,
            influence_counts=[2] * n_players,
            revealed_cards=[[] for _ in range(n_players)],
            observer_idx=0,
            known_hand=hand,
        )
    if with_opp_model:
        opp_model = OpponentModel(n_players=n_players, observer_idx=0)

    return Observation(
        my_idx=0,
        my_hand=hand,
        my_coins=2,
        my_revealed=revealed,
        others=others,
        deck_size=7,
        current_player_idx=0,
        turn_number=3,
        belief_state=belief,
        opponent_model=opp_model,
    )


def _phase3_config(max_players=6):
    return FeatureConfig(
        max_players=max_players,
        include_belief=True,
        include_opp_model=True,
    )


def _phase2_config(max_players=6):
    return FeatureConfig(
        max_players=max_players,
        include_belief=False,
        include_opp_model=False,
    )


# ===========================================================================
# Observation dataclass
# ===========================================================================

class TestObservationFields:

    def test_belief_state_defaults_to_none(self):
        obs = _obs(with_belief=False)
        assert obs.belief_state is None

    def test_opponent_model_defaults_to_none(self):
        obs = _obs(with_opp_model=False)
        assert obs.opponent_model is None

    def test_belief_state_set_when_provided(self):
        obs = _obs(with_belief=True)
        assert obs.belief_state is not None
        assert isinstance(obs.belief_state, BeliefState)

    def test_opponent_model_set_when_provided(self):
        obs = _obs(with_opp_model=True)
        assert obs.opponent_model is not None
        assert isinstance(obs.opponent_model, OpponentModel)

    def test_to_dict_without_belief(self):
        obs = _obs()
        d = obs.to_dict()
        assert "belief_state" not in d
        assert "opponent_model" not in d

    def test_to_dict_with_belief(self):
        obs = _obs(with_belief=True, with_opp_model=True)
        d = obs.to_dict()
        assert "belief_state" in d
        assert "opponent_model" in d

    def test_to_dict_belief_has_probs(self):
        obs = _obs(with_belief=True)
        d = obs.to_dict()
        assert "probs" in d["belief_state"]

    def test_repr_shows_belief_yes(self):
        obs = _obs(with_belief=True)
        assert "belief=yes" in repr(obs)

    def test_repr_shows_belief_no(self):
        obs = _obs(with_belief=False)
        assert "belief=no" in repr(obs)

    def test_phase1_fields_unchanged(self):
        """Existing Phase 1 fields still work exactly as before."""
        obs = _obs()
        assert obs.my_idx == 0
        assert obs.my_coins == 2
        assert obs.turn_number == 3
        assert len(obs.my_hand) == 2

    def test_get_observation_from_game_state_has_no_belief(self):
        """GameState.get_observation() still returns belief=None by default."""
        state = GameState.new_game(["P0", "P1", "P2", "P3"], seed=1)
        obs = state.get_observation(0)
        assert obs.belief_state is None
        assert obs.opponent_model is None


# ===========================================================================
# FeatureConfig
# ===========================================================================

class TestFeatureConfig:

    def test_phase2_defaults(self):
        cfg = FeatureConfig()
        assert cfg.include_belief    is False
        assert cfg.include_opp_model is False

    def test_phase2_feature_dim_unchanged(self):
        """The base 104-dim vector must be preserved for backward compatibility."""
        cfg = _phase2_config()
        assert cfg.feature_dim == 104

    def test_belief_dims_zero_when_disabled(self):
        cfg = _phase2_config()
        assert cfg.belief_dims == 0

    def test_opp_model_dims_zero_when_disabled(self):
        cfg = _phase2_config()
        assert cfg.opp_model_dims == 0

    def test_belief_dims_correct_when_enabled(self):
        cfg = _phase3_config(max_players=6)
        assert cfg.belief_dims == 6 * N_CARDS   # 30

    def test_opp_model_dims_correct_when_enabled(self):
        cfg = _phase3_config(max_players=6)
        assert cfg.opp_model_dims == 6

    def test_phase3_feature_dim(self):
        cfg = _phase3_config(max_players=6)
        assert cfg.feature_dim == 104 + 30 + 6   # 140

    def test_belief_only_dim(self):
        cfg = FeatureConfig(include_belief=True, include_opp_model=False)
        assert cfg.feature_dim == 104 + 30       # 134

    def test_opp_only_dim(self):
        cfg = FeatureConfig(include_belief=False, include_opp_model=True)
        assert cfg.feature_dim == 104 + 6        # 110

    @pytest.mark.parametrize("max_players", [2, 3, 4, 5, 6])
    def test_phase3_dim_scales_with_players(self, max_players):
        cfg = _phase3_config(max_players=max_players)
        expected_belief = max_players * N_CARDS
        expected_opp    = max_players
        assert cfg.belief_dims    == expected_belief
        assert cfg.opp_model_dims == expected_opp


# ===========================================================================
# extract_features — Phase 2 backward compatibility
# ===========================================================================

class TestExtractFeaturesBackwardCompat:

    def test_phase2_config_produces_104_dim_samples(self):
        if not os.path.isdir(RAW_LOGS_DIR):
            pytest.skip("data/raw_logs/ not found")
        games = load_logs(os.path.join(RAW_LOGS_DIR,
                          next(e.name for e in os.scandir(RAW_LOGS_DIR)
                               if e.is_dir())))
        cfg = _phase2_config()
        for s in extract_features(games[0], cfg):
            assert s.features.shape == (104,)

    def test_default_config_still_104(self):
        if not os.path.isdir(RAW_LOGS_DIR):
            pytest.skip("data/raw_logs/ not found")
        games = load_logs(os.path.join(RAW_LOGS_DIR,
                          next(e.name for e in os.scandir(RAW_LOGS_DIR)
                               if e.is_dir())))
        for s in extract_features(games[0]):   # no config = default
            assert s.features.shape == (104,)


# ===========================================================================
# features_from_observation — Phase 3 path
# ===========================================================================

class TestFeaturesFromObservation:

    def test_phase2_shape(self):
        obs = _obs()
        feat = features_from_observation(obs, "Tax", None, "Duke", [], _phase2_config())
        assert feat.shape == (104,)

    def test_phase3_shape(self):
        obs = _obs(with_belief=True, with_opp_model=True)
        feat = features_from_observation(obs, "Tax", None, "Duke", [], _phase3_config())
        assert feat.shape == (140,)

    def test_belief_only_shape(self):
        obs = _obs(with_belief=True)
        cfg = FeatureConfig(include_belief=True, include_opp_model=False)
        feat = features_from_observation(obs, "Income", None, None, [], cfg)
        assert feat.shape == (134,)

    def test_dtype_float32(self):
        obs = _obs(with_belief=True, with_opp_model=True)
        feat = features_from_observation(obs, "Tax", None, "Duke", [], _phase3_config())
        assert feat.dtype == np.float32

    def test_no_nan_or_inf(self):
        obs = _obs(with_belief=True, with_opp_model=True)
        feat = features_from_observation(obs, "Steal", 1, "Captain", ["Income"], _phase3_config())
        assert not np.any(np.isnan(feat))
        assert not np.any(np.isinf(feat))

    def test_belief_section_nonzero_when_provided(self):
        obs = _obs(with_belief=True, with_opp_model=True)
        cfg = _phase3_config()
        feat = features_from_observation(obs, "Tax", None, "Duke", [], cfg)
        # Belief section starts after base 104 dims
        belief_section = feat[104: 104 + cfg.belief_dims]
        assert np.any(belief_section > 0.0)

    def test_belief_section_zero_when_not_provided(self):
        obs = _obs(with_belief=False)   # no belief state
        cfg = _phase3_config()
        feat = features_from_observation(obs, "Tax", None, "Duke", [], cfg)
        belief_section = feat[104: 104 + cfg.belief_dims]
        assert np.all(belief_section == 0.0)

    def test_opp_model_section_all_ones_when_not_provided(self):
        """Without an opponent model, credibility defaults to 1.0 for all."""
        obs = _obs(with_opp_model=False)
        cfg = _phase3_config()
        feat = features_from_observation(obs, "Income", None, None, [], cfg)
        opp_start = 104 + cfg.belief_dims
        opp_section = feat[opp_start: opp_start + cfg.opp_model_dims]
        np.testing.assert_array_almost_equal(opp_section, np.ones(cfg.max_players))

    def test_opp_model_section_reflects_recorded_bluffs(self):
        """After recording bluffs, the credibility section should differ from 1.0."""
        obs = _obs(with_belief=True, with_opp_model=True)
        # Record P1 bluffing 5 times
        for _ in range(5):
            obs.opponent_model.record_challenge_result(actor_idx=1, actor_won=False)

        cfg = _phase3_config()
        feat = features_from_observation(obs, "Tax", None, "Duke", [], cfg)
        opp_start = 104 + cfg.belief_dims
        opp_section = feat[opp_start: opp_start + cfg.opp_model_dims]

        # P1's credibility should be < 1.0 (known bluffer)
        # Player indices: opp_section[1] corresponds to player 1
        assert opp_section[1] < 1.0

    def test_base_section_unchanged_by_belief(self):
        """The first 104 dims must be identical whether belief is included or not."""
        obs = _obs(with_belief=True, with_opp_model=True)
        feat_p2 = features_from_observation(obs, "Tax", None, "Duke", [], _phase2_config())
        feat_p3 = features_from_observation(obs, "Tax", None, "Duke", [], _phase3_config())
        np.testing.assert_array_almost_equal(feat_p2, feat_p3[:104])

    def test_with_action_history(self):
        obs = _obs()
        history = ["Income", "Tax", "ForeignAid"]
        feat = features_from_observation(obs, "Steal", 1, "Captain", history, _phase2_config())
        assert feat.shape == (104,)
        # History section should be non-zero
        base_without_history = 104 - N_ACTIONS * HISTORY_LENGTH
        hist_section = feat[base_without_history: base_without_history + N_ACTIONS * HISTORY_LENGTH]
        assert np.any(hist_section > 0.0)

    def test_with_target(self):
        obs = _obs()
        feat = features_from_observation(obs, "Assassinate", 2, "Assassin", [], _phase2_config())
        assert feat.shape == (104,)
        assert not np.any(np.isnan(feat))

    def test_all_player_counts(self):
        for n in [2, 3, 4, 5, 6]:
            obs = _obs(n_players=n, with_belief=True, with_opp_model=True)
            cfg = FeatureConfig(max_players=6, include_belief=True, include_opp_model=True)
            feat = features_from_observation(obs, "Tax", None, "Duke", [], cfg)
            assert feat.shape == (cfg.feature_dim,)


# ===========================================================================
# Belief section encoding correctness
# ===========================================================================

class TestBeliefSectionEncoding:

    def test_observer_certainty_reflected_in_belief_section(self):
        """Observer holds Duke → their row in belief section should have Duke=1.0."""
        obs = _obs(with_belief=True, with_opp_model=False)
        cfg = _phase3_config(max_players=4)
        feat = features_from_observation(obs, "Tax", None, "Duke", [], cfg)

        # Belief section layout: [P0_probs(5), P1_probs(5), P2_probs(5), P3_probs(5)]
        belief_start = FeatureConfig(max_players=4).feature_dim  # base without belief
        # Actually compute base dim
        base_cfg = FeatureConfig(max_players=4, include_belief=False, include_opp_model=False)
        belief_start = base_cfg.feature_dim

        p0_duke_idx = belief_start + CARD_TO_IDX[Card.DUKE]
        p0_assn_idx = belief_start + CARD_TO_IDX[Card.ASSASSIN]

        # Observer holds Duke and Assassin → both should be 1.0
        assert abs(feat[p0_duke_idx] - 1.0) < 1e-6
        assert abs(feat[p0_assn_idx] - 1.0) < 1e-6

    def test_belief_section_values_in_unit_interval(self):
        obs = _obs(with_belief=True)
        cfg = _phase3_config()
        feat = features_from_observation(obs, "Income", None, None, [], cfg)
        belief_section = feat[104: 104 + cfg.belief_dims]
        assert np.all(belief_section >= 0.0)
        assert np.all(belief_section <= 1.0 + 1e-6)


# ===========================================================================
# Backward compatibility: Phase 1/2 agent interface unchanged
# ===========================================================================

class TestBackwardCompatibility:

    def test_game_state_get_observation_still_works(self):
        """GameState.get_observation() must still work identically as before."""
        state = GameState.new_game(["P0", "P1", "P2", "P3"], seed=42)
        obs = state.get_observation(0)
        assert obs.my_idx == 0
        assert len(obs.my_hand) == 2
        assert obs.belief_state is None
        assert obs.opponent_model is None

    def test_observation_to_dict_phase1_keys_present(self):
        state = GameState.new_game(["P0", "P1", "P2", "P3"], seed=1)
        obs = state.get_observation(1)
        d = obs.to_dict()
        for key in ["my_idx", "my_hand", "my_coins", "my_revealed",
                    "others", "deck_size", "current_player_idx", "turn_number"]:
            assert key in d

    def test_random_agent_works_without_belief(self):
        """RandomAgent should work unchanged — it doesn't use belief fields."""
        from coup.agents.random_agent import RandomAgent
        from coup.actions import get_legal_actions
        state = GameState.new_game(["P0", "P1", "P2", "P3"], seed=5)
        agent = RandomAgent()
        obs = state.get_observation(0)
        legal = get_legal_actions(state, 0)
        action = agent.choose_action(obs, legal)
        assert action in legal
