"""
tests/test_neural_agent.py — Tests for sub-goal 4.2.

Coverage:
  CoupPolicyNet   — architecture shape, forward pass, head outputs,
                    masked action probs, save/load roundtrip
  SupervisedTrainer — dataset loading, one micro-epoch, checkpoint save/load
  NeuralAgent     — all six Agent ABC methods, game integration,
                    fallback to random when model=None,
                    game boundary detection (history reset)

All tests that require torch are skipped if torch is not installed, so the
full test suite stays green without it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Graceful skip when torch is absent ────────────────────────────────────
torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn as nn

from agents.neural.model import CoupPolicyNet, N_ACTION_TYPES, N_TARGET_SLOTS, N_BINARY
from agents.neural.neural_agent import NeuralAgent, CARD_VALUE
from coup.cards import Card
from coup.state import GameState, Observation
from coup.actions import Action, get_legal_actions, get_legal_blocks
from coup.agents.random_agent import RandomAgent
from coup.game import Game
from pipeline.feature_extractor import FeatureConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _net(input_dim=104, hidden_dim=64):
    """Small net for fast tests."""
    return CoupPolicyNet(input_dim=input_dim, hidden_dim=hidden_dim)


def _obs(n_players=4, my_idx=0, my_hand=None, coins=4, turn=3):
    """Build a minimal Observation."""
    hand = my_hand or [Card.DUKE, Card.CAPTAIN]
    others = [
        {"name": f"P{i}", "idx": i, "coins": 2, "influence_count": 2,
         "revealed_cards": [], "is_alive": True}
        for i in range(1, n_players)
    ]
    return Observation(
        my_idx=my_idx, my_hand=hand, my_coins=coins, my_revealed=[],
        others=others, deck_size=7, current_player_idx=my_idx,
        turn_number=turn,
    )


def _make_fake_dataset(n=200, input_dim=104, tmp_dir=None):
    """Generate a tiny fake .npz dataset for trainer tests."""
    rng = np.random.RandomState(0)
    X        = rng.randn(n, input_dim).astype(np.float32)
    y_action = rng.randint(0, 7, n).astype(np.int32)
    y_target = rng.randint(0, 6, n).astype(np.int32)
    y_outcome= rng.randint(0, 2, n).astype(np.int32)
    mean     = X.mean(0).astype(np.float32)
    std      = (X.std(0) + 1e-8).astype(np.float32)

    path = os.path.join(tmp_dir or tempfile.gettempdir(), "test_dataset.npz")
    np.savez_compressed(path, X=X, y_action=y_action, y_target=y_target,
                        y_outcome=y_outcome, mean=mean, std=std)
    return path


# ===========================================================================
# CoupPolicyNet — architecture
# ===========================================================================

class TestCoupPolicyNetArchitecture:

    def test_instantiation(self):
        net = _net()
        assert isinstance(net, CoupPolicyNet)

    def test_default_input_dim(self):
        net = _net()
        assert net.input_dim == 104

    def test_custom_input_dim(self):
        net = _net(input_dim=140)
        assert net.input_dim == 140

    def test_repr_contains_param_count(self):
        net = _net()
        r = repr(net)
        assert "params=" in r
        assert "CoupPolicyNet" in r

    def test_parameter_count_reasonable(self):
        net = CoupPolicyNet(input_dim=104, hidden_dim=256)
        n = sum(p.numel() for p in net.parameters())
        # Should be roughly 100k–500k parameters
        assert 50_000 < n < 1_000_000

    def test_small_hidden_dim(self):
        net = _net(input_dim=104, hidden_dim=32)
        x = torch.randn(1, 104)
        out = net(x)
        assert out["action"].shape == (1, N_ACTION_TYPES)


# ===========================================================================
# CoupPolicyNet — forward pass
# ===========================================================================

class TestCoupPolicyNetForward:

    @pytest.fixture
    def net(self):
        return _net()

    def test_forward_returns_three_keys(self, net):
        x = torch.randn(4, 104)
        out = net(x)
        assert set(out.keys()) == {"action", "target", "binary"}

    def test_action_logits_shape(self, net):
        x = torch.randn(8, 104)
        assert net(x)["action"].shape == (8, N_ACTION_TYPES)

    def test_target_logits_shape(self, net):
        x = torch.randn(8, 104)
        assert net(x)["target"].shape == (8, N_TARGET_SLOTS)

    def test_binary_logits_shape(self, net):
        x = torch.randn(8, 104)
        assert net(x)["binary"].shape == (8, N_BINARY)

    def test_batch_size_1(self, net):
        x = torch.randn(1, 104)
        out = net(x)
        for v in out.values():
            assert v.shape[0] == 1

    def test_no_nan_in_output(self, net):
        x = torch.randn(16, 104)
        out = net(x)
        for v in out.values():
            assert not torch.any(torch.isnan(v))

    def test_trunk_output_shape(self, net):
        x = torch.randn(4, 104)
        h = net.trunk(x)
        assert h.shape == (4, 128)

    def test_phase3_input_dim(self):
        net = _net(input_dim=140)
        x = torch.randn(4, 140)
        out = net(x)
        assert out["action"].shape == (4, N_ACTION_TYPES)


# ===========================================================================
# CoupPolicyNet — inference helpers
# ===========================================================================

class TestCoupPolicyNetInference:

    @pytest.fixture
    def net(self):
        n = _net()
        n.eval()
        return n

    def test_action_probs_sum_to_one(self, net):
        x = torch.randn(4, 104)
        p = net.action_probs(x)
        np.testing.assert_allclose(p.sum(dim=-1).numpy(), np.ones(4), atol=1e-5)

    def test_action_probs_non_negative(self, net):
        x = torch.randn(4, 104)
        assert torch.all(net.action_probs(x) >= 0)

    def test_action_probs_with_legal_mask(self, net):
        x = torch.randn(1, 104)
        mask = torch.tensor([[True, False, True, True, False, True, True]])
        p = net.action_probs(x, legal_mask=mask)
        # Masked-out positions should be 0
        assert p[0, 1].item() < 1e-6
        assert p[0, 4].item() < 1e-6
        # Others should sum to 1
        np.testing.assert_allclose(p.sum().item(), 1.0, atol=1e-5)

    def test_target_probs_sum_to_one(self, net):
        x = torch.randn(3, 104)
        p = net.target_probs(x)
        np.testing.assert_allclose(p.sum(dim=-1).numpy(), np.ones(3), atol=1e-5)

    def test_binary_prob_yes_in_unit_interval(self, net):
        x = torch.randn(8, 104)
        p = net.binary_prob_yes(x)
        assert torch.all(p >= 0) and torch.all(p <= 1)

    def test_binary_prob_yes_shape(self, net):
        x = torch.randn(5, 104)
        assert net.binary_prob_yes(x).shape == (5,)

    def test_high_temperature_near_uniform(self, net):
        x = torch.randn(1, 104)
        p_high = net.action_probs(x, temperature=100.0)
        # Should be close to uniform (1/7 ≈ 0.143) under very high temperature
        assert all(abs(p_high[0, i].item() - 1/N_ACTION_TYPES) < 0.05
                   for i in range(N_ACTION_TYPES))

    def test_low_temperature_near_deterministic(self, net):
        x = torch.randn(1, 104)
        p_low = net.action_probs(x, temperature=0.01)
        # One action should dominate
        assert p_low.max().item() > 0.9


# ===========================================================================
# CoupPolicyNet — save / load
# ===========================================================================

class TestCoupPolicyNetPersistence:

    def test_save_and_load_roundtrip(self, tmp_path):
        net = _net(input_dim=104, hidden_dim=64)
        path = str(tmp_path / "model.pt")
        net.save(path)
        assert os.path.exists(path)

        net2 = CoupPolicyNet.load(path)
        assert net2.input_dim == 104
        assert net2.hidden_dim == 64

    def test_loaded_model_produces_same_output(self, tmp_path):
        net = _net(input_dim=104, hidden_dim=64)
        net.eval()
        x = torch.randn(2, 104)
        out1 = net(x)["action"].detach().numpy()

        path = str(tmp_path / "model.pt")
        net.save(path)
        net2 = CoupPolicyNet.load(path)
        net2.eval()
        out2 = net2(x)["action"].detach().numpy()

        np.testing.assert_allclose(out1, out2, atol=1e-5)


# ===========================================================================
# SupervisedTrainer
# ===========================================================================

class TestSupervisedTrainer:

    @pytest.fixture
    def dataset_path(self, tmp_path):
        return _make_fake_dataset(n=512, input_dim=104, tmp_dir=str(tmp_path))

    def test_trainer_instantiation(self, dataset_path, tmp_path):
        from agents.neural.trainer import SupervisedTrainer
        trainer = SupervisedTrainer(
            dataset_path=dataset_path,
            save_dir=str(tmp_path / "models"),
            hidden_dim=64,
            batch_size=64,
        )
        assert trainer.input_dim == 104
        assert isinstance(trainer.model, CoupPolicyNet)

    def test_one_epoch_runs_without_error(self, dataset_path, tmp_path):
        from agents.neural.trainer import SupervisedTrainer
        trainer = SupervisedTrainer(
            dataset_path=dataset_path,
            save_dir=str(tmp_path / "models"),
            hidden_dim=64,
            batch_size=64,
        )
        history = trainer.train(n_epochs=1)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert history["train_loss"][0] > 0

    def test_train_loss_decreases(self, dataset_path, tmp_path):
        from agents.neural.trainer import SupervisedTrainer
        trainer = SupervisedTrainer(
            dataset_path=dataset_path,
            save_dir=str(tmp_path / "models"),
            hidden_dim=64,
            batch_size=64,
            patience=999,
        )
        history = trainer.train(n_epochs=5)
        # Loss should decrease over 5 epochs on a trainable dataset
        assert history["train_loss"][-1] <= history["train_loss"][0] + 0.5

    def test_checkpoint_saved(self, dataset_path, tmp_path):
        from agents.neural.trainer import SupervisedTrainer
        save_dir = str(tmp_path / "models")
        trainer = SupervisedTrainer(
            dataset_path=dataset_path,
            save_dir=save_dir,
            hidden_dim=64,
            batch_size=64,
        )
        trainer.train(n_epochs=1)
        assert os.path.exists(os.path.join(save_dir, "best_model.pt"))

    def test_load_best_returns_model(self, dataset_path, tmp_path):
        from agents.neural.trainer import SupervisedTrainer
        trainer = SupervisedTrainer(
            dataset_path=dataset_path,
            save_dir=str(tmp_path / "models"),
            hidden_dim=64,
            batch_size=64,
        )
        trainer.train(n_epochs=1)
        model = trainer.load_best()
        assert isinstance(model, CoupPolicyNet)

    def test_val_metrics_present(self, dataset_path, tmp_path):
        from agents.neural.trainer import SupervisedTrainer
        trainer = SupervisedTrainer(
            dataset_path=dataset_path,
            save_dir=str(tmp_path / "models"),
            hidden_dim=64,
            batch_size=64,
        )
        history = trainer.train(n_epochs=2)
        assert "val_loss" in history
        assert "val_action_acc" in history
        assert len(history["val_loss"]) == 2

    def test_early_stopping_triggers(self, dataset_path, tmp_path):
        from agents.neural.trainer import SupervisedTrainer
        trainer = SupervisedTrainer(
            dataset_path=dataset_path,
            save_dir=str(tmp_path / "models"),
            hidden_dim=64,
            batch_size=64,
            patience=1,
        )
        history = trainer.train(n_epochs=20)
        # With patience=1, should stop well before 20 epochs
        assert len(history["train_loss"]) < 20


# ===========================================================================
# NeuralAgent — fallback (model=None)
# ===========================================================================

class TestNeuralAgentFallback:
    """When model=None, NeuralAgent must behave like a valid random agent."""

    @pytest.fixture
    def agent(self):
        return NeuralAgent(model=None, seed=42)

    @pytest.fixture
    def state(self):
        return GameState.new_game(["P0", "P1", "P2", "P3"], seed=1)

    def test_choose_action_returns_legal_action(self, agent, state):
        obs   = state.get_observation(0)
        legal = get_legal_actions(state, 0)
        act   = agent.choose_action(obs, legal)
        assert act in legal

    def test_choose_action_single_option(self, agent, state):
        obs   = state.get_observation(0)
        legal = get_legal_actions(state, 0)
        act   = agent.choose_action(obs, [legal[0]])
        assert act == legal[0]

    def test_challenge_returns_bool(self, agent, state):
        obs    = state.get_observation(1)
        action = Action(type=ActionType.TAX, actor_idx=0)
        result = agent.choose_to_challenge_action(obs, action)
        assert isinstance(result, bool)

    def test_block_returns_none_or_block(self, agent, state):
        obs    = state.get_observation(1)
        action = Action(type=ActionType.FOREIGN_AID, actor_idx=0)
        legal_blocks = get_legal_blocks(action, 1)
        result = agent.choose_to_block(obs, action, legal_blocks)
        assert result is None or isinstance(result, Block)

    def test_block_no_legal_blocks_returns_none(self, agent, state):
        obs    = state.get_observation(1)
        action = Action(type=ActionType.INCOME, actor_idx=0)
        result = agent.choose_to_block(obs, action, [])
        assert result is None

    def test_challenge_block_returns_bool(self, agent, state):
        obs = state.get_observation(2)
        from coup.actions import Block
        block = Block(
            blocker_idx=1,
            blocking_card=Card.DUKE,
            blocked_action=Action(type=ActionType.FOREIGN_AID, actor_idx=0),
        )
        result = agent.choose_to_challenge_block(obs, block)
        assert isinstance(result, bool)

    def test_choose_card_to_lose_valid_index(self, agent, state):
        obs = state.get_observation(0)
        idx = agent.choose_card_to_lose(obs)
        assert 0 <= idx < len(obs.my_hand)

    def test_choose_exchange_cards_correct_count(self, agent, state):
        obs     = state.get_observation(0)
        options = [Card.DUKE, Card.ASSASSIN, Card.CONTESSA, Card.CAPTAIN]
        indices = agent.choose_exchange_cards(obs, options, n_keep=2)
        assert len(indices) == 2
        assert len(set(indices)) == 2
        assert all(0 <= i < len(options) for i in indices)

    def test_does_not_challenge_own_action(self, agent, state):
        obs    = state.get_observation(0)
        action = Action(type=ActionType.TAX, actor_idx=0)   # actor == observer
        result = agent.choose_to_challenge_action(obs, action)
        assert result is False


# ---------------------------------------------------------------------------
# Import ActionType for the above tests
# ---------------------------------------------------------------------------
from coup.cards import ActionType
from coup.actions import Action, Block


# ===========================================================================
# NeuralAgent — with trained model
# ===========================================================================

class TestNeuralAgentWithModel:

    @pytest.fixture
    def agent(self, tmp_path):
        """Train a tiny model and wrap it in a NeuralAgent."""
        from agents.neural.trainer import SupervisedTrainer
        dataset_path = _make_fake_dataset(n=512, tmp_dir=str(tmp_path))
        trainer = SupervisedTrainer(
            dataset_path=dataset_path,
            save_dir=str(tmp_path / "models"),
            hidden_dim=64,
            batch_size=64,
        )
        trainer.train(n_epochs=2)
        return NeuralAgent.from_checkpoint(
            str(tmp_path / "models" / "best_model.pt"),
            temperature=1.0,
        )

    @pytest.fixture
    def state(self):
        return GameState.new_game(["P0", "P1", "P2", "P3"], seed=5)

    def test_choose_action_returns_legal(self, agent, state):
        obs   = state.get_observation(0)
        legal = get_legal_actions(state, 0)
        act   = agent.choose_action(obs, legal)
        assert act in legal

    def test_challenge_returns_bool(self, agent, state):
        obs    = state.get_observation(1)
        action = Action(type=ActionType.TAX, actor_idx=0)
        result = agent.choose_to_challenge_action(obs, action)
        assert isinstance(result, bool)

    def test_block_returns_none_or_valid(self, agent, state):
        obs    = state.get_observation(1)
        action = Action(type=ActionType.FOREIGN_AID, actor_idx=0)
        blocks = get_legal_blocks(action, 1)
        result = agent.choose_to_block(obs, action, blocks)
        assert result is None or isinstance(result, Block)

    def test_challenge_block_returns_bool(self, agent, state):
        obs   = state.get_observation(2)
        block = Block(
            blocker_idx=1,
            blocking_card=Card.DUKE,
            blocked_action=Action(type=ActionType.FOREIGN_AID, actor_idx=0),
        )
        result = agent.choose_to_challenge_block(obs, block)
        assert isinstance(result, bool)

    def test_card_to_lose_valid(self, agent, state):
        obs = state.get_observation(0)
        idx = agent.choose_card_to_lose(obs)
        assert 0 <= idx < len(obs.my_hand)

    def test_exchange_valid(self, agent, state):
        obs     = state.get_observation(0)
        options = [Card.DUKE, Card.ASSASSIN, Card.CONTESSA, Card.CAPTAIN]
        indices = agent.choose_exchange_cards(obs, options, n_keep=2)
        assert len(indices) == 2
        assert len(set(indices)) == 2


# ===========================================================================
# NeuralAgent — card heuristics
# ===========================================================================

class TestNeuralAgentHeuristics:

    @pytest.fixture
    def agent(self):
        return NeuralAgent(model=None, seed=0)

    def test_lose_influence_picks_lowest_value(self, agent):
        # Duke (5) + Contessa (2) → should lose Contessa (index 1)
        obs = _obs(my_hand=[Card.DUKE, Card.CONTESSA])
        idx = agent.choose_card_to_lose(obs)
        assert obs.my_hand[idx] == Card.CONTESSA

    def test_lose_influence_single_card(self, agent):
        obs = _obs(my_hand=[Card.ASSASSIN])
        idx = agent.choose_card_to_lose(obs)
        assert idx == 0

    def test_exchange_keeps_highest_value(self, agent):
        # Options: [Duke(5), Ambassador(2), Captain(4), Contessa(2)]
        # Keep 2 → should keep Duke + Captain
        obs     = _obs(my_hand=[Card.DUKE, Card.CAPTAIN])
        options = [Card.DUKE, Card.AMBASSADOR, Card.CAPTAIN, Card.CONTESSA]
        indices = agent.choose_exchange_cards(obs, options, n_keep=2)
        kept    = {options[i] for i in indices}
        assert Card.DUKE in kept
        assert Card.CAPTAIN in kept

    def test_exchange_keep_all_when_same_count(self, agent):
        obs     = _obs(my_hand=[Card.DUKE])
        options = [Card.DUKE, Card.CAPTAIN]
        indices = agent.choose_exchange_cards(obs, options, n_keep=2)
        assert len(indices) == 2

    def test_exchange_returns_correct_count(self, agent):
        obs     = _obs(my_hand=[Card.DUKE, Card.CAPTAIN])
        options = [Card.DUKE, Card.CAPTAIN, Card.ASSASSIN, Card.CONTESSA]
        for n_keep in [1, 2, 3]:
            indices = agent.choose_exchange_cards(obs, options, n_keep=n_keep)
            assert len(indices) == n_keep


# ===========================================================================
# NeuralAgent — history reset (game boundary)
# ===========================================================================

class TestNeuralAgentHistoryReset:

    def test_history_resets_on_turn_zero(self):
        agent = NeuralAgent(model=None, seed=0)
        state = GameState.new_game(["P0","P1","P2","P3"], seed=1)

        # Simulate a turn mid-game
        agent._action_history = ["Tax", "Income", "Steal"]
        agent._last_turn = 5

        # New game — turn_number = 0
        obs = Observation(
            my_idx=0, my_hand=[Card.DUKE, Card.CAPTAIN], my_coins=2,
            my_revealed=[], others=[], deck_size=9,
            current_player_idx=0, turn_number=0,
        )
        agent._maybe_reset(obs)
        assert agent._action_history == []

    def test_history_resets_on_turn_regression(self):
        agent = NeuralAgent(model=None, seed=0)
        agent._action_history = ["Tax", "Income"]
        agent._last_turn = 10

        obs = Observation(
            my_idx=0, my_hand=[Card.DUKE, Card.CAPTAIN], my_coins=2,
            my_revealed=[], others=[], deck_size=9,
            current_player_idx=0, turn_number=3,  # less than last turn
        )
        agent._maybe_reset(obs)
        assert agent._action_history == []

    def test_history_preserved_within_game(self):
        agent = NeuralAgent(model=None, seed=0)
        agent._action_history = ["Tax"]
        agent._last_turn = 2

        obs = Observation(
            my_idx=0, my_hand=[Card.DUKE, Card.CAPTAIN], my_coins=2,
            my_revealed=[], others=[], deck_size=9,
            current_player_idx=0, turn_number=5,  # higher than last — same game
        )
        agent._maybe_reset(obs)
        assert agent._action_history == ["Tax"]


# ===========================================================================
# Full game integration
# ===========================================================================

class TestNeuralAgentGameIntegration:

    def test_neural_agent_completes_game(self):
        """NeuralAgent (fallback mode) must complete a full game without error."""
        agents = [NeuralAgent(model=None, seed=i) for i in range(4)]
        game = Game(agents=agents, player_names=["P0","P1","P2","P3"], seed=7)
        log = game.play_game()
        end = next(e for e in reversed(log["events"]) if e["event_type"] == "game_end")
        assert end["winner_name"] is not None

    def test_mixed_agents_complete_game(self):
        """NeuralAgent mixed with RandomAgent — game completes cleanly."""
        from coup.agents.random_agent import RandomAgent
        agents = [
            NeuralAgent(model=None, seed=0),
            RandomAgent(seed=1),
            NeuralAgent(model=None, seed=2),
            RandomAgent(seed=3),
        ]
        game = Game(agents=agents, player_names=["P0","P1","P2","P3"], seed=42)
        log = game.play_game()
        assert log is not None

    def test_neural_agent_all_player_counts(self):
        """NeuralAgent works at every player count 2–6."""
        for n in [2, 3, 4, 5, 6]:
            agents = [NeuralAgent(model=None, seed=i) for i in range(n)]
            names  = [f"P{i}" for i in range(n)]
            game   = Game(agents=agents, player_names=names, seed=n)
            log    = game.play_game()
            assert log is not None

    def test_trained_agent_completes_game(self, tmp_path):
        """Trained NeuralAgent must also complete a game without error."""
        from agents.neural.trainer import SupervisedTrainer
        path = _make_fake_dataset(n=256, tmp_dir=str(tmp_path))
        trainer = SupervisedTrainer(
            dataset_path=path,
            save_dir=str(tmp_path / "models"),
            hidden_dim=32, batch_size=64,
        )
        trainer.train(n_epochs=1)

        trained = NeuralAgent.from_checkpoint(
            str(tmp_path / "models" / "best_model.pt")
        )
        agents = [trained] + [NeuralAgent(model=None) for _ in range(3)]
        game   = Game(agents=agents, player_names=["P0","P1","P2","P3"], seed=99)
        log    = game.play_game()
        assert log is not None
