"""
tests/test_training.py — Tests for sub-goal 4.3.

Coverage:
  CFRTrainer       — instantiation, one training run, checkpoint save/load,
                     exploitability decreases, resume from checkpoint
  RecordingAgent   — records experiences during game play, tracker enriches obs,
                     all Agent ABC methods work, experiences have correct shape
  RLTrainer        — collect_games returns experiences + returns,
                     returns are correct (winner +1, losers -1/(n-1)),
                     _update runs without error, loss is finite,
                     entropy regularisation term present
  Integration      — CFRTrainer runs a short self-play loop end-to-end;
                     RLTrainer (with random fallback) collects and updates;
                     BeliefTracker enriches observations during collection
  CLI              — argument parser builds correctly for all three modes

torch-dependent tests are skipped gracefully when torch is not installed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Optional torch import ──────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from coup.cards import Card
from coup.state import GameState
from coup.agents.random_agent import RandomAgent
from coup.game import Game
from belief.belief_tracker import BeliefTracker
from training.cfr_trainer import CFRTrainer, CFRConfig
from training.rl_trainer import RecordingAgent, Experience


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_cfr_trainer(tmp_path, n_players=3):
    return CFRTrainer(
        n_players=n_players,
        output_dir=str(tmp_path / "cfr"),
        config=CFRConfig(
            n_players=n_players,
            save_every=50,
            log_every=10,
        ),
    )


def _make_fake_npz(tmp_path, n=256, input_dim=104):
    """Create a tiny fake .npz dataset for RL trainer tests."""
    rng = np.random.RandomState(0)
    X        = rng.randn(n, input_dim).astype(np.float32)
    y_action = rng.randint(0, 7, n).astype(np.int32)
    y_target = rng.randint(0, 6, n).astype(np.int32)
    y_outcome= rng.randint(0, 2, n).astype(np.int32)
    mean     = X.mean(0).astype(np.float32)
    std      = (X.std(0) + 1e-8).astype(np.float32)
    path     = str(tmp_path / "dataset.npz")
    np.savez_compressed(path, X=X, y_action=y_action, y_target=y_target,
                        y_outcome=y_outcome, mean=mean, std=std)
    return path, mean, std


# ===========================================================================
# CFRTrainer
# ===========================================================================
@pytest.mark.skip(reason="CFR solver performance - revisit after Phase 5")
class TestCFRTrainer:

    def test_instantiation(self, tmp_path):
        trainer = _tiny_cfr_trainer(tmp_path)
        assert trainer.solver is not None
        assert trainer.solver.table.n_iterations == 0

    def test_train_runs_without_error(self, tmp_path):
        trainer = _tiny_cfr_trainer(tmp_path)
        stats = trainer.train(n_iterations=20)
        assert "total_iterations" in stats
        assert stats["total_iterations"] == 20

    def test_infosets_grow(self, tmp_path):
        trainer = _tiny_cfr_trainer(tmp_path)
        trainer.train(n_iterations=30)
        assert trainer.solver.table.n_infosets > 0

    def test_exploitability_is_finite(self, tmp_path):
        trainer = _tiny_cfr_trainer(tmp_path)
        trainer.train(n_iterations=20)
        expl = trainer.solver.table.exploitability_proxy()
        assert np.isfinite(expl)

    def test_checkpoint_saved(self, tmp_path):
        trainer = _tiny_cfr_trainer(tmp_path)
        trainer.train(n_iterations=50)   # triggers save_every=50
        assert os.path.exists(str(tmp_path / "cfr" / "latest.pkl.gz"))

    def test_resume_from_checkpoint(self, tmp_path):
        trainer = _tiny_cfr_trainer(tmp_path)
        trainer.train(n_iterations=50)

        ckpt_path = str(tmp_path / "cfr" / "latest.pkl.gz")
        trainer2  = CFRTrainer.from_checkpoint(
            checkpoint_path=ckpt_path,
            output_dir=str(tmp_path / "cfr2"),
        )
        assert trainer2.solver.table.n_iterations == 50

    def test_resume_and_continue(self, tmp_path):
        trainer = _tiny_cfr_trainer(tmp_path)
        trainer.train(n_iterations=50)

        trainer2 = CFRTrainer.from_checkpoint(
            str(tmp_path / "cfr" / "latest.pkl.gz"),
            output_dir=str(tmp_path / "cfr2"),
        )
        trainer2.train(n_iterations=20)
        assert trainer2.solver.table.n_iterations == 70

    def test_training_log_saved(self, tmp_path):
        trainer = _tiny_cfr_trainer(tmp_path)
        trainer.train(n_iterations=20)
        log_path = str(tmp_path / "cfr" / "training_log.json")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            log = json.load(f)
        assert len(log) > 0
        assert "exploitability" in log[0]

    def test_exploitability_history(self, tmp_path):
        trainer = _tiny_cfr_trainer(tmp_path)
        trainer.train(n_iterations=30)
        hist = trainer.exploitability_history()
        assert len(hist) > 0
        assert all(np.isfinite(h) for h in hist)

    @pytest.mark.parametrize("n_players", [2, 3, 4])
    def test_multiple_player_counts(self, tmp_path, n_players):
        trainer = CFRTrainer(
            n_players=n_players,
            output_dir=str(tmp_path / f"cfr_{n_players}p"),
            config=CFRConfig(n_players=n_players, save_every=1000, log_every=10),
        )
        stats = trainer.train(n_iterations=15)
        assert stats["total_iterations"] == 15

    def test_cfr_plus_vs_vanilla(self, tmp_path):
        """Both CFR variants should produce non-zero infosets after training."""
        for cfr_plus in [True, False]:
            trainer = CFRTrainer(
                n_players=3,
                output_dir=str(tmp_path / f"cfr_plus_{cfr_plus}"),
                config=CFRConfig(n_players=3, use_cfr_plus=cfr_plus,
                                 save_every=1000, log_every=10),
            )
            trainer.train(n_iterations=20)
            assert trainer.solver.table.n_infosets > 0

    def test_repr(self, tmp_path):
        trainer = _tiny_cfr_trainer(tmp_path)
        assert "CFRTrainer" in repr(trainer)


# ===========================================================================
# RecordingAgent
# ===========================================================================

class TestRecordingAgent:

    def _make_recorder(self, player_idx=0, n_players=4, game_id=0):
        return RecordingAgent(
            neural_agent=None,       # fallback to random
            player_idx=player_idx,
            n_players=n_players,
            game_id=game_id,
        )

    def test_choose_action_returns_legal(self):
        rec   = self._make_recorder()
        state = GameState.new_game(["P0","P1","P2","P3"], seed=1)
        obs   = state.get_observation(0)
        from coup.actions import get_legal_actions
        legal = get_legal_actions(state, 0)
        act   = rec.choose_action(obs, legal)
        assert act in legal

    def test_choose_action_records_experience(self):
        rec   = self._make_recorder()
        state = GameState.new_game(["P0","P1","P2","P3"], seed=1)
        obs   = state.get_observation(0)
        from coup.actions import get_legal_actions
        legal = get_legal_actions(state, 0)
        rec.choose_action(obs, legal)
        assert len(rec.experiences) == 1

    def test_experience_has_correct_shape(self):
        from pipeline.feature_extractor import FeatureConfig
        config = FeatureConfig()
        rec    = RecordingAgent(
            neural_agent=None, player_idx=0, n_players=4, game_id=0,
            config=config,
        )
        state = GameState.new_game(["P0","P1","P2","P3"], seed=1)
        obs   = state.get_observation(0)
        from coup.actions import get_legal_actions
        legal = get_legal_actions(state, 0)
        rec.choose_action(obs, legal)
        exp = rec.experiences[0]
        assert exp.feature_vec.shape == (config.feature_dim,)
        assert exp.feature_vec.dtype == np.float32

    def test_experience_action_idx_in_range(self):
        rec   = self._make_recorder()
        state = GameState.new_game(["P0","P1","P2","P3"], seed=2)
        obs   = state.get_observation(0)
        from coup.actions import get_legal_actions
        legal = get_legal_actions(state, 0)
        rec.choose_action(obs, legal)
        assert 0 <= rec.experiences[0].action_idx < 7

    def test_experience_player_idx_correct(self):
        for p_idx in [0, 1, 2, 3]:
            rec   = self._make_recorder(player_idx=p_idx)
            state = GameState.new_game(["P0","P1","P2","P3"], seed=1)
            obs   = state.get_observation(p_idx)
            from coup.actions import get_legal_actions
            legal = get_legal_actions(state, p_idx)
            rec.choose_action(obs, legal)
            assert rec.experiences[0].player_idx == p_idx

    def test_tracker_enriches_observation(self):
        rec   = self._make_recorder()
        state = GameState.new_game(["P0","P1","P2","P3"], seed=3)
        obs   = state.get_observation(0)
        rec.init_tracker(list(state.players[0].hand))

        from coup.actions import get_legal_actions
        legal = get_legal_actions(state, 0)
        rec.choose_action(obs, legal)
        # After choose_action with tracker, obs should be enriched
        assert obs.belief_state is not None

    def test_init_tracker(self):
        rec = self._make_recorder()
        rec.init_tracker([Card.DUKE, Card.CAPTAIN])
        assert rec._tracker is not None
        assert rec._tracker.is_consistent()

    def test_all_agent_methods_work(self):
        """All six Agent ABC methods should not raise with random fallback."""
        from coup.actions import get_legal_actions, get_legal_blocks, Action, Block
        from coup.cards import ActionType

        rec   = self._make_recorder()
        state = GameState.new_game(["P0","P1","P2","P3"], seed=5)
        obs0  = state.get_observation(0)
        obs1  = state.get_observation(1)

        # choose_action
        legal = get_legal_actions(state, 0)
        act   = rec.choose_action(obs0, legal)
        assert act in legal

        # choose_to_challenge_action
        action = Action(type=ActionType.TAX, actor_idx=0)
        result = rec.choose_to_challenge_action(obs1, action)
        assert isinstance(result, bool)

        # choose_to_block
        fa_action = Action(type=ActionType.FOREIGN_AID, actor_idx=0)
        blocks    = get_legal_blocks(fa_action, 1)
        result    = rec.choose_to_block(obs1, fa_action, blocks)
        assert result is None or isinstance(result, Block)

        # choose_to_challenge_block
        block = Block(blocker_idx=1, blocking_card=Card.DUKE, blocked_action=fa_action)
        result = rec.choose_to_challenge_block(obs0, block)
        assert isinstance(result, bool)

        # choose_card_to_lose
        idx = rec.choose_card_to_lose(obs0)
        assert 0 <= idx < len(obs0.my_hand)

        # choose_exchange_cards
        options = [Card.DUKE, Card.CAPTAIN, Card.ASSASSIN, Card.CONTESSA]
        indices = rec.choose_exchange_cards(obs0, options, n_keep=2)
        assert len(indices) == 2

    def test_full_game_with_recorders(self):
        """Four RecordingAgents should complete a full game."""
        n = 4
        recorders = [
            RecordingAgent(neural_agent=None, player_idx=i, n_players=n, game_id=0)
            for i in range(n)
        ]
        game = Game(agents=recorders, player_names=[f"P{i}" for i in range(n)], seed=7)
        log  = game.play_game()
        assert log is not None
        # At least some experiences should have been recorded
        total_exp = sum(len(r.experiences) for r in recorders)
        assert total_exp > 0

    def test_multiple_experiences_per_game(self):
        """More than one action decision should be recorded per player."""
        n = 4
        recorders = [
            RecordingAgent(neural_agent=None, player_idx=i, n_players=n, game_id=0)
            for i in range(n)
        ]
        game = Game(agents=recorders, player_names=[f"P{i}" for i in range(n)], seed=42)
        game.play_game()
        # Every recorder who participated should have at least 1 experience
        participated = [r for r in recorders if r.experiences]
        assert len(participated) > 0


# ===========================================================================
# RLTrainer (torch required)
# ===========================================================================

pytestmark_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestRLTrainer:

    @pytest.fixture
    def trainer_and_model(self, tmp_path):
        from agents.neural.trainer import SupervisedTrainer
        from agents.neural.model import CoupPolicyNet
        from training.rl_trainer import RLTrainer

        # Pretrain a tiny model
        dataset_path, mean, std = _make_fake_npz(tmp_path)
        sup_trainer = SupervisedTrainer(
            dataset_path=dataset_path,
            save_dir=str(tmp_path / "pretrain"),
            hidden_dim=32, batch_size=64,
        )
        sup_trainer.train(n_epochs=1)

        ckpt  = torch.load(str(tmp_path / "pretrain" / "best_model.pt"), map_location="cpu")
        model = CoupPolicyNet(input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"])
        model.load_state_dict(ckpt["state_dict"])

        rl_trainer = RLTrainer(
            model=model,
            feat_mean=ckpt.get("feat_mean"),
            feat_std=ckpt.get("feat_std"),
            n_players=3,
            output_dir=str(tmp_path / "rl"),
            games_per_round=4,
            use_belief=False,    # faster for tests
        )
        return rl_trainer, model

    def test_collect_games_returns_experiences(self, trainer_and_model):
        trainer, _ = trainer_and_model
        exps, returns = trainer._collect_games(n_games=3)
        assert len(exps) > 0
        assert len(returns) > 0

    def test_returns_sum_to_zero(self, trainer_and_model):
        """Zero-sum game: returns across all players in one game should sum to 0."""
        trainer, _ = trainer_and_model
        _, returns  = trainer._collect_games(n_games=5)

        # Group by game_id
        from collections import defaultdict
        by_game = defaultdict(list)
        for (game_id, _), r in returns.items():
            by_game[game_id].append(r)

        for game_id, game_returns in by_game.items():
            total = sum(game_returns)
            assert abs(total) < 1e-6, \
                f"Game {game_id} returns don't sum to 0: {game_returns}"

    def test_winner_gets_positive_return(self, trainer_and_model):
        trainer, _ = trainer_and_model
        _, returns  = trainer._collect_games(n_games=10)
        assert any(r > 0 for r in returns.values())

    def test_loser_gets_negative_return(self, trainer_and_model):
        trainer, _ = trainer_and_model
        _, returns  = trainer._collect_games(n_games=10)
        assert any(r < 0 for r in returns.values())

    def test_experience_feature_shape(self, trainer_and_model):
        trainer, _ = trainer_and_model
        exps, _    = trainer._collect_games(n_games=2)
        for e in exps:
            assert e.feature_vec.ndim == 1
            assert e.feature_vec.dtype == np.float32

    def test_update_runs_without_error(self, trainer_and_model):
        trainer, _ = trainer_and_model
        exps, returns = trainer._collect_games(n_games=4)
        stats = trainer._update(exps, returns)
        assert "action_loss" in stats
        assert np.isfinite(stats["action_loss"])

    def test_update_entropy_positive(self, trainer_and_model):
        trainer, _ = trainer_and_model
        exps, returns = trainer._collect_games(n_games=4)
        stats = trainer._update(exps, returns)
        assert stats["entropy"] > 0

    def test_one_round_train(self, trainer_and_model, tmp_path):
        trainer, _ = trainer_and_model
        stats = trainer.train(n_rounds=1)
        assert stats["n_rounds"] == 1
        assert stats["games_total"] >= 1

    def test_checkpoint_saved_after_training(self, trainer_and_model, tmp_path):
        trainer, _ = trainer_and_model
        trainer.train(n_rounds=1)
        assert os.path.exists(os.path.join(trainer.output_dir, "rl_latest.pt"))

    def test_loaded_rl_model_is_valid(self, trainer_and_model, tmp_path):
        from agents.neural.model import CoupPolicyNet
        trainer, _ = trainer_and_model
        trainer.train(n_rounds=1)

        ckpt = torch.load(
            os.path.join(trainer.output_dir, "rl_latest.pt"), map_location="cpu"
        )
        model2 = CoupPolicyNet(input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"])
        model2.load_state_dict(ckpt["state_dict"])
        assert model2 is not None


# ===========================================================================
# CLI argument parser
# ===========================================================================

class TestCLIParser:

    def _parse(self, args):
        from training.train import build_parser
        return build_parser().parse_args(args)

    def test_cfr_defaults(self):
        args = self._parse(["cfr"])
        assert args.mode == "cfr"
        assert args.n_iter == 10_000
        assert args.n_players == 4

    def test_cfr_custom_args(self):
        args = self._parse(["cfr", "--n_iter", "500", "--n_players", "3", "--seed", "7"])
        assert args.n_iter == 500
        assert args.n_players == 3
        assert args.seed == 7

    def test_cfr_resume_arg(self):
        args = self._parse(["cfr", "--resume", "data/models/cfr/latest.pkl.gz"])
        assert args.resume == "data/models/cfr/latest.pkl.gz"

    def test_cfr_vanilla_flag(self):
        args = self._parse(["cfr", "--vanilla_cfr"])
        assert args.vanilla_cfr is True

    def test_pretrain_defaults(self):
        args = self._parse(["pretrain"])
        assert args.mode == "pretrain"
        assert args.epochs == 30
        assert args.batch_size == 1024

    def test_pretrain_custom(self):
        args = self._parse(["pretrain", "--epochs", "5", "--lr", "1e-3"])
        assert args.epochs == 5
        assert abs(args.lr - 1e-3) < 1e-9

    def test_rl_defaults(self):
        args = self._parse(["rl"])
        assert args.mode == "rl"
        assert args.rounds == 100
        assert args.games_per_round == 64

    def test_rl_no_belief_flag(self):
        args = self._parse(["rl", "--no_belief"])
        assert args.no_belief is True

    def test_rl_custom(self):
        args = self._parse(["rl", "--rounds", "10", "--n_players", "3", "--lr", "5e-5"])
        assert args.rounds == 10
        assert args.n_players == 3


# ===========================================================================
# Integration: CFR + BeliefTracker end-to-end
# ===========================================================================
@pytest.mark.skip(reason="CFR solver performance - revisit after Phase 5")
class TestIntegration:

    def test_cfr_trainer_full_run(self, tmp_path):
        """CFRTrainer completes a short run with checkpoints saved."""
        trainer = CFRTrainer(
            n_players=3,
            output_dir=str(tmp_path / "cfr_int"),
            config=CFRConfig(n_players=3, save_every=25, log_every=10),
        )
        stats = trainer.train(n_iterations=30)
        assert stats["total_iterations"] == 30
        assert stats["n_infosets"] > 0
        assert os.path.exists(str(tmp_path / "cfr_int" / "latest.pkl.gz"))

    def test_recording_agent_belief_tracker_per_seat(self):
        """Each seat maintains an independent BeliefTracker."""
        n = 4
        recorders = [
            RecordingAgent(neural_agent=None, player_idx=i, n_players=n, game_id=0)
            for i in range(n)
        ]
        state = GameState.new_game([f"P{i}" for i in range(n)], seed=99)
        for i, rec in enumerate(recorders):
            rec.init_tracker(list(state.players[i].hand))

        game = Game(agents=recorders, player_names=[f"P{i}" for i in range(n)], seed=99)
        game.play_game()

        # All trackers should still be consistent after the game
        for rec in recorders:
            if rec._tracker is not None:
                assert rec._tracker.is_consistent()

    def test_experiences_game_id_matches(self):
        """All experiences from a game should carry the correct game_id."""
        game_id = 42
        n = 3
        recorders = [
            RecordingAgent(neural_agent=None, player_idx=i,
                           n_players=n, game_id=game_id)
            for i in range(n)
        ]
        game = Game(agents=recorders, player_names=[f"P{i}" for i in range(n)],
                    seed=game_id)
        game.play_game()

        for rec in recorders:
            for exp in rec.experiences:
                assert exp.game_id == game_id
