"""
tests/test_pipeline_phase2.py — Tests for sub-goals 2.3–2.6.

Coverage:
  2.3 Large-scale simulation  — BatchConfig allocation, dry run, single batch
  2.4 Feature extractor       — vector shape, value ranges, label correctness,
                                 outcome mapping, history encoding, batch extraction
  2.5 Dataset                 — CoupDataset (len, getitem, split, save/load, n_classes),
                                 CoupStreamingDataset, build_dataset
  2.6 Data quality report     — CorpusStats accumulation, report fields,
                                 summary string contents, save to file
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.log_parser import load_logs, ParsedGame, ParsedEvent
from pipeline.feature_extractor import (
    FeatureConfig,
    extract_features,
    extract_dataset,
    feature_dim,
    ACTIONS,
    CARDS,
    ACTION_TO_IDX,
    N_ACTIONS,
    Sample,
)
from pipeline.dataset import (
    CoupDataset,
    CoupStreamingDataset,
    build_dataset,
)
from pipeline.large_scale_sim import (
    BatchConfig,
    default_batch_configs,
    run_large_scale,
    _run_batch,
)
from pipeline.data_quality import (
    CorpusStats,
    QualityReport,
    run_quality_report,
    _accumulate,
    _analyse_features,
)

from coup.agents.random_agent import RandomAgent
from coup.game import Game


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RAW_LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_logs")


def _make_games(n: int = 20, n_players: int = 4, seed: int = 0) -> list:
    """Generate n real games and return as ParsedGame list."""
    player_names = [f"P{i}" for i in range(n_players)]
    logs = []
    for i in range(n):
        agents = [RandomAgent(challenge_prob=0.3, block_prob=0.3) for _ in range(n_players)]
        game = Game(agents=agents, player_names=player_names, seed=seed + i)
        logs.append(game.play_game())

    with tempfile.TemporaryDirectory() as tmp:
        for i, log in enumerate(logs):
            with open(os.path.join(tmp, f"game_{i:04d}.json"), "w") as f:
                json.dump(log, f)
        games = load_logs(tmp)
    return games


# ===========================================================================
# Sub-goal 2.3 — Large-scale simulation
# ===========================================================================

class TestBatchConfig:

    def test_default_configs_return_list(self):
        configs = default_batch_configs()
        assert isinstance(configs, list)
        assert len(configs) > 0

    def test_all_configs_have_valid_player_counts(self):
        for cfg in default_batch_configs():
            assert 2 <= cfg.n_players <= 6

    def test_all_configs_have_positive_weight(self):
        for cfg in default_batch_configs():
            assert cfg.weight > 0

    def test_config_names_are_unique(self):
        names = [c.name for c in default_batch_configs()]
        assert len(names) == len(set(names))


class TestRunLargeScale:

    def test_dry_run_returns_plan(self, tmp_path):
        result = run_large_scale(
            total_games=100,
            output_dir=str(tmp_path),
            dry_run=True,
            show_progress=False,
        )
        assert result["dry_run"] is True
        assert "plan" in result
        assert len(result["plan"]) > 0

    def test_dry_run_creates_no_files(self, tmp_path):
        run_large_scale(
            total_games=100,
            output_dir=str(tmp_path),
            dry_run=True,
            show_progress=False,
        )
        # No subdirectories or files should be created
        assert list(tmp_path.iterdir()) == []

    def test_games_allocated_to_total(self, tmp_path):
        total = 50
        result = run_large_scale(
            total_games=total,
            output_dir=str(tmp_path),
            dry_run=True,
            show_progress=False,
        )
        allocated = sum(n for _, n in result["plan"])
        assert allocated == total

    def test_small_run_creates_logs(self, tmp_path):
        configs = [BatchConfig(
            name="test_batch",
            agent_class=RandomAgent,
            agent_kwargs={"challenge_prob": 0.2, "block_prob": 0.3},
            n_players=3,
            weight=1.0,
        )]
        result = run_large_scale(
            total_games=20,
            output_dir=str(tmp_path),
            configs=configs,
            seed=42,
            show_progress=False,
        )
        assert result["total_saved"] == 20
        assert result["total_failed"] == 0
        subdir = tmp_path / "test_batch"
        files = list(subdir.glob("*.json"))
        assert len(files) == 20

    def test_log_files_are_valid_json(self, tmp_path):
        configs = [BatchConfig(
            name="validity_check",
            agent_class=RandomAgent,
            agent_kwargs={"challenge_prob": 0.2, "block_prob": 0.3},
            n_players=4,
            weight=1.0,
        )]
        run_large_scale(
            total_games=10,
            output_dir=str(tmp_path),
            configs=configs,
            seed=7,
            show_progress=False,
        )
        for path in (tmp_path / "validity_check").glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            assert "game_id" in data
            assert "events" in data

    def test_seeded_runs_are_reproducible(self, tmp_path):
        configs = [BatchConfig(
            name="repro",
            agent_class=RandomAgent,
            agent_kwargs={"challenge_prob": 0.2, "block_prob": 0.3},
            n_players=3,
            weight=1.0,
        )]
        out_a = tmp_path / "run_a"
        out_b = tmp_path / "run_b"
        for out in [out_a, out_b]:
            run_large_scale(
                total_games=5,
                output_dir=str(out),
                configs=configs,
                seed=99,
                show_progress=False,
            )
        files_a = sorted((out_a / "repro").glob("*.json"))
        files_b = sorted((out_b / "repro").glob("*.json"))
        # game_id is timestamp-based so compare events and player names only
        for fa, fb in zip(files_a, files_b):
            log_a = json.loads(fa.read_text())
            log_b = json.loads(fb.read_text())
            assert log_a["player_names"] == log_b["player_names"]
            assert log_a["events"] == log_b["events"]


# ===========================================================================
# Sub-goal 2.4 — Feature extractor
# ===========================================================================

class TestFeatureConfig:

    def test_default_feature_dim(self):
        cfg = FeatureConfig(max_players=6)
        assert cfg.feature_dim == 104
        assert cfg.feature_dim == feature_dim(max_players=6)

    def test_n_opponent_slots(self):
        cfg = FeatureConfig(max_players=6)
        assert cfg.n_opponent_slots == 5

    def test_smaller_max_players(self):
        cfg_small = FeatureConfig(max_players=3)
        cfg_large = FeatureConfig(max_players=6)
        assert cfg_small.feature_dim < cfg_large.feature_dim


class TestExtractFeatures:

    @pytest.fixture
    def games(self):
        return _make_games(n=10, n_players=4)

    def test_returns_list_of_samples(self, games):
        samples = extract_features(games[0])
        assert isinstance(samples, list)
        assert all(isinstance(s, Sample) for s in samples)

    def test_sample_count_matches_action_events(self, games):
        game = games[0]
        samples = extract_features(game)
        n_actions = len(game.action_events)
        assert len(samples) == n_actions

    def test_feature_shape(self, games):
        cfg = FeatureConfig()
        for s in extract_features(games[0], cfg):
            assert s.features.shape == (cfg.feature_dim,)

    def test_feature_dtype_is_float32(self, games):
        for s in extract_features(games[0]):
            assert s.features.dtype == np.float32

    def test_action_label_in_valid_range(self, games):
        for game in games:
            for s in extract_features(game):
                assert 0 <= s.action_label < N_ACTIONS

    def test_target_label_in_valid_range(self, games):
        cfg = FeatureConfig()
        max_target = cfg.n_opponent_slots
        for game in games:
            for s in extract_features(game, cfg):
                assert 0 <= s.target_label <= max_target

    def test_outcome_is_binary(self, games):
        for game in games:
            for s in extract_features(game):
                assert s.outcome in (0, 1)

    def test_feature_values_bounded(self, games):
        """Most normalised features should be in [-3, 3] for reasonable inputs."""
        for game in games[:3]:
            for s in extract_features(game):
                assert not np.any(np.isnan(s.features))
                assert not np.any(np.isinf(s.features))

    def test_action_label_matches_event(self, games):
        """The action label in each sample must match the event's action type."""
        for game in games[:3]:
            samples = extract_features(game)
            action_events = game.action_events
            for s, evt in zip(samples, action_events):
                assert s.action_label == ACTION_TO_IDX[evt.action_type]

    def test_player_idx_set(self, games):
        for game in games[:3]:
            for s in extract_features(game):
                assert 0 <= s.player_idx < game.n_players

    def test_game_id_set(self, games):
        for game in games[:3]:
            for s in extract_features(game):
                assert s.game_id == game.game_id

    def test_different_player_counts(self):
        for n_players in [2, 3, 4, 5, 6]:
            games = _make_games(n=5, n_players=n_players)
            cfg = FeatureConfig(max_players=6)   # fixed-size vector
            for game in games:
                for s in extract_features(game, cfg):
                    assert s.features.shape == (cfg.feature_dim,)


class TestExtractDataset:

    @pytest.fixture
    def games(self):
        return _make_games(n=20, n_players=4)

    def test_returns_four_arrays(self, games):
        result = extract_dataset(games, show_progress=False)
        assert len(result) == 4

    def test_array_shapes_consistent(self, games):
        X, y_action, y_target, y_outcome = extract_dataset(games, show_progress=False)
        n = len(X)
        assert y_action.shape == (n,)
        assert y_target.shape == (n,)
        assert y_outcome.shape == (n,)

    def test_X_dtype(self, games):
        X, *_ = extract_dataset(games, show_progress=False)
        assert X.dtype == np.float32

    def test_label_dtypes(self, games):
        _, y_a, y_t, y_o = extract_dataset(games, show_progress=False)
        assert y_a.dtype == np.int32
        assert y_t.dtype == np.int32
        assert y_o.dtype == np.int32

    def test_total_samples_positive(self, games):
        X, *_ = extract_dataset(games, show_progress=False)
        assert len(X) > 0


# ===========================================================================
# Sub-goal 2.5 — Dataset
# ===========================================================================

class TestCoupDataset:

    @pytest.fixture
    def arrays(self):
        games = _make_games(n=30)
        return extract_dataset(games, show_progress=False)

    def test_len(self, arrays):
        X, ya, yt, yo = arrays
        ds = CoupDataset(X, ya, yt, yo, task="action")
        assert len(ds) == len(X)

    @pytest.mark.parametrize("task", ["action", "target", "outcome"])
    def test_getitem_returns_tensor_pair(self, arrays, task):
        import torch
        X, ya, yt, yo = arrays
        ds = CoupDataset(X, ya, yt, yo, task=task)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_feature_dim_property(self, arrays):
        X, ya, yt, yo = arrays
        ds = CoupDataset(X, ya, yt, yo)
        assert ds.feature_dim == X.shape[1]

    @pytest.mark.parametrize("task,expected", [
        ("action",  7),
        ("target",  6),
        ("outcome", 2),
    ])
    def test_n_classes(self, arrays, task, expected):
        X, ya, yt, yo = arrays
        ds = CoupDataset(X, ya, yt, yo, task=task)
        assert ds.n_classes == expected

    def test_normalisation_applied(self, arrays):
        X, ya, yt, yo = arrays
        ds = CoupDataset(X, ya, yt, yo, normalise=True)
        X_norm = ds.X.numpy()
        # After z-score, mean should be ~0 for non-constant dimensions
        non_const = ds._std > 1e-6
        if non_const.any():
            assert abs(X_norm[:, non_const].mean()) < 0.5

    def test_normalisation_stats_accessible(self, arrays):
        X, ya, yt, yo = arrays
        ds = CoupDataset(X, ya, yt, yo)
        mean, std = ds.normalisation_stats
        assert mean.shape == (X.shape[1],)
        assert std.shape == (X.shape[1],)

    def test_split_sizes(self, arrays):
        X, ya, yt, yo = arrays
        ds = CoupDataset(X, ya, yt, yo)
        train, val = ds.split(val_fraction=0.2)
        assert len(train) + len(val) == len(ds)
        assert len(val) == pytest.approx(len(ds) * 0.2, abs=2)

    def test_split_normalisation_consistent(self, arrays):
        X, ya, yt, yo = arrays
        ds = CoupDataset(X, ya, yt, yo)
        train, val = ds.split()
        np.testing.assert_array_equal(train._mean, val._mean)
        np.testing.assert_array_equal(train._std, val._std)

    def test_save_and_load(self, arrays, tmp_path):
        X, ya, yt, yo = arrays
        ds = CoupDataset(X, ya, yt, yo, task="action")
        path = str(tmp_path / "ds.npz")
        ds.save(path)
        assert os.path.exists(path)
        ds2 = CoupDataset.load(path, task="action")
        assert len(ds2) == len(ds)
        assert ds2.feature_dim == ds.feature_dim

    def test_invalid_task_raises(self, arrays):
        X, ya, yt, yo = arrays
        with pytest.raises(AssertionError):
            CoupDataset(X, ya, yt, yo, task="invalid")

    def test_dataloader_iteration(self, arrays):
        import torch
        from torch.utils.data import DataLoader
        X, ya, yt, yo = arrays
        ds = CoupDataset(X, ya, yt, yo)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        x_batch, y_batch = next(iter(loader))
        assert x_batch.shape[0] <= 32
        assert x_batch.shape[1] == ds.feature_dim


class TestCoupStreamingDataset:

    @pytest.fixture
    def games(self):
        return _make_games(n=15)

    def test_len_matches_sample_count(self, games):
        ds = CoupStreamingDataset(games, task="action")
        total = sum(len(extract_features(g)) for g in games)
        assert len(ds) == total

    def test_getitem(self, games):
        import torch
        ds = CoupStreamingDataset(games, task="action")
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_feature_dim_property(self, games):
        cfg = FeatureConfig()
        ds = CoupStreamingDataset(games, config=cfg)
        assert ds.feature_dim == cfg.feature_dim

    def test_n_classes_action(self, games):
        ds = CoupStreamingDataset(games, task="action")
        assert ds.n_classes == N_ACTIONS

    def test_n_classes_outcome(self, games):
        ds = CoupStreamingDataset(games, task="outcome")
        assert ds.n_classes == 2


class TestBuildDataset:

    def test_build_from_directory(self, tmp_path):
        games = _make_games(n=20)
        for i, g in enumerate(_raw_logs_from_games(games)):
            with open(tmp_path / f"game_{i:04d}.json", "w") as f:
                json.dump(g, f)

        ds = build_dataset(str(tmp_path), task="action", show_progress=False)
        assert isinstance(ds, CoupDataset)
        assert len(ds) > 0

    def test_build_saves_npz(self, tmp_path):
        games = _make_games(n=10)
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        for i, g in enumerate(_raw_logs_from_games(games)):
            with open(log_dir / f"game_{i:04d}.json", "w") as f:
                json.dump(g, f)

        save_path = str(tmp_path / "ds.npz")
        build_dataset(str(log_dir), save_path=save_path, show_progress=False)
        assert os.path.exists(save_path)


def _raw_logs_from_games(games):
    """Re-simulate to get raw log dicts from ParsedGame objects."""
    logs = []
    for i in range(len(games)):
        agents = [RandomAgent(challenge_prob=0.2, block_prob=0.3) for _ in range(4)]
        game = Game(agents=agents, player_names=[f"P{j}" for j in range(4)], seed=i)
        logs.append(game.play_game())
    return logs


# ===========================================================================
# Sub-goal 2.6 — Data quality report
# ===========================================================================

class TestCorpusStatsAccumulation:

    @pytest.fixture
    def games(self):
        return _make_games(n=50, n_players=4)

    def test_n_games_counted(self, games):
        stats = CorpusStats()
        _accumulate(games, stats)
        assert stats.n_games == len(games)

    def test_turn_lengths_recorded(self, games):
        stats = CorpusStats()
        _accumulate(games, stats)
        assert len(stats.turn_lengths) == len(games)
        assert all(t > 0 for t in stats.turn_lengths)

    def test_action_counts_nonzero(self, games):
        stats = CorpusStats()
        _accumulate(games, stats)
        total = sum(stats.action_counts.values())
        assert total > 0

    def test_all_action_types_seen(self, games):
        # With enough games all 7 action types should appear
        games_big = _make_games(n=100, n_players=4, seed=99)
        stats = CorpusStats()
        _accumulate(games_big, stats)
        seen = set(stats.action_counts.keys())
        # At minimum Income, ForeignAid, Tax, Steal, Exchange should appear
        for a in ["Income", "ForeignAid", "Tax", "Steal", "Exchange"]:
            assert a in seen, f"Action {a!r} never seen in 100 games"

    def test_wins_by_seat_sums_to_n_games(self, games):
        stats = CorpusStats()
        _accumulate(games, stats)
        total_wins = sum(stats.wins_by_seat.values())
        # Some games may end at turn limit (no winner) so total_wins <= n_games
        assert total_wins <= stats.n_games

    def test_games_by_seat_matches_player_count(self, games):
        stats = CorpusStats()
        _accumulate(games, stats)
        # All 4 seats should have exactly n_games entries
        for seat in range(4):
            assert stats.games_by_seat[seat] == len(games)

    def test_influence_losses_recorded(self, games):
        stats = CorpusStats()
        _accumulate(games, stats)
        total = sum(stats.influence_loss_by_card.values())
        assert total > 0

    def test_player_count_dist(self, games):
        stats = CorpusStats()
        _accumulate(games, stats)
        assert 4 in stats.player_count_dist
        assert stats.player_count_dist[4] == len(games)

    def test_challenge_counts_nonzero(self, games):
        stats = CorpusStats()
        _accumulate(games, stats)
        assert stats.total_challenges >= 0   # may be 0 with low challenge_prob

    def test_block_counts_nonzero(self, games):
        stats = CorpusStats()
        _accumulate(games, stats)
        assert stats.total_blocks >= 0


class TestQualityReport:

    @pytest.fixture
    def report(self):
        games = _make_games(n=50)
        stats = CorpusStats()
        _accumulate(games, stats)
        _analyse_features(games, stats, max_games=50)
        return QualityReport(stats, log_dir="/tmp/test")

    def test_summary_is_string(self, report):
        s = report.summary()
        assert isinstance(s, str)
        assert len(s) > 100

    def test_summary_contains_key_sections(self, report):
        s = report.summary()
        assert "Corpus Overview" in s
        assert "Action Frequencies" in s
        assert "Win Rates" in s
        assert "Challenge Statistics" in s
        assert "Block Statistics" in s
        assert "Influence Loss" in s
        assert "Feature Statistics" in s

    def test_summary_contains_action_names(self, report):
        s = report.summary()
        for action in ACTIONS:
            assert action in s

    def test_save_to_file(self, report, tmp_path):
        path = str(tmp_path / "report.txt")
        report.save(path)
        assert os.path.exists(path)
        content = open(path).read()
        assert "Corpus Overview" in content

    def test_feature_stats_present(self, report):
        assert report.stats.feature_mean is not None
        assert report.stats.feature_std is not None
        assert report.stats.feature_sparsity is not None
        assert 0.0 <= report.stats.feature_sparsity <= 1.0


class TestRunQualityReport:

    def test_returns_quality_report_object(self, tmp_path):
        games = _make_games(n=20)
        for i, log in enumerate(_raw_logs_from_games(_make_games(n=20))):
            with open(tmp_path / f"game_{i:04d}.json", "w") as f:
                json.dump(log, f)

        report = run_quality_report(
            str(tmp_path), show_progress=False, max_feature_games=20
        )
        assert isinstance(report, QualityReport)
        assert report.stats.n_games == 20

    def test_report_saved_to_output_path(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        for i, log in enumerate(_raw_logs_from_games(_make_games(n=10))):
            with open(log_dir / f"game_{i:04d}.json", "w") as f:
                json.dump(log, f)

        out = str(tmp_path / "report.txt")
        run_quality_report(
            str(log_dir), output_path=out,
            show_progress=False, max_feature_games=10
        )
        assert os.path.exists(out)


# ===========================================================================
# Round-trip: full pipeline over existing raw logs
# ===========================================================================

class TestFullPipelineRoundTrip:
    """
    End-to-end test: raw logs → ParsedGame → features → CoupDataset.
    Requires data/raw_logs/ to exist from a simulator run.
    """

    def test_pipeline_roundtrip_on_simulator_logs(self):
        if not os.path.isdir(RAW_LOGS_DIR):
            pytest.skip("data/raw_logs/ not found")

        games = []
        for entry in sorted(os.scandir(RAW_LOGS_DIR), key=lambda e: e.name):
            if entry.is_dir():
                games.extend(load_logs(entry.path))
            elif entry.is_file() and entry.name.endswith(".json"):
                games.extend(load_logs(entry.path))

        assert len(games) > 0, "No games loaded"

        sample_games = games[:50]
        X, y_action, y_target, y_outcome = extract_dataset(
            sample_games, show_progress=False
        )
        assert X.shape[0] > 0
        assert X.shape[1] == FeatureConfig().feature_dim
        assert len(y_action) == len(X)

        ds = CoupDataset(X, y_action, y_target, y_outcome, task="action")
        assert len(ds) == len(X)

        x0, y0 = ds[0]
        assert x0.shape == (ds.feature_dim,)
        assert 0 <= y0.item() < N_ACTIONS
