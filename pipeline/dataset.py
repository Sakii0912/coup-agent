"""
pipeline/dataset.py — PyTorch Dataset for Coup training data.

Two Dataset classes:

  CoupDataset          — loads a pre-extracted numpy array pair (X, y_*)
                         from disk or from in-memory arrays. Fast, used for
                         training loops after feature extraction is done.

  CoupStreamingDataset — streams directly from ParsedGame objects, extracting
                         features on-the-fly. Slower per epoch but uses less
                         RAM; useful during development or when the full
                         feature matrix doesn't fit in memory.

Three task heads are supported (controlled by `task` argument):
  "action"   — predict which action the actor chose (7-class)
  "target"   — predict which opponent was targeted (6-class, incl. no-target)
  "outcome"  — predict whether the action resolved successfully (binary)

Usage:
    from pipeline.dataset import CoupDataset, CoupStreamingDataset
    from pipeline.feature_extractor import extract_dataset, FeatureConfig
    from pipeline.log_parser import load_logs
    from torch.utils.data import DataLoader

    # Pre-extracted (recommended for training)
    X, y_action, y_target, y_outcome = extract_dataset(games)
    ds = CoupDataset(X, y_action, task="action")
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    # Streaming (development / low-memory)
    games = load_logs("data/raw_logs/random_4p_med/")
    ds = CoupStreamingDataset(games, task="action")
    loader = DataLoader(ds, batch_size=64, shuffle=False)
"""

from __future__ import annotations

import os
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_extractor import (
    FeatureConfig,
    extract_features,
    extract_dataset,
    ACTIONS,
    N_ACTIONS,
    feature_dim,
)
from .log_parser import ParsedGame

# Task type alias
Task = Literal["action", "target", "outcome"]


# ---------------------------------------------------------------------------
# CoupDataset — pre-extracted arrays
# ---------------------------------------------------------------------------

class CoupDataset(Dataset):
    """
    Dataset backed by pre-extracted numpy arrays.

    Args:
        X:        Feature matrix, shape (N, feature_dim), float32.
        y_action: Action labels, shape (N,), int.
        y_target: Target labels, shape (N,), int.
        y_outcome: Outcome labels, shape (N,), int.
        task:     Which label to return as y. One of "action", "target", "outcome".
        normalise: If True, z-score normalise X along axis 0.
                   Pass pre-computed (mean, std) to avoid fitting on the test set.
        stats:    Optional (mean, std) tuple for normalisation. If None and
                  normalise=True, computed from X.
    """

    def __init__(
        self,
        X: np.ndarray,
        y_action: np.ndarray,
        y_target: np.ndarray,
        y_outcome: np.ndarray,
        task: Task = "action",
        normalise: bool = True,
        stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        assert task in ("action", "target", "outcome"), \
            f"task must be 'action', 'target', or 'outcome', got {task!r}"
        assert len(X) == len(y_action) == len(y_target) == len(y_outcome), \
            "All arrays must have the same length"

        self.task = task

        # Normalise features
        if normalise:
            if stats is not None:
                self._mean, self._std = stats
            else:
                self._mean = X.mean(axis=0)
                self._std  = X.std(axis=0) + 1e-8   # avoid div-by-zero
            X = (X - self._mean) / self._std
        else:
            self._mean = np.zeros(X.shape[1], dtype=np.float32)
            self._std  = np.ones(X.shape[1],  dtype=np.float32)

        self.X        = torch.from_numpy(X.astype(np.float32))
        self.y_action = torch.from_numpy(y_action.astype(np.int64))
        self.y_target = torch.from_numpy(y_target.astype(np.int64))
        self.y_outcome = torch.from_numpy(y_outcome.astype(np.int64))

    # ── Torch Dataset interface ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.task == "action":
            return x, self.y_action[idx]
        elif self.task == "target":
            return x, self.y_target[idx]
        else:
            return x, self.y_outcome[idx]

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        if self.task == "action":
            return N_ACTIONS                      # 7
        elif self.task == "target":
            return 6                              # 5 opponent slots + no-target
        else:
            return 2                              # binary outcome

    @property
    def normalisation_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) for reuse on a held-out test split."""
        return self._mean, self._std

    # ── Serialisation ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the dataset to a single .npz file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X.numpy(),
            y_action=self.y_action.numpy(),
            y_target=self.y_target.numpy(),
            y_outcome=self.y_outcome.numpy(),
            mean=self._mean,
            std=self._std,
        )

    @classmethod
    def load(cls, path: str, task: Task = "action") -> "CoupDataset":
        """Load a dataset from a .npz file saved by `save()`."""
        data = np.load(path)
        ds = cls(
            X=data["X"],
            y_action=data["y_action"],
            y_target=data["y_target"],
            y_outcome=data["y_outcome"],
            task=task,
            normalise=True,
            stats=(data["mean"], data["std"]),
        )
        return ds

    # ── Train / val split ───────────────────────────────────────────────

    def split(
        self, val_fraction: float = 0.1, seed: int = 42
    ) -> Tuple["CoupDataset", "CoupDataset"]:
        """
        Split into train and validation sets. Normalisation stats are
        computed on the training split and applied to both.

        Returns:
            (train_dataset, val_dataset)
        """
        rng  = np.random.RandomState(seed)
        N    = len(self)
        perm = rng.permutation(N)
        cut  = int(N * (1 - val_fraction))
        tr_idx, vl_idx = perm[:cut], perm[cut:]

        X_np  = self.X.numpy()
        ya_np = self.y_action.numpy()
        yt_np = self.y_target.numpy()
        yo_np = self.y_outcome.numpy()

        # Re-fit stats on training split only
        X_tr = X_np[tr_idx]
        tr_mean = X_tr.mean(axis=0)
        tr_std  = X_tr.std(axis=0) + 1e-8

        train_ds = CoupDataset(
            X=X_np[tr_idx], y_action=ya_np[tr_idx],
            y_target=yt_np[tr_idx], y_outcome=yo_np[tr_idx],
            task=self.task, normalise=True, stats=(tr_mean, tr_std),
        )
        val_ds = CoupDataset(
            X=X_np[vl_idx], y_action=ya_np[vl_idx],
            y_target=yt_np[vl_idx], y_outcome=yo_np[vl_idx],
            task=self.task, normalise=True, stats=(tr_mean, tr_std),
        )
        return train_ds, val_ds

    def __repr__(self) -> str:
        return (
            f"CoupDataset(n={len(self)}, feature_dim={self.feature_dim}, "
            f"task={self.task!r}, n_classes={self.n_classes})"
        )


# ---------------------------------------------------------------------------
# CoupStreamingDataset — on-the-fly extraction
# ---------------------------------------------------------------------------

class CoupStreamingDataset(Dataset):
    """
    Dataset that extracts features on-the-fly from ParsedGame objects.

    Slower than CoupDataset (no pre-computation) but uses very little
    memory — only one game is held in working memory at a time (via indexing).
    Use this when iterating through data once during development.

    Note: __getitem__ by sample index requires a pre-built index mapping
    sample → (game_idx, within-game sample offset). This index is built
    on first access and cached.
    """

    def __init__(
        self,
        games: List[ParsedGame],
        task: Task = "action",
        config: Optional[FeatureConfig] = None,
    ) -> None:
        assert task in ("action", "target", "outcome")
        self.games  = games
        self.task   = task
        self.config = config or FeatureConfig()

        # Pre-extract all samples into memory (simpler and fast enough for <50k games)
        self._samples = []
        for game in games:
            self._samples.extend(extract_features(game, self.config))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self._samples[idx]
        x = torch.from_numpy(s.features)
        if self.task == "action":
            y = torch.tensor(s.action_label, dtype=torch.long)
        elif self.task == "target":
            y = torch.tensor(s.target_label, dtype=torch.long)
        else:
            y = torch.tensor(s.outcome, dtype=torch.long)
        return x, y

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    @property
    def n_classes(self) -> int:
        if self.task == "action":
            return N_ACTIONS
        elif self.task == "target":
            return 6
        else:
            return 2

    def __repr__(self) -> str:
        return (
            f"CoupStreamingDataset(games={len(self.games)}, "
            f"samples={len(self)}, task={self.task!r})"
        )


# ---------------------------------------------------------------------------
# Factory: build a full dataset from a log directory
# ---------------------------------------------------------------------------

def build_dataset(
    log_dir: str,
    task: Task = "action",
    config: Optional[FeatureConfig] = None,
    save_path: Optional[str] = None,
    show_progress: bool = True,
) -> CoupDataset:
    """
    End-to-end: load logs → extract features → return a CoupDataset.

    Args:
        log_dir:       Directory of .json game logs (searched recursively).
        task:          Which label to use as y.
        config:        Feature config. Defaults to FeatureConfig(max_players=6).
        save_path:     If set, save the resulting dataset to this .npz path.
        show_progress: Print progress during loading and extraction.

    Returns:
        A CoupDataset ready for use with DataLoader.
    """
    from .log_parser import load_logs as _load_logs

    if show_progress:
        print(f"Loading logs from {log_dir!r}...")

    # Collect logs recursively from all subdirectories
    games: List[ParsedGame] = []
    if os.path.isdir(log_dir):
        for entry in sorted(os.scandir(log_dir), key=lambda e: e.name):
            if entry.is_dir():
                sub_games = _load_logs(entry.path)
                games.extend(sub_games)
                if show_progress:
                    print(f"  {entry.name:<30} {len(sub_games):>6} games")
            elif entry.name.endswith(".json"):
                pass   # top-level files handled by load_logs below

        top_games = _load_logs(log_dir)
        games.extend(top_games)
    else:
        games = _load_logs(log_dir, show_progress=False)

    if show_progress:
        print(f"  Total: {len(games):,} games loaded\n")

    if show_progress:
        print("Extracting features...")
    X, y_action, y_target, y_outcome = extract_dataset(
        games, config=config, show_progress=show_progress
    )

    ds = CoupDataset(
        X=X, y_action=y_action, y_target=y_target, y_outcome=y_outcome,
        task=task, normalise=True,
    )

    if save_path:
        ds.save(save_path)
        if show_progress:
            print(f"Dataset saved to {save_path!r}")

    return ds
