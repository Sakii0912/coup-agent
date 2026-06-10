"""
agents/neural/trainer.py — Supervised pretraining for CoupPolicyNet (sub-goal 4.2).

─────────────────────────────────────────────────────────────────────────────
What is supervised pretraining?
─────────────────────────────────────────────────────────────────────────────

Before self-play RL (Phase 4.3), we pre-train the model via supervised
learning on the Phase 2 dataset. The model learns to imitate the behaviour
recorded in the game logs (random + honest agents). This is much faster than
learning from scratch via RL and provides a useful warm start.

The dataset provides three label types:
  y_action   (7-class)  — which action type the actor chose
  y_target   (6-class)  — which opponent slot they targeted (5 = no target)
  y_outcome  (binary)   — did the action resolve successfully (optional)

We train on action + target jointly with a combined loss:
  L = w_action × CrossEntropy(action_logits, y_action)
    + w_target × CrossEntropy(target_logits, y_target)

─────────────────────────────────────────────────────────────────────────────
Training setup
─────────────────────────────────────────────────────────────────────────────

  Optimiser:    AdamW (weight decay = 1e-4)
  Scheduler:    CosineAnnealingLR (restarts every n_epochs // 3)
  Batch size:   1024 (default)
  Early stop:   patience = 5 epochs on val action accuracy
  Checkpoint:   saves best model by val action accuracy

─────────────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────────────

    from agents.neural.trainer import SupervisedTrainer
    from pipeline.dataset import CoupDataset

    ds = CoupDataset.load("data/dataset_action.npz", task="action")
    trainer = SupervisedTrainer(
        dataset_path="data/dataset_action.npz",
        save_dir="data/models/",
    )
    trainer.train(n_epochs=30)

Or from the CLI:
    python -m agents.neural.trainer --dataset data/dataset_action.npz --epochs 30
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import CoupPolicyNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    """Pick the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_tensor_dataset(
    npz_path: str,
) -> Tuple[TensorDataset, TensorDataset, np.ndarray, np.ndarray]:
    """
    Load an .npz file, apply train/val split, return two TensorDatasets.
    Also returns the (mean, std) normalisation arrays used.
    """
    data = np.load(npz_path)
    X        = data["X"].astype(np.float32)
    y_action = data["y_action"].astype(np.int64)
    y_target = data["y_target"].astype(np.int64)

    # Use stored normalisation stats if available (from CoupDataset.save())
    mean = data["mean"].astype(np.float32) if "mean" in data else X.mean(0)
    std  = data["std"].astype(np.float32)  if "std"  in data else (X.std(0) + 1e-8)

    X_norm = (X - mean) / std

    # 90 / 10 split (reproducible)
    rng   = np.random.RandomState(42)
    perm  = rng.permutation(len(X_norm))
    cut   = int(len(X_norm) * 0.9)
    tr_i, vl_i = perm[:cut], perm[cut:]

    def _ds(idx):
        return TensorDataset(
            torch.from_numpy(X_norm[idx]),
            torch.from_numpy(y_action[idx]),
            torch.from_numpy(y_target[idx]),
        )

    return _ds(tr_i), _ds(vl_i), mean, std


# ---------------------------------------------------------------------------
# SupervisedTrainer
# ---------------------------------------------------------------------------

class SupervisedTrainer:
    """
    Trains CoupPolicyNet via supervised learning on a pre-extracted dataset.

    Args:
        dataset_path:  Path to .npz file saved by CoupDataset.save().
        save_dir:      Directory to save model checkpoints.
        input_dim:     Feature vector size. If None, inferred from dataset.
        hidden_dim:    Trunk hidden width. Default 256.
        batch_size:    Training batch size. Default 1024.
        lr:            Initial learning rate. Default 3e-4.
        weight_decay:  AdamW weight decay. Default 1e-4.
        action_weight: Loss weight for action prediction. Default 1.0.
        target_weight: Loss weight for target prediction. Default 0.5.
        patience:      Early-stopping patience in epochs. Default 7.
        device:        Torch device. If None, auto-detected.
    """

    def __init__(
        self,
        dataset_path:   str,
        save_dir:       str             = "data/models",
        input_dim:      Optional[int]   = None,
        hidden_dim:     int             = 256,
        batch_size:     int             = 1024,
        lr:             float           = 3e-4,
        weight_decay:   float           = 1e-4,
        action_weight:  float           = 1.0,
        target_weight:  float           = 0.5,
        patience:       int             = 7,
        device:         Optional[str]   = None,
    ) -> None:
        self.dataset_path  = dataset_path
        self.save_dir      = save_dir
        self.hidden_dim    = hidden_dim
        self.batch_size    = batch_size
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.action_weight = action_weight
        self.target_weight = target_weight
        self.patience      = patience
        self.device        = torch.device(device) if device else _get_device()

        os.makedirs(save_dir, exist_ok=True)

        # Load dataset and build data loaders
        print(f"Loading dataset from {dataset_path!r} ...")
        tr_ds, vl_ds, self.feat_mean, self.feat_std = _build_tensor_dataset(dataset_path)
        print(f"  Train: {len(tr_ds):,}  Val: {len(vl_ds):,} samples")

        if input_dim is None:
            # Infer from first sample
            input_dim = tr_ds[0][0].shape[0]
        self.input_dim = input_dim

        self.train_loader = DataLoader(
            tr_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            vl_ds, batch_size=batch_size * 4, shuffle=False,
            num_workers=0,
        )

        # Build model
        self.model = CoupPolicyNet(
            input_dim=input_dim, hidden_dim=hidden_dim
        ).to(self.device)
        print(f"  {self.model}")
        print(f"  Device: {self.device}")

    # ------------------------------------------------------------------ #
    #  Training loop                                                      #
    # ------------------------------------------------------------------ #

    def train(self, n_epochs: int = 30) -> Dict[str, Any]:
        """
        Run the supervised training loop.

        Returns:
            Training history dict with per-epoch metrics.
        """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, n_epochs // 3), T_mult=1,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc    = float("-inf")
        best_epoch      = 0
        epochs_no_improve = 0
        history: Dict[str, list] = {
            "train_loss": [], "train_action_acc": [], "train_target_acc": [],
            "val_loss":   [], "val_action_acc":   [], "val_target_acc":   [],
        }

        print(f"\nTraining for up to {n_epochs} epochs (patience={self.patience}) ...")
        print(f"{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc%':>7}  "
              f"{'VlLoss':>8}  {'VlAcc%':>7}  {'LR':>8}")
        print("-" * 60)

        t0 = time.time()

        for epoch in range(1, n_epochs + 1):
            # ── Train ─────────────────────────────────────────────────────
            tr_metrics = self._run_epoch(optimizer, criterion, train=True)

            # ── Validate ──────────────────────────────────────────────────
            vl_metrics = self._run_epoch(None, criterion, train=False)

            scheduler.step()

            # Record
            for k, v in tr_metrics.items():
                history[f"train_{k}"].append(v)
            for k, v in vl_metrics.items():
                history[f"val_{k}"].append(v)

            val_acc = vl_metrics["action_acc"]
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"{epoch:>6}  "
                f"{tr_metrics['loss']:>8.4f}  {100*tr_metrics['action_acc']:>6.2f}%  "
                f"{vl_metrics['loss']:>8.4f}  {100*val_acc:>6.2f}%  "
                f"{current_lr:>8.2e}"
            )

            # ── Save best ─────────────────────────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch   = epoch
                self._save_checkpoint(epoch, val_acc, optimizer)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print(f"\nEarly stop at epoch {epoch} "
                      f"(best val acc {100*best_val_acc:.2f}% at epoch {best_epoch})")
                break

        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s. "
              f"Best val action accuracy: {100*best_val_acc:.2f}% (epoch {best_epoch})")

        history["best_val_acc"]   = best_val_acc
        history["best_epoch"]     = best_epoch
        history["elapsed_s"]      = elapsed
        return history

    def _run_epoch(
        self,
        optimizer: Optional[optim.Optimizer],
        criterion: nn.Module,
        train:     bool,
    ) -> Dict[str, float]:
        """Run one epoch (train or validate). Returns loss + accuracy metrics."""
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader

        total_loss     = 0.0
        correct_action = 0
        correct_target = 0
        n_samples      = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for X_batch, y_action, y_target in loader:
                X_batch  = X_batch.to(self.device)
                y_action = y_action.to(self.device)
                y_target = y_target.to(self.device)

                out   = self.model(X_batch)
                l_act = criterion(out["action"], y_action)
                l_tgt = criterion(out["target"], y_target)
                loss  = self.action_weight * l_act + self.target_weight * l_tgt

                if train and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                bs              = X_batch.size(0)
                total_loss     += loss.item() * bs
                correct_action += (out["action"].argmax(1) == y_action).sum().item()
                correct_target += (out["target"].argmax(1) == y_target).sum().item()
                n_samples      += bs

        return {
            "loss":       total_loss     / max(n_samples, 1),
            "action_acc": correct_action / max(n_samples, 1),
            "target_acc": correct_target / max(n_samples, 1),
        }

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                      #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(
        self, epoch: int, val_acc: float, optimizer: optim.Optimizer
    ) -> None:
        """Save the current best model to disk."""
        path = os.path.join(self.save_dir, "best_model.pt")
        torch.save({
            "epoch":      epoch,
            "val_acc":    val_acc,
            "state_dict": self.model.state_dict(),
            "input_dim":  self.input_dim,
            "hidden_dim": self.hidden_dim,
            "feat_mean":  torch.as_tensor(self.feat_mean),
            "feat_std":   torch.as_tensor(self.feat_std),
        }, path)

    def load_best(self) -> CoupPolicyNet:
        """Load and return the best saved model."""
        path = os.path.join(self.save_dir, "best_model.pt")
        ckpt = torch.load(path, map_location=self.device)
        net  = CoupPolicyNet(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
        ).to(self.device)
        net.load_state_dict(ckpt["state_dict"])

        feat_mean = ckpt.get("feat_mean")
        feat_std = ckpt.get("feat_std")
        self.feat_mean = feat_mean.detach().cpu().numpy() if isinstance(feat_mean, torch.Tensor) else feat_mean
        self.feat_std = feat_std.detach().cpu().numpy() if isinstance(feat_std, torch.Tensor) else feat_std
        return net


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Supervised pretraining for CoupPolicyNet")
    parser.add_argument("--dataset",    type=str,   default="data/dataset_action.npz")
    parser.add_argument("--save_dir",   type=str,   default="data/models")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=1024)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--patience",   type=int,   default=7)
    args = parser.parse_args()

    trainer = SupervisedTrainer(
        dataset_path=args.dataset,
        save_dir=args.save_dir,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
    trainer.train(n_epochs=args.epochs)
