"""
training/train.py — Unified CLI for Phase 4 training (sub-goal 4.3).

Modes:
  cfr       Run CFR self-play training (no torch required)
  pretrain  Supervised pretraining of NeuralAgent on Phase 2 dataset
  rl        REINFORCE self-play fine-tuning of NeuralAgent

Examples:
  # CFR — 10k iterations, 4 players
  python -m training.train cfr --n_iter 10000 --n_players 4

  # Supervised pretraining
  python -m training.train pretrain --dataset data/dataset_action.npz --epochs 30

  # RL fine-tuning from a pretrained checkpoint
  python -m training.train rl --checkpoint data/models/best_model.pt --rounds 100

  # Resume CFR from latest checkpoint
  python -m training.train cfr --resume data/models/cfr/latest.pkl.gz --n_iter 5000
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_cfr(args: argparse.Namespace) -> None:
    from training.cfr_trainer import CFRTrainer, CFRConfig

    config = CFRConfig(
        n_players    = args.n_players,
        use_cfr_plus = not args.vanilla_cfr,
        seed         = args.seed,
        save_every   = args.save_every,
        log_every    = args.log_every,
    )

    if args.resume:
        trainer = CFRTrainer.from_checkpoint(
            checkpoint_path = args.resume,
            output_dir      = args.output_dir,
            config          = config,
        )
    else:
        trainer = CFRTrainer(
            n_players  = args.n_players,
            output_dir = args.output_dir,
            config     = config,
        )

    stats = trainer.train(n_iterations=args.n_iter)
    print(f"\nFinal stats: {stats}")


def cmd_pretrain(args: argparse.Namespace) -> None:
    from agents.neural.trainer import SupervisedTrainer

    trainer = SupervisedTrainer(
        dataset_path = args.dataset,
        save_dir     = args.output_dir,
        hidden_dim   = args.hidden_dim,
        batch_size   = args.batch_size,
        lr           = args.lr,
        patience     = args.patience,
    )
    history = trainer.train(n_epochs=args.epochs)
    print(f"\nBest val accuracy: {100*history['best_val_acc']:.2f}% "
          f"(epoch {history['best_epoch']})")
    print(f"Model saved to:    {args.output_dir}/best_model.pt")


def cmd_rl(args: argparse.Namespace) -> None:
    import torch
    from agents.neural.model import CoupPolicyNet
    from training.rl_trainer import RLTrainer
    import numpy as np

    # Load pretrained model
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint!r}")
        print("Run 'python -m training.train pretrain' first.")
        sys.exit(1)

    ckpt       = torch.load(args.checkpoint, map_location="cpu")
    model      = CoupPolicyNet(input_dim=ckpt["input_dim"],
                               hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["state_dict"])
    feat_mean  = ckpt.get("feat_mean")
    feat_std   = ckpt.get("feat_std")

    print(f"Loaded model from {args.checkpoint!r}")
    print(f"  input_dim={ckpt['input_dim']}, hidden_dim={ckpt['hidden_dim']}")

    trainer = RLTrainer(
        model           = model,
        feat_mean       = feat_mean,
        feat_std        = feat_std,
        n_players       = args.n_players,
        output_dir      = args.output_dir,
        lr              = args.lr,
        games_per_round = args.games_per_round,
        use_belief      = not args.no_belief,
    )
    stats = trainer.train(n_rounds=args.rounds)
    print(f"\nFinal stats: {stats}")
    print(f"RL model saved to: {args.output_dir}/rl_latest.pt")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="training.train",
        description="Coup AI — Phase 4 training CLI",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── cfr ─────────────────────────────────────────────────────────────
    p_cfr = sub.add_parser("cfr", help="CFR self-play training")
    p_cfr.add_argument("--n_iter",      type=int,   default=10_000)
    p_cfr.add_argument("--n_players",   type=int,   default=4)
    p_cfr.add_argument("--output_dir",  type=str,   default="data/models/cfr")
    p_cfr.add_argument("--resume",      type=str,   default=None,
                       help="Path to checkpoint to resume from")
    p_cfr.add_argument("--seed",        type=int,   default=0)
    p_cfr.add_argument("--save_every",  type=int,   default=1_000)
    p_cfr.add_argument("--log_every",   type=int,   default=100)
    p_cfr.add_argument("--vanilla_cfr", action="store_true",
                       help="Use vanilla CFR instead of CFR+")

    # ── pretrain ────────────────────────────────────────────────────────
    p_pre = sub.add_parser("pretrain", help="Supervised pretraining on Phase 2 dataset")
    p_pre.add_argument("--dataset",    type=str,   default="data/dataset_action.npz")
    p_pre.add_argument("--output_dir", type=str,   default="data/models")
    p_pre.add_argument("--epochs",     type=int,   default=30)
    p_pre.add_argument("--batch_size", type=int,   default=1024)
    p_pre.add_argument("--lr",         type=float, default=3e-4)
    p_pre.add_argument("--hidden_dim", type=int,   default=256)
    p_pre.add_argument("--patience",   type=int,   default=7)

    # ── rl ──────────────────────────────────────────────────────────────
    p_rl = sub.add_parser("rl", help="REINFORCE self-play fine-tuning")
    p_rl.add_argument("--checkpoint",    type=str,   default="data/models/best_model.pt",
                      help="Pretrained model checkpoint (.pt file)")
    p_rl.add_argument("--output_dir",    type=str,   default="data/models/rl")
    p_rl.add_argument("--rounds",        type=int,   default=100)
    p_rl.add_argument("--games_per_round", type=int, default=64)
    p_rl.add_argument("--n_players",     type=int,   default=4)
    p_rl.add_argument("--lr",            type=float, default=1e-4)
    p_rl.add_argument("--no_belief",     action="store_true",
                      help="Disable BeliefTracker (use 104-dim features)")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if args.mode == "cfr":
        cmd_cfr(args)
    elif args.mode == "pretrain":
        cmd_pretrain(args)
    elif args.mode == "rl":
        cmd_rl(args)
