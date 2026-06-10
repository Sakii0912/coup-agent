"""
training — Phase 4.3: self-play training loops.

  cfr_trainer.py  CFRTrainer      — wraps CFRSolver, checkpointing, convergence log
  rl_trainer.py   RLTrainer       — REINFORCE self-play with BeliefTracker integration
  train.py                        — unified CLI entry point

Quick-start:

    # 1. Supervised pretraining (requires torch + dataset_action.npz)
    python -m training.train pretrain --epochs 30

    # 2. CFR self-play (no torch required)
    python -m training.train cfr --n_iter 10000

    # 3. RL fine-tuning (requires torch + pretrained model)
    python -m training.train rl --checkpoint data/models/best_model.pt --rounds 100
"""

from .cfr_trainer import CFRTrainer, CFRConfig

__all__ = ["CFRTrainer", "CFRConfig"]
