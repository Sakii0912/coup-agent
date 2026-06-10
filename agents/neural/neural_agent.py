"""
agents/neural/neural_agent.py — NeuralAgent implementing the Agent ABC (sub-goal 4.2).

Wraps a trained CoupPolicyNet behind the six Agent decision points:

  choose_action            → action_head  + target_head (combined score)
  choose_to_challenge_action → binary_head
  choose_to_block          → binary_head (gated by legal_blocks availability)
  choose_to_challenge_block → binary_head
  choose_card_to_lose      → heuristic (keep highest-value card)
  choose_exchange_cards    → heuristic (keep highest-value combination)

─────────────────────────────────────────────────────────────────────────────
Action selection
─────────────────────────────────────────────────────────────────────────────

The model outputs action-type logits (7 dims) and target-slot logits (6 dims)
independently. To select a legal action, we score every legal action:

    score(action) = action_prob[action.type] × target_prob[slot(action)]

where slot(action) = relative position of the target among alive opponents,
and target_prob[5] = "no target" probability for untargeted actions.

Legal actions are masked: actions with 0 probability are excluded before
sampling. We sample from the normalised scores rather than taking argmax,
which gives a mixed strategy that's harder to exploit.

─────────────────────────────────────────────────────────────────────────────
Feature construction
─────────────────────────────────────────────────────────────────────────────

Uses `features_from_observation()` from the pipeline. The agent maintains
a rolling action_history across decisions within the same game, used for
the history encoding in the feature vector.

Game boundary detection: when obs.turn_number < self._last_turn, we know
a new game has started and reset the action history.

─────────────────────────────────────────────────────────────────────────────
Fallback behaviour
─────────────────────────────────────────────────────────────────────────────

If the model is None or inference fails, every method falls back to the
behaviour of RandomAgent. This allows the NeuralAgent to be used as a
drop-in even before a model is trained (it just plays randomly until
a checkpoint is loaded).
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch

from coup.cards import Card, ActionType
from coup.state import Observation
from coup.actions import Action, Block, get_legal_actions, get_legal_blocks
from coup.agents.base import Agent
from pipeline.feature_extractor import (
    FeatureConfig,
    features_from_observation,
    ACTIONS,
    ACTION_TO_IDX,
    N_ACTIONS,
)
from .model import CoupPolicyNet


# ---------------------------------------------------------------------------
# Card value heuristic
# ---------------------------------------------------------------------------

# Rough strategic value of each card for lose-influence and exchange decisions.
# Duke and Captain are high value (coins + blocking power).
# Assassin is medium (offensive, costs 3 coins to use).
# Contessa and Ambassador are situational.
CARD_VALUE: dict = {
    Card.DUKE:       5,
    Card.CAPTAIN:    4,
    Card.ASSASSIN:   3,
    Card.AMBASSADOR: 2,
    Card.CONTESSA:   2,
}


# ---------------------------------------------------------------------------
# NeuralAgent
# ---------------------------------------------------------------------------

class NeuralAgent(Agent):
    """
    Coup agent driven by a trained CoupPolicyNet.

    Args:
        model:       Trained CoupPolicyNet. If None, falls back to random.
        feat_mean:   Feature normalisation mean (from trainer). Shape (input_dim,).
        feat_std:    Feature normalisation std  (from trainer). Shape (input_dim,).
        config:      FeatureConfig controlling vector size.
        temperature: Softmax temperature for sampling. 1.0 = as trained.
                     <1 = more deterministic, >1 = more random.
        seed:        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        model:       Optional[CoupPolicyNet] = None,
        feat_mean:   Optional[np.ndarray]    = None,
        feat_std:    Optional[np.ndarray]    = None,
        config:      Optional[FeatureConfig] = None,
        temperature: float                   = 1.0,
        seed:        Optional[int]           = None,
    ) -> None:
        self.model       = model
        self.feat_mean   = feat_mean
        self.feat_std    = feat_std
        self.config      = config or FeatureConfig()
        self.temperature = temperature
        self._rng        = random.Random(seed)

        # Per-game state (reset on new game detection)
        self._action_history: List[str] = []
        self._last_turn: int = -1

        if self.model is not None:
            self.model.eval()

    # ------------------------------------------------------------------ #
    #  Factories                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        temperature:     float           = 1.0,
        config:          Optional[FeatureConfig] = None,
        device:          str             = "cpu",
    ) -> "NeuralAgent":
        """
        Load a NeuralAgent from a trainer checkpoint (.pt file).

        Args:
            checkpoint_path: Path to the .pt file saved by SupervisedTrainer.
            temperature:     Sampling temperature.
            config:          FeatureConfig. If None, inferred from checkpoint.
            device:          Torch device for inference.
        """
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = CoupPolicyNet(
            input_dim  = ckpt["input_dim"],
            hidden_dim = ckpt["hidden_dim"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        feat_mean = ckpt.get("feat_mean")
        feat_std  = ckpt.get("feat_std")

        if isinstance(feat_mean, torch.Tensor):
            feat_mean = feat_mean.detach().cpu().numpy()
        if isinstance(feat_std, torch.Tensor):
            feat_std = feat_std.detach().cpu().numpy()

        if config is None:
            config = FeatureConfig(max_players=6)

        return cls(
            model=model,
            feat_mean=feat_mean,
            feat_std=feat_std,
            config=config,
            temperature=temperature,
        )

    # ------------------------------------------------------------------ #
    #  Agent interface                                                    #
    # ------------------------------------------------------------------ #

    def choose_action(
        self, obs: Observation, legal_actions: List[Action]
    ) -> Action:
        self._maybe_reset(obs)

        if self.model is None or not legal_actions:
            return self._rng.choice(legal_actions)

        feat = self._featurise(obs, "Income", None, None)

        try:
            x = torch.from_numpy(feat).unsqueeze(0)          # (1, input_dim)
            out = self.model(x)
            act_probs = torch.softmax(
                out["action"] / max(self.temperature, 1e-6), dim=-1
            ).squeeze(0).numpy()                              # (7,)
            tgt_probs = torch.softmax(
                out["target"] / max(self.temperature, 1e-6), dim=-1
            ).squeeze(0).numpy()                              # (6,)
        except Exception:
            return self._rng.choice(legal_actions)

        # Score each legal action by combining action-type and target probabilities
        scores = []
        for action in legal_actions:
            a_idx = ACTION_TO_IDX.get(action.type.value, 0)
            a_p   = float(act_probs[a_idx])

            if action.target_idx is not None:
                slot  = self._target_slot(obs, action.target_idx)
                t_p   = float(tgt_probs[min(slot, 5)])
            else:
                t_p   = float(tgt_probs[5])   # "no target" slot

            scores.append(max(a_p * t_p, 1e-9))

        # Sample from normalised scores
        total  = sum(scores)
        probs  = [s / total for s in scores]
        chosen = self._rng.choices(legal_actions, weights=probs, k=1)[0]

        # Record action type in history
        self._action_history.append(chosen.type.value)
        return chosen

    def choose_to_challenge_action(
        self, obs: Observation, action: Action
    ) -> bool:
        if action.actor_idx == obs.my_idx:
            return False
        if self.model is None:
            return self._rng.random() < 0.2

        claimed = action.claimed_card.value if action.claimed_card else None
        return self._binary_yes(obs, action.type.value, None, claimed)

    def choose_to_block(
        self,
        obs:          Observation,
        action:       Action,
        legal_blocks: List[Block],
    ) -> Optional[Block]:
        if not legal_blocks:
            return None
        if self.model is None:
            return self._rng.choice(legal_blocks) if self._rng.random() < 0.3 else None

        should_block = self._binary_yes(
            obs, action.type.value, None, None
        )
        if should_block:
            # Among legal blocks, prefer ones we actually hold
            honest_blocks = [b for b in legal_blocks if b.blocking_card in obs.my_hand]
            if honest_blocks:
                return self._rng.choice(honest_blocks)
            return self._rng.choice(legal_blocks)
        return None

    def choose_to_challenge_block(
        self, obs: Observation, block: Block
    ) -> bool:
        if block.blocker_idx == obs.my_idx:
            return False
        if self.model is None:
            return self._rng.random() < 0.2

        return self._binary_yes(obs, "block", None, block.blocking_card.value)

    def choose_card_to_lose(self, obs: Observation) -> int:
        """
        Heuristic: reveal the card with the LOWEST strategic value.
        If two cards have equal value, reveal the one we already revealed
        (though that shouldn't happen — we're choosing from hidden cards).
        """
        if not obs.my_hand:
            return 0
        values = [CARD_VALUE.get(c, 1) for c in obs.my_hand]
        return int(np.argmin(values))

    def choose_exchange_cards(
        self,
        obs:      Observation,
        options:  List[Card],
        n_keep:   int,
    ) -> List[int]:
        """
        Heuristic: keep the n_keep cards with the highest strategic value.
        Ties broken by card position in options list.
        """
        if len(options) <= n_keep:
            return list(range(len(options)))
        values  = [(CARD_VALUE.get(c, 1), i) for i, c in enumerate(options)]
        top     = sorted(values, key=lambda x: -x[0])[:n_keep]
        return sorted(idx for _, idx in top)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _maybe_reset(self, obs: Observation) -> None:
        """Reset per-game state if we detect a new game has started."""
        if obs.turn_number < self._last_turn or obs.turn_number == 0:
            self._action_history = []
        self._last_turn = obs.turn_number

    def _featurise(
        self,
        obs:          Observation,
        action_type:  str,
        target_idx:   Optional[int],
        claimed_card: Optional[str],
    ) -> np.ndarray:
        """Build a normalised feature vector from an Observation."""
        feat = features_from_observation(
            obs=obs,
            action_type=action_type,
            target_idx=target_idx,
            claimed_card=claimed_card,
            action_history=self._action_history,
            config=self.config,
        )
        # Apply stored normalisation if available
        if self.feat_mean is not None and self.feat_std is not None:
            feat = (feat - self.feat_mean) / (self.feat_std + 1e-8)
        return feat.astype(np.float32)

    def _binary_yes(
        self,
        obs:          Observation,
        action_type:  str,
        target_idx:   Optional[int],
        claimed_card: Optional[str],
    ) -> bool:
        """Run binary head and sample yes/no."""
        try:
            feat = self._featurise(obs, action_type, target_idx, claimed_card)
            x    = torch.from_numpy(feat).unsqueeze(0)
            p_yes = self.model.binary_prob_yes(x, self.temperature).item()
            return self._rng.random() < p_yes
        except Exception:
            return self._rng.random() < 0.2

    def _target_slot(self, obs: Observation, target_idx: int) -> int:
        """Convert absolute target_idx to a relative opponent slot (0–4)."""
        n_total = len(obs.others) + 1
        return (target_idx - obs.my_idx - 1) % n_total
