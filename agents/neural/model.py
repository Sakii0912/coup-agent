"""
agents/neural/model.py — CoupPolicyNet architecture (sub-goal 4.2).

─────────────────────────────────────────────────────────────────────────────
Architecture
─────────────────────────────────────────────────────────────────────────────

  Input → Trunk → three task heads

  Trunk (shared):
    Linear(input_dim, 256) → LayerNorm → ReLU
    Linear(256, 256) → LayerNorm → ReLU  }  residual connection
    Linear(256, 128) → LayerNorm → ReLU

  Task heads (separate):
    action_head  : Linear(128, 7)  — which action type to take
    target_head  : Linear(128, 6)  — which opponent slot to target (5 = no target)
    binary_head  : Linear(128, 2)  — yes / no for challenge and block decisions

  Works with both the 104-dim Phase 2 feature vector and the 140-dim Phase 3
  vector (with belief state). Pass input_dim=140 when using Phase 3 features.

─────────────────────────────────────────────────────────────────────────────
Design choices
─────────────────────────────────────────────────────────────────────────────

  LayerNorm instead of BatchNorm: stable at small batch sizes (single-game
  inference during self-play), unlike BatchNorm which requires batch > 1.

  Residual connection in the second block: helps gradient flow with a deeper
  trunk without introducing skip-connection complexity in the first block.

  Shared trunk: the game-state encoding is the same regardless of which
  decision is being made. Only the final layer differs per task. This also
  means supervised training on one task (action selection) implicitly
  improves the features used for the other tasks.

  Separate binary head for challenges AND blocks: both are binary yes/no
  decisions with similar inputs. Sharing a head reduces parameters and
  regularises the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Constants matching feature_extractor.py
# ---------------------------------------------------------------------------

N_ACTION_TYPES  = 7    # Income, ForeignAid, Coup, Tax, Assassinate, Steal, Exchange
N_TARGET_SLOTS  = 6    # 5 opponent slots + 1 "no target" slot
N_BINARY        = 2    # yes / no


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Two-layer residual block with LayerNorm and ReLU."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


# ---------------------------------------------------------------------------
# CoupPolicyNet
# ---------------------------------------------------------------------------

class CoupPolicyNet(nn.Module):
    """
    Multi-task policy network for Coup.

    Args:
        input_dim:   Feature vector size. 104 (Phase 2) or 140 (Phase 3).
        hidden_dim:  Width of the trunk hidden layers. Default 256.
        dropout:     Dropout probability in residual blocks. Default 0.1.
    """

    def __init__(
        self,
        input_dim:  int   = 104,
        hidden_dim: int   = 256,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # ── Trunk ──────────────────────────────────────────────────────────
        # Block 1: project input to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        # Block 2: residual block at full width
        self.res_block = ResidualBlock(hidden_dim, dropout=dropout)
        # Block 3: compress to 128
        self.compress = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # ── Task heads ────────────────────────────────────────────────────
        self.action_head = nn.Linear(128, N_ACTION_TYPES)   # 7 action types
        self.target_head = nn.Linear(128, N_TARGET_SLOTS)   # 6 target slots
        self.binary_head = nn.Linear(128, N_BINARY)         # yes/no

        # Weight initialisation: small uniform for heads → near-uniform initial policy
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared trunk and return the 128-dim feature representation."""
        h = self.input_proj(x)
        h = self.res_block(h)
        h = self.compress(h)
        return h

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Float tensor of shape (..., input_dim).

        Returns:
            Dict with keys "action", "target", "binary" → raw logits.
        """
        h = self.trunk(x)
        return {
            "action": self.action_head(h),    # (..., 7)
            "target": self.target_head(h),    # (..., 6)
            "binary": self.binary_head(h),    # (..., 2)
        }

    # ------------------------------------------------------------------ #
    #  Convenience inference methods                                      #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def action_probs(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Action-type probabilities with optional legal-action masking and temperature.

        Args:
            x:           Input tensor (..., input_dim).
            legal_mask:  Boolean tensor (..., 7). True = legal. If None, all legal.
            temperature: Softmax temperature. <1 = sharper, >1 = more uniform.

        Returns:
            Probability tensor (..., 7).
        """
        logits = self.forward(x)["action"] / max(temperature, 1e-6)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def target_probs(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Target-slot probabilities with optional masking."""
        logits = self.forward(x)["target"] / max(temperature, 1e-6)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def binary_prob_yes(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """P(yes) from the binary head. Shape (...,)."""
        logits = self.forward(x)["binary"] / max(temperature, 1e-6)
        return F.softmax(logits, dim=-1)[..., 0]   # index 0 = "yes"

    # ------------------------------------------------------------------ #
    #  Serialisation                                                      #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save model weights and config."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim":  self.input_dim,
            "hidden_dim": self.hidden_dim,
        }, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "CoupPolicyNet":
        """Load model from a saved checkpoint."""
        ckpt = torch.load(path, map_location=map_location)
        net  = cls(input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"])
        net.load_state_dict(ckpt["state_dict"])
        return net

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"CoupPolicyNet(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"params={n_params:,})"
        )
