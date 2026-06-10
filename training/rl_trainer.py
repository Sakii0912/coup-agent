"""
training/rl_trainer.py — REINFORCE self-play trainer for NeuralAgent (sub-goal 4.3).

─────────────────────────────────────────────────────────────────────────────
Algorithm: REINFORCE (Williams 1992) with self-play
─────────────────────────────────────────────────────────────────────────────

Each training round:
  1. COLLECT: play K games between copies of the current NeuralAgent.
              At every action decision, record:
                (feature_vector, action_idx, target_idx, player_idx, game_id)
  2. ASSIGN:  when each game ends, compute per-player returns:
                winner  → +1
                loser   → -(1 / (n_players - 1))
              Broadcast each player's return to all their recorded decisions.
  3. UPDATE:  REINFORCE gradient update:
                loss = -mean( log_prob(a_t) * G_t )   (action head)
                     + -λ * mean( log_prob(tgt_t) * G_t ) (target head, weighted)
              Clip gradients, step optimizer.
  4. REPEAT for N rounds.

─────────────────────────────────────────────────────────────────────────────
BeliefTracker integration
─────────────────────────────────────────────────────────────────────────────

Each seat maintains a BeliefTracker that is updated after every observable
game event. Before each decision, the Observation is enriched with the
current belief state and opponent model. The feature extractor then produces
a 140-dim vector (if include_belief=True) that captures the agent's full
information state including probabilistic beliefs about hidden cards.

This means the policy network is implicitly trained to use belief information
when it's available — without any architectural changes needed between
pretraining (104-dim, no belief) and RL fine-tuning (140-dim, with belief).

─────────────────────────────────────────────────────────────────────────────
RecordingAgent
─────────────────────────────────────────────────────────────────────────────

A thin wrapper around NeuralAgent that:
  - Keeps BeliefTracker alive across decisions in the same game
  - Records (feature_vec, action_idx) for every choose_action call
  - Exposes experiences after the game ends

This is separate from NeuralAgent to keep the Agent ABC clean.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from coup.cards import Card, ActionType
from coup.state import GameState, Observation
from coup.actions import Action, Block, get_legal_actions
from coup.agents.base import Agent
from coup.game import Game
from belief.belief_tracker import BeliefTracker
from pipeline.feature_extractor import (
    FeatureConfig,
    features_from_observation,
    ACTION_TO_IDX,
    N_ACTIONS,
)


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """One recorded decision point from a self-play game."""
    feature_vec: np.ndarray   # normalised input to the policy net
    action_idx:  int           # index into ACTIONS (0–6)
    target_idx:  int           # index into target slots (0–5)
    player_idx:  int           # which seat made this decision
    game_id:     int           # which game this came from
    turn:        int           # turn number


# ---------------------------------------------------------------------------
# RecordingAgent
# ---------------------------------------------------------------------------

class RecordingAgent(Agent):
    """
    Wraps a NeuralAgent and records experiences during self-play.
    Maintains its own BeliefTracker updated after each observable event.

    The Game class calls the six Agent methods directly; we intercept
    choose_action to record the feature vector and chosen action.
    """

    def __init__(
        self,
        neural_agent,               # NeuralAgent (may be None → random fallback)
        player_idx: int,
        n_players:  int,
        game_id:    int,
        feat_mean:  Optional[np.ndarray] = None,
        feat_std:   Optional[np.ndarray] = None,
        config:     Optional[FeatureConfig] = None,
    ) -> None:
        self._agent      = neural_agent
        self.player_idx  = player_idx
        self.n_players   = n_players
        self.game_id     = game_id
        self.feat_mean   = self._to_numpy(feat_mean)
        self.feat_std    = self._to_numpy(feat_std)
        self.config      = config or FeatureConfig()
        self.experiences: List[Experience] = []
        self._action_history: List[str] = []
        self._tracker: Optional[BeliefTracker] = None

    def init_tracker(self, known_hand: List[Card]) -> None:
        """Call once at game start with the player's initial hand."""
        self._tracker = BeliefTracker.from_game_start(
            n_players=self.n_players,
            observer_idx=self.player_idx,
            known_hand=known_hand,
        )

    def update_tracker(self, event_type: str, **kwargs) -> None:
        """Push a game event to the belief tracker."""
        if self._tracker is not None:
            self._tracker.process_raw(event_type, **kwargs)

    # ── Agent ABC ──────────────────────────────────────────────────────

    def choose_action(self, obs: Observation, legal_actions: List[Action]) -> Action:
        # Enrich observation with belief state before featurising
        if self._tracker is not None:
            self._tracker.enrich_observation(obs)

        # Build feature vector
        feat = self._featurise(obs, "Income", None, None)

        # Delegate decision to neural agent (or random fallback)
        if self._agent is not None:
            action = self._agent.choose_action(obs, legal_actions)
        else:
            import random
            action = random.choice(legal_actions)

        # Record experience
        a_idx = ACTION_TO_IDX.get(action.type.value, 0)
        t_idx = self._target_slot(obs, action.target_idx) if action.target_idx is not None else 5
        self.experiences.append(Experience(
            feature_vec=feat,
            action_idx=a_idx,
            target_idx=t_idx,
            player_idx=self.player_idx,
            game_id=self.game_id,
            turn=obs.turn_number,
        ))
        self._action_history.append(action.type.value)
        return action

    def choose_to_challenge_action(self, obs: Observation, action: Action) -> bool:
        if self._tracker is not None:
            self._tracker.enrich_observation(obs)
        if self._agent is not None:
            return self._agent.choose_to_challenge_action(obs, action)
        import random
        return random.random() < 0.15

    def choose_to_block(self, obs: Observation, action: Action,
                        legal_blocks: List[Block]) -> Optional[Block]:
        if self._tracker is not None:
            self._tracker.enrich_observation(obs)
        if self._agent is not None:
            return self._agent.choose_to_block(obs, action, legal_blocks)
        import random
        return random.choice(legal_blocks) if legal_blocks and random.random() < 0.25 else None

    def choose_to_challenge_block(self, obs: Observation, block: Block) -> bool:
        if self._tracker is not None:
            self._tracker.enrich_observation(obs)
        if self._agent is not None:
            return self._agent.choose_to_challenge_block(obs, block)
        import random
        return random.random() < 0.15

    def choose_card_to_lose(self, obs: Observation) -> int:
        if self._agent is not None:
            return self._agent.choose_card_to_lose(obs)
        return 0

    def choose_exchange_cards(self, obs: Observation, options: List[Card],
                              n_keep: int) -> List[int]:
        if self._agent is not None:
            return self._agent.choose_exchange_cards(obs, options, n_keep)
        import random
        return random.sample(range(len(options)), n_keep)

    # ── Helpers ────────────────────────────────────────────────────────

    def _featurise(self, obs: Observation, action_type: str,
                   target_idx: Optional[int], claimed_card: Optional[str]) -> np.ndarray:
        feat = features_from_observation(
            obs=obs,
            action_type=action_type,
            target_idx=target_idx,
            claimed_card=claimed_card,
            action_history=self._action_history,
            config=self.config,
        )
        if self.feat_mean is not None and self.feat_std is not None:
            feat = (feat - self.feat_mean) / (self.feat_std + 1e-8)
        return feat.astype(np.float32)

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return value

    def _target_slot(self, obs: Observation, target_idx: int) -> int:
        n_total = len(obs.others) + 1
        return min((target_idx - obs.my_idx - 1) % n_total, 5)


# ---------------------------------------------------------------------------
# RLTrainer
# ---------------------------------------------------------------------------

class RLTrainer:
    """
    REINFORCE self-play trainer for NeuralAgent.

    Args:
        model:           Trained CoupPolicyNet (from Phase 4.2 pretraining).
        feat_mean:       Feature normalisation mean from pretraining checkpoint.
        feat_std:        Feature normalisation std from pretraining checkpoint.
        n_players:       Players per game.
        output_dir:      Directory for model checkpoints and logs.
        lr:              Learning rate for RL fine-tuning. Default 1e-4
                         (lower than supervised to avoid catastrophic forgetting).
        games_per_round: Games collected per gradient update. Default 64.
        action_weight:   Loss weight for action head. Default 1.0.
        target_weight:   Loss weight for target head. Default 0.3.
        entropy_coeff:   Entropy regularisation coefficient. Encourages
                         exploration by penalising too-peaked distributions.
        use_belief:      Whether to attach BeliefTracker (140-dim features).
        seed:            RNG seed.
    """

    def __init__(
        self,
        model,
        feat_mean:       Optional[np.ndarray] = None,
        feat_std:        Optional[np.ndarray] = None,
        n_players:       int                  = 4,
        output_dir:      str                  = "data/models/rl",
        lr:              float                = 1e-4,
        games_per_round: int                  = 64,
        action_weight:   float                = 1.0,
        target_weight:   float                = 0.3,
        entropy_coeff:   float                = 0.01,
        use_belief:      bool                 = True,
        seed:            Optional[int]        = None,
    ) -> None:
        import torch, torch.nn as nn, torch.optim as optim
        self.model         = model
        self.feat_mean     = self._to_numpy(feat_mean)
        self.feat_std      = self._to_numpy(feat_std)
        self.n_players     = n_players
        self.output_dir    = output_dir
        self.games_per_round = games_per_round
        self.action_weight = action_weight
        self.target_weight = target_weight
        self.entropy_coeff = entropy_coeff
        self.use_belief    = use_belief

        if use_belief:
            self.config = FeatureConfig(include_belief=True, include_opp_model=True)
        else:
            self.config = FeatureConfig()

        self.optimizer  = optim.Adam(model.parameters(), lr=lr)
        self._game_counter = 0
        self._log: List[Dict[str, Any]] = []
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return value

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def train(self, n_rounds: int) -> Dict[str, Any]:
        """
        Run `n_rounds` of collect → assign returns → update.

        Returns final training stats.
        """
        import torch
        t0 = time.time()

        print(f"\nRL self-play training: {n_rounds} rounds × "
              f"{self.games_per_round} games/round "
              f"({'140-dim belief' if self.use_belief else '104-dim'})")
        print(f"{'Round':>6}  {'Games':>7}  {'AvgRet':>8}  "
              f"{'ActLoss':>9}  {'Entropy':>8}")
        print("-" * 48)

        for rnd in range(1, n_rounds + 1):
            # ── Collect ──────────────────────────────────────────────────
            experiences, returns = self._collect_games(self.games_per_round)

            if not experiences:
                continue

            # ── Update ───────────────────────────────────────────────────
            stats = self._update(experiences, returns)
            stats["round"]      = rnd
            stats["games_total"] = self._game_counter
            self._log.append(stats)

            print(f"{rnd:>6}  {self._game_counter:>7,}  "
                  f"{stats['mean_return']:>8.4f}  "
                  f"{stats['action_loss']:>9.4f}  "
                  f"{stats['entropy']:>8.4f}")

            # Checkpoint every 10 rounds
            if rnd % 10 == 0:
                self._save_checkpoint(rnd)

        self._save_checkpoint(n_rounds)
        self._save_log()

        elapsed = time.time() - t0
        print(f"\nDone. {n_rounds} rounds in {elapsed:.1f}s")
        return {"n_rounds": n_rounds, "games_total": self._game_counter,
                "elapsed_s": round(elapsed, 2)}

    # ------------------------------------------------------------------ #
    #  Collection                                                         #
    # ------------------------------------------------------------------ #

    def _collect_games(
        self, n_games: int
    ) -> Tuple[List[Experience], Dict[Tuple[int, int], float]]:
        """
        Play `n_games` and return all experiences + per-(game,player) returns.
        """
        from agents.neural.neural_agent import NeuralAgent

        all_experiences: List[Experience] = []
        # returns[(game_id, player_idx)] = float reward
        returns: Dict[Tuple[int, int], float] = {}

        player_names = [f"P{i}" for i in range(self.n_players)]

        for _ in range(n_games):
            game_id = self._game_counter
            self._game_counter += 1

            # Build one RecordingAgent per seat
            neural = NeuralAgent(
                model=self.model,
                feat_mean=self.feat_mean,
                feat_std=self.feat_std,
                config=self.config,
            )
            recorders: List[RecordingAgent] = [
                RecordingAgent(
                    neural_agent=neural,
                    player_idx=i,
                    n_players=self.n_players,
                    game_id=game_id,
                    feat_mean=self.feat_mean,
                    feat_std=self.feat_std,
                    config=self.config,
                )
                for i in range(self.n_players)
            ]

            # Run the game
            state = GameState.new_game(player_names, seed=game_id)

            # Initialise trackers with known hands
            for i, rec in enumerate(recorders):
                rec.init_tracker(list(state.players[i].hand))

            game = Game(agents=recorders, player_names=player_names, seed=game_id)
            log  = game.play_game()

            # Determine winner and assign returns
            end  = next((e for e in reversed(log["events"])
                         if e["event_type"] == "game_end"), None)
            winner_idx = end.get("winner_idx") if end else None

            for i in range(self.n_players):
                if winner_idx is not None:
                    r = 1.0 if i == winner_idx else -1.0 / max(self.n_players - 1, 1)
                else:
                    r = 0.0
                returns[(game_id, i)] = r

            # Collect experiences
            for rec in recorders:
                all_experiences.extend(rec.experiences)

        return all_experiences, returns

    # ------------------------------------------------------------------ #
    #  REINFORCE update                                                   #
    # ------------------------------------------------------------------ #

    def _update(
        self,
        experiences: List[Experience],
        returns:     Dict[Tuple[int, int], float],
    ) -> Dict[str, float]:
        """
        REINFORCE gradient update on collected experiences.

        Loss = -E[log π(a|s) * G]  (action head)
             + -λ * E[log π(tgt|s) * G]  (target head)
             - η * H[π(·|s)]  (entropy bonus)
        """
        import torch
        import torch.nn.functional as F

        if not experiences:
            return {"action_loss": 0.0, "target_loss": 0.0,
                    "entropy": 0.0, "mean_return": 0.0}

        # Build tensors
        feats   = np.stack([e.feature_vec for e in experiences])   # (N, D)
        acts    = np.array([e.action_idx  for e in experiences])   # (N,)
        tgts    = np.array([e.target_idx  for e in experiences])   # (N,)
        G_vals  = np.array([
            returns.get((e.game_id, e.player_idx), 0.0)
            for e in experiences
        ], dtype=np.float32)                                        # (N,)

        X = torch.from_numpy(feats)
        a = torch.from_numpy(acts).long()
        t = torch.from_numpy(tgts).long()
        G = torch.from_numpy(G_vals)

        # Normalise returns (reduces variance)
        if G.std() > 1e-6:
            G = (G - G.mean()) / (G.std() + 1e-8)

        # Forward pass (with grad)
        self.model.train()
        out = self.model(X)

        # Action loss
        log_probs_act = F.log_softmax(out["action"], dim=-1)
        sel_log_act   = log_probs_act.gather(1, a.unsqueeze(1)).squeeze(1)
        action_loss   = -(sel_log_act * G).mean()

        # Target loss
        log_probs_tgt = F.log_softmax(out["target"], dim=-1)
        sel_log_tgt   = log_probs_tgt.gather(1, t.unsqueeze(1)).squeeze(1)
        target_loss   = -(sel_log_tgt * G).mean()

        # Entropy bonus (encourages exploration)
        probs_act = F.softmax(out["action"], dim=-1)
        entropy   = -(probs_act * log_probs_act).sum(dim=-1).mean()

        total_loss = (self.action_weight * action_loss
                      + self.target_weight * target_loss
                      - self.entropy_coeff * entropy)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.model.eval()

        return {
            "action_loss": action_loss.item(),
            "target_loss": target_loss.item(),
            "entropy":     entropy.item(),
            "mean_return": float(G_vals.mean()),
            "n_experiences": len(experiences),
        }

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, rnd: int) -> None:
        import torch
        path = os.path.join(self.output_dir, f"rl_model_round{rnd:05d}.pt")
        feat_mean = torch.as_tensor(self.feat_mean) if self.feat_mean is not None else None
        feat_std = torch.as_tensor(self.feat_std) if self.feat_std is not None else None
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim":  self.model.input_dim,
            "hidden_dim": self.model.hidden_dim,
            "round":      rnd,
            "feat_mean":  feat_mean,
            "feat_std":   feat_std,
        }, path)
        # Also save latest
        latest = os.path.join(self.output_dir, "rl_latest.pt")
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim":  self.model.input_dim,
            "hidden_dim": self.model.hidden_dim,
            "round":      rnd,
            "feat_mean":  feat_mean,
            "feat_std":   feat_std,
        }, latest)
        print(f"  ✓ RL checkpoint saved (round {rnd})")

    def _save_log(self) -> None:
        path = os.path.join(self.output_dir, "rl_training_log.json")
        with open(path, "w") as f:
            json.dump(self._log, f, indent=2)
