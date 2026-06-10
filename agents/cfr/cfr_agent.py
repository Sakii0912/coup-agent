"""
agents/cfr/cfr_agent.py — CFRAgent implementing the Agent ABC (sub-goal 4.1).

Wraps a trained StrategyTable behind the six Agent decision points.
At each decision, it:
  1. Builds the correct InformationSet from the Observation.
  2. Looks up the AVERAGE strategy (not current — the average converges
     toward Nash equilibrium, the current strategy oscillates).
  3. Samples an action from that distribution.
  4. Falls back to a uniform random choice for infosets never seen during
     training (the table has no entry → average strategy = uniform).

The agent is stateless across games — no history or belief state is
maintained here. For a richer version with BeliefTracker integration,
see training/rl_trainer.py's RecordingAgent, which wraps NeuralAgent.
"""

from __future__ import annotations

import random
from typing import List, Optional

from coup.cards import Card
from coup.state import Observation
from coup.actions import Action, Block, get_legal_actions, get_legal_blocks
from coup.agents.base import Agent

from .information_set import (
    InformationSet,
    legal_main_action_keys,
    legal_challenge_keys,
    legal_block_keys,
    legal_lose_influence_keys,
    legal_exchange_keys,
    decode_main_action,
    decode_block,
    decode_lose_influence,
    decode_exchange_keep,
)
from .strategy_table import StrategyTable


class CFRAgent(Agent):
    """
    Coup agent driven by a trained CFR strategy table.

    Args:
        table:     Trained StrategyTable (from CFRSolver or loaded from disk).
        n_players: Total players in the game (used for InformationSet context).
        seed:      Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        table:     StrategyTable,
        n_players: int            = 4,
        seed:      Optional[int]  = None,
    ) -> None:
        self.table     = table
        self.n_players = n_players
        self._rng      = random.Random(seed)
        self._history: List[str] = []
        self._last_turn: int = -1

    # ------------------------------------------------------------------ #
    #  Agent ABC                                                          #
    # ------------------------------------------------------------------ #

    def choose_action(
        self, obs: Observation, legal_actions: List[Action]
    ) -> Action:
        self._maybe_reset(obs)

        infoset     = InformationSet.for_main_action(obs, self._history)
        action_keys = legal_main_action_keys(obs, legal_actions)
        strategy    = self.table.average_strategy(infoset, action_keys)
        chosen_key  = self._sample(strategy, action_keys)

        chosen = decode_main_action(chosen_key, obs, legal_actions)
        if chosen is None:
            chosen = self._rng.choice(legal_actions)

        self._history.append(chosen.type.value)
        return chosen

    def choose_to_challenge_action(
        self, obs: Observation, action: Action
    ) -> bool:
        if action.actor_idx == obs.my_idx:
            return False
        claimed  = action.claimed_card.value if action.claimed_card else ""
        infoset  = InformationSet.for_challenge_action(obs, self._history, claimed)
        keys     = legal_challenge_keys()
        strategy = self.table.average_strategy(infoset, keys)
        return self._sample(strategy, keys) == "Challenge"

    def choose_to_block(
        self,
        obs:          Observation,
        action:       Action,
        legal_blocks: List[Block],
    ) -> Optional[Block]:
        if not legal_blocks:
            return None
        infoset  = InformationSet.for_block(obs, self._history, action.type.value)
        keys     = legal_block_keys(legal_blocks)
        strategy = self.table.average_strategy(infoset, keys)
        chosen   = self._sample(strategy, keys)
        return decode_block(chosen, legal_blocks)

    def choose_to_challenge_block(
        self, obs: Observation, block: Block
    ) -> bool:
        if block.blocker_idx == obs.my_idx:
            return False
        infoset  = InformationSet.for_challenge_block(
            obs, self._history, block.blocking_card.value
        )
        keys     = legal_challenge_keys()
        strategy = self.table.average_strategy(infoset, keys)
        return self._sample(strategy, keys) == "Challenge"

    def choose_card_to_lose(self, obs: Observation) -> int:
        if not obs.my_hand:
            return 0
        infoset  = InformationSet.for_lose_influence(obs, self._history)
        keys     = legal_lose_influence_keys(obs)
        if not keys:
            return 0
        strategy = self.table.average_strategy(infoset, keys)
        chosen   = self._sample(strategy, keys)
        return decode_lose_influence(chosen, obs)

    def choose_exchange_cards(
        self,
        obs:     Observation,
        options: List[Card],
        n_keep:  int,
    ) -> List[int]:
        infoset  = InformationSet.for_exchange(obs, self._history)
        keys     = legal_exchange_keys(options, n_keep)
        if not keys:
            return list(range(n_keep))
        strategy = self.table.average_strategy(infoset, keys)
        chosen   = self._sample(strategy, keys)
        return decode_exchange_keep(chosen, options, n_keep)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _sample(self, strategy: dict, keys: List[str]) -> str:
        """Sample one key from a probability distribution."""
        r   = self._rng.random()
        cum = 0.0
        for k in keys:
            cum += strategy.get(k, 1.0 / len(keys))
            if r <= cum:
                return k
        return keys[-1]

    def _maybe_reset(self, obs: Observation) -> None:
        """Detect a new game and reset action history."""
        if obs.turn_number == 0 or obs.turn_number < self._last_turn:
            self._history = []
        self._last_turn = obs.turn_number

    def __repr__(self) -> str:
        return (
            f"CFRAgent(infosets={self.table.n_infosets:,}, "
            f"iterations={self.table.n_iterations:,})"
        )
