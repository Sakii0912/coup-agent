"""
agents/random_agent.py — Uniformly random agent.

Makes every decision randomly (uniformly over legal options).
Configurable challenge and block probabilities.

This agent serves two purposes:
  1. Baseline opponent to benchmark future agents against.
  2. Data generator — run thousands of random games to produce training logs.
"""

from __future__ import annotations

import random
from typing import List, Optional

from .base import Agent
from ..cards import Card
from ..state import Observation
from ..actions import Action, Block


class RandomAgent(Agent):
    """
    Agent that makes uniformly random legal decisions.

    Args:
        challenge_prob: Probability of challenging any given action or block.
                        Default 0.2 (20 %) matches rough human bluff-call rate.
        block_prob:     Probability of blocking when a block is available.
                        Default 0.3.
        seed:           Optional per-agent RNG seed for reproducibility.
    """

    def __init__(
        self,
        challenge_prob: float = 0.2,
        block_prob: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        self.challenge_prob = challenge_prob
        self.block_prob = block_prob
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    #  Agent interface implementation                                      #
    # ------------------------------------------------------------------ #

    def choose_action(
        self, obs: Observation, legal_actions: List[Action]
    ) -> Action:
        return self._rng.choice(legal_actions)

    def choose_to_challenge_action(
        self, obs: Observation, action: Action
    ) -> bool:
        if action.actor_idx == obs.my_idx:
            return False  # Never challenge yourself (safety guard)
        return self._rng.random() < self.challenge_prob

    def choose_to_block(
        self,
        obs: Observation,
        action: Action,
        legal_blocks: List[Block],
    ) -> Optional[Block]:
        if not legal_blocks:
            return None
        if self._rng.random() < self.block_prob:
            return self._rng.choice(legal_blocks)
        return None

    def choose_to_challenge_block(
        self, obs: Observation, block: Block
    ) -> bool:
        if block.blocker_idx == obs.my_idx:
            return False  # Never challenge your own block
        return self._rng.random() < self.challenge_prob

    def choose_card_to_lose(self, obs: Observation) -> int:
        return self._rng.randrange(len(obs.my_hand))

    def choose_exchange_cards(
        self,
        obs: Observation,
        options: List[Card],
        n_keep: int,
    ) -> List[int]:
        return self._rng.sample(range(len(options)), n_keep)


class HonestAgent(RandomAgent):
    """
    Agent that never bluffs — only plays actions it actually holds cards for,
    and always blocks/challenges correctly when it has information.

    Useful as a sanity-check opponent and for studying bluffing value.
    """

    def choose_action(
        self, obs: Observation, legal_actions: List[Action]
    ) -> Action:
        # Filter to actions the agent truly holds the card for
        honest = [
            a for a in legal_actions
            if a.claimed_card is None or a.claimed_card in obs.my_hand
        ]
        if honest:
            return self._rng.choice(honest)
        # Fallback: must take an honest general action
        safe = [
            a for a in legal_actions
            if a.claimed_card is None
        ]
        return self._rng.choice(safe) if safe else self._rng.choice(legal_actions)

    def choose_to_challenge_action(
        self, obs: Observation, action: Action
    ) -> bool:
        if action.actor_idx == obs.my_idx:
            return False
        # Challenge only if we *know* nobody can hold the card
        # (simplified: challenge if claimed card is in our hand and deck is small)
        if action.claimed_card and action.claimed_card in obs.my_hand:
            # We hold this card — more likely they're bluffing
            return self._rng.random() < 0.6
        return False

    def choose_to_block(
        self,
        obs: Observation,
        action: Action,
        legal_blocks: List[Block],
    ) -> Optional[Block]:
        # Only block with cards we actually hold
        honest_blocks = [b for b in legal_blocks if b.blocking_card in obs.my_hand]
        if honest_blocks:
            return self._rng.choice(honest_blocks)
        return None
