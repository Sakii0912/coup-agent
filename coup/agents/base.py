"""
agents/base.py — Abstract base class for all Coup agents.

Every agent (random, rule-based, RL, CFR, human-interface) must implement
these six decision points. The signatures are kept pure and stateless here;
agents may maintain internal state between calls if needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..cards import Card
from ..state import Observation
from ..actions import Action, Block


class Agent(ABC):
    """
    Abstract Coup agent.

    Decision points (called by Game in order during each turn):
      1. choose_action            — active player's main action
      2. choose_to_challenge_action — can any other player challenge it?
      3. choose_to_block          — can target/others block it?
      4. choose_to_challenge_block — can anyone challenge the block?
      5. choose_card_to_lose      — which card to reveal when losing influence
      6. choose_exchange_cards    — which cards to keep during Ambassador exchange
    """

    # ------------------------------------------------------------------ #
    #  Primary turn decision                                               #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def choose_action(
        self, obs: Observation, legal_actions: List[Action]
    ) -> Action:
        """
        Choose an action on your turn.

        Args:
            obs:           What you can observe about the current game state.
            legal_actions: Non-empty list of actions you may legally play.

        Returns:
            One action from `legal_actions`.
        """

    # ------------------------------------------------------------------ #
    #  Reaction decisions (called on every non-active alive player)        #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def choose_to_challenge_action(
        self, obs: Observation, action: Action
    ) -> bool:
        """
        Decide whether to challenge the active player's action claim.

        Only called when `action.is_challengeable` is True.
        Called for every alive player except the actor, in seat order.
        First player to challenge stops the loop.

        Returns:
            True to challenge, False to pass.
        """

    @abstractmethod
    def choose_to_block(
        self,
        obs: Observation,
        action: Action,
        legal_blocks: List[Block],
    ) -> Optional[Block]:
        """
        Decide whether to block an incoming action.

        Called for Foreign Aid: every other alive player.
        Called for Assassinate / Steal: the target only.

        Args:
            legal_blocks: All block claims you can legally make (may be empty).

        Returns:
            A Block from `legal_blocks`, or None to not block.
        """

    @abstractmethod
    def choose_to_challenge_block(
        self, obs: Observation, block: Block
    ) -> bool:
        """
        Decide whether to challenge a blocker's card claim.

        Called for every alive player except the blocker, in seat order.

        Returns:
            True to challenge the block, False to accept it.
        """

    # ------------------------------------------------------------------ #
    #  Forced decisions                                                    #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def choose_card_to_lose(self, obs: Observation) -> int:
        """
        Choose which card to reveal when you must lose one influence.

        Returns:
            Index into `obs.my_hand` of the card to reveal.
        """

    @abstractmethod
    def choose_exchange_cards(
        self,
        obs: Observation,
        options: List[Card],
        n_keep: int,
    ) -> List[int]:
        """
        During an Ambassador Exchange, choose which cards to keep.

        Args:
            options:  Your current hand + up to 2 drawn cards (combined list).
            n_keep:   How many cards you must keep (= your current hand size).

        Returns:
            A list of exactly `n_keep` distinct indices into `options`.
        """
