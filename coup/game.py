"""
game.py — Main game orchestrator.

The Game class drives the full turn loop:
  1. Active player declares an action
  2. Other players may challenge the action (if challengeable)
  3. Eligible players may block the action (if blockable and still alive)
  4. Other players may challenge the block
  5. If action survived all reactions → apply its effect

All agent callbacks flow through here. The Game never peeks at agent
internals — it only calls the six Agent interface methods.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .cards import ActionType, Card
from .state import GameState, PlayerState, Observation
from .actions import Action, Block, get_legal_actions, get_legal_blocks
from .resolution import resolve_challenge, apply_action_effect
from .logger import GameLogger
from .agents.base import Agent


class Game:
    """
    Orchestrates a complete game of Coup.

    Usage:
        agents = [RandomAgent(), RandomAgent(), RandomAgent()]
        game = Game(agents, seed=42)
        log = game.play_game()
    """

    MAX_TURNS = 500  # Safety cap to prevent infinite loops

    def __init__(
        self,
        agents: List[Agent],
        player_names: Optional[List[str]] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        assert 2 <= len(agents) <= 6, "Coup supports 2–6 players"
        if player_names is None:
            player_names = [f"Player{i}" for i in range(len(agents))]
        assert len(player_names) == len(agents), "agents and player_names must match"

        self.agents = agents
        self.verbose = verbose
        self.state = GameState.new_game(player_names, seed=seed)
        self.logger = GameLogger(player_names)

    # ================================================================== #
    #  Public API                                                          #
    # ================================================================== #

    def play_game(self) -> Dict[str, Any]:
        """
        Play a complete game to conclusion.

        Returns:
            Game log dict (JSON-serialisable). Contains all observable events.
        """
        self.logger.log_game_start(self.state)
        self._vprint(f"=== Game start ({len(self.state.players)} players) ===")
        self._vprint(self.state)

        while not self.state.is_game_over:
            if self.state.turn_number >= self.MAX_TURNS:
                self._vprint("⚠ Turn limit reached — stopping.")
                break
            self._play_turn()
            self.state.advance_turn()

        winner = self.state.winner
        self.logger.log_game_end(winner, self.state.turn_number)
        self._vprint(
            f"\n=== Game over | Winner: "
            f"{winner.name if winner else 'Nobody'} "
            f"({self.state.turn_number} turns) ==="
        )
        return self.logger.get_log()

    # ================================================================== #
    #  Turn structure                                                      #
    # ================================================================== #

    def _play_turn(self) -> None:
        actor_idx = self.state.current_player_idx
        actor = self.state.players[actor_idx]

        if not actor.is_alive:
            return

        self._vprint(
            f"\n--- Turn {self.state.turn_number} | {actor.name} "
            f"({actor.influence_count} cards, {actor.coins} coins) ---"
        )

        # ── Step 1: Actor chooses action ──────────────────────────────────
        obs = self.state.get_observation(actor_idx)
        legal = get_legal_actions(self.state, actor_idx)
        action = self.agents[actor_idx].choose_action(obs, legal)

        assert action in legal, f"Agent returned illegal action: {action}"
        self._vprint(f"  {actor.name} → {action}")

        # Pre-deduct Assassinate cost (non-refundable even if challenged/blocked)
        if action.type == ActionType.ASSASSINATE:
            actor.coins -= 3
            self._vprint(f"  [{actor.name} pays 3 coins to assassinate]")

        self.logger.log_action(action, self.state)

        # ── Step 2: Other players may challenge the action ────────────────
        if action.is_challengeable:
            action_survived = self._run_action_challenge_phase(action)
            if not action_survived:
                return  # Actor lost the challenge → action fails

        # ── Step 3: Eligible players may block the action ─────────────────
        if action.is_blockable:
            block = self._run_block_phase(action)
            if block is not None:
                return  # Action was blocked (and block survived any challenge)

        # ── Step 4: Apply action effect ───────────────────────────────────
        apply_action_effect(
            state=self.state,
            action=action,
            lose_influence_callback=self._ask_lose_influence,
            exchange_callback=self._ask_exchange,
        )
        self.logger.log_action_resolved(action, self.state)
        self._vprint(f"  ✓ {action.type.value} resolved")

    # ================================================================== #
    #  Challenge phases                                                    #
    # ================================================================== #

    def _run_action_challenge_phase(self, action: Action) -> bool:
        """
        Poll each other alive player in seat order for a challenge.
        First challenger stops the loop.

        Returns:
            True  → action survives (no challenge, or actor won the challenge).
            False → action fails (challenger won).
        """
        for player in self._other_alive(action.actor_idx):
            obs = self.state.get_observation(player.idx)
            will_challenge = self.agents[player.idx].choose_to_challenge_action(obs, action)

            if will_challenge:
                self._vprint(
                    f"  ⚡ {player.name} challenges "
                    f"{action.type.value} ({action.claimed_card.value})"
                )
                self.logger.log_challenge(player.idx, action, self.state)

                actor_wins = resolve_challenge(
                    state=self.state,
                    challenger_idx=player.idx,
                    actor_idx=action.actor_idx,
                    claimed_card=action.claimed_card,
                    lose_influence_callback=self._ask_lose_influence,
                )

                if actor_wins:
                    self._vprint(f"  → Challenge failed: {player.name} loses influence")
                    self.logger.log_challenge_result(
                        winner_idx=action.actor_idx,
                        loser_idx=player.idx,
                        state=self.state,
                    )
                    return True   # Action proceeds

                else:
                    self._vprint(
                        f"  → Challenge succeeded: "
                        f"{self.state.players[action.actor_idx].name} loses influence"
                    )
                    self.logger.log_challenge_result(
                        winner_idx=player.idx,
                        loser_idx=action.actor_idx,
                        state=self.state,
                    )
                    return False  # Action fails

        return True  # No challenge

    def _run_block_phase(self, action: Action) -> Optional[Block]:
        """
        Poll eligible players for a block. If blocked, poll for a block challenge.

        Who can block:
          - Foreign Aid: any other alive player
          - Assassinate / Steal: only the target

        Returns:
            The Block if the action was successfully blocked, else None.
        """
        if action.type == ActionType.FOREIGN_AID:
            potential_blockers = self._other_alive(action.actor_idx)
        else:
            if action.target_idx is None:
                return None
            target = self.state.players[action.target_idx]
            if not target.is_alive:
                return None
            potential_blockers = [target]

        for blocker in potential_blockers:
            obs = self.state.get_observation(blocker.idx)
            legal_blocks = get_legal_blocks(action, blocker.idx)
            block = self.agents[blocker.idx].choose_to_block(obs, action, legal_blocks)

            if block is not None:
                self._vprint(
                    f"  🛡 {blocker.name} blocks with {block.blocking_card.value}"
                )
                self.logger.log_block(block, self.state)

                block_survived = self._run_block_challenge_phase(block)
                if block_survived:
                    self._vprint(f"  → Block stands — {action.type.value} fails")
                    return block
                else:
                    self._vprint(f"  → Block failed — {action.type.value} proceeds")
                    return None

        return None  # No block issued

    def _run_block_challenge_phase(self, block: Block) -> bool:
        """
        Poll each other alive player for a challenge to the block.
        First challenger stops the loop.

        Returns:
            True  → block survives (no challenge, or blocker won).
            False → block fails (challenger won).
        """
        for player in self._other_alive(block.blocker_idx):
            obs = self.state.get_observation(player.idx)
            will_challenge = self.agents[player.idx].choose_to_challenge_block(obs, block)

            if will_challenge:
                self._vprint(
                    f"  ⚡ {player.name} challenges block "
                    f"({block.blocking_card.value})"
                )
                self.logger.log_block_challenge(player.idx, block, self.state)

                blocker_wins = resolve_challenge(
                    state=self.state,
                    challenger_idx=player.idx,
                    actor_idx=block.blocker_idx,
                    claimed_card=block.blocking_card,
                    lose_influence_callback=self._ask_lose_influence,
                )

                if blocker_wins:
                    self._vprint(
                        f"  → Block challenge failed: {player.name} loses influence"
                    )
                    self.logger.log_challenge_result(
                        winner_idx=block.blocker_idx,
                        loser_idx=player.idx,
                        state=self.state,
                    )
                    return True   # Block stands

                else:
                    self._vprint(
                        f"  → Block challenge succeeded: "
                        f"{self.state.players[block.blocker_idx].name} loses influence"
                    )
                    self.logger.log_challenge_result(
                        winner_idx=player.idx,
                        loser_idx=block.blocker_idx,
                        state=self.state,
                    )
                    return False  # Block fails

        return True  # No challenge to block

    # ================================================================== #
    #  Agent callbacks                                                     #
    # ================================================================== #

    def _ask_lose_influence(self, player_idx: int) -> int:
        """Ask player which card to reveal; log the loss."""
        player = self.state.players[player_idx]
        obs = self.state.get_observation(player_idx)
        card_idx = self.agents[player_idx].choose_card_to_lose(obs)

        assert 0 <= card_idx < len(player.hand), (
            f"choose_card_to_lose returned out-of-range index {card_idx} "
            f"(hand size {len(player.hand)})"
        )

        card_to_lose = player.hand[card_idx]
        self.logger.log_influence_loss(player_idx, card_to_lose, self.state)
        self._vprint(f"  💀 {player.name} reveals {card_to_lose.value}")
        return card_idx

    def _ask_exchange(
        self, player_idx: int, options: List[Card], n_keep: int
    ) -> List[int]:
        """Ask Ambassador player which cards to keep."""
        obs = self.state.get_observation(player_idx)
        keep_indices = self.agents[player_idx].choose_exchange_cards(obs, options, n_keep)
        assert len(keep_indices) == n_keep, (
            f"choose_exchange_cards must return exactly {n_keep} indices"
        )
        return keep_indices

    # ================================================================== #
    #  Utilities                                                           #
    # ================================================================== #

    def _other_alive(self, exclude_idx: int) -> List[PlayerState]:
        """All alive players except the one at `exclude_idx`."""
        return [p for p in self.state.alive_players if p.idx != exclude_idx]

    def _vprint(self, msg: str) -> None:
        if self.verbose:
            print(msg)
