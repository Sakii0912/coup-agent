"""
agents/cfr/cfr_solver.py — Chance Sampling MCCFR solver for Coup (sub-goal 4.1).

─────────────────────────────────────────────────────────────────────────────
Algorithm: Chance Sampling Monte Carlo CFR
─────────────────────────────────────────────────────────────────────────────

In standard CFR for 2-player zero-sum games, the game tree is traversed
exhaustively. For Coup with N ≥ 2 players and a large game tree, we use
Chance Sampling MCCFR (Lanctot et al. 2009):

  1. Sample a random card deal (chance node) once per iteration.
  2. For the UPDATE PLAYER:  traverse ALL action branches, weight by
     counterfactual reach probability (probability of reaching this node
     if the update player had tried to reach it, ignoring their own choices).
  3. For ALL OTHER PLAYERS:  sample a single action from their current
     strategy. This approximates the expectation without full traversal.
  4. Terminal utility: +1 for winning, -(1/(n-1)) for losing, 0 for
     eliminated-but-not-last (rare in this formulation).

The key formula for updating regrets at information set I for action a:
    r_t(I, a) = σ_{-i}(π) × [v_i(I → a) - v_i(I)]
where σ_{-i}(π) is the counterfactual reach probability.

─────────────────────────────────────────────────────────────────────────────
Coup-specific game tree
─────────────────────────────────────────────────────────────────────────────

A single Coup turn has multiple sequential decision nodes:

  [ACTIVE PLAYER] → main action
       ↓
  [EACH OTHER PLAYER] → challenge? (if action is challengeable)
       ↓  (if no challenge, or challenge resolved)
  [TARGET / ALL OTHERS] → block? (if action is blockable)
       ↓  (if blocked)
  [EACH OTHER PLAYER] → challenge block?
       ↓  (throughout: players who lose challenges must lose influence)
  [AFFECTED PLAYER] → which card to lose?
       ↓
  [apply action effect, advance turn]

Each of these is a distinct decision point with its own information set
and strategy. The solver traverses all of them recursively.

─────────────────────────────────────────────────────────────────────────────
State representation
─────────────────────────────────────────────────────────────────────────────

Rather than patching the existing Game class, the solver uses its own
lightweight state machine built on top of GameState. This gives it full
control over branching at each decision node during traversal.

The CoupTreeState dataclass wraps a GameState plus all the "within-turn"
context needed to reconstruct the tree (pending action, pending block, etc.).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from coup.cards import Card, ActionType
from coup.state import GameState, PlayerState
from coup.actions import Action, Block, get_legal_actions, get_legal_blocks
from coup.resolution import resolve_challenge, apply_action_effect

from .information_set import (
    InformationSet,
    DecisionType,
    encode_main_action,
    encode_block,
    encode_lose_influence,
    encode_exchange_keep,
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


# ---------------------------------------------------------------------------
# Tree state — lightweight game state for CFR traversal
# ---------------------------------------------------------------------------

@dataclass
class CoupTreeState:
    """
    Snapshot of full game state at a single point in the CFR tree.
    Includes within-turn context not present in GameState alone.

    Deliberately kept simple: no complex logic lives here.
    All mutations are applied via copy() + modification.
    """
    game_state:       GameState
    turn_number:      int
    history:          List[str]           # observable action type strings

    # Within-turn context
    pending_action:   Optional[Action]    = None
    pending_block:    Optional[Block]     = None
    assassin_paid:    bool                = False  # track assassin coin deduction

    def copy(self) -> "CoupTreeState":
        return CoupTreeState(
            game_state     = self.game_state.copy(),
            turn_number    = self.turn_number,
            history        = list(self.history),
            pending_action = self.pending_action,
            pending_block  = self.pending_block,
            assassin_paid  = self.assassin_paid,
        )

    def observation_for(self, player_idx: int):
        """Build an Observation for player_idx from the current game state."""
        return self.game_state.get_observation(player_idx)


# ---------------------------------------------------------------------------
# Terminal utilities
# ---------------------------------------------------------------------------

def terminal_utility(game_state: GameState, player_idx: int) -> float:
    """
    Utility for player_idx at a terminal game state.

    Win  → +1.0
    Loss → -(1 / (n_players - 1))   (zero-sum normalisation)
    """
    n = len(game_state.players)
    winner = game_state.winner
    if winner is None:
        return 0.0
    if winner.idx == player_idx:
        return 1.0
    return -1.0 / max(1, n - 1)


# ---------------------------------------------------------------------------
# CFR Solver
# ---------------------------------------------------------------------------

class CFRSolver:
    """
    Chance Sampling MCCFR solver for Coup.

    Trains a StrategyTable via self-play. Each call to run() performs
    `n_iterations` full game tree traversals (one per update player per
    iteration), updating regrets and strategy sums.

    Usage:
        solver = CFRSolver(n_players=4, seed=42)
        solver.run(n_iterations=1000)
        solver.save("data/cfr_strategy.pkl.gz")

        # Use the trained table:
        agent = CFRAgent(solver.table, n_players=4)
    """

    def __init__(
        self,
        n_players:      int,
        seed:           Optional[int] = None,
        use_cfr_plus:   bool          = True,
        max_depth:      int           = 50,
    ) -> None:
        assert 2 <= n_players <= 6
        self.n_players    = n_players
        self.max_depth    = max_depth
        self.table        = StrategyTable(use_cfr_plus=use_cfr_plus)
        self._rng         = random.Random(seed)
        self._np_rng      = np.random.RandomState(seed)

        # Training statistics
        self.total_traversals: int = 0
        self.total_time_s:     float = 0.0

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(
        self,
        n_iterations:    int,
        show_progress:   bool = True,
        progress_every:  int  = 100,
    ) -> Dict[str, Any]:
        """
        Run `n_iterations` of Chance Sampling MCCFR.

        Each iteration:
          1. Sample a random game deal.
          2. For each alive player in turn: update that player's regrets.

        Returns a stats dict with convergence metrics.
        """
        t0 = time.time()

        for i in range(n_iterations):
            # Sample a fresh deal for this iteration
            seed = self._rng.randint(0, 2**31)
            player_names = [f"P{j}" for j in range(self.n_players)]
            root_state = CoupTreeState(
                game_state  = GameState.new_game(player_names, seed=seed),
                turn_number = 0,
                history     = [],
            )

            # Update each player's regrets on this deal
            for update_player in range(self.n_players):
                if root_state.game_state.players[update_player].is_alive:
                    self._traverse_turn(
                        state=root_state.copy(),
                        update_player=update_player,
                        reach_probs=np.ones(self.n_players, dtype=np.float64),
                        depth=0,
                    )
                    self.total_traversals += 1

            self.table.n_iterations += 1

            if show_progress and (i + 1) % progress_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(
                    f"  [{i+1:>6}/{n_iterations}] "
                    f"infosets={self.table.n_infosets:,}  "
                    f"expl={self.table.exploitability_proxy():.4f}  "
                    f"{rate:.1f} iter/s",
                    flush=True,
                )

        self.total_time_s += time.time() - t0

        return {
            "n_iterations":     self.table.n_iterations,
            "n_infosets":       self.table.n_infosets,
            "exploitability":   self.table.exploitability_proxy(),
            "total_time_s":     self.total_time_s,
        }

    def save(self, path: str) -> None:
        """Save the trained strategy table."""
        self.table.save(path)
        print(f"Strategy table saved to {path!r} "
              f"({self.table.n_infosets:,} infosets, "
              f"{self.table.n_iterations:,} iterations)")

    # ------------------------------------------------------------------ #
    #  Core traversal — main turn structure                               #
    # ------------------------------------------------------------------ #

    def _traverse_turn(
        self,
        state:         CoupTreeState,
        update_player: int,
        reach_probs:   np.ndarray,    # shape (n_players,)
        depth:         int,
    ) -> float:
        """
        Traverse one complete turn: action → challenge? → block? →
        block_challenge? → lose_influence? → apply effect → next turn.

        Returns the utility for update_player from this point.
        """
        gs = state.game_state

        # Terminal check
        if gs.is_game_over or depth >= self.max_depth:
            return terminal_utility(gs, update_player)

        actor_idx = gs.current_player_idx
        actor = gs.players[actor_idx]

        if not actor.is_alive:
            gs.advance_turn()
            state.turn_number += 1
            return self._traverse_turn(state, update_player, reach_probs, depth + 1)

        # ── Step 1: Main action ───────────────────────────────────────────
        legal = get_legal_actions(gs, actor_idx)
        actor_obs = state.observation_for(actor_idx)
        infoset = InformationSet.for_main_action(actor_obs, state.history)
        action_keys = legal_main_action_keys(actor_obs, legal)

        strategy = self.table.current_strategy(infoset, action_keys)

        if actor_idx == update_player:
            # Traverse ALL branches for update player
            self.table.accumulate_strategy(
                infoset, strategy,
                weight=self._counterfactual_reach(reach_probs, update_player)
            )
            action_values: Dict[str, float] = {}
            for key, action in zip(action_keys, legal):
                child = state.copy()
                self._apply_main_action(child, action)
                new_reach = reach_probs.copy()
                new_reach[actor_idx] *= strategy[key]
                action_values[key] = self._after_action(
                    child, action, update_player, new_reach, depth + 1
                )
            node_value = sum(strategy[k] * v for k, v in action_values.items())
            cf_reach = self._counterfactual_reach(reach_probs, update_player)
            self.table.accumulate_regrets(
                infoset,
                {k: cf_reach * (v - node_value) for k, v in action_values.items()}
            )
            return node_value
        else:
            # Sample one action for other players
            key = self._sample_action(strategy, action_keys)
            action = decode_main_action(key, actor_obs, legal)
            if action is None:
                action = legal[0]
            child = state.copy()
            self._apply_main_action(child, action)
            new_reach = reach_probs.copy()
            new_reach[actor_idx] *= strategy[key]
            self.table.accumulate_strategy(
                infoset, strategy,
                weight=self._counterfactual_reach(reach_probs, actor_idx)
            )
            return self._after_action(child, action, update_player, new_reach, depth + 1)

    # ------------------------------------------------------------------ #
    #  After-action traversal (challenge → block → effect)               #
    # ------------------------------------------------------------------ #

    def _after_action(
        self,
        state:         CoupTreeState,
        action:        Action,
        update_player: int,
        reach_probs:   np.ndarray,
        depth:         int,
    ) -> float:
        """
        Handle the reaction phase after a main action is declared:
        challenges first, then blocks, then effect resolution.
        """
        gs = state.game_state

        # ── Challenge phase ───────────────────────────────────────────────
        if action.is_challengeable:
            result = self._traverse_challenge_phase(
                state, action.actor_idx, action.claimed_card,
                update_player, reach_probs, depth
            )
            if result is not None:
                return result   # Action failed or challenge succeeded — turn ends here

        if gs.is_game_over:
            return terminal_utility(gs, update_player)

        # ── Block phase ───────────────────────────────────────────────────
        if action.is_blockable and not gs.is_game_over:
            blocked = self._traverse_block_phase(
                state, action, update_player, reach_probs, depth
            )
            if blocked:
                # Action was blocked — advance turn
                gs.advance_turn()
                state.turn_number += 1
                return self._traverse_turn(state, update_player, reach_probs, depth + 1)

        if gs.is_game_over:
            return terminal_utility(gs, update_player)

        # ── Apply action effect ───────────────────────────────────────────
        self._apply_effect(state, action, update_player, reach_probs, depth)

        if gs.is_game_over:
            return terminal_utility(gs, update_player)

        gs.advance_turn()
        state.turn_number += 1
        return self._traverse_turn(state, update_player, reach_probs, depth + 1)

    # ------------------------------------------------------------------ #
    #  Challenge traversal                                                #
    # ------------------------------------------------------------------ #

    def _traverse_challenge_phase(
        self,
        state:         CoupTreeState,
        actor_idx:     int,
        claimed_card:  Optional[Card],
        update_player: int,
        reach_probs:   np.ndarray,
        depth:         int,
    ) -> Optional[float]:
        """
        Let each non-actor alive player decide whether to challenge.
        Returns a utility float if the action was resolved (failed), else None.
        """
        gs = state.game_state
        if claimed_card is None:
            return None

        for player in [p for p in gs.alive_players if p.idx != actor_idx]:
            p_obs = state.observation_for(player.idx)
            infoset = InformationSet.for_challenge_action(
                p_obs, state.history, claimed_card.value
            )
            strategy = self.table.current_strategy(infoset, legal_challenge_keys())

            if player.idx == update_player:
                self.table.accumulate_strategy(
                    infoset, strategy,
                    weight=self._counterfactual_reach(reach_probs, update_player)
                )
                challenge_val = self._resolve_challenge_branch(
                    state, actor_idx, claimed_card, player.idx,
                    update_player, reach_probs, depth
                )
                pass_val = None   # computed below if needed

                # Challenge value
                child_c = state.copy()
                actor_wins = child_c.game_state.players[actor_idx].has_card(claimed_card)
                if actor_wins:
                    loser_obs = child_c.observation_for(player.idx)
                    lose_key = self._choose_card_to_lose(
                        child_c, player.idx, update_player, reach_probs, depth
                    )
                    child_c.game_state.players[player.idx].lose_influence(
                        decode_lose_influence(lose_key, loser_obs)
                    )
                    child_c.game_state.players[actor_idx].swap_card(
                        claimed_card, child_c.game_state.deck
                    )
                    cv = self._after_challenge_continues(
                        child_c, state.pending_action, update_player, reach_probs, depth
                    )
                else:
                    lose_key = self._choose_card_to_lose(
                        child_c, actor_idx, update_player, reach_probs, depth
                    )
                    child_c.game_state.players[actor_idx].lose_influence(
                        decode_lose_influence(lose_key, child_c.observation_for(actor_idx))
                    )
                    if child_c.game_state.is_game_over:
                        cv = terminal_utility(child_c.game_state, update_player)
                    else:
                        child_c.game_state.advance_turn()
                        child_c.turn_number += 1
                        cv = self._traverse_turn(child_c, update_player, reach_probs, depth+1)

                # Pass value — continue without challenging
                child_p = state.copy()
                pv = self._after_challenge_continues(
                    child_p, None, update_player, reach_probs, depth
                )

                node_value = strategy["Challenge"] * cv + strategy["Pass"] * pv
                cf = self._counterfactual_reach(reach_probs, update_player)
                self.table.accumulate_regrets(infoset, {
                    "Challenge": cf * (cv - node_value),
                    "Pass":      cf * (pv - node_value),
                })
                # If challenger would challenge in expectation, resolve it
                if self._sample_action(strategy, ["Challenge", "Pass"]) == "Challenge":
                    return cv if actor_wins else cv
                return None   # Pass — continue

            else:
                self.table.accumulate_strategy(
                    infoset, strategy,
                    weight=self._counterfactual_reach(reach_probs, player.idx)
                )
                chosen = self._sample_action(strategy, ["Challenge", "Pass"])
                new_reach = reach_probs.copy()
                new_reach[player.idx] *= strategy[chosen]

                if chosen == "Challenge":
                    actor_wins = gs.players[actor_idx].has_card(claimed_card)
                    if actor_wins:
                        lose_key = self._choose_card_to_lose(
                            state, player.idx, update_player, new_reach, depth
                        )
                        gs.players[player.idx].lose_influence(
                            decode_lose_influence(lose_key, state.observation_for(player.idx))
                        )
                        gs.players[actor_idx].swap_card(claimed_card, gs.deck)
                        return None  # Challenge failed — action continues
                    else:
                        lose_key = self._choose_card_to_lose(
                            state, actor_idx, update_player, new_reach, depth
                        )
                        gs.players[actor_idx].lose_influence(
                            decode_lose_influence(lose_key, state.observation_for(actor_idx))
                        )
                        if gs.is_game_over:
                            return terminal_utility(gs, update_player)
                        gs.advance_turn()
                        state.turn_number += 1
                        return self._traverse_turn(state, update_player, new_reach, depth+1)

        return None  # No challenge issued

    def _after_challenge_continues(
        self, state, pending_action, update_player, reach_probs, depth
    ) -> float:
        """Helper: continue the turn after a challenge resolved in actor's favor."""
        if state.game_state.is_game_over:
            return terminal_utility(state.game_state, update_player)
        return 0.0  # Simplified: let the caller handle the continuation

    # ------------------------------------------------------------------ #
    #  Block traversal                                                    #
    # ------------------------------------------------------------------ #

    def _traverse_block_phase(
        self,
        state:         CoupTreeState,
        action:        Action,
        update_player: int,
        reach_probs:   np.ndarray,
        depth:         int,
    ) -> bool:
        """
        Let eligible players decide whether to block.
        Returns True if the action was blocked (and block survived any challenge).
        """
        gs = state.game_state

        # Who can block?
        if action.type == ActionType.FOREIGN_AID:
            potential_blockers = [p for p in gs.alive_players if p.idx != action.actor_idx]
        elif action.target_idx is not None and gs.players[action.target_idx].is_alive:
            potential_blockers = [gs.players[action.target_idx]]
        else:
            return False

        for blocker in potential_blockers:
            legal_blocks = get_legal_blocks(action, blocker.idx)
            if not legal_blocks:
                continue

            b_obs = state.observation_for(blocker.idx)
            infoset = InformationSet.for_block(
                b_obs, state.history, action.type.value
            )
            block_keys = legal_block_keys(legal_blocks)
            strategy = self.table.current_strategy(infoset, block_keys)

            if blocker.idx == update_player:
                self.table.accumulate_strategy(
                    infoset, strategy,
                    weight=self._counterfactual_reach(reach_probs, update_player)
                )

            chosen_key = self._sample_action(strategy, block_keys)
            if blocker.idx == update_player:
                new_reach = reach_probs.copy()
                new_reach[blocker.idx] *= strategy[chosen_key]
            else:
                self.table.accumulate_strategy(
                    infoset, strategy,
                    weight=self._counterfactual_reach(reach_probs, blocker.idx)
                )
                new_reach = reach_probs.copy()
                new_reach[blocker.idx] *= strategy[chosen_key]

            if chosen_key == "Pass":
                continue

            chosen_block = decode_block(chosen_key, legal_blocks)
            if chosen_block is None:
                continue

            # Block was declared — can others challenge?
            block_survived = self._traverse_block_challenge_phase(
                state, chosen_block, action.actor_idx,
                update_player, new_reach, depth
            )
            return block_survived

        return False

    def _traverse_block_challenge_phase(
        self,
        state:         CoupTreeState,
        block:         Block,
        actor_idx:     int,
        update_player: int,
        reach_probs:   np.ndarray,
        depth:         int,
    ) -> bool:
        """
        Let non-blocker alive players decide whether to challenge the block.
        Returns True if block survives (nobody challenges, or challenger loses).
        """
        gs = state.game_state

        for player in [p for p in gs.alive_players if p.idx != block.blocker_idx]:
            p_obs = state.observation_for(player.idx)
            infoset = InformationSet.for_challenge_block(
                p_obs, state.history, block.blocking_card.value
            )
            strategy = self.table.current_strategy(infoset, legal_challenge_keys())

            if player.idx == update_player:
                self.table.accumulate_strategy(
                    infoset, strategy,
                    weight=self._counterfactual_reach(reach_probs, update_player)
                )

            chosen = self._sample_action(strategy, ["Challenge", "Pass"])
            new_reach = reach_probs.copy()
            new_reach[player.idx] *= strategy[chosen]

            if chosen == "Pass":
                if player.idx == update_player:
                    self.table.accumulate_strategy(infoset, strategy,
                        weight=self._counterfactual_reach(reach_probs, update_player))
                continue

            # Block challenged
            blocker = gs.players[block.blocker_idx]
            blocker_has_card = blocker.has_card(block.blocking_card)

            if blocker_has_card:
                # Challenger loses — block stands
                lose_key = self._choose_card_to_lose(
                    state, player.idx, update_player, new_reach, depth
                )
                gs.players[player.idx].lose_influence(
                    decode_lose_influence(lose_key, state.observation_for(player.idx))
                )
                blocker.swap_card(block.blocking_card, gs.deck)
                return True   # Block survived
            else:
                # Blocker loses — block fails
                lose_key = self._choose_card_to_lose(
                    state, block.blocker_idx, update_player, new_reach, depth
                )
                gs.players[block.blocker_idx].lose_influence(
                    decode_lose_influence(lose_key, state.observation_for(block.blocker_idx))
                )
                return False  # Block failed — action proceeds

        return True  # No challenge — block stands

    # ------------------------------------------------------------------ #
    #  Influence loss traversal                                           #
    # ------------------------------------------------------------------ #

    def _choose_card_to_lose(
        self,
        state:         CoupTreeState,
        player_idx:    int,
        update_player: int,
        reach_probs:   np.ndarray,
        depth:         int,
    ) -> str:
        """
        Ask player_idx which card to reveal. Uses CFR strategy when it's
        the update player, else samples from current strategy.
        """
        gs = state.game_state
        player = gs.players[player_idx]
        if not player.hand:
            return "Lose:Duke"  # Safety fallback

        p_obs = state.observation_for(player_idx)
        infoset = InformationSet.for_lose_influence(p_obs, state.history)
        lose_keys = legal_lose_influence_keys(p_obs)
        if not lose_keys:
            return f"Lose:{player.hand[0].value}"

        strategy = self.table.current_strategy(infoset, lose_keys)

        if player_idx == update_player:
            self.table.accumulate_strategy(
                infoset, strategy,
                weight=self._counterfactual_reach(reach_probs, update_player)
            )

        return self._sample_action(strategy, lose_keys)

    # ------------------------------------------------------------------ #
    #  Effect application                                                 #
    # ------------------------------------------------------------------ #

    def _apply_main_action(self, state: CoupTreeState, action: Action) -> None:
        """Apply coin deductions at declaration time (Assassinate only)."""
        if action.type == ActionType.ASSASSINATE:
            state.game_state.players[action.actor_idx].coins -= 3
            state.assassin_paid = True
        state.pending_action = action
        state.history.append(action.type.value)

    def _apply_effect(
        self,
        state:         CoupTreeState,
        action:        Action,
        update_player: int,
        reach_probs:   np.ndarray,
        depth:         int,
    ) -> None:
        """Apply the resolved action's effect to the game state."""
        gs = state.game_state

        if action.type == ActionType.INCOME:
            gs.players[action.actor_idx].coins += 1

        elif action.type == ActionType.FOREIGN_AID:
            gs.players[action.actor_idx].coins += 2

        elif action.type == ActionType.COUP:
            gs.players[action.actor_idx].coins -= 7
            if action.target_idx is not None:
                lose_key = self._choose_card_to_lose(
                    state, action.target_idx, update_player, reach_probs, depth
                )
                gs.players[action.target_idx].lose_influence(
                    decode_lose_influence(lose_key, state.observation_for(action.target_idx))
                )

        elif action.type == ActionType.TAX:
            gs.players[action.actor_idx].coins += 3

        elif action.type == ActionType.ASSASSINATE:
            if action.target_idx is not None and gs.players[action.target_idx].is_alive:
                lose_key = self._choose_card_to_lose(
                    state, action.target_idx, update_player, reach_probs, depth
                )
                gs.players[action.target_idx].lose_influence(
                    decode_lose_influence(lose_key, state.observation_for(action.target_idx))
                )

        elif action.type == ActionType.STEAL:
            if action.target_idx is not None:
                stolen = min(2, gs.players[action.target_idx].coins)
                gs.players[action.actor_idx].coins += stolen
                gs.players[action.target_idx].coins -= stolen

        elif action.type == ActionType.EXCHANGE:
            self._apply_exchange(state, action.actor_idx, update_player, reach_probs, depth)

    def _apply_exchange(
        self,
        state:         CoupTreeState,
        actor_idx:     int,
        update_player: int,
        reach_probs:   np.ndarray,
        depth:         int,
    ) -> None:
        """Apply Ambassador Exchange: draw 2 cards, let actor choose what to keep."""
        import random as _rnd
        gs = state.game_state
        actor = gs.players[actor_idx]
        n_keep = actor.influence_count
        n_draw = min(2, len(gs.deck))
        drawn = [gs.deck.pop() for _ in range(n_draw)]
        options = actor.hand + drawn

        a_obs = state.observation_for(actor_idx)
        infoset = InformationSet.for_exchange(a_obs, state.history)
        exchange_keys = legal_exchange_keys(options, n_keep)
        if not exchange_keys:
            gs.deck.extend(drawn)
            return

        strategy = self.table.current_strategy(infoset, exchange_keys)
        if actor_idx == update_player:
            self.table.accumulate_strategy(
                infoset, strategy,
                weight=self._counterfactual_reach(reach_probs, update_player)
            )

        chosen_key = self._sample_action(strategy, exchange_keys)
        keep_indices = decode_exchange_keep(chosen_key, options, n_keep)
        kept     = [options[i] for i in keep_indices]
        returned = [c for i, c in enumerate(options) if i not in keep_indices]

        actor.hand = kept
        gs.deck.extend(returned)
        _rnd.shuffle(gs.deck)

    def _resolve_challenge_branch(
        self, state, actor_idx, claimed_card, challenger_idx,
        update_player, reach_probs, depth
    ) -> float:
        return 0.0  # handled inline in _traverse_challenge_phase

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _counterfactual_reach(
        self, reach_probs: np.ndarray, player_idx: int
    ) -> float:
        """
        Counterfactual reach probability for player_idx:
        product of all OTHER players' reach probabilities.
        """
        result = 1.0
        for i, p in enumerate(reach_probs):
            if i != player_idx:
                result *= p
        return result

    def _sample_action(
        self, strategy: Dict[str, float], actions: List[str]
    ) -> str:
        """Sample one action from a probability distribution."""
        r = self._rng.random()
        cumulative = 0.0
        for a in actions:
            cumulative += strategy.get(a, 0.0)
            if r <= cumulative:
                return a
        return actions[-1]
