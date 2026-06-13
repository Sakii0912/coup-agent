"""
live/recommender.py — Core recommendation engine for the live interface.

Takes a LiveGameState, runs it through the full Phase 3 pipeline
(BeliefTracker → features_from_observation → agent), and returns a ranked
list of recommended actions with confidence scores and human-readable reasoning.

─────────────────────────────────────────────────────────────────────────────
Recommendation pipeline
─────────────────────────────────────────────────────────────────────────────

  LiveGameState
       │
       ▼
  BeliefTracker.from_public_snapshot()   ← initialise or update
       │  probs[i, c] = P(player i holds card c)
       │  credibility[i] = how much to trust player i's claims
       ▼
  Observation (enriched with belief_state + opponent_model)
       │
       ▼
  features_from_observation()            ← 140-dim feature vector
       │
       ▼
  Agent.choose_action() / binary head    ← policy network or CFR table
       │  action_probs[k] = P(choose action k)
       ▼
  Ranked recommendations with reasoning

─────────────────────────────────────────────────────────────────────────────
Reasoning generation
─────────────────────────────────────────────────────────────────────────────

For each candidate action, reasoning is generated from:
  - The action's probability score
  - The belief state (e.g. "P2 likely holds Captain (67%)")
  - The opponent model (e.g. "P1 has bluff rate 73% — consider challenging")
  - Game-state heuristics (e.g. "you have 7+ coins — coup is dominant")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from coup.cards import Card, ActionType
from coup.state import Observation
from coup.actions import Action, Block, get_legal_actions, get_legal_blocks
from belief.belief_tracker import BeliefTracker
from belief.belief_state import CARDS, CARD_TO_IDX
from live.state_input import LiveGameState, TurnEvent


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """
    One recommended action with its confidence score and reasoning.
    """
    action:      Action
    score:       float             # probability / confidence (0–1)
    rank:        int               # 1 = best
    reasoning:   List[str]         # bullet-point reasoning lines
    risk_level:  str               # "low" | "medium" | "high"

    @property
    def action_label(self) -> str:
        """Human-readable action label."""
        a = self.action
        label = a.type.value
        if a.target_idx is not None:
            label += f" → Player {a.target_idx}"
        if a.claimed_card:
            label += f" (claim {a.claimed_card.value})"
        return label

    def display(self) -> str:
        risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
            self.risk_level, "⚪"
        )
        lines = [
            f"  #{self.rank}  {self.action_label:<36}  "
            f"score={self.score:.1%}  {risk_icon} {self.risk_level}"
        ]
        for r in self.reasoning:
            lines.append(f"       • {r}")
        return "\n".join(lines)


@dataclass
class ReactionRecommendation:
    """
    Recommendation for a reaction decision (challenge / block / challenge block).
    """
    decision:    str        # "challenge" | "block" | "pass" | "challenge_block"
    block_card:  Optional[Card] = None   # which card to claim when blocking
    score:       float = 0.5
    reasoning:   List[str] = field(default_factory=list)

    def display(self) -> str:
        label = self.decision.upper()
        if self.block_card:
            label += f" with {self.block_card.value}"
        lines = [f"  Recommendation: {label}  (confidence={self.score:.1%})"]
        for r in self.reasoning:
            lines.append(f"    • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class Recommender:
    """
    Recommendation engine for live Coup play.

    Args:
        agent:          A trained Agent (NeuralAgent, CFRAgent, or None for
                        heuristic-only mode).
        config:         FeatureConfig. Defaults to Phase 3 (140-dim with belief).
        top_k:          Number of recommendations to return. Default 3.
    """

    def __init__(
        self,
        agent=None,
        config=None,
        top_k: int = 3,
    ) -> None:
        self.agent  = agent
        self.top_k  = top_k

        if config is None:
            from pipeline.feature_extractor import FeatureConfig
            # Use belief if available, fall back to base
            try:
                config = FeatureConfig(include_belief=True, include_opp_model=True)
            except Exception:
                config = FeatureConfig()
        self.config = config

        # Session state: tracker is built once at game start, updated each turn
        self._tracker:         Optional[BeliefTracker] = None
        self._action_history:  List[str] = []
        self._n_players:       int = 4

    # ------------------------------------------------------------------ #
    #  Session management                                                 #
    # ------------------------------------------------------------------ #

    def start_session(self, state: LiveGameState) -> None:
        """
        Initialise the BeliefTracker for a new game.
        Call once at the beginning of the game.
        """
        self._n_players     = state.n_players
        self._action_history = []

        snapshot = state.all_player_public_views()
        self._tracker = BeliefTracker.from_public_snapshot(
            snapshot     = snapshot,
            observer_idx = state.my_state.idx,
            known_hand   = list(state.my_state.hand),
        )

    def feed_event(self, event: TurnEvent) -> None:
        """
        Update the BeliefTracker with a completed turn's events.
        Call after each turn resolves to keep beliefs current.
        """
        if self._tracker is None:
            return

        # Action claim
        self._tracker.process_raw(
            "action",
            actor_idx=event.actor_idx,
            claimed_card=event.claimed_card,
        )
        self._action_history.append(event.action_type.value)

        # Challenge result
        if event.challenged and event.challenger_idx is not None:
            self._tracker.process_raw(
                "challenge_result",
                actor_idx=event.actor_idx,
                claimed_card=event.claimed_card,
                challenger_idx=event.challenger_idx,
                actor_won=event.actor_won_challenge,
            )

        # Block claim
        if event.blocked and event.blocker_idx is not None:
            self._tracker.process_raw(
                "block",
                blocker_idx=event.blocker_idx,
                blocking_card=event.blocking_card,
            )
            if event.block_challenged and event.blocker_won is not None:
                self._tracker.process_raw(
                    "challenge_result",
                    actor_idx=event.blocker_idx,
                    claimed_card=event.blocking_card,
                    challenger_idx=event.actor_idx,
                    actor_won=event.blocker_won,
                )

        # Influence losses
        for player_idx, card_lost in event.influence_losses:
            self._tracker.process_raw(
                "influence_loss",
                player_idx=player_idx,
                card_lost=card_lost,
            )

        # Exchange
        if event.action_resolved and event.action_type == ActionType.EXCHANGE:
            self._tracker.process_raw("exchange", actor_idx=event.actor_idx)

    # ------------------------------------------------------------------ #
    #  Main recommendation API                                           #
    # ------------------------------------------------------------------ #

    def recommend_action(self, state: LiveGameState) -> List[Recommendation]:
        """
        Given the current LiveGameState (on the human's turn), return a
        ranked list of recommended actions.
        """
        if self._tracker is None:
            self.start_session(state)

        obs = self._build_observation(state)
        if self._tracker:
            self._tracker.enrich_observation(obs)

        # Get legal actions
        from coup.state import GameState, PlayerState
        legal = self._get_legal_actions(state)
        if not legal:
            return []

        # Score each action
        scores = self._score_actions(obs, legal, state)

        # Build ranked recommendations
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        total  = sum(s for _, s in ranked)
        recs   = []
        for rank, (action, score) in enumerate(ranked[:self.top_k], start=1):
            norm_score = score / max(total, 1e-9)
            reasoning  = self._build_action_reasoning(action, state, norm_score)
            risk       = self._assess_risk(action, state)
            recs.append(Recommendation(
                action=action,
                score=norm_score,
                rank=rank,
                reasoning=reasoning,
                risk_level=risk,
            ))
        return recs

    def recommend_reaction(
        self,
        state:        LiveGameState,
        decision:     str,    # "challenge" | "block" | "challenge_block"
    ) -> ReactionRecommendation:
        """
        Recommend whether to challenge, block, or pass in reaction to
        another player's action.
        """
        if self._tracker is None:
            self.start_session(state)

        obs = self._build_observation(state)
        if self._tracker:
            self._tracker.enrich_observation(obs)

        if decision == "challenge":
            return self._recommend_challenge(state, obs)
        elif decision == "block":
            return self._recommend_block(state, obs)
        elif decision == "challenge_block":
            return self._recommend_challenge_block(state, obs)

        return ReactionRecommendation(decision="pass", score=0.5,
                                      reasoning=["No recommendation available."])

    def recommend_card_to_lose(self, state: LiveGameState) -> Card:
        """Recommend which card to reveal when losing an influence."""
        hand = state.my_state.hand
        if not hand:
            return None
        if len(hand) == 1:
            return hand[0]
        # Keep higher-value card
        from agents.neural.neural_agent import CARD_VALUE
        return min(hand, key=lambda c: CARD_VALUE.get(c, 1))

    # ------------------------------------------------------------------ #
    #  Belief state summary                                               #
    # ------------------------------------------------------------------ #

    def belief_summary(self, state: LiveGameState) -> str:
        """Return a formatted summary of current beliefs for display."""
        if self._tracker is None:
            return "  No belief state (session not started)"

        lines = ["  Opponent Belief State:"]
        for opp in state.opponents:
            if not opp.is_alive:
                continue
            lines.append(f"    {opp.name} (P{opp.idx}) — {opp.influence_count} card(s):")
            probs = self._tracker.belief.probs[opp.idx]
            top   = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
            for c_idx, p in top:
                if p > 0.05:
                    bar = "█" * int(p * 20)
                    lines.append(
                        f"      {CARDS[c_idx].value:<12} {p:5.1%}  {bar}"
                    )
            cred = self._tracker.credibility(opp.idx)
            bluff = self._tracker.model.bluff_rate(opp.idx)
            lines.append(
                f"      credibility={cred:.2f}  bluff_rate={bluff:.1%}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_observation(self, state: LiveGameState) -> Observation:
        """Convert LiveGameState into an Observation."""
        others = [o.to_public_dict() for o in state.opponents]
        return Observation(
            my_idx=state.my_state.idx,
            my_hand=list(state.my_state.hand),
            my_coins=state.my_state.coins,
            my_revealed=list(state.my_state.revealed),
            others=others,
            deck_size=max(0, 15 - len(state.my_state.hand)
                          - len(state.my_state.revealed)
                          - sum(o.influence_count + len(o.revealed_cards)
                                for o in state.opponents)),
            current_player_idx=state.current_turn,
            turn_number=state.turn_number,
        )

    def _get_legal_actions(self, state: LiveGameState) -> List[Action]:
        """
        Derive legal actions from the current game state without running
        the full game engine (which would need a proper GameState object).
        """
        from coup.actions import Action, ActionType, TARGETED_ACTIONS
        from coup.cards import BLOCK_MAP, ACTION_CLAIMS

        me        = state.my_state
        my_coins  = me.coins
        alive_opps = [o for o in state.opponents if o.is_alive]
        legal     = []

        # Must coup at 10+
        if my_coins >= 10:
            for opp in alive_opps:
                legal.append(Action(
                    type=ActionType.COUP,
                    actor_idx=me.idx,
                    target_idx=opp.idx,
                ))
            return legal

        # General actions
        legal.append(Action(type=ActionType.INCOME, actor_idx=me.idx))
        legal.append(Action(type=ActionType.FOREIGN_AID, actor_idx=me.idx))

        if my_coins >= 7:
            for opp in alive_opps:
                legal.append(Action(
                    type=ActionType.COUP, actor_idx=me.idx, target_idx=opp.idx
                ))

        legal.append(Action(type=ActionType.TAX, actor_idx=me.idx))

        if my_coins >= 3:
            for opp in alive_opps:
                legal.append(Action(
                    type=ActionType.ASSASSINATE, actor_idx=me.idx, target_idx=opp.idx
                ))

        for opp in alive_opps:
            if opp.coins > 0:
                legal.append(Action(
                    type=ActionType.STEAL, actor_idx=me.idx, target_idx=opp.idx
                ))

        legal.append(Action(type=ActionType.EXCHANGE, actor_idx=me.idx))
        return legal

    def _score_actions(
        self,
        obs:     Observation,
        legal:   List[Action],
        state:   LiveGameState,
    ) -> Dict[Action, float]:
        """Score each legal action. Uses agent if available, else heuristics."""
        scores: Dict[Action, float] = {}

        if self.agent is not None:
            try:
                return self._score_with_agent(obs, legal, state)
            except Exception:
                pass

        # Heuristic scoring fallback
        return self._heuristic_scores(legal, state)

    def _score_with_agent(
        self, obs: Observation, legal: List[Action], state: LiveGameState
    ) -> Dict[Action, float]:
        """Use the policy agent to score actions."""
        import torch
        from pipeline.feature_extractor import features_from_observation, ACTION_TO_IDX

        feat = features_from_observation(
            obs=obs,
            action_type="Income",
            target_idx=None,
            claimed_card=None,
            action_history=self._action_history,
            config=self.config,
        )
        # Apply normalisation if agent has it
        if hasattr(self.agent, 'feat_mean') and self.agent.feat_mean is not None:
            feat = (feat - self.agent.feat_mean) / (self.agent.feat_std + 1e-8)

        x = torch.from_numpy(feat).unsqueeze(0)
        act_probs = self.agent.model.action_probs(x).squeeze(0).numpy()
        tgt_probs = self.agent.model.target_probs(x).squeeze(0).numpy()

        scores: Dict[Action, float] = {}
        for action in legal:
            a_p   = float(act_probs[ACTION_TO_IDX.get(action.type.value, 0)])
            if action.target_idx is not None:
                n_total = len(obs.others) + 1
                slot    = min((action.target_idx - obs.my_idx - 1) % n_total, 5)
                t_p     = float(tgt_probs[slot])
            else:
                t_p = float(tgt_probs[5])
            scores[action] = max(a_p * t_p, 1e-9)
        return scores

    def _heuristic_scores(
        self, legal: List[Action], state: LiveGameState
    ) -> Dict[Action, float]:
        """
        Rule-based heuristic scoring when no agent is available.
        Encodes basic Coup strategy:
          - Prefer Coup when >= 7 coins
          - Prefer Tax if we hold Duke
          - Prefer Assassinate if we hold Assassin and can afford it
          - Steal weakest-coin opponent (disrupts their income)
          - Income / Foreign Aid as fallback
        """
        my_hand  = set(state.my_state.hand)
        my_coins = state.my_state.coins
        scores: Dict[Action, float] = {}

        for action in legal:
            score = 1.0   # base
            at    = action.type

            if at == ActionType.COUP:
                score = 4.0 if my_coins >= 7 else 0.1
            elif at == ActionType.ASSASSINATE:
                score = 3.5 if Card.ASSASSIN in my_hand else 1.5
            elif at == ActionType.TAX:
                score = 3.0 if Card.DUKE in my_hand else 1.8
            elif at == ActionType.STEAL:
                # Prefer stealing from the richest opponent
                if action.target_idx is not None:
                    opp = next(
                        (o for o in state.opponents if o.idx == action.target_idx),
                        None,
                    )
                    score = 2.0 + (opp.coins * 0.3 if opp else 0)
                    if Card.CAPTAIN in my_hand:
                        score += 1.0
            elif at == ActionType.EXCHANGE:
                score = 1.5 if Card.AMBASSADOR in my_hand else 0.8
            elif at == ActionType.FOREIGN_AID:
                score = 1.4
            elif at == ActionType.INCOME:
                score = 1.0

            # Penalise bluffing risky cards if we don't hold them
            if at in (ActionType.TAX, ActionType.ASSASSINATE, ActionType.STEAL,
                      ActionType.EXCHANGE):
                from coup.cards import ACTION_CLAIMS
                needed = ACTION_CLAIMS.get(at)
                if needed and needed not in my_hand:
                    score *= 0.7   # slight penalty for bluffing

            scores[action] = score

        return scores

    def _recommend_challenge(
        self, state: LiveGameState, obs: Observation
    ) -> ReactionRecommendation:
        """Should we challenge the pending action claim?"""
        if state.pending_claimed is None:
            return ReactionRecommendation(
                decision="pass", score=0.8,
                reasoning=["No card claimed — cannot challenge."],
            )

        reasoning = []
        score     = 0.3   # default lean toward passing

        claimed   = state.pending_claimed
        actor_idx = state.pending_actor

        # How many copies of the claimed card are visible?
        visible_count = sum(
            1 for c in state.my_state.hand if c == claimed
        ) + sum(
            1 for opp in state.opponents
            for c in opp.revealed_cards if c == claimed
        )
        remaining = 3 - visible_count
        reasoning.append(
            f"{claimed.value}: {visible_count} accounted for, "
            f"{remaining} still possible in hidden hands."
        )

        if visible_count >= 3:
            score = 0.95
            reasoning.append(
                f"All 3 {claimed.value}s are accounted for — they MUST be bluffing!"
            )
        elif visible_count == 2:
            score = 0.75
            reasoning.append(
                f"Only 1 {claimed.value} remains unaccounted — "
                f"high chance of bluff."
            )
        elif claimed in state.my_state.hand:
            copies_i_hold = sum(1 for c in state.my_state.hand if c == claimed)
            score = 0.5 + 0.15 * copies_i_hold
            reasoning.append(
                f"You hold {copies_i_hold} {claimed.value}(s) — "
                f"reduces chance they have it."
            )

        # Opponent model
        if self._tracker and actor_idx is not None:
            bluff_rate = self._tracker.model.bluff_rate(actor_idx)
            cred       = self._tracker.credibility(actor_idx)
            reasoning.append(
                f"P{actor_idx} bluff rate: {bluff_rate:.0%}  "
                f"(credibility: {cred:.2f})"
            )
            score += (bluff_rate - 0.5) * 0.3   # high bluffer → challenge more

        score = float(np.clip(score, 0.05, 0.95))
        decision = "challenge" if score > 0.5 else "pass"
        return ReactionRecommendation(
            decision=decision, score=score, reasoning=reasoning
        )

    def _recommend_block(
        self, state: LiveGameState, obs: Observation
    ) -> ReactionRecommendation:
        """Should we block the pending action?"""
        from coup.cards import BLOCK_MAP

        if state.pending_action is None:
            return ReactionRecommendation(decision="pass", score=0.5)

        blocking_cards = BLOCK_MAP.get(state.pending_action, [])
        if not blocking_cards:
            return ReactionRecommendation(
                decision="pass", score=0.9,
                reasoning=["This action cannot be blocked."],
            )

        reasoning = []
        # Do we hold a blocking card?
        honest_blocks = [c for c in blocking_cards if c in state.my_state.hand]
        if honest_blocks:
            card    = honest_blocks[0]
            score   = 0.80
            decision = "block"
            reasoning.append(
                f"You hold {card.value} — can block honestly (no challenge risk)."
            )
        else:
            # Bluff block — risky
            card    = blocking_cards[0]
            score   = 0.35
            decision = "pass"
            reasoning.append(
                f"You don't hold {card.value} — blocking requires a bluff."
            )
            reasoning.append(
                "Bluff blocking is risky if anyone challenges."
            )

        return ReactionRecommendation(
            decision=decision,
            block_card=card if decision == "block" else None,
            score=score,
            reasoning=reasoning,
        )

    def _recommend_challenge_block(
        self, state: LiveGameState, obs: Observation
    ) -> ReactionRecommendation:
        """Should we challenge the blocker's card claim?"""
        if state.pending_claimed is None:
            return ReactionRecommendation(decision="pass", score=0.7)

        # Reuse challenge logic — same reasoning applies
        result = self._recommend_challenge(state, obs)
        result.decision = "challenge_block" if result.decision == "challenge" else "pass"
        return result

    def _build_action_reasoning(
        self, action: Action, state: LiveGameState, score: float
    ) -> List[str]:
        """Generate bullet-point reasoning for a recommended action."""
        reasons = []
        at      = action.type
        my_hand = state.my_state.hand

        # Card honesty
        from coup.cards import ACTION_CLAIMS
        needed = ACTION_CLAIMS.get(at)
        if needed:
            if needed in my_hand:
                reasons.append(f"You hold {needed.value} — no bluff risk.")
            else:
                reasons.append(
                    f"Requires claiming {needed.value} (you don't hold it — bluff)."
                )

        # Target reasoning
        if action.target_idx is not None:
            target = next(
                (o for o in state.opponents if o.idx == action.target_idx), None
            )
            if target and self._tracker:
                if at == ActionType.STEAL:
                    reasons.append(
                        f"P{target.idx} ({target.name}) has {target.coins} coins — "
                        f"steal {min(2, target.coins)}."
                    )
                    cap_p = self._tracker.prob(target.idx, Card.CAPTAIN)
                    amb_p = self._tracker.prob(target.idx, Card.AMBASSADOR)
                    block_p = max(cap_p, amb_p)
                    reasons.append(
                        f"Block risk: {block_p:.0%} chance they can block "
                        f"(Captain {cap_p:.0%} / Ambassador {amb_p:.0%})."
                    )
                elif at == ActionType.ASSASSINATE:
                    cont_p = self._tracker.prob(target.idx, Card.CONTESSA)
                    reasons.append(
                        f"Block risk: {cont_p:.0%} chance they hold Contessa."
                    )
                elif at == ActionType.COUP:
                    reasons.append(
                        f"Coup cannot be blocked or challenged — guaranteed influence loss."
                    )

        # Coin context
        if at == ActionType.COUP:
            reasons.append(f"You have {state.my_state.coins} coins — Coup is available.")
        if at in (ActionType.INCOME, ActionType.FOREIGN_AID):
            reasons.append(
                f"Builds toward Coup threshold "
                f"({state.my_state.coins} → "
                f"{state.my_state.coins + (1 if at == ActionType.INCOME else 2)} coins)."
            )

        return reasons

    def _assess_risk(self, action: Action, state: LiveGameState) -> str:
        """Classify action risk as low/medium/high."""
        from coup.cards import ACTION_CLAIMS
        at     = action.type
        needed = ACTION_CLAIMS.get(at)

        if at == ActionType.COUP:
            return "low"
        if at == ActionType.INCOME:
            return "low"
        if needed and needed in state.my_state.hand:
            return "low"
        if at in (ActionType.TAX, ActionType.EXCHANGE):
            return "medium"
        if at in (ActionType.STEAL, ActionType.ASSASSINATE):
            return "medium" if needed in state.my_state.hand else "high"
        if at == ActionType.FOREIGN_AID:
            return "low"
        return "medium"
