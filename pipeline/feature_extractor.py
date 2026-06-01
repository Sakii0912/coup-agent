"""
pipeline/feature_extractor.py — Convert game log events into feature vectors.

Each observable decision point in a game log is converted to a flat numpy
vector that captures everything the acting player could see at that moment.

Feature vector layout (Phase 2 base: 104 dims; Phase 3 full: 134 dims)
─────────────────────────────────────────────────────────────────────────────
  Section                  Dims    Description
─────────────────────────────────────────────────────────────────────────────
  My state                 12      Revealed cards (×5), coins/12, influence/2,
                                   revealed proxy
  Opponents (×5 slots)     40      Per slot: influence/2, coins/12, revealed
                                   cards (×5), is_alive; zero-padded if absent
  Action type               7      One-hot (Income … Exchange)
  Claimed card              6      One-hot + "no claim" dim
  Meta scalars              4      target_is_me, actor_is_me, turn/500, table_coins/72
  Action history           35      One-hot counts of last 5 action types
  ── Phase 3 (optional) ──────────────────────────────────────────────────
  Belief state probs       30      Flattened (n_players × N_CARDS) prob matrix
                                   (include_belief=True only)
  Opponent credibilities    6      Per-player credibility score from OpponentModel
                                   (include_opp_model=True only)
─────────────────────────────────────────────────────────────────────────────

Phase 2 (default):  FeatureConfig()                       → 104 dims
Phase 3 (belief):   FeatureConfig(include_belief=True,
                                  include_opp_model=True)  → 134 dims

Label vector layout:
  Action label  7   Index into ActionType for the chosen action
  Target label  6   Index into player slot (5 opponents + "no target")
  Outcome       1   1 = action resolved, 0 = blocked/failed
─────────────────────────────────────────────────────────────────────────────

Usage:
    # From game logs (Phase 2 path)
    from pipeline.feature_extractor import extract_features, FeatureConfig
    config = FeatureConfig(include_belief=False)
    samples = extract_features(parsed_game, config)

    # From a live Observation (Phase 3 / Phase 5 path)
    from pipeline.feature_extractor import features_from_observation
    feat = features_from_observation(obs, config)   # shape (134,) with belief
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .log_parser import ParsedGame, ParsedEvent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CARDS = ["Duke", "Assassin", "Contessa", "Captain", "Ambassador"]
CARD_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CARDS)}
N_CARDS = len(CARDS)                    # 5

ACTIONS = ["Income", "ForeignAid", "Coup", "Tax", "Assassinate", "Steal", "Exchange"]
ACTION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)               # 7

MAX_COINS_NORM = 12.0                  # normalisation denominator for coins
MAX_TURNS_NORM = 500.0                 # normalisation denominator for turn number
MAX_TABLE_COINS = 72.0                 # 6 players × 12 coins max
HISTORY_LENGTH = 5                     # number of past actions to encode


# ---------------------------------------------------------------------------
# Feature config
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """
    Controls dimensionality of the feature vector.

    max_players:       Maximum number of opponent slots to encode.
                       Set to 6 (default) for a fixed-size vector that works
                       for any game size.
    include_belief:    If True, append belief state probs (n_players × N_CARDS
                       = 30 dims for max_players=6) to the feature vector.
    include_opp_model: If True, append opponent credibility scores (max_players
                       dims, one per player) to the feature vector.

    Phase 2 code (include_belief=False, include_opp_model=False) produces
    the original 104-dim vector — fully backward compatible.
    Phase 3 code (include_belief=True, include_opp_model=True) produces a
    134-dim vector.
    """
    max_players:       int  = 6
    include_belief:    bool = False   # Phase 3: off by default for backward compat
    include_opp_model: bool = False   # Phase 3: off by default for backward compat

    @property
    def n_opponent_slots(self) -> int:
        return self.max_players - 1     # 5 for max_players=6

    @property
    def belief_dims(self) -> int:
        """Dims added by the belief state: one prob per (player, card) pair."""
        return self.max_players * N_CARDS if self.include_belief else 0

    @property
    def opp_model_dims(self) -> int:
        """Dims added by the opponent model: one credibility score per player."""
        return self.max_players if self.include_opp_model else 0

    @property
    def feature_dim(self) -> int:
        """Total feature vector length."""
        my_dims      = N_CARDS + 1 + 1 + N_CARDS           # hand + coins + influence + revealed
        opp_dims     = self.n_opponent_slots * (1 + 1 + N_CARDS + 1)
        action_dims  = N_ACTIONS
        claimed_dims = N_CARDS + 1
        meta_dims    = 4                                    # target_is_me, actor_is_me, turn, table_coins
        history_dims = N_ACTIONS * HISTORY_LENGTH
        base = my_dims + opp_dims + action_dims + claimed_dims + meta_dims + history_dims
        return base + self.belief_dims + self.opp_model_dims


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """
    One labelled training example, corresponding to a single action event
    in a game log.

    features:      Float32 numpy array, shape (feature_dim,)
    action_label:  Int index into ACTIONS for the chosen action
    target_label:  Int index into opponent slots (0–4), or 5 if no target
    outcome:       1 if the action resolved successfully, 0 if it was blocked/failed
    game_id:       Source game identifier (for debugging)
    turn:          Turn number this action occurred on
    player_idx:    Which player took this action
    """
    features: np.ndarray
    action_label: int
    target_label: int
    outcome: int
    game_id: int
    turn: int
    player_idx: int


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_features(
    game: ParsedGame,
    config: Optional[FeatureConfig] = None,
) -> List[Sample]:
    """
    Extract all labelled training samples from a single ParsedGame.

    One sample is produced per action event. The feature vector captures
    the publicly observable state at the moment the action is declared.

    Args:
        game:    A validated ParsedGame object.
        config:  FeatureConfig controlling vector size. Defaults to max_players=6.

    Returns:
        List of Sample objects (one per action event in the game).
    """
    if config is None:
        config = FeatureConfig()

    samples: List[Sample] = []
    events = game.events

    # Build a quick index of which events follow each action event
    action_outcomes = _build_outcome_map(events)

    # Maintain a rolling action history window
    action_history: List[str] = []

    # Maintain a live state snapshot for context (updated on action_resolved)
    live_state: Optional[List[Dict[str, Any]]] = None

    # First event is game_start — grab initial state
    if events and events[0].type == "game_start" and events[0].state:
        live_state = events[0].state

    for evt_idx, evt in enumerate(events):
        # Update live_state on any event that carries a state snapshot
        if evt.state is not None:
            live_state = evt.state

        if evt.type != "action":
            continue
        if live_state is None:
            continue
        if evt.actor_idx is None or evt.action_type is None:
            continue

        actor_idx   = evt.actor_idx
        action_type = evt.action_type
        target_idx  = evt.target_idx

        # Build the state the actor saw when they chose this action.
        # We use the snapshot from the most recent preceding state-bearing event.
        feat = _build_feature_vector(
            actor_idx=actor_idx,
            action_type=action_type,
            target_idx=target_idx,
            claimed_card=evt.claimed_card,
            state=live_state,
            n_players=game.n_players,
            turn=evt.turn,
            action_history=action_history,
            config=config,
        )

        # Labels
        action_label = ACTION_TO_IDX.get(action_type, 0)
        target_label = _encode_target(actor_idx, target_idx, game.n_players, config)
        outcome      = action_outcomes.get(evt_idx, 0)

        samples.append(Sample(
            features=feat,
            action_label=action_label,
            target_label=target_label,
            outcome=outcome,
            game_id=game.game_id,
            turn=evt.turn,
            player_idx=actor_idx,
        ))

        # Roll the history forward
        action_history.append(action_type)

    return samples


def _build_outcome_map(events: List[ParsedEvent]) -> Dict[int, int]:
    """
    For each action event index, determine if the action was successful.
    1 = action_resolved followed the action (possibly after challenge/block).
    0 = no action_resolved followed before the next action (blocked/failed).
    """
    outcome: Dict[int, int] = {}
    n = len(events)
    for i, evt in enumerate(events):
        if evt.type != "action":
            continue
        for j in range(i + 1, n):
            et = events[j].type
            if et == "action_resolved":
                outcome[i] = 1
                break
            if et == "action":
                outcome[i] = 0
                break
        else:
            outcome[i] = 0
    return outcome


def _encode_target(
    actor_idx: int,
    target_idx: Optional[int],
    n_players: int,
    config: FeatureConfig,
) -> int:
    """
    Encode target as an index into opponent slots (0 … n_opponent_slots-1),
    or `n_opponent_slots` if there is no target.

    Opponent slots are numbered relative to the actor:
      slot 0 = next player clockwise, slot 1 = next after that, etc.
    """
    if target_idx is None:
        return config.n_opponent_slots   # "no target" class
    # Convert absolute player index to relative opponent slot
    slot = (target_idx - actor_idx - 1) % n_players
    return min(slot, config.n_opponent_slots - 1)


def _build_feature_vector(
    actor_idx: int,
    action_type: str,
    target_idx: Optional[int],
    claimed_card: Optional[str],
    state: List[Dict[str, Any]],
    n_players: int,
    turn: int,
    action_history: List[str],
    config: FeatureConfig,
    belief_probs: Optional[np.ndarray] = None,      # shape (n_players, N_CARDS) or None
    opp_credibilities: Optional[np.ndarray] = None, # shape (n_players,) or None
) -> np.ndarray:
    """Assemble the full feature vector as a float32 numpy array."""
    parts: List[np.ndarray] = []

    # ── My features ──────────────────────────────────────────────────────
    me = state[actor_idx] if actor_idx < len(state) else {}

    # My hand: we can only encode revealed cards from the state snapshot
    # (hidden hand is unknown to opponents — we'll encode own hand in Phase 3
    # when the BeliefState adds hand inference. For now, encode revealed only.)
    my_revealed = _card_counts(me.get("revealed_cards", []))
    my_coins    = np.array([me.get("coins", 0) / MAX_COINS_NORM], dtype=np.float32)
    my_influence = np.array([me.get("influence_count", 0) / 2.0], dtype=np.float32)

    parts += [my_revealed, my_coins, my_influence, my_revealed]  # revealed twice (hand proxy)

    # ── Opponent features ─────────────────────────────────────────────────
    slot = 0
    for rel in range(1, n_players):
        opp_idx = (actor_idx + rel) % n_players
        if opp_idx < len(state):
            opp = state[opp_idx]
            opp_influence = np.array([opp.get("influence_count", 0) / 2.0], dtype=np.float32)
            opp_coins     = np.array([opp.get("coins", 0) / MAX_COINS_NORM], dtype=np.float32)
            opp_revealed  = _card_counts(opp.get("revealed_cards", []))
            opp_alive     = np.array([float(opp.get("is_alive", False))], dtype=np.float32)
            parts += [opp_influence, opp_coins, opp_revealed, opp_alive]
        else:
            # Pad absent opponent slot with zeros
            parts.append(np.zeros(1 + 1 + N_CARDS + 1, dtype=np.float32))
        slot += 1
        if slot >= config.n_opponent_slots:
            break

    # Pad remaining opponent slots
    while slot < config.n_opponent_slots:
        parts.append(np.zeros(1 + 1 + N_CARDS + 1, dtype=np.float32))
        slot += 1

    # ── Action one-hot ────────────────────────────────────────────────────
    action_vec = np.zeros(N_ACTIONS, dtype=np.float32)
    if action_type in ACTION_TO_IDX:
        action_vec[ACTION_TO_IDX[action_type]] = 1.0
    parts.append(action_vec)

    # ── Claimed card one-hot (+ "no claim" dim) ───────────────────────────
    claimed_vec = np.zeros(N_CARDS + 1, dtype=np.float32)
    if claimed_card is not None and claimed_card in CARD_TO_IDX:
        claimed_vec[CARD_TO_IDX[claimed_card]] = 1.0
    else:
        claimed_vec[N_CARDS] = 1.0   # "no claim"
    parts.append(claimed_vec)

    # ── Meta scalars ──────────────────────────────────────────────────────
    target_is_me = float(target_idx == actor_idx)     # always 0 for main actions
    actor_is_me  = 1.0                                 # always 1 (we encode actor's perspective)
    turn_norm    = min(turn / MAX_TURNS_NORM, 1.0)
    table_coins  = sum(p.get("coins", 0) for p in state) / MAX_TABLE_COINS
    parts.append(np.array([target_is_me, actor_is_me, turn_norm, table_coins], dtype=np.float32))

    # ── Action history (last HISTORY_LENGTH actions) ──────────────────────
    history_vec = np.zeros(N_ACTIONS * HISTORY_LENGTH, dtype=np.float32)
    recent = action_history[-HISTORY_LENGTH:]
    for slot_h, past_action in enumerate(recent):
        if past_action in ACTION_TO_IDX:
            history_vec[slot_h * N_ACTIONS + ACTION_TO_IDX[past_action]] = 1.0
    parts.append(history_vec)

    # ── Belief state (Phase 3) ────────────────────────────────────────────
    if config.include_belief:
        belief_vec = np.zeros(config.max_players * N_CARDS, dtype=np.float32)
        if belief_probs is not None:
            # Flatten (n_players, N_CARDS) → ordered by player then card
            flat = belief_probs.flatten().astype(np.float32)
            n = min(len(flat), len(belief_vec))
            belief_vec[:n] = flat[:n]
        parts.append(belief_vec)

    # ── Opponent model credibilities (Phase 3) ────────────────────────────
    if config.include_opp_model:
        cred_vec = np.ones(config.max_players, dtype=np.float32)
        if opp_credibilities is not None:
            n = min(len(opp_credibilities), config.max_players)
            cred_vec[:n] = opp_credibilities[:n].astype(np.float32)
        parts.append(cred_vec)

    return np.concatenate(parts).astype(np.float32)


def _card_counts(cards: List[str]) -> np.ndarray:
    """Return a float32 vector of shape (N_CARDS,) counting occurrences."""
    vec = np.zeros(N_CARDS, dtype=np.float32)
    for c in cards:
        if c in CARD_TO_IDX:
            vec[CARD_TO_IDX[c]] += 1.0
    return vec


# ---------------------------------------------------------------------------
# Live extraction from Observation (Phase 3 / Phase 5 path)
# ---------------------------------------------------------------------------

def features_from_observation(
    obs,                              # coup.state.Observation
    action_type: str,
    target_idx: Optional[int],
    claimed_card: Optional[str],
    action_history: List[str],
    config: Optional[FeatureConfig] = None,
) -> np.ndarray:
    """
    Extract a feature vector directly from a live Observation object.

    This is the entry point used by the Phase 5 live interface and by agents
    during self-play (Phase 4). It builds the same feature vector as
    `_build_feature_vector` but reads its inputs from an `Observation` rather
    than from log event dicts.

    If the Observation contains a `belief_state` and/or `opponent_model` and
    `config.include_belief` / `config.include_opp_model` are True, those
    sections are populated from the live belief tracker.

    Args:
        obs:            Observation object (from coup.state).
        action_type:    String name of the action being taken.
        target_idx:     Absolute player index of the target, or None.
        claimed_card:   String card name claimed, or None.
        action_history: List of action type strings seen so far this game.
        config:         FeatureConfig. Defaults to FeatureConfig() (Phase 2 dims).

    Returns:
        Float32 numpy array of shape (config.feature_dim,).
    """
    if config is None:
        config = FeatureConfig()

    # Build a state snapshot from the Observation
    # The observer's own slot uses known hand data; others use public view.
    state: List[Dict[str, Any]] = []
    n_players = len(obs.others) + 1

    # Reconstruct full player list in absolute index order
    others_by_idx = {p["idx"]: p for p in obs.others}
    for p_idx in range(n_players):
        if p_idx == obs.my_idx:
            state.append({
                "name": f"P{p_idx}",
                "idx": p_idx,
                "coins": obs.my_coins,
                "influence_count": len(obs.my_hand),
                "revealed_cards": [c.value for c in obs.my_revealed],
                "is_alive": len(obs.my_hand) > 0,
            })
        elif p_idx in others_by_idx:
            state.append(others_by_idx[p_idx])
        else:
            # Absent slot (shouldn't happen in valid games)
            state.append({
                "name": f"P{p_idx}", "idx": p_idx, "coins": 0,
                "influence_count": 0, "revealed_cards": [], "is_alive": False,
            })

    # Extract belief probs and credibilities if available
    belief_probs     = None
    opp_credibilities = None

    if config.include_belief and obs.belief_state is not None:
        belief_probs = obs.belief_state.probs   # (n_players, N_CARDS)

    if config.include_opp_model and obs.opponent_model is not None:
        opp_credibilities = obs.opponent_model.all_credibilities()  # (n_players,)

    return _build_feature_vector(
        actor_idx=obs.my_idx,
        action_type=action_type,
        target_idx=target_idx,
        claimed_card=claimed_card,
        state=state,
        n_players=n_players,
        turn=obs.turn_number,
        action_history=action_history,
        config=config,
        belief_probs=belief_probs,
        opp_credibilities=opp_credibilities,
    )


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_dataset(
    games: List[ParsedGame],
    config: Optional[FeatureConfig] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features and labels from a list of ParsedGame objects.

    Returns four arrays:
        X        shape (N, feature_dim)   float32  feature matrix
        y_action shape (N,)               int32    action label (0–6)
        y_target shape (N,)               int32    target label (0–5, 5=no target)
        y_outcome shape (N,)              int32    1=resolved, 0=blocked/failed
    """
    if config is None:
        config = FeatureConfig()

    all_features: List[np.ndarray] = []
    all_action:   List[int]        = []
    all_target:   List[int]        = []
    all_outcome:  List[int]        = []

    for i, game in enumerate(games):
        if show_progress and i % max(1, len(games) // 10) == 0:
            pct = 100 * i / len(games)
            print(f"  Extracting features [{pct:5.1f}%] {i}/{len(games)}", flush=True)

        for sample in extract_features(game, config):
            all_features.append(sample.features)
            all_action.append(sample.action_label)
            all_target.append(sample.target_label)
            all_outcome.append(sample.outcome)

    X        = np.stack(all_features, axis=0)
    y_action = np.array(all_action,  dtype=np.int32)
    y_target = np.array(all_target,  dtype=np.int32)
    y_outcome = np.array(all_outcome, dtype=np.int32)

    if show_progress:
        print(f"  Done — {len(X):,} samples from {len(games):,} games")

    return X, y_action, y_target, y_outcome


# ---------------------------------------------------------------------------
# Convenience: feature dimension for a given config
# ---------------------------------------------------------------------------

def feature_dim(max_players: int = 6) -> int:
    return FeatureConfig(max_players=max_players).feature_dim
