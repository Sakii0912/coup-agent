"""
pipeline/feature_extractor.py — Convert game log events into feature vectors.

Each observable decision point in a game log is converted to a flat numpy
vector that captures everything the acting player could see at that moment.

Feature vector layout (total: 135 dims for 6-player games, scales with n_players):
─────────────────────────────────────────────────────────────────────────────────
  Section               Dims    Description
─────────────────────────────────────────────────────────────────────────────────
  My hand               5       One-hot per card (Duke/Assassin/Contessa/Captain/Ambassador)
                                Double-count for two copies (can be 0, 1, or 2 per card)
  My coins              1       Normalised (÷ 12)
  My influence          1       Remaining influence (0–2, normalised ÷ 2)
  My revealed cards     5       One-hot count of my revealed cards
  Opponents (×5 slots)  6 each  [influence_count/2, coins/12, 4×revealed card one-hot count,
                                 is_alive] — padded with zeros for absent players
  Action type           7       One-hot (Income/ForeignAid/Coup/Tax/Assassinate/Steal/Exchange)
  Claimed card          6       One-hot + "no claim" dim (None = index 5)
  Target is me          1       Binary: is the action targeting this player?
  Actor is me           1       Binary: is this player the actor?
  Turn number           1       Normalised (÷ 500)
  Coins on table        1       Sum of all players' coins, normalised (÷ 72)
  Action history        7×5=35  One-hot counts of last 5 action types seen this game
─────────────────────────────────────────────────────────────────────────────────

Label vector layout:
  Action label          7       One-hot index into ActionType for the chosen action
  Target label          6       One-hot index into player slot (5 opponents + "no target")
  Outcome               1       1 = actor won this turn (action resolved), 0 = blocked/failed
─────────────────────────────────────────────────────────────────────────────────

Usage:
    from pipeline.feature_extractor import extract_features, FeatureConfig, FEATURE_DIM

    config = FeatureConfig(max_players=6)
    samples = extract_features(parsed_game, config)   # List[Sample]
    for s in samples:
        print(s.features.shape, s.action_label, s.outcome)
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

    max_players: maximum number of opponent slots to encode.
                 Set to 5 (for up to 6 total players) for a fixed-size
                 vector that works for any game size.
    """
    max_players: int = 6                # total including self → 5 opponent slots

    @property
    def n_opponent_slots(self) -> int:
        return self.max_players - 1     # 5 for max_players=6

    @property
    def feature_dim(self) -> int:
        """Total feature vector length."""
        my_dims       = N_CARDS + 1 + 1 + N_CARDS          # hand + coins + influence + revealed
        opp_dims      = self.n_opponent_slots * (1 + 1 + N_CARDS + 1)  # per slot: influence, coins, revealed×5, is_alive
        action_dims   = N_ACTIONS                           # action one-hot
        claimed_dims  = N_CARDS + 1                         # card + "no claim"
        meta_dims     = 1 + 1 + 1 + 1                      # target_is_me, actor_is_me, turn, table_coins
        history_dims  = N_ACTIONS * HISTORY_LENGTH
        return my_dims + opp_dims + action_dims + claimed_dims + meta_dims + history_dims


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

    return np.concatenate(parts).astype(np.float32)


def _card_counts(cards: List[str]) -> np.ndarray:
    """Return a float32 vector of shape (N_CARDS,) counting occurrences."""
    vec = np.zeros(N_CARDS, dtype=np.float32)
    for c in cards:
        if c in CARD_TO_IDX:
            vec[CARD_TO_IDX[c]] += 1.0
    return vec


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
