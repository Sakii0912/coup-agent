"""
pipeline/schema.py — Single source of truth for the log event schema.

Every event type has a set of REQUIRED fields and a set of OPTIONAL fields.
The validator in log_parser.py checks against these definitions.

Design note: we keep schema knowledge here (not in logger.py or cards.py) so
that Phase 3/4 code can import it without importing the full game engine. This
also makes it easy to extend when new event fields are added.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Set

# ---------------------------------------------------------------------------
# Valid enum values (mirrors coup/cards.py — duplicated intentionally so
# the pipeline has zero dependency on the game engine at parse time)
# ---------------------------------------------------------------------------

VALID_CARDS: FrozenSet[str] = frozenset(
    ["Duke", "Assassin", "Contessa", "Captain", "Ambassador"]
)

VALID_ACTION_TYPES: FrozenSet[str] = frozenset(
    ["Income", "ForeignAid", "Coup", "Tax", "Assassinate", "Steal", "Exchange"]
)

VALID_EVENT_TYPES: FrozenSet[str] = frozenset([
    "game_start",
    "action",
    "challenge",
    "challenge_result",
    "block",
    "block_challenge",
    "influence_loss",
    "action_resolved",
    "game_end",
])

# ---------------------------------------------------------------------------
# Required fields per event type
# ---------------------------------------------------------------------------

# Fields that every event must have regardless of type
EVENT_BASE_REQUIRED: FrozenSet[str] = frozenset(["event_type", "turn_number"])

# Required fields per event_type (on top of base)
EVENT_REQUIRED_FIELDS: Dict[str, FrozenSet[str]] = {
    "game_start": frozenset(["player_names", "n_players", "state"]),
    "action": frozenset([
        "actor_idx", "actor_name", "action_type",
        "target_idx", "target_name", "claimed_card", "state",
    ]),
    "challenge": frozenset([
        "challenger_idx", "challenger_name",
        "challenged_action", "challenged_card",
        "actor_idx", "actor_name",
    ]),
    "challenge_result": frozenset([
        "winner_idx", "winner_name",
        "loser_idx", "loser_name",
        "state",
    ]),
    "block": frozenset([
        "blocker_idx", "blocker_name",
        "blocking_card", "blocked_action",
    ]),
    "block_challenge": frozenset([
        "challenger_idx", "challenger_name",
        "challenged_card",
        "blocker_idx", "blocker_name",
    ]),
    "influence_loss": frozenset([
        "player_idx", "player_name",
        "card_lost", "influence_remaining",
    ]),
    "action_resolved": frozenset([
        "action_type", "actor_idx", "actor_name", "state",
    ]),
    "game_end": frozenset([
        "winner_idx", "winner_name", "total_turns",
    ]),
}

# Required fields for a top-level game log dict
LOG_REQUIRED_FIELDS: FrozenSet[str] = frozenset(
    ["game_id", "player_names", "n_players", "events"]
)

# Required fields for a public player snapshot (inside "state" lists)
PLAYER_SNAPSHOT_REQUIRED: FrozenSet[str] = frozenset([
    "name", "idx", "coins", "influence_count", "revealed_cards", "is_alive"
])

# ---------------------------------------------------------------------------
# Value constraints
# ---------------------------------------------------------------------------

PLAYER_COUNT_RANGE: range = range(2, 7)   # 2–6 inclusive
MAX_COINS: int = 12                        # should never see > 12 in a valid game
MAX_INFLUENCE: int = 2
MAX_TURNS: int = 600                       # safety ceiling (simulator uses 500)
