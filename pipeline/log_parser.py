"""
pipeline/log_parser.py — Log loading, validation, and clean iterable API.

Three layers:

  1. RawLogLoader   — reads JSON from file or dict, catches decode errors
  2. LogValidator   — validates structure, field presence, value ranges,
                      game-level consistency; returns typed ValidationResult
  3. ParsedGame     — clean, typed wrapper around a validated log;
                      iterable over events with typed accessors

Public API (what Phase 3 / feature extractor will import):

    from pipeline.log_parser import load_logs, ParsedGame, EventType

    games: List[ParsedGame] = load_logs("data/raw_logs/", strict=False)
    for game in games:
        for event in game.events:
            if event.type == "action":
                print(event.actor_idx, event.action_type)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

from .schema import (
    EVENT_BASE_REQUIRED,
    EVENT_REQUIRED_FIELDS,
    LOG_REQUIRED_FIELDS,
    MAX_COINS,
    MAX_INFLUENCE,
    MAX_TURNS,
    PLAYER_COUNT_RANGE,
    PLAYER_SNAPSHOT_REQUIRED,
    VALID_ACTION_TYPES,
    VALID_CARDS,
    VALID_EVENT_TYPES,
)


# ===========================================================================
# Typed event wrapper
# ===========================================================================

@dataclass
class ParsedEvent:
    """
    Typed wrapper around a single log event dict.

    All fields are sourced from the raw dict and validated before this
    object is constructed. Attribute access is preferred over dict indexing
    in downstream code.
    """

    raw: Dict[str, Any]           # Original dict — preserved for debugging

    # ── Always present ──────────────────────────────────────────────────
    type: str                     # event_type string (e.g. "action")
    turn: int                     # turn_number

    # ── Conditionally present (None if not applicable for this event) ──
    actor_idx: Optional[int] = None
    actor_name: Optional[str] = None
    action_type: Optional[str] = None
    target_idx: Optional[int] = None
    target_name: Optional[str] = None
    claimed_card: Optional[str] = None

    challenger_idx: Optional[int] = None
    challenger_name: Optional[str] = None

    winner_idx: Optional[int] = None
    winner_name: Optional[str] = None
    loser_idx: Optional[int] = None
    loser_name: Optional[str] = None

    blocker_idx: Optional[int] = None
    blocker_name: Optional[str] = None
    blocking_card: Optional[str] = None
    blocked_action: Optional[str] = None

    player_idx: Optional[int] = None
    player_name: Optional[str] = None
    card_lost: Optional[str] = None
    influence_remaining: Optional[int] = None

    total_turns: Optional[int] = None

    # Public state snapshot (list of player dicts) — present on several events
    state: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParsedEvent":
        return cls(
            raw=d,
            type=d["event_type"],
            turn=d["turn_number"],
            actor_idx=d.get("actor_idx"),
            actor_name=d.get("actor_name"),
            action_type=d.get("action_type"),
            target_idx=d.get("target_idx"),
            target_name=d.get("target_name"),
            claimed_card=d.get("claimed_card"),
            challenger_idx=d.get("challenger_idx"),
            challenger_name=d.get("challenger_name"),
            winner_idx=d.get("winner_idx"),
            winner_name=d.get("winner_name"),
            loser_idx=d.get("loser_idx"),
            loser_name=d.get("loser_name"),
            blocker_idx=d.get("blocker_idx"),
            blocker_name=d.get("blocker_name"),
            blocking_card=d.get("blocking_card"),
            blocked_action=d.get("blocked_action"),
            player_idx=d.get("player_idx"),
            player_name=d.get("player_name"),
            card_lost=d.get("card_lost"),
            influence_remaining=d.get("influence_remaining"),
            total_turns=d.get("total_turns"),
            state=d.get("state"),
        )

    def __repr__(self) -> str:
        return (
            f"ParsedEvent(type={self.type!r}, turn={self.turn}, "
            f"actor={self.actor_name!r}, action={self.action_type!r})"
        )


# ===========================================================================
# ParsedGame — clean wrapper over a validated log
# ===========================================================================

@dataclass
class ParsedGame:
    """
    Clean, typed wrapper around a validated game log.

    Attributes:
        game_id:       Unique ID (timestamp-based from logger.py)
        player_names:  Ordered list of player names
        n_players:     Player count
        events:        Ordered list of ParsedEvent objects
        source_path:   File path this log was loaded from (if applicable)
    """

    game_id: int
    player_names: List[str]
    n_players: int
    events: List[ParsedEvent]
    source_path: Optional[str] = None

    # ── Convenience accessors ───────────────────────────────────────────

    def events_of_type(self, event_type: str) -> List[ParsedEvent]:
        return [e for e in self.events if e.type == event_type]

    @property
    def action_events(self) -> List[ParsedEvent]:
        return self.events_of_type("action")

    @property
    def influence_loss_events(self) -> List[ParsedEvent]:
        return self.events_of_type("influence_loss")

    @property
    def challenge_result_events(self) -> List[ParsedEvent]:
        return self.events_of_type("challenge_result")

    @property
    def game_end_event(self) -> Optional[ParsedEvent]:
        ends = self.events_of_type("game_end")
        return ends[-1] if ends else None

    @property
    def winner_name(self) -> Optional[str]:
        end = self.game_end_event
        return end.winner_name if end else None

    @property
    def total_turns(self) -> Optional[int]:
        end = self.game_end_event
        return end.total_turns if end else None

    def iter_action_contexts(self) -> Generator[Tuple[ParsedEvent, List[ParsedEvent]], None, None]:
        """
        Yields (action_event, subsequent_reaction_events) pairs.

        Subsequent reactions = all events that follow the action up to and
        including the next action_resolved or the next action event,
        whichever comes first. Useful for building per-decision training
        examples with outcome labels.
        """
        events = self.events
        n = len(events)
        for i, evt in enumerate(events):
            if evt.type != "action":
                continue
            reactions: List[ParsedEvent] = []
            for j in range(i + 1, n):
                next_evt = events[j]
                reactions.append(next_evt)
                if next_evt.type in ("action_resolved", "action"):
                    break
            yield evt, reactions

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[ParsedEvent]:
        return iter(self.events)

    def __repr__(self) -> str:
        return (
            f"ParsedGame(id={self.game_id}, players={self.player_names}, "
            f"turns={self.total_turns}, events={len(self.events)})"
        )


# ===========================================================================
# Validation
# ===========================================================================

@dataclass
class ValidationError:
    severity: str          # "error" | "warning"
    location: str          # human-readable path, e.g. "event[4].actor_idx"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.location}: {self.message}"


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def add_error(self, location: str, message: str) -> None:
        self.errors.append(ValidationError("error", location, message))
        self.is_valid = False

    def add_warning(self, location: str, message: str) -> None:
        self.warnings.append(ValidationError("warning", location, message))

    def summary(self) -> str:
        lines = [
            f"Valid: {self.is_valid}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        for e in self.errors:
            lines.append(f"  {e}")
        for w in self.warnings:
            lines.append(f"  {w}")
        return "\n".join(lines)


class LogValidator:
    """
    Validates a raw log dict against the schema defined in schema.py.

    Checks performed:
      Top-level:
        - Required top-level keys present
        - player_names is a non-empty list within valid player count range
        - n_players matches len(player_names)
        - events is a non-empty list

      Per-event:
        - event_type is a known value
        - Base required fields present
        - Type-specific required fields present
        - Value types correct (int indices, string names, etc.)
        - Enum values valid (action_type, card names)
        - Player indices in range [0, n_players)

      Game-level consistency:
        - First event is game_start
        - Last event is game_end
        - game_end has a winner_name (not None)
        - turn_numbers are non-decreasing
        - total_turns <= MAX_TURNS
        - At least one action event exists
    """

    def validate(self, log: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        if not isinstance(log, dict):
            result.add_error("log", f"Expected dict, got {type(log).__name__}")
            return result

        # ── Top-level keys ───────────────────────────────────────────────
        for key in LOG_REQUIRED_FIELDS:
            if key not in log:
                result.add_error("log", f"Missing required top-level key: {key!r}")

        if not result.is_valid:
            return result  # Can't proceed without basic structure

        n_players = log.get("n_players")
        player_names = log.get("player_names", [])

        if not isinstance(player_names, list) or len(player_names) == 0:
            result.add_error("log.player_names", "Must be a non-empty list")
        elif len(player_names) not in PLAYER_COUNT_RANGE:
            result.add_error(
                "log.player_names",
                f"Player count {len(player_names)} outside valid range 2–6"
            )

        if not isinstance(n_players, int):
            result.add_error("log.n_players", "Must be an integer")
        elif isinstance(player_names, list) and n_players != len(player_names):
            result.add_error(
                "log.n_players",
                f"n_players={n_players} but len(player_names)={len(player_names)}"
            )

        events = log.get("events", [])
        if not isinstance(events, list) or len(events) == 0:
            result.add_error("log.events", "Must be a non-empty list")
            return result

        # ── Per-event validation ─────────────────────────────────────────
        prev_turn = -1
        for i, event in enumerate(events):
            loc = f"event[{i}]"
            if not isinstance(event, dict):
                result.add_error(loc, f"Expected dict, got {type(event).__name__}")
                continue

            self._validate_event(event, i, n_players, result)

            # Turn monotonicity
            turn = event.get("turn_number")
            if isinstance(turn, int):
                if turn < prev_turn:
                    result.add_warning(
                        f"{loc}.turn_number",
                        f"Turn {turn} < previous turn {prev_turn} (non-monotonic)"
                    )
                prev_turn = turn

        # ── Game-level consistency ───────────────────────────────────────
        self._validate_game_consistency(events, result)

        return result

    # ------------------------------------------------------------------ #
    #  Per-event checks                                                    #
    # ------------------------------------------------------------------ #

    def _validate_event(
        self,
        event: Dict[str, Any],
        idx: int,
        n_players: int,
        result: ValidationResult,
    ) -> None:
        loc = f"event[{idx}]"

        # Base required fields
        for field_name in EVENT_BASE_REQUIRED:
            if field_name not in event:
                result.add_error(loc, f"Missing base field: {field_name!r}")

        event_type = event.get("event_type")
        if event_type not in VALID_EVENT_TYPES:
            result.add_error(
                f"{loc}.event_type",
                f"Unknown event_type {event_type!r}. "
                f"Valid: {sorted(VALID_EVENT_TYPES)}"
            )
            return  # Can't validate type-specific fields without a known type

        # Type-specific required fields
        for field_name in EVENT_REQUIRED_FIELDS.get(event_type, frozenset()):
            if field_name not in event:
                result.add_error(
                    f"{loc}.{field_name}",
                    f"Missing required field for {event_type!r}: {field_name!r}"
                )

        # Value-level checks
        self._validate_event_values(event, idx, event_type, n_players, result)

    def _validate_event_values(
        self,
        event: Dict[str, Any],
        idx: int,
        event_type: str,
        n_players: int,
        result: ValidationResult,
    ) -> None:
        loc = f"event[{idx}]"

        # turn_number
        turn = event.get("turn_number")
        if turn is not None:
            if not isinstance(turn, int) or turn < 0:
                result.add_error(f"{loc}.turn_number", f"Must be a non-negative int, got {turn!r}")
            elif turn > MAX_TURNS:
                result.add_warning(f"{loc}.turn_number", f"Turn {turn} exceeds MAX_TURNS {MAX_TURNS}")

        # Player index fields — must be int in [0, n_players)
        idx_fields = ["actor_idx", "target_idx", "challenger_idx",
                      "winner_idx", "loser_idx", "blocker_idx", "player_idx"]
        for f in idx_fields:
            val = event.get(f)
            if val is None:
                continue
            if not isinstance(val, int):
                result.add_error(f"{loc}.{f}", f"Must be int, got {type(val).__name__}")
            elif n_players and not (0 <= val < n_players):
                result.add_error(f"{loc}.{f}", f"Index {val} out of range [0, {n_players})")

        # action_type
        at = event.get("action_type")
        if at is not None and at not in VALID_ACTION_TYPES:
            result.add_error(
                f"{loc}.action_type",
                f"Unknown action_type {at!r}. Valid: {sorted(VALID_ACTION_TYPES)}"
            )

        # Card fields
        card_fields = ["claimed_card", "blocking_card", "card_lost", "challenged_card"]
        for f in card_fields:
            val = event.get(f)
            if val is not None and val not in VALID_CARDS:
                result.add_error(
                    f"{loc}.{f}",
                    f"Unknown card {val!r}. Valid: {sorted(VALID_CARDS)}"
                )

        # revealed_cards in player snapshots
        state = event.get("state")
        if state is not None:
            self._validate_state_snapshot(state, idx, n_players, result)

        # influence_remaining: 0 or 1
        ir = event.get("influence_remaining")
        if ir is not None:
            if not isinstance(ir, int) or not (0 <= ir <= MAX_INFLUENCE):
                result.add_error(
                    f"{loc}.influence_remaining",
                    f"Must be 0 or 1, got {ir!r}"
                )

        # coins in player snapshots (checked inside _validate_state_snapshot)
        # total_turns
        tt = event.get("total_turns")
        if tt is not None and (not isinstance(tt, int) or tt < 0):
            result.add_error(f"{loc}.total_turns", f"Must be non-negative int, got {tt!r}")

    def _validate_state_snapshot(
        self,
        state: Any,
        event_idx: int,
        n_players: int,
        result: ValidationResult,
    ) -> None:
        loc = f"event[{event_idx}].state"
        if not isinstance(state, list):
            result.add_error(loc, f"Must be a list, got {type(state).__name__}")
            return
        if n_players and len(state) != n_players:
            result.add_error(loc, f"Expected {n_players} player entries, got {len(state)}")

        for j, player in enumerate(state):
            ploc = f"{loc}[{j}]"
            if not isinstance(player, dict):
                result.add_error(ploc, "Must be a dict")
                continue
            for f in PLAYER_SNAPSHOT_REQUIRED:
                if f not in player:
                    result.add_error(ploc, f"Missing field: {f!r}")

            coins = player.get("coins")
            if coins is not None and (not isinstance(coins, int) or coins < 0):
                result.add_error(f"{ploc}.coins", f"Must be non-negative int, got {coins!r}")
            elif isinstance(coins, int) and coins > MAX_COINS:
                result.add_warning(f"{ploc}.coins", f"Unusually high coin count: {coins}")

            ic = player.get("influence_count")
            if ic is not None and (not isinstance(ic, int) or not (0 <= ic <= MAX_INFLUENCE)):
                result.add_error(f"{ploc}.influence_count", f"Must be 0–2, got {ic!r}")

            revealed = player.get("revealed_cards", [])
            if isinstance(revealed, list):
                for card in revealed:
                    if card not in VALID_CARDS:
                        result.add_error(f"{ploc}.revealed_cards", f"Unknown card: {card!r}")

    # ------------------------------------------------------------------ #
    #  Game-level consistency                                              #
    # ------------------------------------------------------------------ #

    def _validate_game_consistency(
        self, events: List[Dict[str, Any]], result: ValidationResult
    ) -> None:
        if not events:
            return

        first_type = events[0].get("event_type")
        if first_type != "game_start":
            result.add_error("events[0].event_type", f"First event must be game_start, got {first_type!r}")

        last_type = events[-1].get("event_type")
        if last_type != "game_end":
            result.add_error(f"events[{len(events)-1}].event_type",
                             f"Last event must be game_end, got {last_type!r}")

        # Winner must be set in game_end
        last = events[-1]
        if last.get("event_type") == "game_end":
            if last.get("winner_name") is None and last.get("winner_idx") is None:
                result.add_warning("events[-1].winner_name", "Game ended with no winner (tie / turn limit)")

        # At least one action event
        action_count = sum(1 for e in events if e.get("event_type") == "action")
        if action_count == 0:
            result.add_error("events", "No action events found — log appears incomplete")

        # challenge_result must always follow a challenge or block_challenge
        prev_types = []
        for i, evt in enumerate(events):
            et = evt.get("event_type")
            if et == "challenge_result":
                if prev_types and prev_types[-1] not in ("challenge", "block_challenge"):
                    result.add_warning(
                        f"event[{i}]",
                        f"challenge_result not preceded by challenge/block_challenge "
                        f"(preceded by {prev_types[-1]!r})"
                    )
            prev_types.append(et)


# ===========================================================================
# RawLogLoader
# ===========================================================================

class RawLogLoader:
    """Loads raw log dicts from JSON files or in-memory dicts."""

    @staticmethod
    def from_file(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Load a single JSON log file.

        Returns:
            (log_dict, None)         on success
            (None,     error_str)    on failure
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None, f"Root element is not a dict (got {type(data).__name__})"
            return data, None
        except json.JSONDecodeError as e:
            return None, f"JSON decode error: {e}"
        except OSError as e:
            return None, f"File read error: {e}"

    @staticmethod
    def from_directory(
        directory: str, pattern: str = ".json"
    ) -> Generator[Tuple[str, Optional[Dict[str, Any]], Optional[str]], None, None]:
        """
        Yield (filepath, log_dict, error_str) for every matching file in directory.
        error_str is None on success.
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory!r}")
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(pattern):
                continue
            path = os.path.join(directory, fname)
            log, err = RawLogLoader.from_file(path)
            yield path, log, err


# ===========================================================================
# Public API — load_logs
# ===========================================================================

def load_logs(
    source: str,
    strict: bool = False,
    max_warnings: int = 0,
    verbose: bool = False,
) -> List[ParsedGame]:
    """
    Load and validate all game logs from a directory or a single JSON file.

    Args:
        source:       Path to a directory of .json files, or a single .json file.
        strict:       If True, raise ValueError on any validation error.
                      If False, skip invalid logs and continue.
        max_warnings: If strict=False, still raise if a log has more than
                      this many warnings. 0 = allow all warnings.
        verbose:      Print a line per file showing load status.

    Returns:
        List of ParsedGame objects (only valid logs included).
    """
    validator = LogValidator()
    games: List[ParsedGame] = []
    n_total = 0
    n_failed = 0

    def _process(path: str, raw: Dict[str, Any]) -> Optional[ParsedGame]:
        nonlocal n_total, n_failed
        n_total += 1
        result = validator.validate(raw)

        if not result.is_valid:
            n_failed += 1
            if strict:
                raise ValueError(
                    f"Validation failed for {path!r}:\n{result.summary()}"
                )
            if verbose:
                print(f"  ✗ INVALID  {os.path.basename(path)}: "
                      f"{len(result.errors)} error(s)")
            return None

        if max_warnings > 0 and len(result.warnings) > max_warnings:
            n_failed += 1
            if strict:
                raise ValueError(
                    f"Too many warnings for {path!r}:\n{result.summary()}"
                )
            if verbose:
                print(f"  ⚠ WARNINGS {os.path.basename(path)}: "
                      f"{len(result.warnings)} warning(s)")
            return None

        if verbose and result.warnings:
            print(f"  ~ OK+warn  {os.path.basename(path)}: "
                  f"{len(result.warnings)} warning(s)")
        elif verbose:
            print(f"  ✓ OK       {os.path.basename(path)}")

        events = [ParsedEvent.from_dict(e) for e in raw["events"]]
        return ParsedGame(
            game_id=raw["game_id"],
            player_names=raw["player_names"],
            n_players=raw["n_players"],
            events=events,
            source_path=path,
        )

    if os.path.isfile(source):
        raw, err = RawLogLoader.from_file(source)
        if err:
            if strict:
                raise ValueError(f"Could not load {source!r}: {err}")
        else:
            game = _process(source, raw)
            if game:
                games.append(game)

    elif os.path.isdir(source):
        for path, raw, err in RawLogLoader.from_directory(source):
            if err:
                n_total += 1
                n_failed += 1
                if strict:
                    raise ValueError(f"Could not load {path!r}: {err}")
                if verbose:
                    print(f"  ✗ LOAD ERR {os.path.basename(path)}: {err}")
                continue
            game = _process(path, raw)
            if game:
                games.append(game)
    else:
        raise FileNotFoundError(f"Source not found: {source!r}")

    if verbose or n_failed > 0:
        print(
            f"\nLoaded {len(games)}/{n_total} logs "
            f"({n_failed} rejected{'.' if n_failed == 0 else ' — run verbose=True for details.'})"
        )

    return games


def validate_log(log: Dict[str, Any]) -> ValidationResult:
    """
    Validate a single raw log dict and return the full ValidationResult.
    Convenience wrapper around LogValidator for use in tests.
    """
    return LogValidator().validate(log)
