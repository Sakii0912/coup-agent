"""
pipeline — Phase 2: Data Pipeline

Sub-goal 2: Log parsing and validation
  schema.py    — field definitions, valid enum values, value constraints
  log_parser.py — RawLogLoader, LogValidator, ParsedGame, load_logs()

Sub-goal 3 (next): large-scale simulation run
Sub-goal 4 (next): feature extraction
Sub-goal 5 (next): PyTorch Dataset class
Sub-goal 6 (next): data quality report
"""

from .schema import VALID_CARDS, VALID_ACTION_TYPES, VALID_EVENT_TYPES
from .log_parser import (
    ParsedEvent,
    ParsedGame,
    ValidationResult,
    ValidationError,
    LogValidator,
    RawLogLoader,
    load_logs,
    validate_log,
)

__all__ = [
    "VALID_CARDS", "VALID_ACTION_TYPES", "VALID_EVENT_TYPES",
    "ParsedEvent", "ParsedGame",
    "ValidationResult", "ValidationError",
    "LogValidator", "RawLogLoader",
    "load_logs", "validate_log",
]
