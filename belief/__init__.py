"""
belief — Phase 3: Belief Tracker

Sub-goal 3.1: Prior distribution over hidden cards
  belief_state.py  — BeliefState dataclass, hypergeometric prior, constructors

Sub-goal 3.2: Bayesian update rules per event type
  belief_updater.py — BeliefUpdater with one rule per observable event type
Sub-goal 3.3 (next): Global deck constraint normalisation
Sub-goal 3.4 (next): Per-opponent bluff rate modelling
Sub-goal 3.5 (next): Observation integration + feature vector extension
"""

from .belief_state import (
    BeliefState,
    CARDS,
    N_CARDS,
    CARD_TO_IDX,
    TOTAL_CARDS,
    COPIES_PER_CARD,
    hypergeometric_at_least_one,
)
from .belief_updater import BeliefUpdater, CLAIM_CREDIBILITY

__all__ = [
    "BeliefState",
    "CARDS",
    "N_CARDS",
    "CARD_TO_IDX",
    "TOTAL_CARDS",
    "COPIES_PER_CARD",
    "hypergeometric_at_least_one",
    "BeliefUpdater",
    "CLAIM_CREDIBILITY",
]
