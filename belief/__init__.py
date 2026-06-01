"""
belief — Phase 3: Belief Tracker

Sub-goal 3.1: Prior distribution over hidden cards
  belief_state.py  — BeliefState dataclass, hypergeometric prior, constructors

Sub-goal 3.2: Bayesian update rules per event type
  belief_updater.py — BeliefUpdater with one rule per observable event type
Sub-goal 3.3: Global deck constraint normalisation
  constraint_normaliser.py — Sinkhorn-style iterative column/row normalisation
Sub-goal 3.4: Per-opponent bluff rate modelling
  opponent_model.py — Beta-posterior bluff rate tracker, credibility scores
Sub-goal 3.6: End-to-end tests and validation
  belief_tracker.py — BeliefTracker orchestrator (BeliefState + BeliefUpdater
                       + OpponentModel + ConstraintNormaliser)
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
from .constraint_normaliser import ConstraintNormaliser, normalise as normalise_belief
from .opponent_model import (
    OpponentModel,
    PlayerBluffStats,
    PRIOR_STRENGTH,
    MIN_CREDIBILITY,
    MAX_CREDIBILITY,
    DEFAULT_CREDIBILITY,
)
from .belief_tracker import BeliefTracker, BeliefSnapshot

__all__ = [
    "BeliefState",
    "CARDS", "N_CARDS", "CARD_TO_IDX", "TOTAL_CARDS", "COPIES_PER_CARD",
    "hypergeometric_at_least_one",
    "BeliefUpdater", "CLAIM_CREDIBILITY",
    "ConstraintNormaliser", "normalise_belief",
    "OpponentModel", "PlayerBluffStats",
    "PRIOR_STRENGTH", "MIN_CREDIBILITY", "MAX_CREDIBILITY", "DEFAULT_CREDIBILITY",
    "BeliefTracker", "BeliefSnapshot",
]
