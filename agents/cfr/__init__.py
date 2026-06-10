"""
agents/cfr — CFR infrastructure (sub-goal 4.1).

  information_set.py  InformationSet  — hashable abstracted infoset key
  strategy_table.py   StrategyTable   — regret matching + CFR+ accumulation
  cfr_solver.py       CFRSolver       — Chance Sampling MCCFR training loop
"""

from .information_set import InformationSet, DecisionType, bucket_coins
from .strategy_table import StrategyTable

__all__ = ["InformationSet", "DecisionType", "bucket_coins", "StrategyTable"]
