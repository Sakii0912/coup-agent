"""
cards.py — All enums and constants that describe the Coup card game's rules.

The five cards and their abilities:
  Duke        → Tax (take 3 coins), Block Foreign Aid
  Assassin    → Assassinate (pay 3, target loses influence)
  Contessa    → Block Assassination
  Captain     → Steal (take 2 coins from target), Block Steal
  Ambassador  → Exchange (swap cards with deck), Block Steal
"""

from enum import Enum
from typing import Dict, List, Set


class Card(Enum):
    DUKE = "Duke"
    ASSASSIN = "Assassin"
    CONTESSA = "Contessa"
    CAPTAIN = "Captain"
    AMBASSADOR = "Ambassador"


class ActionType(Enum):
    # General actions — no card required
    INCOME = "Income"           # +1 coin, no challenge, no block
    FOREIGN_AID = "ForeignAid"  # +2 coins, no challenge, blockable by Duke
    COUP = "Coup"               # Pay 7 coins, target loses influence, no challenge, no block

    # Character actions — claimable by anyone (bluffing is allowed)
    TAX = "Tax"                 # Duke: +3 coins, challengeable
    ASSASSINATE = "Assassinate" # Assassin: pay 3, target loses influence, challengeable, blockable
    STEAL = "Steal"             # Captain: take 2 from target, challengeable, blockable
    EXCHANGE = "Exchange"       # Ambassador: swap cards with deck, challengeable


# Maps character action → the card that claims it
ACTION_CLAIMS: Dict[ActionType, Card] = {
    ActionType.TAX: Card.DUKE,
    ActionType.ASSASSINATE: Card.ASSASSIN,
    ActionType.STEAL: Card.CAPTAIN,
    ActionType.EXCHANGE: Card.AMBASSADOR,
}

# Maps action → list of cards that can block it
BLOCK_MAP: Dict[ActionType, List[Card]] = {
    ActionType.FOREIGN_AID: [Card.DUKE],
    ActionType.ASSASSINATE: [Card.CONTESSA],
    ActionType.STEAL: [Card.CAPTAIN, Card.AMBASSADOR],
}

# Actions that require choosing a live target
TARGETED_ACTIONS: Set[ActionType] = {
    ActionType.COUP,
    ActionType.ASSASSINATE,
    ActionType.STEAL,
}

# Actions that can be challenged (actor must prove they hold the claimed card)
CHALLENGEABLE_ACTIONS: Set[ActionType] = {
    ActionType.TAX,
    ActionType.ASSASSINATE,
    ActionType.STEAL,
    ActionType.EXCHANGE,
}

# Actions that can be blocked by a counter-claim
BLOCKABLE_ACTIONS: Set[ActionType] = {
    ActionType.FOREIGN_AID,
    ActionType.ASSASSINATE,
    ActionType.STEAL,
}

# Standard deck: 3 copies of each card = 15 total
DECK_COMPOSITION: List[Card] = [card for card in Card for _ in range(3)]
