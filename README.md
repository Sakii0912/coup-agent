# Coup AI Agent

A Python project to build a competitive AI agent for the card game **Coup** using Counterfactual Regret Minimization (CFR) and neural network-based opponent modelling.

---

## Project Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **1 — Game Engine** | Model rules, state, actions, resolution, self-play | ✅ Complete |
| **2 — Data Pipeline** | Scrape or generate game logs, define schema | 🔜 Next |
| **3 — Belief Tracker** | Bayesian inference over hidden opponent cards | 🔜 |
| **4 — Agent Training** | CFR self-play + neural net policy | 🔜 |
| **5 — Live Interface** | Input game state → get best action recommendation | 🔜 |

---

## Project Structure

```
coup_agent/
│
├── coup/                        # Core game library
│   ├── __init__.py              # Public API exports
│   ├── cards.py                 # Card/ActionType enums, game constants
│   ├── state.py                 # PlayerState, GameState, Observation dataclasses
│   ├── actions.py               # Action/Block dataclasses, legal action generator
│   ├── resolution.py            # Challenge resolution and action effect application
│   ├── logger.py                # Structured event logger (JSON-serialisable)
│   ├── game.py                  # Game orchestrator — full turn loop
│   └── agents/
│       ├── __init__.py
│       ├── base.py              # Abstract Agent base class
│       └── random_agent.py      # RandomAgent and HonestAgent implementations
│
├── tests/
│   ├── test_cards_and_state.py  # Unit tests: deck, state, observations, legal actions
│   ├── test_resolution.py       # Unit tests: challenge resolution, action effects
│   └── test_game_integration.py # Integration tests: full games, log structure, invariants
│
├── simulator.py                 # Self-play CLI and API for generating training data
├── data/
│   └── raw_logs/                # Output directory for game log JSON files
└── requirements.txt
```

---

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the test suite

```bash
# All tests with verbose output
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_cards_and_state.py -v
python -m pytest tests/test_resolution.py -v
python -m pytest tests/test_game_integration.py -v

# With coverage report
python -m pytest tests/ --cov=coup --cov-report=term-missing
```

### Play a single game (verbose)

```python
from coup.game import Game
from coup.agents.random_agent import RandomAgent

agents = [RandomAgent() for _ in range(4)]
game = Game(agents=agents, seed=42, verbose=True)
log = game.play_game()
```

### Generate training data

```bash
# 1,000 games, 4 players, logs saved to data/raw_logs/
python simulator.py --n_games 1000 --n_players 4 --seed 42

# 5,000 games, 3 players, mixed agent types
python simulator.py --n_games 5000 --n_players 3 --agent mixed

# Programmatic API
from simulator import run_simulation
logs = run_simulation(n_games=500, n_players=4, seed=0)
```

---

## Running the Tests

There are **three test files**, each covering a distinct layer of the engine.

### `tests/test_cards_and_state.py` — Unit tests (46 tests)

Tests the lowest-level building blocks.

| Class | What it tests |
|-------|--------------|
| `TestDeck` | Deck has exactly 15 cards, 3 of each type |
| `TestGameStateInit` | Correct dealing, coin assignment, seeded reproducibility, player count validation |
| `TestPlayerState` | `lose_influence`, `swap_card`, `has_card`, `public_view` (hides hand) |
| `TestObservation` | Player sees own hand, opponents' hands hidden, JSON serialisable, dead-player turn skip |
| `TestLegalActions` | Income/Foreign Aid/Tax/Exchange always available; Coup ≥ 7 coins; must-coup at 10; Assassinate ≥ 3; Steal vs 0-coin targets; correct target lists; dead players excluded |
| `TestLegalBlocks` | Each action blocked by the correct cards; unblockable actions return empty |

```bash
python -m pytest tests/test_cards_and_state.py -v
```

### `tests/test_resolution.py` — Unit tests (15 tests)

Tests the pure resolution functions in isolation, using hand-crafted game states.

| Class | What it tests |
|-------|--------------|
| `TestResolveChallenge` | Actor wins when they hold the card (challenger loses influence, card swapped); bluffer loses when they don't (actor loses influence) |
| `TestApplyActionEffect` | Income (+1), Foreign Aid (+2), Tax (+3), Coup (−7 coins, target loses influence, death), Assassinate (influence loss), Steal (2-coin transfer, partial transfer at 1 coin, zero-coin no-op), Exchange (hand size preserved, cards returned to deck, invalid keep count raises) |

```bash
python -m pytest tests/test_resolution.py -v
```

### `tests/test_game_integration.py` — Integration tests (31 tests)

Runs complete games end-to-end. Does not test internals — only observable outcomes.

| Class | What it tests |
|-------|--------------|
| `TestGameTermination` | Games terminate with exactly one winner for all player counts (2–6) and multiple seeds; winner is the only alive player; log starts with `game_start`, ends with `game_end` |
| `TestLogStructure` | Log has required keys; all action events carry `actor_idx` and `action_type`; log is JSON-serialisable; influence loss events name the card |
| `TestInvariants` | No player ever has negative coins; no player holds more than 2 cards; total cards in deck + hands + revealed = 15 (conservation) |
| `TestAgentVariety` | HonestAgent-only games; mixed RandomAgent/HonestAgent; extreme challenge/block probabilities (0.0 and 0.9) still terminate |
| `TestSimulator` | `run_simulation` returns exactly N logs; logs are valid JSON with a `game_end` event |

```bash
python -m pytest tests/test_game_integration.py -v
```

---

## Phase 1 — Implementation Notes

### Design philosophy

The engine is built around three principles that matter for all later phases:

**1. Strict separation of god-view and agent-view.** `GameState` holds all information including hidden cards. Agents never receive a `GameState` — they receive an `Observation`, which is the subset of information that player is entitled to see. This separation is enforced structurally, not by convention. When Phase 3 adds a belief tracker, the Observation gains a `belief_state` field and no other code changes.

**2. Immutable action objects.** `Action` and `Block` are frozen dataclasses. This means they can be stored in logs, compared safely, and passed to multiple functions without risk of mutation. It also makes it straightforward to enumerate all legal actions as a plain Python list.

**3. Callbacks instead of coupling.** Resolution functions (`resolve_challenge`, `apply_action_effect`) never call agent methods directly. They receive callbacks (`lose_influence_callback`, `exchange_callback`) injected by `Game`. This keeps the resolution logic pure and testable with simple lambdas, and means Phase 4 agents slot in without touching resolution code.

---

### `cards.py` — Enums and constants

All game rules that are pure data (no logic) live here.

`ACTION_CLAIMS` maps each character action to the card it requires. During a challenge, the engine looks up the claimed card from this dict — if the actor doesn't hold it, they were bluffing.

`BLOCK_MAP` maps each blockable action to the list of cards that can block it. Foreign Aid is blocked by Duke. Steal is blocked by Captain or Ambassador (either card is a valid block claim, which is why the value is a list). This dict drives `get_legal_blocks` in `actions.py`.

`DECK_COMPOSITION` is the canonical 15-card deck (3× each card). Every new game shuffles a copy of this list and deals from it.

---

### `state.py` — Data structures

**`PlayerState`** is intentionally mutable. Fields mutated during a game: `hand` (cards drawn/lost/swapped), `revealed` (cards flipped face-up), `coins`. The `public_view()` method returns a dict with `influence_count` (number of remaining hidden cards) but omits `hand` entirely — this is what opponents see.

**`GameState.new_game()`** is the only constructor used in normal play. It shuffles the deck, deals 2 cards and 2 coins to each player, and sets `current_player_idx = 0`. Passing a `seed` makes the deal fully reproducible, which is critical for running the same test scenario repeatedly.

**`GameState.advance_turn()`** skips dead players automatically. Dead players keep their slot in `players` (so indices stay stable) but are simply skipped when iterating.

**`Observation`** is the data structure passed to every agent decision method. It currently contains: own hand, own coins, own revealed cards, public view of all other players, deck size, and whose turn it is. In Phase 3, a `belief_state` field will be added here — a probability matrix over each opponent's possible cards — without any other changes to the agent interface.

---

### `actions.py` — Legal move generation

`get_legal_actions` enforces all coin constraints and the must-coup rule. The must-coup rule is important to implement correctly: at 10+ coins the function returns *only* Coup actions (one per alive opponent). No other action is legal. This is what the rule actually says — it is not just a strong recommendation.

Steal has a special constraint: `target.coins > 0`. A player with 0 coins is not a valid steal target. This prevents generating actions whose effect would be a no-op, which could confuse a policy network.

Every targeted action generates one `Action` per valid target (rather than one action with a target list). This keeps actions atomic and makes the policy output a flat choice over a well-defined set.

`get_legal_blocks` returns one `Block` per card that can legally be claimed. For Steal, two blocks are generated (claiming Captain, or claiming Ambassador). The blocker does not have to actually hold the card — bluffing a block is always legal. The challenge system handles proof of the claim.

---

### `resolution.py` — Challenge and effect logic

**Challenge resolution** is the core mechanic where hidden information becomes observable. The sequence is:

1. Check if actor's hand contains the claimed card.
2. If yes (actor wins): call `lose_influence_callback(challenger_idx)` then `PlayerState.swap_card()` on the actor. The swap is mandatory — the actor returns their proved card to the deck and draws a fresh one, removing the information advantage the challenger just gained.
3. If no (challenger wins): call `lose_influence_callback(actor_idx)`.

The `lose_influence_callback` pattern is how the engine asks an agent which card to reveal without coupling resolution logic to the agent class hierarchy. In tests, a lambda (`return 0`) is sufficient.

**`apply_action_effect`** is only called after all challenges and blocks have resolved in favour of the action proceeding. The Assassinate coin cost (3) is deducted in `game.py` when the action is *declared*, before any challenge. This matches the real rules: the coins are spent on the attempt, not the outcome. If the assassination is blocked or the assassin loses a challenge, the coins are still gone.

**Exchange** uses a second callback (`exchange_callback`) to ask the actor which cards to keep. The function draws up to 2 cards from the deck, combines them with the actor's hand into `all_options`, calls the callback to get keep indices, then returns unchosen cards to the deck and reshuffles. The deck reshuffle after every exchange is important — it prevents an observer from inferring deck composition by watching what is returned.

---

### `game.py` — Turn orchestrator

`Game._play_turn()` implements the full reactive turn structure:

```
1. Actor chooses action
   → if action costs coins (Assassinate): deduct immediately
2. If action is challengeable:
   → poll other alive players in seat order
   → first challenger triggers resolve_challenge()
   → actor loses → action fails and turn ends
   → actor wins → action continues
3. If action is blockable (and step 2 passed or was skipped):
   → poll eligible blockers (all players for Foreign Aid, target only for Steal/Assassinate)
   → if blocked: run block challenge phase (same structure as step 2)
   → block survives → action fails and turn ends
   → block challenged and fails → action continues
4. Apply action effect
```

Players are polled in seat order starting from the player immediately after the actor. The first player who challenges or blocks stops the polling — subsequent players in that round don't get to react. This matches the actual rules.

The `MAX_TURNS = 500` guard prevents infinite loops in degenerate cases (e.g., all passive agents accumulating coins and never couping). In practice, random games rarely exceed 80–100 turns.

---

### `logger.py` — Event log

Every observable event is recorded as a `LogEvent` with its turn number and a dict of relevant fields. The log is the training data for all later phases. The key design decision is that the log only records *observable* information — it never logs hidden cards unless they are revealed through a challenge or influence loss. This mirrors what a human player can see and keeps the training signal honest.

The full log schema (what each event type records) is documented in the module docstring. Events you should know:

- `action` — actor, action type, claimed card, target, public state snapshot
- `challenge_result` — winner and loser indices (lets you reconstruct who was bluffing)
- `influence_loss` — which card was revealed (the most information-rich event)
- `game_end` — winner, total turns

---

### `agents/` — Agent architecture

`Agent` (abstract base class) defines six decision points that every agent must implement. The six methods map directly to the six moments in a turn where a player must make a choice:

| Method | Called when |
|--------|-------------|
| `choose_action` | It is your turn |
| `choose_to_challenge_action` | Another player declared a character action |
| `choose_to_block` | An action targeting you (or anyone, for Foreign Aid) was declared |
| `choose_to_challenge_block` | Someone claimed to block an action |
| `choose_card_to_lose` | You must reveal an influence card |
| `choose_exchange_cards` | You played Ambassador Exchange |

`RandomAgent` implements all six uniformly at random, with configurable `challenge_prob` and `block_prob`. These two parameters are the main dials for generating diverse training data — games with `challenge_prob=0.0` play very differently from games with `challenge_prob=0.8`.

`HonestAgent` extends `RandomAgent` but overrides `choose_action` (only plays actions it holds the card for) and `choose_to_block` (only blocks with cards it actually holds). It challenges more often when it holds the claimed card (reducing the probability the claim is true). This agent is useful as a benchmark: a policy that loses to `HonestAgent` has not learned anything useful.

---

## What Phase 2 will add

- A scraper/search for Coup game logs on BoardGameArena or GitHub
- If no public dataset exists: a high-volume simulator run (100k+ games with mixed agents) to generate a synthetic training corpus
- A log replay tool (feed a saved JSON log back through the engine and verify it is valid)
- Feature extraction: convert a game log into a tensor suitable for a neural network

---

## What Phase 3 will add

- `BeliefState` class: a `(n_players, n_cards)` probability matrix updated after every observable event
- Bayesian update rules: seeing player X claim Duke reduces the prior probability that other players hold Duke
- Consistency tracking: if two players both claim Duke in a 3-player game, one must be bluffing — the belief state captures this
- The `Observation` dataclass gets a `belief_state` field (no other interface changes)
