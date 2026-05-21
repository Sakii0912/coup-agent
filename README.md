# Coup AI Agent

A Python project to build a competitive AI agent for the card game **Coup** using Counterfactual Regret Minimization (CFR) and neural network-based opponent modelling.

---

## Project Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **1 — Game Engine** | Model rules, state, actions, resolution, self-play | ✅ Complete |
| **2 — Data Pipeline** | Dataset search, log parsing, large-scale simulation, feature extraction | 🔄 In Progress |
| **3 — Belief Tracker** | Bayesian inference over hidden opponent cards | 🔜 |
| **4 — Agent Training** | CFR self-play + neural net policy | 🔜 |
| **5 — Live Interface** | Input game state → get best action recommendation | 🔜 |

### Phase 2 sub-goals

| Sub-goal | Description | Status |
|----------|-------------|--------|
| 2.1 — Dataset search | Search BGA / GitHub for real Coup game logs | ✅ Complete |
| 2.2 — Log parser & validator | Load, validate, and expose a typed API over JSON logs | ✅ Complete |
| 2.3 — Large-scale simulation | Generate 100k+ synthetic games with diverse agent configs | 🔜 Next |
| 2.4 — Feature extraction | Convert log events into flat numpy feature vectors | 🔜 |
| 2.5 — PyTorch Dataset | Wrap feature extractor in a `torch.utils.data.Dataset` | 🔜 |
| 2.6 — Data quality report | Action distributions, game lengths, win rates, bluff rates | 🔜 |

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
├── pipeline/                    # Phase 2: data pipeline
│   ├── __init__.py
│   ├── schema.py                # Field definitions, valid enum values, value constraints
│   └── log_parser.py            # RawLogLoader, LogValidator, ParsedGame, load_logs()
│
├── tests/
│   ├── test_cards_and_state.py  # Unit tests: deck, state, observations, legal actions
│   ├── test_resolution.py       # Unit tests: challenge resolution, action effects
│   ├── test_game_integration.py # Integration tests: full games, log structure, invariants
│   └── test_log_parser.py       # Unit + round-trip tests for the log parser
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

### Load and validate game logs

```python
from pipeline.log_parser import load_logs

# Load all logs from a directory — invalid logs are skipped silently
games = load_logs("data/raw_logs/")
print(f"{len(games)} games loaded")

# Strict mode — raises ValueError on any invalid log
games = load_logs("data/raw_logs/", strict=True)

# Verbose mode — prints a status line per file
games = load_logs("data/raw_logs/", verbose=True)

# Inspect a game
game = games[0]
print(game.winner_name, game.total_turns)
for action in game.action_events:
    print(action.actor_name, action.action_type, action.claimed_card)
```

### Validate a single log dict

```python
from pipeline.log_parser import validate_log

result = validate_log(my_log_dict)
print(result.is_valid)   # True / False
print(result.summary())  # Full error + warning report
```

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

There are **four test files**, each covering a distinct layer of the project. Total: **164 tests**.

```bash
# Run everything
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=coup --cov-report=term-missing
```

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

### `tests/test_log_parser.py` — Unit + round-trip tests (72 tests)

Tests the full Phase 2 parsing and validation pipeline.

| Class | What it tests |
|-------|--------------|
| `TestTopLevelValidation` | Required top-level keys; player count range (2–6); n_players/player_names consistency; non-dict input |
| `TestEventValidation` | Unknown event types; missing turn number; missing actor fields; player index out of range; invalid action types and card names; all valid enum values pass; `influence_remaining` bounds |
| `TestStateSnapshotValidation` | Wrong player count in snapshot; negative coins; invalid revealed cards; missing snapshot fields; high coin count triggers warning not error |
| `TestGameConsistency` | First event must be `game_start`; last event must be `game_end`; no action events → invalid; missing winner produces warning (not error) for turn-limit games |
| `TestRawLogLoader` | Valid file loading; malformed JSON error capture; missing file error; directory iteration skips non-JSON files; malformed files in directory don't stop iteration |
| `TestParsedGame` | `events_of_type`, property accessors, `winner_name`, `total_turns`, `len`, iteration, `iter_action_contexts`, typed field access on `ParsedEvent` |
| `TestLoadLogs` | Directory loading; single file loading; invalid logs silently skipped; strict mode raises; source not found raises; `source_path` set; malformed JSON skipped |
| `TestRoundTrip` | Every simulator-generated log in `data/raw_logs/` loads without error and passes strict validation; all `ParsedGame` objects have correct structure |

```bash
python -m pytest tests/test_log_parser.py -v
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

## Phase 2 — Implementation Notes

### Sub-goal 2.1 — Dataset search

A thorough search of BoardGameArena, GitHub, Kaggle, and academic sources confirmed that **no public Coup game log dataset exists**. The only known dataset is used internally by the official Coup mobile app and is private. Every published Coup AI project independently reaches the same conclusion and uses self-play simulation instead.

BGA does have replay data accessible via scraping, but it presents three blockers: it requires a paid account, enforces daily rate limits on replay access, and the raw HTML log format would require substantial parsing work for uncertain yield. Crucially, even scraped human games would have the same hidden-information structure as synthetic ones — opponents' hands are not revealed unless challenged — so human data offers no structural advantage over high-volume simulation at this stage.

**Decision:** generate all training data synthetically using the Phase 1 simulator with diverse agent configurations (varying `challenge_prob`, `block_prob`, player counts, and agent types). This is the standard approach for hidden-information game AI.

---

### Sub-goal 2.2 — Log parser and validator

The parser lives in `pipeline/` and is split into two files following the same zero-coupling principle as the game engine: `schema.py` holds all rule knowledge as pure data, `log_parser.py` holds all logic.

**`pipeline/schema.py`** is the single source of truth for what a valid log looks like. It defines the required fields for every event type, the set of valid enum values for cards and action types, and numeric constraints (max coins, max influence, player count range). It has zero imports from the game engine — the pipeline can be used, tested, and extended without the game library installed.

**`pipeline/log_parser.py`** is structured as three layers that compose cleanly:

`RawLogLoader` handles I/O: reading a single JSON file or iterating a directory. It returns `(dict, error_string)` tuples so every failure mode is captured without exceptions propagating. A non-JSON file in the directory is silently skipped; a malformed JSON file returns an error string but does not stop iteration. This makes batch loading robust to partial corruption.

`LogValidator` runs three tiers of checks in order, stopping early if the structure is too broken to continue. Tier 1 checks top-level keys and player count consistency. Tier 2 iterates every event and checks field presence, value types, enum membership, and player index bounds. Tier 3 checks game-level consistency: event ordering, turn monotonicity, and that a winner is recorded. The validator distinguishes errors (structural violations that make the log unusable for training) from warnings (anomalies that are legal but worth knowing about, like a game that ended at the turn limit without a winner).

`ParsedGame` is a typed wrapper that exposes convenient accessors for downstream code. The feature extractor in Sub-goal 2.4 will call `game.action_events` and `game.iter_action_contexts()` rather than scanning raw dicts. `iter_action_contexts()` is the key method: it yields each action event paired with the reactions that followed it (challenges, blocks, resolutions), which is exactly the context window needed to build labelled training examples.

`load_logs(source, strict, verbose)` is the single public entry point for all downstream consumers. In non-strict mode (the default) it silently filters out invalid logs and reports a summary count. In strict mode it raises on the first invalid log, useful in CI to catch any simulator regression.

The round-trip test class (`TestRoundTrip`) validates every log currently in `data/raw_logs/` under strict mode. This test runs as part of the normal test suite and will catch any future changes to `logger.py` that break the schema.

---

## What Phase 2 still needs (sub-goals 2.3–2.6)

- **2.3 Large-scale simulation** — run 100k+ games with mixed agent configurations and diverse player counts; save efficiently to `data/raw_logs/`
- **2.4 Feature extraction** — convert a `(ParsedGame, event_index)` pair into a flat numpy vector ready for model input
- **2.5 PyTorch Dataset** — wrap the feature extractor in `torch.utils.data.Dataset` for use in training loops
- **2.6 Data quality report** — action frequency distributions, game length histograms, win rates by seat, bluff detection rates

---

## What Phase 3 will add

- `BeliefState` class: a `(n_players, n_cards)` probability matrix updated after every observable event
- Bayesian update rules: seeing player X claim Duke reduces the prior probability that other players hold Duke
- Consistency tracking: if two players both claim Duke in a 3-player game, one must be bluffing — the belief state captures this
- The `Observation` dataclass gets a `belief_state` field (no other interface changes)
