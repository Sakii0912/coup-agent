# Coup AI Agent

A Python project to build a competitive AI agent for the card game **Coup** using Counterfactual Regret Minimization (CFR) and neural network-based opponent modelling.

---

## Project Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **1 — Game Engine** | Model rules, state, actions, resolution, self-play | ✅ Complete |
| **2 — Data Pipeline** | Dataset search, log parsing, large-scale simulation, feature extraction | ✅ Complete |
| **3 — Belief Tracker** | Bayesian inference over hidden opponent cards | 🔜 Next |
| **4 — Agent Training** | CFR self-play + neural net policy | 🔜 |
| **5 — Live Interface** | Input game state → get best action recommendation | 🔜 |

### Phase 2 sub-goals

| Sub-goal | Description | Status |
|----------|-------------|--------|
| 2.1 — Dataset search | Search BGA / GitHub for real Coup game logs | ✅ Complete |
| 2.2 — Log parser & validator | Load, validate, and expose a typed API over JSON logs | ✅ Complete |
| 2.3 — Large-scale simulation | 18-batch corpus generator across agent types, player counts, bluff rates | ✅ Complete |
| 2.4 — Feature extraction | Convert log events into 104-dim float32 numpy vectors | ✅ Complete |
| 2.5 — PyTorch Dataset | `CoupDataset` and `CoupStreamingDataset` with split, save, load | ✅ Complete |
| 2.6 — Data quality report | Action distributions, win rates, bluff rates, feature stats | ✅ Complete |

---

## Project Structure

```
coup_agent/
│
├── coup/                           # Phase 1: core game library
│   ├── __init__.py                 # Public API exports
│   ├── cards.py                    # Card/ActionType enums, game constants
│   ├── state.py                    # PlayerState, GameState, Observation dataclasses
│   ├── actions.py                  # Action/Block dataclasses, legal action generator
│   ├── resolution.py               # Challenge resolution and action effect application
│   ├── logger.py                   # Structured event logger (JSON-serialisable)
│   ├── game.py                     # Game orchestrator — full turn loop
│   └── agents/
│       ├── __init__.py
│       ├── base.py                 # Abstract Agent base class (6 decision points)
│       └── random_agent.py         # RandomAgent and HonestAgent implementations
│
├── pipeline/                       # Phase 2: data pipeline
│   ├── __init__.py
│   ├── schema.py                   # Field definitions, valid enum values, value constraints
│   ├── log_parser.py               # RawLogLoader, LogValidator, ParsedGame, load_logs()
│   ├── large_scale_sim.py          # 18-config corpus generator, ~370 games/sec
│   ├── feature_extractor.py        # ParsedGame → 104-dim float32 feature vectors + labels
│   ├── dataset.py                  # CoupDataset, CoupStreamingDataset, build_dataset()
│   └── data_quality.py             # Corpus statistics and formatted quality report
│
├── tests/
│   ├── test_cards_and_state.py     # Unit: deck, state, observations, legal actions (46)
│   ├── test_resolution.py          # Unit: challenge resolution, action effects (15)
│   ├── test_game_integration.py    # Integration: full games, log structure, invariants (31)
│   ├── test_log_parser.py          # Unit + round-trip: parser and validator (72)
│   └── test_pipeline_phase2.py     # Unit + integration: sim, features, dataset, report (70)
│
├── simulator.py                    # Single-config self-play CLI (quick runs / dev)
├── data/
│   ├── raw_logs/                   # Game log JSON files, one subdir per batch config
│   │   ├── random_2p_low/
│   │   ├── random_4p_med/
│   │   ├── honest_4p/
│   │   ├── mixed_4p/
│   │   └── …                       # 18 subdirectories total
│   ├── dataset_action.npz          # Pre-extracted CoupDataset (action prediction task)
│   └── quality_report.txt          # Latest data quality report
└── requirements.txt
```

---

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the full test suite

```bash
python -m pytest tests/ -v          # 234 tests, all should pass
python -m pytest tests/ -q          # quiet summary
python -m pytest tests/ --cov=coup --cov-report=term-missing
```

### Play a single game (verbose turn-by-turn output)

```python
from coup.game import Game
from coup.agents.random_agent import RandomAgent

agents = [RandomAgent() for _ in range(4)]
game = Game(agents=agents, seed=42, verbose=True)
log = game.play_game()
```

### Generate the full training corpus (100k games, ~5 min)

```bash
python -m pipeline.large_scale_sim --total 100000 --output_dir data/raw_logs --seed 0
python -m pipeline.large_scale_sim --total 100000 --dry_run    # preview plan only
```

### Load and inspect game logs

```python
from pipeline.log_parser import load_logs

games = load_logs("data/raw_logs/random_4p_med/")
game = games[0]
print(game.winner_name, game.total_turns)
for action in game.action_events:
    print(action.actor_name, action.action_type, action.claimed_card)
```

### Extract features and build a training dataset

```python
from pipeline.dataset import build_dataset
from torch.utils.data import DataLoader

ds = build_dataset("data/raw_logs", task="action", save_path="data/dataset_action.npz")
train, val = ds.split(val_fraction=0.1)

loader = DataLoader(train, batch_size=256, shuffle=True)
x_batch, y_batch = next(iter(loader))   # x: (256, 104)  y: (256,)
```

### Load a pre-built dataset

```python
from pipeline.dataset import CoupDataset
ds = CoupDataset.load("data/dataset_action.npz", task="action")
```

### Run the data quality report

```bash
python -m pipeline.data_quality --log_dir data/raw_logs --output data/quality_report.txt
```

---

## Running the Tests

Five test files, **234 tests** total.

```bash
python -m pytest tests/ -v
python -m pytest tests/test_cards_and_state.py -v
python -m pytest tests/test_resolution.py -v
python -m pytest tests/test_game_integration.py -v
python -m pytest tests/test_log_parser.py -v
python -m pytest tests/test_pipeline_phase2.py -v
```

### `tests/test_cards_and_state.py` — 46 tests

| Class | What it tests |
|-------|--------------|
| `TestDeck` | Deck has exactly 15 cards, 3 of each type |
| `TestGameStateInit` | Correct dealing, coin assignment, seeded reproducibility, player count validation |
| `TestPlayerState` | `lose_influence`, `swap_card`, `has_card`, `public_view` (hides hand) |
| `TestObservation` | Player sees own hand; opponents' hands hidden; JSON-serialisable; dead-player turn skip |
| `TestLegalActions` | All coin rules; must-coup at 10; Steal vs 0-coin targets; dead players excluded |
| `TestLegalBlocks` | Each action blocked by correct cards; unblockable actions return empty list |

### `tests/test_resolution.py` — 15 tests

| Class | What it tests |
|-------|--------------|
| `TestResolveChallenge` | Actor wins when card held (challenger loses, card swapped); bluffer loses when not |
| `TestApplyActionEffect` | All 7 action effects: coin changes, influence loss, steal partial transfer, Exchange hand-size invariant |

### `tests/test_game_integration.py` — 31 tests

| Class | What it tests |
|-------|--------------|
| `TestGameTermination` | Games terminate with exactly one winner for all player counts (2–6) and multiple seeds |
| `TestLogStructure` | Log has required keys; JSON-serialisable; influence loss events name the card |
| `TestInvariants` | No negative coins; no player holds >2 cards; total cards always = 15 (conservation) |
| `TestAgentVariety` | HonestAgent-only games; mixed agents; extreme challenge/block probabilities |
| `TestSimulator` | `run_simulation` returns exactly N logs; logs contain `game_end` event |

### `tests/test_log_parser.py` — 72 tests

| Class | What it tests |
|-------|--------------|
| `TestTopLevelValidation` | Required keys; player count range; n_players/player_names consistency |
| `TestEventValidation` | Unknown event types; missing fields; player index bounds; invalid enum values |
| `TestStateSnapshotValidation` | Wrong player count; negative coins; invalid revealed cards; missing snapshot fields |
| `TestGameConsistency` | First/last event ordering; no action events; missing winner warning |
| `TestRawLogLoader` | Valid file; malformed JSON; missing file; directory iteration; mixed valid/invalid |
| `TestParsedGame` | Typed accessors, `winner_name`, `total_turns`, `iter_action_contexts` |
| `TestLoadLogs` | Directory loading; strict mode; invalid logs skipped; `source_path` set |
| `TestRoundTrip` | Every simulator-generated log in `data/raw_logs/` passes strict validation |

### `tests/test_pipeline_phase2.py` — 70 tests

| Class | What it tests |
|-------|--------------|
| `TestBatchConfig` | Config list non-empty; valid player counts; unique names; positive weights |
| `TestRunLargeScale` | Dry run creates no files; game allocation sums to total; logs are valid JSON; seeded reproducibility |
| `TestFeatureConfig` | Feature dim = 104; opponent slots; smaller configs produce shorter vectors |
| `TestExtractFeatures` | Sample count matches action events; shape (104,); dtype float32; label ranges; no NaN/Inf; action label matches event; all player counts produce same-size vector |
| `TestExtractDataset` | Four arrays returned; shapes consistent; correct dtypes |
| `TestCoupDataset` | len, getitem, all three tasks, feature_dim, n_classes, normalisation, split, save/load, DataLoader |
| `TestCoupStreamingDataset` | len matches sample count; getitem; feature_dim; n_classes |
| `TestBuildDataset` | End-to-end from directory; saves .npz |
| `TestCorpusStatsAccumulation` | Game count; turn lengths; all action types seen; win totals; influence losses; player count dist |
| `TestQualityReport` | Summary is string; all sections present; all action names present; save to file; feature stats populated |
| `TestRunQualityReport` | Returns QualityReport; saves output file |
| `TestFullPipelineRoundTrip` | Raw logs → ParsedGame → features → CoupDataset end-to-end |

---

## Phase 1 — Implementation Notes

### Design philosophy

Three principles underpin the engine and carry forward to all later phases:

**1. Strict god-view / agent-view separation.** `GameState` holds all information including hidden cards. Agents never receive a `GameState` — they receive an `Observation`, the subset of information they are entitled to see. This is enforced structurally. In Phase 3, `Observation` gains a `belief_state` field and nothing else in the agent interface changes.

**2. Immutable action objects.** `Action` and `Block` are frozen dataclasses. They can be stored in logs, compared safely, and passed to multiple functions without mutation risk. Enumerating legal actions returns a plain Python list of distinct atomic actions — one per (type, target) pair.

**3. Callbacks instead of coupling.** Resolution functions (`resolve_challenge`, `apply_action_effect`) never call agent methods directly. They receive callbacks (`lose_influence_callback`, `exchange_callback`) injected by `Game`. Resolution logic stays pure and testable with simple lambdas, and Phase 4 agents slot in without touching it.

---

### `cards.py` — Enums and constants

All game rules that are pure data live here. `ACTION_CLAIMS` maps each character action to its required card. `BLOCK_MAP` maps each blockable action to the list of cards that can block it — Steal maps to `[Captain, Ambassador]` because either is a valid block claim. `DECK_COMPOSITION` is the canonical 15-card deck (3× each card) copied and shuffled at game start.

---

### `state.py` — Data structures

`PlayerState` is intentionally mutable: `hand`, `revealed`, and `coins` change during a game. `public_view()` returns `influence_count` (number of hidden cards) but omits `hand` entirely — this is all opponents can see.

`GameState.new_game()` is the only normal constructor. Passing `seed` makes the deal fully reproducible, critical for test scenarios. `advance_turn()` skips dead players while keeping their slot so indices stay stable.

`Observation` is the per-player view passed to every agent decision. It currently carries: own hand, own coins, own revealed cards, public view of all other players, deck size, current player index, and turn number. In Phase 3 a `belief_state` field will be added here with no other interface changes.

---

### `actions.py` — Legal move generation

`get_legal_actions` enforces all coin constraints and the must-coup rule. At 10+ coins the function returns *only* Coup actions — no other action is legal, not even Income. Steal only targets players with ≥ 1 coin. Every targeted action generates one `Action` per valid target, keeping actions atomic for the policy output.

`get_legal_blocks` returns one `Block` per claimable card. Bluffing a block is always legal — the challenge system handles proof.

---

### `resolution.py` — Challenge and effect logic

**Challenge resolution:** if the actor holds the claimed card they win — the challenger loses an influence and the actor swaps their proved card for a fresh draw (removing the information gain). If the actor was bluffing, the challenger wins and the actor loses an influence.

**`apply_action_effect`** is called only after all reactions resolve in the action's favour. The Assassinate coin cost (3) is deducted in `game.py` when declared, not here — coins are spent on the attempt regardless of outcome. Exchange draws up to 2 cards, lets the actor choose which to keep, and reshuffles the deck after returning unchosen cards to prevent deck composition inference.

---

### `game.py` — Turn orchestrator

`_play_turn()` implements the reactive sequence: declare action → optional challenge → optional block → optional block challenge → apply effect. Players are polled in seat order; the first challenger or blocker stops the loop. `MAX_TURNS = 500` guards against infinite loops — in practice random games rarely exceed 80 turns.

---

### `agents/` — Agent architecture

`Agent` (abstract base class) defines six decision points:

| Method | Called when |
|--------|-------------|
| `choose_action` | It is your turn |
| `choose_to_challenge_action` | Another player declared a character action |
| `choose_to_block` | An action targeting you (or anyone, for Foreign Aid) was declared |
| `choose_to_challenge_block` | Someone claimed to block an action |
| `choose_card_to_lose` | You must reveal an influence card |
| `choose_exchange_cards` | You played Ambassador Exchange |

`RandomAgent` implements all six uniformly at random with configurable `challenge_prob` and `block_prob`. `HonestAgent` extends it: only plays actions it actually holds cards for, only blocks with cards it holds, and challenges more often when it already holds the claimed card.

---

## Phase 2 — Implementation Notes

### Sub-goal 2.1 — Dataset search

A thorough search of BoardGameArena, GitHub, Kaggle, and academic sources confirmed **no public Coup game log dataset exists**. BGA replay data is accessible via scraping but requires a paid account, enforces daily rate limits, and even scraped human games have the same hidden-information structure — opponents' hands are not revealed unless challenged. Human data offers no structural advantage over high-volume simulation at this stage.

**Decision:** generate all training data synthetically with diverse agent configurations. This is the standard approach for hidden-information game AI.

---

### Sub-goal 2.2 — Log parser and validator

`pipeline/schema.py` is the single source of truth for valid log structure. It has zero imports from the game engine — the pipeline can run standalone.

`pipeline/log_parser.py` has three layers:

`RawLogLoader` handles I/O and returns `(dict, error_string)` tuples so every failure is captured without propagating exceptions.

`LogValidator` runs three tiers: top-level structure → per-event field presence and value ranges → game-level consistency. It distinguishes errors (make the log unusable) from warnings (anomalies that are legal, e.g. turn-limit games with no winner).

`ParsedGame` exposes typed accessors (`action_events`, `winner_name`, `iter_action_contexts()`) used directly by the feature extractor. `iter_action_contexts()` yields each action paired with its subsequent reactions — the exact context window for building labelled training examples.

`load_logs(source, strict, verbose)` is the single public entry point. Strict mode raises on the first invalid log; non-strict silently filters and reports a count.

---

### Sub-goal 2.3 — Large-scale simulation

`pipeline/large_scale_sim.py` runs 18 named `BatchConfig`s covering 3 challenge/block probability tiers (low/med/high) × 3 player counts (2/4/6) × 2 agent types (Random/Honest), plus one mixed-seat batch. Games are distributed proportionally by weight; each config writes to its own subdirectory under `data/raw_logs/`. Agent RNGs are seeded deterministically from `game_seed × 100 + seat` so runs are fully reproducible. Throughput: ~370 games/sec; 100k games takes ~4.5 minutes.

**Observed corpus statistics (2,200-game sample):**
- Average game length: 17 turns (median 15, p95 35)
- Most common action: Steal (24.4%), least: Coup (3.2%)
- Challenge rate: 31.7% of actions; bluff rate: ~63%
- Win rates by seat: roughly uniform (22–27%), no significant first-mover advantage

---

### Sub-goal 2.4 — Feature extraction

`pipeline/feature_extractor.py` converts each action event in a `ParsedGame` into a 104-dim float32 vector. Fixed size regardless of player count (padded to 6-player maximum).

| Section | Dims | Description |
|---------|------|-------------|
| My state | 12 | Revealed cards (×5), coins (÷12), influence (÷2), revealed proxy |
| Opponents (×5 slots) | 40 | Per slot: influence, coins, revealed cards (×5), is_alive; zero-padded for absent players |
| Action one-hot | 7 | Which of 7 action types was declared |
| Claimed card | 6 | One-hot + "no claim" dim |
| Meta scalars | 4 | target_is_me, actor_is_me, turn (÷500), table coins (÷72) |
| Action history | 35 | One-hot counts of last 5 action types seen this game |

Three label targets per sample: `action_label` (7-class), `target_label` (6-class including no-target), `outcome` (binary: did the action resolve?). A 100k-game corpus produces ~1.4 million training samples.

---

### Sub-goal 2.5 — PyTorch Dataset

`pipeline/dataset.py` provides two Dataset classes:

`CoupDataset` wraps pre-extracted numpy arrays. It applies z-score normalisation (fitted on training split only), supports `.split(val_fraction)` with consistent normalisation stats across splits, and serialises to/from `.npz` via `.save()` / `.load()`. Supports all three task heads (`"action"`, `"target"`, `"outcome"`) via the `task` parameter.

`CoupStreamingDataset` extracts features on-the-fly from `ParsedGame` objects — useful during development when the full feature matrix doesn't fit in memory.

`build_dataset(log_dir, task, save_path)` is the one-call end-to-end factory: loads all logs recursively → extracts features → returns a normalised `CoupDataset`, optionally saving to disk.

---

### Sub-goal 2.6 — Data quality report

`pipeline/data_quality.py` walks the full corpus once and computes: action frequency distribution, win rates by seat, challenge rate and bluff detection rate per action type, block rate and hold rate per action type, influence loss breakdown by card, and feature-level statistics (mean, std, sparsity). Output is a formatted text report. Feature sparsity of ~78% is expected for this one-hot-heavy encoding.

```bash
python -m pipeline.data_quality --log_dir data/raw_logs --output data/quality_report.txt
```

---

## What Phase 3 will add

Phase 3 builds the `BeliefState` — a probability distribution over each opponent's hidden cards, updated after every observable event.

- `BeliefState` class: a `(n_players, n_cards)` probability matrix with Bayesian update rules
- Update triggers: action claims (seeing a Duke claim reduces Duke probability for others), challenge results (a lost challenge proves the actor held the card), influence losses (revealed cards are removed from the distribution), Ambassador exchanges (hand composition shifts without revealing cards)
- Consistency tracking: if two players claim Duke in a 3-player game, one must be bluffing — the belief state captures this tension
- The `Observation` dataclass gains a `belief_state` field; no other agent interface changes
- The feature vector in Phase 2.4 will be extended with belief state columns, increasing from 104 to ~154 dims
ENDOFREADME
