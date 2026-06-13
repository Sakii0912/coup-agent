# Coup AI Agent

A modular Python project for building Coup-playing agents with:
- a full game engine and simulator,
- a data pipeline for large synthetic corpora,
- probabilistic belief tracking for hidden information,
- CFR and neural training workflows,
- evaluation tooling,
- and a live recommendation engine for real games.

## Current Status

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Core game engine and baseline agents | Complete |
| Phase 2 | Data pipeline and large-scale simulation | Complete |
| Phase 3 | Belief state + opponent modelling | Complete |
| Phase 4 | CFR/Neural training + evaluation | Complete |
| Phase 5 | Live recommendation pipeline | Complete |

## Project Layout

```text
coup-agent/
├── coup/                  # Core game engine (rules, state, resolution, game loop)
│   ├── actions.py
│   ├── cards.py
│   ├── game.py
│   ├── resolution.py
│   ├── state.py
│   └── agents/
│       ├── base.py
│       ├── random_agent.py
│       ├── crf_agent.py   # currently empty stub
│       ├── nn_agent.py    # currently empty stub
│       └── rule_agent.py  # currently empty stub
├── pipeline/              # Log parsing, feature extraction, datasets, quality checks
├── belief/                # BeliefState, updates, normalisation, opponent model
├── agents/
│   ├── cfr/               # CFR solver + table + runtime CFRAgent
│   └── neural/            # CoupPolicyNet + NeuralAgent + supervised trainer
├── training/              # Unified training CLI + CFR/RL trainers
├── evaluation/            # Match and tournament evaluation tooling
├── live/                  # Live state input + recommender engine
├── tests/                 # Unit and integration tests
├── simulator.py           # Simple simulation CLI
├── requirements.txt
└── data/                  # Logs, datasets, models, reports
```

## Installation

### 1) Create environment and install base dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` currently includes test/core numeric dependencies (`pytest`, `pytest-cov`, `numpy`).

### 2) Install optional ML dependency for neural training/evaluation

```bash
pip install torch
```

Neural components in `agents/neural/`, `training rl`, and optional neural tournament entrants require `torch`.

## Agent Types

### Baseline engine agents (`coup/agents/`)

- `RandomAgent`: uniformly random legal decisions with configurable challenge/block probabilities.
- `HonestAgent`: constrained random policy that avoids bluffing where possible.

### Trained/runtime agents (`agents/`)

- `CFRAgent` (`agents/cfr/cfr_agent.py`): plays using average strategy from a trained CFR `StrategyTable`.
- `NeuralAgent` (`agents/neural/neural_agent.py`): wraps `CoupPolicyNet` checkpoint with Agent interface methods.

## Quickstart Workflows

### Run a quick simulation

```bash
python simulator.py --n_games 1000 --n_players 4 --agent random --output_dir data/raw_logs
```

Agent choices:
- `random`
- `honest`
- `mixed`

Useful flags:
- `--challenge_prob`
- `--block_prob`
- `--seed`

### Generate a large corpus (Phase 2)

```bash
python -m pipeline.large_scale_sim --total 100000 --output_dir data/raw_logs --seed 0
```

Preview batch allocation without running:

```bash
python -m pipeline.large_scale_sim --total 100000 --dry_run
```

### Build dataset from logs

```python
from pipeline.dataset import build_dataset

ds = build_dataset(
    log_dir="data/raw_logs",
    task="action",
    save_path="data/dataset_action.npz",
)
print(ds)
```

### Run quality report

```bash
python -m pipeline.data_quality --log_dir data/raw_logs --output data/quality_report.txt
```

## Training (Phase 4)

Unified entrypoint:

```bash
python -m training.train <mode> [args]
```

Modes:

### 1) CFR self-play

```bash
python -m training.train cfr --n_iter 10000 --n_players 4 --output_dir data/models/cfr
```

Resume from checkpoint:

```bash
python -m training.train cfr --resume data/models/cfr/latest.pkl.gz --n_iter 5000
```

Important CFR flags:
- `--n_iter`
- `--n_players`
- `--save_every`
- `--log_every`
- `--max_depth`
- `--vanilla_cfr` (default is CFR+)

### 2) Supervised neural pretraining

```bash
python -m training.train pretrain \
  --dataset data/dataset_action.npz \
  --epochs 30 \
  --output_dir data/models
```

Important pretrain flags:
- `--dataset`
- `--epochs`
- `--batch_size`
- `--lr`
- `--hidden_dim`
- `--patience`

### 3) RL fine-tuning (REINFORCE self-play)

```bash
python -m training.train rl \
  --checkpoint data/models/best_model.pt \
  --rounds 100 \
  --games_per_round 64 \
  --output_dir data/models/rl
```

Important RL flags:
- `--checkpoint`
- `--rounds`
- `--games_per_round`
- `--n_players`
- `--lr`
- `--no_belief` (use base features without belief augmentation)

## Evaluation CLI (Phase 4)

Entrypoint:

```bash
python -m evaluation.evaluate <mode> [args]
```

### Quick head-to-head match

```bash
python -m evaluation.evaluate match \
  --subject random \
  --opponent honest \
  --n_games 200 \
  --n_players 4 \
  --output data/eval_match.txt
```

`match` supports subject/opponent baselines:
- `random`
- `honest`
- `random-low`
- `random-high`

### Full tournament

```bash
python -m evaluation.evaluate tournament \
  --n_games 200 \
  --n_players 4 \
  --neural_ckpt data/models/best_model.pt \
  --rl_ckpt data/models/rl/rl_latest.pt \
  --cfr_ckpt data/models/cfr/latest.pkl.gz \
  --output data/evaluation_report.txt
```

Outputs:
- text report (`.txt`)
- JSON report alongside text file path (`.json`)

## Live Recommendations (Phase 5)

`live/recommender.py` provides recommendation APIs for real-table usage.
There is currently no dedicated standalone CLI script in `live/`; the intended interface is Python API integration.

### Action recommendation example

```python
from live.recommender import Recommender
from live.state_input import LiveGameState, MyState, OpponentState
from coup.cards import Card

state = LiveGameState(
    n_players=4,
    my_state=MyState(idx=0, name="Me", hand=[Card.DUKE, Card.CAPTAIN], coins=2, revealed=[]),
    opponents=[
        OpponentState(name="P1", idx=1, coins=2, influence_count=2, revealed_cards=[], is_alive=True),
        OpponentState(name="P2", idx=2, coins=3, influence_count=2, revealed_cards=[], is_alive=True),
        OpponentState(name="P3", idx=3, coins=1, influence_count=2, revealed_cards=[], is_alive=True),
    ],
    current_turn=0,
    turn_number=5,
    decision="action",
)

rec = Recommender(agent=None, top_k=3)  # heuristic-only mode
for r in rec.recommend_action(state):
    print(r.display())
```

### Reaction recommendations

- `recommend_reaction(state, decision="challenge")`
- `recommend_reaction(state, decision="block")`
- `recommend_reaction(state, decision="challenge_block")`
- `recommend_card_to_lose(state)`
- `belief_summary(state)`

Internally this uses:
1. `BeliefTracker` updates,
2. `Observation` enrichment,
3. feature extraction,
4. agent/heuristic scoring,
5. ranked action reasoning output.

## Using Different Agents in a Game

### Baselines only

```python
from coup.game import Game
from coup.agents.random_agent import RandomAgent, HonestAgent

agents = [
    RandomAgent(challenge_prob=0.2, block_prob=0.3),
    HonestAgent(challenge_prob=0.35, block_prob=0.3),
    RandomAgent(),
    HonestAgent(),
]

game = Game(agents=agents, seed=42, verbose=False)
log = game.play_game()
print("winner:", log.get("winner_name"))
```

### Mix in Neural and CFR agents

```python
from coup.game import Game
from coup.agents.random_agent import RandomAgent
from agents.neural.neural_agent import NeuralAgent
from agents.cfr.strategy_table import StrategyTable
from agents.cfr.cfr_agent import CFRAgent

neural = NeuralAgent.from_checkpoint("data/models/best_model.pt")
table = StrategyTable.load("data/models/cfr/latest.pkl.gz")
cfr = CFRAgent(table=table, n_players=4)

agents = [neural, cfr, RandomAgent(), RandomAgent()]
game = Game(agents=agents, seed=1)
log = game.play_game()
```

## Test Suite

Run everything:

```bash
python -m pytest tests/ -v
```

Run targeted areas:

```bash
python -m pytest tests/test_cfr.py -v
python -m pytest tests/test_training.py -v
python -m pytest tests/test_neural_agent.py -v
python -m pytest tests/test_belief_tracker.py -v
python -m pytest tests/test_observation_integration.py -v
```

## CLI Summary

| Command | Purpose |
|---|---|
| `python simulator.py ...` | Single-mode game simulation and log generation |
| `python -m pipeline.large_scale_sim ...` | Multi-batch large-scale corpus generation |
| `python -m pipeline.data_quality ...` | Corpus quality and feature diagnostics |
| `python -m training.train cfr ...` | CFR self-play training |
| `python -m training.train pretrain ...` | Supervised neural pretraining |
| `python -m training.train rl ...` | RL fine-tuning from checkpoint |
| `python -m evaluation.evaluate match ...` | Head-to-head evaluation |
| `python -m evaluation.evaluate tournament ...` | Round-robin tournament and ranking |

## Notes and Caveats

- `coup/agents/crf_agent.py`, `coup/agents/nn_agent.py`, and `coup/agents/rule_agent.py` are currently empty placeholders.
- CFR runtime agent is implemented in `agents/cfr/cfr_agent.py`.
- Neural runtime agent is implemented in `agents/neural/neural_agent.py`.
- For reproducibility, most training/simulation entry points accept a `--seed` argument.

