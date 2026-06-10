"""
agents/neural — Neural policy agent (sub-goal 4.2).

  model.py        CoupPolicyNet  — shared trunk + action/target/binary heads
  trainer.py      SupervisedTrainer — pre-training on Phase 2 dataset
  neural_agent.py NeuralAgent    — Agent ABC wrapper around the policy net

Quick-start:

    # Pre-train from the Phase 2 dataset
    from agents.neural.trainer import SupervisedTrainer
    trainer = SupervisedTrainer(dataset_path="data/dataset_action.npz",
                                save_dir="data/models/")
    trainer.train(n_epochs=30)

    # Load the trained agent
    from agents.neural.neural_agent import NeuralAgent
    agent = NeuralAgent.from_checkpoint("data/models/best_model.pt")

    # Use it in a game
    from coup.game import Game
    agents = [agent] + [RandomAgent() for _ in range(3)]
    game = Game(agents, seed=42)
    log = game.play_game()
"""

from .model import CoupPolicyNet
from .neural_agent import NeuralAgent

__all__ = ["CoupPolicyNet", "NeuralAgent"]
