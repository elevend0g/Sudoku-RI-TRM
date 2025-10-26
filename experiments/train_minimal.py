"""
Minimal training experiment for RI-TRM Sudoku.

Train on 100 tasks to validate the complete system works.

Expected results (from Sudoku.md):
- Loss decreases
- Can fix some violations
- Path memory grows to ~50-100 paths
- Some tasks solve completely
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from tqdm import tqdm

from src.rules import SudokuRuleGraph
from src.dataset import RuleBasedSudokuDataset
from src.path_memory import HebbianPathMemory
from src.action_network import create_action_network
from src.action_solver import ActionBasedSolver
from src.action_trainer import ActionBasedTrainer


def main():
    """
    Minimal training experiment.

    Train RI-TRM on 100 synthetic Sudoku tasks.
    """
    print("=" * 60)
    print("RI-TRM SUDOKU - MINIMAL TRAINING EXPERIMENT")
    print("=" * 60)
    print()

    # Configuration
    NUM_TASKS = 10000
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Configuration:")
    print(f"  Tasks: {NUM_TASKS}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Device: {DEVICE}")
    print()

    # Initialize components
    print("Initializing RI-TRM components...")
    rule_graph = SudokuRuleGraph()
    dataset = RuleBasedSudokuDataset(
        rule_graph,
        num_tasks=NUM_TASKS,
        violation_rate=0.7,  # 70% with violations for training
        random_seed=42
    )
    path_memory = None
    network = create_action_network()

    print(f"  Network parameters: {sum(p.numel() for p in network.parameters() if p.requires_grad):,}")
    print()

    # Create solver and trainer
    solver = ActionBasedSolver(network, rule_graph)
    trainer = ActionBasedTrainer(solver, dataset, rule_graph, lr=LEARNING_RATE)

    print("Components initialized!")
    print()

    # Training loop
    print("Starting training...")
    print("-" * 60)

    epoch_losses = []
    epoch_violations = []

    for epoch in range(NUM_EPOCHS):
        stats = trainer.train_epoch(num_tasks=NUM_TASKS)
        epoch_loss = stats['loss']
        avg_viols = stats['avg_violations_reduced']

        epoch_losses.append(epoch_loss)
        epoch_violations.append(avg_viols)

        print(f"  Epoch {epoch+1}: Loss={epoch_loss:.4f}, Avg Violations Reduced={avg_viols:.2f}")

    print()
    print("-" * 60)
    print("Training complete!")
    print()

    # Results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    print(f"Final loss: {epoch_losses[-1]:.4f}")
    print(f"Loss reduction: {epoch_losses[0] - epoch_losses[-1]:.4f}")
    print(f"  ({(epoch_losses[0] - epoch_losses[-1]) / (epoch_losses[0] + 1e-9) * 100:.1f}% decrease)")
    print()

    # Test on validation set
    print("Testing on validation set...")
    dataset_val = RuleBasedSudokuDataset(
        rule_graph,
        num_tasks=20,
        violation_rate=0.7,
        random_seed=999
    )

    solver.epsilon = 0.0 # No exploration

    solved_count = 0
    improved_count = 0

    for task in tqdm(list(dataset_val), desc="Validation"):
        grid = task["grid"]
        initial_violations = len(task["violations"])

        if initial_violations == 0:
            continue

        # Solve
        result, trace, solved = solver.solve(grid)

        final_violations = len(rule_graph.verify(result))

        if solved:
            solved_count += 1
        elif final_violations < initial_violations:
            improved_count += 1

    print()
    print(f"Validation results:")
    print(f"  Solved completely: {solved_count}/20")
    print(f"  Improved: {improved_count}/20")
    print()

    # Show example reasoning trace
    print("=" * 60)
    print("EXAMPLE REASONING TRACE")
    print("=" * 60)
    print()

    # Get a task with violations
    example_grid = None
    example_task = None
    for task in list(dataset_val):
        if len(task["violations"]) > 0:
            example_grid = task["grid"]
            example_task = task
            break

    if example_grid is not None:
        result, trace, solved = solver.solve(example_grid, verbose=True)

        print(f"Initial violations: {len(example_task['violations'])}")
        print(f"Final violations: {len(rule_graph.verify(result))}")
        print(f"Solved: {solved}")
        print()

        if trace:
            print("Reasoning steps:")
            for step in trace[:5]:  # Show first 5 steps
                print(f"  {step}")
    else:
        print("No tasks with violations in validation set.")

    print()
    print("=" * 60)
    print("Experiment complete!")
    print("=" * 60)

    # Save model
    torch.save({
        'network_state_dict': network.state_dict(),
        'epoch_losses': epoch_losses,
    }, 'experiments/ri_trm_final.pt')

    print()
    print("Model saved to: experiments/ri_trm_final.pt")


if __name__ == "__main__":
    main()
