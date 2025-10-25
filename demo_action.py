#!/usr/bin/env python3
"""
Demo of action-based RI-TRM solver.
"""

import numpy as np
from src.rules import SudokuRuleGraph
from src.dataset import RuleBasedSudokuDataset
from src.action_network import create_action_network
from src.action_solver import ActionBasedSolver
from src.action_trainer import ActionBasedTrainer


def main():
    print("Action-Based RI-TRM Demo")
    print("=" * 60)

    # Create components
    rule_graph = SudokuRuleGraph()
    network = create_action_network(hidden_size=512, num_heads=8, num_layers=2)
    solver = ActionBasedSolver(network, rule_graph, epsilon=0.3)

    # Test grid
    test_grid = np.zeros((9, 9), dtype=int)
    test_grid[0, 0] = 3
    test_grid[0, 5] = 3  # Violation

    print("\nBefore training:")
    result, trace, solved = solver.solve(test_grid, verbose=True)
    print(f"Solved: {solved}")

    # Create dataset
    dataset = RuleBasedSudokuDataset(rule_graph, num_tasks=100)

    # Train
    print("\nTraining...")
    trainer = ActionBasedTrainer(solver, dataset, rule_graph)

    for epoch in range(10):
        stats = trainer.train_epoch(num_tasks=20)
        print(f"Epoch {epoch}: loss={stats['loss']:.3f}, "
              f"reduced={stats['avg_violations_reduced']:.2f}, "
              f"success={stats['success_rate']:.1%}, "
              f"ε={stats['epsilon']:.3f}")

    # Test after training
    print("\nAfter training:")
    solver.epsilon = 0.0  # No exploration
    result, trace, solved = solver.solve(test_grid, verbose=True)
    print(f"Solved: {solved}")


if __name__ == "__main__":
    main()
