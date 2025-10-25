#!/usr/bin/env python3
"""
Demo of action-based RI-TRM solver on a fixable violation.
"""

import numpy as np
import torch
from src.rules import SudokuRuleGraph
from src.action_network import create_action_network
from src.action_solver import ActionBasedSolver

def main():
    print("Action-Based RI-TRM Demo on Fixable Violation")
    print("=" * 60)

    # Create components
    rule_graph = SudokuRuleGraph()
    network = create_action_network(hidden_size=512, num_heads=8, num_layers=2)
    
    # Load the trained model
    try:
        network.load_state_dict(torch.load('experiments/trained_model_minimal.pt')['network_state_dict'])
        print("Loaded trained model from experiments/trained_model_minimal.pt")
    except FileNotFoundError:
        print("No trained model found. Please run experiments/train_minimal.py first.")
        return

    solver = ActionBasedSolver(network, rule_graph, epsilon=0.0) # No exploration

    # Test grid with a fixable violation
    # The violation is at (0, 1), where a 3 is placed, but there is already a 3 in the same row.
    # The cell (0, 1) is not an original clue.
    test_grid = np.zeros((9, 9), dtype=int)
    test_grid[0, 0] = 3
    test_grid[0, 5] = 5
    original_clues = (test_grid != 0)
    test_grid[0, 1] = 3 # This is the fixable violation
    

    print("\nGrid with fixable violation:")
    print(test_grid)

    print("\nSolving...")
    result, trace, solved = solver.solve(test_grid, original_clues=original_clues, verbose=True)
    
    print("\nFinal grid:")
    print(result)
    print(f"\nSolved: {solved}")


if __name__ == "__main__":
    main()
