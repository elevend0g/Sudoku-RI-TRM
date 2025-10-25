"""
Evaluation script for the RI-TRM model.

Evaluate the trained RI-TRM model on a test set of hard Sudoku puzzles.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from tqdm import tqdm

from src.rules import SudokuRuleGraph
from src.dataset import SudokuSolver
from src.action_network import create_action_network
from src.action_solver import ActionBasedSolver

def main():
    """
    Evaluation script for the RI-TRM model.
    """
    print("=" * 60)
    print("RI-TRM - EVALUATION")
    print("=" * 60)
    print()

    # Configuration
    NUM_TEST_SAMPLES = 1000
    NUM_HOLES = 60  # Hard puzzles
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Configuration:")
    print(f"  Test Samples: {NUM_TEST_SAMPLES}")
    print(f"  Holes: {NUM_HOLES}")
    print(f"  Device: {DEVICE}")
    print()

    # Create test set
    print("Creating test set...")
    rule_graph = SudokuRuleGraph()
    solver_gen = SudokuSolver(rule_graph)
    test_puzzles = []
    for _ in tqdm(range(NUM_TEST_SAMPLES), desc="Generating puzzles"):
        grid = np.zeros((9, 9), dtype=int)
        solved_grid = solver_gen.solve(grid)
        if solved_grid is None:
            continue

        puzzle = solved_grid.flatten().copy()
        holes = np.random.choice(81, NUM_HOLES, replace=False)
        puzzle[holes] = 0
        test_puzzles.append(puzzle.reshape(9, 9))
    print()

    # Load trained model
    print("Loading trained RI-TRM model...")
    network = create_action_network()
    try:
        network.load_state_dict(torch.load('experiments/ri_trm_final.pt')['network_state_dict'])
    except FileNotFoundError:
        print("Trained model not found. Please run experiments/train_minimal.py first.")
        return
    network.to(DEVICE)
    network.eval()
    print()

    # Create solver
    solver = ActionBasedSolver(network, rule_graph, epsilon=0.0) # No exploration

    # Evaluate
    print("Evaluating model on the test set...")
    solved_count = 0
    failed_puzzles = []
    for puzzle in tqdm(test_puzzles, desc="Evaluating"):
        result, trace, solved = solver.solve(puzzle)
        if solved:
            solved_count += 1
        else:
            failed_puzzles.append((puzzle, result, trace))
    print()

    # Results
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print()

    accuracy = solved_count / NUM_TEST_SAMPLES
    print(f"Accuracy (pass@1): {accuracy:.2%}")
    print()

    # Show reasoning trace for a failed puzzle
    if failed_puzzles:
        print("=" * 60)
        print("EXAMPLE FAILED PUZZLE - REASONING TRACE")
        print("=" * 60)
        print()

        puzzle, result, trace = failed_puzzles[0]
        initial_violations = len(rule_graph.verify(puzzle))
        final_violations = len(rule_graph.verify(result))

        print(f"Initial puzzle:")
        print(puzzle)
        print()
        print(f"Initial violations: {initial_violations}")
        print(f"Final violations: {final_violations}")
        print()

        if trace:
            print("Reasoning steps:")
            for step in trace[:10]:  # Show first 10 steps
                print(f"  {step}")
        else:
            print("No reasoning trace available.")

    print()
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
