#!/usr/bin/env python3
"""
Demonstration script that shows the RI-TRM model solving 10 random hard puzzles.
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
    Demonstration script.
    """
    print("=" * 60)
    print("RI-TRM - 10 HARD PUZZLES DEMO")
    print("=" * 60)
    print()

    # Configuration
    NUM_PUZZLES = 10
    NUM_HOLES = 20  # Partially filled puzzles
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Create puzzles
    print(f"Generating {NUM_PUZZLES} hard puzzles...")
    rule_graph = SudokuRuleGraph()
    solver_gen = SudokuSolver(rule_graph)
    puzzles = []
    for _ in range(NUM_PUZZLES):
        grid = np.zeros((9, 9), dtype=int)
        solved_grid = solver_gen.solve(grid)
        if solved_grid is None:
            continue

        puzzle = solved_grid.flatten().copy()
        holes = np.random.choice(81, NUM_HOLES, replace=False)
        puzzle[holes] = 0
        puzzles.append(puzzle.reshape(9, 9))
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
    solver = ActionBasedSolver(network, rule_graph)

    # Solve puzzles
    print(f"Solving {NUM_PUZZLES} hard puzzles...")
    solved_count = 0
    for i, puzzle in enumerate(puzzles):
        print("-" * 60)
        print(f"Puzzle {i+1}/{NUM_PUZZLES}")
        print("Initial puzzle:")
        print(puzzle)
        print()

        result, trace, solved = solver.solve(puzzle)

        print("Final grid:")
        print(result)
        print()
        print(f"Solved: {solved}")

        if solved:
            solved_count += 1
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"Solved {solved_count}/{NUM_PUZZLES} hard puzzles.")
    print()


if __name__ == "__main__":
    main()
