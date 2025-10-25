#!/usr/bin/env python3
"""
Test with a fixable violation (where network can actually edit cells).
"""

import numpy as np
from src.rules import SudokuRuleGraph
from src.network import create_sudoku_network
from src.solver import RecursiveSolver

def print_grid(grid, title="Grid"):
    """Print a Sudoku grid."""
    print(f"\n{title}:")
    for i in range(9):
        if i > 0 and i % 3 == 0:
            print("  " + "-" * 21)
        row_str = "  "
        for j in range(9):
            val = grid[i, j]
            row_str += ("·" if val == 0 else str(val)) + " "
            if (j + 1) % 3 == 0:
                row_str += " "
        print(row_str.rstrip())
    print()


def main():
    print("=" * 70)
    print("FIXABLE VIOLATION TEST")
    print("=" * 70)

    # Create test grid where violation involves a non-clue cell
    # This simulates a partially-solved grid where the network made an error
    grid = np.zeros((9, 9), dtype=int)

    # Set up some original clues
    grid[0, 0] = 3  # Original clue
    grid[1, 1] = 5  # Original clue
    grid[2, 2] = 7  # Original clue

    # Network previously filled these cells (simulate partially solved state)
    # But it made an error - duplicate 3 in row 0
    grid[0, 5] = 3  # ERROR: duplicates (0,0)

    # Mark which cells are original clues vs filled by network
    original_grid = np.zeros((9, 9), dtype=int)
    original_grid[0, 0] = 3
    original_grid[1, 1] = 5
    original_grid[2, 2] = 7
    # Note: (0,5) is NOT in original_grid, so it can be edited

    print_grid(grid, "Current Grid (network made an error)")
    print("Original clues: (0,0)=3, (1,1)=5, (2,2)=7")
    print("Network filled: (0,5)=3 ← ERROR! Duplicates (0,0)")

    rule_graph = SudokuRuleGraph()
    violations = rule_graph.verify(grid)

    print(f"\nViolations: {len(violations)}")
    for v in violations:
        print(f"  - {v.type}: value {v.conflicting_values[0]} at cells {v.cells}")

    # Create solver - but use original_grid as the input
    # This way, the solver knows (0,5) is NOT an original clue
    network = create_sudoku_network()
    solver = RecursiveSolver(
        network=network,
        rule_graph=rule_graph,
        path_memory=None,
        max_iterations=10,  # More iterations
        reasoning_steps=5
    )

    print("\n" + "=" * 70)
    print("RUNNING SOLVER...")
    print("=" * 70)
    print("Cell (0,5) CAN be edited (not an original clue)")
    print("Expected: Network might change (0,5) to a different value")

    # Use original_grid so solver knows what can be edited
    result = solver.solve(original_grid, return_trace=True)

    print_grid(result.grid, "Final Grid")

    print(f"\nFinal violations: {len(result.violations)}")
    if result.violations:
        for v in result.violations:
            print(f"  - {v.type}: value {v.conflicting_values[0]}")

    # Check editable cell
    if result.grid[0, 5] != original_grid[0, 5]:
        print(f"\n✅ Cell (0,5) was edited: {original_grid[0,5]} → {result.grid[0,5]}")
    else:
        print(f"\n⚠️  Cell (0,5) was NOT edited (network too conservative)")

    # Check original clues preserved
    original_clues = (original_grid != 0)
    clues_preserved = np.all(result.grid[original_clues] == original_grid[original_clues])
    print(f"Original clues preserved: {clues_preserved} {'✅' if clues_preserved else '❌'}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if len(result.violations) < len(violations):
        print("✅ SUCCESS: Violations reduced!")
    elif len(result.violations) == len(violations):
        print("⚠️  STABLE: Violations unchanged")
        print("   This is expected for untrained network")
        print("   Network needs training to learn how to fix violations")
    else:
        print("❌ FAILURE: Violations increased")

    print("\nNote: An untrained network won't know how to fix violations.")
    print("After training, it should learn to change (0,5) to a non-3 value.")


if __name__ == "__main__":
    main()
