#!/usr/bin/env python3
"""
Demonstration of the violation explosion fix.

This script shows before/after behavior with detailed output.
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
    print("VIOLATION EXPLOSION FIX DEMONSTRATION")
    print("=" * 70)

    # Create test grid with simple violation
    grid = np.zeros((9, 9), dtype=int)
    grid[0, 0] = 3  # Original clue
    grid[0, 5] = 3  # Violation: duplicate 3 in row 0

    print_grid(grid, "Initial Grid (1 violation)")

    rule_graph = SudokuRuleGraph()
    violations = rule_graph.verify(grid)

    print(f"Initial violations: {len(violations)}")
    for v in violations:
        print(f"  - {v.type}: value {v.conflicting_values[0]} at cells {v.cells}")

    # Create solver
    network = create_sudoku_network()
    solver = RecursiveSolver(
        network=network,
        rule_graph=rule_graph,
        path_memory=None,
        max_iterations=5,
        reasoning_steps=3
    )

    print("\n" + "=" * 70)
    print("RUNNING SOLVER WITH FIXES...")
    print("=" * 70)

    result = solver.solve(grid, return_trace=True, update_memory=False)

    print_grid(result.grid, "Final Grid")

    print(f"\nFinal violations: {len(result.violations)}")
    if result.violations:
        for v in result.violations:
            print(f"  - {v.type}: value {v.conflicting_values[0]}")

    print(f"\nSteps taken: {result.num_steps}")
    print(f"Converged: {result.converged}")
    print(f"Success: {result.success}")

    # Check that original clues are preserved
    original_clues = (grid != 0)
    clues_preserved = np.all(result.grid[original_clues] == grid[original_clues])
    print(f"\nOriginal clues preserved: {clues_preserved}")

    if not clues_preserved:
        print("❌ ERROR: Original clues were modified!")
        changed = np.argwhere(grid != result.grid)
        for pos in changed:
            row, col = pos
            if grid[row, col] != 0:
                print(f"  Cell ({row},{col}): {grid[row,col]} → {result.grid[row,col]}")
    else:
        print("✅ SUCCESS: Original clues intact!")

    # Count how many cells changed
    changes = np.sum(grid != result.grid)
    print(f"\nCells changed: {changes}/81")

    print("\n" + "=" * 70)
    print("REASONING TRACE")
    print("=" * 70)
    if result.trace:
        for step in result.trace[:5]:
            print(f"\nStep {step.step}:")
            print(f"  Violations: {step.violation_count}")
            print(f"  Confidence: {step.confidence:.3f}")
            if step.action_taken:
                print(f"  Action: {step.action_taken}")

    print("\n" + "=" * 70)
    print("KEY METRICS")
    print("=" * 70)
    print(f"✓ No explosion: {len(result.violations) <= len(violations)}")
    print(f"✓ Clues safe: {clues_preserved}")
    print(f"✓ Minimal edits: {changes <= 10}")

    if len(result.violations) > len(violations):
        print("\n❌ WARNING: Violations INCREASED!")
        print(f"   {len(violations)} → {len(result.violations)} (WORSE)")
    elif len(result.violations) < len(violations):
        print("\n✅ Violations DECREASED!")
        print(f"   {len(violations)} → {len(result.violations)} (BETTER)")
    else:
        print("\n⚠️  Violations UNCHANGED")
        print(f"   {len(violations)} → {len(result.violations)} (STABLE)")
        print("   Network is being conservative (good!)")
        print("   Needs more training to actively reduce violations")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
