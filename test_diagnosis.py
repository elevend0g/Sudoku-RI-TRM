#!/usr/bin/env python3
"""
Diagnose why the network isn't making any changes.
"""

import numpy as np
import torch
from src.rules import SudokuRuleGraph
from src.network import create_sudoku_network
from src.solver import RecursiveSolver

def main():
    print("=" * 70)
    print("DIAGNOSING WHY NETWORK DOESN'T CHANGE GRID")
    print("=" * 70)

    # Create test grid where we CAN edit cells
    grid = np.zeros((9, 9), dtype=int)
    grid[0, 0] = 3  # Original clue
    grid[1, 1] = 5  # Original clue

    # We'll manually add a violation by setting a non-clue cell
    # Simulate this is from a previous network step (not an original clue)
    grid[0, 5] = 3  # Creates violation with (0,0)

    print("\nInitial Grid:")
    print(f"  Cell (0,0) = {grid[0,0]} - Original Clue")
    print(f"  Cell (0,5) = {grid[0,5]} - Should be editable")
    print(f"  Cell (1,1) = {grid[1,1]} - Original Clue")

    rule_graph = SudokuRuleGraph()
    violations = rule_graph.verify(grid)

    print(f"\nViolations: {len(violations)}")
    for v in violations:
        print(f"  {v.type}: cells {v.cells}")

    # Create solver - use ONLY the original clues
    original_clues_only = np.zeros((9, 9), dtype=int)
    original_clues_only[0, 0] = 3
    original_clues_only[1, 1] = 5

    network = create_sudoku_network()
    solver = RecursiveSolver(
        network=network,
        rule_graph=rule_graph,
        max_iterations=3,
        reasoning_steps=2
    )

    # Manually call _refine_step to see what happens
    print("\n" + "=" * 70)
    print("STEP-BY-STEP DIAGNOSIS")
    print("=" * 70)

    solver.original_clues = (original_clues_only != 0)

    print(f"\nOriginal clues mask:")
    print(f"  (0,0): {solver.original_clues[0,0]} - Should be True")
    print(f"  (0,5): {solver.original_clues[0,5]} - Should be False")
    print(f"  (1,1): {solver.original_clues[1,1]} - Should be True")

    # Get violation cells
    violation_cells = solver._get_violation_cells(violations)
    print(f"\nViolation cells: {violation_cells}")

    # Create edit mask
    edit_mask = np.zeros((9, 9), dtype=bool)
    for row, col in violation_cells:
        if not solver.original_clues[row, col]:
            edit_mask[row, col] = True

    print(f"\nEdit mask (cells that CAN be changed):")
    for row, col in violation_cells:
        can_edit = edit_mask[row, col]
        is_clue = solver.original_clues[row, col]
        print(f"  ({row},{col}): can_edit={can_edit}, is_clue={is_clue}")

    num_editable = np.sum(edit_mask)
    print(f"\nTotal editable cells: {num_editable}")

    if num_editable == 0:
        print("\n❌ PROBLEM FOUND: No cells are editable!")
        print("   Both violated cells are marked as original clues")
        print("   This prevents any changes from being made")
    else:
        print(f"\n✅ {num_editable} cells can be edited")

        # Actually try to refine
        print("\nCalling _refine_step...")
        improved_grid, confidence, state = solver._refine_step(
            grid, violations, None
        )

        changes = np.sum(grid != improved_grid)
        print(f"\nCells changed: {changes}")

        if changes == 0:
            print("❌ Grid didn't change even though cells were editable!")
            print("   This suggests network is outputting same values")

            # Check what network is predicting
            grid_tensor = torch.from_numpy(grid.flatten()).long().unsqueeze(0)
            violations_tensor = solver._violations_to_tensor(violations)

            with torch.no_grad():
                logits, conf, _ = network(grid_tensor, violations_tensor, None)

            # Check predictions for editable cells
            print("\nNetwork predictions for editable cells:")
            for row, col in violation_cells:
                if edit_mask[row, col]:
                    pos = row * 9 + col
                    current_val = grid[row, col]
                    predicted_probs = torch.softmax(logits[0, pos], dim=0)
                    predicted_val = torch.argmax(predicted_probs).item()
                    predicted_prob = predicted_probs[predicted_val].item()

                    print(f"  ({row},{col}): current={current_val}, "
                          f"predicted={predicted_val} (prob={predicted_prob:.3f})")
                    print(f"    Top 3 predictions: ", end="")
                    top3_vals = torch.topk(predicted_probs, 3)
                    for i, (prob, val) in enumerate(zip(top3_vals.values, top3_vals.indices)):
                        print(f"{val.item()}({prob:.3f}) ", end="")
                    print()
        else:
            print(f"✅ Grid changed! {changes} cells modified")
            for row in range(9):
                for col in range(9):
                    if grid[row, col] != improved_grid[row, col]:
                        print(f"  ({row},{col}): {grid[row,col]} → {improved_grid[row,col]}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if num_editable == 0:
        print("\n🔧 FIX NEEDED: Expand edit mask beyond just violation cells")
        print("   Option 1: Allow editing empty cells (value=0) in same row/col/box")
        print("   Option 2: Use action-based architecture")
    elif changes == 0:
        print("\n🔧 FIX NEEDED: Network predicting same values as input")
        print("   Option 1: Increase exploration (temperature/epsilon)")
        print("   Option 2: Use action-based architecture with forced changes")
    else:
        print("\n✅ System is working! Grid is being modified")


if __name__ == "__main__":
    main()
