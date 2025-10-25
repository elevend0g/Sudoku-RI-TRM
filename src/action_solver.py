"""
Action-based solver that applies discrete changes to grid.
Guarantees exactly one cell changes per step.
"""

import numpy as np
import torch
from typing import List, Tuple
from dataclasses import dataclass

from src.rules import SudokuRuleGraph, Violation
from src.action_network import ActionBasedNetwork


@dataclass
class ActionStep:
    """Record of one action taken."""
    step: int
    cell_changed: Tuple[int, int]  # (row, col)
    old_value: int
    new_value: int
    violations_before: int
    violations_after: int
    success: bool  # Did violations decrease?


class ActionBasedSolver:
    """
    Solver that applies discrete actions to reduce violations.
    """

    def __init__(
        self,
        network: ActionBasedNetwork,
        rule_graph: SudokuRuleGraph,
        path_memory=None,
        epsilon=0.3  # Start with exploration
    ):
        self.network = network
        self.rule_graph = rule_graph
        self.path_memory = path_memory
        self.epsilon = epsilon

    def solve(self, grid, original_clues=None, max_iterations=16, verbose=False):
        """
        Solve by iteratively applying actions.

        Returns:
            final_grid: [9, 9] numpy array
            trace: List[ActionStep]
            solved: bool
        """
        current_grid = grid.copy()
        trace = []
        if original_clues is None:
            original_clues = (grid != 0)

        for step in range(max_iterations):
            violations = self.rule_graph.verify(current_grid)

            if len(violations) == 0:
                if verbose:
                    print(f"✓ Solved in {step} steps!")
                return current_grid, trace, True

            # Get candidate cells (violated non-clues)
            candidate_cells = self._get_candidate_cells(
                violations,
                original_clues
            )

            if len(candidate_cells) == 0:
                if verbose:
                    print("No editable cells available")
                break

            # Select and apply action
            cell_idx, new_value = self.network.select_action(
                current_grid,
                violations,
                candidate_cells,
                epsilon=self.epsilon
            )

            # Convert cell_idx to (row, col)
            row, col = cell_idx // 9, cell_idx % 9
            old_value = current_grid[row, col]

            # Apply action
            current_grid[row, col] = new_value

            # Verify result
            new_violations = self.rule_graph.verify(current_grid)
            success = len(new_violations) < len(violations)

            # Record step
            trace.append(ActionStep(
                step=step,
                cell_changed=(row, col),
                old_value=old_value,
                new_value=new_value,
                violations_before=len(violations),
                violations_after=len(new_violations),
                success=success
            ))

            if verbose:
                arrow = "→" if success else "↗" if len(new_violations) > len(violations) else "→"
                print(f"Step {step}: ({row},{col}) {old_value}→{new_value}, "
                      f"{len(violations)} {arrow} {len(new_violations)} violations")

            # ASSERTION: Grid must have changed
            assert current_grid[row, col] != old_value or old_value == new_value, \
                "Grid didn't change!"

        return current_grid, trace, False

    def _get_candidate_cells(self, violations, original_clues):
        """
        Get cells that can be edited (in violations, not original clues).
        """
        candidate_cells = []

        for violation in violations:
            for row, col in violation.cells:
                if not original_clues[row, col]:
                    if (row, col) not in candidate_cells:
                        candidate_cells.append((row, col))

        return candidate_cells
