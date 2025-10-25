"""
Rule-based Sudoku dataset generation.

Generates training tasks by creating grids with violations.
Trains on fixing violations, NOT matching ground truth.

This is fundamentally different from traditional ML datasets:
- No ground truth complete grids needed
- Rules tell us what's wrong
- Model learns to fix violations, not predict values
"""

import numpy as np
from typing import Dict, List, Optional
from src.rules import SudokuRuleGraph, Violation


class RuleBasedSudokuDataset:
    """
    Generate training tasks for RI-TRM.

    Each task is a grid (potentially with violations) that needs to be fixed.
    The rules (K_R) tell us what's wrong - no ground truth needed.

    Task generation strategy:
    1. Start with empty grid
    2. Randomly fill cells
    3. Allow violations to occur naturally
    4. Rules verify what's wrong
    """

    def __init__(
        self,
        rule_graph: SudokuRuleGraph,
        num_tasks: Optional[int] = None,
        min_filled: int = 17,  # Minimum cells for solvable Sudoku
        max_filled: int = 40,
        violation_rate: float = 0.3,  # 30% of tasks have violations
        random_seed: Optional[int] = None
    ):
        """
        Initialize dataset generator.

        Args:
            rule_graph: SudokuRuleGraph for verification
            num_tasks: Number of tasks to generate (None for infinite)
            min_filled: Minimum number of filled cells
            max_filled: Maximum number of filled cells
            violation_rate: Probability of intentionally creating violations
            random_seed: Random seed for reproducibility
        """
        self.rule_graph = rule_graph
        self.num_tasks = num_tasks
        self.min_filled = min_filled
        self.max_filled = max_filled
        self.violation_rate = violation_rate

        # Use instance-specific random number generator for reproducibility
        self.rng = np.random.default_rng(random_seed)

        self.task_count = 0

    def generate_task(self) -> Dict:
        """
        Generate a single training task.

        Returns:
            Dict with:
                - "grid": 9x9 numpy array
                - "violations": List of Violation objects
                - "task_id": Unique task identifier
                - "num_filled": Number of filled cells
                - "original_clues": 9x9 numpy array of booleans
        """
        self.task_count += 1

        # Decide number of cells to fill
        num_filled = self.rng.integers(self.min_filled, self.max_filled + 1)

        # Decide if we want violations
        force_violations = self.rng.random() < self.violation_rate

        # Generate grid
        if force_violations:
            grid, original_clues = self._generate_grid_with_violations(num_filled)
        else:
            grid = self._generate_valid_partial_grid(num_filled)
            original_clues = (grid != 0)

        # Verify and get violations
        violations = self.rule_graph.verify(grid)

        return {
            "grid": grid,
            "violations": violations,
            "task_id": self.task_count,
            "num_filled": num_filled,
            "original_clues": original_clues,
        }

    def _generate_valid_partial_grid(self, num_filled: int) -> np.ndarray:
        """
        Generate a valid partial Sudoku grid.

        Strategy: Fill cells one by one, checking validity.
        If we can't place a value, backtrack and try another.
        """
        grid = np.zeros((9, 9), dtype=int)
        cells_to_fill = self._random_cell_order()[:num_filled]

        for row, col in cells_to_fill:
            # Get valid values for this cell
            valid_values = self._get_valid_values(grid, row, col)

            if len(valid_values) > 0:
                # Choose random valid value
                value = self.rng.choice(valid_values)
                grid[row, col] = value

        return grid

    def _generate_grid_with_violations(self, num_filled: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a grid with intentional, fixable violations.

        Strategy:
        1. Generate a valid partial grid.
        2. Add a few more cells with values that are guaranteed to
           create violations with the existing valid cells.
        """
        # 1. Generate a valid partial grid
        grid = self._generate_valid_partial_grid(num_filled - 1)
        original_clues = (grid != 0)

        # 2. Find an empty cell
        empty_cells = np.argwhere(grid == 0)
        if len(empty_cells) == 0:
            return grid, original_clues  # Should not happen with num_filled < 81

        # 3. Pick a random empty cell
        cell_to_violate = tuple(self.rng.choice(empty_cells))
        row, col = cell_to_violate

        # 4. Find a value that creates a violation
        # Get values already in the row, column, and box
        used_values = set(grid[row, :]) | set(grid[:, col])
        box_row, box_col = (row // 3) * 3, (col // 3) * 3
        used_values.update(grid[box_row:box_row+3, box_col:box_col+3].flatten())
        used_values.discard(0) # remove 0 from the set

        if len(used_values) > 0:
            # Pick a value that is already used
            violating_value = self.rng.choice(list(used_values))
            grid[row, col] = violating_value
        else:
            # If there are no used values, just pick a random value
            grid[row, col] = self.rng.integers(1, 10)

        return grid, original_clues

    def _get_valid_values(self, grid: np.ndarray, row: int, col: int) -> List[int]:
        """
        Get valid values for a cell (values that don't violate rules).

        Args:
            grid: Current grid state
            row: Row index
            col: Column index

        Returns:
            List of valid values (1-9)
        """
        valid = []

        for value in range(1, 10):
            # Try placing value
            test_grid = grid.copy()
            test_grid[row, col] = value

            # Check if it creates violations
            violations = self.rule_graph.verify(test_grid)

            # Check if any violation involves this cell
            creates_violation = False
            for v in violations:
                if (row, col) in v.cells:
                    creates_violation = True
                    break

            if not creates_violation:
                valid.append(value)

        return valid

    def _random_cell_order(self) -> List[tuple]:
        """
        Get random ordering of all cells in grid.

        Returns:
            List of (row, col) tuples in random order
        """
        cells = [(i, j) for i in range(9) for j in range(9)]
        self.rng.shuffle(cells)
        return cells

    def __iter__(self):
        """Make dataset iterable."""
        return self

    def __next__(self):
        """Generate next task."""
        if self.num_tasks is not None and self.task_count >= self.num_tasks:
            raise StopIteration

        return self.generate_task()

    def __len__(self):
        """Return number of tasks (if finite)."""
        if self.num_tasks is None:
            raise ValueError("Dataset is infinite (num_tasks=None)")
        return self.num_tasks

    def get_batch(self, batch_size: int) -> List[Dict]:
        """
        Generate a batch of tasks.

        Args:
            batch_size: Number of tasks to generate

        Returns:
            List of task dictionaries
        """
        return [self.generate_task() for _ in range(batch_size)]

    def get_statistics(self, num_samples: int = 100) -> Dict:
        """
        Get statistics about generated tasks.

        Args:
            num_samples: Number of tasks to sample

        Returns:
            Dictionary with statistics
        """
        tasks = self.get_batch(num_samples)

        num_violations = [len(t["violations"]) for t in tasks]
        num_filled = [t["num_filled"] for t in tasks]

        violation_types = {
            "row_duplicate": 0,
            "col_duplicate": 0,
            "box_duplicate": 0,
        }

        for task in tasks:
            for v in task["violations"]:
                violation_types[v.type] += 1

        return {
            "num_samples": num_samples,
            "avg_violations": np.mean(num_violations),
            "std_violations": np.std(num_violations),
            "max_violations": np.max(num_violations),
            "min_violations": np.min(num_violations),
            "pct_valid": sum(1 for n in num_violations if n == 0) / num_samples * 100,
            "avg_filled": np.mean(num_filled),
            "violation_types": violation_types,
        }


class SudokuSolver:
    """
    Simple backtracking Sudoku solver for generating valid complete grids.

    Note: This is NOT used for training - it's only for testing/validation.
    RI-TRM learns to fix violations without seeing complete solutions.
    """

    def __init__(self, rule_graph: SudokuRuleGraph):
        self.rule_graph = rule_graph

    def solve(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve a Sudoku puzzle using backtracking.

        Args:
            grid: 9x9 numpy array (0 = empty)

        Returns:
            Solved grid or None if unsolvable
        """
        grid = grid.copy()

        # Find empty cell
        empty = self._find_empty(grid)
        if empty is None:
            # No empty cells - check if valid
            if self.rule_graph.is_valid(grid):
                return grid
            return None

        row, col = empty

        # Try values 1-9
        for value in range(1, 10):
            grid[row, col] = value

            if self.rule_graph.is_valid(grid):
                # Recursively solve
                result = self.solve(grid)
                if result is not None:
                    return result

            # Backtrack
            grid[row, col] = 0

        return None

    def _find_empty(self, grid: np.ndarray) -> Optional[tuple]:
        """Find first empty cell."""
        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0:
                    return (i, j)
        return None
