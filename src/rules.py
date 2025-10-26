"""
Explicit Sudoku rules - K_R (Layer 2) in RI-TRM architecture.

This provides zero-shot verification competence without any training.
The rule graph can verify any Sudoku grid immediately.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Violation:
    """
    Represents a rule violation in a Sudoku grid.

    Attributes:
        type: Type of violation (row_duplicate, col_duplicate, box_duplicate)
        location: Location of the violation (row, col) or box number
        conflicting_values: List of values that are duplicated
        cells: List of cell coordinates involved in the violation
    """
    type: str
    location: Tuple[int, ...]
    conflicting_values: List[int]
    cells: List[Tuple[int, int]]

    def __repr__(self):
        return f"Violation(type={self.type}, location={self.location}, values={self.conflicting_values})"


class SudokuRuleGraph:
    """
    Explicit Sudoku rule verification.

    Encodes the formal rules of Sudoku:
    1. Each row must contain unique values (1-9, ignoring 0s)
    2. Each column must contain unique values (1-9, ignoring 0s)
    3. Each 3x3 box must contain unique values (1-9, ignoring 0s)

    This is Layer 2 (K_R) in the RI-TRM architecture - structural knowledge
    that is provided explicitly, not learned.
    """

    def __init__(self):
        self.grid_size = 9
        self.box_size = 3

    def verify(self, grid: np.ndarray) -> List[Violation]:
        """
        Verify a Sudoku grid against all rules.

        Args:
            grid: 9x9 numpy array with values 0-9 (0 = empty)

        Returns:
            List of Violation objects. Empty list means valid grid.
        """
        if grid.shape != (9, 9):
            raise ValueError(f"Grid must be 9x9, got {grid.shape}")

        violations = []

        # Check row violations
        violations.extend(self._check_rows(grid))

        # Check column violations
        violations.extend(self._check_columns(grid))

        # Check box violations
        violations.extend(self._check_boxes(grid))

        return violations

    def _check_rows(self, grid: np.ndarray) -> List[Violation]:
        """Check for duplicate values in rows."""
        violations = []

        for row_idx in range(self.grid_size):
            row = grid[row_idx, :]
            # Only check non-zero values
            non_zero = row[row > 0]

            if len(non_zero) != len(set(non_zero)):
                # Found duplicates
                duplicates = self._find_duplicates(non_zero)
                for value in duplicates:
                    # Find all cells with this duplicate value
                    cells = [(row_idx, col_idx) for col_idx in range(self.grid_size)
                            if grid[row_idx, col_idx] == value]

                    violations.append(Violation(
                        type="row_duplicate",
                        location=(row_idx,),
                        conflicting_values=[value],
                        cells=cells
                    ))

        return violations

    def _check_columns(self, grid: np.ndarray) -> List[Violation]:
        """Check for duplicate values in columns."""
        violations = []

        for col_idx in range(self.grid_size):
            col = grid[:, col_idx]
            # Only check non-zero values
            non_zero = col[col > 0]

            if len(non_zero) != len(set(non_zero)):
                # Found duplicates
                duplicates = self._find_duplicates(non_zero)
                for value in duplicates:
                    # Find all cells with this duplicate value
                    cells = [(row_idx, col_idx) for row_idx in range(self.grid_size)
                            if grid[row_idx, col_idx] == value]

                    violations.append(Violation(
                        type="col_duplicate",
                        location=(col_idx,),
                        conflicting_values=[value],
                        cells=cells
                    ))

        return violations

    def _check_boxes(self, grid: np.ndarray) -> List[Violation]:
        """Check for duplicate values in 3x3 boxes."""
        violations = []

        for box_row in range(3):
            for box_col in range(3):
                # Extract 3x3 box
                box = grid[
                    box_row * 3:(box_row + 1) * 3,
                    box_col * 3:(box_col + 1) * 3
                ]

                # Flatten and check non-zero values
                flat_box = box.flatten()
                non_zero = flat_box[flat_box > 0]

                if len(non_zero) != len(set(non_zero)):
                    # Found duplicates
                    duplicates = self._find_duplicates(non_zero)
                    box_num = box_row * 3 + box_col

                    for value in duplicates:
                        # Find all cells with this duplicate value
                        cells = []
                        for i in range(3):
                            for j in range(3):
                                if box[i, j] == value:
                                    cells.append((box_row * 3 + i, box_col * 3 + j))

                        violations.append(Violation(
                            type="box_duplicate",
                            location=(box_num,),
                            conflicting_values=[value],
                            cells=cells
                        ))

        return violations

    @staticmethod
    def _find_duplicates(arr: np.ndarray) -> List[int]:
        """Find duplicate values in an array."""
        seen = set()
        duplicates = set()

        for value in arr:
            if value in seen:
                duplicates.add(int(value))
            seen.add(value)

        return list(duplicates)

    def is_valid(self, grid: np.ndarray) -> bool:
        """
        Quick check if grid is valid (no violations).

        Args:
            grid: 9x9 numpy array

        Returns:
            True if no violations, False otherwise
        """
        return len(self.verify(grid)) == 0

    def get_violation_summary(self, grid: np.ndarray) -> dict:
        """
        Get a summary of violations by type.

        Returns:
            Dict with counts of each violation type
        """
        violations = self.verify(grid)

        summary = {
            "total": len(violations),
            "row_duplicates": sum(1 for v in violations if v.type == "row_duplicate"),
            "col_duplicates": sum(1 for v in violations if v.type == "col_duplicate"),
            "box_duplicates": sum(1 for v in violations if v.type == "box_duplicate"),
        }

        return summary

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
            violations = self.verify(test_grid)

            # Check if any violation involves this cell
            creates_violation = False
            for v in violations:
                if (row, col) in v.cells:
                    creates_violation = True
                    break

            if not creates_violation:
                valid.append(value)

        return valid
