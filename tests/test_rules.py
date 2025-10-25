"""
Tests for Sudoku rule verification (K_R - Layer 2).

These tests validate zero-shot verification competence:
the ability to verify correctness without any training.
"""

import pytest
import numpy as np
from src.rules import SudokuRuleGraph, Violation


class TestZeroShotVerification:
    """Test that rules work immediately without training."""

    def test_valid_empty_grid(self):
        """Empty grid should have no violations."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        violations = rule_graph.verify(grid)

        assert len(violations) == 0, "Empty grid should be valid"
        assert rule_graph.is_valid(grid), "is_valid should return True"

    def test_valid_partial_grid(self):
        """Partially filled valid grid should have no violations."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Fill some cells without conflicts
        grid[0, 0] = 1
        grid[0, 1] = 2
        grid[0, 2] = 3
        grid[1, 0] = 4
        grid[2, 0] = 7

        violations = rule_graph.verify(grid)

        assert len(violations) == 0, "Valid partial grid should have no violations"
        assert rule_graph.is_valid(grid)

    def test_valid_complete_grid(self):
        """A complete valid Sudoku should have no violations."""
        rule_graph = SudokuRuleGraph()

        # Valid complete Sudoku (classic example)
        grid = np.array([
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ])

        violations = rule_graph.verify(grid)

        assert len(violations) == 0, "Valid complete grid should have no violations"
        assert rule_graph.is_valid(grid)


class TestRowViolations:
    """Test detection of row duplicate violations."""

    def test_row_duplicate_simple(self):
        """Detect duplicate in a single row."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Put duplicate 1s in first row
        grid[0, 0] = 1
        grid[0, 5] = 1

        violations = rule_graph.verify(grid)

        assert len(violations) > 0, "Should detect row duplicate"
        assert any(v.type == "row_duplicate" for v in violations)

        # Find the row violation
        row_violations = [v for v in violations if v.type == "row_duplicate"]
        assert len(row_violations) == 1
        assert row_violations[0].location == (0,)
        assert 1 in row_violations[0].conflicting_values
        assert len(row_violations[0].cells) == 2

    def test_multiple_row_duplicates(self):
        """Detect multiple duplicate values in same row."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Duplicate 1s and 5s in first row
        grid[0, 0] = 1
        grid[0, 1] = 1
        grid[0, 2] = 5
        grid[0, 3] = 5

        violations = rule_graph.verify(grid)

        row_violations = [v for v in violations if v.type == "row_duplicate"]
        assert len(row_violations) == 2, "Should detect both duplicates"

        values = set()
        for v in row_violations:
            values.update(v.conflicting_values)

        assert 1 in values and 5 in values

    def test_row_duplicates_in_different_rows(self):
        """Detect violations in multiple rows."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Row 0 has duplicate 3
        grid[0, 0] = 3
        grid[0, 1] = 3

        # Row 5 has duplicate 7
        grid[5, 2] = 7
        grid[5, 8] = 7

        violations = rule_graph.verify(grid)

        row_violations = [v for v in violations if v.type == "row_duplicate"]
        assert len(row_violations) == 2

        locations = [v.location for v in row_violations]
        assert (0,) in locations
        assert (5,) in locations


class TestColumnViolations:
    """Test detection of column duplicate violations."""

    def test_col_duplicate_simple(self):
        """Detect duplicate in a single column."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Put duplicate 2s in first column
        grid[0, 0] = 2
        grid[5, 0] = 2

        violations = rule_graph.verify(grid)

        assert len(violations) > 0
        col_violations = [v for v in violations if v.type == "col_duplicate"]
        assert len(col_violations) == 1
        assert col_violations[0].location == (0,)
        assert 2 in col_violations[0].conflicting_values

    def test_multiple_column_duplicates(self):
        """Detect violations in multiple columns."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Column 0 has duplicate 4
        grid[0, 0] = 4
        grid[3, 0] = 4

        # Column 8 has duplicate 9
        grid[1, 8] = 9
        grid[7, 8] = 9

        violations = rule_graph.verify(grid)

        col_violations = [v for v in violations if v.type == "col_duplicate"]
        assert len(col_violations) == 2

        locations = [v.location for v in col_violations]
        assert (0,) in locations
        assert (8,) in locations


class TestBoxViolations:
    """Test detection of 3x3 box duplicate violations."""

    def test_box_duplicate_simple(self):
        """Detect duplicate in a single box."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Put duplicate 5s in top-left box (box 0)
        grid[0, 0] = 5
        grid[1, 2] = 5

        violations = rule_graph.verify(grid)

        box_violations = [v for v in violations if v.type == "box_duplicate"]
        assert len(box_violations) == 1
        assert box_violations[0].location == (0,)
        assert 5 in box_violations[0].conflicting_values

    def test_all_boxes(self):
        """Test detection across different boxes."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Box 0 (top-left): duplicate 1
        grid[0, 0] = 1
        grid[1, 1] = 1

        # Box 4 (center): duplicate 8
        grid[3, 3] = 8
        grid[4, 5] = 8

        # Box 8 (bottom-right): duplicate 9
        grid[6, 6] = 9
        grid[8, 8] = 9

        violations = rule_graph.verify(grid)

        box_violations = [v for v in violations if v.type == "box_duplicate"]
        assert len(box_violations) == 3

        box_nums = [v.location[0] for v in box_violations]
        assert 0 in box_nums  # top-left
        assert 4 in box_nums  # center
        assert 8 in box_nums  # bottom-right

    def test_box_boundaries(self):
        """Test that boxes are correctly bounded."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # These should NOT conflict (different boxes)
        grid[0, 2] = 7  # Box 0 (top-left)
        grid[0, 3] = 7  # Box 1 (top-center)

        violations = rule_graph.verify(grid)

        # Should have row duplicate but NOT box duplicate
        box_violations = [v for v in violations if v.type == "box_duplicate"]
        row_violations = [v for v in violations if v.type == "row_duplicate"]

        assert len(box_violations) == 0, "Should not detect box violation across box boundary"
        assert len(row_violations) == 1, "Should detect row violation"


class TestMixedViolations:
    """Test grids with multiple types of violations."""

    def test_row_and_column_violation(self):
        """Grid with both row and column violations."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Create a violation that's both row and column
        grid[0, 0] = 3
        grid[0, 5] = 3  # Row duplicate
        grid[4, 0] = 3  # Column duplicate

        violations = rule_graph.verify(grid)

        assert len(violations) >= 2
        types = [v.type for v in violations]
        assert "row_duplicate" in types
        assert "col_duplicate" in types

    def test_all_violation_types(self):
        """Grid with row, column, and box violations."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Row violation
        grid[0, 0] = 1
        grid[0, 8] = 1

        # Column violation
        grid[1, 1] = 2
        grid[8, 1] = 2

        # Box violation
        grid[3, 3] = 5
        grid[4, 5] = 5

        violations = rule_graph.verify(grid)

        types = set(v.type for v in violations)
        assert "row_duplicate" in types
        assert "col_duplicate" in types
        assert "box_duplicate" in types


class TestViolationObject:
    """Test Violation object structure."""

    def test_violation_attributes(self):
        """Verify Violation objects have correct attributes."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        grid[0, 0] = 4
        grid[0, 3] = 4

        violations = rule_graph.verify(grid)

        assert len(violations) > 0
        v = violations[0]

        # Check all required attributes exist
        assert hasattr(v, 'type')
        assert hasattr(v, 'location')
        assert hasattr(v, 'conflicting_values')
        assert hasattr(v, 'cells')

        # Check types
        assert isinstance(v.type, str)
        assert isinstance(v.location, tuple)
        assert isinstance(v.conflicting_values, list)
        assert isinstance(v.cells, list)

        # Check cells are coordinate tuples
        for cell in v.cells:
            assert isinstance(cell, tuple)
            assert len(cell) == 2


class TestViolationSummary:
    """Test violation summary functionality."""

    def test_summary_counts(self):
        """Test get_violation_summary returns correct counts."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Add violations that don't overlap in boxes
        # Row violation (not in same box)
        grid[0, 0] = 1
        grid[0, 1] = 1  # Row violation AND box violation (both in box 0)

        # Column violations (in different boxes)
        grid[1, 6] = 2  # Box 2
        grid[5, 6] = 2  # Box 5 - Col violation only

        grid[2, 7] = 3  # Box 2
        grid[7, 7] = 3  # Box 8 - Col violation only

        # Box violation (not in same row/col)
        grid[3, 3] = 5  # Box 4, row 3, col 3
        grid[4, 4] = 5  # Box 4, row 4, col 4 - Box violation only

        summary = rule_graph.get_violation_summary(grid)

        # grid[0,0]=1 and grid[0,1]=1 create BOTH row AND box violations
        # So: 1 row, 2 col, 2 box = 5 total
        assert summary["total"] == 5
        assert summary["row_duplicates"] == 1
        assert summary["col_duplicates"] == 2
        assert summary["box_duplicates"] == 2

    def test_summary_empty_grid(self):
        """Empty grid should have zero violations in summary."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        summary = rule_graph.get_violation_summary(grid)

        assert summary["total"] == 0
        assert summary["row_duplicates"] == 0
        assert summary["col_duplicates"] == 0
        assert summary["box_duplicates"] == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_grid_shape(self):
        """Should raise error for non-9x9 grid."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((8, 8), dtype=int)

        with pytest.raises(ValueError, match="Grid must be 9x9"):
            rule_graph.verify(grid)

    def test_triple_duplicates(self):
        """Test detection when value appears 3+ times."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Put 6 three times in row 0
        grid[0, 0] = 6
        grid[0, 3] = 6
        grid[0, 8] = 6

        violations = rule_graph.verify(grid)

        row_violations = [v for v in violations if v.type == "row_duplicate"]
        assert len(row_violations) == 1
        assert len(row_violations[0].cells) == 3

    def test_ignores_zeros(self):
        """Multiple zeros should not be considered duplicates."""
        rule_graph = SudokuRuleGraph()
        grid = np.zeros((9, 9), dtype=int)

        # Grid full of zeros should be valid
        violations = rule_graph.verify(grid)
        assert len(violations) == 0
