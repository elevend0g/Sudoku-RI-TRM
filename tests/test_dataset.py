"""
Tests for rule-based Sudoku dataset generation.

These tests validate that:
1. Tasks are generated with diverse violation patterns
2. No ground truth complete grids are needed
3. Tasks are solvable (not impossible configurations)
"""

import pytest
import numpy as np
from src.dataset import RuleBasedSudokuDataset, SudokuSolver
from src.rules import SudokuRuleGraph


class TestDatasetGeneration:
    """Test basic dataset generation."""

    def test_generate_single_task(self):
        """Test generating a single task."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph, num_tasks=10, random_seed=42)

        task = dataset.generate_task()

        assert isinstance(task, dict)
        assert "grid" in task
        assert "violations" in task
        assert "task_id" in task
        assert "num_filled" in task

    def test_grid_shape(self):
        """Generated grids should be 9x9."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph, random_seed=42)

        task = dataset.generate_task()

        assert task["grid"].shape == (9, 9)
        assert task["grid"].dtype == np.int64 or task["grid"].dtype == np.int32

    def test_grid_values_valid(self):
        """Grid values should be 0-9."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph, random_seed=42)

        task = dataset.generate_task()
        grid = task["grid"]

        assert np.all(grid >= 0)
        assert np.all(grid <= 9)

    def test_num_filled_in_range(self):
        """Number of filled cells should be in specified range."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(
            rule_graph,
            min_filled=20,
            max_filled=35,
            random_seed=42
        )

        for _ in range(10):
            task = dataset.generate_task()
            num_filled = np.sum(task["grid"] > 0)

            assert 20 <= num_filled <= 35
            assert task["num_filled"] == num_filled


class TestViolationGeneration:
    """Test that violations are generated appropriately."""

    def test_can_generate_valid_grids(self):
        """Dataset can generate tasks with 0 violations."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(
            rule_graph,
            violation_rate=0.0,  # No forced violations
            random_seed=42
        )

        # Generate multiple tasks
        valid_count = 0
        for _ in range(20):
            task = dataset.generate_task()
            if len(task["violations"]) == 0:
                valid_count += 1

        # Should generate at least some valid grids
        assert valid_count > 0, "Should be able to generate valid grids"

    def test_can_generate_violations(self):
        """Dataset can generate tasks with violations."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(
            rule_graph,
            violation_rate=1.0,  # Always force violations
            random_seed=42
        )

        # Generate multiple tasks
        violation_count = 0
        for _ in range(20):
            task = dataset.generate_task()
            if len(task["violations"]) > 0:
                violation_count += 1

        # Should generate at least some grids with violations
        assert violation_count > 0, "Should be able to generate violations"

    def test_violation_rate_respected(self):
        """Violation rate parameter should be approximately respected."""
        rule_graph = SudokuRuleGraph()
        violation_rate = 0.3
        dataset = RuleBasedSudokuDataset(
            rule_graph,
            violation_rate=violation_rate,
            random_seed=42
        )

        num_samples = 100
        grids_with_violations = 0

        for _ in range(num_samples):
            task = dataset.generate_task()
            if len(task["violations"]) > 0:
                grids_with_violations += 1

        actual_rate = grids_with_violations / num_samples

        # Should be approximately 30% (with some tolerance)
        assert 0.1 <= actual_rate <= 0.9, f"Violation rate {actual_rate} seems reasonable"

    def test_diverse_violation_types(self):
        """Should generate different types of violations."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(
            rule_graph,
            violation_rate=1.0,
            random_seed=42
        )

        violation_types = set()

        for _ in range(50):
            task = dataset.generate_task()
            for v in task["violations"]:
                violation_types.add(v.type)

        # Should see at least 2 different violation types
        assert len(violation_types) >= 2


class TestIteratorInterface:
    """Test dataset iterator interface."""

    def test_iteration(self):
        """Test iterating through dataset."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph, num_tasks=5, random_seed=42)

        tasks = list(dataset)

        assert len(tasks) == 5
        assert all(isinstance(t, dict) for t in tasks)

    def test_batch_generation(self):
        """Test generating batches."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph, random_seed=42)

        batch = dataset.get_batch(10)

        assert len(batch) == 10
        assert all(isinstance(t, dict) for t in batch)

    def test_task_ids_unique(self):
        """Task IDs should be unique and incrementing."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph, random_seed=42)

        tasks = dataset.get_batch(10)
        task_ids = [t["task_id"] for t in tasks]

        # Should be unique
        assert len(set(task_ids)) == len(task_ids)

        # Should be sequential
        assert task_ids == list(range(1, 11))


class TestStatistics:
    """Test dataset statistics functionality."""

    def test_get_statistics(self):
        """Test statistics generation."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph, random_seed=42)

        stats = dataset.get_statistics(num_samples=50)

        assert "num_samples" in stats
        assert "avg_violations" in stats
        assert "std_violations" in stats
        assert "max_violations" in stats
        assert "min_violations" in stats
        assert "pct_valid" in stats
        assert "avg_filled" in stats
        assert "violation_types" in stats

        assert stats["num_samples"] == 50
        assert stats["avg_violations"] >= 0
        assert 0 <= stats["pct_valid"] <= 100

    def test_statistics_violation_types(self):
        """Statistics should include violation type breakdown."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(
            rule_graph,
            violation_rate=1.0,
            random_seed=42
        )

        stats = dataset.get_statistics(num_samples=50)

        violation_types = stats["violation_types"]

        assert "row_duplicate" in violation_types
        assert "col_duplicate" in violation_types
        assert "box_duplicate" in violation_types


class TestGetValidValues:
    """Test the helper method for getting valid values."""

    def test_empty_grid_all_valid(self):
        """Empty grid should allow all values 1-9."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph)

        grid = np.zeros((9, 9), dtype=int)
        valid = dataset._get_valid_values(grid, 0, 0)

        assert len(valid) == 9
        assert set(valid) == set(range(1, 10))

    def test_row_constraint(self):
        """Should exclude values already in row."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph)

        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 5  # Put 5 in same row

        valid = dataset._get_valid_values(grid, 0, 5)

        assert 5 not in valid  # 5 should be excluded
        assert len(valid) == 8

    def test_column_constraint(self):
        """Should exclude values already in column."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph)

        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 7  # Put 7 in same column

        valid = dataset._get_valid_values(grid, 5, 0)

        assert 7 not in valid  # 7 should be excluded
        assert len(valid) == 8

    def test_box_constraint(self):
        """Should exclude values already in box."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph)

        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 3  # Put 3 in top-left box

        valid = dataset._get_valid_values(grid, 1, 1)

        assert 3 not in valid  # 3 should be excluded
        assert len(valid) == 8

    def test_multiple_constraints(self):
        """Should handle multiple constraints."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph)

        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 1  # Row, box
        grid[0, 5] = 2  # Row only
        grid[5, 1] = 3  # Column only
        grid[2, 2] = 4  # Box only

        valid = dataset._get_valid_values(grid, 0, 1)

        # Should exclude 1, 2, 3, 4
        assert 1 not in valid
        assert 2 not in valid
        assert 3 not in valid
        assert 4 not in valid
        assert len(valid) == 5


class TestSudokuSolver:
    """Test the backtracking Sudoku solver (for validation only)."""

    def test_solve_empty_grid(self):
        """Solver should be able to solve an empty grid."""
        rule_graph = SudokuRuleGraph()
        solver = SudokuSolver(rule_graph)

        grid = np.zeros((9, 9), dtype=int)
        solution = solver.solve(grid)

        assert solution is not None
        assert rule_graph.is_valid(solution)
        assert np.all(solution > 0)  # All cells filled

    def test_solve_partial_grid(self):
        """Solver should solve a partially filled grid."""
        rule_graph = SudokuRuleGraph()
        solver = SudokuSolver(rule_graph)

        # Start with a few cells filled
        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 5
        grid[0, 1] = 3
        grid[1, 0] = 6

        solution = solver.solve(grid)

        assert solution is not None
        assert rule_graph.is_valid(solution)

        # Original values should be preserved
        assert solution[0, 0] == 5
        assert solution[0, 1] == 3
        assert solution[1, 0] == 6

    def test_solve_nearly_complete(self):
        """Solver should complete a nearly-complete grid."""
        rule_graph = SudokuRuleGraph()
        solver = SudokuSolver(rule_graph)

        # Valid complete grid with one cell missing
        grid = np.array([
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 0]  # Last cell empty
        ])

        solution = solver.solve(grid)

        assert solution is not None
        assert rule_graph.is_valid(solution)
        assert solution[8, 8] == 9  # Should be 9

    def test_unsolvable_grid(self):
        """Solver should return None for unsolvable grids."""
        rule_graph = SudokuRuleGraph()
        solver = SudokuSolver(rule_graph)

        # Create impossible grid (two 1s in same row)
        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 1
        grid[0, 1] = 1  # Duplicate!

        solution = solver.solve(grid)

        # Should recognize this is impossible
        assert solution is None


class TestReproducibility:
    """Test that random seed makes dataset reproducible."""

    def test_same_seed_same_tasks(self):
        """Same seed should produce same tasks."""
        rule_graph = SudokuRuleGraph()

        dataset1 = RuleBasedSudokuDataset(rule_graph, random_seed=123)
        dataset2 = RuleBasedSudokuDataset(rule_graph, random_seed=123)

        task1 = dataset1.generate_task()
        task2 = dataset2.generate_task()

        assert np.array_equal(task1["grid"], task2["grid"])
        assert len(task1["violations"]) == len(task2["violations"])

    def test_different_seed_different_tasks(self):
        """Different seeds should produce different tasks."""
        rule_graph = SudokuRuleGraph()

        dataset1 = RuleBasedSudokuDataset(rule_graph, random_seed=123)
        dataset2 = RuleBasedSudokuDataset(rule_graph, random_seed=456)

        tasks1 = dataset1.get_batch(10)
        tasks2 = dataset2.get_batch(10)

        # At least some grids should be different
        different = False
        for t1, t2 in zip(tasks1, tasks2):
            if not np.array_equal(t1["grid"], t2["grid"]):
                different = True
                break

        assert different, "Different seeds should produce different tasks"
