"""
Integration tests for RI-TRM Sudoku implementation.

These are the critical tests from Sudoku.md that validate:
1. Learning from rule violations (not ground truth)
2. Path memory knowledge accumulation
3. Zero-shot verification competence
"""

import pytest
import numpy as np
from src.rules import SudokuRuleGraph, Violation
from src.dataset import RuleBasedSudokuDataset
from src.path_memory import HebbianPathMemory


class TestZeroShotVerification:
    """
    Test that rules work immediately without training.

    This is the core RI-TRM principle: Layer 2 (K_R) provides
    verification competence from initialization.
    """

    def test_zero_shot_verification(self):
        """Rules should work without any training."""
        rule_graph = SudokuRuleGraph()

        # Valid grid should pass
        valid_grid = np.zeros((9, 9), dtype=int)
        valid_grid[0, 0] = 1
        valid_grid[0, 1] = 2
        valid_grid[1, 0] = 3

        violations = rule_graph.verify(valid_grid)
        assert len(violations) == 0, "Valid grid should have no violations"

        # Invalid grid should be detected
        invalid_grid = np.zeros((9, 9), dtype=int)
        invalid_grid[0, 0] = 5
        invalid_grid[0, 1] = 5  # Duplicate!

        violations = rule_graph.verify(invalid_grid)
        assert len(violations) > 0, "Should detect duplicate without training"
        assert any(v.type == "row_duplicate" for v in violations)

    def test_zero_shot_on_complete_grid(self):
        """Should verify complete valid Sudoku without training."""
        rule_graph = SudokuRuleGraph()

        # Valid complete Sudoku
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
        assert len(violations) == 0, "Should verify complete grid without training"

    def test_zero_shot_all_violation_types(self):
        """Should detect all violation types without training."""
        rule_graph = SudokuRuleGraph()

        # Test row violations
        grid_row = np.zeros((9, 9), dtype=int)
        grid_row[0, 0] = 3
        grid_row[0, 5] = 3
        violations = rule_graph.verify(grid_row)
        assert any(v.type == "row_duplicate" for v in violations)

        # Test column violations
        grid_col = np.zeros((9, 9), dtype=int)
        grid_col[0, 0] = 7
        grid_col[5, 0] = 7
        violations = rule_graph.verify(grid_col)
        assert any(v.type == "col_duplicate" for v in violations)

        # Test box violations
        grid_box = np.zeros((9, 9), dtype=int)
        grid_box[0, 0] = 9
        grid_box[1, 2] = 9
        violations = rule_graph.verify(grid_box)
        assert any(v.type == "box_duplicate" for v in violations)


class TestPathMemoryGrowth:
    """
    Test that path memory grows and strengthens over time.

    From Sudoku.md:
    - After 100 tasks: ~500 paths, avg weight 0.45
    - After 500 tasks: ~2000 paths, avg weight 0.67
    - After 1000 tasks: ~5000 paths, avg weight 0.78
    """

    def test_path_memory_grows(self):
        """Path memory should accumulate knowledge over training."""
        memory = HebbianPathMemory()

        # Initially empty
        assert len(memory) == 0

        # Simulate learning from 100 violations
        patterns = [
            frozenset(["row_duplicate"]),
            frozenset(["col_duplicate"]),
            frozenset(["box_duplicate"]),
            frozenset(["row_duplicate", "col_duplicate"]),
        ]

        actions = ["fix_cell", "swap_values", "try_different_value"]

        for i in range(100):
            pattern = patterns[i % len(patterns)]
            action = actions[i % len(actions)]
            success = (i % 3 != 0)  # ~67% success rate

            memory.update(pattern, action, success)

        # Memory should have grown
        assert len(memory) > 0
        assert len(memory) <= 12  # 4 patterns × 3 actions = 12 max

        # Get statistics
        stats = memory.get_statistics()
        assert stats["num_paths"] > 0
        assert stats["total_updates"] == 100
        assert stats["successful_updates"] > 0
        assert stats["failed_updates"] > 0

    def test_path_weights_increase_with_success(self):
        """Successful paths should strengthen over time."""
        memory = HebbianPathMemory(alpha=0.1)

        pattern = frozenset(["row_duplicate"])
        action = "effective_fix"

        weights = []

        # Apply 20 successful updates
        for _ in range(20):
            path = memory.update(pattern, action, success=True)
            weights.append(path.weight)

        # Weights should increase monotonically
        for i in range(len(weights) - 1):
            assert weights[i + 1] > weights[i] or weights[i + 1] > 0.95  # Near max

        # Final weight should be high
        assert weights[-1] > 0.8

    def test_strong_paths_emerge(self):
        """Strong paths (weight > 0.8) should emerge with experience."""
        memory = HebbianPathMemory(alpha=0.15)

        # Simulate successful pattern
        pattern = frozenset(["col_duplicate"])
        action = "expert_fix"

        # Apply many successes
        for _ in range(15):
            memory.update(pattern, action, success=True)

        stats = memory.get_statistics()
        assert stats["strong_paths"] >= 1, "Should have at least one strong path"

    def test_pattern_diversity_grows(self):
        """Should learn patterns for different violation types."""
        memory = HebbianPathMemory()

        patterns = [
            frozenset(["row_duplicate"]),
            frozenset(["col_duplicate"]),
            frozenset(["box_duplicate"]),
        ]

        for pattern in patterns:
            memory.update(pattern, "fix", success=True)

        diversity = memory.get_pattern_diversity()
        assert diversity == 3, "Should have 3 unique patterns"


class TestRuleBasedDataset:
    """Test that dataset generates tasks based on rules, not ground truth."""

    def test_no_ground_truth_needed(self):
        """Dataset should work without complete Sudoku solutions."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph, num_tasks=10, random_seed=42)

        # Generate tasks
        tasks = list(dataset)

        assert len(tasks) == 10

        # Each task has violations detected by rules
        for task in tasks:
            assert "grid" in task
            assert "violations" in task

            # Violations come from rule verification
            violations = rule_graph.verify(task["grid"])
            assert task["violations"] == violations

    def test_learns_from_violations(self):
        """Training should be based on fixing violations."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(
            rule_graph,
            violation_rate=0.8,  # Mostly violations
            random_seed=42
        )

        tasks = dataset.get_batch(20)

        # Count tasks with violations
        tasks_with_violations = sum(1 for t in tasks if len(t["violations"]) > 0)

        # Most should have violations
        assert tasks_with_violations > 10

    def test_violation_patterns_diverse(self):
        """Dataset should generate diverse violation patterns."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(
            rule_graph,
            violation_rate=1.0,
            random_seed=42
        )

        tasks = dataset.get_batch(50)

        # Collect violation types
        violation_types = set()
        for task in tasks:
            for v in task["violations"]:
                violation_types.add(v.type)

        # Should see multiple types
        assert len(violation_types) >= 2


class TestLearningFromRules:
    """
    Test the core RI-TRM principle: learning from rule violations.

    The model should learn to reduce violations, not match examples.
    """

    def test_simulated_learning_reduces_violations(self):
        """Simulate learning loop that reduces violations."""
        rule_graph = SudokuRuleGraph()
        memory = HebbianPathMemory()

        # Create grid with violations
        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 4
        grid[0, 3] = 4  # Row duplicate

        initial_violations = rule_graph.verify(grid)
        assert len(initial_violations) > 0

        # Simulate fix attempt
        violation_pattern = frozenset(v.type for v in initial_violations)

        # Try action: change grid[0, 3]
        grid[0, 3] = 5
        new_violations = rule_graph.verify(grid)

        # Check if violations reduced
        success = len(new_violations) < len(initial_violations)

        # Update path memory
        memory.update(violation_pattern, "change_cell_0_3_to_5", success)

        # Memory should learn from this
        assert len(memory) > 0

        if success:
            key = (violation_pattern, "change_cell_0_3_to_5")
            assert memory.paths[key].weight > 0.5

    def test_iterative_improvement(self):
        """Multiple refinement steps should reduce violations."""
        rule_graph = SudokuRuleGraph()

        # Start with violations
        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 2
        grid[0, 1] = 2  # Duplicate
        grid[1, 0] = 3
        grid[1, 1] = 3  # Another duplicate

        violations_history = []
        violations_history.append(len(rule_graph.verify(grid)))

        # Fix first violation
        grid[0, 1] = 7
        violations_history.append(len(rule_graph.verify(grid)))

        # Fix second violation
        grid[1, 1] = 8
        violations_history.append(len(rule_graph.verify(grid)))

        # Violations should decrease
        assert violations_history[0] > violations_history[1]
        assert violations_history[1] > violations_history[2]
        assert violations_history[2] == 0  # Fully fixed


class TestEndToEndWorkflow:
    """Test complete workflow: rules → dataset → learning → memory."""

    def test_complete_workflow(self):
        """Test end-to-end: generate task, detect violations, learn fix."""
        # Initialize components
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph, random_seed=42)
        memory = HebbianPathMemory()

        # Generate task
        task = dataset.generate_task()
        grid = task["grid"]
        violations = task["violations"]

        # Rules work zero-shot
        verified_violations = rule_graph.verify(grid)
        assert violations == verified_violations

        if len(violations) > 0:
            # Create violation pattern
            pattern = frozenset(v.type for v in violations)

            # Simulate fix attempt (randomly change a cell)
            modified_grid = grid.copy()
            # Find first non-zero cell and change it
            for i in range(9):
                for j in range(9):
                    if modified_grid[i, j] > 0:
                        modified_grid[i, j] = (modified_grid[i, j] % 9) + 1
                        break
                else:
                    continue
                break

            # Check if improved
            new_violations = rule_graph.verify(modified_grid)
            success = len(new_violations) < len(violations)

            # Learn from outcome
            memory.update(pattern, "modify_cell", success)

            # Memory should have learned
            assert len(memory) > 0

    def test_learns_to_fix_row_duplicates(self):
        """
        Test from Sudoku.md: Learn to fix row duplicates.

        Generate multiple tasks with row duplicates, learn patterns.
        """
        rule_graph = SudokuRuleGraph()
        memory = HebbianPathMemory()

        # Generate 10 grids with row duplicates
        for seed in range(10):
            grid = np.zeros((9, 9), dtype=int)
            rng = np.random.default_rng(seed)

            # Add row duplicate
            row = rng.integers(0, 9)
            col1, col2 = rng.choice(9, size=2, replace=False)
            value = rng.integers(1, 10)

            grid[row, col1] = value
            grid[row, col2] = value

            # Verify violation
            violations = rule_graph.verify(grid)
            assert any(v.type == "row_duplicate" for v in violations)

            # Try fix: change one cell
            grid[row, col2] = (value % 9) + 1
            new_violations = rule_graph.verify(grid)

            # Update memory
            pattern = frozenset(["row_duplicate"])
            success = len(new_violations) < len(violations)
            memory.update(pattern, "change_duplicate_cell", success)

        # After 10 examples, should have learned
        assert len(memory) > 0
        stats = memory.get_statistics()
        assert stats["total_updates"] == 10

        # Query for row duplicates
        test_violations = [
            Violation("row_duplicate", (0,), [5], [(0, 0), (0, 5)])
        ]
        candidate_paths = memory.query(test_violations)
        assert len(candidate_paths) > 0

    def test_memory_persistence(self):
        """Test saving and loading trained memory."""
        memory1 = HebbianPathMemory()

        # Train memory
        patterns = [
            frozenset(["row_duplicate"]),
            frozenset(["col_duplicate"]),
        ]

        for i in range(50):
            pattern = patterns[i % 2]
            action = f"fix_{i % 5}"
            success = (i % 3 != 0)
            memory1.update(pattern, action, success)

        # Save
        data = memory1.save_to_dict()

        # Load
        memory2 = HebbianPathMemory.load_from_dict(data)

        # Should have same statistics
        stats1 = memory1.get_statistics()
        stats2 = memory2.get_statistics()

        assert stats1["num_paths"] == stats2["num_paths"]
        assert stats1["total_updates"] == stats2["total_updates"]
        assert abs(stats1["avg_weight"] - stats2["avg_weight"]) < 1e-6


class TestExpectedBehaviors:
    """Test expected behaviors from RI-TRM paper."""

    def test_interpretable_reasoning(self):
        """Path memory should provide interpretable reasoning."""
        memory = HebbianPathMemory()

        # Train with clear pattern
        pattern = frozenset(["row_duplicate"])
        for _ in range(15):
            memory.update(pattern, "change_rightmost_duplicate", success=True)

        # Query should return interpretable path
        violations = [
            Violation("row_duplicate", (3,), [7], [(3, 2), (3, 8)])
        ]
        paths = memory.query(violations, top_k=1)

        assert len(paths) > 0
        best_path = paths[0]

        # Path should have high weight
        assert best_path.weight > 0.8

        # Action should be interpretable
        assert "change_rightmost_duplicate" in best_path.action

    def test_confidence_tracking(self):
        """Path weights represent confidence in the fix."""
        memory = HebbianPathMemory()

        pattern = frozenset(["col_duplicate"])

        # Low confidence: few examples
        for _ in range(2):
            memory.update(pattern, "uncertain_fix", success=True)

        key_uncertain = (pattern, "uncertain_fix")
        weight_low = memory.paths[key_uncertain].weight

        # High confidence: many examples
        for _ in range(15):
            memory.update(pattern, "confident_fix", success=True)

        key_confident = (pattern, "confident_fix")
        weight_high = memory.paths[key_confident].weight

        # Confident path should have higher weight
        assert weight_high > weight_low
        assert weight_high > 0.85


class TestScaling:
    """Test behavior as dataset and memory scale."""

    def test_dataset_scales(self):
        """Dataset should handle large number of tasks."""
        rule_graph = SudokuRuleGraph()
        dataset = RuleBasedSudokuDataset(rule_graph, random_seed=42)

        # Generate 100 tasks
        tasks = dataset.get_batch(100)

        assert len(tasks) == 100

        # Check statistics
        stats = dataset.get_statistics(num_samples=100)
        assert stats["num_samples"] == 100
        assert stats["avg_violations"] >= 0

    def test_memory_scales(self):
        """Path memory should handle large number of paths."""
        memory = HebbianPathMemory()

        # Create many paths
        for i in range(100):
            pattern = frozenset([f"type_{i % 10}"])
            action = f"action_{i % 20}"
            success = (i % 2 == 0)
            memory.update(pattern, action, success)

        assert len(memory) > 0
        assert len(memory) <= 200  # 10 types × 20 actions max

        # Statistics should work
        stats = memory.get_statistics()
        assert stats["num_paths"] == len(memory)
        assert stats["total_updates"] == 100
