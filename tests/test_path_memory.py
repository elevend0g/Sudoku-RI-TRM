"""
Tests for Hebbian path memory (K_P - Layer 3).

These tests validate:
1. Long-Term Potentiation (successful paths strengthen)
2. Long-Term Depression (failed paths weaken)
3. Myelination (heavily-used paths get boosted)
4. ε-greedy exploration/exploitation
5. Path query and selection
"""

import pytest
import numpy as np
from src.path_memory import HebbianPathMemory, Path
from src.rules import Violation


class TestLongTermPotentiation:
    """Test that successful paths strengthen (LTP)."""

    def test_ltp_increases_weight(self):
        """Successful update should increase path weight."""
        memory = HebbianPathMemory(alpha=0.1)

        pattern = frozenset(["row_duplicate"])
        action = "fix_duplicate"
        initial_weight = 0.5

        # Update with success
        path = memory.update(pattern, action, success=True)

        # Weight should increase
        assert path.weight > initial_weight

    def test_ltp_formula(self):
        """Test LTP formula: w_new = w_old + α(1 - w_old)."""
        memory = HebbianPathMemory(alpha=0.1)

        pattern = frozenset(["col_duplicate"])
        action = "change_cell"

        # First update creates path with weight 0.5
        memory.update(pattern, action, success=True)

        # Get current weight
        key = (pattern, action)
        w_old = memory.paths[key].weight

        # Second successful update
        memory.update(pattern, action, success=True)
        w_new = memory.paths[key].weight

        # Check formula: w_new = w_old + 0.1 * (1 - w_old)
        expected = w_old + 0.1 * (1 - w_old)
        assert abs(w_new - expected) < 1e-6

    def test_multiple_successes(self):
        """Multiple successful updates should progressively strengthen path."""
        memory = HebbianPathMemory(alpha=0.1)

        pattern = frozenset(["box_duplicate"])
        action = "swap_values"

        weights = [0.5]  # Initial weight

        # Apply 10 successful updates
        for _ in range(10):
            path = memory.update(pattern, action, success=True)
            weights.append(path.weight)

        # Weights should increase monotonically
        for i in range(len(weights) - 1):
            assert weights[i + 1] > weights[i]

        # Final weight should approach 1.0
        assert weights[-1] > 0.9

    def test_usage_count_increases(self):
        """Successful updates should increment usage count."""
        memory = HebbianPathMemory()

        pattern = frozenset(["row_duplicate"])
        action = "fix"

        # Apply 5 successful updates
        for _ in range(5):
            path = memory.update(pattern, action, success=True)

        assert path.usage_count == 5


class TestLongTermDepression:
    """Test that failed paths weaken (LTD)."""

    def test_ltd_decreases_weight(self):
        """Failed update should decrease path weight."""
        memory = HebbianPathMemory(gamma=0.95)

        pattern = frozenset(["row_duplicate"])
        action = "bad_fix"

        # Create path with high weight first
        for _ in range(10):
            memory.update(pattern, action, success=True)

        key = (pattern, action)
        w_before = memory.paths[key].weight

        # Failed update
        memory.update(pattern, action, success=False)
        w_after = memory.paths[key].weight

        # Weight should decrease
        assert w_after < w_before

    def test_ltd_formula(self):
        """Test LTD formula: w_new = w_old * γ."""
        memory = HebbianPathMemory(gamma=0.95)

        pattern = frozenset(["col_duplicate"])
        action = "wrong_fix"

        # Start with some weight
        memory.update(pattern, action, success=True)
        memory.update(pattern, action, success=True)

        key = (pattern, action)
        w_old = memory.paths[key].weight

        # Failed update
        memory.update(pattern, action, success=False)
        w_new = memory.paths[key].weight

        # Check formula: w_new = w_old * 0.95
        expected = w_old * 0.95
        assert abs(w_new - expected) < 1e-6

    def test_multiple_failures(self):
        """Multiple failures should progressively weaken path."""
        memory = HebbianPathMemory(gamma=0.95)

        pattern = frozenset(["box_duplicate"])
        action = "ineffective_fix"

        # Start with some weight
        for _ in range(5):
            memory.update(pattern, action, success=True)

        key = (pattern, action)
        weights = [memory.paths[key].weight]

        # Apply 10 failed updates
        for _ in range(10):
            memory.update(pattern, action, success=False)
            weights.append(memory.paths[key].weight)

        # Weights should decrease monotonically
        for i in range(len(weights) - 1):
            assert weights[i + 1] < weights[i]

    def test_weight_lower_bound(self):
        """Weight should not go below min_weight."""
        memory = HebbianPathMemory(gamma=0.95, min_weight=0.01)

        pattern = frozenset(["row_duplicate"])
        action = "bad_fix"

        # Start with some weight
        memory.update(pattern, action, success=True)

        # Apply many failures
        for _ in range(100):
            path = memory.update(pattern, action, success=False)

        # Should not go below min_weight
        assert path.weight >= 0.01


class TestMyelination:
    """Test myelination boost for heavily-used paths."""

    def test_myelination_after_threshold(self):
        """Path should get boosted after threshold uses."""
        memory = HebbianPathMemory(
            alpha=0.1,
            beta=1.1,
            theta_myelination=10
        )

        pattern = frozenset(["row_duplicate"])
        action = "effective_fix"

        # Apply 9 successful updates (below threshold)
        for _ in range(9):
            memory.update(pattern, action, success=True)

        key = (pattern, action)
        weight_before_threshold = memory.paths[key].weight

        # 10th update should trigger myelination
        path = memory.update(pattern, action, success=True)

        # Weight should have extra boost from myelination
        expected_without_myelination = weight_before_threshold + 0.1 * (1 - weight_before_threshold)
        assert path.weight > expected_without_myelination

    def test_myelination_applies_repeatedly(self):
        """Myelination should apply on every success after threshold."""
        memory = HebbianPathMemory(
            alpha=0.1,
            beta=1.1,
            theta_myelination=5
        )

        pattern = frozenset(["col_duplicate"])
        action = "expert_fix"

        # Apply 10 successful updates
        for _ in range(10):
            memory.update(pattern, action, success=True)

        key = (pattern, action)
        path = memory.paths[key]

        # Usage count should be 10
        assert path.usage_count == 10

        # Weight should be very high due to repeated myelination
        assert path.weight > 0.95

    def test_weight_upper_bound(self):
        """Weight should not exceed max_weight."""
        memory = HebbianPathMemory(
            alpha=0.1,
            beta=1.2,
            theta_myelination=2,
            max_weight=0.99
        )

        pattern = frozenset(["box_duplicate"])
        action = "perfect_fix"

        # Apply many successful updates
        for _ in range(50):
            path = memory.update(pattern, action, success=True)

        # Should not exceed max_weight
        assert path.weight <= 0.99


class TestPathQuery:
    """Test querying paths from memory."""

    def test_query_empty_memory(self):
        """Querying empty memory should return empty list."""
        memory = HebbianPathMemory()

        violations = [
            Violation("row_duplicate", (0,), [1], [(0, 0), (0, 1)])
        ]

        paths = memory.query(violations)
        assert len(paths) == 0

    def test_query_matching_pattern(self):
        """Should return paths matching violation pattern."""
        memory = HebbianPathMemory()

        pattern = frozenset(["row_duplicate"])
        memory.update(pattern, "action1", success=True)
        memory.update(pattern, "action2", success=True)

        violations = [
            Violation("row_duplicate", (0,), [1], [(0, 0), (0, 1)])
        ]

        paths = memory.query(violations)
        assert len(paths) == 2

    def test_query_sorted_by_weight(self):
        """Should return paths sorted by weight (descending)."""
        memory = HebbianPathMemory()

        pattern = frozenset(["col_duplicate"])

        # Create three paths with different weights
        for _ in range(2):
            memory.update(pattern, "weak_fix", success=True)

        for _ in range(10):
            memory.update(pattern, "strong_fix", success=True)

        for _ in range(5):
            memory.update(pattern, "medium_fix", success=True)

        violations = [
            Violation("col_duplicate", (0,), [2], [(0, 0), (5, 0)])
        ]

        paths = memory.query(violations)

        # Should be sorted by weight
        assert len(paths) == 3
        assert paths[0].action == "strong_fix"
        assert paths[1].action == "medium_fix"
        assert paths[2].action == "weak_fix"

        # Verify ordering
        for i in range(len(paths) - 1):
            assert paths[i].weight >= paths[i + 1].weight

    def test_query_top_k(self):
        """Should return only top_k paths."""
        memory = HebbianPathMemory()

        pattern = frozenset(["box_duplicate"])

        # Create 10 paths
        for i in range(10):
            memory.update(pattern, f"action_{i}", success=True)

        violations = [
            Violation("box_duplicate", (0,), [3], [(0, 0), (1, 1)])
        ]

        paths = memory.query(violations, top_k=3)
        assert len(paths) == 3

    def test_query_no_violations(self):
        """Query with no violations should return empty list."""
        memory = HebbianPathMemory()

        pattern = frozenset(["row_duplicate"])
        memory.update(pattern, "action", success=True)

        paths = memory.query([])
        assert len(paths) == 0

    def test_query_multiple_violation_types(self):
        """Should match pattern with multiple violation types."""
        memory = HebbianPathMemory()

        pattern = frozenset(["row_duplicate", "col_duplicate"])
        memory.update(pattern, "complex_fix", success=True)

        # Single violation type shouldn't match
        violations1 = [
            Violation("row_duplicate", (0,), [1], [(0, 0), (0, 1)])
        ]
        paths1 = memory.query(violations1)
        assert len(paths1) == 0

        # Both types should match
        violations2 = [
            Violation("row_duplicate", (0,), [1], [(0, 0), (0, 1)]),
            Violation("col_duplicate", (1,), [2], [(0, 1), (5, 1)])
        ]
        paths2 = memory.query(violations2)
        assert len(paths2) == 1


class TestEpsilonGreedy:
    """Test ε-greedy path selection."""

    def test_select_from_empty_list(self):
        """Selecting from empty list should return None."""
        memory = HebbianPathMemory()
        path = memory.select_path([])
        assert path is None

    def test_greedy_selects_best(self):
        """With epsilon=0, should always select highest weight."""
        memory = HebbianPathMemory(epsilon=0.0)

        # Create paths with different weights
        paths = [
            Path(frozenset(["row_duplicate"]), "weak", True, weight=0.3),
            Path(frozenset(["row_duplicate"]), "strong", True, weight=0.9),
            Path(frozenset(["row_duplicate"]), "medium", True, weight=0.6),
        ]

        # Should always select strongest
        for _ in range(10):
            selected = memory.select_path(paths)
            assert selected.action == "strong"

    def test_exploration_randomizes(self):
        """With epsilon=1.0, should randomize selection."""
        np.random.seed(42)
        memory = HebbianPathMemory(epsilon=1.0)

        paths = [
            Path(frozenset(["row_duplicate"]), f"action_{i}", True, weight=0.5)
            for i in range(5)
        ]

        # Collect selections
        selections = [memory.select_path(paths).action for _ in range(100)]

        # Should have variety (not all the same)
        unique_selections = len(set(selections))
        assert unique_selections > 1

    def test_epsilon_decay(self):
        """Epsilon should decay over time."""
        memory = HebbianPathMemory(epsilon=0.9, epsilon_decay=0.9, min_epsilon=0.05)

        initial_epsilon = memory.epsilon

        # Decay 10 times
        for _ in range(10):
            memory.decay_epsilon()

        assert memory.epsilon < initial_epsilon
        assert memory.epsilon >= 0.05  # Should not go below min

    def test_epsilon_min_bound(self):
        """Epsilon should not decay below min_epsilon."""
        memory = HebbianPathMemory(epsilon=0.5, epsilon_decay=0.5, min_epsilon=0.1)

        # Decay many times
        for _ in range(100):
            memory.decay_epsilon()

        assert memory.epsilon == 0.1


class TestStatistics:
    """Test statistics functionality."""

    def test_empty_memory_stats(self):
        """Empty memory should return valid statistics."""
        memory = HebbianPathMemory()
        stats = memory.get_statistics()

        assert stats["num_paths"] == 0
        assert stats["avg_weight"] == 0.0
        assert stats["total_updates"] == 0

    def test_statistics_after_updates(self):
        """Statistics should reflect updates."""
        memory = HebbianPathMemory()

        pattern = frozenset(["row_duplicate"])

        # 7 successes, 3 failures
        for _ in range(7):
            memory.update(pattern, "action1", success=True)

        for _ in range(3):
            memory.update(pattern, "action2", success=False)

        stats = memory.get_statistics()

        assert stats["num_paths"] == 2
        assert stats["total_updates"] == 10
        assert stats["successful_updates"] == 7
        assert stats["failed_updates"] == 3
        assert abs(stats["success_rate"] - 0.7) < 0.01

    def test_top_paths(self):
        """Should return top N strongest paths."""
        memory = HebbianPathMemory()

        # Create paths with varying strengths
        for i in range(10):
            pattern = frozenset([f"type_{i}"])
            for _ in range(i + 1):  # Different number of successes
                memory.update(pattern, f"action_{i}", success=True)

        top_paths = memory.get_top_paths(n=3)

        assert len(top_paths) == 3
        # Should be sorted by weight
        for i in range(2):
            assert top_paths[i].weight >= top_paths[i + 1].weight

    def test_pattern_diversity(self):
        """Should count unique violation patterns."""
        memory = HebbianPathMemory()

        # Create paths with 3 unique patterns
        memory.update(frozenset(["row_duplicate"]), "action1", True)
        memory.update(frozenset(["row_duplicate"]), "action2", True)
        memory.update(frozenset(["col_duplicate"]), "action3", True)
        memory.update(frozenset(["box_duplicate"]), "action4", True)

        diversity = memory.get_pattern_diversity()
        assert diversity == 3

    def test_strong_paths_count(self):
        """Should count paths with weight > 0.8."""
        memory = HebbianPathMemory(alpha=0.2)

        # Create one strong path
        pattern1 = frozenset(["row_duplicate"])
        for _ in range(15):
            memory.update(pattern1, "strong_action", success=True)

        # Create one weak path
        pattern2 = frozenset(["col_duplicate"])
        memory.update(pattern2, "weak_action", success=True)

        stats = memory.get_statistics()
        assert stats["strong_paths"] >= 1


class TestPruning:
    """Test pruning weak paths."""

    def test_prune_weak_paths(self):
        """Should remove paths below threshold."""
        memory = HebbianPathMemory()

        # Create strong path
        pattern1 = frozenset(["row_duplicate"])
        for _ in range(10):
            memory.update(pattern1, "strong", success=True)

        # Create weak path (many failures to make it weak)
        pattern2 = frozenset(["col_duplicate"])
        memory.update(pattern2, "weak", success=True)
        for _ in range(15):
            memory.update(pattern2, "weak", success=False)

        initial_count = len(memory)

        # Prune paths below 0.3
        removed = memory.prune_weak_paths(threshold=0.3)

        assert removed > 0
        assert len(memory) < initial_count
        assert len(memory) == 1  # Only strong path should remain


class TestSerialization:
    """Test saving and loading path memory."""

    def test_save_to_dict(self):
        """Should serialize to dictionary."""
        memory = HebbianPathMemory(alpha=0.15, gamma=0.93)

        pattern = frozenset(["row_duplicate"])
        memory.update(pattern, "action", success=True)

        data = memory.save_to_dict()

        assert "hyperparameters" in data
        assert "paths" in data
        assert "statistics" in data
        assert data["hyperparameters"]["alpha"] == 0.15
        assert data["hyperparameters"]["gamma"] == 0.93

    def test_load_from_dict(self):
        """Should deserialize from dictionary."""
        memory1 = HebbianPathMemory(alpha=0.12, gamma=0.94)

        pattern = frozenset(["col_duplicate"])
        for _ in range(5):
            memory1.update(pattern, "test_action", success=True)

        # Serialize
        data = memory1.save_to_dict()

        # Deserialize
        memory2 = HebbianPathMemory.load_from_dict(data)

        # Check hyperparameters
        assert memory2.alpha == 0.12
        assert memory2.gamma == 0.94

        # Check paths
        assert len(memory2) == len(memory1)

        # Check path contents
        key = (pattern, "test_action")
        assert key in memory2.paths
        assert memory2.paths[key].weight == memory1.paths[key].weight
        assert memory2.paths[key].usage_count == memory1.paths[key].usage_count

    def test_round_trip(self):
        """Save and load should preserve memory."""
        memory1 = HebbianPathMemory()

        # Create diverse paths
        for i in range(5):
            pattern = frozenset([f"type_{i % 3}"])
            for _ in range(i + 1):
                memory1.update(pattern, f"action_{i}", success=(i % 2 == 0))

        # Round trip
        data = memory1.save_to_dict()
        memory2 = HebbianPathMemory.load_from_dict(data)

        # Should have same statistics
        stats1 = memory1.get_statistics()
        stats2 = memory2.get_statistics()

        assert stats1["num_paths"] == stats2["num_paths"]
        assert abs(stats1["avg_weight"] - stats2["avg_weight"]) < 1e-6


class TestRepr:
    """Test string representations."""

    def test_path_repr(self):
        """Path should have readable repr."""
        path = Path(
            violation_pattern=frozenset(["row_duplicate"]),
            action="fix_cell",
            result=True,
            weight=0.87,
            usage_count=15
        )

        repr_str = repr(path)
        assert "row_duplicate" in repr_str
        assert "fix_cell" in repr_str
        assert "0.87" in repr_str or "0.870" in repr_str

    def test_memory_repr(self):
        """Memory should have readable repr."""
        memory = HebbianPathMemory()

        pattern = frozenset(["row_duplicate"])
        for _ in range(5):
            memory.update(pattern, "action", success=True)

        repr_str = repr(memory)
        assert "HebbianPathMemory" in repr_str
        assert "paths" in repr_str.lower()
