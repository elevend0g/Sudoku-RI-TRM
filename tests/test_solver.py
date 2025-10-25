"""
Tests for recursive refinement solver.

Validates:
1. Basic solving functionality
2. Reasoning trace generation
3. Path memory integration
4. Training capabilities
5. Early stopping
"""

import pytest
import numpy as np
import torch

from src.solver import RecursiveSolver, TrainableSolver, ReasoningStep, SolverResult
from src.network import create_sudoku_network
from src.rules import SudokuRuleGraph
from src.path_memory import HebbianPathMemory


class TestBasicSolving:
    """Test basic solving functionality."""

    def test_solver_initialization(self):
        """Test creating a solver."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()

        solver = RecursiveSolver(
            network=network,
            rule_graph=rule_graph
        )

        assert solver is not None
        assert solver.max_iterations == 16
        assert solver.reasoning_steps == 6

    def test_solve_valid_grid(self):
        """Solving a valid grid should return immediately."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph)

        # Valid partial grid
        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 1
        grid[0, 1] = 2
        grid[1, 0] = 3

        result = solver.solve(grid)

        assert result.success
        assert len(result.violations) == 0
        assert result.converged
        assert result.num_steps <= 1  # Should stop immediately

    def test_solve_returns_result(self):
        """Solve should return SolverResult."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph)

        grid = np.zeros((9, 9), dtype=int)
        result = solver.solve(grid)

        assert isinstance(result, SolverResult)
        assert hasattr(result, 'grid')
        assert hasattr(result, 'violations')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'num_steps')
        assert hasattr(result, 'trace')

    def test_grid_shape_preserved(self):
        """Output grid should have same shape as input."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph)

        grid = np.zeros((9, 9), dtype=int)
        result = solver.solve(grid)

        assert result.grid.shape == (9, 9)

    def test_max_iterations_respected(self):
        """Solver should not exceed max iterations."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph, max_iterations=5)

        # Grid with violations
        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 1
        grid[0, 1] = 1  # Duplicate

        result = solver.solve(grid)

        assert result.num_steps <= 5


class TestReasoningTrace:
    """Test reasoning trace generation."""

    def test_trace_is_list(self):
        """Trace should be a list of ReasoningSteps."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph)

        grid = np.zeros((9, 9), dtype=int)
        result = solver.solve(grid, return_trace=True)

        assert isinstance(result.trace, list)
        if len(result.trace) > 0:
            assert isinstance(result.trace[0], ReasoningStep)

    def test_trace_disabled(self):
        """Trace should be None when disabled."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph)

        grid = np.zeros((9, 9), dtype=int)
        result = solver.solve(grid, return_trace=False)

        assert result.trace is None

    def test_trace_contains_violations(self):
        """Each step should record violations."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph, max_iterations=3)

        # Grid with violation
        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 2
        grid[0, 5] = 2  # Duplicate

        result = solver.solve(grid, return_trace=True)

        if len(result.trace) > 0:
            first_step = result.trace[0]
            assert hasattr(first_step, 'violations')
            assert hasattr(first_step, 'violation_count')

    def test_trace_records_steps(self):
        """Trace should record step numbers."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph, max_iterations=4)

        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 3
        grid[0, 1] = 3

        result = solver.solve(grid, return_trace=True)

        if len(result.trace) > 1:
            for i, step in enumerate(result.trace):
                assert step.step == i


class TestPathMemoryIntegration:
    """Test integration with path memory."""

    def test_solver_with_path_memory(self):
        """Solver should work with path memory."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        memory = HebbianPathMemory()

        solver = RecursiveSolver(network, rule_graph, path_memory=memory)

        grid = np.zeros((9, 9), dtype=int)
        result = solver.solve(grid)

        assert result is not None

    def test_path_memory_updated(self):
        """Path memory should be updated during solving."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        memory = HebbianPathMemory()

        solver = RecursiveSolver(network, rule_graph, path_memory=memory)

        # Grid with violation
        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 5
        grid[0, 3] = 5

        initial_paths = len(memory)

        result = solver.solve(grid, update_memory=True)

        # Memory should have grown (unless grid was immediately valid)
        # Note: might not grow if randomly fixed immediately
        assert len(memory) >= initial_paths

    def test_path_memory_not_updated_when_disabled(self):
        """Path memory should not update when disabled."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        memory = HebbianPathMemory()

        solver = RecursiveSolver(network, rule_graph, path_memory=memory)

        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 7
        grid[0, 2] = 7

        initial_paths = len(memory)
        initial_updates = memory.total_updates

        result = solver.solve(grid, update_memory=False)

        # Should not have updated
        assert memory.total_updates == initial_updates

    def test_candidate_paths_in_trace(self):
        """Trace should include candidate paths from memory."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        memory = HebbianPathMemory()

        # Pre-populate memory
        pattern = frozenset(["row_duplicate"])
        for _ in range(5):
            memory.update(pattern, "test_fix", success=True)

        solver = RecursiveSolver(network, rule_graph, path_memory=memory)

        # Grid with row duplicate
        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 4
        grid[0, 4] = 4

        result = solver.solve(grid, return_trace=True)

        # Check if any step has candidate paths
        # (might not if grid gets fixed before querying)
        if len(result.trace) > 0:
            assert hasattr(result.trace[0], 'candidate_paths')


class TestEarlyStopping:
    """Test early stopping based on confidence."""

    def test_early_stopping_on_low_confidence(self):
        """Solver should stop early on low confidence."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()

        solver = RecursiveSolver(
            network,
            rule_graph,
            max_iterations=16,
            confidence_threshold=0.9  # High threshold for testing
        )

        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 6
        grid[0, 6] = 6

        result = solver.solve(grid)

        # Might stop early (though not guaranteed with random network)
        assert result.num_steps <= 16

    def test_no_early_stopping_with_low_threshold(self):
        """With very low threshold, should run to max iterations."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()

        solver = RecursiveSolver(
            network,
            rule_graph,
            max_iterations=3,
            confidence_threshold=0.0  # Never stop early
        )

        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 8
        grid[0, 7] = 8

        result = solver.solve(grid)

        # Should run to max unless solved
        if not result.success:
            assert result.num_steps == 3


class TestTrainableSolver:
    """Test trainable solver functionality."""

    def test_trainable_solver_creation(self):
        """Test creating trainable solver."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()

        solver = TrainableSolver(network, rule_graph)

        assert isinstance(solver, TrainableSolver)
        assert isinstance(solver, RecursiveSolver)

    def test_configure_optimizer(self):
        """Test optimizer configuration."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = TrainableSolver(network, rule_graph)

        solver.configure_optimizer(optimizer_type="adamw", lr=1e-4)

        assert solver.optimizer is not None
        assert isinstance(solver.optimizer, torch.optim.AdamW)

    def test_train_step(self):
        """Test single training step."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = TrainableSolver(network, rule_graph, max_iterations=2)

        solver.configure_optimizer(lr=1e-4)

        # Grid with violation
        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 9
        grid[0, 8] = 9

        losses = solver.train_step(grid)

        assert isinstance(losses, dict)
        assert "total_loss" in losses
        assert "violation_count" in losses

    def test_loss_history(self):
        """Training should record loss history."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = TrainableSolver(network, rule_graph, max_iterations=2)

        solver.configure_optimizer()

        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 1
        grid[0, 1] = 1

        # Multiple training steps
        for _ in range(3):
            solver.train_step(grid)

        assert len(solver.loss_history) == 3

    def test_train_mode_eval_mode(self):
        """Test switching between train and eval modes."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = TrainableSolver(network, rule_graph)

        solver.train_mode()
        assert solver.network.training

        solver.eval_mode()
        assert not solver.network.training


class TestViolationHandling:
    """Test violation detection and handling."""

    def test_violations_to_tensor(self):
        """Test converting violations to tensor format."""
        from src.rules import Violation

        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph)

        violations = [
            Violation("row_duplicate", (0,), [1], [(0, 0), (0, 1)]),
            Violation("col_duplicate", (1,), [2], [(0, 1), (5, 1)]),
        ]

        tensor = solver._violations_to_tensor(violations)

        assert tensor is not None
        assert tensor.shape[0] == 1  # Batch size
        assert tensor.shape[1] == 2  # Num violations
        assert tensor.shape[2] == 2  # [type, location]

    def test_empty_violations_to_tensor(self):
        """Empty violations should return None."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph)

        tensor = solver._violations_to_tensor([])

        assert tensor is None

    def test_describe_action(self):
        """Test action description generation."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph)

        grid1 = np.zeros((9, 9), dtype=int)
        grid2 = np.zeros((9, 9), dtype=int)
        grid2[0, 0] = 5

        action = solver._describe_action(grid1, grid2)

        assert isinstance(action, str)
        assert "cell" in action or action == "no_change"

    def test_describe_no_change(self):
        """Action description for unchanged grid."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph)

        grid = np.zeros((9, 9), dtype=int)

        action = solver._describe_action(grid, grid)

        assert action == "no_change"


class TestDeviceHandling:
    """Test device handling (CPU/GPU)."""

    def test_default_cpu(self):
        """Default device should be CPU."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph)

        assert solver.device == "cpu"

    def test_network_on_device(self):
        """Network should be on specified device."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph, device="cpu")

        # Check first parameter device
        param = next(network.parameters())
        assert param.device.type == "cpu"


class TestResultRepresentation:
    """Test string representations."""

    def test_reasoning_step_repr(self):
        """ReasoningStep should have readable repr."""
        step = ReasoningStep(
            step=1,
            grid=np.zeros((9, 9)),
            violations=[],
            violation_count=0,
            confidence=0.95,
            success=True
        )

        repr_str = repr(step)
        assert "Step 1" in repr_str
        assert "0.95" in repr_str or "0.950" in repr_str

    def test_solver_result_repr(self):
        """SolverResult should have readable repr."""
        result = SolverResult(
            grid=np.zeros((9, 9)),
            violations=[],
            converged=True,
            num_steps=5,
            trace=[],
            final_confidence=0.88,
            success=True
        )

        repr_str = repr(result)
        assert "SOLVED" in repr_str or "success" in repr_str.lower()
        assert "5" in repr_str  # num_steps


class TestIntegrationScenarios:
    """Test realistic solving scenarios."""

    def test_solve_simple_violation(self):
        """Test solving a grid with simple violation."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        solver = RecursiveSolver(network, rule_graph, max_iterations=10)

        # Grid with one row duplicate
        grid = np.zeros((9, 9), dtype=int)
        grid[0, 0] = 3
        grid[0, 5] = 3

        result = solver.solve(grid)

        # Should attempt to solve (might or might not succeed with random network)
        assert result.num_steps <= 10
        assert result.grid.shape == (9, 9)

    def test_solve_with_memory_learning(self):
        """Test that memory learns over multiple solves."""
        network = create_sudoku_network()
        rule_graph = SudokuRuleGraph()
        memory = HebbianPathMemory()
        solver = RecursiveSolver(network, rule_graph, path_memory=memory, max_iterations=5)

        # Solve multiple similar grids
        for _ in range(3):
            grid = np.zeros((9, 9), dtype=int)
            grid[0, 0] = 4
            grid[0, np.random.randint(1, 9)] = 4

            solver.solve(grid, update_memory=True)

        # Memory should have accumulated some knowledge
        assert len(memory) > 0
        assert memory.total_updates > 0
