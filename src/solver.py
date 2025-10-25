"""
Recursive refinement solver for RI-TRM.

Implements Algorithm 1 from the RI-TRM paper:
- Iterative refinement with rule verification
- Path memory integration
- Confidence-based early stopping
- Interpretable reasoning traces
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np
import torch

from src.rules import SudokuRuleGraph, Violation
from src.path_memory import HebbianPathMemory
from src.network import TinyRecursiveNetwork


@dataclass
class ReasoningStep:
    """
    Single step in the reasoning trace.

    Provides full interpretability of the solving process.
    """
    step: int
    grid: np.ndarray
    violations: List[Violation]
    violation_count: int
    action_taken: Optional[str] = None
    candidate_paths: List = field(default_factory=list)
    selected_path: Optional = None
    confidence: float = 0.0
    success: bool = False

    def __repr__(self):
        return (
            f"Step {self.step}: {self.violation_count} violations, "
            f"confidence={self.confidence:.3f}, success={self.success}"
        )


@dataclass
class SolverResult:
    """
    Result from recursive refinement.

    Contains final grid, reasoning trace, and metadata.
    """
    grid: np.ndarray
    violations: List[Violation]
    converged: bool
    num_steps: int
    trace: List[ReasoningStep]
    final_confidence: float
    success: bool  # True if no violations

    def __repr__(self):
        status = "✓ SOLVED" if self.success else f"✗ {len(self.violations)} violations"
        return (
            f"SolverResult({status}, {self.num_steps} steps, "
            f"confidence={self.final_confidence:.3f})"
        )


class RecursiveSolver:
    """
    Recursive refinement solver for Sudoku using RI-TRM.

    Implements Algorithm 1 from the paper:
    1. Verify current solution using rules (K_R)
    2. If no violations, success!
    3. Query path memory for candidate fixes (K_P)
    4. Recursive reasoning: n steps through network
    5. Generate improved solution
    6. Update path memory based on success
    7. Repeat until convergence or max iterations

    The key insight: Train on reducing violations, not matching ground truth.
    """

    def __init__(
        self,
        network: TinyRecursiveNetwork,
        rule_graph: SudokuRuleGraph,
        path_memory: Optional[HebbianPathMemory] = None,
        max_iterations: int = 16,
        reasoning_steps: int = 6,
        confidence_threshold: float = 0.1,
        device: str = "cpu",
    ):
        """
        Initialize recursive solver.

        Args:
            network: TinyRecursiveNetwork for reasoning
            rule_graph: SudokuRuleGraph for verification (K_R)
            path_memory: HebbianPathMemory for learned patterns (K_P)
            max_iterations: Maximum refinement iterations (N_sup)
            reasoning_steps: Number of recursive reasoning steps (n)
            confidence_threshold: Early stopping threshold
            device: Device for network computation
        """
        self.network = network
        self.rule_graph = rule_graph
        self.path_memory = path_memory
        self.max_iterations = max_iterations
        self.reasoning_steps = reasoning_steps
        self.confidence_threshold = confidence_threshold
        self.device = device

        # Move network to device
        self.network.to(device)
        self.network.eval()  # Start in eval mode

    def solve(
        self,
        grid: np.ndarray,
        return_trace: bool = True,
        update_memory: bool = True
    ) -> SolverResult:
        """
        Solve Sudoku puzzle using recursive refinement.

        Args:
            grid: 9x9 numpy array (0 = empty)
            return_trace: Whether to return full reasoning trace
            update_memory: Whether to update path memory

        Returns:
            SolverResult with solution and trace
        """
        # Initialize
        current_grid = grid.copy()
        reasoning_state = None
        trace = [] if return_trace else None

        # Track original clues (non-zero cells) - these should NEVER be modified
        self.original_clues = (grid != 0)

        # Main refinement loop
        for step in range(self.max_iterations):
            # Step 1: Verify current solution (K_R)
            violations = self.rule_graph.verify(current_grid)
            violation_count = len(violations)

            # Check for success
            if violation_count == 0:
                if return_trace:
                    trace.append(ReasoningStep(
                        step=step,
                        grid=current_grid.copy(),
                        violations=[],
                        violation_count=0,
                        confidence=1.0,
                        success=True
                    ))
                return SolverResult(
                    grid=current_grid,
                    violations=[],
                    converged=True,
                    num_steps=step + 1,
                    trace=trace,
                    final_confidence=1.0,
                    success=True
                )

            # Step 2: Query path memory for candidate fixes (K_P)
            candidate_paths = []
            selected_path = None
            if self.path_memory is not None:
                candidate_paths = self.path_memory.query(violations, top_k=5)
                if candidate_paths:
                    selected_path = self.path_memory.select_path(candidate_paths)

            # Step 3: Recursive reasoning through network
            grid_improved, confidence, reasoning_state = self._refine_step(
                current_grid,
                violations,
                reasoning_state
            )

            # Record step in trace
            if return_trace:
                trace.append(ReasoningStep(
                    step=step,
                    grid=current_grid.copy(),
                    violations=violations.copy(),
                    violation_count=violation_count,
                    candidate_paths=candidate_paths,
                    selected_path=selected_path,
                    confidence=confidence,
                    success=False
                ))

            # Step 4: Update path memory (Hebbian learning)
            if update_memory and self.path_memory is not None:
                # Check if violations decreased
                new_violations = self.rule_graph.verify(grid_improved)
                success = len(new_violations) < violation_count

                # Create violation pattern
                violation_pattern = frozenset(v.type for v in violations)

                # Create action description
                action = self._describe_action(current_grid, grid_improved)

                # Update memory
                self.path_memory.update(violation_pattern, action, success)

            # Update grid
            current_grid = grid_improved

            # Step 5: Early stopping based on confidence
            if confidence < self.confidence_threshold:
                break

        # Didn't converge
        final_violations = self.rule_graph.verify(current_grid)
        return SolverResult(
            grid=current_grid,
            violations=final_violations,
            converged=False,
            num_steps=step + 1,
            trace=trace,
            final_confidence=confidence,
            success=len(final_violations) == 0
        )

    def _get_violation_cells(self, violations: List[Violation]) -> set:
        """
        Get all cells involved in violations.

        Args:
            violations: List of Violation objects

        Returns:
            Set of (row, col) tuples for cells involved in violations
        """
        violation_cells = set()
        for violation in violations:
            for cell in violation.cells:
                violation_cells.add(cell)
        return violation_cells

    def _get_valid_values(self, grid: np.ndarray, row: int, col: int) -> set:
        """
        Get values that won't violate constraints if placed at (row, col).

        Args:
            grid: Current grid
            row, col: Position to check

        Returns:
            Set of valid values (1-9)
        """
        used = set()

        # Check row
        used.update(grid[row, :])

        # Check column
        used.update(grid[:, col])

        # Check 3x3 box
        box_row, box_col = (row // 3) * 3, (col // 3) * 3
        used.update(grid[box_row:box_row+3, box_col:box_col+3].flatten())

        # Remove 0 (empty) and return valid values
        used.discard(0)
        valid = set(range(1, 10)) - used

        return valid

    def _refine_step(
        self,
        grid: np.ndarray,
        violations: List[Violation],
        reasoning_state: Optional[torch.Tensor]
    ) -> Tuple[np.ndarray, float, torch.Tensor]:
        """
        Perform one refinement step.

        Args:
            grid: Current grid (9, 9)
            violations: Current violations
            reasoning_state: Previous reasoning state

        Returns:
            improved_grid: (9, 9) numpy array
            confidence: float
            reasoning_state: updated state
        """
        # Convert grid to tensor
        grid_flat = torch.from_numpy(grid.flatten()).long().unsqueeze(0)  # (1, 81)
        grid_flat = grid_flat.to(self.device)

        # Convert violations to tensor
        violations_tensor = self._violations_to_tensor(violations)
        if violations_tensor is not None:
            violations_tensor = violations_tensor.to(self.device)

        # Recursive reasoning: n steps through network
        with torch.no_grad():
            for _ in range(self.reasoning_steps):
                logits, confidence, reasoning_state = self.network(
                    grid_flat,
                    violations_tensor,
                    reasoning_state
                )

        # Create mask: only modify cells involved in violations that are not original clues
        violation_cells = self._get_violation_cells(violations)
        edit_mask = np.zeros((9, 9), dtype=bool)
        for row, col in violation_cells:
            # Only allow editing if NOT an original clue
            if not self.original_clues[row, col]:
                edit_mask[row, col] = True

        edit_mask_flat = torch.from_numpy(edit_mask.flatten()).to(self.device)

        # Sample improved grid from logits - but ONLY for editable cells
        improved_flat = grid_flat.squeeze(0).clone()  # Start with current grid

        if edit_mask_flat.any():
            # Only sample cells that are editable
            probs = torch.softmax(logits.squeeze(0)[edit_mask_flat] / 1.0, dim=-1)
            sampled_values = torch.multinomial(probs, num_samples=1).squeeze(-1)
            improved_flat[edit_mask_flat] = sampled_values

        # Convert back to numpy
        improved_grid = improved_flat.cpu().numpy().reshape(9, 9)
        confidence_val = confidence.item()

        return improved_grid, confidence_val, reasoning_state

    def _violations_to_tensor(
        self,
        violations: List[Violation]
    ) -> Optional[torch.Tensor]:
        """
        Convert violations to tensor format.

        Args:
            violations: List of Violation objects

        Returns:
            Tensor of shape (1, num_violations, 2) or None
        """
        if len(violations) == 0:
            return None

        # Map violation types to indices
        type_map = {
            "row_duplicate": 0,
            "col_duplicate": 1,
            "box_duplicate": 2,
        }

        # Extract type and location
        violation_data = []
        for v in violations:
            v_type = type_map.get(v.type, 0)
            # Use first cell location
            v_loc = v.cells[0][0] * 9 + v.cells[0][1] if v.cells else 0
            violation_data.append([v_type, v_loc])

        # Convert to tensor
        tensor = torch.tensor(violation_data, dtype=torch.long).unsqueeze(0)
        return tensor

    def _describe_action(
        self,
        old_grid: np.ndarray,
        new_grid: np.ndarray
    ) -> str:
        """
        Describe the action taken (for path memory).

        Args:
            old_grid: Grid before action
            new_grid: Grid after action

        Returns:
            Action description string
        """
        # Find changed cells
        diff = (old_grid != new_grid)
        changed_positions = np.argwhere(diff)

        if len(changed_positions) == 0:
            return "no_change"

        # Describe first change (for simplicity)
        row, col = changed_positions[0]
        old_val = old_grid[row, col]
        new_val = new_grid[row, col]

        return f"cell_{row}_{col}_from_{old_val}_to_{new_val}"

    def train_mode(self):
        """Set network to training mode."""
        self.network.train()

    def eval_mode(self):
        """Set network to evaluation mode."""
        self.network.eval()


class TrainableSolver(RecursiveSolver):
    """
    Solver with training capabilities.

    Extends RecursiveSolver to support gradient-based training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.loss_history = []

    def configure_optimizer(
        self,
        optimizer_type: str = "adamw",
        lr: float = 1e-4,
        weight_decay: float = 0.01
    ):
        """
        Configure optimizer for training.

        Args:
            optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
            lr: Learning rate
            weight_decay: Weight decay
        """
        if optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=lr
            )
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def train_step(
        self,
        grid: np.ndarray,
        target_grid: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            grid: Input grid with violations
            target_grid: Optional target (for supervised training)

        Returns:
            Dictionary of losses
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not configured. Call configure_optimizer() first.")

        self.network.train()

        # Track original clues (same as in solve method)
        self.original_clues = (grid != 0)

        # Convert to tensor
        grid_flat = torch.from_numpy(grid.flatten()).long().unsqueeze(0)
        grid_flat = grid_flat.to(self.device)

        # Get violations
        violations = self.rule_graph.verify(grid)
        violations_tensor = self._violations_to_tensor(violations)
        if violations_tensor is not None:
            violations_tensor = violations_tensor.to(self.device)

        # Forward pass
        reasoning_state = None
        total_loss = 0.0

        # Track initial violation count for reward signal
        initial_violation_count = len(violations)

        # Deep supervision: train at each step
        for i in range(self.max_iterations):
            logits, confidence, reasoning_state = self.network(
                grid_flat,
                violations_tensor,
                reasoning_state
            )

            # Loss 1: Violation reduction loss (now includes preservation penalty)
            violation_loss = self._violation_loss(
                logits,
                grid_flat,
                violations
            )

            # Loss 2: Confidence calibration
            # High confidence when no violations
            target_confidence = 1.0 if len(violations) == 0 else 0.0
            confidence_loss = torch.nn.functional.mse_loss(
                confidence,
                torch.tensor([target_confidence], device=self.device)
            )

            # Total loss for this step
            step_loss = violation_loss + 0.1 * confidence_loss
            total_loss += step_loss

            # Sample next grid - use masking like in _refine_step
            with torch.no_grad():
                violation_cells = self._get_violation_cells(violations)
                edit_mask = np.zeros((9, 9), dtype=bool)
                for row, col in violation_cells:
                    if not self.original_clues[row, col]:
                        edit_mask[row, col] = True

                edit_mask_flat = torch.from_numpy(edit_mask.flatten()).to(self.device)

                # Start with current grid
                new_grid_flat = grid_flat.squeeze(0).clone()

                if edit_mask_flat.any():
                    # Only sample cells that are editable
                    probs = torch.softmax(logits.squeeze(0)[edit_mask_flat], dim=-1)
                    sampled_values = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    new_grid_flat[edit_mask_flat] = sampled_values

                grid_flat = new_grid_flat.unsqueeze(0)

            # Update violations
            new_grid = grid_flat.squeeze(0).cpu().numpy().reshape(9, 9)
            new_violations = self.rule_graph.verify(new_grid)

            # Violation reduction reward: strong signal for improvement
            # This is detached from the computation graph but influences total_loss
            violation_delta = initial_violation_count - len(new_violations)
            if violation_delta > 0:
                # Reduced violations - apply reward (negative loss)
                reward = -violation_delta * 5.0
                total_loss = total_loss + torch.tensor(reward, device=self.device, requires_grad=True)
            elif violation_delta < 0:
                # Increased violations - heavy penalty
                penalty = -violation_delta * 20.0
                total_loss = total_loss + torch.tensor(penalty, device=self.device, requires_grad=True)

            violations = new_violations
            violations_tensor = self._violations_to_tensor(violations)
            if violations_tensor is not None:
                violations_tensor = violations_tensor.to(self.device)

            # Update initial count for next iteration
            initial_violation_count = len(violations)

            # Early stop if solved
            if len(violations) == 0:
                break

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        # Record loss
        loss_val = total_loss.item()
        self.loss_history.append(loss_val)

        return {
            "total_loss": loss_val,
            "violation_count": len(violations),
        }

    def _violation_loss(
        self,
        logits: torch.Tensor,
        current_grid: torch.Tensor,
        violations: List[Violation]
    ) -> torch.Tensor:
        """
        Compute loss based on violation reduction.

        Args:
            logits: (1, 81, 10)
            current_grid: (1, 81)
            violations: List of violations

        Returns:
            Loss tensor
        """
        if len(violations) == 0:
            # No violations - minimize changes
            target = current_grid.squeeze(0)
            loss = torch.nn.functional.cross_entropy(
                logits.squeeze(0),
                target
            )
            return loss * 0.1

        # Get cells involved in violations
        violation_cells = self._get_violation_cells(violations)
        violation_positions = set()
        for row, col in violation_cells:
            violation_positions.add(row * 9 + col)

        # Compute loss that encourages fixing violations
        violation_loss = 0.0
        violation_count = 0

        # Get current grid as numpy for constraint checking
        current_grid_np = current_grid.squeeze(0).cpu().numpy().reshape(9, 9)

        for pos in violation_positions:
            # Only consider non-clue cells
            row, col = pos // 9, pos % 9
            if hasattr(self, 'original_clues') and self.original_clues[row, col]:
                continue  # Don't try to change original clues

            current_val = current_grid[0, pos].item()

            # Get valid values for this position
            valid_values = self._get_valid_values(current_grid_np, row, col)

            if len(valid_values) > 0:
                # Encourage network to predict valid values
                # Create target distribution favoring valid values
                target_dist = torch.zeros(10, device=self.device)
                for v in valid_values:
                    target_dist[v] = 1.0 / len(valid_values)

                # KL divergence to encourage valid value distribution
                pred_probs = torch.softmax(logits[0, pos], dim=0)
                kl_loss = torch.nn.functional.kl_div(
                    pred_probs.log(),
                    target_dist,
                    reduction='sum'
                )
                violation_loss += kl_loss
                violation_count += 1
            else:
                # No valid values available (overconstrained)
                # Encourage changing to 0 (empty) or anything != current
                logits_cell = logits[0, pos].clone()
                logits_cell[current_val] = float('-inf')

                # Loss: maximize probability of changing
                prob_change = torch.softmax(logits_cell, dim=0).sum()
                violation_loss += -torch.log(prob_change + 1e-8)
                violation_count += 1

        if violation_count > 0:
            violation_loss = violation_loss / violation_count

        # Preservation loss: penalize changing non-violated cells
        preservation_loss = 0.0
        preservation_count = 0

        for pos in range(81):
            if pos not in violation_positions:
                # This cell should be preserved
                target_val = current_grid[0, pos]
                cell_loss = torch.nn.functional.cross_entropy(
                    logits[0, pos].unsqueeze(0),
                    target_val.unsqueeze(0)
                )
                preservation_loss += cell_loss
                preservation_count += 1

        if preservation_count > 0:
            preservation_loss = preservation_loss / preservation_count

        # Combine losses with balanced weight on preservation
        # Reduced from 10.0 to 2.0 to allow more exploration
        total_loss = violation_loss + 2.0 * preservation_loss

        return total_loss
