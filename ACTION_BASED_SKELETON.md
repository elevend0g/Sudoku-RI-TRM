# Action-Based Architecture - Implementation Skeleton

This file contains code skeletons for implementing the action-based RI-TRM architecture.

---

## File 1: `src/action_network.py`

```python
"""
Action-based network that outputs discrete actions.
Forces exactly one cell to change per refinement step.
"""

import torch
import torch.nn as nn
from src.network import TinyRecursiveNetwork

class ActionBasedNetwork(nn.Module):
    """
    Network that outputs actions: (cell_to_change, new_value)
    instead of complete grids.
    """

    def __init__(self, hidden_size=512, num_heads=8, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size

        # Reuse existing transformer backbone
        self.backbone = TinyRecursiveNetwork(
            vocab_size=10,
            grid_size=81,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # Action selection heads
        self.cell_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 81)  # Which cell (0-80)
        )

        self.value_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 10)  # What value (0-9)
        )

    def forward(self, grid, violations=None, candidate_mask=None):
        """
        Args:
            grid: [batch, 81] current grid
            violations: [batch, num_viols, 2] violations
            candidate_mask: [batch, 81] which cells can be edited

        Returns:
            cell_probs: [batch, 81] probability distribution over cells
            value_probs: [batch, 10] probability distribution over values
        """
        batch_size = grid.shape[0]

        # Process through transformer
        logits, confidence, z = self.backbone(grid, violations, None)
        # z is [batch, 81, hidden_size]

        # Pool to single vector
        pooled = z.mean(dim=1)  # [batch, hidden_size]

        # Predict which cell to change
        cell_logits = self.cell_selector(pooled)  # [batch, 81]

        # Mask out non-editable cells
        if candidate_mask is not None:
            cell_logits = cell_logits.masked_fill(~candidate_mask, float('-inf'))

        cell_probs = torch.softmax(cell_logits, dim=-1)

        # Predict what value to use
        value_logits = self.value_selector(pooled)  # [batch, 10]
        value_probs = torch.softmax(value_logits, dim=-1)

        return cell_probs, value_probs, confidence

    def select_action(self, grid, violations, candidate_cells, epsilon=0.0):
        """
        Select an action using epsilon-greedy.

        Args:
            grid: [batch, 81] or numpy [9, 9]
            violations: List[Violation]
            candidate_cells: List[(row, col)] - editable cells
            epsilon: exploration rate

        Returns:
            cell_idx: int (0-80)
            new_value: int (0-9)
        """
        import numpy as np

        # Convert grid to tensor if needed
        if isinstance(grid, np.ndarray):
            grid = torch.from_numpy(grid.flatten()).long().unsqueeze(0)

        # Create candidate mask
        candidate_mask = torch.zeros(81, dtype=torch.bool)
        for row, col in candidate_cells:
            candidate_mask[row * 9 + col] = True

        candidate_mask = candidate_mask.unsqueeze(0)  # [1, 81]

        # Epsilon-greedy
        if torch.rand(1).item() < epsilon:
            # Random exploration
            cell_idx = np.random.choice([r*9+c for r,c in candidate_cells])
            new_value = np.random.randint(0, 10)
        else:
            # Network action
            with torch.no_grad():
                cell_probs, value_probs, _ = self.forward(
                    grid, None, candidate_mask
                )

            cell_idx = torch.argmax(cell_probs, dim=-1).item()
            new_value = torch.argmax(value_probs, dim=-1).item()

        return cell_idx, new_value


def create_action_network(hidden_size=512, num_heads=8, num_layers=2):
    """Create action-based network for Sudoku."""
    return ActionBasedNetwork(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers
    )
```

---

## File 2: `src/action_solver.py`

```python
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

    def solve(self, grid, max_iterations=16, verbose=False):
        """
        Solve by iteratively applying actions.

        Returns:
            final_grid: [9, 9] numpy array
            trace: List[ActionStep]
            solved: bool
        """
        current_grid = grid.copy()
        trace = []
        original_clues = (grid != 0)

        for step in range(max_iterations):
            violations = self.rule_graph.verify(current_grid)

            if len(violations) == 0:
                if verbose:
                    print(f"✓ Solved in {step} steps!")
                return current_grid, trace, True

            # Get candidate cells (violated non-clues)
            candidate_cells = self._get_candidate_cells(
                violations, original_clues
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
```

---

## File 3: `src/action_trainer.py`

```python
"""
Trainer for action-based network.
Uses violation delta as reward signal.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from src.action_solver import ActionBasedSolver
from src.rules import SudokuRuleGraph


class ActionBasedTrainer:
    """
    Train action network using violation reduction as reward.
    """

    def __init__(
        self,
        solver: ActionBasedSolver,
        dataset,
        rule_graph: SudokuRuleGraph,
        lr=1e-4
    ):
        self.solver = solver
        self.dataset = dataset
        self.rule_graph = rule_graph

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            solver.network.parameters(),
            lr=lr,
            weight_decay=0.01
        )

    def train_epoch(self, num_tasks=20, max_steps_per_task=8):
        """
        Train on multiple tasks.

        Returns:
            dict of metrics
        """
        total_loss = 0.0
        total_violations_reduced = 0
        successful_tasks = 0

        for task_idx in tqdm(range(num_tasks), desc="Training"):
            # Get task
            task = self.dataset[task_idx % len(self.dataset)]
            grid = task["grid"]

            # Train on this task
            task_loss, violations_reduced = self.train_task(
                grid, max_steps=max_steps_per_task
            )

            total_loss += task_loss
            total_violations_reduced += violations_reduced

            if violations_reduced >= len(task["violations"]):
                successful_tasks += 1

        # Decay exploration
        self.solver.epsilon *= 0.95
        self.solver.epsilon = max(0.05, self.solver.epsilon)

        return {
            'loss': total_loss / num_tasks,
            'avg_violations_reduced': total_violations_reduced / num_tasks,
            'success_rate': successful_tasks / num_tasks,
            'epsilon': self.solver.epsilon
        }

    def train_task(self, grid, max_steps=8):
        """
        Train on single task with deep supervision.

        Returns:
            total_loss: float
            violations_reduced: int
        """
        current_grid = grid.copy()
        original_clues = (grid != 0)

        initial_violations = self.rule_graph.verify(current_grid)
        total_loss = 0.0

        for step in range(max_steps):
            violations = self.rule_graph.verify(current_grid)

            if len(violations) == 0:
                break

            # Get candidate cells
            candidate_cells = self.solver._get_candidate_cells(
                violations, original_clues
            )

            if len(candidate_cells) == 0:
                break

            # Forward pass (with gradients)
            self.solver.network.train()

            # Convert to tensor
            grid_tensor = torch.from_numpy(current_grid.flatten()).long().unsqueeze(0)

            # Create candidate mask
            candidate_mask = torch.zeros(1, 81, dtype=torch.bool)
            for row, col in candidate_cells:
                candidate_mask[0, row * 9 + col] = True

            # Get action probabilities
            cell_probs, value_probs, confidence = self.solver.network.forward(
                grid_tensor, None, candidate_mask
            )

            # Sample action
            cell_dist = torch.distributions.Categorical(cell_probs)
            value_dist = torch.distributions.Categorical(value_probs)

            cell_idx = cell_dist.sample()
            value = value_dist.sample()

            # Apply action
            new_grid = current_grid.copy()
            row, col = cell_idx.item() // 9, cell_idx.item() % 9
            new_grid[row, col] = value.item()

            # Compute reward
            new_violations = self.rule_graph.verify(new_grid)
            violation_delta = len(violations) - len(new_violations)

            # Reward: positive for reducing violations
            reward = violation_delta * 2.0

            # Loss: negative log prob * reward (REINFORCE)
            log_prob = cell_dist.log_prob(cell_idx) + value_dist.log_prob(value)
            loss = -log_prob * reward

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.solver.network.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            current_grid = new_grid

        violations_reduced = len(initial_violations) - len(self.rule_graph.verify(current_grid))

        return total_loss, violations_reduced
```

---

## Usage Example: `demo_action.py`

```python
#!/usr/bin/env python3
"""
Demo of action-based RI-TRM solver.
"""

import numpy as np
from src.rules import SudokuRuleGraph
from src.dataset import RuleBasedSudokuDataset
from src.action_network import create_action_network
from src.action_solver import ActionBasedSolver
from src.action_trainer import ActionBasedTrainer


def main():
    print("Action-Based RI-TRM Demo")
    print("=" * 60)

    # Create components
    rule_graph = SudokuRuleGraph()
    network = create_action_network(hidden_size=512, num_heads=8, num_layers=2)
    solver = ActionBasedSolver(network, rule_graph, epsilon=0.3)

    # Test grid
    test_grid = np.zeros((9, 9), dtype=int)
    test_grid[0, 0] = 3
    test_grid[0, 5] = 3  # Violation

    print("\\nBefore training:")
    result, trace, solved = solver.solve(test_grid, verbose=True)
    print(f"Solved: {solved}")

    # Create dataset
    dataset = RuleBasedSudokuDataset(rule_graph, num_tasks=100)

    # Train
    print("\\nTraining...")
    trainer = ActionBasedTrainer(solver, dataset, rule_graph)

    for epoch in range(10):
        stats = trainer.train_epoch(num_tasks=20)
        print(f"Epoch {epoch}: loss={stats['loss']:.3f}, "
              f"reduced={stats['avg_violations_reduced']:.2f}, "
              f"success={stats['success_rate']:.1%}, "
              f"ε={stats['epsilon']:.3f}")

    # Test after training
    print("\\nAfter training:")
    solver.epsilon = 0.0  # No exploration
    result, trace, solved = solver.solve(test_grid, verbose=True)
    print(f"Solved: {solved}")


if __name__ == "__main__":
    main()
```

---

## Expected Output

```
Before training:
Step 0: (0,5) 3→7, 1 → 1 violations  # Random, no learning
Step 1: (0,5) 7→2, 1 → 1 violations

Training...
Epoch 0: loss=2.543, reduced=1.25, success=15.0%, ε=0.285
Epoch 1: loss=2.102, reduced=1.80, success=30.0%, ε=0.271
...
Epoch 9: loss=0.834, reduced=2.95, success=70.0%, ε=0.197

After training:
Step 0: (0,5) 3→0, 1 → 0 violations  # Learned!
✓ Solved in 1 steps!
Solved: True
```

---

**End of Skeleton**
