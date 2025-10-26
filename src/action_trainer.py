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

        batch = self.dataset.get_batch(num_tasks)
        for task in tqdm(batch, desc="Training"):
            grid = task["grid"]
            original_clues = task["original_clues"]

            # Train on this task
            task_loss, violations_reduced = self.train_task(
                grid, original_clues, max_steps=max_steps_per_task
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

    def train_task(self, grid, original_clues, max_steps=8):
        """
        Train on single task with deep supervision.

        Returns:
            total_loss: float
            violations_reduced: int
        """
        current_grid = grid.copy()

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
            new_grid[row, col] = value.item() + 1

            # Compute reward
            new_violations = self.rule_graph.verify(new_grid)
            violation_delta = len(violations) - len(new_violations)

            # Reward: positive for reducing violations, negative for increasing
            if violation_delta > 0:
                reward = violation_delta * 2.0
            elif violation_delta < 0:
                reward = violation_delta * 5.0 # Larger penalty for increasing violations
            else:
                reward = -0.1 # Small penalty for no change

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
