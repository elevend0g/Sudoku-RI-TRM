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
        candidate_indices = []
        for row, col in candidate_cells:
            idx = row * 9 + col
            candidate_mask[idx] = True
            candidate_indices.append(idx)

        candidate_mask = candidate_mask.unsqueeze(0)  # [1, 81]

        # Epsilon-greedy
        if torch.rand(1).item() < epsilon:
            # Random exploration
            cell_idx = np.random.choice(candidate_indices)
            new_value = np.random.randint(1, 10) # Values 1-9
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
