"""
Baseline TRM training experiment for Sudoku.

Train a standard Transformer model on a large dataset of (puzzle, solution) pairs.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.rules import SudokuRuleGraph
from src.dataset import SudokuSolver


class SudokuDataset(Dataset):
    """Sudoku dataset for the baseline model."""

    def __init__(self, num_samples=1000000, num_holes=40):
        self.num_samples = num_samples
        self.num_holes = num_holes
        self.solver = SudokuSolver(SudokuRuleGraph())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        grid = np.zeros((9, 9), dtype=int)
        solved_grid = self.solver.solve(grid)
        if solved_grid is None:
            # Should not happen with an empty grid
            return self.__getitem__(idx)

        puzzle = solved_grid.flatten().copy()
        holes = np.random.choice(81, self.num_holes, replace=False)
        puzzle[holes] = 0

        return torch.from_numpy(puzzle).long(), torch.from_numpy(solved_grid.flatten()).long()


class BaselineTRM(nn.Module):
    """Baseline Transformer model for Sudoku."""

    def __init__(self, vocab_size=10, hidden_size=512, num_layers=2, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size * 4,
        )
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        return self.output_head(output)


def main():
    """
    Baseline TRM training experiment.
    """
    print("=" * 60)
    print("BASELINE TRM - TRAINING EXPERIMENT")
    print("=" * 60)
    print()

    # Configuration
    NUM_SAMPLES = 1000000
    NUM_EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Configuration:")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Device: {DEVICE}")
    print()

    # Create dataset and dataloader
    dataset = SudokuDataset(num_samples=NUM_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create model
    model = BaselineTRM().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Network parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()

    # Training loop
    print("Starting training...")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for src, tgt in pbar:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)

            optimizer.zero_grad()

            # The target for the transformer decoder is the same as the source
            output = model(src, src)
            loss = criterion(output.view(-1, 10), tgt.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}")

    print()
    print("-" * 60)
    print("Training complete!")
    print()

    # Save model
    torch.save(model.state_dict(), 'experiments/baseline_trm_final.pt')
    print("Model saved to: experiments/baseline_trm_final.pt")


if __name__ == "__main__":
    main()
