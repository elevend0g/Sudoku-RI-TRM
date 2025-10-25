"""
Tiny 2-layer recursive network for RI-TRM.

Architecture from TRM paper:
- 2 transformer layers (not 4!)
- Hidden dim: 512
- 8 attention heads
- RMSNorm, no bias, RoPE, SwiGLU
- ~7M parameters total

Input: (x, y, z, violations)
- x: embedded task specification (initial grid)
- y: current solution (current grid state)
- z: reasoning latent state
- violations: embedded violation information

Output: updated z (reasoning latent)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    More efficient than LayerNorm and used in modern transformers.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            Normalized x: (batch, seq_len, dim)
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_norm = x / rms
        return self.weight * x_norm


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    More effective than absolute positional embeddings for transformers.
    """

    def __init__(self, dim: int, max_seq_len: int = 256):
        super().__init__()
        self.dim = dim

        # Precompute frequency matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute position indices
        position = torch.arange(max_seq_len).float()
        self.register_buffer("position", position)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cos, sin: (seq_len, dim) each
        """
        # Get positions for this sequence length
        position = self.position[:seq_len]

        # Compute angles
        freqs = torch.outer(position, self.inv_freq)

        # Duplicate for real and imaginary parts
        emb = torch.cat([freqs, freqs], dim=-1)

        return torch.cos(emb), torch.sin(emb)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key.

    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)

    Returns:
        q_rotated, k_rotated
    """
    # Reshape cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Split into pairs for rotation
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper for RoPE: rotate second half by 90 degrees."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.

    More effective than standard FFN for transformers.
    SwiGLU(x) = Swish(xW) ⊗ (xV)
    where Swish(x) = x * sigmoid(x)
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w = nn.Linear(dim, hidden_dim, bias=bias)
        self.v = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
        """
        # Swish activation: x * sigmoid(x)
        swish = self.w(x)
        swish = swish * torch.sigmoid(swish)

        # Gate with v projection
        v = self.v(x)
        hidden = swish * v

        # Project back to dim
        return self.w2(hidden)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with RoPE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len) or (batch, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, head_dim)

        # Apply RoPE
        cos, sin = self.rope(seq_len)
        cos, sin = cos.to(q.device), sin.to(q.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # (batch, num_heads, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # (batch, seq_len) -> (batch, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        # (batch, num_heads, seq_len, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.dim)

        # Output projection
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """
    Single transformer block with RMSNorm, attention, and SwiGLU FFN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()

        # Pre-norm architecture
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, bias=False, dropout=dropout)

        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, dim)
        """
        # Pre-norm + attention + residual
        x = x + self.dropout(self.attn(self.norm1(x), mask))

        # Pre-norm + FFN + residual
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


class TinyRecursiveNetwork(nn.Module):
    """
    Tiny 2-layer recursive transformer network for RI-TRM.

    Architecture:
    - 2 transformer layers
    - 512 hidden dim
    - 8 attention heads
    - RMSNorm, no bias, RoPE, SwiGLU
    - ~7M parameters

    For Sudoku:
    - Input grid: 9x9 = 81 cells, each with value 0-9
    - Violations: embedded violation information
    - Reasoning state: z
    """

    def __init__(
        self,
        vocab_size: int = 10,  # 0-9 for Sudoku
        grid_size: int = 81,  # 9x9 Sudoku
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 2048,  # 4 * hidden_size
        max_seq_len: int = 256,
        dropout: float = 0.0,
        num_violation_types: int = 3,  # row, col, box
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embeddings
        self.cell_embedding = nn.Embedding(vocab_size, hidden_size)

        # Violation embedding (for each violation type)
        self.violation_embedding = nn.Embedding(num_violation_types, hidden_size)

        # Position embedding for grid cells
        self.position_embedding = nn.Embedding(grid_size, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(hidden_size)

        # Output head for grid predictions
        self.output_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1, bias=False),
            nn.Sigmoid()
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        grid: torch.Tensor,
        violations: Optional[torch.Tensor] = None,
        reasoning_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            grid: (batch, 81) - flattened Sudoku grid with values 0-9
            violations: (batch, num_violations, 2) - violation (type, location)
            reasoning_state: (batch, 81, hidden_size) - previous reasoning state

        Returns:
            output: (batch, 81, vocab_size) - logits for each cell
            confidence: (batch,) - confidence score
            reasoning_state: (batch, 81, hidden_size) - updated reasoning state
        """
        batch_size = grid.shape[0]
        device = grid.device

        # Embed grid cells
        x = self.cell_embedding(grid)  # (batch, 81, hidden)

        # Add positional embeddings
        positions = torch.arange(self.grid_size, device=device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        x = x + self.position_embedding(positions)

        # Add violation information if provided
        if violations is not None and violations.shape[1] > 0:
            # violations: (batch, num_violations, 2) where each is [type, location]
            violation_types = violations[:, :, 0]  # (batch, num_violations)
            violation_locs = violations[:, :, 1]  # (batch, num_violations)

            # Embed violation types
            v_embed = self.violation_embedding(violation_types)  # (batch, num_violations, hidden)

            # Add to corresponding grid positions
            # For simplicity, add to first position (could be more sophisticated)
            if v_embed.shape[1] > 0:
                v_mean = v_embed.mean(dim=1, keepdim=True)  # (batch, 1, hidden)
                x = x + v_mean.expand(-1, self.grid_size, -1) * 0.1

        # Combine with previous reasoning state if provided
        if reasoning_state is not None:
            x = x + reasoning_state * 0.5

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final norm
        x = self.norm(x)

        # Output logits for each cell
        logits = self.output_head(x)  # (batch, 81, vocab_size)

        # Confidence score (pooled over sequence)
        pooled = x.mean(dim=1)  # (batch, hidden)
        confidence = self.confidence_head(pooled).squeeze(-1)  # (batch,)

        # Return logits, confidence, and updated reasoning state
        return logits, confidence, x

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> dict:
        """Get detailed parameter breakdown."""
        breakdown = {}

        breakdown["cell_embedding"] = self.cell_embedding.weight.numel()
        breakdown["violation_embedding"] = self.violation_embedding.weight.numel()
        breakdown["position_embedding"] = self.position_embedding.weight.numel()

        # Transformer layers
        layer_params = 0
        for layer in self.layers:
            for name, param in layer.named_parameters():
                layer_params += param.numel()
        breakdown["transformer_layers"] = layer_params

        breakdown["norm"] = self.norm.weight.numel()
        breakdown["output_head"] = self.output_head.weight.numel()

        # Confidence head
        conf_params = sum(p.numel() for p in self.confidence_head.parameters())
        breakdown["confidence_head"] = conf_params

        breakdown["total"] = sum(breakdown.values())

        return breakdown


def create_sudoku_network(
    hidden_size: int = 512,
    num_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.0
) -> TinyRecursiveNetwork:
    """
    Create a network for Sudoku with ~7M parameters.

    Args:
        hidden_size: Hidden dimension (default 512)
        num_layers: Number of layers (default 2)
        num_heads: Number of attention heads (default 8)
        dropout: Dropout rate (default 0.0)

    Returns:
        TinyRecursiveNetwork
    """
    network = TinyRecursiveNetwork(
        vocab_size=10,  # 0-9
        grid_size=81,  # 9x9
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=int(hidden_size * 3),  # 3x instead of 4x to get ~7M params
        dropout=dropout,
        num_violation_types=3,  # row, col, box
    )

    return network
