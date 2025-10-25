"""
Tests for the 2-layer recursive network.

Validates:
1. Parameter count (~7M)
2. Forward pass with correct shapes
3. Gradient flow
4. Component functionality (RMSNorm, RoPE, SwiGLU, Attention)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.network import (
    TinyRecursiveNetwork,
    create_sudoku_network,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    MultiHeadAttention,
    TransformerBlock,
    apply_rotary_pos_emb,
    rotate_half,
)


class TestParameterCount:
    """Test that network has ~7M parameters."""

    def test_default_network_size(self):
        """Default configuration should have ~7M parameters."""
        network = create_sudoku_network()
        params = network.count_parameters()

        # Should be close to 7M (within 10%)
        assert 6_000_000 < params < 8_000_000, f"Expected ~7M params, got {params:,}"

    def test_parameter_breakdown(self):
        """Test parameter breakdown is reasonable."""
        network = create_sudoku_network()
        breakdown = network.get_parameter_breakdown()

        # Check all components present
        assert "cell_embedding" in breakdown
        assert "violation_embedding" in breakdown
        assert "position_embedding" in breakdown
        assert "transformer_layers" in breakdown
        assert "norm" in breakdown
        assert "output_head" in breakdown
        assert "confidence_head" in breakdown

        # Total should match
        assert breakdown["total"] == network.count_parameters()

        # Transformer layers should dominate
        assert breakdown["transformer_layers"] > breakdown["total"] * 0.5

    def test_2_layers_vs_4_layers(self):
        """2-layer network should have fewer parameters than 4-layer."""
        network_2 = TinyRecursiveNetwork(num_layers=2)
        network_4 = TinyRecursiveNetwork(num_layers=4)

        params_2 = network_2.count_parameters()
        params_4 = network_4.count_parameters()

        assert params_4 > params_2
        # 4-layer should be roughly 2x the transformer params
        assert params_4 > params_2 * 1.5


class TestForwardPass:
    """Test forward pass with correct shapes."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        network = create_sudoku_network()
        batch_size = 4

        # Create input grid (batch, 81)
        grid = torch.randint(0, 10, (batch_size, 81))

        # Forward pass
        logits, confidence, reasoning_state = network(grid)

        # Check shapes
        assert logits.shape == (batch_size, 81, 10), "Logits shape incorrect"
        assert confidence.shape == (batch_size,), "Confidence shape incorrect"
        assert reasoning_state.shape == (batch_size, 81, 512), "Reasoning state shape incorrect"

    def test_with_violations(self):
        """Test forward pass with violation information."""
        network = create_sudoku_network()
        batch_size = 4

        grid = torch.randint(0, 10, (batch_size, 81))

        # Create violation tensor (batch, num_violations, 2)
        # Each violation: [type (0-2), location (0-80)]
        violations = torch.tensor([
            [[0, 5], [1, 10]],  # Batch 0: 2 violations
            [[2, 20], [0, 30]],  # Batch 1: 2 violations
            [[1, 15], [2, 25]],  # Batch 2: 2 violations
            [[0, 40], [1, 50]],  # Batch 3: 2 violations
        ])

        logits, confidence, reasoning_state = network(grid, violations)

        assert logits.shape == (batch_size, 81, 10)
        assert confidence.shape == (batch_size,)

    def test_with_reasoning_state(self):
        """Test forward pass with previous reasoning state."""
        network = create_sudoku_network()
        batch_size = 4

        grid = torch.randint(0, 10, (batch_size, 81))

        # Create previous reasoning state
        prev_reasoning = torch.randn(batch_size, 81, 512)

        logits, confidence, reasoning_state = network(
            grid,
            reasoning_state=prev_reasoning
        )

        assert logits.shape == (batch_size, 81, 10)
        assert confidence.shape == (batch_size,)

        # New reasoning state should be different from input
        assert not torch.allclose(reasoning_state, prev_reasoning)

    def test_confidence_in_range(self):
        """Confidence should be in [0, 1]."""
        network = create_sudoku_network()
        grid = torch.randint(0, 10, (4, 81))

        _, confidence, _ = network(grid)

        assert torch.all(confidence >= 0.0)
        assert torch.all(confidence <= 1.0)

    def test_batch_size_1(self):
        """Test with single example."""
        network = create_sudoku_network()
        grid = torch.randint(0, 10, (1, 81))

        logits, confidence, reasoning_state = network(grid)

        assert logits.shape == (1, 81, 10)
        assert confidence.shape == (1,)

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        network = create_sudoku_network()

        for batch_size in [1, 2, 8, 16]:
            grid = torch.randint(0, 10, (batch_size, 81))
            logits, confidence, reasoning_state = network(grid)

            assert logits.shape == (batch_size, 81, 10)
            assert confidence.shape == (batch_size,)


class TestGradientFlow:
    """Test that gradients flow correctly."""

    def test_gradients_flow(self):
        """Test that gradients flow through network."""
        network = create_sudoku_network()
        grid = torch.randint(0, 10, (4, 81))

        # Add violations to ensure violation_embedding gets gradients
        violations = torch.tensor([
            [[0, 5], [1, 10]],
            [[2, 20], [0, 30]],
            [[1, 15], [2, 25]],
            [[0, 40], [1, 50]],
        ])

        # Forward pass
        logits, confidence, _ = network(grid, violations)

        # Create dummy loss (mean to avoid large values)
        loss = logits.mean() + confidence.mean()

        # Backward pass
        loss.backward()

        # Check gradients exist
        for name, param in network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_gradient_magnitude_reasonable(self):
        """Test that gradients are not exploding or vanishing."""
        network = create_sudoku_network()
        grid = torch.randint(0, 10, (4, 81))

        # Add violations
        violations = torch.tensor([
            [[0, 5], [1, 10]],
            [[2, 20], [0, 30]],
            [[1, 15], [2, 25]],
            [[0, 40], [1, 50]],
        ])

        logits, confidence, _ = network(grid, violations)
        # Use mean loss to keep gradients reasonable
        loss = logits.mean() + confidence.mean()
        loss.backward()

        # Check gradient magnitudes
        grad_norms = []
        for param in network.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        avg_grad_norm = np.mean(grad_norms)

        # Should not be too large or too small
        assert 0.00001 < avg_grad_norm < 100.0, f"Unusual gradient norm: {avg_grad_norm}"


class TestRMSNorm:
    """Test RMSNorm implementation."""

    def test_rmsnorm_shape(self):
        """RMSNorm should preserve shape."""
        norm = RMSNorm(512)
        x = torch.randn(4, 10, 512)

        out = norm(x)

        assert out.shape == x.shape

    def test_rmsnorm_normalizes(self):
        """RMSNorm should normalize."""
        norm = RMSNorm(512)
        x = torch.randn(4, 10, 512) * 10  # Large values

        out = norm(x)

        # Output should have smaller magnitude
        assert out.abs().mean() < x.abs().mean()

    def test_rmsnorm_gradient(self):
        """RMSNorm should have gradients."""
        norm = RMSNorm(512)
        x = torch.randn(4, 10, 512, requires_grad=True)

        out = norm(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None


class TestRotaryPositionalEmbedding:
    """Test RoPE implementation."""

    def test_rope_shape(self):
        """RoPE should return correct shapes."""
        rope = RotaryPositionalEmbedding(dim=64)

        cos, sin = rope(seq_len=20)

        assert cos.shape == (20, 64)
        assert sin.shape == (20, 64)

    def test_rope_different_seq_lens(self):
        """RoPE should work for different sequence lengths."""
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=256)

        for seq_len in [10, 50, 100, 200]:
            cos, sin = rope(seq_len)
            assert cos.shape == (seq_len, 64)
            assert sin.shape == (seq_len, 64)

    def test_apply_rotary_emb(self):
        """Test applying rotary embeddings."""
        batch, num_heads, seq_len, head_dim = 2, 8, 20, 64

        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads, seq_len, head_dim)

        rope = RotaryPositionalEmbedding(dim=head_dim)
        cos, sin = rope(seq_len)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

        # Should be different from original
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)

    def test_rotate_half(self):
        """Test rotate_half helper."""
        x = torch.randn(2, 8, 20, 64)
        x_rot = rotate_half(x)

        assert x_rot.shape == x.shape
        assert not torch.allclose(x_rot, x)


class TestSwiGLU:
    """Test SwiGLU activation."""

    def test_swiglu_shape(self):
        """SwiGLU should preserve input shape."""
        swiglu = SwiGLU(dim=512, hidden_dim=2048)
        x = torch.randn(4, 10, 512)

        out = swiglu(x)

        assert out.shape == x.shape

    def test_swiglu_nonlinear(self):
        """SwiGLU should be nonlinear."""
        swiglu = SwiGLU(dim=512, hidden_dim=2048)

        x1 = torch.randn(4, 10, 512)
        x2 = torch.randn(4, 10, 512)

        out1 = swiglu(x1)
        out2 = swiglu(x2)
        out_combined = swiglu(x1 + x2)

        # Should not be linear
        assert not torch.allclose(out1 + out2, out_combined, atol=1e-5)

    def test_swiglu_gradient(self):
        """SwiGLU should have gradients."""
        swiglu = SwiGLU(dim=512, hidden_dim=2048)
        x = torch.randn(4, 10, 512, requires_grad=True)

        out = swiglu(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


class TestMultiHeadAttention:
    """Test multi-head attention."""

    def test_attention_shape(self):
        """Attention should preserve shape."""
        attn = MultiHeadAttention(dim=512, num_heads=8)
        x = torch.randn(4, 20, 512)

        out = attn(x)

        assert out.shape == x.shape

    def test_attention_with_mask(self):
        """Attention should work with mask."""
        attn = MultiHeadAttention(dim=512, num_heads=8)
        x = torch.randn(4, 20, 512)

        # Create causal mask
        mask = torch.tril(torch.ones(20, 20)).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(4, -1, -1, -1)

        out = attn(x, mask)

        assert out.shape == x.shape

    def test_attention_self_attention(self):
        """Test that it's doing self-attention."""
        attn = MultiHeadAttention(dim=512, num_heads=8)

        # Create input where one position is very different
        x = torch.zeros(1, 10, 512)
        x[:, 5, :] = 10.0  # Position 5 is special

        out = attn(x)

        # Output at position 5 should be influenced by other positions
        assert not torch.allclose(out[:, 5, :], x[:, 5, :])

    def test_attention_gradient(self):
        """Attention should have gradients."""
        attn = MultiHeadAttention(dim=512, num_heads=8)
        x = torch.randn(4, 20, 512, requires_grad=True)

        out = attn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


class TestTransformerBlock:
    """Test transformer block."""

    def test_block_shape(self):
        """Transformer block should preserve shape."""
        block = TransformerBlock(dim=512, num_heads=8, ffn_dim=2048)
        x = torch.randn(4, 20, 512)

        out = block(x)

        assert out.shape == x.shape

    def test_block_residual(self):
        """Block should use residual connections."""
        block = TransformerBlock(dim=512, num_heads=8, ffn_dim=2048)
        x = torch.randn(4, 20, 512)

        out = block(x)

        # Output should be similar to input (due to residuals)
        # but not identical
        similarity = F.cosine_similarity(
            out.flatten(1), x.flatten(1), dim=1
        ).mean()

        assert similarity > 0.5, "Block should preserve information via residuals"

    def test_block_gradient(self):
        """Block should have gradients."""
        block = TransformerBlock(dim=512, num_heads=8, ffn_dim=2048)
        x = torch.randn(4, 20, 512, requires_grad=True)

        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


class TestNetworkArchitecture:
    """Test overall network architecture."""

    def test_network_has_2_layers(self):
        """Network should have exactly 2 transformer layers."""
        network = create_sudoku_network()

        assert len(network.layers) == 2

    def test_network_uses_rmsnorm(self):
        """Network should use RMSNorm, not LayerNorm."""
        network = create_sudoku_network()

        # Check that norm is RMSNorm
        assert isinstance(network.norm, RMSNorm)

        # Check transformer blocks use RMSNorm
        for layer in network.layers:
            assert isinstance(layer.norm1, RMSNorm)
            assert isinstance(layer.norm2, RMSNorm)

    def test_network_uses_swiglu(self):
        """Network should use SwiGLU activation."""
        network = create_sudoku_network()

        for layer in network.layers:
            assert isinstance(layer.ffn, SwiGLU)

    def test_no_bias_in_linear_layers(self):
        """Linear layers should not have bias (as per spec)."""
        network = create_sudoku_network()

        # Check key linear layers
        for layer in network.layers:
            assert layer.attn.q_proj.bias is None
            assert layer.attn.k_proj.bias is None
            assert layer.attn.v_proj.bias is None
            assert layer.attn.out_proj.bias is None

        assert network.output_head.bias is None

    def test_8_attention_heads(self):
        """Network should have 8 attention heads."""
        network = create_sudoku_network()

        for layer in network.layers:
            assert layer.attn.num_heads == 8

    def test_512_hidden_size(self):
        """Network should have 512 hidden size."""
        network = create_sudoku_network()

        assert network.hidden_size == 512


class TestDeterminism:
    """Test network determinism."""

    def test_same_input_same_output(self):
        """Same input should give same output."""
        torch.manual_seed(42)
        network1 = create_sudoku_network()

        torch.manual_seed(42)
        network2 = create_sudoku_network()

        grid = torch.randint(0, 10, (4, 81))

        torch.manual_seed(123)
        out1, conf1, state1 = network1(grid)

        torch.manual_seed(123)
        out2, conf2, state2 = network2(grid)

        assert torch.allclose(out1, out2)
        assert torch.allclose(conf1, conf2)
        assert torch.allclose(state1, state2)


class TestNumericalStability:
    """Test numerical stability."""

    def test_no_nans(self):
        """Network should not produce NaNs."""
        network = create_sudoku_network()
        grid = torch.randint(0, 10, (4, 81))

        logits, confidence, reasoning_state = network(grid)

        assert not torch.isnan(logits).any()
        assert not torch.isnan(confidence).any()
        assert not torch.isnan(reasoning_state).any()

    def test_no_infs(self):
        """Network should not produce Infs."""
        network = create_sudoku_network()
        grid = torch.randint(0, 10, (4, 81))

        logits, confidence, reasoning_state = network(grid)

        assert not torch.isinf(logits).any()
        assert not torch.isinf(confidence).any()
        assert not torch.isinf(reasoning_state).any()

    def test_reasonable_output_range(self):
        """Outputs should be in reasonable range."""
        network = create_sudoku_network()
        grid = torch.randint(0, 10, (4, 81))

        logits, confidence, reasoning_state = network(grid)

        # Logits should be reasonable (not too extreme)
        assert logits.abs().max() < 100

        # Confidence in [0, 1]
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0

        # Reasoning state should be reasonable
        assert reasoning_state.abs().max() < 100


import torch.nn.functional as F
