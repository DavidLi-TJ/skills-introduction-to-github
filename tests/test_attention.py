"""Unit tests for attention mechanisms."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from deep_learning.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    create_causal_mask,
    create_padding_mask
)


class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention."""
    
    def test_attention_output_shape(self):
        """Test that attention produces correct output shape."""
        batch_size, seq_len_q, seq_len_k, d_k, d_v = 2, 5, 10, 64, 128
        
        Q = np.random.randn(batch_size, seq_len_q, d_k)
        K = np.random.randn(batch_size, seq_len_k, d_k)
        V = np.random.randn(batch_size, seq_len_k, d_v)
        
        output, weights = scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len_q, d_v)
        assert weights.shape == (batch_size, seq_len_q, seq_len_k)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        batch_size, seq_len, d_k = 2, 5, 64
        
        Q = np.random.randn(batch_size, seq_len, d_k)
        K = np.random.randn(batch_size, seq_len, d_k)
        V = np.random.randn(batch_size, seq_len, d_k)
        
        _, weights = scaled_dot_product_attention(Q, K, V)
        
        # Sum along the key dimension (last axis)
        sums = np.sum(weights, axis=-1)
        assert np.allclose(sums, 1.0)
    
    def test_attention_with_mask(self):
        """Test attention with masking."""
        batch_size, seq_len, d_k = 1, 3, 8
        
        Q = np.random.randn(batch_size, seq_len, d_k)
        K = np.random.randn(batch_size, seq_len, d_k)
        V = np.random.randn(batch_size, seq_len, d_k)
        
        # Mask out last position
        mask = np.array([[[0.0, 0.0, -np.inf]]])
        
        _, weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Last position should have zero attention
        assert np.allclose(weights[0, :, -1], 0.0)


class TestMultiHeadAttention:
    """Tests for multi-head attention."""
    
    def test_mha_initialization(self):
        """Test multi-head attention initialization."""
        d_model, num_heads = 512, 8
        mha = MultiHeadAttention(d_model, num_heads)
        
        assert mha.d_model == d_model
        assert mha.num_heads == num_heads
        assert mha.d_k == d_model // num_heads
    
    def test_mha_invalid_dimensions(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(AssertionError):
            # d_model not divisible by num_heads
            MultiHeadAttention(d_model=100, num_heads=7)
    
    def test_mha_forward_shape(self):
        """Test multi-head attention output shape."""
        d_model, num_heads = 512, 8
        batch_size, seq_len = 2, 10
        
        mha = MultiHeadAttention(d_model, num_heads)
        
        x = np.random.randn(batch_size, seq_len, d_model)
        output = mha.forward(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_mha_different_qkv_lengths(self):
        """Test multi-head attention with different Q, K, V lengths."""
        d_model, num_heads = 128, 4
        batch_size = 2
        seq_len_q, seq_len_k = 5, 10
        
        mha = MultiHeadAttention(d_model, num_heads)
        
        Q = np.random.randn(batch_size, seq_len_q, d_model)
        K = np.random.randn(batch_size, seq_len_k, d_model)
        V = np.random.randn(batch_size, seq_len_k, d_model)
        
        output = mha.forward(Q, K, V)
        
        # Output should have same sequence length as query
        assert output.shape == (batch_size, seq_len_q, d_model)


class TestMasks:
    """Tests for masking functions."""
    
    def test_causal_mask_shape(self):
        """Test causal mask has correct shape."""
        seq_len = 5
        mask = create_causal_mask(seq_len)
        assert mask.shape == (seq_len, seq_len)
    
    def test_causal_mask_structure(self):
        """Test causal mask has correct structure."""
        seq_len = 4
        mask = create_causal_mask(seq_len)
        
        # Upper triangle (excluding diagonal) should be -inf
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert mask[i, j] == -np.inf
                else:
                    assert mask[i, j] == 0.0
    
    def test_padding_mask_shape(self):
        """Test padding mask has correct shape."""
        batch_size, seq_len = 2, 10
        seq = np.random.randint(1, 100, (batch_size, seq_len))
        seq[:, -3:] = 0  # Add padding
        
        mask = create_padding_mask(seq, pad_token=0)
        assert mask.shape == (batch_size, 1, seq_len)
    
    def test_padding_mask_values(self):
        """Test padding mask marks padding correctly."""
        seq = np.array([[1, 2, 3, 0, 0],
                        [4, 5, 0, 0, 0]])
        
        mask = create_padding_mask(seq, pad_token=0)
        
        # First sequence: last 2 positions are padding
        assert mask[0, 0, 3] == -np.inf
        assert mask[0, 0, 4] == -np.inf
        assert mask[0, 0, 0] == 0.0
        
        # Second sequence: last 3 positions are padding
        assert mask[1, 0, 2] == -np.inf
        assert mask[1, 0, 1] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
