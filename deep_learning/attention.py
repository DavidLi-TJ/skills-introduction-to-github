"""
Attention Mechanisms for Neural Networks

This module implements attention mechanisms, which are the core building blocks
of modern architectures like Transformers.
"""

import numpy as np
from .activations import softmax


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention.
    
    This is the fundamental attention mechanism used in Transformers.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    The scaling by sqrt(d_k) prevents the dot products from growing too large,
    which would push the softmax into regions with extremely small gradients.
    
    Args:
        query (np.ndarray): Query matrix of shape (batch_size, seq_len_q, d_k)
        key (np.ndarray): Key matrix of shape (batch_size, seq_len_k, d_k)
        value (np.ndarray): Value matrix of shape (batch_size, seq_len_v, d_v)
        mask (np.ndarray, optional): Mask of shape (batch_size, seq_len_q, seq_len_k)
            Values should be 0 (keep) or -inf (mask out)
    
    Returns:
        tuple: (output, attention_weights)
            - output: Weighted sum of values (batch_size, seq_len_q, d_v)
            - attention_weights: Attention scores (batch_size, seq_len_q, seq_len_k)
    
    Examples:
        >>> # Simple example with single sequence
        >>> Q = np.random.randn(1, 5, 64)  # 5 queries, dimension 64
        >>> K = np.random.randn(1, 10, 64) # 10 keys
        >>> V = np.random.randn(1, 10, 128) # 10 values, dimension 128
        >>> output, weights = scaled_dot_product_attention(Q, K, V)
        >>> print(output.shape)
        (1, 5, 128)
    """
    # Get the dimension of keys (d_k)
    d_k = query.shape[-1]
    
    # Compute attention scores: QK^T / sqrt(d_k)
    # Shape: (batch_size, seq_len_q, seq_len_k)
    scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply mask if provided (e.g., for padding or causal masking)
    if mask is not None:
        scores = scores + mask
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)
    
    # Compute weighted sum of values
    # Shape: (batch_size, seq_len_q, d_v)
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.
    
    Instead of performing a single attention function, multi-head attention
    runs multiple attention operations in parallel, then concatenates the results.
    This allows the model to attend to information from different representation
    subspaces at different positions.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    """
    
    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.
        
        Args:
            d_model (int): Dimension of the model (must be divisible by num_heads)
            num_heads (int): Number of attention heads
        
        Raises:
            AssertionError: If d_model is not divisible by num_heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Initialize projection matrices
        # W^Q, W^K, W^V for all heads (combined)
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        
        # Output projection W^O
        self.W_o = np.random.randn(d_model, d_model) * 0.01
        
        # Cache for backpropagation
        self.cache = None
    
    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        
        Args:
            x (np.ndarray): Input of shape (batch_size, seq_len, d_model)
        
        Returns:
            np.ndarray: Reshaped to (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x):
        """
        Combine heads back into a single dimension.
        
        Args:
            x (np.ndarray): Input of shape (batch_size, num_heads, seq_len, d_k)
        
        Returns:
            np.ndarray: Reshaped to (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        # Transpose to (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(0, 2, 1, 3)
        # Reshape to (batch_size, seq_len, d_model)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass through multi-head attention.
        
        Args:
            query (np.ndarray): Query of shape (batch_size, seq_len_q, d_model)
            key (np.ndarray): Key of shape (batch_size, seq_len_k, d_model)
            value (np.ndarray): Value of shape (batch_size, seq_len_v, d_model)
            mask (np.ndarray, optional): Attention mask
        
        Returns:
            np.ndarray: Output of shape (batch_size, seq_len_q, d_model)
        """
        batch_size = query.shape[0]
        
        # Linear projections
        Q = np.matmul(query, self.W_q)  # (batch_size, seq_len_q, d_model)
        K = np.matmul(key, self.W_k)    # (batch_size, seq_len_k, d_model)
        V = np.matmul(value, self.W_v)  # (batch_size, seq_len_v, d_model)
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len_v, d_k)
        
        # Apply attention for each head
        # Reshape to apply attention to all heads at once
        Q_reshaped = Q.reshape(-1, Q.shape[2], Q.shape[3])
        K_reshaped = K.reshape(-1, K.shape[2], K.shape[3])
        V_reshaped = V.reshape(-1, V.shape[2], V.shape[3])
        
        if mask is not None:
            # Expand mask for all heads
            mask = np.expand_dims(mask, 1)  # (batch_size, 1, seq_len_q, seq_len_k)
            mask = np.repeat(mask, self.num_heads, axis=1)
            mask = mask.reshape(-1, mask.shape[2], mask.shape[3])
        
        # Compute attention
        attention_output, attention_weights = scaled_dot_product_attention(
            Q_reshaped, K_reshaped, V_reshaped, mask
        )
        
        # Reshape back
        attention_output = attention_output.reshape(
            batch_size, self.num_heads, -1, self.d_k
        )
        
        # Combine heads
        concat_attention = self.combine_heads(attention_output)
        
        # Final linear projection
        output = np.matmul(concat_attention, self.W_o)
        
        # Cache for backpropagation
        self.cache = (query, key, value, Q, K, V, attention_weights)
        
        return output
    
    def __repr__(self):
        return f"MultiHeadAttention(d_model={self.d_model}, num_heads={self.num_heads})"


def create_causal_mask(seq_len):
    """
    Create a causal (autoregressive) mask for self-attention.
    
    This mask prevents positions from attending to future positions,
    which is necessary for autoregressive models like GPT.
    
    Args:
        seq_len (int): Length of the sequence
    
    Returns:
        np.ndarray: Mask of shape (seq_len, seq_len) with -inf for masked positions
    
    Examples:
        >>> mask = create_causal_mask(4)
        >>> print(mask)
        [[  0. -inf -inf -inf]
         [  0.   0. -inf -inf]
         [  0.   0.   0. -inf]
         [  0.   0.   0.   0.]]
    """
    # Create lower triangular matrix
    mask = np.tril(np.ones((seq_len, seq_len)))
    # Replace 0s with -inf and 1s with 0
    mask = np.where(mask == 0, -np.inf, 0)
    return mask


def create_padding_mask(seq, pad_token=0):
    """
    Create a padding mask to prevent attention to padding tokens.
    
    Args:
        seq (np.ndarray): Sequence with padding (batch_size, seq_len)
        pad_token (int): Token ID used for padding (default: 0)
    
    Returns:
        np.ndarray: Mask of shape (batch_size, 1, seq_len) with -inf for padding
    """
    # Create mask: 1 for padding, 0 for real tokens
    mask = (seq == pad_token).astype(float)
    # Convert to -inf for padding, 0 for real tokens
    mask = np.where(mask == 1, -np.inf, 0)
    # Add dimension for broadcasting with attention scores
    return mask[:, np.newaxis, :]
