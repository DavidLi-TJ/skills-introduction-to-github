"""
Transformer Architecture Demo

This script demonstrates the Transformer model and its components.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from deep_learning.transformer import Transformer, PositionalEncoding
from deep_learning.attention import (
    scaled_dot_product_attention, 
    MultiHeadAttention,
    create_causal_mask
)


def demo_positional_encoding():
    """Demonstrate positional encoding."""
    print("=" * 60)
    print("Positional Encoding")
    print("=" * 60)
    
    d_model = 64
    max_len = 100
    
    pos_enc = PositionalEncoding(d_model, max_len)
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Add positional encoding
    x_with_pos = pos_enc.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_with_pos.shape}")
    print(f"Positional encoding for first position (first 10 dims):")
    print(pos_enc.pe[0, :10])
    print(f"Positional encoding for second position (first 10 dims):")
    print(pos_enc.pe[1, :10])
    print()


def demo_scaled_dot_product_attention():
    """Demonstrate scaled dot-product attention."""
    print("=" * 60)
    print("Scaled Dot-Product Attention")
    print("=" * 60)
    
    batch_size = 2
    seq_len_q = 5
    seq_len_k = 10
    d_k = 64
    d_v = 64
    
    # Create random queries, keys, and values
    Q = np.random.randn(batch_size, seq_len_q, d_k)
    K = np.random.randn(batch_size, seq_len_k, d_k)
    V = np.random.randn(batch_size, seq_len_k, d_v)
    
    print(f"Query shape: {Q.shape}")
    print(f"Key shape: {K.shape}")
    print(f"Value shape: {V.shape}")
    
    # Compute attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights sum (should be 1.0): {attention_weights[0, 0, :].sum():.6f}")
    print()


def demo_causal_mask():
    """Demonstrate causal (autoregressive) masking."""
    print("=" * 60)
    print("Causal Masking for Autoregressive Models")
    print("=" * 60)
    
    seq_len = 5
    mask = create_causal_mask(seq_len)
    
    print(f"Causal mask for sequence length {seq_len}:")
    print(mask)
    print("\nThis mask ensures each position can only attend to")
    print("previous positions (including itself), preventing")
    print("information leakage from future tokens.")
    print()


def demo_multi_head_attention():
    """Demonstrate multi-head attention."""
    print("=" * 60)
    print("Multi-Head Attention")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # Create multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    print(f"Multi-head attention: {mha}")
    print(f"Model dimension: {d_model}")
    print(f"Number of heads: {num_heads}")
    print(f"Dimension per head: {mha.d_k}")
    
    # Create random input
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Self-attention (Q, K, V all come from the same input)
    output = mha.forward(x, x, x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()


def demo_transformer_simple():
    """Demonstrate a simple Transformer forward pass."""
    print("=" * 60)
    print("Transformer Forward Pass")
    print("=" * 60)
    
    # Small vocabulary for demonstration
    src_vocab_size = 100
    tgt_vocab_size = 100
    
    # Model parameters
    d_model = 128
    num_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff = 512
    dropout = 0.1
    
    # Create transformer
    transformer = Transformer(
        src_vocab_size, 
        tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    
    print(f"Transformer: {transformer}")
    print()
    
    # Create sample input (token indices)
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src = np.random.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = np.random.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    
    # Forward pass
    output = transformer.forward(src, tgt)
    
    print(f"Output shape: {output.shape}")
    print(f"Output represents logits for {tgt_vocab_size} vocabulary items")
    print()


def demo_transformer_components():
    """Show the components of a Transformer."""
    print("=" * 60)
    print("Transformer Architecture Components")
    print("=" * 60)
    
    print("""
    A Transformer consists of:
    
    1. INPUT EMBEDDINGS + POSITIONAL ENCODING
       - Convert token IDs to dense vectors
       - Add positional information
    
    2. ENCODER STACK (N layers)
       Each encoder layer has:
       - Multi-head self-attention
       - Layer normalization
       - Position-wise feed-forward network
       - Residual connections
    
    3. DECODER STACK (N layers)
       Each decoder layer has:
       - Masked multi-head self-attention
       - Multi-head cross-attention (to encoder)
       - Layer normalization
       - Position-wise feed-forward network
       - Residual connections
    
    4. OUTPUT PROJECTION
       - Linear layer to vocabulary size
       - Softmax to get probabilities
    
    Key innovations:
    - Self-attention allows parallel processing
    - Multi-head attention captures different relationships
    - Positional encoding injects sequence information
    - Layer normalization stabilizes training
    """)


def main():
    """Run all transformer demonstrations."""
    print("\n")
    print("*" * 60)
    print("TRANSFORMER ARCHITECTURE DEMONSTRATION")
    print("*" * 60)
    print("\n")
    
    demo_positional_encoding()
    demo_scaled_dot_product_attention()
    demo_causal_mask()
    demo_multi_head_attention()
    demo_transformer_simple()
    demo_transformer_components()
    
    print("*" * 60)
    print("Demo completed!")
    print("*" * 60)


if __name__ == "__main__":
    main()
