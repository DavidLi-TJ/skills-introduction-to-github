"""
Attention Mechanism Demo

This script demonstrates attention mechanisms and their behavior.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from deep_learning.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    create_causal_mask,
    create_padding_mask
)


def demo_self_attention_behavior():
    """Demonstrate how self-attention works."""
    print("=" * 60)
    print("Self-Attention Behavior")
    print("=" * 60)
    
    # Create a simple sequence with known patterns
    # Each token is represented by a 4-dimensional vector
    seq_len = 5
    d_model = 4
    
    # Create a simple sequence where tokens are similar to their neighbors
    sequence = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Token 0
        [0.9, 0.1, 0.0, 0.0],  # Token 1 (similar to 0)
        [0.0, 1.0, 0.0, 0.0],  # Token 2 (different)
        [0.0, 0.9, 0.1, 0.0],  # Token 3 (similar to 2)
        [0.0, 0.0, 1.0, 0.0],  # Token 4 (different)
    ]).reshape(1, seq_len, d_model)
    
    print(f"Input sequence shape: {sequence.shape}")
    print("Tokens with similar vectors should attend to each other.\n")
    
    # Self-attention: Q, K, V are all the same
    output, attention_weights = scaled_dot_product_attention(
        sequence, sequence, sequence
    )
    
    print("Attention weights (each row shows what a token attends to):")
    print(attention_weights[0])
    print("\nNotice:")
    print("- Token 0 attends mostly to token 0 and 1 (similar vectors)")
    print("- Token 2 attends mostly to token 2 and 3 (similar vectors)")
    print()


def demo_query_key_relationships():
    """Demonstrate the query-key-value mechanism."""
    print("=" * 60)
    print("Query-Key-Value Mechanism")
    print("=" * 60)
    
    print("""
    In attention, we have three key components:
    
    1. QUERY: "What am I looking for?"
       - Represents the current position's requirements
    
    2. KEY: "What do I offer?"
       - Represents what information each position contains
    
    3. VALUE: "Here's my information"
       - The actual information to be aggregated
    
    The attention mechanism:
    - Compares queries with keys (dot product)
    - Creates weights (softmax of similarities)
    - Uses weights to aggregate values
    """)
    
    # Simple example with interpretable values
    batch_size = 1
    seq_len = 3
    d_k = 2
    
    # Query: "I want information about dimension 0"
    Q = np.array([[[1.0, 0.0],   # Position 0 looks for dim 0
                   [0.0, 1.0],   # Position 1 looks for dim 1
                   [1.0, 1.0]]]) # Position 2 looks for both
    
    # Keys: "I have information about these dimensions"
    K = np.array([[[1.0, 0.0],   # Position 0 has dim 0
                   [0.0, 1.0],   # Position 1 has dim 1
                   [1.0, 0.0]]]) # Position 2 has dim 0
    
    # Values: "This is my actual information"
    V = np.array([[[10.0, 0.0],  # Position 0's info
                   [0.0, 20.0],  # Position 1's info
                   [30.0, 0.0]]])# Position 2's info
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("Attention weights:")
    print(weights[0])
    print("\nOutput (aggregated information):")
    print(output[0])
    print()


def demo_causal_masking_effect():
    """Demonstrate causal masking for autoregressive models."""
    print("=" * 60)
    print("Causal Masking (for GPT-style models)")
    print("=" * 60)
    
    seq_len = 4
    d_k = 8
    
    # Create sequence
    sequence = np.random.randn(1, seq_len, d_k)
    
    # Without mask
    _, attention_no_mask = scaled_dot_product_attention(
        sequence, sequence, sequence
    )
    
    # With causal mask
    mask = create_causal_mask(seq_len)
    mask = mask.reshape(1, seq_len, seq_len)  # Add batch dimension
    
    _, attention_with_mask = scaled_dot_product_attention(
        sequence, sequence, sequence, mask
    )
    
    print("Attention WITHOUT causal mask:")
    print("(each position can attend to all positions)")
    print(attention_no_mask[0])
    print()
    
    print("Attention WITH causal mask:")
    print("(each position can only attend to previous positions)")
    print(attention_with_mask[0])
    print("\nNotice how future positions (upper triangle) have zero attention!")
    print()


def demo_multi_head_vs_single_head():
    """Compare single-head and multi-head attention."""
    print("=" * 60)
    print("Multi-Head vs Single-Head Attention")
    print("=" * 60)
    
    print("""
    Why use multiple attention heads?
    
    Multi-head attention allows the model to:
    1. Attend to different aspects of the input simultaneously
    2. Capture different types of relationships
    3. Have multiple "representation subspaces"
    
    Think of it like having multiple experts looking at the same data,
    each focusing on different patterns.
    """)
    
    d_model = 128
    num_heads = 8
    batch_size = 2
    seq_len = 10
    
    # Create multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    print(f"Configuration:")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Number of heads: {num_heads}")
    print(f"  - Dimension per head: {d_model // num_heads}")
    print()
    
    # Create input
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Apply multi-head attention
    output = mha.forward(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nEach head processes a {d_model // num_heads}-dimensional")
    print(f"subspace independently, then results are concatenated.")
    print()


def visualize_attention_patterns():
    """Visualize attention weight patterns."""
    print("=" * 60)
    print("Visualizing Attention Patterns")
    print("=" * 60)
    
    seq_len = 10
    d_k = 16
    
    # Create a sequence with some structure
    sequence = np.random.randn(1, seq_len, d_k)
    
    # Compute self-attention
    _, attention_weights = scaled_dot_product_attention(
        sequence, sequence, sequence
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Attention weights heatmap
    im1 = axes[0].imshow(attention_weights[0], cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    axes[0].set_title('Self-Attention Weights')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Attention weights with causal mask
    mask = create_causal_mask(seq_len).reshape(1, seq_len, seq_len)
    _, masked_attention = scaled_dot_product_attention(
        sequence, sequence, sequence, mask
    )
    im2 = axes[1].imshow(masked_attention[0], cmap='viridis', aspect='auto')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    axes[1].set_title('Self-Attention with Causal Mask')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    output_path = 'attention_visualization.png'
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to: {output_path}")
    print()


def main():
    """Run all attention demonstrations."""
    print("\n")
    print("*" * 60)
    print("ATTENTION MECHANISM DEMONSTRATION")
    print("*" * 60)
    print("\n")
    
    demo_self_attention_behavior()
    demo_query_key_relationships()
    demo_causal_masking_effect()
    demo_multi_head_vs_single_head()
    
    try:
        visualize_attention_patterns()
    except Exception as e:
        print(f"Note: Visualization skipped ({e})")
    
    print("*" * 60)
    print("Demo completed!")
    print("*" * 60)


if __name__ == "__main__":
    main()
