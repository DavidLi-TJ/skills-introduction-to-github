"""
Softmax Activation Function Demo

This script demonstrates the softmax function and its properties.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from deep_learning.activations import softmax


def demo_basic_softmax():
    """Demonstrate basic softmax computation."""
    print("=" * 60)
    print("Basic Softmax Example")
    print("=" * 60)
    
    # Simple logits
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    
    print(f"Input logits: {logits}")
    print(f"Softmax probabilities: {probs}")
    print(f"Sum of probabilities: {np.sum(probs):.6f} (should be 1.0)")
    print()


def demo_temperature_scaling():
    """Demonstrate the effect of temperature on softmax."""
    print("=" * 60)
    print("Temperature Scaling")
    print("=" * 60)
    
    logits = np.array([2.0, 1.0, 0.1])
    temperatures = [0.5, 1.0, 2.0, 5.0]
    
    print(f"Original logits: {logits}\n")
    
    for temp in temperatures:
        scaled_logits = logits / temp
        probs = softmax(scaled_logits)
        print(f"Temperature {temp}:")
        print(f"  Probabilities: {probs}")
        print(f"  Max probability: {np.max(probs):.4f}")
        print()


def demo_numerical_stability():
    """Demonstrate numerical stability of softmax implementation."""
    print("=" * 60)
    print("Numerical Stability")
    print("=" * 60)
    
    # Large values that could cause overflow
    large_logits = np.array([1000.0, 1001.0, 1002.0])
    
    print(f"Large logits: {large_logits}")
    
    # Our stable implementation
    stable_probs = softmax(large_logits)
    print(f"Stable softmax: {stable_probs}")
    print(f"Sum: {np.sum(stable_probs):.6f}")
    
    # Naive implementation (for comparison)
    try:
        naive_exp = np.exp(large_logits)
        naive_probs = naive_exp / np.sum(naive_exp)
        print(f"Naive softmax: {naive_probs}")
    except:
        print("Naive implementation would overflow!")
    
    print()


def demo_batch_softmax():
    """Demonstrate softmax on batches of data."""
    print("=" * 60)
    print("Batch Softmax")
    print("=" * 60)
    
    # Batch of logits (e.g., from a neural network)
    batch_logits = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 0.5, 0.5],
        [1.0, 2.0, 3.0]
    ])
    
    print("Batch of logits:")
    print(batch_logits)
    print()
    
    # Apply softmax along last axis
    batch_probs = softmax(batch_logits, axis=-1)
    
    print("Batch of probabilities:")
    print(batch_probs)
    print()
    
    # Verify each row sums to 1
    print("Row sums (should all be 1.0):")
    print(np.sum(batch_probs, axis=-1))
    print()


def visualize_softmax():
    """Create visualization of softmax behavior."""
    print("=" * 60)
    print("Visualizing Softmax")
    print("=" * 60)
    
    # Create a range of values
    x = np.linspace(-5, 5, 100)
    
    # Create 3-class logits where first class varies, others are fixed
    logits = np.zeros((100, 3))
    logits[:, 0] = x
    logits[:, 1] = 0.0
    logits[:, 2] = 0.0
    
    # Compute softmax
    probs = softmax(logits, axis=-1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, probs[:, 0], label='Class 0 (varying)', linewidth=2)
    plt.plot(x, probs[:, 1], label='Class 1 (fixed at 0)', linewidth=2)
    plt.plot(x, probs[:, 2], label='Class 2 (fixed at 0)', linewidth=2)
    plt.xlabel('Logit value for Class 0', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Softmax Probabilities as One Logit Varies', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    output_path = 'softmax_visualization.png'
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to: {output_path}")
    print()


def main():
    """Run all softmax demonstrations."""
    print("\n")
    print("*" * 60)
    print("SOFTMAX ACTIVATION FUNCTION DEMONSTRATION")
    print("*" * 60)
    print("\n")
    
    demo_basic_softmax()
    demo_temperature_scaling()
    demo_numerical_stability()
    demo_batch_softmax()
    
    try:
        visualize_softmax()
    except Exception as e:
        print(f"Note: Visualization skipped ({e})")
    
    print("*" * 60)
    print("Demo completed!")
    print("*" * 60)


if __name__ == "__main__":
    main()
