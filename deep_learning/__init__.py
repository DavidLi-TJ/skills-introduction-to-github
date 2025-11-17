"""
Deep Learning Fundamentals - Core Implementations

This package provides educational implementations of fundamental deep learning concepts.
"""

__version__ = "0.1.0"

from .activations import softmax, relu, sigmoid, tanh
from .layers import Linear, BatchNorm, Dropout
from .attention import scaled_dot_product_attention, MultiHeadAttention
from .transformer import Transformer, PositionalEncoding

__all__ = [
    'softmax', 'relu', 'sigmoid', 'tanh',
    'Linear', 'BatchNorm', 'Dropout',
    'scaled_dot_product_attention', 'MultiHeadAttention',
    'Transformer', 'PositionalEncoding'
]
