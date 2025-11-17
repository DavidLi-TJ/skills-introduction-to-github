"""Unit tests for activation functions."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from deep_learning.activations import (
    softmax, relu, sigmoid, tanh, 
    leaky_relu, relu_derivative, sigmoid_derivative
)


class TestSoftmax:
    """Tests for softmax activation function."""
    
    def test_softmax_sums_to_one(self):
        """Test that softmax outputs sum to 1."""
        x = np.array([2.0, 1.0, 0.1])
        result = softmax(x)
        assert np.allclose(np.sum(result), 1.0)
    
    def test_softmax_all_positive(self):
        """Test that all softmax outputs are positive."""
        x = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        result = softmax(x)
        assert np.all(result > 0)
    
    def test_softmax_numerical_stability(self):
        """Test softmax with large values doesn't overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.allclose(np.sum(result), 1.0)
    
    def test_softmax_batch(self):
        """Test softmax on batched input."""
        x = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
        result = softmax(x, axis=-1)
        assert result.shape == x.shape
        assert np.allclose(np.sum(result, axis=-1), [1.0, 1.0])
    
    def test_softmax_zero_input(self):
        """Test softmax with all zeros."""
        x = np.array([0.0, 0.0, 0.0])
        result = softmax(x)
        expected = np.array([1/3, 1/3, 1/3])
        assert np.allclose(result, expected)


class TestReLU:
    """Tests for ReLU activation function."""
    
    def test_relu_positive_values(self):
        """Test ReLU preserves positive values."""
        x = np.array([1.0, 2.0, 3.0])
        result = relu(x)
        assert np.array_equal(result, x)
    
    def test_relu_negative_values(self):
        """Test ReLU zeros out negative values."""
        x = np.array([-1.0, -2.0, -3.0])
        result = relu(x)
        expected = np.array([0.0, 0.0, 0.0])
        assert np.array_equal(result, expected)
    
    def test_relu_mixed_values(self):
        """Test ReLU with mixed positive and negative."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = relu(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert np.array_equal(result, expected)
    
    def test_relu_derivative(self):
        """Test ReLU derivative."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = relu_derivative(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        assert np.array_equal(result, expected)


class TestSigmoid:
    """Tests for sigmoid activation function."""
    
    def test_sigmoid_range(self):
        """Test sigmoid output is in (0, 1)."""
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = sigmoid(x)
        assert np.all(result > 0)
        assert np.all(result < 1)
    
    def test_sigmoid_zero(self):
        """Test sigmoid(0) = 0.5."""
        result = sigmoid(np.array([0.0]))
        assert np.allclose(result, 0.5)
    
    def test_sigmoid_symmetry(self):
        """Test sigmoid(-x) = 1 - sigmoid(x)."""
        x = np.array([1.0, 2.0, 3.0])
        pos_result = sigmoid(x)
        neg_result = sigmoid(-x)
        assert np.allclose(pos_result + neg_result, 1.0)
    
    def test_sigmoid_derivative(self):
        """Test sigmoid derivative."""
        x = np.array([0.0])
        sig_x = sigmoid(x)
        deriv = sigmoid_derivative(sig_x)
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        assert np.allclose(deriv, 0.25)


class TestTanh:
    """Tests for tanh activation function."""
    
    def test_tanh_range(self):
        """Test tanh output is in (-1, 1)."""
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = tanh(x)
        assert np.all(result > -1)
        assert np.all(result < 1)
    
    def test_tanh_zero(self):
        """Test tanh(0) = 0."""
        result = tanh(np.array([0.0]))
        assert np.allclose(result, 0.0)
    
    def test_tanh_symmetry(self):
        """Test tanh(-x) = -tanh(x)."""
        x = np.array([1.0, 2.0, 3.0])
        pos_result = tanh(x)
        neg_result = tanh(-x)
        assert np.allclose(pos_result, -neg_result)


class TestLeakyReLU:
    """Tests for Leaky ReLU activation function."""
    
    def test_leaky_relu_positive(self):
        """Test Leaky ReLU preserves positive values."""
        x = np.array([1.0, 2.0, 3.0])
        result = leaky_relu(x)
        assert np.array_equal(result, x)
    
    def test_leaky_relu_negative(self):
        """Test Leaky ReLU scales negative values."""
        x = np.array([-1.0, -2.0])
        result = leaky_relu(x, alpha=0.01)
        expected = np.array([-0.01, -0.02])
        assert np.allclose(result, expected)
    
    def test_leaky_relu_custom_alpha(self):
        """Test Leaky ReLU with custom alpha."""
        x = np.array([-1.0])
        result = leaky_relu(x, alpha=0.2)
        assert np.allclose(result, [-0.2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
