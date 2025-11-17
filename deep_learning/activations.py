"""
Activation Functions for Neural Networks

This module implements common activation functions used in deep learning,
with a focus on educational clarity and numerical stability.
"""

import numpy as np


def softmax(x, axis=-1):
    """
    Compute the softmax function with numerical stability.
    
    The softmax function converts a vector of real numbers into a probability
    distribution. It's commonly used in the output layer of classification networks.
    
    Mathematical formula:
        softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    
    For numerical stability, we subtract the maximum value:
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    Args:
        x (np.ndarray): Input array of any shape
        axis (int): Axis along which to compute softmax (default: -1)
    
    Returns:
        np.ndarray: Softmax probabilities with the same shape as input
    
    Examples:
        >>> logits = np.array([2.0, 1.0, 0.1])
        >>> probs = softmax(logits)
        >>> print(probs)
        [0.659 0.242 0.099]
        >>> print(np.sum(probs))  # Should sum to 1
        1.0
    """
    # Subtract max for numerical stability (prevents overflow)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    
    # Compute exponentials
    exp_x = np.exp(x_shifted)
    
    # Normalize to get probabilities
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_derivative(softmax_output):
    """
    Compute the derivative of softmax for backpropagation.
    
    For a single softmax output s, the Jacobian is:
        ds_i/dx_j = s_i * (δ_ij - s_j)
    where δ_ij is the Kronecker delta.
    
    Args:
        softmax_output (np.ndarray): Output from softmax function
    
    Returns:
        np.ndarray: Jacobian matrix of derivatives
    """
    s = softmax_output.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.
    
    ReLU is the most commonly used activation function in hidden layers.
    It introduces non-linearity while being computationally efficient.
    
    Formula: relu(x) = max(0, x)
    
    Args:
        x (np.ndarray): Input array
    
    Returns:
        np.ndarray: Output with negative values set to 0
    
    Examples:
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> relu(x)
        array([0, 0, 0, 1, 2])
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU for backpropagation.
    
    The derivative is 1 for x > 0, and 0 for x <= 0.
    
    Args:
        x (np.ndarray): Input array
    
    Returns:
        np.ndarray: Gradient (1 where x > 0, 0 elsewhere)
    """
    return (x > 0).astype(float)


def sigmoid(x):
    """
    Sigmoid activation function.
    
    The sigmoid function squashes input values to the range (0, 1).
    It's commonly used for binary classification.
    
    Formula: sigmoid(x) = 1 / (1 + exp(-x))
    
    Args:
        x (np.ndarray): Input array
    
    Returns:
        np.ndarray: Output in range (0, 1)
    
    Examples:
        >>> x = np.array([-2, 0, 2])
        >>> sigmoid(x)
        array([0.119, 0.5, 0.881])
    """
    # Clip to prevent overflow
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))


def sigmoid_derivative(sigmoid_output):
    """
    Derivative of sigmoid for backpropagation.
    
    The derivative in terms of the sigmoid output is:
        sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    
    Args:
        sigmoid_output (np.ndarray): Output from sigmoid function
    
    Returns:
        np.ndarray: Gradient
    """
    return sigmoid_output * (1 - sigmoid_output)


def tanh(x):
    """
    Hyperbolic tangent activation function.
    
    Tanh squashes input values to the range (-1, 1).
    It's zero-centered, which can make optimization easier.
    
    Formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Args:
        x (np.ndarray): Input array
    
    Returns:
        np.ndarray: Output in range (-1, 1)
    
    Examples:
        >>> x = np.array([-2, 0, 2])
        >>> tanh(x)
        array([-0.964, 0.0, 0.964])
    """
    return np.tanh(x)


def tanh_derivative(tanh_output):
    """
    Derivative of tanh for backpropagation.
    
    The derivative in terms of the tanh output is:
        tanh'(x) = 1 - tanh(x)^2
    
    Args:
        tanh_output (np.ndarray): Output from tanh function
    
    Returns:
        np.ndarray: Gradient
    """
    return 1 - tanh_output ** 2


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    
    Leaky ReLU addresses the "dying ReLU" problem by allowing
    a small gradient when x < 0.
    
    Formula: leaky_relu(x) = x if x > 0 else alpha * x
    
    Args:
        x (np.ndarray): Input array
        alpha (float): Slope for negative values (default: 0.01)
    
    Returns:
        np.ndarray: Output array
    
    Examples:
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> leaky_relu(x)
        array([-0.02, -0.01, 0., 1., 2.])
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of Leaky ReLU for backpropagation.
    
    Args:
        x (np.ndarray): Input array
        alpha (float): Slope for negative values
    
    Returns:
        np.ndarray: Gradient
    """
    return np.where(x > 0, 1, alpha)
