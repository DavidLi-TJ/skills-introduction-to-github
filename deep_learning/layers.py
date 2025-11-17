"""
Neural Network Layer Implementations

This module provides fundamental building blocks for neural networks.
"""

import numpy as np


class Linear:
    """
    Fully connected (dense) layer for neural networks.
    
    Implements the operation: y = xW^T + b
    where W is the weight matrix and b is the bias vector.
    
    Attributes:
        in_features (int): Number of input features
        out_features (int): Number of output features
        weights (np.ndarray): Weight matrix of shape (out_features, in_features)
        bias (np.ndarray): Bias vector of shape (out_features,)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize linear layer with Xavier/Glorot initialization.
        
        Args:
            in_features (int): Size of input features
            out_features (int): Size of output features
            bias (bool): Whether to include bias term (default: True)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Xavier initialization for better gradient flow
        limit = np.sqrt(6 / (in_features + out_features))
        self.weights = np.random.uniform(-limit, limit, 
                                        (out_features, in_features))
        
        if self.use_bias:
            self.bias = np.zeros(out_features)
        else:
            self.bias = None
        
        # For storing gradients during backpropagation
        self.grad_weights = None
        self.grad_bias = None
        self.input_cache = None
    
    def forward(self, x):
        """
        Forward pass through the linear layer.
        
        Args:
            x (np.ndarray): Input of shape (batch_size, in_features)
        
        Returns:
            np.ndarray: Output of shape (batch_size, out_features)
        """
        self.input_cache = x
        output = np.dot(x, self.weights.T)
        
        if self.use_bias:
            output += self.bias
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass to compute gradients.
        
        Args:
            grad_output (np.ndarray): Gradient from next layer
        
        Returns:
            np.ndarray: Gradient with respect to input
        """
        # Gradient with respect to weights
        self.grad_weights = np.dot(grad_output.T, self.input_cache)
        
        # Gradient with respect to bias
        if self.use_bias:
            self.grad_bias = np.sum(grad_output, axis=0)
        
        # Gradient with respect to input
        grad_input = np.dot(grad_output, self.weights)
        
        return grad_input
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"


class BatchNorm:
    """
    Batch Normalization layer.
    
    Normalizes inputs across the batch dimension to have zero mean
    and unit variance, which helps with training stability and speed.
    
    Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Initialize batch normalization layer.
        
        Args:
            num_features (int): Number of features/channels
            eps (float): Small constant for numerical stability
            momentum (float): Momentum for running statistics
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)  # Scale
        self.beta = np.zeros(num_features)   # Shift
        
        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache for backpropagation
        self.cache = None
        self.training = True
    
    def forward(self, x):
        """
        Forward pass through batch normalization.
        
        Args:
            x (np.ndarray): Input of shape (batch_size, num_features)
        
        Returns:
            np.ndarray: Normalized output
        """
        if self.training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * batch_var
            
            # Normalize
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Cache values for backward pass
            self.cache = (x, x_normalized, batch_mean, batch_var)
        else:
            # Use running statistics for inference
            x_normalized = (x - self.running_mean) / \
                          np.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_normalized + self.beta
        
        return out
    
    def backward(self, grad_output):
        """
        Backward pass through batch normalization.
        
        Args:
            grad_output (np.ndarray): Gradient from next layer
        
        Returns:
            np.ndarray: Gradient with respect to input
        """
        x, x_normalized, batch_mean, batch_var = self.cache
        batch_size = x.shape[0]
        
        # Gradients with respect to gamma and beta
        grad_gamma = np.sum(grad_output * x_normalized, axis=0)
        grad_beta = np.sum(grad_output, axis=0)
        
        # Gradient with respect to normalized input
        grad_x_normalized = grad_output * self.gamma
        
        # Gradient with respect to variance
        grad_var = np.sum(grad_x_normalized * (x - batch_mean) * 
                         -0.5 * (batch_var + self.eps) ** (-1.5), axis=0)
        
        # Gradient with respect to mean
        grad_mean = np.sum(grad_x_normalized * -1 / np.sqrt(batch_var + self.eps), axis=0) + \
                   grad_var * np.sum(-2 * (x - batch_mean), axis=0) / batch_size
        
        # Gradient with respect to input
        grad_x = grad_x_normalized / np.sqrt(batch_var + self.eps) + \
                grad_var * 2 * (x - batch_mean) / batch_size + \
                grad_mean / batch_size
        
        return grad_x
    
    def train(self):
        """Set layer to training mode."""
        self.training = True
    
    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False


class Dropout:
    """
    Dropout layer for regularization.
    
    During training, randomly sets a fraction of input units to 0
    to prevent overfitting. During inference, all units are kept.
    """
    
    def __init__(self, p=0.5):
        """
        Initialize dropout layer.
        
        Args:
            p (float): Probability of dropping a unit (default: 0.5)
        """
        self.p = p
        self.training = True
        self.mask = None
    
    def forward(self, x):
        """
        Forward pass through dropout.
        
        Args:
            x (np.ndarray): Input array
        
        Returns:
            np.ndarray: Output with dropout applied
        """
        if self.training and self.p > 0:
            # Create dropout mask
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape)
            # Scale by 1/(1-p) to maintain expected value (inverted dropout)
            return x * self.mask / (1 - self.p)
        else:
            return x
    
    def backward(self, grad_output):
        """
        Backward pass through dropout.
        
        Args:
            grad_output (np.ndarray): Gradient from next layer
        
        Returns:
            np.ndarray: Gradient with dropout mask applied
        """
        if self.training and self.p > 0:
            return grad_output * self.mask / (1 - self.p)
        else:
            return grad_output
    
    def train(self):
        """Set layer to training mode."""
        self.training = True
    
    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False
    
    def __repr__(self):
        return f"Dropout(p={self.p})"
