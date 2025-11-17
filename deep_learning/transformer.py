"""
Transformer Architecture Implementation

This module implements the Transformer architecture introduced in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

import numpy as np
from .attention import MultiHeadAttention
from .layers import Linear, Dropout
from .activations import relu


class PositionalEncoding:
    """
    Positional Encoding for Transformers.
    
    Since Transformers don't have recurrence or convolution, we need to inject
    information about the position of tokens in the sequence. This is done using
    sinusoidal positional encodings.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Dimension of the model
            max_len (int): Maximum sequence length (default: 5000)
        """
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        
        # Compute the div_term for the sinusoidal functions
        div_term = np.exp(np.arange(0, d_model, 2) * 
                         -(np.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (np.ndarray): Input embeddings of shape (batch_size, seq_len, d_model)
        
        Returns:
            np.ndarray: Input with positional encoding added
        """
        seq_len = x.shape[1]
        # Add positional encoding (broadcasting across batch dimension)
        return x + self.pe[:seq_len, :]
    
    def __repr__(self):
        return f"PositionalEncoding(d_model={self.d_model})"


class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    This is applied to each position separately and identically.
    Consists of two linear transformations with a ReLU activation in between.
    
    FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize feed-forward network.
        
        Args:
            d_model (int): Dimension of the model
            d_ff (int): Dimension of the feed-forward hidden layer
            dropout (float): Dropout probability
        """
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass through feed-forward network.
        
        Args:
            x (np.ndarray): Input of shape (batch_size, seq_len, d_model)
        
        Returns:
            np.ndarray: Output of shape (batch_size, seq_len, d_model)
        """
        # First linear transformation + ReLU
        x = self.linear1.forward(x)
        x = relu(x)
        x = self.dropout.forward(x)
        
        # Second linear transformation
        x = self.linear2.forward(x)
        
        return x


class TransformerEncoderLayer:
    """
    Single Transformer Encoder Layer.
    
    Consists of:
    1. Multi-head self-attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-forward network
    4. Add & Norm
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize encoder layer.
        
        Args:
            d_model (int): Dimension of the model
            num_heads (int): Number of attention heads
            d_ff (int): Dimension of feed-forward hidden layer
            dropout (float): Dropout probability
        """
        self.d_model = d_model
        
        # Multi-head attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Dropout layers
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        # Layer normalization parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        self.eps = 1e-6
    
    def layer_norm(self, x, gamma, beta):
        """
        Apply layer normalization.
        
        Args:
            x (np.ndarray): Input
            gamma (np.ndarray): Scale parameter
            beta (np.ndarray): Shift parameter
        
        Returns:
            np.ndarray: Normalized output
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return gamma * x_norm + beta
    
    def forward(self, x, mask=None):
        """
        Forward pass through encoder layer.
        
        Args:
            x (np.ndarray): Input of shape (batch_size, seq_len, d_model)
            mask (np.ndarray, optional): Attention mask
        
        Returns:
            np.ndarray: Output of shape (batch_size, seq_len, d_model)
        """
        # Multi-head self-attention with residual connection
        attn_output = self.self_attention.forward(x, x, x, mask)
        attn_output = self.dropout1.forward(attn_output)
        x = x + attn_output
        x = self.layer_norm(x, self.gamma1, self.beta1)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward.forward(x)
        ff_output = self.dropout2.forward(ff_output)
        x = x + ff_output
        x = self.layer_norm(x, self.gamma2, self.beta2)
        
        return x


class TransformerDecoderLayer:
    """
    Single Transformer Decoder Layer.
    
    Consists of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head cross-attention (attending to encoder output)
    4. Add & Norm
    5. Feed-forward network
    6. Add & Norm
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize decoder layer.
        
        Args:
            d_model (int): Dimension of the model
            num_heads (int): Number of attention heads
            d_ff (int): Dimension of feed-forward hidden layer
            dropout (float): Dropout probability
        """
        self.d_model = d_model
        
        # Self-attention (masked)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Cross-attention (attending to encoder)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Dropout layers
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        # Layer normalization parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        self.gamma3 = np.ones(d_model)
        self.beta3 = np.zeros(d_model)
        self.eps = 1e-6
    
    def layer_norm(self, x, gamma, beta):
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return gamma * x_norm + beta
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass through decoder layer.
        
        Args:
            x (np.ndarray): Decoder input (batch_size, tgt_seq_len, d_model)
            encoder_output (np.ndarray): Encoder output (batch_size, src_seq_len, d_model)
            src_mask (np.ndarray, optional): Source mask
            tgt_mask (np.ndarray, optional): Target mask (causal)
        
        Returns:
            np.ndarray: Output of shape (batch_size, tgt_seq_len, d_model)
        """
        # Masked self-attention with residual
        self_attn_output = self.self_attention.forward(x, x, x, tgt_mask)
        self_attn_output = self.dropout1.forward(self_attn_output)
        x = x + self_attn_output
        x = self.layer_norm(x, self.gamma1, self.beta1)
        
        # Cross-attention with encoder output
        cross_attn_output = self.cross_attention.forward(
            x, encoder_output, encoder_output, src_mask
        )
        cross_attn_output = self.dropout2.forward(cross_attn_output)
        x = x + cross_attn_output
        x = self.layer_norm(x, self.gamma2, self.beta2)
        
        # Feed-forward with residual
        ff_output = self.feed_forward.forward(x)
        ff_output = self.dropout3.forward(ff_output)
        x = x + ff_output
        x = self.layer_norm(x, self.gamma3, self.beta3)
        
        return x


class Transformer:
    """
    Complete Transformer model for sequence-to-sequence tasks.
    
    The Transformer uses stacked encoder and decoder layers with
    multi-head attention to process sequential data without recurrence.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, dropout=0.1, max_len=5000):
        """
        Initialize Transformer model.
        
        Args:
            src_vocab_size (int): Size of source vocabulary
            tgt_vocab_size (int): Size of target vocabulary
            d_model (int): Dimension of the model
            num_heads (int): Number of attention heads
            num_encoder_layers (int): Number of encoder layers
            num_decoder_layers (int): Number of decoder layers
            d_ff (int): Dimension of feed-forward hidden layer
            dropout (float): Dropout probability
            max_len (int): Maximum sequence length
        """
        self.d_model = d_model
        
        # Embedding layers
        self.src_embedding = np.random.randn(src_vocab_size, d_model) * 0.01
        self.tgt_embedding = np.random.randn(tgt_vocab_size, d_model) * 0.01
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ]
        
        # Decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ]
        
        # Output projection
        self.output_projection = Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = Dropout(dropout)
    
    def encode(self, src, src_mask=None):
        """
        Encode source sequence.
        
        Args:
            src (np.ndarray): Source token indices (batch_size, src_seq_len)
            src_mask (np.ndarray, optional): Source mask
        
        Returns:
            np.ndarray: Encoder output (batch_size, src_seq_len, d_model)
        """
        # Embed and scale
        x = self.src_embedding[src] * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding.forward(x)
        x = self.dropout.forward(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer.forward(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decode target sequence.
        
        Args:
            tgt (np.ndarray): Target token indices (batch_size, tgt_seq_len)
            encoder_output (np.ndarray): Encoder output
            src_mask (np.ndarray, optional): Source mask
            tgt_mask (np.ndarray, optional): Target mask
        
        Returns:
            np.ndarray: Decoder output (batch_size, tgt_seq_len, d_model)
        """
        # Embed and scale
        x = self.tgt_embedding[tgt] * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding.forward(x)
        x = self.dropout.forward(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer.forward(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass through the entire Transformer.
        
        Args:
            src (np.ndarray): Source token indices (batch_size, src_seq_len)
            tgt (np.ndarray): Target token indices (batch_size, tgt_seq_len)
            src_mask (np.ndarray, optional): Source mask
            tgt_mask (np.ndarray, optional): Target mask
        
        Returns:
            np.ndarray: Logits for each token (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Encode source
        encoder_output = self.encode(src, src_mask)
        
        # Decode target
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        # Reshape for linear layer
        batch_size, seq_len, d_model = decoder_output.shape
        decoder_output_flat = decoder_output.reshape(-1, d_model)
        output_flat = self.output_projection.forward(decoder_output_flat)
        output = output_flat.reshape(batch_size, seq_len, -1)
        
        return output
    
    def __repr__(self):
        return (f"Transformer(d_model={self.d_model}, "
                f"num_encoder_layers={len(self.encoder_layers)}, "
                f"num_decoder_layers={len(self.decoder_layers)})")
