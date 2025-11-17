# Deep Learning Fundamentals

_Learn deep learning by implementing fundamental concepts from scratch._

## Overview

This repository provides educational implementations of core deep learning concepts, built from the ground up to help you understand how modern neural networks work. Each implementation is designed for clarity and learning, with detailed comments and examples.

## What You'll Learn

- **Who is this for**: Students, researchers, and practitioners wanting to understand deep learning fundamentals
- **What you'll learn**: Core concepts including softmax, attention mechanisms, transformers, and more
- **What you'll build**: Working implementations of fundamental deep learning components
- **Prerequisites**: Basic Python programming and familiarity with linear algebra
- **Approach**: Learn by implementing - understand the math and code behind modern AI

## Contents

### 1. **Activation Functions**
   - Softmax implementation with numerical stability
   - ReLU, Sigmoid, Tanh variants
   - Forward and backward passes

### 2. **Neural Network Components**
   - Linear layers with weight initialization
   - Batch normalization
   - Dropout for regularization

### 3. **Attention Mechanisms**
   - Scaled dot-product attention
   - Multi-head attention
   - Self-attention visualization

### 4. **Transformer Architecture**
   - Complete transformer implementation
   - Positional encoding
   - Encoder and decoder blocks

## Getting Started

```bash
# Clone the repository
git clone https://github.com/DavidLi-TJ/skills-introduction-to-github.git
cd skills-introduction-to-github

# Install dependencies
pip install -r requirements.txt

# Run examples
python examples/softmax_demo.py
python examples/transformer_demo.py
```

## Project Structure

```
├── deep_learning/          # Core implementations
│   ├── activations.py      # Activation functions (softmax, relu, etc.)
│   ├── layers.py           # Neural network layers
│   ├── attention.py        # Attention mechanisms
│   └── transformer.py      # Transformer architecture
├── examples/               # Usage examples and demos
├── tests/                  # Unit tests
└── requirements.txt        # Python dependencies
```

## Learning Path

1. **Start with Activations** (`deep_learning/activations.py`)
   - Understand softmax and its numerical stability
   - Learn about gradient computation

2. **Build Neural Network Layers** (`deep_learning/layers.py`)
   - Implement linear transformations
   - Add normalization and regularization

3. **Master Attention** (`deep_learning/attention.py`)
   - Implement scaled dot-product attention
   - Build multi-head attention from scratch

4. **Create Transformers** (`deep_learning/transformer.py`)
   - Combine all components into a transformer
   - Understand positional encoding

## Examples

Each concept includes practical examples:

```python
# Softmax example
from deep_learning.activations import softmax
import numpy as np

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Probabilities: {probs}")
# Output: [0.659, 0.242, 0.099]
```

## Testing

Run unit tests to verify implementations:

```bash
python -m pytest tests/
```

## Resources

- **Mathematical Foundations**: Linear algebra, calculus basics
- **Papers**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Further Reading**: Deep Learning Book by Goodfellow et al.

## Contributing

This is an educational repository. Feel free to:
- Add more examples
- Improve documentation
- Fix bugs or add tests
- Suggest new topics

## License

MIT License - See LICENSE file for details

---

Get help: [Post in discussions](https://github.com/DavidLi-TJ/skills-introduction-to-github/discussions) • [Report issues](https://github.com/DavidLi-TJ/skills-introduction-to-github/issues)

© 2024 Educational Deep Learning Project
