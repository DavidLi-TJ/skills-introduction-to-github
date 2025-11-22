# 深度学习基础教程

_从零开始学习深度学习的基本概念_

[English](README.md) | 简体中文

## 欢迎 👋

这个仓库是为想要深入理解深度学习的初学者准备的。我们会从最基础的概念开始，一步步实现现代深度学习中的核心组件。

## 🎯 你会学到什么

- **适合人群**: 想要理解深度学习原理的初学者、学生和研究人员
- **学习内容**: Softmax、注意力机制、Transformer 等核心概念
- **实践项目**: 从零开始实现深度学习的基础组件
- **前置知识**: 基础的 Python 编程和线性代数知识
- **学习方法**: 通过实现来学习 - 理解现代 AI 背后的数学和代码

## 📚 学习路径（从零开始）

### 第一步：环境准备

```bash
# 1. 克隆这个仓库
git clone https://github.com/DavidLi-TJ/skills-introduction-to-github.git
cd skills-introduction-to-github

# 2. 安装依赖包
pip install -r requirements.txt

# 3. 运行第一个示例
python examples/softmax_demo.py
```

### 第二步：理解激活函数

**什么是激活函数？**

激活函数是神经网络中的"开关"，它决定神经元是否应该被激活。就像我们的大脑神经元一样，只有当信号足够强时才会传递信息。

#### 1. Softmax - 概率转换器 🎲

**用途**: 将任意数字转换为概率分布（所有值加起来等于 1）

**示例**:
```python
from deep_learning.activations import softmax
import numpy as np

# 假设有三个类别的分数
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

print(f"输入分数: {logits}")
print(f"输出概率: {probs}")
print(f"概率总和: {np.sum(probs)}")  # 总是等于 1.0

# 输出:
# 输入分数: [2.  1.  0.1]
# 输出概率: [0.659 0.242 0.099]
# 概率总和: 1.0
```

**应用场景**: 图像分类（判断图片是猫、狗还是鸟）

**数学公式**:
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

**重要特性**:
- ✅ 输出总是在 0 到 1 之间
- ✅ 所有输出的和等于 1
- ✅ 可以解释为概率

**运行示例**:
```bash
python examples/softmax_demo.py
```

#### 2. ReLU - 简单而强大 ⚡

**用途**: 保留正值，将负值变为零

**示例**:
```python
from deep_learning.activations import relu
import numpy as np

x = np.array([-2, -1, 0, 1, 2])
output = relu(x)

print(f"输入: {x}")
print(f"输出: {output}")

# 输出:
# 输入: [-2 -1  0  1  2]
# 输出: [0  0  0  1  2]
```

**为什么使用 ReLU？**
- 计算速度快
- 避免梯度消失问题
- 是现代神经网络中最常用的激活函数

#### 3. Sigmoid - 二分类专家 📊

**用途**: 将数值压缩到 0 和 1 之间

**示例**:
```python
from deep_learning.activations import sigmoid
import numpy as np

x = np.array([-2, 0, 2])
output = sigmoid(x)

print(f"输入: {x}")
print(f"输出: {output}")

# 输出:
# 输入: [-2  0  2]
# 输出: [0.119 0.5 0.881]
```

**应用场景**: 判断邮件是否是垃圾邮件（是/否）

### 第三步：构建神经网络层

#### 1. Linear 层 - 线性变换 🔗

**作用**: 对输入进行线性变换（矩阵乘法 + 偏置）

**示例**:
```python
from deep_learning.layers import Linear
import numpy as np

# 创建一个从 10 维到 5 维的线性层
layer = Linear(in_features=10, out_features=5)

# 输入: 批次大小为 2，每个样本 10 维
x = np.random.randn(2, 10)
output = layer.forward(x)

print(f"输入形状: {x.shape}")      # (2, 10)
print(f"输出形状: {output.shape}")  # (2, 5)
```

**数学公式**:
```
y = xW^T + b
```

其中:
- x: 输入
- W: 权重矩阵
- b: 偏置向量
- y: 输出

#### 2. Batch Normalization - 稳定训练 📈

**作用**: 标准化每一批数据，使训练更稳定、更快

**为什么重要？**
- 加速训练
- 允许使用更大的学习率
- 减少对初始化的依赖

#### 3. Dropout - 防止过拟合 🛡️

**作用**: 训练时随机"关闭"一些神经元

**示例**:
```python
from deep_learning.layers import Dropout
import numpy as np

dropout = Dropout(p=0.5)  # 50% 的神经元会被关闭
dropout.train()  # 训练模式

x = np.array([[1, 2, 3, 4, 5]])
output = dropout.forward(x)

print(f"输入: {x}")
print(f"输出: {output}")  # 约一半的值会变成 0
```

### 第四步：理解注意力机制 👀

**什么是注意力？**

想象你在读一本书，你的眼睛会自动关注重要的词语，而忽略不重要的部分。注意力机制让神经网络也能做同样的事情。

#### 核心概念：Query、Key、Value

这就像在图书馆找书：

- **Query (查询)**: "我想找关于机器学习的书"
- **Key (键)**: 每本书的标题和分类
- **Value (值)**: 书的实际内容

注意力机制会：
1. 比较你的查询和每本书的标签
2. 给每本书打分（相关性分数）
3. 根据分数加权组合书的内容

**示例**:
```python
from deep_learning.attention import scaled_dot_product_attention
import numpy as np

# 创建查询、键、值
batch_size, seq_len, d_k = 1, 3, 4

Q = np.random.randn(batch_size, seq_len, d_k)  # 查询
K = np.random.randn(batch_size, seq_len, d_k)  # 键
V = np.random.randn(batch_size, seq_len, d_k)  # 值

# 计算注意力
output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"注意力权重形状: {attention_weights.shape}")
print(f"注意力权重:\n{attention_weights[0]}")
print(f"每行的和: {attention_weights[0].sum(axis=1)}")  # 都是 1.0
```

**运行注意力示例**:
```bash
python examples/attention_demo.py
```

#### 多头注意力 (Multi-Head Attention) 🎭

就像从多个角度看同一个问题，多头注意力让模型从不同的"视角"关注输入。

**示例**:
```python
from deep_learning.attention import MultiHeadAttention
import numpy as np

# 创建多头注意力
mha = MultiHeadAttention(d_model=128, num_heads=8)

# 输入
x = np.random.randn(2, 10, 128)  # (批次, 序列长度, 维度)

# 自注意力
output = mha.forward(x, x, x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # 形状保持不变
```

### 第五步：构建 Transformer 🚀

**什么是 Transformer？**

Transformer 是现代 AI（如 ChatGPT、BERT）的核心架构。它完全基于注意力机制，不需要循环结构。

#### Transformer 的组成部分

1. **位置编码 (Positional Encoding)**: 告诉模型每个词的位置
2. **编码器 (Encoder)**: 理解输入
3. **解码器 (Decoder)**: 生成输出
4. **多头注意力**: 从多个角度理解内容
5. **前馈网络**: 进一步处理信息

**示例**:
```python
from deep_learning.transformer import Transformer
import numpy as np

# 创建一个小型 Transformer
transformer = Transformer(
    src_vocab_size=1000,   # 源语言词汇量
    tgt_vocab_size=1000,   # 目标语言词汇量
    d_model=128,           # 模型维度
    num_heads=4,           # 注意力头数
    num_encoder_layers=2,  # 编码器层数
    num_decoder_layers=2,  # 解码器层数
    d_ff=512,              # 前馈网络维度
    dropout=0.1            # Dropout 比率
)

# 创建输入
batch_size = 2
src = np.random.randint(0, 1000, (batch_size, 10))  # 源序列
tgt = np.random.randint(0, 1000, (batch_size, 8))   # 目标序列

# 前向传播
output = transformer.forward(src, tgt)

print(f"源序列形状: {src.shape}")
print(f"目标序列形状: {tgt.shape}")
print(f"输出形状: {output.shape}")  # (批次, 目标序列长度, 词汇量)
```

**运行 Transformer 示例**:
```bash
python examples/transformer_demo.py
```

## 🧪 运行测试

我们为所有实现编写了测试，确保代码正确：

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_activations.py -v
python -m pytest tests/test_attention.py -v
```

## 📖 详细教程

### 教程 1: 从零理解 Softmax

Softmax 是神经网络中最重要的函数之一。让我们深入理解它：

**问题**: 我们有三个类别的分数 [2.0, 1.0, 0.1]，如何转换成概率？

**步骤**:

1. **计算指数**:
   ```
   exp(2.0) = 7.389
   exp(1.0) = 2.718
   exp(0.1) = 1.105
   ```

2. **求和**:
   ```
   总和 = 7.389 + 2.718 + 1.105 = 11.212
   ```

3. **归一化**:
   ```
   P(类别0) = 7.389 / 11.212 = 0.659
   P(类别1) = 2.718 / 11.212 = 0.242
   P(类别2) = 1.105 / 11.212 = 0.099
   ```

**数值稳定性问题**:

当数字很大时（如 1000），exp(1000) 会溢出！

**解决方案**: 减去最大值
```python
# 不稳定的实现
def naive_softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# 稳定的实现
def stable_softmax(x):
    x_max = np.max(x)
    x_shifted = x - x_max  # 减去最大值
    return np.exp(x_shifted) / np.sum(np.exp(x_shifted))
```

### 教程 2: 注意力机制详解

**场景**: 机器翻译

假设我们要把 "I love you" 翻译成中文 "我爱你"。

当翻译 "爱" 这个字时，注意力机制会：
1. 查看英文句子的所有词
2. 发现 "love" 最相关
3. 主要使用 "love" 的信息来生成 "爱"

**数学表达**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

其中:
- Q (Query): 当前要翻译的词的"需求"
- K (Key): 每个英文词的"标签"
- V (Value): 每个英文词的"内容"

### 教程 3: Transformer 架构

Transformer 的创新之处：
1. **并行处理**: 不像 RNN 需要逐个处理，可以同时处理所有词
2. **长距离依赖**: 可以轻松捕捉相距很远的词之间的关系
3. **可扩展性**: 容易扩展到更大的模型

**编码器的工作流程**:
```
输入文本 → 词嵌入 → 位置编码 → 
    → 多头注意力 → 残差连接 → 层归一化 →
    → 前馈网络 → 残差连接 → 层归一化 →
    → 输出
```

**解码器的工作流程**:
```
目标文本 → 词嵌入 → 位置编码 →
    → 掩码多头注意力（只看之前的词）→ 残差连接 → 层归一化 →
    → 多头注意力（看编码器输出）→ 残差连接 → 层归一化 →
    → 前馈网络 → 残差连接 → 层归一化 →
    → 输出
```

## 🎓 学习建议

### 对于完全的初学者：

1. **第一周**: 
   - 理解 softmax 和 ReLU
   - 运行 `softmax_demo.py`
   - 尝试修改参数，观察输出变化

2. **第二周**:
   - 学习 Linear 层
   - 理解矩阵乘法
   - 手动计算一个简单的例子

3. **第三周**:
   - 理解注意力机制
   - 运行 `attention_demo.py`
   - 可视化注意力权重

4. **第四周**:
   - 学习 Transformer
   - 运行 `transformer_demo.py`
   - 尝试修改层数和维度

### 推荐学习资源：

- **数学基础**: 线性代数、微积分基础
- **经典论文**: "Attention Is All You Need" (Vaswani et al., 2017)
- **进阶阅读**: Deep Learning Book by Goodfellow et al.

## 🔧 项目结构

```
├── deep_learning/          # 核心实现
│   ├── __init__.py
│   ├── activations.py      # 激活函数
│   ├── layers.py           # 神经网络层
│   ├── attention.py        # 注意力机制
│   └── transformer.py      # Transformer 架构
├── examples/               # 示例代码
│   ├── softmax_demo.py     # Softmax 演示
│   ├── attention_demo.py   # 注意力演示
│   └── transformer_demo.py # Transformer 演示
├── tests/                  # 单元测试
│   ├── test_activations.py
│   └── test_attention.py
├── README.md               # 英文文档
├── README_CN.md            # 中文文档（本文件）
└── requirements.txt        # 依赖包
```

## ❓ 常见问题

### Q1: 我需要什么基础知识？

**A**: 
- Python 基础语法（变量、函数、类）
- 基础线性代数（向量、矩阵）
- NumPy 基础操作

### Q2: 代码运行不了怎么办？

**A**: 
```bash
# 1. 检查 Python 版本（需要 3.7+）
python --version

# 2. 重新安装依赖
pip install -r requirements.txt

# 3. 如果还有问题，创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Q3: 如何调试代码？

**A**:
```python
# 添加打印语句
print(f"变量 x 的形状: {x.shape}")
print(f"变量 x 的值:\n{x}")

# 使用 Python 调试器
import pdb; pdb.set_trace()
```

### Q4: 为什么需要这么多数学？

**A**: 数学是深度学习的语言。但不要担心：
- 我们从简单的例子开始
- 每个公式都有代码实现
- 重点是理解直觉，而不是记忆公式

### Q5: 学完这个仓库后，下一步学什么？

**A**:
1. 尝试在真实数据集上应用（MNIST、CIFAR-10）
2. 学习 PyTorch 或 TensorFlow 框架
3. 实现更复杂的模型（GPT、BERT）
4. 参与开源项目

## 🤝 如何贡献

欢迎贡献！你可以：
- 添加更多示例
- 改进中文文档
- 修复 bug
- 添加新的功能

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 📮 获取帮助

- [发起讨论](https://github.com/DavidLi-TJ/skills-introduction-to-github/discussions)
- [报告问题](https://github.com/DavidLi-TJ/skills-introduction-to-github/issues)

---

**记住**: 学习深度学习是一个循序渐进的过程。不要着急，每次理解一个概念，然后动手实践。祝你学习愉快！🚀

© 2024 深度学习教育项目
