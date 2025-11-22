# 从零开始：完全新手指南

## 👋 欢迎！

如果你是完全的新手，这份指南会帮助你从零开始学习深度学习。不要担心，我们会一步一步来！

## 🎯 学习前的准备

### 你需要知道的

**必须**:
- ✅ 基础 Python（变量、函数、列表）
- ✅ 能够运行 Python 程序

**不需要**:
- ❌ 不需要深度学习经验
- ❌ 不需要高等数学
- ❌ 不需要 GPU

### 安装环境

#### 第一步：安装 Python

确保你有 Python 3.7 或更高版本：

```bash
python --version
```

如果没有，去 [python.org](https://python.org) 下载安装。

#### 第二步：克隆仓库

```bash
git clone https://github.com/DavidLi-TJ/skills-introduction-to-github.git
cd skills-introduction-to-github
```

#### 第三步：安装依赖

```bash
pip install -r requirements.txt
```

如果遇到问题，试试创建虚拟环境：

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate

# 然后安装
pip install -r requirements.txt
```

#### 第四步：测试安装

```bash
python examples/softmax_demo.py
```

如果看到输出，说明安装成功！🎉

## 📚 学习路线图

### 第 1 周：基础概念

#### 第 1 天：什么是神经网络？

神经网络就像一个"函数"：
- **输入**: 图片、文字、数字...
- **输出**: 分类、预测、翻译...

**例子**: 判断图片是猫还是狗
```
图片 → [神经网络] → "这是猫！"
```

#### 第 2 天：理解激活函数

激活函数是神经网络的"开关"。

**最简单的例子 - ReLU**:
```python
def relu(x):
    if x > 0:
        return x
    else:
        return 0

print(relu(5))   # 输出: 5
print(relu(-3))  # 输出: 0
```

**运行代码**:
```python
from deep_learning.activations import relu
import numpy as np

x = np.array([-2, -1, 0, 1, 2])
print(relu(x))
# 输出: [0 0 0 1 2]
```

#### 第 3-4 天：学习 Softmax

Softmax 把数字变成概率。

**阅读**: [Softmax 教程](01_softmax_tutorial_cn.md)

**运行**:
```bash
python examples/softmax_demo.py
```

**动手练习**:
```python
from deep_learning.activations import softmax
import numpy as np

# 三个类别的分数
scores = np.array([3.0, 1.0, 0.2])

# 转成概率
probs = softmax(scores)
print(f"猫: {probs[0]:.1%}")
print(f"狗: {probs[1]:.1%}")
print(f"鸟: {probs[2]:.1%}")

# 输出:
# 猫: 73.1%
# 狗: 19.9%
# 鸟: 7.0%
```

#### 第 5-7 天：理解神经网络层

**Linear 层是什么？**

就是矩阵乘法！

```python
from deep_learning.layers import Linear
import numpy as np

# 创建一个 3 → 2 的层
layer = Linear(in_features=3, out_features=2)

# 输入: 一个 3 维向量
x = np.array([[1.0, 2.0, 3.0]])

# 输出: 变成 2 维
output = layer.forward(x)
print(f"输入形状: {x.shape}")      # (1, 3)
print(f"输出形状: {output.shape}")  # (1, 2)
```

### 第 2 周：注意力机制

#### 第 8-10 天：理解注意力

**阅读**: [注意力机制教程](02_attention_tutorial_cn.md)

**运行**:
```bash
python examples/attention_demo.py
```

**核心概念**:
- Query: 我要找什么？
- Key: 这里有什么？
- Value: 实际内容

#### 第 11-14 天：多头注意力

多头注意力就像从多个角度看问题。

```python
from deep_learning.attention import MultiHeadAttention
import numpy as np

# 8 个头，每个头看不同的方面
mha = MultiHeadAttention(d_model=64, num_heads=8)

# 输入
x = np.random.randn(1, 5, 64)  # 5 个词，每个 64 维

# 计算注意力
output = mha.forward(x, x, x)
print(output.shape)  # (1, 5, 64)
```

### 第 3 周：Transformer

#### 第 15-17 天：Transformer 基础

**运行**:
```bash
python examples/transformer_demo.py
```

**Transformer 的核心**:
1. 位置编码（告诉模型词的位置）
2. 多头注意力（理解词之间的关系）
3. 前馈网络（进一步处理）

#### 第 18-21 天：实践 Transformer

```python
from deep_learning.transformer import Transformer
import numpy as np

# 创建一个小 Transformer
model = Transformer(
    src_vocab_size=1000,   # 源词汇量
    tgt_vocab_size=1000,   # 目标词汇量
    d_model=64,            # 小一点的维度
    num_heads=4,           # 4 个头
    num_encoder_layers=2,  # 2 层编码器
    num_decoder_layers=2   # 2 层解码器
)

# 模拟输入
src = np.array([[1, 2, 3, 4, 5]])  # 源句子
tgt = np.array([[1, 2, 3]])        # 目标句子

# 前向传播
output = model.forward(src, tgt)
print(f"输出形状: {output.shape}")
```

### 第 4 周：综合实践

#### 第 22-28 天：项目实践

选择一个小项目：

1. **文本分类**
   - 判断评论是正面还是负面
   
2. **简单翻译**
   - 数字序列转换
   
3. **序列预测**
   - 预测下一个数字

## 🎓 学习技巧

### 1. 每天坚持

每天 30 分钟比一周学一次效果好！

### 2. 动手实践

看懂代码不等于会写代码。一定要自己敲！

### 3. 改代码玩

```python
# 试着改这些参数
probs = softmax([1, 2, 3])  # 改成 [1, 1, 1] 会怎样？
layer = Linear(5, 3)         # 改成 Linear(10, 2) 呢？
```

### 4. 画图理解

```python
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = relu(x)

plt.plot(x, y)
plt.title('ReLU Function')
plt.show()
```

### 5. 问问题

不懂就问！可以：
- 在 [Issues](https://github.com/DavidLi-TJ/skills-introduction-to-github/issues) 提问
- 在 [Discussions](https://github.com/DavidLi-TJ/skills-introduction-to-github/discussions) 讨论

## 📖 必备资源

### Python 基础

如果 Python 不熟练：
- [廖雪峰 Python 教程](https://www.liaoxuefeng.com/wiki/1016959663602400)
- [菜鸟教程](https://www.runoob.com/python3/python3-tutorial.html)

### NumPy 基础

NumPy 是必须的：
```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3])

# 数组运算
b = a * 2  # [2, 4, 6]

# 矩阵乘法
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)
```

### 线性代数

需要了解的概念：
- 向量和矩阵
- 矩阵乘法
- 点积

**不需要**: 复杂的证明和推导

## 🐛 常见问题

### Q: 数学不好怎么办？

**A**: 不用担心！
- 我们从直觉开始
- 代码比公式更重要
- 边学边补数学

### Q: 代码报错怎么办？

**A**: 按这个顺序检查：

1. **看错误信息**
```python
# 如果是 ImportError
pip install -r requirements.txt

# 如果是 ModuleNotFoundError
# 确保在正确的目录
cd /path/to/skills-introduction-to-github
```

2. **检查 Python 版本**
```bash
python --version  # 应该是 3.7+
```

3. **重新安装**
```bash
pip uninstall numpy matplotlib pytest
pip install -r requirements.txt
```

### Q: 我应该记住所有公式吗？

**A**: 不需要！
- 理解概念比记公式重要
- 代码就是公式的实现
- 用的时候查就行

### Q: 学习需要多长时间？

**A**: 因人而异，但大概：
- 1 个月：掌握基础
- 3 个月：能实现简单模型
- 6 个月：能看懂论文

## 🎯 学习检查清单

### 基础级（第 1-2 周）

- [ ] 能运行所有示例代码
- [ ] 理解 Softmax 的作用
- [ ] 能解释 ReLU 为什么有用
- [ ] 知道什么是 Linear 层
- [ ] 理解矩阵乘法的形状变化

### 进阶级（第 3-4 周）

- [ ] 理解注意力机制的直觉
- [ ] 能解释 Query、Key、Value
- [ ] 知道为什么需要多头注意力
- [ ] 理解 Transformer 的结构
- [ ] 能修改代码参数并预测结果

### 高级（之后）

- [ ] 能从零实现简单的注意力
- [ ] 能解释位置编码的作用
- [ ] 理解残差连接和层归一化
- [ ] 能在小数据集上训练模型

## 💪 今天就开始！

不要想太多，从最简单的开始：

```bash
# 1. 运行第一个示例
python examples/softmax_demo.py

# 2. 打开代码看看
# 用任何文本编辑器打开 deep_learning/activations.py

# 3. 试着修改
# 改变 softmax_demo.py 里的输入值，看输出变化
```

记住：
- 🐢 慢慢来，不要急
- 💪 每天进步一点点
- 🤝 遇到问题就问
- 🎉 享受学习的过程

**祝你学习愉快！你一定可以的！🚀**

---

## 📞 需要帮助？

- 📧 [提交 Issue](https://github.com/DavidLi-TJ/skills-introduction-to-github/issues)
- 💬 [参与讨论](https://github.com/DavidLi-TJ/skills-introduction-to-github/discussions)
- 📚 查看其他教程：
  - [Softmax 详解](01_softmax_tutorial_cn.md)
  - [注意力机制详解](02_attention_tutorial_cn.md)
