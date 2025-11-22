# 教程 1: Softmax 从零开始

## 🎯 学习目标

学完本教程后，你将能够：
1. 理解什么是 Softmax 函数
2. 知道为什么需要 Softmax
3. 理解数值稳定性问题
4. 能够自己实现 Softmax

## 📚 什么是 Softmax？

### 简单解释

想象你在参加一个选择题考试，有三个选项 A、B、C。你对每个选项的"信心分数"是：
- A: 2.0 分
- B: 1.0 分  
- C: 0.1 分

但是老师要求你给出每个选项的"概率"（加起来必须等于 100%）。这时候就需要 Softmax！

### 数学定义

Softmax 函数的公式是：

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

翻译成人话：
1. 对每个分数求指数（exp）
2. 把所有指数加起来
3. 用每个指数除以总和

## 🔢 手工计算示例

让我们用上面的例子计算：

**输入**: [2.0, 1.0, 0.1]

### 步骤 1: 计算指数

```python
import numpy as np

x = np.array([2.0, 1.0, 0.1])
exp_x = np.exp(x)
print(exp_x)
# 输出: [7.389, 2.718, 1.105]
```

为什么用指数？
- 指数函数总是正数
- 能放大差异（大的变得更大）

### 步骤 2: 求和

```python
sum_exp = np.sum(exp_x)
print(sum_exp)
# 输出: 11.212
```

### 步骤 3: 归一化

```python
probs = exp_x / sum_exp
print(probs)
# 输出: [0.659, 0.242, 0.099]

print(np.sum(probs))
# 输出: 1.0 ✓ 正确！
```

**解释结果**:
- 选项 A: 65.9% 的概率
- 选项 B: 24.2% 的概率
- 选项 C: 9.9% 的概率

## 💡 为什么需要 Softmax？

### 1. 分类任务

在图像分类中：
```python
# 模型对一张图片的判断分数
scores = {
    "猫": 3.5,
    "狗": 2.1,
    "鸟": 0.8
}

# 使用 Softmax 转换成概率
probs = softmax([3.5, 2.1, 0.8])
# 结果: [0.72, 0.21, 0.07]
# 意思是：72% 是猫，21% 是狗，7% 是鸟
```

### 2. 可解释性

Softmax 的输出：
- ✅ 都是正数
- ✅ 加起来等于 1
- ✅ 可以当作概率理解

### 3. 可微分

Softmax 是光滑的函数，可以求导，适合神经网络训练。

## ⚠️ 数值稳定性问题

### 问题：数值溢出

```python
# 危险的实现！
def bad_softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# 尝试大数字
x = np.array([1000, 1001, 1002])
result = bad_softmax(x)
print(result)
# 输出: [nan nan nan]  ❌ 溢出了！

# 原因：exp(1000) ≈ 10^434，超过浮点数上限
```

### 解决方案：减去最大值

**数学技巧**:
```
softmax(x) = softmax(x - max(x))
```

这不会改变结果，但能避免溢出！

**证明**:
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
             = exp(x_i - c) / Σ exp(x_j - c)  (c 是任意常数)
```

**安全的实现**:
```python
def safe_softmax(x):
    x_max = np.max(x)
    x_shifted = x - x_max  # 减去最大值
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

# 现在可以处理大数字了
x = np.array([1000, 1001, 1002])
result = safe_softmax(x)
print(result)
# 输出: [0.09, 0.24, 0.67]  ✅ 正确！
```

## 🧪 实践练习

### 练习 1: 手工计算

给定输入 `[1.0, 2.0, 3.0]`，手工计算 Softmax：

<details>
<summary>点击查看答案</summary>

```python
# 步骤 1: 计算指数
exp(1.0) = 2.718
exp(2.0) = 7.389
exp(3.0) = 20.086

# 步骤 2: 求和
sum = 2.718 + 7.389 + 20.086 = 30.193

# 步骤 3: 归一化
P(0) = 2.718 / 30.193 = 0.090
P(1) = 7.389 / 30.193 = 0.245
P(2) = 20.086 / 30.193 = 0.665
```
</details>

### 练习 2: 实现 Softmax

尝试自己实现一个稳定的 Softmax 函数：

```python
def my_softmax(x):
    # 你的代码写在这里
    pass

# 测试
x = np.array([2.0, 1.0, 0.1])
result = my_softmax(x)
print(result)
# 应该输出: [0.659, 0.242, 0.099]
```

<details>
<summary>点击查看答案</summary>

```python
def my_softmax(x):
    # 1. 减去最大值（数值稳定）
    x_max = np.max(x)
    x_shifted = x - x_max
    
    # 2. 计算指数
    exp_x = np.exp(x_shifted)
    
    # 3. 归一化
    sum_exp = np.sum(exp_x)
    return exp_x / sum_exp
```
</details>

### 练习 3: 批处理 Softmax

如何对一批数据（矩阵）应用 Softmax？

```python
# 一批三个样本，每个样本有 4 个类别
batch = np.array([
    [2.0, 1.0, 0.1, -1.0],
    [3.0, 2.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 1.0]
])

# 对每一行应用 Softmax
result = softmax_batch(batch)
print(result)
```

<details>
<summary>点击查看答案</summary>

```python
def softmax_batch(x, axis=-1):
    # 沿着指定轴减去最大值
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    
    # 计算指数和归一化
    exp_x = np.exp(x_shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    
    return exp_x / sum_exp

# 使用内置的 softmax
from deep_learning.activations import softmax
result = softmax(batch, axis=-1)

# 验证每行和为 1
print(np.sum(result, axis=-1))
# 输出: [1. 1. 1.]
```
</details>

## 🎮 温度参数（Temperature）

Softmax 可以加一个"温度"参数来控制输出的"锐度"：

```python
def softmax_with_temperature(x, temperature=1.0):
    x_scaled = x / temperature
    return softmax(x_scaled)

x = np.array([2.0, 1.0, 0.1])

# 低温：更"确定"的预测
print("温度 0.5:", softmax_with_temperature(x, 0.5))
# 输出: [0.86, 0.12, 0.02] - 更偏向最大值

# 正常温度
print("温度 1.0:", softmax_with_temperature(x, 1.0))
# 输出: [0.66, 0.24, 0.10] - 平衡

# 高温：更"模糊"的预测
print("温度 2.0:", softmax_with_temperature(x, 2.0))
# 输出: [0.50, 0.30, 0.19] - 更平均
```

**应用**:
- 低温：让模型更"自信"
- 高温：让模型更"谨慎"，增加多样性

## 🔍 运行演示代码

```bash
# 运行完整的 Softmax 演示
python examples/softmax_demo.py
```

这个演示会展示：
1. 基础 Softmax 计算
2. 温度缩放效果
3. 数值稳定性对比
4. 批处理示例
5. 可视化图表

## 📝 总结

### 关键要点

1. **Softmax 的作用**: 将任意实数转换为概率分布
2. **使用场景**: 多分类问题的输出层
3. **数值稳定**: 必须减去最大值避免溢出
4. **温度参数**: 控制输出的"尖锐"程度

### 下一步

学完 Softmax 后，可以继续学习：
- 交叉熵损失函数（Cross-Entropy Loss）
- 其他激活函数（ReLU, Sigmoid）
- 神经网络的前向传播

## 🤔 思考题

1. 为什么 Softmax 用指数函数而不是线性变换？
2. 如果所有输入都相同，Softmax 会输出什么？
3. Softmax 的导数是什么？（提示：雅可比矩阵）

---

**恭喜！** 你已经掌握了 Softmax 的基础知识。继续加油！🎉
