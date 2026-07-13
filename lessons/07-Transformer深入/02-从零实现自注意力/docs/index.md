# 从零实现自注意力

> 注意力是一个查询表，每个词都在问"谁对我重要？"——并学习答案。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 3（深度学习核心）、阶段 05 · 09（序列到序列）
**时间：** ~90 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 用 NumPy 从零实现缩放点积自注意力——包括 Q/K/V 投影和 softmax 加权求和
- [ ] 构建多头注意力层——拆分头、并行计算、拼接结果
- [ ] 追踪注意力矩阵如何捕获词元关系，解释 √d_k 缩放为什么防止 softmax 饱和
- [ ] 应用因果掩码将双向注意力转换为自回归（解码器风格）注意力

---

## 1. 问题

RNN 一个词元一个词元地处理序列。到第 50 个词元时，第 1 个词元的信息已经被压缩过 50 个非线性变换。长距离依赖被压缩到固定大小的隐藏状态——这是 LSTM 门控无法完全解决的瓶颈。

2014 年 Bahdanau 注意力论文指出了修复方法：让解码器回看每个编码器位置并决定哪些对当前步骤重要。但它是附加在 RNN 上的。2017 年"Attention Is All You Need"问了一个更尖锐的问题：**如果注意力是唯一的机制会怎样？** 没有递归。没有卷积。只有注意力。

自注意力让序列中的每个位置在一个并行步骤中关注其他所有位置。这就是 Transformer 快速、可扩展、占主导的原因。

---

## 2. 概念

### 2.1 数据库查询类比

把注意力想象成一个软数据库查询：

```
传统数据库：
  查询: "法国首都"  →  精确匹配  →  "巴黎"

注意力：
  查询: "法国首都"  →  与所有键的相似度  →  所有值的加权混合
```

每个词元生成三个向量：
- **查询 (Q)：** "我在找什么？"
- **键 (K)：** "我包含什么？"
- **值 (V)：** "如果被选中，我提供什么信息？"

查询与所有键的点积产生注意力分数。高分意味着"这个键匹配我的查询"。这些分数对值加权。输出是值的加权求和。

### 2.2 Q、K、V 计算

每个词元嵌入通过三个可学习的权重矩阵投影：

```
输入嵌入（n 个词元，每个 d 维）：

  X = [x1, x2, ..., xn]    shape: (n, d)

三个权重矩阵：
  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

投影：
  Q = X @ Wq    shape: (n, dk)    每个词元的查询
  K = X @ Wk    shape: (n, dk)    每个词元的键
  V = X @ Wv    shape: (n, dv)    每个词元的值
```

### 2.3 注意力矩阵

所有词元的 Q、K、V 得到注意力分数矩阵：

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   ← q1 对每个 key 的关注程度
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        ...
```

### 2.4 为什么要缩放？

点积随维度 dk 增长。如果 dk=64，点积可达几十，将 softmax 推入梯度消失区域。修复：除以 √dk。

```
缩放分数 = (Q @ K^T) / sqrt(dk)
```

### 2.5 完整流程

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

---

## 3. 从零实现

### Step 1：Softmax

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

减去最大值保证数值稳定——不溢出。

### Step 2：缩放点积注意力

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### Step 3：可学习投影的自注意力类

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### Step 4：在句子上运行

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
d_model, dk, dv = 16, 8, 8
rng = np.random.default_rng(42)
X = rng.normal(0, 1, (len(sentence), d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("注意力权重（每行：该词关注哪些词）:")
# [cat] -> 0.52 * [cat] + 0.09 * [sat] + ... → 最关注自己
```

### Step 5：多头注意力

```python
class MultiHeadSelfAttention:
    def __init__(self, d_model, n_heads, seed=42):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dk = self.dv = d_model // n_heads
        self.heads = [
            SelfAttention(d_model, self.dk, self.dv, seed=seed+i)
            for i in range(n_heads)
        ]
        rng = np.random.default_rng(seed + n_heads)
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.Wo = rng.normal(0, scale, (n_heads * self.dv, d_model))

    def forward(self, X):
        head_outputs = []
        for head in self.heads:
            out, _ = head.forward(X)
            head_outputs.append(out)
        concatenated = np.concatenate(head_outputs, axis=-1)
        return concatenated @ self.Wo, [head.forward(X)[1] for head in self.heads]
```

完整代码见 `code/main.py`——纯 NumPy，可立即运行。

---

## 4. 工业工具

### PyTorch MultiheadAttention

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
X_torch = torch.randn(1, 6, 8)
output, attn_weights = mha(X_torch, X_torch, X_torch)
# output: (1, 6, 8), attn_weights: (2, 6, 6)
```

关键差异：多头注意力并行运行多个注意力函数，每个有独立的 Q/K/V 投影。这让模型同时关注不同类型的关系。

---

## 5. 常见错误

### 错误 1：忘记 √d_k 缩放

**现象：** 训练初期 loss 不降，梯度为 NaN。

**原因：** dk=64 时点积可达几十。Softmax 在大值上饱和——梯度趋近于 0。除以 √64=8 将值拉回合理范围。

### 错误 2：混淆 Q/K/V 投影矩阵

**现象：** 注意力权重几乎均匀——模型没有学到任何关系。

**原因：** Wq 和 Wk 必须是不同的可学习参数——如果用同一个矩阵，Q=K 导致自注意力永远给自身最高分。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 缩放点积注意力 | "注意力公式" | softmax(QK^T/√dk)·V——缩放防止高维 softmax 饱和 |
| Q/K/V | "三个向量" | Q=查询（找什么），K=键（含什么），V=值（提供什么） |
| 多头注意力 | "并行注意力" | 运行多个注意力函数并拼接结果，捕获不同类型的关系 |
| 因果掩码 | "解码器掩码" | 将未来位置的权重设为 -∞，确保当前位置只看过去 |

---

## 📚 小结

自注意力 = 数据库查询的软版本：每个词元问"谁对我重要？"并学习答案。核心公式：Attention(Q,K,V) = softmax(QK^T/√dk)·V。√dk 缩放防止 softmax 在高维时饱和。多头注意力并行运行多个注意力函数，捕获不同类型的关系。因果掩码将双向注意力转为解码器式的自回归注意力。

---

## ✏️ 练习

1. 修改 `scaled_dot_product_attention` 接受可选掩码矩阵——在 softmax 前将某些位置设为 -∞。这是因果/解码器掩码的工作方式。
2. 从零实现多头注意力：将 Q/K/V 拆分为 `n_heads` 个块，每块独立注意力，拼接后通过 Wo 投影。
3. 对两个不同长度相同的句子运行同一个 SelfAttention 实例，对比注意力模式——什么变了？什么没变？

---

## 📖 参考资料

1. [论文] Vaswani et al. "Attention Is All You Need". NeurIPS, 2017. https://arxiv.org/abs/1706.03762
2. [博客] Jay Alammar. "The Illustrated Transformer". https://jalammar.github.io/illustrated-transformer/
3. [代码] Harvard NLP. "The Annotated Transformer". https://nlp.seas.harvard.edu/annotated-transformer/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
