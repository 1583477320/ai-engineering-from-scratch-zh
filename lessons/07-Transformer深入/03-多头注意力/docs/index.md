# 多头注意力

> 一个注意力头看到一种关系。多个头看到多种关系。拼接它们，你得到一个能同时理解语法、语义和位置的表示。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 07 · 02（从零实现自注意力）
**时间：** ~75 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解多头注意力的动机——为什么单头无法同时捕获不同类型的词元关系
- [ ] 实现多头注意力层——将 Q/K/V 拆分为 n_heads 个子空间，独立注意力，拼接后投影
- [ ] 解释输出投影矩阵 Wo 的作用——合并后的拼接向量需要重新映射回原始维度

---

## 1. 问题

单头注意力只能学到一种"关系模式"。但自然语言中，"cat"和"sat"之间有语法关系（主语-谓语），"cat"和"dog"之间有语义关系（都是宠物），"cat"和"mat"之间有位置关系（on the mat）。一个注意力头不可能同时学会这三种关系——它被迫在语法、语义、位置之间做出妥协。

**多头注意力的解决方案：** 将输入投影到 n_heads 个独立的子空间，每个子空间独立计算注意力。第 1 个头可能关注语法关系，第 2 个头关注语义关系，第 3 个头关注位置关系。拼接所有头的输出后，通过一个可学习的投影矩阵 Wo 将它们合并回原始维度。

---

## 2. 概念

### 2.1 多头注意力架构

```
输入 X: (batch, seq_len, d_model)
        ↓
    ┌───┴───┐
    │ Q=X·Wq│  (d_model → dk)
    │ K=X·Wk│  (d_model → dk)
    │ V=X·Wv│  (d_model → dv)
    └───┬───┘
        ↓
┌─────────────┐
│ 头 1: 注意力(Q1, K1, V1) → 输出1
│ 头 2: 注意力(Q2, K2, V2) → 输出2
│ ...            ...
│ 头 n: 注意力(Qn, Kn, Vn) → 输出n
└─────────────┘
        ↓ 拼接
  输出 = concat(输出1, 输出2, ..., 输出n)
        ↓
  最终输出 = 输出 @ Wo
```

**关键公式：**

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$$

其中 $\text{head}_i = \text{Attention}(QW_i^K, KW_i^K, VW_i^V)$

### 2.2 为什么要多个头

| 头的角色 | 捕获的关系 |
|---|---|
| 头 1 | 语法依赖——"cat" → "sat"（主语→谓语） |
| 头 2 | 语义相似性——"cat" ↔ "dog"（都是宠物） |
| 头 3 | 位置邻近——"cat" ↔ "on"（位置修饰） |
| 头 4 | 长距离依赖——"the" ↔ "mat"（首尾词） |

每个头有自己的 Q/K/V 投影——它们学到完全不同的"关系视角"。

### 2.3 参数量计算

```python
# 输入维度 d_model = 512
# 头数 n_heads = 8
# 每头维度 dk = dv = d_model / n_heads = 64

# 每头参数：Wq + Wk + Wv = 3 × (512 × 64) = 98,304
# 八个头总计：8 × 98,304 = 786,432

# 输出投影 Wo：(8 × 64) × 512 = 262,144

# 多头注意力总参数：786,432 + 262,144 = 1,048,576（约 1M）
```

---

## 3. 从零实现

### Step 1：拆分为多头

```python
import numpy as np

class MultiHeadSelfAttention:
    def __init__(self, d_model, n_heads, seed=42):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dk = self.dv = d_model // n_heads
        
        # 为每个头创建独立的 Q/K/V 投影
        self.heads = []
        for i in range(n_heads):
            rng = np.random.default_rng(seed + i)
            scale = np.sqrt(2.0 / (d_model + self.dk))
            Wq = rng.normal(0, scale, (d_model, self.dk))
            Wk = rng.normal(0, scale, (d_model, self.dk))
            Wv = rng.normal(0, scale, (d_model, self.dv))
            self.heads.append((Wq, Wk, Wv))
        
        # 输出投影矩阵
        rng = np.random.default_rng(seed + n_heads)
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.Wo = rng.normal(0, scale, (n_heads * self.dv, d_model))
```

### Step 2：并行注意力计算

```python
def forward(self, X):
    head_outputs = []
    for Wq, Wk, Wv in self.heads:
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        # 缩放点积注意力
        scores = Q @ K.T / np.sqrt(self.dk)
        weights = softmax(scores)
        output = weights @ V
        head_outputs.append(output)
    
    # 拼接所有头的输出
    concatenated = np.concatenate(head_outputs, axis=-1)
    # 输出投影
    return concatenated @ self.Wo
```

### Step 3：可视化——每个头学到了什么

```python
# 假设 4 个头的注意力矩阵
head_1_weights = np.array([[0.8, 0.1, 0.05, 0.05],   # cat→cat（自我关注）
                            [0.3, 0.5, 0.1, 0.1],     # cat→sat（语法关系）
                            [0.1, 0.6, 0.2, 0.1],     # sat→cat（语法关系）
                            [0.1, 0.1, 0.3, 0.5]])     # mat→on（位置关系）

head_2_weights = np.array([[0.2, 0.5, 0.2, 0.1],   # cat→dog（语义相似）
                            [0.5, 0.2, 0.2, 0.1],   # sat→cat（语义相似）
                            [0.2, 0.3, 0.2, 0.3],   # sat→mat（语义相似）
                            [0.1, 0.3, 0.3, 0.3]])   # mat→cat（语义相似）

# 头 1 学到了语法关系，头 2 学到了语义关系
```

### Step 4：PyTorch 实现

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
X_torch = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
output, attn_weights = mha(X_torch, X_torch, X_torch)
print(f"输出: {output.shape}")       # (2, 10, 512)
print(f"注意力权重: {attn_weights.shape}")  # (2, 8, 10, 10)
```

---

## 4. 常见错误

### 错误 1：所有头共享同一个 Wq/Wk/Wv

**现象：** 多头注意力的输出与单头没有区别。

**原因：** 如果所有头使用相同的 Q/K/V 投影——它们计算的是同一个注意力——拼接后没有新增信息。每个头必须有独立的参数。

### 错误 2：忘记输出投影 Wo

**现象：** 拼接后的维度是 n_heads × dv，而下一层期望 d_model。

**原因：** 拼接多个头的输出后，维度 = n_heads × dv = d_model（因为 dk = dv = d_model/n_heads）。但直接使用拼接向量没有意义——Wo 将拼接向量映射到一个新的 d_model 空间，让模型学习如何组合各头的信息。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 多头注意力 | "多个注意力函数并行" | 拆分为 n_heads 个子空间，每个子空间独立注意力，拼接后投影 |
| 头维度 dk | "每个头看到的维度" | d_model / n_heads。每个头的 Q/K 在 dk 维空间操作 |
| 输出投影 Wo | "拼接后的融合" | 将 n_heads × dv 维拼接向量映射回 d_model 维 |

---

## 📚 小结

多头注意力让模型同时捕获不同类型的词元关系——语法、语义、位置、长距离依赖。每个头有独立的 Q/K/V 投影，输出拼接后通过 Wo 投影回原始维度。8 头 × 64 维 = 1M 参数，比单头 512 维的 260K 参数多——但信息量大得多。

---

## ✏️ 练习

1. 修改多头注意力，让第 1 个头只关注位置邻近的词元（用掩码限制注意力窗口）。与全注意力对比——哪个捕获语法关系更好？
2. 实现一个 `AttentionHeadAnalyzer` 类——对每个头计算：平均注意力范围、最大注意力跨度、注意力熵（分布的分散程度）。

---

## 📖 参考资料

1. [论文] Vaswani et al. "Attention Is All You Need". NeurIPS, 2017. https://arxiv.org/abs/1706.03762
2. [博客] Jay Alammar. "The Illustrated Transformer". https://jalammar.github.io/illustrated-transformer/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
