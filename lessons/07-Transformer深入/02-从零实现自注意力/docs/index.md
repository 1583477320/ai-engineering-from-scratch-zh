# 从零实现自注意力

> 注意力是一个查询表，每个词都在问"谁对我重要？"——并学习答案。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 3（深度学习核心）、阶段 05 · 09（序列到序列）
**时间：** ~90 分钟
**所处阶段：** Tier 2
**关联课程：** 第 7 阶段 · 03（多头注意力）— 理解注意力机制如何扩展

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

### 4.1 PyTorch MultiheadAttention

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
X_torch = torch.randn(1, 6, 8)
output, attn_weights = mha(X_torch, X_torch, X_torch)
# output: (1, 6, 8), attn_weights: (2, 6, 6)
```

关键差异：多头注意力并行运行多个注意力函数，每个有独立的 Q/K/V 投影。这让模型同时关注不同类型的关系。

### 4.2 HuggingFace Transformers

```python
from transformers import AutoModel, AutoTokenizer

# 加载预训练的 BERT 模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese", output_attentions=True)

# 输入文本
text = "注意力机制是 Transformer 的核心"
inputs = tokenizer(text, return_tensors="pt")

# 前向传播，获取注意力权重
outputs = model(**inputs)
attentions = outputs.attentions  # 每层的注意力权重

print(f"层数: {len(attentions)}")           # 12 层
print(f"头数: {attentions[0].shape[1]}")     # 12 个头
print(f"注意力权重形状: {attentions[0].shape}")  # (1, 12, seq_len, seq_len)
```

### 4.3 性能对比

| 实现方式 | 速度 | 内存 | 适用场景 |
|---|---|---|---|
| 我们的 NumPy 版 | 慢 | 低 | 学习理解 |
| PyTorch MHA | 快 | 中 | 训练 / 研究 |
| FlashAttention | 极快 | 低 | 生产环境 |
| vLLM PagedAttention | 极快 | 极低 | 大语言模型推理 |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

GPT-4、Claude、Llama 3 等大语言模型的核心架构都是 Decoder-only Transformer。它们的每一层都包含自注意力机制——这是模型理解上下文、建立词元间关系的关键。

具体来说，GPT-4 使用了 128 个注意力头，每个头的维度为 128。这意味着模型在每个位置上同时关注 128 种不同类型的关系——语法关系、语义关系、位置关系、长距离依赖等等。你在第 3 步实现的多头注意力，正是这个机制的简化版本。

### 5.2 LLM 时代什么变了？

**规模变了。** 你在本课实现的自注意力处理 6 个词元，GPT-4 处理 128K 个词元。注意力矩阵从 (6, 6) 变成 (128K, 128K)——这是 4600 亿个浮点数。

**优化变了。** 朴素的 O(n²) 注意力在长上下文场景下不可接受。工业界使用 FlashAttention（IO 感知的分块计算）将内存占用从 O(n²) 降到 O(n)，速度提升 2-4 倍。vLLM 的 PagedAttention 进一步优化了 KV 缓存的内存管理。

**位置编码变了。** 你在第 3 步没有实现位置编码——这是自注意力的缺陷（它是置换不变的）。现代大语言模型使用 RoPE（旋转位置嵌入）来编码位置信息，这是第 4 课的内容。

### 5.3 什么没变？

**核心公式没变。** 你在第 2 步实现的 `softmax(QK^T/√dk)V` 从 2017 年至今没有改变。GPT-4、Claude、Llama 3 的注意力层本质上都在执行这个公式。

**Q/K/V 分离没变。** 查询、键、值的分离是注意力机制的核心设计决策。你在第 5 步看到的 Wq/Wk/Wv 三个独立投影，在所有现代大语言模型中都保留着。

**缩放因子没变。** √d_k 缩放仍然是防止梯度消失的标准做法。没有哪个大语言模型跳过了这一步。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你在 ChatGPT 中输入一段话，模型生成回复时，它在每个生成步骤都在执行自注意力——每个新生成的词元都要与之前的所有词元计算注意力权重。

如果你输入 100 个词元，模型生成 50 个词元，那么注意力机制总共执行了 100 + 101 + 102 + ... + 149 = 6225 次注意力计算（每次一个位置关注所有之前的位置）。这就是为什么长对话比短对话慢——注意力计算量随序列长度平方增长。

你在第 5 步实现的因果掩码，正是保证"生成时不能偷看未来"的关键机制。没有它，模型就能在训练时作弊——直接看到答案。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 学习 / 实验 | PyTorch `nn.MultiheadAttention` | 开箱即用 |
| 训练（< 2K 上下文）| PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`) | PyTorch 2.0+ 内置，自动选取最优实现 |
| 训练（长上下文）| FlashAttention-2 | IO 感知，内存 O(n)，速度 2-4x |
| LLM 推理 | vLLM PagedAttention | KV 缓存分页管理，吞吐量提升 10-20x |

### 6.2 中文场景特别建议

- 中文分词器训练时，确保训练语料中中文占比 ≥ 30%，否则中文会被过度切分
- 使用 `bert-base-chinese` 的分词器时注意：它是字级别的，不产生子词
- 中文 + 代码混合场景，优先使用 Llama 3 或 GPT-4 的分词器（128K+ 词表，对多语言更友好）

### 6.3 踩坑经验

- 训练分词器时未设置 `byte_fallback=True`，导致部署时遇到生僻字直接报错
- `padding_side="left"` 还是 `"right"`？自回归模型用 left，编码器模型用 right
- 不要用训练集的平均长度作为 `max_length`——留 20% 余量，否则截断率太高
- 多头注意力的头数必须能整除 d_model，否则会报错
- 因果掩码方向搞反是常见 bug——上三角遮挡未来，下三角保留过去

---

## 7. 常见错误

### 错误 1：忘记 √d_k 缩放

**现象：** 训练初期 loss 不降，梯度为 NaN。

**原因：** dk=64 时点积可达几十。Softmax 在大值上饱和——梯度趋近于 0。除以 √64=8 将值拉回合理范围。

**修复：**
```python
# ❌ 错误写法
scores = Q @ K.T

# ✓ 正确写法
scores = Q @ K.T / np.sqrt(Q.shape[-1])
```

### 错误 2：混淆 Q/K/V 投影矩阵

**现象：** 注意力权重几乎均匀——模型没有学到任何关系。

**原因：** Wq 和 Wk 必须是不同的可学习参数——如果用同一个矩阵，Q=K 导致自注意力永远给自身最高分。

**修复：**
```python
# ❌ 错误写法：Wq 和 Wk 共享权重
self.Wq = rng.normal(0, scale, (d_model, dk))
self.Wk = self.Wq  # Q=K，自注意力永远给自身最高分

# ✓ 正确写法：三个独立的投影矩阵
self.Wq = rng.normal(0, scale, (d_model, dk))
self.Wk = rng.normal(0, scale, (d_model, dk))
self.Wv = rng.normal(0, scale, (d_model, dv))
```

### 错误 3：因果掩码方向搞反

**现象：** 模型生成时每个位置都能"看到未来"——推理 loss 异常低，但生成结果完全不对。

**原因：** 掩码矩阵构建时上三角和下三角搞反了。解码器中，位置 i 应该只能看到 0~i（下三角），不能看到 i+1~n（上三角）。

**修复：**
```python
# ❌ 错误写法：遮住过去，露出未来
mask = np.tril(np.ones((n, n))) * -1e9

# ✓ 正确写法：上三角为 -inf，遮住未来
mask = np.triu(np.ones((n, n)), k=1) * -1e9
```

### 错误 4：多头注意力头数不能整除 d_model

**现象：** 运行时报错 `AssertionError`。

**原因：** 每个头的维度是 `d_model // n_heads`，必须是整数。如果 d_model=512，n_heads=7，512/7=73.14 不是整数。

**修复：**
```python
# ❌ 错误写法：头数不能整除
n_heads = 7  # 512 / 7 = 73.14（不是整数）

# ✓ 正确写法：选择能整除的头数
n_heads = 8  # 512 / 8 = 64（整数）
assert d_model % n_heads == 0, f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"
```

### 错误 5：注意力权重未归一化

**现象：** 注意力权重之和不为 1，输出值异常大。

**原因：** 忘记应用 Softmax，或者 Softmax 的 axis 参数错误。

**修复：**
```python
# ❌ 错误写法：未归一化
weights = scores  # scores 的值域是 (-∞, +∞)

# ✓ 正确写法：Softmax 归一化
weights = softmax(scores)  # 每行之和为 1
```

---

## 8. 面试考点

### Q1：Self-Attention 中 Q、K、V 分别代表什么？为什么不用同一个？（难度：⭐⭐）

**参考答案：**
Q（查询）代表当前词元"想找什么"，K（键）代表每个词元"含有什么"，V（值）代表每个词元"提供什么信息"。

分离 Q、K、V 的关键原因是**非对称性**——一个词元想找的信息（Q）和它自己能提供的信息（K、V）是不同的。例如"苹果"这个词元，当它作为主语时（Q），它想找动词；当它作为宾语时（V），它被动词寻找。同样的词元在不同上下文中扮演不同角色，用同一组权重无法同时表达。

### Q2：为什么 Softmax 之前要除以 √d_k？（难度：⭐⭐）

**参考答案：**
当 d_k 较大时，点积 QK^T 的值量级增大，推入 Softmax 的饱和区（梯度接近零）。除以 √d_k 将方差控制为 1，使 Softmax 保持在梯度有效的区域。如果不做缩放，模型训练初期梯度消失，无法学习。

### Q3：手写 Multi-Head Attention 的前向传播（难度：⭐⭐⭐）

**参考答案：**
```python
def multi_head_attention(Q, K, V, Wq, Wk, Wv, Wo, n_heads):
    # 1. 线性投影
    Q_proj = Q @ Wq
    K_proj = K @ Wk
    V_proj = V @ Wv

    # 2. 拆分为多头
    seq_len, d_model = Q.shape
    d_k = d_model // n_heads
    Q_heads = Q_proj.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K_heads = K_proj.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V_heads = V_proj.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    # 3. 每个头独立计算注意力
    scores = Q_heads @ K_heads.transpose(0, 2, 1) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    head_outputs = weights @ V_heads

    # 4. 拼接所有头的结果
    concatenated = head_outputs.transpose(1, 0, 2).reshape(seq_len, d_model)

    # 5. 最终投影
    output = concatenated @ Wo
    return output
```

### Q4：自注意力的时间复杂度是多少？有什么优化方法？（难度：⭐⭐⭐）

**参考答案：**
自注意力的时间复杂度是 O(n²·d)，其中 n 是序列长度，d 是维度。这是因为每个位置都要与所有其他位置计算注意力分数。

优化方法包括：
- **FlashAttention**：IO 感知的分块计算，减少 GPU 内存访问，速度提升 2-4 倍
- **稀疏注意力**：只计算部分位置的注意力（如 Local Attention、Global Attention）
- **线性注意力**：将复杂度从 O(n²) 降到 O(n)，但牺牲了部分表达能力

### Q5：如何理解注意力权重矩阵的对称性？（难度：⭐⭐）

**参考答案：**
在不使用因果掩码的双向注意力中，注意力权重矩阵是对称的（W[i][j] ≈ W[j][i]），因为 Q 和 K 来自同一组输入。但使用因果掩码后，上三角被遮挡，矩阵不再对称——位置 i 只能看到 0~i，位置 j 只能看到 0~j。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 缩放点积注意力 | "注意力公式" | softmax(QK^T/√dk)·V——缩放防止高维 softmax 饱和 |
| Q/K/V | "三个向量" | Q=查询（找什么），K=键（含什么），V=值（提供什么） |
| 多头注意力 | "并行注意力" | 运行多个注意力函数并拼接结果，捕获不同类型的关系 |
| 因果掩码 | "解码器掩码" | 将未来位置的权重设为 -∞，确保当前位置只看过去 |
| 自注意力 | "内部注意力" | 序列中的每个位置关注所有其他位置——与交叉注意力（关注另一个序列）相对 |
| 注意力权重 | "模型在关注哪里" | Softmax 归一化后的分数——每个位置对其他所有位置的概率分布，所有值之和为 1 |
| 缩放因子 | "除以根号 dk" | √dk 防止点积过大导致 Softmax 梯度消失——这是训练稳定性的关键 |
| 输出投影 | "最终的线性层" | 将多头拼接后的结果投影回 d_model 维度——让模型学习如何组合不同头的信息 |

---

## 📚 小结

自注意力 = 数据库查询的软版本：每个词元问"谁对我重要？"并学习答案。核心公式：Attention(Q,K,V) = softmax(QK^T/√dk)·V。√dk 缩放防止 softmax 在高维时饱和。多头注意力并行运行多个注意力函数，捕获不同类型的关系。因果掩码将双向注意力转为解码器式的自回归注意力。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释 Self-Attention 中 Q、K、V 三个矩阵的作用。为什么不能用同一个矩阵？写 200 字以内的说明，让一个没有 ML 背景的程序员也能听懂。

2. **【实现】** 修改 `scaled_dot_product_attention` 接受可选掩码矩阵——在 softmax 前将某些位置设为 -∞。这是因果/解码器掩码的工作方式。

3. **【实现】** 从零实现多头注意力：将 Q/K/V 拆分为 `n_heads` 个块，每块独立注意力，拼接后通过 Wo 投影。

4. **【实验】** 对两个不同长度相同的句子运行同一个 SelfAttention 实例，对比注意力模式——什么变了？什么没变？

5. **【思考】** FlashAttention 是目前工业界的标配。阅读 FlashAttention 论文的摘要，用你自己的话解释它为什么比标准注意力快。（提示：思考 GPU 的内存层次结构）

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 自注意力实现 | `code/main.py` | 从零实现的自注意力、多头注意力、因果掩码，纯 NumPy |
| 注意力可视化工具 | `outputs/attention-visualizer.md` | 可视化注意力权重矩阵，帮助理解模型关注模式 |

---

## 📖 参考资料

1. [论文] Vaswani et al. "Attention Is All You Need". NeurIPS, 2017. https://arxiv.org/abs/1706.03762
2. [博客] Jay Alammar. "The Illustrated Transformer". https://jalammar.github.io/illustrated-transformer/
3. [代码] Harvard NLP. "The Annotated Transformer". https://nlp.seas.harvard.edu/annotated-transformer/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
