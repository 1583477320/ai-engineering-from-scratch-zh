# 完整 Transformer

> 从词元嵌入到注意力到前馈网络到残差连接——组装一个可以训练的 Transformer 块。这是阶段 07 的毕业设计。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 07 · 02-04
**时间：** ~120 分钟
**所处阶段：** Tier 2
**关联课程：** 第 7 阶段 · 06（BERT 掩码语言建模）— 理解 Transformer 编码器如何用于预训练

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零组装一个完整的 Transformer 块——嵌入 + 位置编码 + 多头注意力 + 前馈网络 + 残差连接 + 层归一化
- [ ] 构建编码器和解码器栈——包括交叉注意力和因果掩码
- [ ] 在简单分类任务上验证模型可以训练——理解从理论到实现的关键细节

---

## 1. 问题

前面三课分别构建了自注意力、多头注意力和位置编码。现在需要把它们组装成一个可以训练的完整架构——这是从"理解组件"到"构建系统"的关键跃迁。

---

## 2. 概念——Transformer 块的解剖

```
输入嵌入 + 位置编码
        ↓
┌─────────────────────┐
│ 多头自注意力         │
│ 残差连接 + 层归一化  │
│ 前馈网络 (FFN)       │
│ 残差连接 + 层归一化  │
└─────────────────────┘
        ↓
N 层堆叠
        ↓
输出
```

**每个组件的作用：**

| 组件 | 作用 |
|---|---|
| 嵌入 + 位置编码 | 将词元 ID 转换为带位置信息的向量 |
| 多头注意力 | 让每个词元关注其他词元——学习关系 |
| 前馈网络 (FFN) | 两层线性层 + GELU — 对每个位置独立处理 |
| 残差连接 | `x + attention(x)` — 梯度直通，防止梯度消失 |
| 层归一化 | 稳定训练——归一化每个层的输出 |

---

## 3. 从零实现

```python
import numpy as np

class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, seed=42):
        # 多头注意力
        self.mha = MultiHeadSelfAttention(d_model, n_heads, seed)
        # 前馈网络：d_model → d_ff → d_model
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / (d_model + d_ff)), (d_model, d_ff))
        self.W2 = rng.normal(0, np.sqrt(2.0 / (d_ff + d_model)), (d_ff, d_model))
        self.d_model = d_model

    def forward(self, x):
        # 自注意力 + 残差连接
        attn_out, _ = self.mha.forward(x)
        x = self.layernorm(x + attn_out)
        # 前馈网络 + 残差连接
        ffn_out = np.maximum(0, x @ self.W1) @ self.W2  # GELU 近似
        x = self.layernorm(x + ffn_out)
        return x

    def layernorm(self, x):
        """层归一化：减均值，除标准差。"""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-8
        return (x - mean) / std
```

### 从零到可训练的清单

| 组件 | 状态 | 验证 |
|---|---|---|
| 词元嵌入 | ✓ 阶段 05 · 03 | 形状 (vocab, d_model) |
| 位置编码 | ✓ 阶段 07 · 04 | 可以加到嵌入上 |
| 多头自注意力 | ✓ 阶段 07 · 03 | 输出形状 (batch, seq, d_model) |
| FFN | ✓ 本课 | GELU 激活 |
| 残差连接 | ✓ 本课 | x + attn(x) |
| 层归一化 | ✓ 本课 | 稳定训练 |

---

## 4. 工具——PyTorch 实现

```python
import torch
import torch.nn as nn

class TransformerBlockPytorch(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 自注意力 + 残差 + 层归一化
        attn_out, _ = self.mha(x, x, x)
        x = self.ln1(x + attn_out)
        # FFN + 残差 + 层归一化
        ffn_out = self.ffn(x)
        return self.ln2(x + ffn_out)

# 一个 6 层、512 维、8 头的 Transformer 编码器
encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True),
    num_layers=6
)
x = torch.randn(2, 20, 512)  # (batch, seq_len, d_model)
out = encoder(x)
print(f"输入: {x.shape}, 输出: {out.shape}")
# 输入: torch.Size([2, 20, 512]), 输出: torch.Size([2, 20, 512])
```

完整代码见 `code/main.py`——纯 NumPy，可立即运行。

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

GPT-4、Claude、Llama 3 等大语言模型的核心架构都是 Decoder-only Transformer。它们的每一层都包含你在本课实现的所有组件——多头注意力、前馈网络、残差连接、层归一化。

具体来说，GPT-4 使用了 128 层 Transformer 块，每层包含 128 个注意力头，每个头的维度为 128。你在本课实现的 2 层 Transformer 块，正是这个机制的简化版本。

### 5.2 LLM 时代什么变了？

**规模变了。** 你在本课实现的 Transformer 块处理 20 个词元，GPT-4 处理 128K 个词元。层数从 2 层变成 128 层——这是 64 倍的深度。

**优化变了。** 朴素的 O(n²) 注意力在长上下文场景下不可接受。工业界使用 FlashAttention（IO 感知的分块计算）将内存占用从 O(n²) 降到 O(n)，速度提升 2-4 倍。vLLM 的 PagedAttention 进一步优化了 KV 缓存的内存管理。

**激活函数变了。** 你在本课使用 GELU 近似，现代大语言模型可能使用 SwiGLU 或其他变体。但核心思想——两层线性层 + 非线性激活——没有改变。

### 5.3 什么没变？

**核心架构没变。** 你在本课实现的 Transformer 块结构（注意力 + FFN + 残差 + 层归一化）从 2017 年至今没有改变。GPT-4、Claude、Llama 3 的每一层都在执行这个结构。

**残差连接没变。** 残差连接是训练深层网络的关键。没有它，梯度会在深层网络中消失。所有现代大语言模型都保留了残差连接。

**层归一化没变。** 层归一化是稳定训练的关键。所有现代大语言模型都使用层归一化（通常是 Pre-LN 变体）。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你在 ChatGPT 中输入一段话，模型生成回复时，它在每个生成步骤都在执行多层 Transformer 块——每层包含多头注意力、前馈网络、残差连接、层归一化。

如果你输入 100 个词元，模型生成 50 个词元，那么 Transformer 块总共执行了 150 × 128 = 19200 次前向传播（150 个位置 × 128 层）。这就是为什么大语言模型需要强大的 GPU——每一步生成都需要大量计算。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 学习 / 实验 | PyTorch `TransformerEncoderLayer` | 开箱即用 |
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
- Pre-LN 比 Post-LN 训练更稳定——现代 Transformer 默认使用 Pre-LN
- 因果掩码方向搞反是常见 bug——上三角遮挡未来，下三角保留过去

---

## 7. 常见错误

### 错误 1：层归一化放在注意力之前

**现象：** 训练不稳定，loss 震荡。

**原因：** Pre-LN（先归一化再注意力）比 Post-LN（先注意力再归一化）在深层网络上更稳定。现代 Transformer 默认 Pre-LN。

**修复：**
```python
# ❌ 错误写法：Post-LN（不稳定）
x = layernorm(x + attention(x))

# ✓ 正确写法：Pre-LN（稳定）
x = x + attention(layernorm(x))
```

### 错误 2：因果掩码缺失

**现象：** 解码器在训练时可以看到未来位置。

**原因：** 解码器必须是自回归的——位置 i 只能看位置 0..i。没有掩码的解码器训练会在推理时产生不一致的结果。

**修复：**
```python
# ❌ 错误写法：没有掩码
output = attention(Q, K, V)  # 所有位置都能看到所有位置

# ✓ 正确写法：因果掩码
mask = np.triu(np.ones((n, n), dtype=bool), k=1)
output = attention(Q, K, V, mask=mask)  # 位置 i 只能看到 0..i
```

### 错误 3：前馈网络维度不匹配

**现象：** 运行时报错 `ValueError: shapes not aligned`。

**原因：** FFN 的第一层将 d_model 映射到 d_ff，第二层将 d_ff 映射回 d_model。如果维度不匹配，无法进行矩阵乘法。

**修复：**
```python
# ❌ 错误写法：维度不匹配
W1 = np.random.randn(d_model, d_model)  # 应该是 d_model → d_ff
W2 = np.random.randn(d_ff, d_model)     # 应该是 d_ff → d_model

# ✓ 正确写法：维度匹配
W1 = np.random.randn(d_model, d_ff)  # d_model → d_ff
W2 = np.random.randn(d_ff, d_model)  # d_ff → d_model
```

### 错误 4：残差连接维度不匹配

**现象：** 运行时报错 `ValueError: operands could not be broadcast together`。

**原因：** 残差连接要求输入和输出维度一致。如果注意力输出的维度与输入不同，无法相加。

**修复：**
```python
# ❌ 错误写法：维度不匹配
x = np.random.randn(10, 512)  # 512 维
attn_out = np.random.randn(10, 256)  # 256 维
x = x + attn_out  # 报错

# ✓ 正确写法：维度一致
x = np.random.randn(10, 512)  # 512 维
attn_out = np.random.randn(10, 512)  # 512 维
x = x + attn_out  # 正常
```

### 错误 5：忘记 GELU 激活

**现象：** FFN 退化为线性层，表达能力不足。

**原因：** FFN 需要非线性激活函数（如 GELU）来增加表达能力。没有激活函数，两层线性层等价于一层线性层。

**修复：**
```python
# ❌ 错误写法：没有激活函数
ffn_out = x @ W1 @ W2  # 等价于 x @ (W1 @ W2)，退化为线性层

# ✓ 正确写法：GELU 激活
linear1 = x @ W1
gelu = 0.5 * linear1 * (1 + np.tanh(np.sqrt(2 / np.pi) * (linear1 + 0.044715 * linear1**3)))
ffn_out = gelu @ W2
```

---

## 8. 面试考点

### Q1：Transformer 块由哪些组件组成？（难度：⭐⭐）

**参考答案：**
Transformer 块由以下组件组成：
1. 多头自注意力
2. 前馈网络（FFN）
3. 残差连接
4. 层归一化

每个组件的作用：
- 多头注意力：学习词元间关系
- FFN：对每个位置独立处理，增加非线性表达能力
- 残差连接：梯度直通，防止梯度消失
- 层归一化：稳定训练

### Q2：为什么需要残差连接？（难度：⭐⭐）

**参考答案：**
残差连接解决深层网络的梯度消失问题。在没有残差连接的深层网络中，梯度需要经过很多层才能到达浅层，每经过一层都会衰减。残差连接提供了"梯度直通"的路径——梯度可以直接从输出层传到输入层，不需要经过每一层。

数学上，残差连接将 $y = f(x)$ 变为 $y = x + f(x)$。梯度计算时，$\frac{\partial y}{\partial x} = 1 + \frac{\partial f}{\partial x}$——即使 $\frac{\partial f}{\partial x}$ 很小，梯度也不会消失。

### Q3：Pre-LN 和 Post-LN 有什么区别？（难度：⭐⭐⭐）

**参考答案：**
Pre-LN 和 Post-LN 是层归一化的两种放置方式：
- **Post-LN**：$x = \text{LN}(x + \text{sublayer}(x))$ — 原始 Transformer 使用
- **Pre-LN**：$x = x + \text{sublayer}(\text{LN}(x))$ — 现代 Transformer 默认使用

Pre-LN 更稳定的原因：
1. 残差连接直接传递原始信号，不会被归一化破坏
2. 梯度可以通过残差连接直接传播，不需要经过归一化层
3. 训练深层网络时不容易出现梯度爆炸或消失

### Q4：FFN 的维度为什么要先升后降？（难度：⭐⭐⭐）

**参考答案：**
FFN 的第一层将 d_model 映射到 d_ff（通常 d_ff = 4 × d_model），第二层映射回 d_model。这种"升维-降维"的设计有两个目的：
1. **增加表达能力**：高维空间可以表达更复杂的非线性变换
2. **参数效率**：如果两层都是 d_model × d_model，参数量是 $2 \times d_{model}^2$；升维后参数量是 $d_{model} \times d_{ff} + d_{ff} \times d_{model} = 2 \times d_{model} \times d_{ff}$，但表达能力更强

### Q5：如何计算 Transformer 的参数量？（难度：⭐⭐⭐）

**参考答案：**
Transformer 参数量计算：
1. **词元嵌入**：vocab_size × d_model
2. **位置编码**：正弦编码为 0，可学习编码为 max_len × d_model
3. **每层多头注意力**：4 × d_model²（Wq, Wk, Wv, Wo）
4. **每层 FFN**：2 × d_model × d_ff（W1, W2）
5. **每层层归一化**：4 × d_model（gamma 和 beta，各 2 个）
6. **总计**：词元嵌入 + 位置编码 + n_layers × (每层参数)

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 前馈网络 | "中间的小网络" | 两层线性层 + GELU，对每个位置独立处理 |
| 残差连接 | "快捷连接" | x + attention(x)——梯度直通，防止深层网络梯度消失 |
| 层归一化 | "每层归一化" | 减均值除标准差，稳定训练 |
| 因果掩码 | "解码器掩码" | 位置 i 只能看位置 0..i，防止"看到未来" |
| Pre-LN | "先归一化再注意力" | 现代 Transformer 默认选择，训练更稳定 |
| Post-LN | "先注意力再归一化" | 原始 Transformer 使用，深层网络不稳定 |
| GELU | "激活函数" | 高斯误差线性单元，比 ReLU 更平滑 |
| 残差流 | "梯度的高速公路" | 残差连接提供的梯度直通路径 |

---

## 📚 小结

完整 Transformer 块 = 多头注意力 + FFN + 残差连接 + 层归一化。从零组装到训练，关键细节是：残差让梯度直通、层归一化稳定训练、因果掩码保持自回归性质。PyTorch 的 `TransformerEncoderLayer` 封装了这一切——理解每个组件的作用比记住 API 更重要。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释为什么 Transformer 需要残差连接和层归一化。如果没有它们，训练会有什么问题？写 200 字以内的说明。

2. **【实现】** 从零实现完整的 Transformer 编码器（6 层 × 512 维 × 8 头），在玩具分类任务上验证可以训练。

3. **【实验】** 比较 Pre-LN vs Post-LN：在相同超参下训练，记录训练损失曲线——哪个更稳定？

4. **【实现】** 实现一个完整的 Transformer 解码器（带因果掩码），验证它可以在简单序列生成任务上训练。

5. **【思考】** 阅读 FlashAttention 论文的摘要，用你自己的话解释它为什么比标准注意力快。（提示：思考 GPU 的内存层次结构）

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 完整 Transformer 实现 | `code/main.py` | Transformer 编码器和解码器的完整实现 |
| Transformer 架构指南 | `outputs/transformer-architecture-guide.md` | 组件作用、Pre-LN vs Post-LN、参数量估算 |

---

## 📖 参考资料

1. [论文] Vaswani et al. "Attention Is All You Need". NeurPR, 2017.
2. [代码] Harvard NLP. "The Annotated Transformer". https://nlp.seas.harvard.edu/annotated-transformer/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
