# 多头注意力

> 一个注意力头看到一种关系。多个头看到多种关系。拼接它们，你得到一个能同时理解语法、语义和位置的表示。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 07 · 02（从零实现自注意力）
**时间：** ~75 分钟
**所处阶段：** Tier 2
**关联课程：** 第 7 阶段 · 04（位置编码）— 理解多头注意力如何与位置信息结合

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

完整代码见 `code/main.py`——纯 NumPy，可立即运行。

---

## 4. 工业工具

### 4.1 PyTorch MultiheadAttention

```python
import torch.nn as nn

# 标准多头注意力
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
X_torch = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
output, attn_weights = mha(X_torch, X_torch, X_torch)
print(f"输出: {output.shape}")       # (2, 10, 512)
print(f"注意力权重: {attn_weights.shape}")  # (2, 8, 10, 10)
```

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

GPT-4、Claude、Llama 3 等大语言模型的核心架构都是 Decoder-only Transformer。它们的每一层都包含多头注意力——这是模型理解上下文、建立词元间关系的关键。

具体来说，GPT-4 使用了 128 个注意力头，每个头的维度为 128。这意味着模型在每个位置上同时关注 128 种不同类型的关系——语法关系、语义关系、位置关系、长距离依赖等等。你在本课实现的多头注意力，正是这个机制的简化版本。

### 5.2 LLM 时代什么变了？

**规模变了。** 你在本课实现的多头注意力处理 6 个词元，GPT-4 处理 128K 个词元。注意力矩阵从 (6, 6) 变成 (128K, 128K)——这是 4600 亿个浮点数。

**优化变了。** 朴素的 O(n²) 注意力在长上下文场景下不可接受。工业界使用 FlashAttention（IO 感知的分块计算）将内存占用从 O(n²) 降到 O(n)，速度提升 2-4 倍。vLLM 的 PagedAttention 进一步优化了 KV 缓存的内存管理。

**位置编码变了。** 你在本课没有实现位置编码——这是自注意力的缺陷（它是置换不变的）。现代大语言模型使用 RoPE（旋转位置嵌入）来编码位置信息，这是第 4 课的内容。

### 5.3 什么没变？

**核心公式没变。** 你在第 2 步实现的 `Concat(head_1,...,head_h)Wo` 从 2017 年至今没有改变。GPT-4、Claude、Llama 3 的注意力层本质上都在执行这个公式。

**Q/K/V 分离没变。** 查询、键、值的分离是注意力机制的核心设计决策。你在本课看到的每个头独立的 Q/K/V 投影，在所有现代大语言模型中都保留着。

**缩放因子没变。** √d_k 缩放仍然是防止梯度消失的标准做法。没有哪个大语言模型跳过了这一步。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你在 ChatGPT 中输入一段话，模型生成回复时，它在每个生成步骤都在执行多头注意力——每个新生成的词元都要与之前的所有词元计算注意力权重。

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

### 错误 1：所有头共享同一个 Wq/Wk/Wv

**现象：** 多头注意力的输出与单头没有区别。

**原因：** 如果所有头使用相同的 Q/K/V 投影——它们计算的是同一个注意力——拼接后没有新增信息。每个头必须有独立的参数。

**修复：**
```python
# ❌ 错误写法：所有头共享权重
self.Wq = rng.normal(0, scale, (d_model, dk))
self.Wk = rng.normal(0, scale, (d_model, dk))
self.Wv = rng.normal(0, scale, (d_model, dk))

# ✓ 正确写法：每个头独立的权重
for i in range(n_heads):
    rng = np.random.default_rng(seed + i)
    Wq = rng.normal(0, scale, (d_model, dk))
    Wk = rng.normal(0, scale, (d_model, dk))
    Wv = rng.normal(0, scale, (d_model, dk))
    self.heads.append((Wq, Wk, Wv))
```

### 错误 2：忘记输出投影 Wo

**现象：** 拼接后的维度是 n_heads × dv，而下一层期望 d_model。

**原因：** 拼接多个头的输出后，维度 = n_heads × dv = d_model（因为 dk = dv = d_model/n_heads）。但直接使用拼接向量没有意义——Wo 将拼接向量映射到一个新的 d_model 空间，让模型学习如何组合各头的信息。

**修复：**
```python
# ❌ 错误写法：直接返回拼接结果
return concatenated  # 形状 (n, d_model)，但没有学习如何组合各头信息

# ✓ 正确写法：通过输出投影
return concatenated @ self.Wo  # 形状 (n, d_model)，学习组合各头信息
```

### 错误 3：头数不能整除 d_model

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

### 错误 4：注意力权重未归一化

**现象：** 注意力权重之和不为 1，输出值异常大。

**原因：** 忘记应用 Softmax，或者 Softmax 的 axis 参数错误。

**修复：**
```python
# ❌ 错误写法：未归一化
weights = scores  # scores 的值域是 (-∞, +∞)

# ✓ 正确写法：Softmax 归一化
weights = softmax(scores)  # 每行之和为 1
```

### 错误 5：因果掩码方向搞反

**现象：** 模型生成时每个位置都能"看到未来"——推理 loss 异常低，但生成结果完全不对。

**原因：** 掩码矩阵构建时上三角和下三角搞反了。解码器中，位置 i 应该只能看到 0~i（下三角），不能看到 i+1~n（上三角）。

**修复：**
```python
# ❌ 错误写法：遮住过去，露出未来
mask = np.tril(np.ones((n, n))) * -1e9

# ✓ 正确写法：上三角为 -inf，遮住未来
mask = np.triu(np.ones((n, n)), k=1) * -1e9
```

---

## 8. 面试考点

### Q1：多头注意力为什么比单头更好？（难度：⭐⭐）

**参考答案：**
单头注意力被迫在语法、语义、位置关系之间妥协。多头注意力将输入投影到 n_heads 个独立的子空间，每个子空间独立计算注意力。第 1 个头可能关注语法关系，第 2 个头关注语义关系，第 3 个头关注位置关系。拼接所有头的输出后，通过 Wo 投影回原始维度。

### Q2：输出投影矩阵 Wo 的作用是什么？（难度：⭐⭐）

**参考答案：**
拼接多个头的输出后，维度 = n_heads × dv = d_model。但直接使用拼接向量没有意义——Wo 将拼接向量映射到一个新的 d_model 空间，让模型学习如何组合各头的信息。没有 Wo，每个头的信息是孤立的；有了 Wo，模型可以学习"语法头的信息占 60%，语义头的信息占 40%"这样的组合策略。

### Q3：多头注意力的时间复杂度是多少？（难度：⭐⭐⭐）

**参考答案：**
多头注意力的时间复杂度是 O(n²·d)，其中 n 是序列长度，d 是维度。这是因为每个头都要计算 n×n 的注意力矩阵，共有 n_heads 个头。但由于每个头的维度是 d_model/n_heads，总计算量与单头注意力相同。

### Q4：如何理解注意力头的冗余性？（难度：⭐⭐⭐）

**参考答案：**
研究表明，训练好的 Transformer 中存在大量冗余的注意力头。移除某些头对模型性能影响很小，而移除另一些头会导致性能急剧下降。这说明：
1. 不是所有头都学到有用的信息
2. 某些头可能学到重复的模式
3. 注意力头的剪枝是模型压缩的重要方向

### Q5：FlashAttention 如何优化多头注意力？（难度：⭐⭐⭐）

**参考答案：**
FlashAttention 通过以下方式优化：
1. **分块计算**：将注意力矩阵分块，避免一次性加载整个 n×n 矩阵到 GPU 内存
2. **IO 感知**：优化 GPU 内存访问模式，减少 HBM（高带宽内存）访问次数
3. **重计算**：在反向传播时重新计算注意力权重，而不是存储它们

结果：内存占用从 O(n²) 降到 O(n)，速度提升 2-4 倍。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 多头注意力 | "多个注意力函数并行" | 拆分为 n_heads 个子空间，每个子空间独立注意力，拼接后投影 |
| 头维度 dk | "每个头看到的维度" | d_model / n_heads。每个头的 Q/K 在 dk 维空间操作 |
| 输出投影 Wo | "拼接后的融合" | 将 n_heads × dv 维拼接向量映射回 d_model 维 |
| 注意力头 | "一个注意力函数" | 一组独立的 Q/K/V 投影 + 缩放点积注意力，捕获特定类型的关系 |
| 子空间 | "降维后的空间" | 通过 Q/K/V 投影将 d_model 维输入映射到 dk 维的低维空间 |
| 头剪枝 | "删除不重要的头" | 移除冗余的注意力头，压缩模型大小而不显著影响性能 |
| 注意力熵 | "分布的分散程度" | 注意力权重的熵——高熵表示分散关注，低熵表示集中关注 |
| KV 缓存 | "缓存键值对" | 在自回归生成时缓存之前步骤的 K/V，避免重复计算 |

---

## 📚 小结

多头注意力让模型同时捕获不同类型的词元关系——语法、语义、位置、长距离依赖。每个头有独立的 Q/K/V 投影，输出拼接后通过 Wo 投影回原始维度。8 头 × 64 维 = 1M 参数，比单头 512 维的 260K 参数多——但信息量大得多。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释为什么需要多个注意力头。如果一个头已经能学到"cat→sat"的语法关系，为什么还需要其他头？写 200 字以内的说明。

2. **【实现】** 修改多头注意力，让第 1 个头只关注位置邻近的词元（用掩码限制注意力窗口）。与全注意力对比——哪个捕获语法关系更好？

3. **【实现】** 实现一个 `AttentionHeadAnalyzer` 类——对每个头计算：平均注意力范围、最大注意力跨度、注意力熵（分布的分散程度）。

4. **【实验】** 对两个不同长度相同的句子运行同一个 MultiHeadSelfAttention 实例，对比每个头的注意力模式——什么变了？什么没变？

5. **【思考】** 阅读 FlashAttention 论文的摘要，用你自己的话解释它为什么比标准注意力快。（提示：思考 GPU 的内存层次结构）

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 多头注意力实现 | `code/main.py` | 从零实现的多头注意力、注意力头分析器，纯 NumPy |
| 注意力头分析指南 | `outputs/attention-head-guide.md` | 分析每个注意力头的学习模式 |

---

## 📖 参考资料

1. [论文] Vaswani et al. "Attention Is All You Need". NeurIPS, 2017. https://arxiv.org/abs/1706.03762
2. [博客] Jay Alammar. "The Illustrated Transformer". https://jalammar.github.io/illustrated-transformer/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
