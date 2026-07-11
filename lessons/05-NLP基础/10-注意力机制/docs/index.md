# 注意力机制——突破瓶颈

> 解码器不再眯着眼盯一个压缩摘要。它开始看整个源序列。此后的一切，都是注意力加工程。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 05 · 09（序列到序列模型）
**预计时间：** ~45 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 07 · 02（自注意力从零）— 同一套数学，从"解码器看编码器"变成"序列自己看自己"

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 Bahdanau 加性注意力和 Luong 点积/一般注意力——在 NumPy 中追踪每一步的形状
- [ ] 解释 Bahdanau 的 s_{t-1} 和 Luong 的 s_t 约定差异——为什么混用是注意力实现中最难调试的 bug
- [ ] 用数值示例演示注意力权重从"猫"到"垫子"的转移——理解注意力就是显式的对齐
- [ ] 解读 Q/K/V 框架——从经典注意力到自注意力的概念桥梁，数学相同、只有来源不同

---

## 1. 问题

阶段 05 · 09 的结尾是一个可测量的失败。GRU 编码器-解码器在玩具复制任务上——序列长度 5 时准确率 89%，长度 80 时几乎随机。原因不是训练的 bug——是结构性瓶颈：**编码器提炼的全部信息都被塞进一个固定大小的隐藏状态，解码器永远看不到其他东西。**

Bahdanau、Cho、Bengio 在 2014 年发表了一个三行的修复。不要只给解码器最后一个编码器状态——**保留每一步的编码器状态。** 在解码器的每一步，计算所有编码器状态的加权平均，其中权重表示"解码器现在需要看编码器位置 `i` 的程度"。这个加权平均就是上下文——它**每一步都在变化**。

这就是注意力的全部想法。Transformer 扩展了它。自注意力把它应用到了单个序列上。多头注意力并行地运行它。但 2014 年的版本已经打破了瓶颈——理解了它，从 Seq2Seq 到 Transformer 的跳跃就不再是概念上的挑战，而是工程上的。

---

## 2. 概念

### 2.1 注意力——加权平均，权重来自查询与键的相似度

在解码器的每一步 `t`：

1. 用上一步的解码器隐藏状态 `s_{t-1}` 作为**查询**
2. 查询与每个编码器隐藏状态 `h_1,...,h_T` 打分 → 每个位置一个标量
3. Softmax 将所有分数转化为**注意力权重** `α_{t,1},...,α_{t,T}`（和为 1）
4. **上下文向量** `c_t = Σ α_{t,i} × h_i`（编码器状态的加权平均）
5. 解码器用 `c_t` + 上一步输出词元 → 生成下一个词元

**加权平均是整个机制的重点。** 当解码器需要把"Je"翻译成"I"时，注意力权重高度集中在编码器处理"Je"的位置上。当需要翻译"not"时，权重集中在"pas"上。上下文向量每一步都在重塑——不再是 Seq2Seq 的固定管道。

### 2.2 形状表——每个实现第一次都会错的地方

这是注意力实现中最容易出错的地方。逐行读。

| 张量 | 形状 | 说明 |
|---|---|---|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 如果 BiLSTM，`d_h = 2 × d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 单个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` 做 Softmax 后 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态同维度 |

### 2.3 两种打分函数

**Bahdanau（加性）分数。** `e_{t,i} = v_α^T · tanh(W_a · s_{t-1} + U_a · h_i)`。

- `s_{t-1}` 的形状是 `(d_s,)`，`h_i` 的形状是 `(d_h,)`
- `W_a` 的形状是 `(d_attn, d_s)`。`U_a` 的形状是 `(d_attn, d_h)`
- 两者在 tanh 内的和是 `(d_attn,)`
- `v_α` 的形状是 `(d_attn,)`。与 `v_α` 的内积将一个 `d_attn` 维的向量压缩为一个标量分数。**这就是 `v_α` 的作用。** 它不是魔法——是一个可学习的投影，回答"这个编码器位置对当前解码有多重要"

**Luong（乘性）分数。** 三种变体：

- `dot`（点积）：`e_{t,i} = s_t^T · h_i`。要求 `d_s == d_h`——硬约束。编码器是双向时跳过
- `general`（一般）：`e_{t,i} = s_t^T · W · h_i`，其中 `W` 的形状是 `(d_s, d_h)`。解除了等维度约束
- `concat`（拼接）：本质上是 Bahdanau 的形式。因为前两种更便宜，很少使用

**一个值得命名的 Bahdanau/Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（**生成当前词之前**的解码器状态）。Luong 使用 `s_t`（**生成当前词之后**的状态）。混用两者会产生微妙错误的梯度，极其难调试。**选一篇论文，遵守它的时间步约定。**

---

## 3. 从零实现

### 第 1 步：加性（Bahdanau）注意力

```python
import numpy as np

def additive_attention(decoder_state,      # (d_s,)
                       encoder_states,     # (T_enc, d_h)
                       W_a, U_a, v_a):
    """Bahdanau 加性注意力。

    形状追踪：
      projected_dec:  (d_attn,)         ← W_a @ decoder_state
      projected_enc:  (T_enc, d_attn)   ← encoder_states @ U_a.T
      combined:       (T_enc, d_attn)   ← tanh(projected_enc + projected_dec)  广播
      scores:         (T_enc,)          ← combined @ v_a  ← v_a 的魔法
      weights:        (T_enc,)          ← softmax(scores)
      context:        (d_h,)            ← weights @ encoder_states
    """
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights

def softmax(x):
    x = x - np.max(x)   # 减去最大值保证数值稳定性
    e = np.exp(x)
    return e / e.sum()
```

对照上面的形状表检查每一步。`encoder_states` 的形状是 `(T_enc, d_h)`。`projected_enc` 的形状是 `(T_enc, d_attn)`。`projected_dec` 的形状是 `(d_attn,)` 并广播。`combined` 的形状是 `(T_enc, d_attn)`。`scores` 的形状是 `(T_enc,)`。`weights` 的形状是 `(T_enc,)`。`context` 的形状是 `(d_h,)`。一切吻合。

### 第 2 步：Luong 点积和一般注意力

```python
def dot_attention(decoder_state, encoder_states):
    """Luong 点积注意力——三行。前提：d_s == d_h。"""
    scores = encoder_states @ decoder_state   # (T_enc,)
    weights = softmax(scores)
    return weights @ encoder_states, weights

def general_attention(decoder_state, encoder_states, W):
    """Luong 一般注意力——W 解除 d_s == d_h 的约束。"""
    projected = W.T @ decoder_state           # (d_h,)
    scores = encoder_states @ projected       # (T_enc,)
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

三行一个函数。这就是 Luong 论文落地的原因——同样准确率，少了一大半代码，解除了 Bahdanau 的额外维度 `d_attn` 和参数 `v_a`。

### 第 3 步：数值示例——注意力就是显式对齐

给定三个编码器状态（大致对应"猫"、"坐"、"垫子"）和一个最接近第一个的解码器状态——注意力分布集中在位置 0。把解码器状态移到接近第三个——注意力移到位置 2。上下文向量随之变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],   # "猫"
    [0.5, 0.5, 0.1],   # "坐"
    [0.1, 0.9, 0.3],   # "垫子"
])

# 解码器状态接近"猫" → 注意力集中在位置 0
s_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_cat, H)
print("weights:", w.round(3))        # [0.464 0.305 0.231] → 位置 0 最高

# 解码器状态接近"垫子" → 注意力转移到位置 2
s_mat = np.array([0.2, 0.8, 0.4])
ctx, w = dot_attention(s_mat, H)
print("weights:", w.round(3))        # [0.245 0.318 0.437] → 位置 2 最高
```

第一行赢了。然后把解码器状态移到接近第三个编码器状态，观察权重的转移。**这就是注意力——显式的对齐。**

### 第 4 步：为什么这是通往 Transformer 的桥梁

将上面的语言翻译为 Q/K/V：

- **查询 (Q)** = 解码器状态 `s_{t-1}`——"我现在要找什么？"
- **键 (K)** = 编码器状态——"我含有什么可以用来匹配？"
- **值 (V)** = 编码器状态——"如果被选中，我提供什么信息？"

在经典注意力中，键和值是同一组编码器状态。**自注意力将它们分离**——你可以让一个序列自己查询自己，用不同的学习投影来生成 K 和 V。多头注意力并行运行它，每个头有不同的学习投影。Transformer 把这个阶段堆叠很多层，并完全丢弃 RNN。

数学相同。形状相同。从 Bahdanau 注意力到缩放点积注意力（Transformer 的注意力层），大部分只是符号变化。

完整代码见 `code/attention_demo.py`。

---

## 4. 工业工具

### 4.1 PyTorch——MultiheadAttention 直接可用

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)     # (batch, seq_q, dim)
key   = torch.randn(2, 10, 128)    # (batch, seq_kv, dim)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
# torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 Transformer 注意力层。5 个查询位置、10 个键/值位置、128 维、8 个头。`output` 是上下文增强后的新查询。`weights` 是 (5, 10) 的可视化对齐矩阵。

### 4.2 经典注意力在 2026 年仍然重要的场景

- **教学。** 单头、单层、基于 RNN 的版本让每一个概念清晰可见
- **设备端序列任务。** Transformer 放不下的地方
- **2014-2017 年的任意论文。** 不懂 Bahdanau 的 s_{t-1} 约定，你会读错那一代全部论文
- **翻译对齐分析。** 原始注意力权重在 Transformer 模型上仍然是一个可解释性工具——即使不完全可靠，至少能提供线索

### 4.3 注意力权重作为"解释"的陷阱

注意力权重看起来太像"解释"了。它们是每个位置上和为 1 的权重；你可以画出来；高 = "模型看了这里"。审稿人喜欢它们。

**它们并不像看起来那么可解释。** Jain 和 Wallace（2019）证明了注意力分布可以被置换和替换，而模型预测在部分任务上保持不变。高权重不必然等于模型依赖这个位置。**永远不要在没有消融实验或反事实检验的情况下，将注意力权重报告为推理证据。**

---

## 5. 知识连线

注意力是 Seq2Seq 和 Transformer 之间唯一的桥：

```
Seq2Seq (2014) → 注意力 (2014-2015) → 自注意力 (2017) → Transformer
     │                │                      │
 固定瓶颈        动态上下文              序列自己看自己
 全部信息挤在     加权平均来自查询×键      多头+堆叠+无RNN
 一个向量里       每一步都在变化           O(1)路径长度
```

- **阶段 05 · 09（Seq2Seq）→** 本课修复了 Seq2Seq 的结构性瓶颈，同时保留了"编码器→解码器"的高层框架
- **阶段 07 · 02（自注意力从零）→** 本课建立的 Q/K/V 框架和形状表在那里完全复用——只有来源从"跨序列"变成"同一序列"
- **阶段 05 · 11（机器翻译）→** 注意力让解码器在生成每个目标词时动态聚焦源句中相关的部分——翻译质量从"勉强能用"跳跃到"基本流畅"

---

## 6. 工程最佳实践

### 6.1 注意力实现的形状三重检查

```python
# 在每个注意力模块的出口写死这三个断言
assert context.shape == (d_h,), f"context shape mismatch: {context.shape}"
assert weights.sum() - 1.0 < 1e-6, f"weights don't sum to 1: {weights.sum()}"
assert len(weights) == T_enc, f"weights length != encoder steps"
```

静默的广播 bug 是注意力实现中最危险的陷阱——NumPy/PyTorch 自动修正形状不匹配而不报错，但输出的语义彻底错误。形状表 + 出口断言是最低成本的防线。

### 6.2 中文场景的注意力可视化

翻译"猫坐在垫子上" → "The cat sat on the mat"时，注意力权重矩阵可以揭示：
- 生成"cat"时模型聚焦在"猫"上
- 生成"sat"时模型聚焦在"坐"上
- 生成"on the mat"时模型在"垫子上"之间分布注意力

这种对齐分析对中英翻译系统的调优有直接价值——如果注意力在生成"on"时仍然聚焦在"猫"上，说明词序对齐出了问题。

---

## 7. 常见错误

### 错误 1：Bahdanau 和 Luong 的 s_{t-1} vs s_t 混用

**现象：** 训练 loss 在下降但 BLEU/准确率几乎不变——模型在优化一个错误的时间步关系。

**原因：** Bahdanau 用 `s_{t-1}`（上一步状态）算当前注意力。Luong 用 `s_t`（当前步状态）算当前注意力。混用 = 你先决定了要看什么（`s_t`），再问自己"我应该看什么"（注意力权重用 `s_t` 算出来的）——这在概念上是自循环，在数值上产生错误梯度。

**修复：** 选定一篇论文，全部代码遵守它的时间步约定。不混合。

### 错误 2：静默的广播 bug

**现象：** 注意力权重和为 1、形状正确、loss 在降——但 BLEU 比基准差 5-8%。

**原因：** 某个维度在不该广播的地方广播了。`(d_s,)` 和 `(T_enc, d_attn)` 相加时 NumPy/PyTorch 自动扩展维度——权重计算"数学上正确"但语义上错误。广播掩盖了维度不匹配。

**修复：** 上面 §6.1 的三个出口断言。加上`assert projected_enc.shape == (T_enc, d_attn)` 等中间形状检查——在生产代码中保留它们，不要只在调试时加。

---

## 8. 面试考点

### Q1：注意力权重和为 1，高权重 = 模型依赖这个位置——这个推理为什么有风险？（难度：⭐⭐）

**参考答案：**
Jain & Wallace（2019）证明注意力分布可以被置换（高权重换到另一个位置，低权重换到原高位置）而模型预测不变。如果高权重 = 模型依赖，置换权重应该改变预测——但事实是不改变。权重高不一定意味着"模型用了这个信息来做决策"——它可能只是过路噪声，被下游的 MLP 层忽略。交叉验证注意力权重的唯一方法是消融实验：把高权重位置的输入去掉/替换，看预测是否变化。

### Q2：形状表为什么是注意力实现的第一道防线？（难度：⭐⭐）

**参考答案：**
注意力的每个中间张量在多个维度（序列长度、隐藏维度、注意力维度、批次）上交错。一个 transpose 缺失或维度不匹配会被广播机制"善意"修正而不报错——但输出的语义完全错误。最糟糕的 bug 是静默的——loss 仍在降，但模型在学一个几何上不正确的东西。形状表 + 每层断言是最低成本的防护。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 注意力 | "看东西" | 值序列的加权平均，权重来自查询-键的相似度。每一步都在动态变化 |
| Q/K/V | "三个投影" | Q=查询（我要找什么），K=键（我含有什么），V=值（我提供什么信息）。经典注意力中 K=V，自注意力中三者来自不同投影 |
| 加性注意力 | "Bahdanau" | 前馈网络打分——v^T·tanh(W·q + U·k)。v 将中间维度压缩为标量 |
| 乘性注意力 | "Luong" | 点积打分——q^T·k 或 q^T·W·k。三行代码，不输准确率 |
| 对齐矩阵 | "那张漂亮的图" | 注意力权重作为 (T_dec, T_enc) 的矩阵。看它来读模型关注了什么 |
| s_{t-1} vs s_t | "时间步陷阱" | Bahdanau 用生成前状态，Luong 用生成后状态。混用是注意力最经典的难调试 bug |

---

## 📚 小结

注意力用三行代码修复了 Seq2Seq 的结构性瓶颈——解码器的每一步不再被一个固定向量限制，而是动态地回看所有编码器位置。从 Bahdanau 的加性注意力到 Luong 的点积再到 Q/K/V 框架——数学没有变，变的是 Q 和 K 的来源（跨序列 → 同一序列）和 V 是否独立投影。

注意力权重看起来像解释——但它们不是。高权重 ≠ 重要。消融实验永远是唯一的验证路径。

下一课是机器翻译——把 Seq2Seq + 注意力编织成真正可用的翻译系统。

---

## ✏️ 练习

1. 【理解】实现 softmax 掩码——将填充位置的编码器注意力权重设为 0。在一批变长序列上验证权重和仍为 1 且填充位置权重为 0。

2. 【实现】在 Luong 一般形式的基础上实现多头注意力：将 d_h 拆分为 n_heads 组，每组独立运行注意力，拼接结果。验证单头情况与原始实现一致。

3. 【实验】为阶段 05 · 09 的复制任务加上 Bahdanau 注意力。绘制准确率 vs 序列长度曲线，与无注意力基线对比。差距应随长度增大而扩大——验证注意力确实提升了瓶颈。

4. 【思考】在解释翻译质量时，同事指着注意力热力图说"这里高亮，说明模型看了这个位置"。你应该礼貌地指出什么？建议什么额外验证？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 注意力从零实现 | `code/attention_demo.py` | Bahdanau + Luong + 数值演示 + Q/K/V 过渡，含形状检查 |
| 可复用提示词 | `outputs/prompt-attention-shapes.md` | 调试注意力形状 bug 的系统化方案 |

---

## 📖 参考资料

1. [论文] Bahdanau, Cho, Bengio. "Neural Machine Translation by Jointly Learning to Align and Translate". ICLR, 2015. https://arxiv.org/abs/1409.0473 — 注意力原论文
2. [论文] Luong, Pham, Manning. "Effective Approaches to Attention-based Neural Machine Translation". EMNLP, 2015. https://arxiv.org/abs/1508.04025 — 三种打分函数的对比
3. [论文] Jain and Wallace. "Attention is not Explanation". NAACL, 2019. https://arxiv.org/abs/1902.10186 — 注意力权重的可解释性警告
4. [教程] Dive into Deep Learning — Bahdanau Attention. https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html — 可运行的 PyTorch 走读

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、Bahdanau/Luong 对比分析、形状追踪注释、工程最佳实践、常见错误、面试考点等均为原创内容。
