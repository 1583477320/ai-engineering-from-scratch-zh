# Transformer 毕业设计

> 从零组装一个完整的编码器-解码器 Transformer——注意力、位置编码、训练循环、推理解码，一个都不能少。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 07 · 02-13 | **预计时间：** ~180 分钟 | **所处阶段：** Tier 2 | **关联课程：** 第 10 阶段（大语言模型从零）——本课的架构是理解 GPT、BERT、T5 的直接前置

---

## 学习目标

完成本课后，你能够：

- [ ] 从零实现一个完整的 Encoder-Decoder Transformer——包含多头注意力、位置编码、前馈网络、层归一化
- [ ] 实现教师强制训练循环——理解训练时和推理时解码器输入的差异
- [ ] 实现贪心解码和束搜索——解释两者的优劣并对比输出质量
- [ ] 诊断教师强制导致的训练-推理分布偏差，并说明缓解策略
- [ ] 将你的 NumPy 实现与 PyTorch `nn.Transformer` 对比，说明工业实现做了哪些优化

---

## 1. 问题

前面 12 课分别构建了自注意力、多头注意力、位置编码、编码器、解码器、BERT、GPT、T5 等组件。你已经理解了每个零件的工作原理。

但理解零件不等于能造出机器。

一个能跑的翻译系统需要：词嵌入把文本变成向量，位置编码注入顺序信息，编码器理解源序列，解码器自回归生成目标序列，交叉注意力连接两端，因果掩码防止信息泄漏，训练循环让所有参数协同工作。漏掉任何一个环节，系统就无法运行。

更关键的是，训练和推理是两个不同的世界。训练时你有教师强制——解码器每一步都能看到"正确答案的前文"。推理时你没有——解码器只能看到自己生成的词元，一步错步步错。

这节课的任务是：把所有零件组装成一个完整的、可以在玩具翻译任务上训练和推理的系统。不是调 API，不是用框架，而是用纯 NumPy 从零构建。做完这节课，你就真正理解了 Transformer 的每一个细节。

---

## 2. 概念

### 2.1 完整架构总览

```
源序列  "the cat sat ."
   │
   ▼
[词嵌入 + 正弦位置编码]
   │
   ▼
[编码器 × N 层]
  ├── 多头自注意力（所有位置互相看）
  ├── Add & LayerNorm
  ├── 前馈网络（GELU 激活）
  └── Add & LayerNorm
   │
   ▼
编码器输出（源序列的上下文表示）
   │
   ▼
解码器 × N 层
  ├── 多头掩码自注意力（只能看过去）
  ├── Add & LayerNorm
  ├── 多头交叉注意力（Q 来自解码器，K/V 来自编码器）
  ├── Add & LayerNorm
  ├── 前馈网络（GELU 激活）
  └── Add & LayerNorm
   │
   ▼
输出投影 → 词表概率分布 → 取下一个词元
```

### 2.2 训练 vs 推理

| | 训练（教师强制） | 推理（自回归） |
|---|---|---|
| 解码器输入 | 目标序列右移一位（BOS + 前 n-1 个词元） | 上一步生成的词元 |
| 因果掩码 | 有 | 有 |
| 交叉注意力 | 看完整编码器输出 | 同上 |
| 损失 | 交叉熵 | 无（贪心/束搜索） |
| 并行度 | 一次前向传播处理整个序列 | 每步只能生成一个词元 |

这个差异是 Transformer 最大的"陷阱"——训练时一切顺利，推理时可能一塌糊涂。后面的常见错误章节会详细讨论。

### 2.3 自注意力 vs 交叉注意力

自注意力中，Q、K、V 都来自同一个序列——每个位置关注序列中的所有位置。编码器和解码器的自注意力层都属于这一类。

交叉注意力中，Q 来自解码器，K 和 V 来自编码器。这让解码器在生成每个词元时，都能回看源序列的任何位置。这是编码器和解码器之间的"信息桥梁"。

---

## 3. 从零实现

完整代码见 `code/main.py`。下面按组件逐步讲解。

### Step 1：基础工具函数

```python
import numpy as np

PAD, BOS, EOS = 0, 1, 2  # 特殊词元 ID


def softmax(x):
    """数值稳定的 Softmax。"""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sinusoidal_encoding(max_len, d):
    """正弦位置编码——无需学习参数，为模型注入位置信息。"""
    pe = np.zeros((max_len, d))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d, 2) * -(np.log(10000) / d))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe
```

正弦位置编码的核心思想：用不同频率的正弦和余弦函数为每个位置生成唯一的向量。偶数维度用 sin，奇数维度用 cos。这样模型可以通过线性变换学到相对位置关系。

### Step 2：多头注意力

```python
class MultiHeadAttention:
    """多头注意力——自注意力和交叉注意力共用同一个类。"""

    def __init__(self, d_model, n_heads, seed=42):
        assert d_model % n_heads == 0
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        rng = np.random.default_rng(seed)
        s = np.sqrt(2.0 / d_model)
        self.Wq = rng.normal(0, s, (d_model, d_model))
        self.Wk = rng.normal(0, s, (d_model, d_model))
        self.Wv = rng.normal(0, s, (d_model, d_model))
        self.Wo = rng.normal(0, s, (d_model, d_model))

    def forward(self, q_in, kv_in, mask=None):
        sq, sk = q_in.shape[0], kv_in.shape[0]
        h, dk = self.n_heads, self.d_k
        Q = (q_in @ self.Wq).reshape(sq, h, dk).transpose(1, 0, 2)
        K = (kv_in @ self.Wk).reshape(sk, h, dk).transpose(1, 0, 2)
        V = (kv_in @ self.Wv).reshape(sk, h, dk).transpose(1, 0, 2)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dk)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        weights = softmax(scores)
        out = (weights @ V).transpose(1, 0, 2).reshape(sq, -1)
        return out @ self.Wo
```

关键设计：`forward` 方法接收 `q_in` 和 `kv_in` 两个参数。自注意力时两者相同（`q_in == kv_in`），交叉注意力时不同（`q_in` 来自解码器，`kv_in` 来自编码器）。一个类同时服务于两种注意力，这与原始论文的设计一致。

### Step 3：编码器块和解码器块

```python
class EncoderBlock:
    """编码器块：自注意力 + 前馈网络，每个子层都有残差连接和层归一化。"""

    def __init__(self, d_model, n_heads, d_ff, seed=42):
        self.attn = MultiHeadAttention(d_model, n_heads, seed)
        self.ffn = FeedForward(d_model, d_ff, seed + 100)
        self.ln1, self.ln2 = LayerNorm(d_model), LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.ln1.forward(x + self.attn.forward(x, x, mask))
        return self.ln2.forward(x + self.ffn.forward(x))


class DecoderBlock:
    """解码器块：掩码自注意力 + 交叉注意力 + 前馈网络。"""

    def __init__(self, d_model, n_heads, d_ff, seed=42):
        self.self_attn = MultiHeadAttention(d_model, n_heads, seed)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, seed + 50)
        self.ffn = FeedForward(d_model, d_ff, seed + 100)
        self.ln1, self.ln2, self.ln3 = (
            LayerNorm(d_model), LayerNorm(d_model), LayerNorm(d_model)
        )

    def forward(self, x, enc_out, causal_mask=None):
        x = self.ln1.forward(x + self.self_attn.forward(x, x, causal_mask))
        x = self.ln2.forward(x + self.cross_attn.forward(x, enc_out))
        return self.ln3.forward(x + self.ffn.forward(x))
```

解码器块比编码器块多一个交叉注意力层。注意交叉注意力不需要因果掩码——解码器可以自由查看编码器的所有位置。但自注意力必须加因果掩码，防止当前位置"偷看"未来的词元。

### Step 4：训练循环（教师强制）

```python
def train(model, pairs, src_vocab, tgt_vocab, epochs=200, lr=0.01):
    """训练循环——教师强制 + 交叉熵损失。"""
    for epoch in range(epochs):
        for src_text, tgt_text in pairs:
            src_ids = np.array(encode(src_text, src_vocab))
            tgt_ids = encode(tgt_text, tgt_vocab)
            # 教师强制：输入是 <bos> + 目标（去掉最后一个词元）
            dec_in = np.array([BOS] + tgt_ids[:-1])
            dec_tgt = np.array(tgt_ids)

            logits = model.forward(src_ids, dec_in)
            loss = cross_entropy(logits, dec_tgt)

            # 输出投影梯度：dL/dW = hidden^T @ (softmax - one_hot)
            probs = softmax(logits)
            grad = probs.copy()
            grad[np.arange(len(dec_tgt)), dec_tgt] -= 1
            model.decoder.proj -= lr * model.decoder._hidden.T @ grad
```

教师强制的关键：解码器的输入是 `[BOS, t_1, t_2, ..., t_{n-1}]`，目标是 `[t_1, t_2, ..., t_n]`。每个位置的输入都是"前面所有正确词元"，模型只需预测下一个词元。这让训练可以并行处理整个序列。

### Step 5：贪心解码

```python
def greedy_decode(model, src_ids, max_len=20):
    """贪心解码——每步取概率最高的词元，直到 EOS。"""
    enc_out = model.encoder.forward(src_ids)
    seq = [BOS]
    for _ in range(max_len):
        logits = model.decoder.forward(np.array(seq), enc_out)
        next_id = int(logits[-1].argmax())
        if next_id == EOS:
            break
        seq.append(next_id)
    return seq
```

贪心解码的核心问题：每步只看当前最优，不考虑未来。一旦选错，后续所有位置都在错误基础上继续。

### Step 6：束搜索

```python
def beam_search(model, src_ids, beam_width=3, max_len=20):
    """束搜索——保持 beam_width 个累积概率最高的候选。"""
    enc_out = model.encoder.forward(src_ids)
    beams = [(0.0, [BOS])]  # (累积对数概率, 序列)

    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == EOS:
                candidates.append((score, seq))
                continue
            logits = model.decoder.forward(np.array(seq), enc_out)
            log_probs = np.log(softmax(logits[-1]) + 1e-8)
            top_k = log_probs.argsort()[-beam_width:]
            for tid in top_k:
                candidates.append((score + log_probs[tid], seq + [int(tid)]))
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

    return beams[0][1]
```

束搜索的关键区别：不是贪心地选1个，而是展开 K 个候选，保留得分最高的 K 个。这相当于在 K 个方向上同时搜索，最终选全局最优。代价是计算量增大 K 倍。

---

## 4. 工业工具

### 4.1 PyTorch 内置实现

```python
import torch
import torch.nn as nn

# PyTorch 内置 Transformer——一个类搞定
model = nn.Transformer(
    d_model=512, nhead=8,
    num_encoder_layers=6, num_decoder_layers=6,
    dim_feedforward=2048,
    batch_first=True
)

# 训练
src = torch.randint(0, 10000, (32, 20))  # (batch, seq_len)
tgt = torch.randint(0, 10000, (32, 20))
output = model(src, tgt)
print(f"输出形状: {output.shape}")  # (32, 20, 512)
```

### 4.2 HuggingFace Transformers

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# T5——Google 的编码器-解码器大语言模型
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

inputs = tokenizer("translate English to French: Hello world", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### 4.3 性能对比

| 实现方式 | 速度 | 内存 | 适用场景 |
|---|---|---|---|
| 我们的 NumPy 版 | 慢 | 低 | 学习理解 |
| PyTorch `nn.Transformer` | 快 | 中 | 训练 / 研究 |
| FlashAttention-2 | 极快 | 低 | 长上下文训练 |
| vLLM PagedAttention | 极快 | 极低 | 大语言模型推理 |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

你在这节课实现的编码器-解码器架构，就是 T5、BART、mBART 等模型的基础。T5 的 "T5" 就是 "Text-to-Text Transfer Transformer"——把所有 NLP 任务都统一为文本到文本的格式。Google 用这个架构在 1T 词元上预训练，得到了一个在翻译、摘要、问答等任务上都表现优秀的通用模型。

GPT 系列只用了解码器部分（去掉交叉注意力），BERT 只用了编码器部分。理解完整的编码器-解码器架构，你就理解了为什么 GPT 擅长生成、BERT 擅长理解——因为它们各自只用了 Transformer 的一半。

### 5.2 LLM 时代什么变了？

原始 Transformer 用正弦位置编码，最大支持 512 词元。现代大语言模型用 RoPE（旋转位置嵌入）支持 128K 甚至更长的上下文。注意力机制本身没变，但工程实现变了——FlashAttention 把 O(n²) 的内存占用降到 O(n)，让百万词元的上下文成为可能。

训练方式也变了。原始 Transformer 用 Adam 优化器从随机初始化开始训练。现代大语言模型用 AdamW + 学习率预热 + 余弦退火，在数万亿词元上预训练。但核心算法——自注意力、层归一化、残差连接——几乎没有变化。

### 5.3 什么没变？

Transformer 的核心公式 Attention(Q, K, V) = softmax(QK^T / √d_k) V 自 2017 年以来从未改变。无论模型是 1 亿参数还是 1.8 万亿参数，无论是翻译任务还是多模态任务，注意力计算的数学形式完全一致。

残差连接和层归一化也保持不变。这两个"不起眼"的组件是 Transformer 能训练到深层的关键——没有它们，梯度消失会让 6 层以上的模型无法训练。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你让 ChatGPT 翻译一段话时，它做的事情和你在这节课实现的完全一样：编码源语言，解码目标语言，自回归逐词生成。区别只是模型更大（数千亿参数）、训练数据更多（数万亿词元）、推理优化更好（KV 缓存 + FlashAttention）。

你在这节课遇到的教师强制偏差问题，在 ChatGPT 中同样存在——这就是为什么大语言模型有时会"幻觉"，生成看似流畅但事实错误的内容。训练时模型看到的是正确答案，推理时只能看到自己的输出，一旦偏移就回不来了。

---

## 6. 工程最佳实践

### 6.1 训练配置推荐

| 场景 | d_model | n_heads | n_layers | d_ff | 学习率 |
|---|---|---|---|---|---|
| 玩具实验 | 32-64 | 4 | 1-2 | 64-128 | 1e-3 |
| 小型研究 | 128-256 | 4-8 | 2-4 | 256-1024 | 1e-4 |
| 中型训练 | 512-1024 | 8-16 | 6-12 | 2048-4096 | 3e-4 |
| 大规模预训练 | 4096+ | 32+ | 24+ | 11008+ | 1e-4 |

### 6.2 中文场景特别建议

- 中文词表大小建议 20K-50K（字级别 6K，子词级别更大）。太小会导致每个中文字被切分为多个词元，序列变长
- 中英混合训练时，确保中文语料占比不低于 30%，否则中文能力退化
- 位置编码的最大长度要覆盖你的目标场景——短文本对话 512 够用，长文档处理需要 4K-32K

### 6.3 踩坑经验

- 训练时忘记加因果掩码：模型"偷看"未来，训练 loss 异常低，推理时输出乱码
- 解码器输入忘记移位：输入和目标是同一个序列，模型学会"复制"而非"生成"
- 嵌入维度不是 `n_heads` 的整数倍：多头注意力拆分时直接报错
- 训练数据太少（< 100 条）：模型记住所有训练样本，泛化能力为零
- 推理时没有 EOS 停止条件：生成无限循环直到达到 max_len

---

## 7. 常见错误

### 错误 1：解码器训练时用了教师强制但推理时不用

**现象：** 训练 loss 很低但推理输出乱码。

**原因：** 训练时解码器输入是目标序列（教师强制），推理时是上一步输出。训练和推理的输入分布不同，导致模型在推理时"没见过"自己生成的错误输出。

**修复：**
```python
# ❌ 训练和推理用完全不同的输入模式
# 训练：解码器输入 = 目标序列
# 推理：解码器输入 = 自己的上一步输出

# ✓ 缓解方法：逐步降低 teacher_forcing_ratio
for epoch in range(epochs):
    ratio = max(0.0, 1.0 - epoch / (epochs * 0.8))
    if random.random() < ratio:
        dec_in = target_ids  # 教师强制
    else:
        dec_in = model_output_ids  # 自回归
```

### 错误 2：忘记添加 EOS token

**现象：** 解码器永远不会停止生成——无限循环。

**原因：** 没有终止条件。解码器不断预测下一个词元直到达到 max_len。

**修复：**
```python
# ❌ 缺少终止条件
for _ in range(max_len):
    next_id = argmax(logits)
    seq.append(next_id)  # 永远不会停

# ✓ 检测 EOS 并停止
for _ in range(max_len):
    next_id = argmax(logits)
    if next_id == EOS:
        break  # 遇到结束标记停止
    seq.append(next_id)
```

### 错误 3：因果掩码方向搞反

**现象：** 模型生成时每个位置都能"看到未来"——训练 loss 异常低，但生成结果完全不对。

**原因：** 掩码矩阵构建时上三角和下三角搞反了。解码器中，位置 i 应该只能看到 0~i（下三角），不能看到 i+1~n（上三角）。

**修复：**
```python
# ❌ 这会遮住过去，露出未来
mask = np.tril(np.ones((n, n)), k=1) * -1e9

# ✓ 上三角为 True（被替换为 -inf），遮住未来
mask = np.triu(np.ones((n, n), dtype=bool), k=1)
scores = np.where(mask, -1e9, scores)
```

### 错误 4：交叉注意力中 Q/K/V 来源搞混

**现象：** 编码器和解码器之间的信息传递中断，模型退化为只用自注意力。

**原因：** 交叉注意力中 Q 应来自解码器，K/V 应来自编码器。如果三者都来自解码器，就变成了自注意力。

**修复：**
```python
# ❌ 三者相同——变成自注意力
cross_attn.forward(dec_hidden, dec_hidden)

# ✓ Q 来自解码器，K/V 来自编码器
cross_attn.forward(dec_hidden, enc_output)
```

### 错误 5：输出投影矩阵维度不匹配

**现象：** 训练时抛出形状错误，或输出 logits 维度与词表大小不一致。

**原因：** 输出投影矩阵的形状应为 `(d_model, vocab_size)`，将隐藏状态映射到词表空间。如果维度搞反，softmax 会作用在错误的轴上。

**修复：**
```python
# ❌ 维度搞反
proj = np.random.randn(vocab_size, d_model)  # (vocab, d_model)

# ✓ 正确：d_model -> vocab_size
proj = np.random.randn(d_model, vocab_size)  # (d_model, vocab)
logits = hidden @ proj  # (seq_len, vocab_size)
```

---

## 8. 面试考点

### Q1：Transformer 用自注意力替代了 RNN，这带来了什么好处和代价？（难度：⭐⭐）

**参考答案：**

好处：**并行化**。RNN 必须按时间步顺序计算，长度为 n 的序列需要 n 步。自注意力可以并行计算所有位置的注意力，训练速度大幅提升。此外，任意两个位置之间的路径长度从 O(n) 降到 O(1)，长距离依赖更容易学习。

代价：**计算复杂度从 O(n) 变为 O(n²)**。每个位置都要与所有其他位置计算注意力分数。当序列长度从 512 增加到 4096 时，计算量增加 64 倍。这就是为什么 FlashAttention 和 KV 缓存等工程优化如此重要。

### Q2：为什么训练时用教师强制而推理时不用？能否统一？（难度：⭐⭐）

**参考答案：**

教师强制在训练时让解码器每一步都看到"正确答案的前文"，这样可以并行计算整个序列的损失，训练速度快且稳定。推理时没有"正确答案"可用，只能用模型自己上一步的输出。

可以部分统一：Scheduled Sampling 在训练时以一定概率用模型自己的输出代替真实词元，逐步增加自回归比例。这让模型在训练时就"习惯"自己可能犯错的情况，缩小训练-推理分布偏差。但完全统一需要改变训练范式（如使用强化学习），代价较大。

### Q3：解释交叉注意力在编码器-解码器 Transformer 中的作用。（难度：⭐⭐）

**参考答案：**

交叉注意力是编码器和解码器之间的信息桥梁。在解码器的每一层中，自注意力让解码器理解"已经生成了什么"，交叉注意力让解码器"回看源序列"。

具体来说，交叉注意力的 Q 来自解码器的当前隐藏状态（"我在生成什么位置"），K 和 V 来自编码器的输出（"源序列每个位置包含什么信息"）。通过注意力权重，解码器可以聚焦于源序列中与当前生成位置最相关的部分。例如翻译 "The cat sat" 时，生成 "le chat" 时交叉注意力会聚焦于 "The cat"。

### Q4：如果让你设计一个支持 100K 上下文长度的 Transformer，你会做哪些修改？（难度：⭐⭐⭐）

**参考答案：**

原始 Transformer 的 O(n²) 注意力在 100K 上下文下不可行——注意力矩阵需要 100K × 100K ≈ 100 亿个浮点数。

关键修改：
1. **FlashAttention-2**：IO 感知的注意力实现，内存从 O(n²) 降到 O(n)，速度提升 2-4 倍
2. **KV 缓存**：推理时缓存已计算的 K 和 V，避免重复计算
3. **位置编码升级**：从正弦编码改为 RoPE（旋转位置嵌入），支持外推到更长序列
4. **分组查询注意力（GQA）**：多个 Q 头共享同一组 K/V 头，减少 KV 缓存内存
5. **Ring Attention**：跨多 GPU 分布式处理超长序列

### Q5：从零实现 Transformer 时，你认为最关键的三个设计决策是什么？（难度：⭐⭐⭐）

**参考答案：**

1. **缩放因子 √d_k**：没有这个缩放，点积 QK^T 的值量级会随维度增大，推入 Softmax 饱和区导致梯度消失。这是让 Transformer 能训练到深层的数学保证。

2. **因果掩码**：解码器自注意力必须加因果掩码，确保位置 i 只能看到 0~i 的位置。没有掩码，模型在训练时就能"偷看"未来词元，推理时做不到这一点。

3. **残差连接 + 层归一化**：这两个组件共同解决了深层网络的训练稳定性问题。残差连接保证梯度能直接流过，层归一化稳定每层的输入分布。去掉任何一个，6 层以上的 Transformer 就很难训练。

---

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 教师强制 | "训练时告诉模型正确答案" | 训练时将目标序列右移一位作为解码器输入，让每个位置都能看到前文的正确词元，从而并行计算整个序列的损失 |
| 束搜索 | "比贪心更好" | 保持 K 个候选序列，每步展开所有候选并保留得分最高的 K 个，最终选全局最优——用 K 倍计算量换取更好的输出质量 |
| 因果掩码 | "让模型只能往前看" | 将注意力矩阵的上三角设为 -inf，确保位置 i 只能关注 0~i，用于自回归生成时防止信息泄漏 |
| 交叉注意力 | "解码器关注编码器" | 解码器的 Q 与编码器的 K/V 计算注意力——让生成每个词元时都能回看源序列的任何位置，是编码器-解码器架构的核心连接 |
| 残差连接 | "跳过一层直接连" | 将子层的输入直接加到输出上（x + sublayer(x)），保证梯度能直接流过，是训练深层 Transformer 的关键 |
| 层归一化 | "每层做一次标准化" | 在每个子层的输出上做归一化（减均值除标准差），稳定训练过程中的激活值分布 |
| 位置编码 | "告诉模型词的顺序" | Transformer 没有循环结构，不知道词元的先后顺序。正弦位置编码为每个位置生成唯一向量，注入顺序信息 |
| GELU 激活 | "比 ReLU 更平滑" | 高斯误差线性单元——0.5 * x * (1 + tanh(...))。比 ReLU 在零点更平滑，被 BERT 和 GPT 广泛使用 |

---

## 小结

你从零实现了一个完整的 Encoder-Decoder Transformer——从词嵌入到位置编码，从多头自注意力到交叉注意力，从训练时的教师强制到推理时的贪心解码和束搜索。理解了每个组件的作用和它们之间的协作关系。

这是阶段 07 的毕业设计。到这里，你已经掌握了 Transformer 架构的全部核心组件。下一课我们将进入阶段 10（大语言模型从零），用这些知识从头构建一个 GPT——你会发现，GPT 只是 Transformer 解码器的放大版。

---

## 练习

1. 【实现】在 `code/main.py` 的基础上，添加 Scheduled Sampling——训练时以一定概率用模型自己的预测代替真实词元。记录训练过程中的损失变化，与纯教师强制对比。

2. 【实验】分别用贪心解码和束搜索（beam_width=5）翻译 10 句测试句子。对比两者的输出，记录哪些句子束搜索更好、哪些差异不大。

3. 【思考】你的 NumPy 实现为什么只能训练输出投影矩阵，而不能训练所有层？要实现完整的反向传播需要什么？写 200 字以内的分析。

4. 【扩展】修改模型配置（增加到 d_model=128, n_heads=8, n_layers=2），在更大的数据集上训练。观察模型是否能学会更复杂的翻译模式。

5. 【对比】用 PyTorch 的 `nn.Transformer` 实现同样的翻译任务。对比你的 NumPy 版本和 PyTorch 版本的训练速度和最终损失。

---

## 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 完整 Transformer 实现 | `code/main.py` | 纯 NumPy 实现的编码器-解码器 Transformer，含训练和推理 |
| 毕业设计自检清单 | `outputs/capstone-checklist.md` | 逐项检查架构组件、训练流程、推理策略的完成度 |

---

## 参考资料

1. [论文] Vaswani et al. "Attention Is All You Need". NeurIPS, 2017. https://arxiv.org/abs/1706.03762
2. [代码] Harvard NLP. "The Annotated Transformer". https://nlp.seas.harvard.edu/annotated-transformer/
3. [官方文档] PyTorch `nn.Transformer`: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
4. [GitHub] FlashAttention: https://github.com/Dao-AILab/flash-attention
5. [论文] Raffel et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". JMLR, 2020. https://arxiv.org/abs/1910.10683

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
