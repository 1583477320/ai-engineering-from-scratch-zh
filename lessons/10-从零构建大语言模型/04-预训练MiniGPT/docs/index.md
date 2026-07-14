# 预训练 Mini GPT（124M 参数）

> GPT-2 Small 有 1.24 亿参数。12 层 Transformer、12 个注意力头、768 维嵌入。你可以在单 GPU 上几小时从零训练它。大多数人从不这样做——他们用预训练 checkpoint。但如果你不自己训练一个，你就没有真正理解你正在构建产品的模型内部发生了什么。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01-03（分词器 + 数据管道）| **时间：** ~120 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 05（缩放与分布式训练）— 从 124M 到 7B+ | 阶段 10 · 06（指令微调）— 从预训练到对齐

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 GPT-2 架构（124M 参数）——词嵌入、位置编码、Transformer 块、LM head
- [ ] 在文本语料上用 next-token prediction 训练 GPT——交叉熵损失
- [ ] 实现自回归文本生成——温度采样和 top-k/top-p 过滤
- [ ] 监控训练损失曲线——验证模型学到连贯的语言模式

---

## 1. 问题

你知道 Transformer 是什么。你能画出图、背出"attention is all you need"。但有一个问题：**你从零训练过一个吗？**

GPT-2 Small（124M 参数）是理解 LLM 内部运作的最佳窗口。它足够小可以在单 GPU 上几小时训练完成，但足够大可以看到真正的语言建模行为——从随机噪声到连贯文本的转变。

你将亲手实现：每个权重的初始化、每一步梯度更新、每一行损失计算。读完这课，当你说"我知道 LLM 怎么工作"时，你真的知道。

---

## 2. 概念

### 2.1 Mini GPT 架构（GPT-2 Small）

| 参数 | 值 |
|------|-----|
| 层数 L | 12 |
| 隐藏维度 d | 768 |
| 注意力头数 h | 12 |
| 注意力维度 d_k | 64 |
| 上下文长度 | 1024 |
| 词表大小 V | 32000 |
| 总参数 | ~124M |

```
输入 token 序列 (B, 1024)
    ↓
词嵌入 + 位置编码 (B, 1024, 768)
    ↓
12 × Transformer Block
  ┌─ Multi-Head Self-Attention（因果掩码）
  ├─ Layer Norm
  ├─ FFN (768 → 3072 → 768)
  └─ Layer Norm
    ↓
LM Head (768 → 32000)
    ↓
下一个词概率分布 (B, 1024, 32000)
```

### 2.2 训练配置

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 优化器 | AdamW | β₁=0.9, β₂=0.95, weight_decay=0.1 |
| 学习率 | 1e-4 峰值 → 1e-5 最小 | 余弦退火 + 1000 步预热 |
| 批次大小 | 32 × 1024 = 32768 tokens | 梯度累积或大批次 |
| 训练 token 数 | 1B | 约 OpenWebText 10% |
| 训练时间 | ~8 小时 | 单 A100 80GB |
| 混合精度 | BF16 | 显存减半，速度提升 |

### 2.3 损失函数

交叉熵损失——预测下一个词元的分布与实际词元的差异：

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i})$$

困惑度（Perplexity）= exp(L)。GPT-2 Small 在 OpenWebText 上的困惑度约 20-30。越低越好——20 意味着模型在每个位置"犹豫"大约 20 个选择。

### 2.4 生成策略

```python
def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8, top_k=40):
    """自回归生成。"""
    tokens = prompt_tokens
    for _ in range(max_new_tokens):
        logits = model(tokens)[:, -1, :] / temperature
        # top-k 过滤
        top_k_values, _ = torch.topk(logits, top_k)
        logits[logits < top_k_values[:, -1:]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens
```

**温度**：>1 更随机/创造性，<1 更确定/重复。**Top-k**：只从概率最高的 k 个词元中采样。

---

## 3. 从零实现

完整代码见 `code/main.py`。核心结构：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    vocab_size = 32000
    block_size = 1024
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

训练循环见 `code/main.py`（使用 nanoGPT 风格的简洁训练）。

---

## 4. 工具

### 4.1 nanoGPT（最推荐）

```bash
# 克隆 nanoGPT
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT

# 训练 GPT-2 Small（124M）
python train.py config/train_gpt2.py
```

nanoGPT 是 Andrej Karpathy 的极简 GPT 训练实现——一个文件完成数据准备、模型定义、训练循环。是理解 GPT 训练的最佳资源。

### 4.2 HuggingFace Transformers

```python
from transformers import GPT2LMHeadModel, GPT2Config

# 从零初始化 GPT-2 Small
config = GPT2Config(
    vocab_size=50257, n_positions=1024, n_embd=768,
    n_layer=12, n_head=12,
)
model = GPT2LMHeadModel(config)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
# 124,439,808
```

### 4.3 工具对比

| 工具 | 用途 | 特点 |
|------|------|------|
| nanoGPT | 从零训练 | 教学用，代码最简洁 |
| HuggingFace | 加载/微调 | 预训练权重，即插即用 |
| LitGPT | 灵活训练 | 支持多种架构 |

---

## 5. LLM 视角

### 5.1 预训练学到了什么

124M 的 GPT-2 Small 在 OpenWebText 上训练后能学到：
- **语法**：正确的词序、主谓一致
- **语义**：词与词之间的关系
- **世界知识**：简单的事实（"法国的首都是巴黎"）
- **推理**：简单的逻辑链条

但 124M 还太小——无法进行复杂推理，也容易产生事实错误。

### 5.2 参数量的直觉

- 124M：可以在笔记本上运行，理解训练过程
- 1.5B：可以生成连贯段落
- 7B：可以进行简单对话
- 70B：可以进行复杂推理、写代码
- 405B：当前最强开源模型级别

### 5.3 使用 ChatGPT / Claude 时的直接体验

当你使用 GPT-4 时，它背后的架构本质上是 GPT-2 的放大版——更多的层、更多的参数、更多的训练数据。GPT-2 Small 是理解 GPT-4 的"最小可行版本"。

---

## 6. 工程最佳实践

### 6.1 训练监控

| 指标 | 正常范围 | 问题信号 |
|------|---------|---------|
| 训练 loss | 稳定下降到 ~3.0-3.5 | 不下降=学习率太小；震荡=太大 |
| 验证 loss | 接近训练 loss | 远高于训练 loss=过拟合 |
| 生成质量 | 逐步从胡言乱语到连贯文本 | 1000 步后仍胡言乱语=数据问题 |

### 6.2 中文场景特别建议

- 用中文语料（如 Wikipedia 中文版）训练中文 MiniGPT
- 词表大小建议 32K-64K（覆盖中英文）
- 中文文本长度分布不同——平均句长比英文短

### 6.3 踩坑经验

- **loss 不下降**：检查学习率（太小）、数据格式（标签是否正确对齐）、掩码是否正确
- **生成重复**：温度太低或 top_k 太小——增大温度或减小 top_k
- **显存不足**：减小 batch_size，或使用梯度累积

---

## 7. 常见错误

### 错误 1：标签没有正确对齐

**现象：** 训练 loss 不下降。

**原因：** GPT 的标签是输入向右移一位——`labels[i] = input_ids[i+1]`。如果没移位，模型在学"预测自己"而非"预测下一个词"。

### 错误 2：位置编码超出范围

**现象：** 长文本生成时位置编码崩溃。

**原因：** 训练时位置编码只覆盖 block_size（如 1024）。生成超过 1024 token 时需要截断或使用 RoPE。

### 错误 3：忘记设置 eval 模式

**现象：** 生成文本中出现随机噪声或重复片段。

**原因：** 模型处于训练模式——Dropout/LayerNorm 统计量不稳定。

---

## 8. 面试考点

### Q1：为什么 next-token prediction 是有效的预训练目标？（难度：⭐⭐）

**参考答案：**
Next-token prediction 迫使模型学习语言的所有方面：语法（正确词序）、语义（词间关系）、世界知识（事实记忆）、推理（逻辑链条）。因为正确的预测需要理解上下文中所有信息——模型不能"作弊"，必须真正理解输入。这也是为什么 LLM 可以做如此多任务——它们学会了"理解文本"，而"理解"的本质是"能预测下一个词"。

### Q2：124M 和 1.5B 的 GPT 在能力上有什么质的差异？（难度：⭐⭐⭐）

**参考答案：**
124M 参数量不足以存储大量世界知识——它能生成语法正确的文本，但事实准确率低。1.5B 开始能"记住"训练数据中的常见事实（如"法国的首都是巴黎"），并进行简单的上下文内推理。参数量的增加不只是"更多相同的东西"——存在涌现能力阈值——某些能力（如多步推理、代码理解）在参数量达到某个阈值后突然出现。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 预训练 | "训练基础模型" | 在大规模文本上用 next-token prediction 训练——学习语言的通用表示 |
| 因果注意力 | "只能往前看" | 下三角掩码——位置 i 只能关注 0..i |
| 困惑度 (PPL) | "模型多困惑" | exp(交叉熵损失)——越低越好，20-30 是 GPT-2 Small 的水平 |
| 温度 | "生成有多随机" | >1 更随机/创造性，<1 更确定/重复 |
| Top-k | "只从最好的 k 个里选" | 只从概率最高的 k 个词元中采样——防止极低概率词元被选中 |

---

## 📚 小结

GPT-2 Small（124M 参数）是理解 LLM 的最小可行版本。12 层 Transformer、768 维、1024 上下文。在 OpenWebText 上用 next-token prediction 训练几小时。训练损失从 ~10 降到 ~3.0，生成从胡言乱语到连贯文本。下一课我们将理解如何将这个 124M 扩展到 7B+——分布式训练、FSDP、DeepSpeed。

---

## ✏️ 练习

1. **【实现】** 用 nanoGPT 在 Shakespeare 数据集上训练一个小型 GPT（12M 参数，4 层，128 维）。观察 loss 曲线——什么时候生成开始变得"莎士比亚"？
2. **【实验】** 对比不同温度（0.5, 0.8, 1.0, 1.2）下的生成质量——哪个温度下生成最好？
3. **【实验】** 测量不同数据量（10M, 100M, 1B tokens）对验证 loss 的影响——画出 scaling law 曲线。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| GPT-2 实现 | `code/main.py` | 124M 参数 GPT 的完整实现 + 训练循环 |

---

## 📖 参考资料

1. [GitHub] nanoGPT: https://github.com/karpathy/nanoGPT — 从零训练 GPT 的最佳教程
2. [论文] Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2). 2019.
3. [论文] Kaplan et al. "Scaling Laws for Neural Language Models". arXiv, 2020. https://arxiv.org/abs/2001.08361

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
