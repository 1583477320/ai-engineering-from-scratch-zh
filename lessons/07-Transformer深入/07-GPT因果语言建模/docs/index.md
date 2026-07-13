# GPT——因果语言建模

> GPT 不预测被遮住的词，而是预测下一个词。这个自回归目标教会了模型生成——以及为什么生成比理解更难。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 07 · 05（完整 Transformer）、06（BERT）
**时间：** ~75 分钟
**所处阶段：** Tier 2
**关联课程：** 第 7 阶段 · 08（T5/BART）— 对比解码器架构与编码器-解码器架构

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分 BERT（MLM）和 GPT（CLM）的训练目标——为什么生成是自回归的
- [ ] 实现 GPT 的因果掩码——位置 i 只能看到位置 0..i-1
- [ ] 解释 GPT 在指令微调中如何从"预测下一个词"变成"遵循指令"

---

## 1. 问题

BERT 做得很好——但它是双向的，不能直接用于生成。生成是自回归的——你必须在写下一个词之前知道已经写了哪些词。GPT 的答案：**只看左边的上下文**——预测下一个词。每个位置的注意力被因果掩码限制到 0..i-1。

这很简单——但有效。GPT 系列从 GPT-1 到 GPT-4，用同一个目标在所有规模上持续进步。

---

## 2. 概念

### 2.1 因果掩码

```
正常注意力:    q1可以看 k1, k2, k3, k4, k5（全部可见）
因果注意力:    q1只能看 k1
               q2只能看 k1, k2
               q3只能看 k1, k2, k3
               ...
```

**实现方式：** 在注意力分数矩阵的上三角位置（i < j）填入 -∞——softmax 后这些位置的概率为 0。

### 2.2 GPT vs BERT

| | BERT | GPT |
|---|---|---|
| 训练目标 | MLM：预测被遮住的词 | CLM：预测下一个词 |
| 注意力方向 | 双向（看全部上下文） | 单向（只看左边） |
| 生成能力 | ❌ 不适合生成 | ✅ 自然生成 |
| 分类能力 | ✅ 天然擅长 | ✅ 通过指令微调 |
| 预训练效率 | 较低（15% 的词被预测） | 较高（100% 的词都被预测） |

### 2.3 指令微调

GPT 的"魔法"不只是预训练——是**指令微调**。在预训练的"预测下一个词"之上，加入大量"指令-响应"对。这让模型从"预测词"变成"理解并执行指令"。

---

## 3. 从零实现

### 因果掩码

```python
import torch

def create_causal_mask(seq_len):
    """创建上三角掩码——位置 i 只能看 0..i-1。"""
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    return mask  # 上三角为 -inf，对角线及以下为 0
```

### GPT 块（因果自注意力）

```python
class CausalSelfAttention:
    def __init__(self, d_model, n_heads):
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    
    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        attn_out, _ = self.mha(x, x, x, attn_mask=mask)
        return attn_out
```
完整代码见 `code/main.py`——纯 NumPy，可立即运行。

---

## 4. 工业工具

### 4.1 HuggingFace Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载预训练的 GPT-2 模型
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 输入文本
text = "The cat sat"
inputs = tokenizer(text, return_tensors="pt")

# 生成下一个词元
with torch.no_grad():
    outputs = model(**inputs)
    next_token_logits = outputs.logits[0, -1, :]
    next_token_id = next_token_logits.argmax()
    print(f"预测下一个词元: {tokenizer.decode(next_token_id)}")
```

### 4.2 性能对比

| 模型 | 参数量 | 层数 | 头数 | 上下文长度 |
|---|---|---|---|---|
| GPT-2 | 124M | 12 | 12 | 1024 |
| GPT-3 | 175B | 96 | 96 | 2048 |
| GPT-4 | ~1.8T | 128 | 128 | 128K |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

GPT 系列是大语言模型的开创者。GPT-3（175B 参数）证明了"预测下一个词"这个简单目标，在足够大时能涌现出理解、翻译、对话、推理等复杂能力。

你在本课实现的因果掩码和自回归生成，正是 GPT 系列的核心机制。每次你与 ChatGPT 对话时，模型都在执行因果掩码——每个位置只能看到左边的上下文。

### 5.2 LLM 时代什么变了？

**规模变了。** 从 GPT-1（117M）到 GPT-3（175B），参数规模扩大了 1500 倍。因果语言模型的目标没有变，但规模带来了质变——涌现能力。

**训练变了。** 从预训练 → 微调 → 指令微调 → RLHF。因果语言模型不再只是"预测下一个词"，而是"理解并执行指令"。

**推理变了。** 从朴素的自回归生成（O(n²)）到 FlashAttention（O(n)）、KV 缓存、推测解码。

### 5.3 什么没变？

**核心目标没变。** GPT-4 仍然在用因果语言建模训练——预测下一个词。这是所有大语言模型的基础。

**因果掩码没变。** 无论模型多大，因果掩码都是确保自回归性质的关键。

**自回归没变。** 生成仍然是一个词一个词地产生——只是在工程上做了大量优化。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你在 ChatGPT 中输入一段话，模型生成回复时，它在每个生成步骤都执行因果自注意力——每个新生成的词元只能看到左边的上下文。

如果你让模型写一段 500 字的文章，它产生了 500 个词元——每个词元都执行了一次完整的注意力计算。这就是为什么长文本生成比短文本慢很多倍。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 学习 / 实验 | GPT-2（124M）| 可在单 GPU 上运行 |
| 生成 / 对话 | GPT-4 / Claude | API 调用 |
| 自托管 | LLaMA 3 / Mistral | 开源模型 |
| 推理优化 | vLLM + KV 缓存 | 吞吐量提升 10-20x |

### 6.2 中文场景特别建议

- 中文生成任务优先使用 LLaMA 3 或 GPT-4 的分词器（128K+ 词表，对多语言更友好）
- 注意：GPT-2 的分词器对中文不友好——中文文本被过度切分

### 6.3 踩坑经验

- 推理时因果掩码不能省略——否则训练/推理分布不匹配
- KV 缓存在大批次下显存消耗大——注意管理
- 生成时建议设置温度和 top-p 采样——贪心解码生成结果太保守

---

## 7. 常见错误

### 错误 1：推理时忘记因果掩码

**现象：** 模型在训练时正常，推理时输出退化。

**原因：** 训练时用因果掩码（自回归）。推理时如果忘记加掩码——模型可以"看到"未来词元——训练/推理分布不匹配。

**修复：**
```python
# ❌ 错误写法：推理时无掩码
output = model(input_ids)  # 所有位置互相可见

# ✓ 正确写法：推理时有因果掩码
mask = create_causal_mask(seq_len)
output = model(input_ids, attn_mask=mask)  # 位置 i 只能看 0..i-1
```

### 错误 2：BERT 的 [MASK] token 不在 GPT 的词表中

**现象：** 尝试将 BERT 用于 GPT 的生成任务，输出混乱。

**原因：** BERT 训练时见过 [MASK]；GPT 的词表中没有 [MASK]。BERT 设计用于双向理解，GPT 设计用于单向生成——架构目标不同。

### 错误 3：温度设置不当导致生成质量差

**现象：** 生成结果重复（温度太低）或混乱（温度太高）。

**原因：** 温度过低时模型总是选概率最高的词——生成内容单调。温度过高时模型随机选择——生成内容混乱。

**修复：**
```python
# ❌ 温度过低
next_token = greedy_sample(logits)  # 总是选概率最高的词

# ✓ 使用温度采样
probs = softmax(logits / temperature)  # temperature=0.8 是常用值
next_token = np.random.choice(vocab_size, p=probs)
```

### 错误 4：未使用 KV 缓存导致推理很慢

**现象：** 生成 100 个词元的时间远超生成 10 个词元的 10 倍。

**原因：** 没有 KV 缓存时，每个生成步骤都重新计算所有位置的 K/V——O(n²) 复杂度。KV 缓存只计算最新的 K/V——O(n) 复杂度。

**修复：**
```python
# ❌ 无 KV 缓存：每一步重新计算全部
for step in range(max_new_tokens):
    logits = model(input_ids)  # 重复计算之前的 K/V

# ✓ 有 KV 缓存：只计算最新的 K/V
for step in range(max_new_tokens):
    logits, past_kv = model(input_ids, past_key_values=past_kv)
```

### 错误 5：注意力分数未缩放

**现象：** 训练初期 loss 不降，梯度为 NaN。

**原因：** 即使有因果掩码，Q @ K.T 的点积也需要除以 √d_k 防止 softmax 饱和。

**修复：**
```python
# ❌ 错误写法：未缩放
scores = Q @ K.T  # d_k=64 时点积可达几十

# ✓ 正确写法：缩放
scores = Q @ K.T / np.sqrt(d_k)  # 除以 √d_k 防止饱和
```

---

## 8. 面试考点

### Q1：因果掩码和普通掩码有什么区别？（难度：⭐⭐）

**参考答案：**
因果掩码用于自回归模型——位置 i 只能看到位置 0..i-1。普通掩码（如填充掩码）用于处理不同长度的序列——填充位置被遮挡。因果掩码的形状是 (seq_len, seq_len) 的上三角矩阵，普通掩码的形状是 (batch, seq_len) 的布尔矩阵。

### Q2：GPT 的自回归生成为什么比 BERT 的分类慢？（难度：⭐⭐）

**参考答案：**
GPT 生成时每个词元需要执行一次完整的前向传播。生成 100 个词元需要 100 次前向传播。BERT 分类时只需要一次前向传播——直接取 [CLS] 的输出。

### Q3：KV 缓存的作用是什么？（难度：⭐⭐⭐）

**参考答案：**
KV 缓存在自回归生成时缓存之前步骤的 K/V 矩阵。没有 KV 缓存时，生成第 i 个词元需要重新计算所有前 i-1 个词元的 K/V——复杂度 O(n²)。有 KV 缓存时，只需计算第 i 个词元的 K/V——复杂度 O(n)。

### Q4：指令微调如何改变 GPT 的行为？（难度：⭐⭐⭐）

**参考答案：**
预训练后的 GPT 只知道"预测下一个词"。指令微调通过大量指令-响应对训练，让模型从"预测下一个词"变成"理解并执行指令"。这是 GPT-3 和 ChatGPT 的核心区别。

### Q5：为什么 GPT 在分类任务上不如 BERT？（难度：⭐⭐）

**参考答案：**
GPT 是单向的——每个词元只能看到左边的上下文。分类需要理解整个句子的语义，双向上下文比单向上下文更丰富。GPT 可以通过指令微调做分类，但同等参数规模下 BERT 的分类准确率更高。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| CLM | "因果语言模型" | 预测下一个词——自回归生成 |
| 因果掩码 | "上三角掩码" | 位置 i 只能看位置 0..i-1 |
| 指令微调 | "让 LLM 听话" | 在预训练基础上用指令-响应对训练 |
| 自回归 | "一次生成一个词" | 生成过程是顺序的——当前词依赖前面所有词 |
| KV 缓存 | "缓存键值对" | 缓存之前步骤的 K/V，避免重复计算 |
| 涌现能力 | "规模带来的质变" | 模型大到一定程度后自动获得的能力 |
| 温度采样 | "控制随机性" | 调整 softmax 分布的平滑程度 |
| 推测解码 | "并行生成" | 用小模型先生成候选，大模型验证 |

---

## 📚 小结

GPT 用因果语言建模预训练——预测下一个词，注意力被掩码限制为只看左边。与 BERT 的双向理解互补：BERT 擅长分类，GPT 擅长生成。指令微调将 GPT 从"预测词"变成"遵循指令"——这是 GPT-3/4 成功的关键转折。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释为什么 GPT 需要因果掩码。如果没有因果掩码，GPT 会学到什么？写 200 字以内的说明。

2. **【实现】** 实现因果掩码并在 GPT 块上验证——对比有/无掩码时的输出差异。

3. **【实现】** 实现带 KV 缓存的 GPT 推理——对比有/无缓存的推理速度差异。

4. **【实验】** 用小型 GPT 在 WikiText 上训练，报告困惑度——对比不同模型大小（125M, 350M, 1.3B）。

5. **【思考】** 阅读 InstructGPT 论文，用你自己的话解释指令微调如何改变 GPT 的行为。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| GPT 因果语言模型实现 | `code/main.py` | 因果掩码、GPT 块、自回归生成 |
| GPT vs BERT 对比指南 | `outputs/gpt-vs-bert-comparison.md` | 两种架构的详细对比 |

---

## 📖 参考资料

1. [论文] Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2). 2019.
2. [论文] Ouyang et al. "Training language models to follow instructions with human feedback" (InstructGPT). 2022.
3. [论文] Brown et al. "Language Models are Few-Shot Learners" (GPT-3). 2020.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
