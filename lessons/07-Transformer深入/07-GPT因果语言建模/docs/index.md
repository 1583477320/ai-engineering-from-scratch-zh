# GPT——因果语言建模

> GPT 不预测被遮住的词，而是预测下一个词。这个自回归目标教会了模型生成——以及为什么生成比理解更难。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 07 · 05（完整 Transformer）、06（BERT）
**时间：** ~75 分钟

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

---

## 4. 常见错误

### 错误 1：推理时忘记因果掩码

**现象：** 模型在训练时正常，推理时输出退化。

**原因：** 训练时用因果掩码（自回归）。推理时如果忘记加掩码——模型可以"看到"未来词元——训练/推理分布不匹配。

### 错误 2：BERT 的 [MASK] token 不在 GPT 的词表中

**现象：** 尝试将 BERT 用于 GPT 的生成任务，输出混乱。

**原因：** BERT 训练时见过 [MASK]；GPT 的词表中没有 [MASK]。BERT 设计用于双向理解，GPT 设计用于单向生成——架构目标不同。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| CLM | "因果语言模型" | 预测下一个词——自回归生成 |
| 因果掩码 | "上三角掩码" | 位置 i 只能看位置 0..i-1 |
| 指令微调 | "让 LLM 听话" | 在预训练基础上用指令-响应对训练 |
| 自回归 | "一次生成一个词" | 生成过程是顺序的——当前词依赖前面所有词 |

---

## 📚 小结

GPT 用因果语言建模预训练——预测下一个词，注意力被掩码限制为只看左边。与 BERT 的双向理解互补：BERT 擅长分类，GPT 擅长生成。指令微调将 GPT 从"预测词"变成"遵循指令"——这是 GPT-3/4 成功的关键转折。

---

## ✏️ 练习

1. 实现因果掩码并在 GPT 块上验证——对比有/无掩码时的输出差异
2. 用小型 GPT 在 WikiText 上训练，报告困惑度——对比不同模型大小（125M, 350M, 1.3B）

---

## 📖 参考资料

1. [论文] Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2). 2019.
2. [论文] Ouyang et al. "Training language models to follow instructions with human feedback" (InstructGPT). 2022.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
