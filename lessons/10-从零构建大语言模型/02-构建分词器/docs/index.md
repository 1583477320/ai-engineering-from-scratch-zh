# 构建分词器——从零训练

> 分词器的质量决定了 LLM 看到什么。训练一个分词器就是决定"什么算一个 token"——这直接影响序列长度、OOV 率和模型质量。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01（分词器）| **时间：** ~60 分钟

---

## 🎯 学习目标

- [ ] 从零训练一个 BPE 分词器——理解合并过程和词表构建
- [ ] 解释词表大小对模型的影响——更大词表 = 更短序列 = 更多参数用于嵌入层
- [ ] 实现中文/英文分词器的差异化处理——为什么中文需要字级初始词表

---

## 1. 问题

预训练分词器（如 BERT tokenizer）假设了特定的词表。如果你训练自己的 LLM——尤其是针对特定领域或语言（如中文）——你需要从零训练分词器。词表大小直接影响模型架构：词表越大→嵌入层越大→但序列越短→注意力计算越少。

---

## 2. 概念

### 2.1 BPE 训练流程

```python
from tokenizers import Tokenizer, models, trainers

# 1. 定义基础模型
tokenizer = Tokenizer(models.BPE())

# 2. 定义训练器
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# 3. 训练
tokenizer.train(files=["corpus.txt"], trainer=trainer)
```

### 2.2 词表大小的权衡

| 词表大小 | 序列长度 | 嵌入层参数 | 推荐场景 |
|---|---|---|---|
| 8K | 长 | 小 | 资源受限 |
| 16K | 中 | 中 | 通用 |
| 32K | 短 | 大 | GPT/LLaMA 级别 |
| 64K | 更短 | 更大 | 多语言 |

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| BPE Trainer | 管理合并过程的训练器——指定词表大小和最小频率 |
| 合并次数 | 从字符到词表大小的迭代次数 |
| 最小频率 | 一个词对需要出现的最低次数才会被合并 |
| SentencePiece | Google 的分词器库——支持 BPE 和 Unigram |

---

## 📚 小结

训练分词器 = 选择"什么算一个 token"。BPE 从字符开始迭代合并，词表大小直接影响模型架构。中文需要字级初始词表。特殊 token（PAD/UNK/CLS/SEP/MASK）是 LLM 的接口。

---

## ✏️ 练习

1. 在 100 万中文 token 上训练 32000 词表的 BPE——打印前 20 个合并规则
2. 对比中文 vs 英文分词器——相同的文本，中文需要多少 token？英文呢？

---

## 📖 参考资料

1. [库] Hugging Face tokenizers. https://github.com/huggingface/tokenizers
2. [论文] Kudo. "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates". 2018.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
