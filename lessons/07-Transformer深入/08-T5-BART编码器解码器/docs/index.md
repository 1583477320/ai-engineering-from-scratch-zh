# T5/BART——编码器-解码器架构

> BERT 理解文本，GPT 生成文本，T5/BART 同时理解并生成。编码器-解码器架构是 NLP 中"最全能"的选择。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 07 · 05（完整 Transformer）、06（BERT）、07（GPT）
**时间：** ~75 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 比较 BERT（编码器）、GPT（解码器）、T5（编码器+解码器）的架构差异
- [ ] 解释 T5 的"Text-to-Text"统一框架——为什么翻译、摘要、QA 都用同一个模型
- [ ] 理解 BART 与 T5 的差异——BART 用去噪自编码预训练，T5 用 span corruption

---

## 1. 问题

BERT 擅长理解（分类、问答），但不能生成。GPT 擅长生成，但生成是单向的。T5 和 BART 将两者结合：**编码器理解输入，解码器生成输出**。翻译、摘要、问答——所有"输入序列→输出序列"的任务都用同一个架构。

---

## 2. 概念

### 2.1 三种 Transformer 变体

| 架构 | 组件 | 生成 | 代表模型 |
|---|---|---|---|
| 仅编码器 | 只有 Transformer 编码器 | ❌ | BERT, RoBERTa |
| 仅解码器 | 只有 Transformer 解码器 | ✅ | GPT, LLaMA |
| 编码器+解码器 | 编码器 + 解码器 + 交叉注意力 | ✅ | T5, BART, BLOOM |

### 2.2 T5——"Text-to-Text"

T5 的核心思想：**所有任务都是文本到文本。**

```
翻译:  "translate English to French: The cat sat." → "Le chat s'est assis."
摘要:  "summarize: [长文本]" → "简短摘要"
分类:  "classify: [文本]" → "positive"
QA:    "answer: 问题 + 上下文" → "答案"
```

一个模型，一个框架，所有 NLP 任务。

### 2.3 BART vs T5

| | BART | T5 |
|---|---|---|
| 预训练 | 去噪自编码（破坏文本→恢复） | 文本到文本（span corruption） |
| 架构 | BERT 编码器 + GPT 解码器 | 标准编码器-解码器 |
| 强项 | 文本生成、摘要、翻译 | 多任务统一 |
| 论文 | Lewis et al. 2020 | Raffel et al. 2020 |

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 编码器-解码器 | "Transformer 两半" | 编码器理解输入，解码器生成输出，交叉注意力连接两者 |
| T5 | "文本到文本" | 所有 NLP 任务都统一为输入文本→输出文本格式 |
| BART | "BERT + GPT" | BERT 风格编码器 + GPT 风格解码器，预训练为去噪自编码 |
| 交叉注意力 | "解码器看编码器" | 解码器每一步关注编码器的所有位置——生成时可以"回看"输入 |

---

## 📚 小结

T5 和 BART 是"最全能"的 Transformer 变体——编码器理解、解码器生成。T5 用"文本到文本"框架统一所有任务；BART 用去噪预训练在生成任务上更强。两者在翻译、摘要、QA 上仍是强基线——虽然 LLM 正在取代它们，但它们更快、更小、在数据有限时更稳。

---

## ✏️ 练习

1. 比较 BERT、GPT、T5 在相同硬件上的参数量和推理速度——画出权衡曲线
2. 用 T5 实现一个简单的翻译任务——理解编码器-解码器在 seq2seq 上的工作方式

---

## 📖 参考资料

1. [论文] Raffel et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5). 2020.
2. [论文] Lewis et al. "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension". 2020.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
