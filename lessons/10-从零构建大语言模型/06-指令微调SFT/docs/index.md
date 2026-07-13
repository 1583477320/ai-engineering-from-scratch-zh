# 指令微调——SFT

> 预训练的GPT会预测下一个词，但它不会遵循指令。SFT用人工标注的指令-响应对教模型听话。

**类型：** 实现课 | **语言：** Python
**前置知识：** 阶段 10 · 04-05（预训练+分布式）
**时间：** ~60 分钟

---

## 🎯 学习目标

- [ ] 理解SFT的训练数据格式——指令+输入+输出
- [ ] 实现SFT的损失函数——交叉熵，只计算输出部分的token
- [ ] 说明SFT vs 预训练的数据量和成本差异

---

## 1. 问题

预训练GPT只会'预测下一个词'。要让模型遵循'把这段文字翻译成英文'或'用三句话总结这篇文章'——你需要SFT。

核心思路：用人工标注的(指令, 答案)对继续训练预训练模型。损失只计算输出部分的token（不计算输入的）。

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| SFT | 监督微调——用人工标注的指令-响应对训练 |
| 忽略损失 | 计算损失时排除输入部分——只计算输出部分的交叉熵 |
| 数据量 | SFT通常只需10K-100K条标注数据——远少于预训练 |

---

## 📚 小结

SFT = 用10K-100K条人工标注的指令-响应对继续预训练。损失只计算输出部分。数据质量和多样性比数量重要。SFT是RLHF/DPO的前置步骤——没有经过SFT的模型不应该直接用于对话。

---

## ✏️ 练习

1. 从ShareGPT或OpenAssistant数据集中抽取1000条对话数据，构建SFT训练格式
2. 在一个预训练MiniGPT（第04课）上做SFT——对比SFT前后的输出差异

---

## 📖 参考资料

1. [论文] Ouyang et al. 'Training language models to follow instructions with human feedback' (InstructGPT). 2022.
2. [数据] OpenAssistant. https://huggingface.co/datasets/OpenAssistant/oasst1

---

> 本课程参考了AI Engineering From Scratch（MIT License）的课程体系。