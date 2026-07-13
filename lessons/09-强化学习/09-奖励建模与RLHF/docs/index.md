# 奖励建模与 RLHF

> PPO 解决了"怎么优化"，但没有解决"优化什么"。RLHF 说：用人类偏好训练一个奖励模型，然后用 PPO 优化它。这是 ChatGPT 的核心训练配方。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 09 · 08（PPO）、阶段 7 · 07（GPT）| **时间：** ~60 分钟

---

## 🎯 学习目标

- [ ] 解释 RLHF 的三阶段流水线——SFT → 奖励模型 → PPO 微调
- [ ] 理解奖励模型的 Bradley-Terry 偏好损失
- [ ] 说明 DPO 如何绕过奖励模型——直接从偏好对优化策略

---

## 1. 问题

预训练的 LLM 会生成流畅但可能有害的文本。SFT（监督微调）解决了"格式"但没有解决"质量"。RLHF 说：**让人类判断什么是好答案，训练一个奖励模型，然后用 PPO 优化 LLM 生成这个奖励模型认为好的答案。**

### RLHF 三阶段

```
阶段1: SFT（监督微调）
  用人工标注的问答对微调 LLM → 基础能力

阶段2: 奖励模型训练
  给人类展示两个回答 → 哪个更好？→ 训练奖励模型

阶段3: PPO 微调
  LLM 生成回答 → 奖励模型打分 → PPO 优化策略 → 重复
```

### DPO——直接偏好优化

RLHF 的缺点：需要单独的奖励模型。DPO 说：**直接从偏好对中学习——不需要奖励模型。** 损失函数：

$$L_{DPO}(\theta) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

简单说：**让好答案更可能，坏答案更不可能**——用参考策略锚定。

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| SFT | 监督微调——用人工问答对微调 LLM |
| 奖励模型 | 从人类偏好中学习的打分器 |
| RLHF | 用 PPO + 奖励模型优化 LLM |
| DPO | 直接偏好优化——绕过奖励模型，直接从偏好对训练 |
| KL 散度惩罚 | 防止 LLM 偏离预训练分布太远 |

---

## 📚 小结

RLHF = SFT（基础能力）+ 奖励模型（什么算好）+ PPO（优化策略）。DPO 直接从偏好对学习——更简单。GRPO（DeepSeek 2024）用组相对策略优化——更高效。这三者是 2026 年 LLM 对齐的三大支柱。

---

## ✏️ 练习

1. 从偏好对数据中实现 DPO 损失——对比 DPO vs PPO 在相同数据上的收敛速度
2. 画出 RLHF 三阶段的计算成本对比——SFT vs RM vs PPO 各占多少

---

## 📖 参考资料

1. [论文] Ouyang et al. "Training language models to follow instructions with human feedback" (InstructGPT). 2022.
2. [论文] Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (DPO). 2023.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
