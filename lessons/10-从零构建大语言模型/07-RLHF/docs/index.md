# RLHF——从人类反馈中学习

> SFT教模型格式，RLHF教模型价值观。用人类偏好训练的奖励模型给PPO提供信号——让模型生成'对的'而非'格式正确的'答案。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 10 · 06（SFT）、阶段 09 · 09（RLHF基础）
**时间：** ~60 分钟

---

## 🎯 学习目标

- [ ] 理解RLHF三阶段——SFT→奖励模型→PPO
- [ ] 解释奖励模型的Bradley-Terry偏好损失
- [ ] 说明PPO在RLHF中的角色——用KL散度惩罚防止模型跑偏

---

## 1. 问题

SFT教会了模型'遵循格式'但没有教会'什么算好的答案'。RLHF用人类偏好判断来定义'好'。

三阶段：1)SFT打好基础→2)训练奖励模型学习偏好→3)PPO用奖励分数优化策略。KL散度惩罚防止模型偏离SFT太远。

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| 奖励模型 | 从人类偏好中学习的打分器——给PPO提供训练信号 |
| KL散度惩罚 | 防止PPO后的模型偏离SFT太远——保持生成质量 |
| PPO-ptx | 在PPO的同时保留预训练损失——防止灾难性遗忘 |

---

## 📚 小结

RLHF = SFT + 奖励模型 + PPO + KL散度惩罚。InstructGPT/ChatGPT的核心训练配方——2022年后被广泛采用。DPO（Direct Preference Optimization）在2024年后简化了这个流程：不需要单独的奖励模型。

---

## ✏️ 练习

1. 从对比学习数据中训练一个奖励模型——评估其与人类偏好的一致性
2. 用你的MiniGPT+奖励模型实现简化的RLHF——对比RLHF前后的输出质量

---

## 📖 参考资料

1. [论文] Ouyang et al. 'Training language models to follow instructions with human feedback' (InstructGPT). 2022.

---

> 本课程参考了AI Engineering From Scratch（MIT License）的课程体系。