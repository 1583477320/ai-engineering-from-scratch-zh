# LLM评估

> 你如何知道你的模型变好了？困惑度不够。你需要人类偏好的代理指标——AlpacaEval、MT-Bench、Chatbot Arena——以及最终的裁判：人类评估。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 10 · 04（预训练MiniGPT）
**时间：** ~45 分钟

---

## 🎯 学习目标

- [ ] 区分内在评估和外在评估——困惑度 vs 人类偏好
- [ ] 解释AlpacaEval和MT-Bench的工作原理——LLM-as-Judge
- [ ] 说明Chatbot Arena的ELO评分——基于用户偏好的LLM排名

---

## 1. 问题

训练完了——怎么知道模型变好了？困惑度衡量'预测下一个词的能力'，但用户的直接问题是'你更想用哪个模型'。

三个流行benchmark：AlpacaEval（单轮问答）、MT-Bench（多轮对话用LLM评分）、Chatbot Arena（用户偏好ELO排名）。

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| 困惑度 | 模型对测试数据的'惊讶程度'——越低越好 |
| AlpacaEval | 单轮问答基准——LLM-as-Judge评分 |
| MT-Bench | 多轮对话基准——GPT-4作为评判者 |
| Chatbot Arena | ELO排名的'LLM世界杯'——最佳的主观质量评估 |

---

## 📚 小结

困惑度 ≠ 质量。AlpacaEval（单轮）、MT-Bench（多轮LLM评分）、Chatbot Arena（用户偏好ELO）三个维度互补。LLM-as-Judge常见但需要校准——位置偏差和长度偏差。

---

## ✏️ 练习

1. 用你的MiniGPT在AlpacaEval上评估——对比SFT前后的得分变化
2. 对比LLM-as-Judge vs 人类评估的一致性——在20个样本上计算相关系数

---

## 📖 参考资料

1. [项目] AlpacaEval. https://github.com/tatsu-lab/alpaca_eval
2. [项目] Chatbot Arena. https://lmsys.org/blog/2023-10-30-arena/

---

> 本课程参考了AI Engineering From Scratch（MIT License）的课程体系。