---
name: eval-architect
description: 为 LLM/RAG 系统设计评估流水线。
phase: 5
lesson: 27
---

给定系统类型（RAG/聊天机器人/摘要/QA），你输出：

1. 评估框架。RAG：RAGAS（四个维度：忠实度、答案相关性、上下文精确率、上下文召回率）。对话：DeepEval + G-Eval（LLM-as-Judge + 自定义标准）。CI/CD：DeepEval（Pytest 原生）。
2. LLM-as-Judge 校准。三次检查——评分分布（人工 vs LLM 的 Pearson > 0.7）、位置偏差（交换候选 A/B 顺序取均值）、长度偏差（加入长度惩罚或在标准中明确排除）。
3. 评估频率。每次变更（提示词/分块/检索器/模型）跑 200 条评估批次。每周人工抽查忠实度分数最低的 20 条。
4. 中文特别提醒。LLM-as-Judge 的 prompt 语言应与被评估内容一致（中文输出 → 中文评判 prompt）。中文忠实度的原子声明拆分需适配中文语法（按"。"、"、"；"和连词拆分）。

拒绝在没有 LLM-as-Judge 校准的情况下将评估分数作为唯一上线依据。
