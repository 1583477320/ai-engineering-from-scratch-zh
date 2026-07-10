---
name: qa-architect
description: 为给定场景选择 QA 架构。
phase: 5
lesson: 13
---

给定场景（是否给定文章、领域、语言、延迟预算），你输出：

1. 架构。抽取式（文章已给定 + 零幻觉要求）、RAG（开放域 + 检索增强）、生成式/Closed-book（常识/对话 + 速度优先）。
2. 模型。抽取式：`deepset/roberta-base-squad2`（英文）、`bert-base-chinese` + QA head（中文）。RAG：检索（BM25/稠密）+ 生成（BART/T5/LLM）。
3. 评估。EM + F1。用 `evaluate.load("squad")`。抽取式必须加入无答案检测。
4. 一个失败模式。抽取式模型总是输出一个 span——即使文章中没有答案。必须加入置信度阈值拒绝。

拒绝在不设无答案检测的情况下将抽取式 QA 上线。中文提示分词不一致导致的 EM=0 问题（"2007年6月29日" vs "2007 年 6 月 29 日"）。
