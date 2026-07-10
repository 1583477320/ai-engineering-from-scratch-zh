---
name: seq2seq-design
description: 为给定任务设计 Seq2Seq 流水线。
phase: 5
lesson: 09
---

给定任务（翻译、摘要、改写、问题重写），你输出：

1. 架构。默认 = 预训练 Transformer 编码器-解码器（BART/T5/mBART/NLLB）。RNN Seq2Seq 仅用于流式/边缘推理/教学。
2. 起始 checkpoint。`facebook/bart-base`、`google/flan-t5-base`、`facebook/nllb-200-distilled-600M`。中文：`fnlp/bart-base-chinese`。
3. 解码策略。贪心 = 确定性输出。束搜索（width 4-5）= 质量优先。带温度采样 = 多样性优先。一句话理由。
4. 上线前验证的一个失败模式。暴露偏差 → 长输出生成漂移。取 20 条 90% 分位长度的输出，人工检查。

拒绝在 < 100 万平行样例下推荐从头训练 Seq2Seq。标记用户向内容用贪心解码为脆弱（循环、重复）。
