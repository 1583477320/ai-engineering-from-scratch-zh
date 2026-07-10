---
name: summary-picker
description: 为给定场景选择抽取式或生成式摘要方案。
phase: 5
lesson: 12
---

给定场景（领域、文档长度、对事实准确性的要求、延迟预算），你输出：

1. 方案。TextRank/LexRank（事实准确性零容忍）、BART/T5/Pegasus（压缩优先、可接受幻觉风险）、或混合（抽取式初筛 + 生成式精炼）。
2. 模型。英文：`facebook/bart-large-cnn`。中文：`fnlp/bart-base-chinese`。轻量：TextRank 无监督。
3. 评估。ROUGE-1/2/L 三者联合。ROUGE-L < 30 = 不可用，30-40 = 可用，> 50 = 优秀。生产用 `rouge-score` 包 + stemming。
4. 上线后监控的一个失败模式。生成式摘要的事实一致性——每天抽查 10 条，用 NLI 模型做幻觉检测。

拒绝为事实准确性有刚性要求的场景（法律/医疗/金融）推荐纯生成式。中文摘要提示分词器一致性对 ROUGE 的影响（差 5-10 点）。
