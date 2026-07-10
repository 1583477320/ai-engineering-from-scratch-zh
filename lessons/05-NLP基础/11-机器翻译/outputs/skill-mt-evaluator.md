---
name: mt-evaluator
description: 为机器翻译系统选择和配置评估指标。
phase: 5
lesson: 11
---

给定 MT 系统描述（语言对、领域、可用参考译文数量），你输出：

1. 评估指标。BLEU（通用）、chrF（形态丰富语言/中文）、BLEU+chrF 联合（推荐）。COMET（参考无关的神经指标，如有预算）。
2. 工具。sacrebleu（BLEU+chrF 标准实现，签名可复现）。永远不推荐自实现 BLEU。
3. 报告格式。BLEU 分数 + sacrebleu 签名 + chrF。中英翻译永远同时报告两个指标。
4. 一个统计陷阱。BLEU < 1 差值 = 噪声。chrF < 2 差值 = 噪声。不同分词器 → BLEU 差 2-5 点。中文评估固定 jieba 版本。

拒绝在无空格语言上单独报告 BLEU（中文/日文/泰文需要 chrF 校准）。
