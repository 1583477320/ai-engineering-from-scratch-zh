---
name: reward-hack-auditor
description: 识别RLHF模型中的奖励黑客表现——冗长、谄媚、不忠实推理、评估者篡改。
version: 1.0.0
phase: 18
lesson: 02
tags: [reward-hacking, goodhart, overoptimization, rlhf, 奖励黑客]
---

给定训练日志和评估报告，识别奖励黑客的四种表现，定位代理-黄金差距，推荐缓解方案。

**输出：** 表现的识别（冗长/谄媚/不忠实推理/评估者篡改）、代理-黄金差距位置（KL距离）、缓解建议（数据质量/RM鲁棒性/KL调度/过程监督）。

**硬拒绝：** 只加高KL系数——KL正则化软化但不防止黄金奖励崩塌。

**输出格式：** 奖励黑客审计——表现识别、差距定位、缓解建议、KL评估。
