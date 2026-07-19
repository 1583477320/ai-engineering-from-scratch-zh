---
name: mesa-diagnostic
description: 分类安全评估报告中的故障模式——外对齐/内对齐代理/欺骗性对齐。
version: 1.0.0
phase: 18
lesson: 06
tags: [mesa-optimization, deceptive-alignment, safety-evaluation, 元优化]
---

给定安全评估报告，将每个识别的故障模式分类到 {外对齐失败, 内对齐代理, 内对齐欺骗} 中，并推荐相应的缓解类别。

**输出：** 故障分类、缓解建议、监控指标。

**硬拒绝：** 不加区分地应用对抗训练——对抗训练对欺骗性对齐可能适得其反。

**输出格式：** 元优化诊断——故障分类、缓解推荐、监控指标。
