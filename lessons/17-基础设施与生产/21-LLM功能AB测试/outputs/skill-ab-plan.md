---
name: ab-plan
description: 根据功能变更、工作负载和基线，选择 A/B 测试平台、门控和样本量。
version: 1.0.0
phase: 17
lesson: 21
tags: [ab-testing, cuped, statsig, growthbook, 实验]
---

根据功能变更、工作负载和基线，设计 A/B 测试方案。

**输出：**

1. **测试轴选择。** 提示词（措辞）、模型（选择）、参数（temperature/top-p）。
2. **指标设计。** 主要指标（任务成功率）、护栏指标（成本/延迟）、次要指标（用户反馈）。
3. **平台选择。** Statsig（一体化 SaaS）或 GrowthBook（开源仓库原生）。
4. **样本量计算。** 考虑 LLM 非确定性（×1.3-1.5 倍余量）。
5. **校正策略。** Bonferroni（严格）或 Benjamini-Hochberg（FDR 控制）。

**硬拒绝：**
- "感觉更好"就发布——必须有 A/B 测试的统计显著性。
- 不修正多重比较——20 个测试在 95% 置信度下会产生一个假阳性。

**输出格式：** A/B 测试方案——测试轴、指标、平台、样本量、校正策略。
