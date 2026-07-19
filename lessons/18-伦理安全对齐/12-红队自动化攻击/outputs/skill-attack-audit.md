---
name: attack-audit
description: 审计红队评估报告——哪些攻击被运行、预算、裁判、有害行为集。
version: 1.0.0
phase: 18
lesson: 12
tags: [red-teaming, pair, gcg, jailbreakbench, harmbench, 红队测试]
---

给定红队评估报告，审计：哪些攻击被运行（PAIR, GCG, TAP, AutoDAN, PAP），每个的预算，使用的裁判，以及评估的行为集（JailbreakBench, HarmBench, 内部）。

**输出：** 攻击覆盖分析、预算合理性评估、裁判选择评估、行为集完整性。

**硬拒绝：** 不带查询预算报告ASR——90% ASR在200次查询与85% ASR在20次查询不可比。

**输出格式：** 攻击审计——覆盖分析、预算评估、裁判评估、行为集完整性。
