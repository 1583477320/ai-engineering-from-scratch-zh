---
name: ipi-audit
description: 智能体部署的 IPI 审计——不受信任内容源枚举和 IFC 检查。
version: 1.0.0
phase: 18
lesson: 15
tags: [ipi, indirect-prompt-injection, rag-injection, ifc, 间接提示注入]
---

给定智能体部署描述，枚举不受信任内容源，检查是否应用了 IFC，标记未经信任标签就到达模型的内容源。

**输出：** 不受信任内容源列表、IFC 应用检查、缺口报告、修复建议。

**硬拒绝：** 声称对所有 IPI 防御——Nasr 等人 2025 年证明 12 个已发表防御全部被自适应攻击打破。

**输出格式：** IPI 审计——内容源枚举、IFC 检查、缺口报告、修复建议。
