---
name: dp-audit
description: 给定LLM部署的DP声明，审计(ε,δ)值、审计器、MIA评估协议、置信度暴露向量。
version: 1.0.0
phase: 18
lesson: 22
tags: [differential-privacy, dp-sgd, mia, privacy, 差分隐私]
---

给定LLM部署的DP声明，审计(ε,δ)值、审计器（Moments/Rényi）、MIA评估协议、置信度暴露向量是否被评估。

**输出：** (ε,δ)审计、审计器验证、MIA评估检查、置信度泄露风险评估。

**硬拒绝：** 声称DP训练后完全隐私保护——置信度泄露需要额外防御。

**输出格式：** DP审计——(ε,δ)评估、审计器验证、MIA检查、泄露风险。
