---
name: llm-security-plan
description: 根据监管范围和安全现状，设计LLM服务安全方案——保险库迁移、PII脱敏器、网络出口策略、审计日志。
version: 1.0.0
phase: 17
lesson: 25
tags: [security, pii, vault, guardrails, audit, 安全]
---

**输出：** 保险库迁移方案、PII脱敏器设计、网络出口白名单策略、审计日志Schema、密钥轮转策略（≤90天）。

**硬拒绝：**
- 密钥硬编码在代码中——必须通过保险库拉取。
- PII后处理清理——必须在推理前脱敏。

**输出格式：** 安全方案——凭据管理、PII脱敏、网络出口、审计日志、轮转策略。
