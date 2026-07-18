---
name: observability-stack
description: 根据技术栈、规模、预算和许可证要求，选择 LLM 可观测性工具。
version: 1.0.0
phase: 17
lesson: 13
tags: [langfuse, langsmith, phoenix, helicone, opik, openllmetry, 可观测性]
---

根据技术栈（LangChain/原生 SDK/多供应商）、规模、预算和许可证要求，选择 LLM 可观测性工具组合。

**输出：**

1. **技术栈分析。** 是否使用 LangChain/LangGraph？是否需要多供应商支持？
2. **工具选择。** 推荐主工具+备用方案。
3. **OpenTelemetry 集成。** 如何通过 OTel 粘合网关和评估平台。
4. **采样策略。** 根据请求量设计采样规则。
5. **成本估算。** 按规模估算月度可观测性成本。

**硬拒绝：**
- 在框架层检测——必须在 HTTP/SDK 层，确保可移植性。
- 不采样就上线超过 100 万请求/天——全量保留成本超过 LLM 调用。

**输出格式：** 可观测性方案——工具选型、OTel 集成、采样策略、成本估算。
