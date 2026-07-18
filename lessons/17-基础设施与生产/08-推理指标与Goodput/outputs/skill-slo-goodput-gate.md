---
name: slo-goodput-gate
description: 生成 CI/CD 就绪的基准测试方案，用 goodput 而不是吞吐量来门控部署，包含 P50/P90/P99 百分位和工具声明。
version: 1.0.0
phase: 17
lesson: 08
tags: [goodput, ttft, tpot, itl, slo, 基准测试, llmperf, genai-perf]
---

根据工作负载和 SLO，生成 CI/CD 就绪的基准测试方案，用 goodput 门控部署。

**输出：**

1. **SLO 定义。** 定义 TTFT P99、TPOT P99、E2E P99 的阈值。
2. **Goodput 目标。** 设定 goodput 目标（如 ≥ 99%）。
3. **基准测试方案。** 指定工具（LLMPerf 或 GenAI-Perf）、参数（提示长度分布、输出长度、并发量）、运行次数。
4. **门控条件。** 好的部署：goodput ≥ 目标。差的部署：goodput < 目标或 P99 恶化。
5. **监控仪表盘。** 部署后持续监控 P50/P90/P99 和 goodput。

**硬拒绝：**
- 只用吞吐量门控——必须用 goodput。
- 不声明工具和定义——GenAI-Perf 和 LLMPerf 对 TPOT 的定义不同。

**输出格式：** 基准测试方案——SLO 定义、goodput 目标、测试参数、门控条件、监控配置。
