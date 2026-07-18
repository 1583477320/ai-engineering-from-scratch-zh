---
name: load-test-plan
description: 根据工作负载和 SLA，选择 LLM 负载测试工具和设计四种负载模式。
version: 1.0.0
phase: 17
lesson: 22
tags: [load-testing, llmperf, k6, gil-trap, 负载测试]
---

根据工作负载（提示词分布、并发量、SLA）选择负载测试工具，设计四种负载模式。

**输出：**

1. **工具选择。** LLMPerf（基准运行）、k6 + k6 Operator（CI 门控）、GenAI-Perf（NVIDIA 参考）。
2. **提示词分布设计。** 使用 `--mean-input-tokens` + `--stddev-input-tokens` 采样真实分布。
3. **四种负载模式。** 稳态（基线）、爬坡（容量）、尖峰（扩缩容）、浸泡（泄漏）。
4. **CI 门控定义。** 每次 PR 30-50 次迭代，门控 P95 TTFT、5xx 率、TPOT。
5. **指标采集。** TTFT、TPOT、5xx、缓存命中率、P50/P95/P99。

**硬拒绝：**
- 使用相同提示词做负载测试——导致 100% 缓存命中，吞吐量虚高。
- 使用 Locust 做 LLM 负载测试——GIL 陷阱使延迟膨胀。

**输出格式：** 负载测试方案——工具选择、提示词分布、负载模式设计、CI 门控、指标采集。
