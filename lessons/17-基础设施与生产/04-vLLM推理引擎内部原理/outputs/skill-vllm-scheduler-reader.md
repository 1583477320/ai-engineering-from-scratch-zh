---
name: vllm-scheduler-reader
description: 诊断 vLLM 在线服务配置，读取调度器级别的参数，识别 PagedAttention、连续批处理和 Chunked Prefill 中哪个是瓶颈。
version: 1.0.0
phase: 17
lesson: 04
tags: [vllm, pagedattention, continuous-batching, chunked-prefill, 调度器]
---

根据 vLLM 服务配置（批大小、KV 内存利用率、Chunked Prefill 大小、投机解码配置、并发量），生成调度器诊断。

**输出：**

1. **瓶颈识别。** 分析三个默认设置（PagedAttention、连续批处理、Chunked Prefill）的配置状态，指出哪个是瓶颈。
2. **TTFT 分析。** 如果 TTFT P99 高，检查 Chunked Prefill 是否开启、块大小是否合适。
3. **ITL 分析。** 如果 P99 ITL 高，检查连续批处理是否开启、批大小限制。
4. **内存分析。** 检查 `--gpu-memory-utilization` 配置是否合理。
5. **调优建议。** 给出具体的参数调整建议。

**硬拒绝：**
- 建议关闭 PagedAttention——它是 vLLM 唯一的分配器。
- 不检查 Chunked Prefill 就诊断 ITL 问题。

**输出格式：** 调度器诊断报告——瓶颈识别、TTFT/ITL 分析、内存分析、调优建议。
