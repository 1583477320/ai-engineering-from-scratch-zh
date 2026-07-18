---
name: vllm-stack-decider
description: 根据工作负载形态和 vLLM 部署，判断使用原生 CPU 卸载、LMCache 还是不启用 KV 卸载。
version: 1.0.0
phase: 17
lesson: 18
tags: [vllm, lmcache, kv-offloading, production-stack, KV卸载]
---

根据工作负载形态和 vLLM 部署，判断 KV 卸载策略。

**输出：**

1. **HBM 压力评估。** 计算 KV 缓存占用 vs HBM 容量。是否经常超过 80%？
2. **跨引擎复用评估。** 是否有共享前缀（RAG 共同系统提示词、多租户共享模板）？
3. **决策：** 原生 CPU 卸载（单引擎 HBM 压力）vs LMCache（多引擎共享前缀）vs 不启用（低占用+无共享）。
4. **LMCache 部署方案。** 如需 LMCache，设计高可用方案（副本、回退到原生）。
5. **监控指标。** 抢占率、HBM 利用率、KV 缓存命中率。

**硬拒绝：**
- KV 缓存 < 50% HBM 就启用 LMCache——开销大于收益。
- 单租户无共享前缀场景使用 LMCache——没有跨引擎复用。

**输出格式：** KV 卸载决策——HBM 评估、复用评估、决策、部署方案、监控指标。
