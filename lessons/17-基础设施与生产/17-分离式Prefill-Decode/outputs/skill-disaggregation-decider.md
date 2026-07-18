---
name: disaggregation-decider
description: 根据工作负载和集群，判断是否采用分离式 prefill/decode。
version: 1.0.0
phase: 17
lesson: 17
tags: [disaggregated-serving, dynamo, llm-d, 分离式]
---

根据工作负载（提示词长度分布、输出长度、并发量）和集群配置，判断是否采用分离式 prefill/decode。

**输出：**

1. **工作负载分析。** 提示词长度分布、输出长度分布、prefill:decode 时间比。
2. **分离式收益估算。** 共置 vs 分离式的吞吐量和成本对比。
3. **Dynamo vs llm-d 决策。** 根据运维偏好选择（托管编排 vs K8s 原生）。
4. **池配置。** prefill 池和 decode 池的 GPU 类型、数量、扩缩信号。
5. **实施路径。** 分阶段迁移——先在非关键路径测试。

**硬拒绝：**
- P50 提示词长度 < 512 词元就采用分离式——传输税超过了收益。
- 没有 RDMA 网络就部署分离式——TCP 传输延迟太高。

**输出格式：** 分离式决策报告——工作负载分析、收益估算、技术选型、池配置、实施路径。
