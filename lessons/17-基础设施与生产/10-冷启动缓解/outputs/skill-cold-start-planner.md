---
name: cold-start-planner
description: 根据SLA、模型大小和流量形状，选择冷启动缓解方案组合。
version: 1.0.0
phase: 17
lesson: 10
tags: [cold-start, warm-pool, model-streamer, gpu-snapshot, serverless-llm]
---

根据SLA、模型大小和流量形状，选择冷启动缓解方案组合。

**输出：**

1. **冷启动时间分解。** 计算各阶段时间（节点供应+镜像拉取+权重加载+引擎初始化+首次前向）。
2. **缓解方案选择。** 根据SLA阈值选择方案组合：TTFT P99 < 60s→必须热池；< 5min→Bottlerocket+Streamer；< 30s→+GPU快照。
3. **热池分层策略。** 交互路径min_workers=1-2，批处理路径缩到零。
4. **成本估算。** 计算热池的空闲GPU小时数和月度成本。
5. **监控方案。** 部署后监控冷启动时间、热池利用率、请求超时率。

**硬拒绝：**
- 不测量冷启动时间分布就选择缓解方案。
- 所有路径都用min_workers=1。

**输出格式：** 冷启动规划报告——时间分解、方案选择、热池策略、成本估算、监控配置。
