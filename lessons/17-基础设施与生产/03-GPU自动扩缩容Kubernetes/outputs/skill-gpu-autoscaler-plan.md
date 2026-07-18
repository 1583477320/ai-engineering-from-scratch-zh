---
name: gpu-autoscaler-plan
description: 为基于 Kubernetes 的 LLM 在线服务集群设计三层 GPU 自动扩缩容方案（Karpenter + KAI Scheduler + 应用层信号）。诊断 DCGM_FI_DEV_GPU_UTIL 陷阱和部分分配故障。
version: 1.0.0
phase: 17
lesson: 03
tags: [karpenter, kai-scheduler, gpu-autoscaling, gang-scheduling, kubernetes]
---

根据集群拓扑（节点类型、GPU 数量、NVLink 拓扑）、工作负载画像（模型大小、并发量、SLO）和成本预算，生成三层扩缩容方案。

**输出：**

1. **第 1 层：节点供应（Karpenter）。** 配置 NodePool，指定 instance-type 要求、`consolidationPolicy: WhenEmpty`（不驱逐运行中作业）、`consolidateAfter: 1h`。
2. **第 2 层：组调度（KAI Scheduler）。** 为多 GPU 作业配置组调度，确保原子分配。指定拓扑感知约束（NVLink 域内）。
3. **第 3 层：应用层信号。** 选择正确的扩缩信号：prefill 端用队列深度，decode 端用 KV 缓存利用率。不用 DCGM_FI_DEV_GPU_UTIL。
4. **冷启动缓解。** 为 SLO 关键路径配置热池（min_workers=1）。
5. **分离式 prefill/decode 扩缩。** 如果使用分离式架构，为 prefill 和 decode Pod 配置独立的 HPA。

**硬拒绝：**
- 使用 DCGM_FI_DEV_GPU_UTIL 作为 HPA 信号——它是占空比，不是负载信号。
- 使用 WhenEmptyOrUnderutilized 合并策略——它会驱逐运行中的推理作业。

**输出格式：** 三层扩缩容方案——Karpenter 配置、KAI 调度器配置、应用层 HPA 配置、冷启动策略、监控指标。
