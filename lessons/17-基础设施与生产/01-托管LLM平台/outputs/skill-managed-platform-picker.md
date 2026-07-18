---
name: managed-platform-picker
description: 根据工作负载、SLA 和合规要求，选择托管 LLM 平台（Bedrock、Azure OpenAI、Vertex AI）和第二家备用平台，并生成 FinOps 检测方案。
version: 1.0.0
phase: 17
lesson: 01
tags: [bedrock, azure-openai, vertex-ai, ptu, finops, 托管平台]
---

根据工作负载画像（所需模型、月度词元数、TTFT SLA P50/P99、合规约束、现有云足迹），生成平台推荐。

**输出：**

1. **主平台。** 说明平台名称、覆盖的具体模型、按需还是预置吞吐（PTU/Provisioned Throughput）更合适。引用盈亏平衡数学（PTU 在约 40-60% 持续利用率时划算）。
2. **备用平台。** 说明两家供应商最低要求的备用方案。论证配对合理性——冗余必须覆盖模型重叠和区域重叠。
3. **FinOps 检测。** 说明第一天需要启用什么：Bedrock Application Inference Profiles、Azure 作用域 + PTU 预留作为成本对象、Vertex 项目制 + BigQuery 计费导出。
4. **SLA 检查。** 对比目标 TTFT P99 与公开基准（Azure OpenAI PTU ≈ 50ms P50；Bedrock 按需 ≈ 75ms P50）。如果 SLA 比按需能提供的更严格，要求 PTU。
5. **合规检查。** 验证 BAA、SOC 2 Type II、HIPAA、欧盟数据驻留。三家都满足基线，但数据保留策略和滥用监控 opt-out 有差异。
6. **迁移路径。** 说明一个本周可执行的可逆步骤和一个长期步骤。

**硬拒绝：**
- 不命名备用平台就推荐单一平台——坚持两家供应商最低要求。
- 没有利用率估计就选择 PTU——要求持续利用率数据。
- 列出归因需求时忽略 Bedrock Application Inference Profiles——它们是最干净的原生方案。

**输出格式：** 一页决策——主平台、备用平台、PTU vs 按需、检测清单、SLA/合规验证、两个迁移步骤。最后指出一个将捕获偏离计划的指标。
