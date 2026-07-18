---
name: eagle3-rollout
description: 生成分阶段的 EAGLE-3 投机解码落地计划，在真实流量上测量接受率 α 后再投产。
version: 1.0.0
phase: 17
lesson: 05
tags: [eagle3, speculative-decoding, acceptance-rate, vllm, 投机解码]
---

根据目标模型、流量分布描述和并发目标，生成分阶段的 EAGLE-3 落地计划。

**输出：**

1. **基线测量。** 不开投机解码，测量 TTFT、ITL、吞吐量。
2. **启用配置。** 通过 vLLM `speculative_config` 启用 EAGLE-3。
3. **测量 α。** 在生产流量分布上测量接受率 α。α < 0.55 就禁用或训练领域特定草稿头。
4. **门控条件。** 确认 P99 ITL 没有变差后才全量发布。
5. **监控。** 上线后持续监控 α、P99 ITL、吞吐量。

**硬拒绝：**
- 不测量 α 就全量发布——α < 0.55 是生产反模式。
- 只看 P50 ITL——必须看 P99。

**输出格式：** 分阶段落地计划——基线→启用→测量→门控→发布→监控。
