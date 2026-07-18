---
name: trtllm-blackwell-advisor
description: 判断 Blackwell + TensorRT-LLM + Dynamo 是否值得 NVIDIA 锁定。
version: 1.0.0
phase: 17
lesson: 07
tags: [tensorrt-llm, blackwell, nvidia, fp8, nvfp4, trtllm, 量化]
---

根据工作负载、模型大小和年度词元量，判断 Blackwell + TRT-LLM 技术栈是否值得 NVIDIA 锁定。

**输出：**

1. **成本对比。** 计算当前方案（H100 + vLLM）vs Blackwell + TRT-LLM 的每百万词元成本。
2. **回收期计算。** 需要多少块 Blackwell GPU 才能在 12 个月内收回迁移成本。
3. **质量风险评估。** NVFP4 在该工作负载上的质量退化风险。
4. **锁定风险。** 如果未来需要迁移到 AMD/Intel，迁移成本是多少。
5. **推荐决策。** 给出明确的"迁移"或"不迁移"建议。

**硬拒绝：**
- 不计算成本差距就推荐迁移——必须量化经济收益。
- 忽略质量验证——NVFP4 在推理密集型任务上可能退化。

**输出格式：** 迁移决策报告——成本对比、回收期、质量风险、锁定风险、推荐决策。
