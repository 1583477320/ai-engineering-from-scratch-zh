---
name: radix-scheduler-advisor
description: 评估 SGLang 采用的适用性，为前缀密集型工作负载提供提示词排序处方。
version: 1.0.0
phase: 17
lesson: 06
tags: [sglang, radixattention, prefix-caching, prompt-ordering, 缓存感知]
---

根据工作负载描述（提示模板形状、检索模式、并发租户数），生成 SGLang 采用建议和提示词排序处方。

**输出：**

1. **前缀命中率评估。** 根据提示模板的重复程度，估算 RadixAttention 的缓存命中率。
2. **提示词排序处方。** 固定组件顺序——不可变部分（系统提示、工具、schema）放最前，检索上下文放中间，用户问题放最后。
3. **SGLang vs vLLM 决策。** 如果前缀命中率 > 50%，推荐 SGLang。如果 < 30%，推荐 vLLM。
4. **迁移路径。** 如果从 vLLM 迁移，第一步是固定提示词排序。

**硬拒绝：**
- 在前缀命中率 < 30% 的场景推荐 SGLang——收益不足以覆盖迁移成本。
- 不固定提示词排序就部署 SGLang——缓存命中率会很低。

**输出格式：** 前缀命中率估算、提示词排序处方、SGLang 采用建议、迁移路径。
