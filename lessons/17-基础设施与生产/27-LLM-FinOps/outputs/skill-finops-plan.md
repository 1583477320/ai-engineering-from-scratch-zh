---
name: finops-plan
description: 根据产品和规模设计LLM支出的归因Schema和执法阶梯。
version: 1.0.0
phase: 17
lesson: 27
tags: [finops, cost-attribution, kill-switch, unit-economics, 成本管理]
---

根据产品类型（聊天/智能体/代码生成）和规模，设计LLM支出的归因Schema和执法阶梯。

**输出：** 归因维度设计（user/task/tenant）、词元层拆分（提示词/工具/记忆/响应）、执法阶梯配置（速率限制/支出上限/终止开关）、成本优化栈设计（缓存/批处理/路由/网关）。

**硬拒绝：**
- 不拆分词元层就开始优化——四层混在一个桶里无法定位成本来源。
- 用$/M词元做产品决策——必须绑定到业务结果。

**输出格式：** FinOps方案——归因Schema、执法阶梯、优化栈、仪表盘配置。
