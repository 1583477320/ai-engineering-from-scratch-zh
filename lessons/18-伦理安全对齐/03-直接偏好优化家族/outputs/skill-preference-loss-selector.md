---
name: preference-loss-selector
description: 根据数据集统计（配对vs非配对、偏好强度、长度分布）推荐偏好损失函数。
version: 1.0.0
phase: 18
lesson: 03
tags: [dpo, ip o, kto, simpo, orpo, bpo, 偏好优化]
---

根据数据集统计和部署目标（单阶段或两阶段），推荐DPO家族中的偏好损失函数。

**输出：** 数据集分析（配对/非配对、偏好强度分布、长度分布）、损失函数推荐（DPO/IPO/KTO/SimPO/ORPO/BPO）、已知故障模式警示。

**硬拒绝：** 不测试就选DPO——每任务应测试所有变体。

**输出格式：** 偏好损失选型——数据集分析、推荐损失、理由、已知故障模式。
