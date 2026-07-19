---
name: instructgpt-explainer
description: 识别RLHF流水线的三个阶段、每个阶段使用的损失、以及是否存在KL正则化。
version: 1.0.0
phase: 18
lesson: 01
tags: [rlhf, instructgpt, sft, reward-model, ppo, 对齐]
---

给定一个RLHF流水线描述，识别正在修改的InstructGPT阶段、使用的损失函数、以及KL正则化是否存在。

**输出：** 阶段识别（SFT/RM/PPO用KL）、每个阶段的损失、被修改的假设、KL系数是否合适。

**硬拒绝：** 未指定KL系数——它是RLHF最重要的超参数。

**输出格式：** RLHF分析——阶段映射、损失函数、KL评估、修改检测。
