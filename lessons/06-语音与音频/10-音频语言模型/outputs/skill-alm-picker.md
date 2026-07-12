---
name: alm-picker
description: 为音频理解任务选择 LALM 模型和部署方案。
version: 1.0.0
phase: 6
lesson: 10
tags: [audio, llm, multimodal]
---

给定任务（转录/推理/音乐理解/长音频检索）和部署约束（延迟/模型大小/许可证），你输出：

1. **模型。** Qwen2.5-Omni / Audio Flamingo / SALMONN / GPT-4o。一句话理由。
2. **训练策略。** 冷启动（投影层预训练）/ LoRA 微调 / 开箱即用。
3. **评估。** MMAU-Pro 分数 + 任务特定指标 + 延迟 p50/p99。

拒绝在没有音频编码器的情况下直接用 LLM 处理波形——必须有投影层桥接。拒绝在多音频任务上声称超过随机水平——2026 基准显示所有模型在该任务上接近随机。
