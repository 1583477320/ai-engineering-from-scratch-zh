---
name: whisper-finetune
description: 为给定任务选择 Whisper 版本、分块策略和 LoRA 微调方案。
version: 1.0.0
phase: 6
lesson: 05
tags: [audio, asr, whisper]
---

给定场景（语言、领域、延迟预算、数据量），你输出：

1. **模型版本。** Tiny/Base/Small/Medium/Large-v3/Large-v3-turbo。理由基于延迟、准确率和部署约束。
2. **分块策略。** 对 >30s 音频：chunk_length_s、stride_s、是否 VAD 门控。短音频直接 padding。
3. **微调策略。** 全量 fine-tune（<100小时领域数据）/ LoRA r=16 q_proj,v_proj（推荐）/ 冻结编码器。
4. **评估。** WER（归一化后）+ 每语言准确率 + 延迟 p50/p99。

拒绝在没有 VAD 门控的情况下对静默音频运行 Whisper——会产生幻觉文本。拒绝在多语言场景中不强制指定语言——Whisper 的自动语言识别在短句上不稳定。拒绝在没有验证 LoRA 可训练参数量（通常是原模型的 1/100+）的情况下声称'高效微调'。
