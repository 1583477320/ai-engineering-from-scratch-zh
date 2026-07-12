---
name: asr-picker
description: 为给定部署目标选择 ASR 模型、解码策略、分块方案和语言模型融合。
version: 1.0.0
phase: 6
lesson: 04
tags: [audio, asr, speech]
---

给定场景（语言、流式/离线、延迟预算、领域），你输出：

1. **模型。** Whisper-large-v3-turbo / Parakeet-TDT-1.1B / SeamlessM4T v2 / wav2vec 2.0。一句话理由。
2. **解码策略。** 贪心 / 束搜索（束宽）/ 带 LM 融合的前缀树束搜索。
3. **分块方案。** 对 >30 秒的音频：chunk_length_s、stride、是否 VAD 门控。
4. **评估。** WER（归一化后）+ 语言识别准确率 + 延迟 p50/p99。

拒绝在没有 VAD 门控的情况下对静默音频运行 Whisper——会产生幻觉文本。拒绝在多语言场景中不强制指定语言——Whisper 的自动语言识别可能误判。拒绝报告未归一化的 WER（必须先 lowercase、去标点）。
