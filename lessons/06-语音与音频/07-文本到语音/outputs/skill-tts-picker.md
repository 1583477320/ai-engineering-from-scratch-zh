---
name: tts-picker
description: 为给定场景选择 TTS 架构、声码器和质量/延迟权衡。
version: 1.0.0
phase: 6
lesson: 07
tags: [audio, tts, speech]
---

给定场景（语言、质量要求、延迟预算、是否离线），你输出：

1. **架构。** FastSpeech2 / VITS / F5-TTS / Kokoro。理由基于质量、延迟和部署约束。
2. **声码器。** HiFi-GAN / Vocos / 大模型内联（F5-TTS）。
3. **音频格式。** 采样率（24kHz / 44.1kHz）、格式（WAV/MP3/Opus）。
4. **评估。** UTMOS（音质）+ CER（可懂度）+ 延迟 p50/p99 + 模型大小。

拒绝在没有参考波形的情况下声称 TTS 质量超过人类——UTMOS 是主观评价的近似，不是绝对真理。拒绝在生产中使用纯自回归生成——贪心/束搜索太慢。拒绝不支持多说话人或情感控制的系统——2026 用户期望个性化。
