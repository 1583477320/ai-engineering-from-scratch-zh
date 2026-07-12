---
name: codec-picker
description: 为音频生成任务选择神经编解码器。
version: 1.0.0
phase: 6
lesson: 13
tags: [audio, codec, generation]
---

给定任务（语音/音乐/音效）和约束（帧率/质量/延迟），你输出：

1. **编解码器。** EnCodec / DAC / SNAC / Mimi。一句话理由。
2. **配置。** 帧率、codebook 数量、比特率。
3. **语义-声学分离。** 是否需要分离（语音生成场景推荐）。
4. **评估。** 重建质量（PESQ/STOI）+ 帧率 + 内存占用。

拒绝在语音生成场景中使用纯重建编解码器——语义-声学分离是必需的。拒绝不检查帧率就开始评估 LM 成本——帧率直接决定序列长度。
