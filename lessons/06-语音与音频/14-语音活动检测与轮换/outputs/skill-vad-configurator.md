---
name: vad-configurator
description: 配置 VAD 和端点检测参数。
version: 1.0.0
phase: 6
lesson: 14
tags: [audio, vad, real-time]
---

给定应用（语音助手/电话/会议）和环境（噪声/安静），你输出：

1. VAD 选择。Silero VAD / WebRTC VAD / Pyannote VAD。基于延迟和准确率要求。
2. 参数配置。阈值、最小语音时长、静默挂起、预滚动缓冲。
3. 端点检测策略。固定静默计数 vs 语义端点检测模型。
4. Flush trick 实现。是否需要 STT flush 信号，延迟影响分析。

拒绝在有明显背景噪声的环境中使用能量门控作为唯一 VAD——它会不断误触发。拒绝静默挂起 < 300ms——会过早切断用户的话。
