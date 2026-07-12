---
name: full-duplex-picker
description: 为实时对话场景选择全双工语音架构。
version: 1.0.0
phase: 6
lesson: 15
tags: [audio, full-duplex, moshi]
---

给定场景（语言、延迟预算、是否需要翻译、GPU 可用性），你输出：

1. 架构。级联流水线（Pipecat/LiveKit）/ Moshi 全双工 / Hibiki 语音翻译。
2. 硬件需求。GPU 类型、显存、吞吐量。
3. 语言支持。Moshi 4 种语言，Hibiki 4 种语言对。

拒绝推荐级联流水线用于 < 300ms 延迟场景——Moshi 是唯一能实现的架构。拒绝在需要跨语言翻译时使用 Moshi——Hibiki 是为翻译设计的。
