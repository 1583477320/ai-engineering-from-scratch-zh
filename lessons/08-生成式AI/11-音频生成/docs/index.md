# 音频生成

> 文本→语音（TTS）是最成熟的音频生成任务；文本→音乐、文本→音效是扩展。Diffusion 和自回归是两大技术路线。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 06 · 07（TTS）、阶段 06 · 13（音频编解码）| **时间：** ~45 分钟

---

## 🎯 学习目标

- [ ] 区分 TTS（语音）、音乐生成、音效生成三类任务的技术路线
- [ ] 解释 AudioCraft/MusicGen 的架构——EnCodec token + Transformer LM
- [ ] 说明 2026 年音频生成的两个前沿——F5-TTS（零样本 TTS）和 Flow Matching

---

## 1. 问题

音频生成包含三个子任务：TTS（文本→语音）、音乐生成（文本/和弦→音乐）、音效生成（文本→环境声）。它们共享相同的底层技术——神经编解码器（将音频压缩为离散 token）+ Transformer LM。

---

## 2. 概念

### 2.1 三类音频生成

| 任务 | 输入 | 输出 | 代表 |
|---|---|---|---|
| TTS | 文本 | 语音 | Kokoro, F5-TTS, VITS |
| 音乐生成 | 文本/和弦 | 音乐 | MusicGen, Stable Audio |
| 音效生成 | 文本 | 环境声/音效 | AudioLDM, AudioCraft |

### 2.2 AudioCraft 的架构

```
文本 → [文本编码器] → 条件向量 → [Transformer LM 在 EnCodec token 上] → [EnCodec 解码器] → 波形
```

### 2.3 2026 年的两个前沿

- **F5-TTS：** Flow Matching 用于语音合成——零样本 TTS（无需音素对齐），3-5 秒参考音频即可克隆
- **Flow Matching 用于音频：** 扩散模型的流匹配版本——更少的采样步数，更高的质量

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| EnCodec | Meta 的音频编解码器——将波形压缩为离散 token |
| AudioCraft | Meta 的音频生成框架——TTS + 音乐 + 音效 |
| F5-TTS | 零样本语音合成——Flow Matching 架构 |
| Flow Matching | 扩散模型的流匹配版本——更快的采样 |

---

## 📚 小结

音频生成 = 神经编解码器（EnCodec）+ Transformer LM（在 token 上生成）。TTS、音乐、音效共享同一底层技术。F5-TTS 用 Flow Matching 实现零样本 TTS。2026 年的趋势是统一的音频 token 范式——不同的生成任务使用同一个 LM 骨干。

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
