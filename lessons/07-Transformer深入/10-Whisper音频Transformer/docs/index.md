# 音频 Transformer——Whisper

> Whisper 用 680,000 小时的多语言弱监督数据训练了一个编码器-解码器 Transformer。一个架构、多个任务（转录/翻译/时间戳）、99 种语言、笔记本运行——2026 年的参考 ASR。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 06 · 04（ASR）、阶段 07 · 05（完整 Transformer）| **时间：** ~75 分钟

---

## 🎯 学习目标

- [ ] 理解 Whisper 如何将音频编码为 log-mel 频谱图再编码为 token
- [ ] 区分 Whisper 编码器（双向 Transformer）和解码器（自回归 Transformer）的作用
- [ ] 说明 Whisper 的"多任务"设计——同一个模型做转录、翻译和时间戳

---

## 1. 问题

ASR 的核心困难是**音频帧与文字之间没有对齐**——"okay" 可能是 200ms 也可能是 1200ms。CTC、RNN-T、Attention 是三种解法。Whisper 选择了 **Attention encoder-decoder**——编码器理解音频，解码器自回归生成文字。

Whisper 用 680,000 小时的弱监督数据训练（大部分来自互联网）——不需要精心标注的平行语料。这种"用噪声数据换规模"的范式是 LLM 时代的核心思路。

---

## 2. 概念

### 2.1 Whisper 架构

```
30 秒 log-mel 频谱图 (80 mel × 3000 帧)
    ↓ 两层 stride-2 卷积（降采样 4x）
    ↓ 24 层 Transformer 编码器
    ↓ 编码器隐藏状态
    ↓ 24 层 Transformer 解码器（因果掩码 + 交叉注意力）
    ↓ 自回归生成：BOS → token1 → token2 → ... → EOS
```

### 2.2 多任务——一个模型做三件事

| 任务 | 启动 token | 用途 |
|---|---|---|
| 转录（语言 X） | `<\|startoftranscript\|><\|lang_X\|><\|transcribe\|>` | 语音识别 |
| 翻译（X→英文） | `<\|startoftranscript\|><\|lang_X\|><\|translate\|>` | 语音翻译 |
| 带时间戳转录 | `<\|startoftranscript\|><\|lang_X\|><\|transcribe\|>` | 字幕生成 |

### 2.3 Whisper 模型家族

| 模型 | 层数 | 参数量 | WER（LibriSpeech test-clean） |
|---|---|---|---|
| tiny | 4 | 39M | 7.6% |
| base | 6 | 74M | 4.9% |
| small | 12 | 244M | 3.5% |
| medium | 24 | 769M | 2.7% |
| **large-v3** | 32 | 1.5B | **1.8%** |
| **large-v3-turbo** | 32 enc + 4 dec | 809M | **1.6%** |

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| Log-mel 频谱图 | Whisper 的输入——音频的频率-时间表示 |
| 弱监督训练 | 用不完美（噪声）的网络数据训练——不需要人工标注 |
| 束搜索解码 | 解码时保持 top-K 个候选序列，选择全局最优 |
| VAD 门控 | 在 Whisper 前加语音活动检测——过滤静默避免幻觉 |

---

## 📚 小结

Whisper = Log-mel 频谱图 + 24 层 Transformer 编码器 + 24 层解码器。680K 小时弱监督数据训练。一个模型做转录、翻译、时间戳。2026 年的基准：large-v3-turbo 1.6% WER（LibriSpeech）。关键实践：VAD 门控避免静默幻觉，分块处理长音频。

---

## ✏️ 练习

1. 用 Whisper 在 10 句中文语音上测试，记录 WER 和是否有静默幻觉
2. 对比 Whisper-tiny 和 Whisper-large-v3 的推理延迟和准确率——找出质量-速度的最优解

---

## 📖 参考资料

1. [论文] Radford et al. "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper). 2022.
2. [代码] openai/whisper. https://github.com/openai/whisper

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
