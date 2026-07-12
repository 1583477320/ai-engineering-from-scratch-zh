# 流式语音到语音——Moshi、Hibiki 与全双工对话

> 2024-2026 重新定义了语音 AI。Moshi 发布了单模型同时听和说、200ms 延迟的全双工对话。Hibiki 实现了逐 chunk 的语音到语音翻译。两者都抛弃了 ASR → LLM → TTS 流水线，转向 Mimi 编解码器 token 上的统一全双工架构。这是新的参考设计。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 06 · 13（神经音频编解码）、阶段 06 · 11（实时音频）、阶段 7 · 05（Transformer）
**时间：** ~75 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述 Moshi 的双 Mimi 流 + 内心独白架构——理解为什么一个模型能同时听和说
- [ ] 解释深度 Transformer 的作用——8 个 codebook 不是并行预测的
- [ ] 说明 Hibiki 如何用 GRPO 强化学习消除词级对齐需求

---

## 1. 问题

每个用第 11+12 课搭建的语音助手都有一个结构性延迟下限：~300-500ms。VAD 触发 → STT 处理 → LLM 推理 → TTS 生成。每个阶段有自己的最小延迟。你可以调优和并行化，但流水线形态限制了你。

Moshi（Kyutai, 2024-2026）问了一个不同的问题：**如果没有流水线会怎样？** 如果一个模型接收音频输入并直接输出音频——持续进行，文本作为中间"内心独白"而非必须的阶段呢？

答案是**全双工语音到语音**。理论延迟 160ms（80ms Mimi 帧 + 80ms 声学延迟）。实际在单 L4 GPU 上 200ms。这是最佳流水线语音助手的一半。

---

## 2. 概念

### 2.1 Moshi 架构

**输入。** 两个 Mimi 编解码器流，均为 12.5Hz × 8 个 codebook：

- 流 1：用户音频（Mimi 编码，持续到达）
- 流 2：Moshi 自己的音频（正在生成）

**Transformer。** 7B 参数的 Temporal Transformer 在每 80ms 步骤处理：

1. 消费用户最新的 Mimi token
2. 消费 Moshi 自己最新的 Mimi token（已生成）
3. 生成下一个 Moshi 文本 token（内心独白）
4. 生成下一个 Moshi Mimi token（8 个 codebook，通过小型 Depth Transformer）

三个流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以在说话时听用户；可以在用户打断时打断自己；可以发出"嗯"等反馈信号而不打断主要话语。

**深度 Transformer。** 在一帧内，8 个 codebook 不是并行预测的——它们之间有依赖。一个 2 层的小型"深度 Transformer"在 80ms 内依次预测它们。这是 AR 编解码 LM 的标准因式分解（VALL-E、VibeVoice 也使用）。

### 2.2 为什么内心独白文本有帮助

没有显式文本时，模型必须隐式地在音频流中建模语言。Moshi 的洞察：强制输出文本 token 与音频并行——这相当于 Moshi 正在说的内容的转录。这改进了语义连贯性，使替换语言模型头更容易，并且免费提供了转录。

### 2.3 Hibiki：流式语音到语音翻译

相同架构，训练在翻译对上。源音频输入，目标语言音频输出，持续进行。Hibiki-Zero（2026 年 2 月）消除了词级对齐训练数据的需求——使用句子级数据 + GRPO 强化学习优化延迟。

---

## 3. 从零实现

### Moshi 核心循环概念

```python
def moshi_step(user_mimi_tokens, moshi_mimi_tokens, text_state):
    """每 80ms 步骤的核心循环。"""
    # 1. 消费用户最新的 Mimi token
    # 2. 消费 Moshi 自己最新的 Mimi token
    # 3. 生成下一个 Moshi 文本 token（内心独白）
    # 4. 生成下一个 Moshi Mimi token（通过深度 Transformer）
    return next_text_token, next_mimi_tokens
```

关键洞察：**文本流是 Moshi 说的话的转录——它使语义连贯性显式化。** 没有文本流，模型必须在声学 token 中隐式建模语言结构——这不可扩展。

---

## 4. 工业工具

| 工具 | 用途 | 许可 |
|---|---|---|
| Moshi（Kyutai） | 全双工语音对话 | CC-BY 4.0 |
| Hibiki（Kyutai） | 语音到语音翻译 | CC-BY 4.0 |
| Kyutai STT-1B/2.6B | 流式语音转文字 | CC-BY 4.0 |
| LiveKit + Deepgram | 级联流水线备选 | 商业 API |

---

## 5. 知识连线

- **阶段 06 · 13（音频编解码器）→** Moshi 和 Hibiki 都基于 Mimi 编解码器——12.5Hz 帧率、8 个 codebook、Codebook 0 从 WavLM 蒸馏
- **阶段 06 · 07（TTS）→** 传统 TTS 是级联流水线的最后一步；Moshi 将 ASR+LLM+TTS 合并到单个 Transformer，消除组件间切换
- **阶段 06 · 11（实时音频）→** 全双工架构将"听→理解→响应→说话"的延迟从流水线的 ~400ms 下限压缩到 200ms

---

## 6. 常见错误

### 错误 1：混淆全双工和半双工

**现象：** 认为 GPT-4o-realtime（320ms）是"全双工"——实际上它是流式但仍然是半双工（用户说话时模型不能同时说话）。

**修复：** 全双工 = 两个方向可以同时传输（人和 AI 同时说话）。半双工 = 同一时刻只有一个方向活跃（轮流说）。Moshi 是真正的全双工；GPT-4o-realtime 是流式半双工。

---

## 7. 面试考点

### Q1：Moshi 的"内心独白"文本流解决了什么问题？（难度：⭐⭐）

**参考答案：** 没有文本流时，模型必须在音频 token 中隐式地建模语言结构——这增加了训练难度（语言和声学特征耦合在同一 codebook 中），无法替换语言模型头（因为语言结构没有被显式编码），且无法免费获得转录。内心独白文本强制模型输出语言 token，分离了语言建模和声学建模——使语言模型头可以被替换，转录可以直接获取。

---

## 📚 小结

Moshi 架构——双 Mimi 流 + 内心独白文本流 + 7B Temporal Transformer——用 200ms 全双工延迟重新定义了语音 AI。核心洞察：文本流是 Moshi 说的话的转录——它使语义连贯性显式化，使替换语言模型头更容易。Hibiki 将相同架构用于语音翻译，用 GRPO 消除词级对齐需求。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 全双工选择提示词 | `outputs/skill-full-duplex-picker.md` | 按场景选择级联/全双工/语音翻译架构 |

---

## 🔑 关键术语

| 术语 | 实际含义 |
|---|---|
| 全双工 | 两个方向同时传输——AI 可以在听用户的同时说话 |
| 深度 Transformer | 预测 8 个 codebook 的内部依赖——2 层，80ms 内完成 |
| 内心独白 | Moshi 文本输出流——相当于说话内容的转录，使语义建模显式化 |
| Hibiki | Kyutai 的语音到语音翻译模型——相同架构，训练在翻译对上 |

---

## ✏️ 练习

1. 【理解】画出 Moshi 的三流架构图（用户音频流 + Moshi 音频流 + 文本流），标注每步操作。
2. 【实验】用 Kyutai 的开源模型尝试 Moshi 对话，记录延迟和响应质量。

---

## 📖 参考资料

1. [论文] Défossez et al. "Moshi: a speech-text foundation model for real-time dialogue". Kyutai, 2024. https://arxiv.org/abs/2410.07874
2. [论文] Défossez et al. "Hibiki: Efficient High-Quality Speech-to-Speech Translation". Kyutai, 2025.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
