# 神经音频编解码器——EnCodec、SNAC、Mimi、DAC

> 2026 年音频生成几乎全是 token。EnCodec、SNAC、Mimi 和 DAC 将连续波形转换为 Transformer 可以预测的离散序列。语义-声学 token 分离——第一个 codebook 编码语义，其余编码声学细节——是 Transformer 音频领域最重要的架构转变。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 06 · 02（频谱图）、阶段 10 · 11（量化）、阶段 5 · 19（子词分词）
**时间：** ~60 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 RVQ（残差向量量化）如何将连续音频压缩为离散 token 序列
- [ ] 区分重建优先（EnCodec, DAC）和语义优先（Mimi, SpeechTokenizer）两种编解码器
- [ ] 说明为什么语义-声学 token 分离是 Moshi 和 Sesame CSM 工作的关键

---

## 1. 问题

语言模型处理离散 token。音频是连续的。如果你想用 LLM 风格的模型处理语音或音乐——MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——你首先需要一个**神经音频编解码器**：一个学习到的编码器，将音频离散化为一个小词汇表的 token，加上一个匹配的解码器重构波形。

两个家族：

1. **重建优先编解码器** —— EnCodec、DAC。优化感知音频质量。Token 是"声学的"——包含说话人身份、音色、背景噪声的一切
2. **语义优先编解码器** —— Mimi（Kyutai）、SpeechTokenizer。强制第一个 codebook 编码语言/语音内容（通常从 WavLM 蒸馏）。后续 codebook 是声学细节

**2024-2026 洞察：** 纯重建编解码器在从文本生成时会产生模糊语音。LLM 在 codec token 上必须同时学习语言结构和声学结构，这不可扩展。将它们分离——语义 codebook 0，声学 codebooks 1-N——是让 Moshi 和 Sesame CSM 工作的关键。

---

## 2. 概念

### 2.1 RVQ（残差向量量化）——核心技巧

不是用一个大 codebook（需要数百万个码），现代音频编解码器使用 **RVQ**：一系列小 codebook 的级联。第一个 codebook 量化编码器输出；第二个量化残差；以此类推。每个 codebook 1024 个码。8 个 codebook = 有效词汇量 1024^8 = 10^24。

推理时，解码器将每一帧所有选择的码相加以重构音频。

### 2.2 四个关键编解码器

| 编解码器 | 帧率 | 特点 |
|---|---|---|
| **EnCodec**（Meta, 2022） | 75 Hz | 基线。1D conv + transformer + 1D conv。24kHz，默认 4 codebook @ 1.5kbps |
| **DAC**（Descript, 2023） | 86 Hz | L2 归一化 codebook + 周期激活。最高重建保真度——12 codebook 时有时与原始语音不可区分。44.1kHz |
| **SNAC**（Hubert Siuzdak, 2024） | ~12 Hz（粗） | 多尺度 RVQ——粗 codebook 在低帧率操作，细 codebook 在高帧率。层级建模 |
| **Mimi**（Kyutai, 2024） | 12.5 Hz | 2026 游戏改变者。Codebook 0 **从 WavLM 蒸馏**——训练为预测 WavLM 语音内容特征。Codebook 1-7 是声学残差。此分离驱动 Moshi 和 Sesame CSM |

### 2.3 帧率对语言模型的重要性

| 编解码器 | 帧率 | 1s = N 帧 | 适用 |
|---|---|---|---|
| EnCodec-24k | 75 Hz | 75 | 音乐、通用音频 |
| DAC-44.1k | 86 Hz | 86 | 高保真音乐 |
| SNAC-24k（粗） | ~12 Hz | 12 | AR-LM 高效 |
| Mimi | 12.5 Hz | 12.5 | 流式语音 |

**更低帧率 = 更短序列 = 更快 LM。** Moshi 用 Mimi 的 12.5Hz 帧率——1 秒音频只生成 12.5 个 token，而非 EnCodec 的 75 个。

---

## 🔑 关键术语

| 术语 | 实际含义 |
|---|---|
| RVQ | 残差向量量化——用一系列小 codebook 级联量化：第一个量化原始信号，后续每个量化残差 |
| Codebook | 量化字典——每个含 1024 个码；编码器输入→最近码；解码器读码→重建信号 |
| 语义 token | 编解码器的第一个 codebook——从 WavLM 蒸馏，编码语言/语音内容，而非声学细节 |
| 声学 token | 编解码器后续 codebook——编码音色、说话人身份、背景噪声等声学信息 |
| 帧率 | 每秒帧数。更低 = 更短序列 = 更快 LM |

---

## ✏️ 练习

1. 【理解】画出 RVQ 的级联结构——8 个 codebook，每个 1024 码，输入→第一个 codebook→残差→第二个 codebook→...
2. 【实验】用 EnCodec 编码一段 3 秒音频到 4 个 codebook token。再用不同数量的 codebook（1, 2, 4, 8）解码，对比重建质量。

---

## 📖 参考资料

1. [论文] Défossez et al. "High Fidelity Neural Audio Compression (EnCodec)". 2022.
2. [论文] Zeghidour et al. "SoundStream: An End-to-End Neural Audio Codec". 2021.
3. [论文] Vitter et al. "Mimi: A Neural Audio Codec". Kyutai, 2024.
4. [论文] Kumar et al. "Descript Audio Codec (DAC)". 2023.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
