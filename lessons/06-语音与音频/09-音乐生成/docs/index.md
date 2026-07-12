# 音乐生成——MusicGen、Stable Audio、Suno 与版权地震

> 2026 音乐生成：Suno v5 和 Udio v4 主导商业；MusicGen、Stable Audio Open 和 ACE-Step 引领开源。技术问题大部分已解决。法律问题（Warner Music 5 亿美元和解）在 2025-2026 年重塑了整个领域。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 06 · 02（频谱图）、阶段 04 · 10（扩散模型） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分音乐生成的三大技术路线——神经编解码 LM、潜在扩散、混合架构
- [ ] 理解 MusicGen 的工作原理——在 EnCodec token 上条件化自回归预测
- [ ] 说明 2025-2026 年版权和解对音乐生成系统的影响——水印和元数据披露是法律要求

---

## 1. 问题

文本 → 30 秒到 4 分钟的音乐片段，包含歌词、人声和结构。三个子问题：

1. **器乐生成。** "lo-fi hip-hop drums with warm keys" → 音频。MusicGen、Stable Audio、AudioLDM
2. **歌曲生成（含人声+歌词）。** "乡村歌曲，关于德州雨夜" → 完整歌曲。Suno、Udio、YuE、ACE-Step
3. **条件/可控生成。** 延长现有片段、重生成桥段、换风格、分轨分离。Udio 的 inpainting + stem separation 是 2026 年对标功能

---

## 2. 概念

### 2.1 Token LM over 神经编解码器 token

Meta 的 **MusicGen**（2023，MIT）：以文本/旋律嵌入为条件，自回归预测 EnCodec token（32kHz，4 codebook），用 EnCodec 解码。300M-3.3B 参数。**强基线；超过 30 秒时变差。**

**ACE-Step**（开源，2026 年 4 月发布 4B XL 版本）扩展了这个方向以支持完整歌曲的歌词条件化生成。

### 2.2 潜在扩散

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：压缩音频上的潜在扩散。擅长循环、音效设计、氛围纹理。不擅长结构化的完整歌曲。

### 2.3 混合架构（生产系统）——Suno、Udio

闭源权重。可能是 AR codec LM + 扩散声码器 + 专用人声/鼓/旋律头。Suno v5（2026）是 ELO 1293 质量领先者。Udio v4 增加了 inpainting + 分轨分离（贝斯、鼓、人声独立下载）。

### 2.4 2026 评估指标

- **FAD（Fréchet Audio Distance）。** VGGish/PANNs 特征的嵌入级距离。越低越好
- **Musicality（主观）。** 人类偏好。Suno v5 ELO 1293 领先
- **文本-音频对齐。** CLAP 分数

### 2.5 2026 模型图谱

| 模型 | 参数 | 时长 | 人声 | 许可证 |
|---|---|---|---|---|
| MusicGen-large | 3.3B | 30s | 无 | MIT |
| Stable Audio Open | 1.2B | 47s | 无 | 非商用 |
| ACE-Step XL | 4B | 2min+ | 有 | Apache-2.0 |
| Suno v5 | 未知 | 4min | 有 | 商业 |

---

## 3. 常见错误

### 错误 1：忽略版权合规

**现象：** 用版权音乐训练模型后发布——Warner/UMG 索赔 5 亿美元。

**修复：** 训练数据必须有合法授权。生成的音乐必须加水印 + 元数据披露（AI 生成标记）。

---

## 🔑 关键术语

| 术语 | 实际含义 |
|---|---|
| MusicGen | Meta 开源音乐生成模型，在 EnCodec token 上做自回归。纯器乐，MIT 许可 |
| Stable Audio | 扩散模型做音乐——擅长循环、氛围，不擅长结构化歌曲 |
| ACE-Step | 开源歌词条件化音乐生成，接近 Suno 质量。Apache-2.0 |
| FAD | 音频距离指标——嵌入级距离，越低越好 |
| 分轨分离 | 将歌曲拆分为独立音轨（人声、贝斯、鼓等）。Udio v4 的核心功能 |

---

## ✏️ 练习

1. 【实验】用 MusicGen API 生成 4 种不同风格的 15 秒片段。主观评估音乐质量和文本对齐度
2. 【思考】在"AI 生成音乐"和"人类创作音乐"之间，版权边界应该如何划定？设计一个判断框架

---

## 📖 参考资料

1. [论文] Copet et al. "Simple and Controllable Music Generation (MusicGen)". NeurIPS, 2023.
2. [论文] Evans et al. "Fast Diffusion-based Text-to-Audio Decoding (Stable Audio)". 2023.
3. [项目] ACE-Step. https://github.com/ace-step/ACE-Step — 开源歌词条件化音乐生成

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
