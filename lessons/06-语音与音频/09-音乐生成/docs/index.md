# 音乐生成——MusicGen、Stable Audio、Suno 与版权地震

> 2026 音乐生成：Suno v5 和 Udio v4 主导商业领域；MusicGen、Stable Audio Open、ACE-Step 引领开源。技术问题基本解决。版权问题（Warner Music 5 亿美元和解）在 2025-2026 年重塑了整个领域。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 06 · 02（频谱图与梅尔）、阶段 04 · 10（扩散模型）
**时间：** ~75 分钟

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
3. **条件/可控生成。** 延长现有片段、重生成桥段、换风格、分轨分离。Udio 的 inpainting + stem separation 是 2026 年的对标功能

---

## 2. 概念

### 2.1 Token LM over 神经编解码器 token

Meta 的 **MusicGen**（2023，MIT 许可）：以文本/旋律嵌入为条件，自回归预测 EnCodec token（32kHz，4 codebook），用 EnCodec 解码。300M - 3.3B 参数。强基线；超过 30 秒时质量下降。

**ACE-Step**（开源，2026 年 4 月发布 4B XL）扩展支持完整歌曲歌词条件化生成。

### 2.2 潜在扩散

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：压缩音频上的潜在扩散。擅长循环、音效、氛围纹理。不擅长结构化完整歌曲。

### 2.3 混合架构（生产系统）——Suno、Udio

闭源权重。可能是 AR codec LM + 扩散声码器 + 专用人声/鼓/旋律头。Suno v5（2026）ELO 1293 质量领先。Udio v4 增加 inpainting + 分轨分离（贝斯、鼓、人声独立下载）。

### 2.4 评估指标

- **FAD（Fréchet Audio Distance）：** VGGish/PANNs 嵌入级距离。越低越好
- **Musicality：** 人类偏好。Suno v5 ELO 1293 领先
- **文本-音频对齐：** CLAP 分数

---

## 3. 从零实现

### Step 1: 符号级和弦/鼓点生成

```python
MAJOR_KEYS = {
    "C": ["C", "Dm", "Em", "F", "G", "Am", "Bdim"],
    "G": ["G", "Am", "Bm", "C", "D", "Em", "F#dim"],
    "D": ["D", "Em", "F#m", "G", "A", "Bm", "C#dim"],
}
COMMON_PROGRESSIONS = {"pop": [1,5,6,4], "ballad": [1,6,4,5], "rock": [1,4,5,1]}

def chord_progression(key, genre, bars=8):
    scale = MAJOR_KEYS[key]
    pat = COMMON_PROGRESSIONS.get(genre, COMMON_PROGRESSIONS["pop"])
    return [scale[i - 1] for i in (pat * (bars // len(pat) + 1))[:bars]]
```

这演示了"随时间变化的 token"的核心思想。真正的音乐生成用神经编解码器 token——同样的理念，但 token 粒度是声学级而非符号级。

完整代码见 `code/main.py`。

---

## 4. 工业工具

| 模型 | 参数 | 时长 | 人声 | 许可证 |
|---|---|---|---|---|
| MusicGen-large | 3.3B | 30s | 否 | MIT |
| Stable Audio Open | 1.2B | 47s | 否 | 非商用 |
| ACE-Step XL | 4B | 2min+ | 有 | Apache-2.0 |
| YuE | 7B | 2min+ | 有 | Apache-2.0 |
| Suno v5 | 未知 | 4min | 有 | 商业 |
| Udio v4 | 未知 | 4min | 有+分轨 | 商业 |

---

## 5. 常见错误

### 错误 1：忽略版权合规

**现象：** 用版权音乐训练模型后发布——Warner/UMG 索赔 5 亿美元。

**修复：** 训练数据必须有合法授权。生成的音乐必须加水印 + 元数据披露（AI 生成标记）。2025-2026 年的版权和解已经定义了安全区。

---

## 🔑 关键术语

| 术语 | 实际含义 |
|---|---|
| MusicGen | Meta 开源音乐生成模型，在 EnCodec token 上做自回归。纯器乐，MIT 许可 |
| Stable Audio | 扩散模型做音乐——擅长循环、氛围，不擅长结构化歌曲 |
| ACE-Step | 开源歌词条件化音乐生成。Apache-2.0。开源社区最接近 Suno 的方案 |
| FAD | 音频距离指标——嵌入级距离，越低越好 |
| 分轨分离 | 将歌曲拆分为独立音轨（人声、贝斯、鼓等）。Udio v4 的核心功能 |

---

## ✏️ 练习

1. 【实验】用 MusicGen API 生成 4 种不同风格的 15 秒片段。主观评估音乐质量和文本对齐度
2. 【思考】在"AI 生成音乐"和"人类创作音乐"之间，版权边界应该如何划定？设计一个判断框架

---

## 📖 参考资料

1. [论文] Copet et al. "Simple and Controllable Music Generation (MusicGen)". NeurIPS, 2023. https://arxiv.org/abs/2306.05284
2. [论文] Evans et al. "Fast Diffusion-based Text-to-Audio Decoding (Stable Audio)". 2023.
3. [项目] ACE-Step. https://github.com/ace-step/ACE-Step — 开源歌词条件化音乐生成
4. [报道] Warner Music / UMG 和解（2025-2026）— 定义 AI 音乐版权安全区

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、法律风险分析、工程最佳实践均为原创内容。
