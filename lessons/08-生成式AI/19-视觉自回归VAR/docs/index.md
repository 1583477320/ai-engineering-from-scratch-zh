# 视觉自回归（VAR）

> 扩散模型是 2024 年的图像生成王者。但自回归在语言上统治了十年——如果图像也能像文本一样"下一个 token"生成呢？VAR 就是这个答案。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 08 · 06（DDPM）、阶段 07 · 07（GPT）| **时间：** ~45 分钟

---

## 🎯 学习目标

- [ ] 理解视觉自回归的核心——将图像离散化为 token 序列，然后用 Transformer 生成
- [ ] 解释图像 tokenizer 的设计——RQ-VAE 或 VQ-VAE 如何将图像压缩为离散 token
- [ ] 说明 2026 年视觉自回归的代表——Gemini、Chameleon、Llama-4

---

## 1. 问题

扩散模型在图像生成上取得了 SOTA——但自回归在语言上统治了十年。自回归的两个优势：(1) 自然支持条件生成（文本→图像就像文本→文本一样）；(2) 与 LLM 的训练范式一致——同一个训练框架处理文本、图像、视频。

**视觉自回归 = 图像 tokenizer + 自回归 Transformer。**

---

## 2. 概念

### 2.1 图像 tokenizer

将图像压缩为离散 token 序列。两种方法：

| 方法 | 原理 | 代表 |
|---|---|---|
| **RQ-VAE** | 残差量化——逐层量化残差 | Chameleon, Emu3 |
| **VQ-VAE** | 向量量化——最近邻匹配 | DALL-E 1 |

### 2.2 VAR 架构

```
图像 → [图像 tokenizer] → token 序列 → [Transformer LM] → token 序列 → [tokenizer 解码] → 图像
```

与 GPT 完全相同的框架——只是输入输出是图像 token 而非文本 token。

### 2.3 2026 年的代表

| 模型 | 类型 | 特点 |
|---|---|---|
| Gemini | 多模态 | 文本+图像统一 token——原生多模态 |
| Chameleon（Meta） | 多模态 | 统一 tokenizer——文本和图像使用同一个词表 |
| Llama-4 | 多模态 | 视觉 token + 文本 token 联合训练 |
| Emu3 | 视觉自回归 | 纯图像 token 自回归——与文本无关 |

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| 图像 tokenizer | 将图像压缩为离散 token 序列——RQ-VAE 或 VQ-VAE |
| 视觉自回归 | 将图像视为 token 序列，用 Transformer 预测下一个 token |
| RQ-VAE | 残差量化——逐层量化残差，将连续嵌入压缩为离散 codebook |
| 统一 tokenizer | Gemini/Chameleon 的做法——文本和图像使用同一个 token 词表 |

---

## 📚 小结

视觉自回归 = 图像 tokenizer（RQ-VAE/VQ-VAE）+ Transformer LM——将图像视为 token 序列，用 GPT 式的"下一个 token"生成。2026 年 Gemini、Chameleon、Llama-4 都采用这个范式。与扩散模型的竞争正在收敛——两者都在向"同一个 token 框架"靠拢。

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
