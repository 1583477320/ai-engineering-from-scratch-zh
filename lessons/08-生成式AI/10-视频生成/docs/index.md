# 视频生成

> 图像是二维的；视频是三维的（时间+空间）。扩散模型加上时间维度——变成视频扩散——就是 Sora、Kling、Runway 的核心技术。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 08 · 07（潜在扩散）| **时间：** ~45 分钟

---

## 🎯 学习目标

- [ ] 说明视频扩散与图像扩散的关键差异——时间维度、帧间一致性、计算开销
- [ ] 区分 Sora、Kling、Runway Gen-3 的架构差异
- [ ] 解释 2026 年视频生成的两个前沿——Consistency Model 和 Rectified Flow

---

## 1. 问题

图像扩散在 512×512 上做 20-50 步已经很快了。但视频是 3D 的（H×W×T）——T=24 帧/秒 × 10 秒 = 240 帧——计算量增长了 240 倍。而且视频有时间一致性要求——连续帧必须看起来连贯，不能闪烁。

---

## 2. 概念

### 2.1 视频扩散的三种架构

| 架构 | 核心思想 | 代表 |
|---|---|---|
| **3D U-Net** | 将时间维度当作第三个空间维度 | Video Diffusion Model |
| **帧级扩散+时序模块** | 每帧独立扩散 + Transformer 时序建模 | Sora, Kling, Runway Gen-3 |
| **一致性模型** | 直接从噪声映射到视频——一步或几步 | Consistency Model（2026） |

### 2.2 Sora 的核心创新

- **时空 patches：** 将视频切分为固定大小的时空块——类似 ViT 的图像 patch，但扩展到时间维度
- **DiT（Diffusion Transformer）：** 用 Transformer 替代 U-Net——每个 patch 同时关注空间和时间邻居
- **可变时长：** 支持 5-60 秒的可变长度视频

### 2.3 2026 年的突破

- **Consistency Model：** 从扩散模型蒸馏出一步生成——1 步出视频
- **Rectified Flow：** Flow Matching 的加速版本——2-5 步生成高质量视频
- **Sora / Kling / Runway：** 商业级视频生成——1080p，5-60 秒

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| 时空 patches | 将视频切分为固定大小的时空块——ViT 的视频扩展 |
| DiT | Diffusion Transformer——用 Transformer 替代 U-Net 的扩散模型 |
| Consistency Model | 直接从噪声映射到输出——1 步或几步生成 |
| Rectified Flow | Flow Matching 的加速版——2-5 步生成 |

---

## 📚 小结

视频扩散 = 图像扩散 + 时间维度。三大架构：3D U-Net、帧级扩散+时序模块、一致性模型。Sora 用 DiT + 时空 patches 实现了 5-60 秒的高质量视频生成。2026 年的前沿是一致性模型——从扩散骨干蒸馏出一步生成，大幅降低推理成本。

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
