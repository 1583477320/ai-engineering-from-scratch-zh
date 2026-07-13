# Flow Matching 与 Rectified Flow

> 扩散模型需要 20-50 步采样。Flow Matching 用更直的路径——2-5 步就能达到相同质量。Rectified Flow 是 2026 年训练扩散骨干的主流方法。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 08 · 06（DDPM）、07（潜在扩散）| **时间：** ~60 分钟

---

## 🎯 学习目标

- [ ] 理解 Flow Matching 的核心——学习从噪声到数据的最优传输路径
- [ ] 解释为什么 Flow Matching 比 DDPM 快 4-10 倍——路径更直意味着更少采样步数
- [ ] 说明 2026 年的 Rectified Flow——如何在不损失质量的前提下将步数压缩到 1-4 步

---

## 1. 问题

DDPM 需要 1000 步采样。DDIM 将其压缩到 20-50 步。但这仍然太慢——20 步 × 每步 100ms = 2 秒。视频生成需要每帧都做一次——不可接受。

Flow Matching 问了一个不同的问题：**在噪声空间和数据空间之间，有没有一条更直的路径？**

---

## 2. 概念

### 2.1 Flow Matching vs 扩散

```
DDPM 路径：  噪声 ──曲折──→ 数据（20-50 步）
Flow Matching: 噪声 ──────→ 数据（2-5 步）
```

**核心差异：** DDPM 的路径是随机微分方程（SDE）——曲折。Flow Matching 学习从噪声到数据的**最优传输路径**——更直、更高效。

### 2.2 Rectified Flow

Flow Matching 的改进——训练一个"校正器"来消除路径中的不必要弯曲。

```
初始路径:  噪声 ──→ 数据（5 步，仍有弯曲）
Rectified: 噪声 ─→ 数据（2 步，几乎直线）
```

### 2.3 2026 年的采样步数演进

| 方法 | 步数 | 质量 | 速度 |
|---|---|---|---|
| DDPM | 1000 | 基准 | 慢 |
| DDIM | 20-50 | 接近基准 | 中 |
| Flow Matching | 2-5 | 接近基准 | 快 |
| Consistency Model | 1 | 略低 | 极快 |

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| Flow Matching | 学习从噪声到数据的最优传输路径——更直的路径=更少步数 |
| Rectified Flow | Flow Matching 的校正版本——路径更直→步数更少 |
| Consistency Model | 从扩散模型蒸馏出一步生成——质量略低于完整扩散 |
| 最优传输 | 从噪声分布到数据分布的"最短路径" |

---

## 📚 小结

Flow Matching 学习更直的噪声→数据路径——将采样步数从 DDPM 的 20-50 步压缩到 2-5 步。Rectified Flow 是训练扩散骨干的主流方法。Consistency Model 蒸馏出一步生成——质量略低但速度极快。2026 年的视频生成和实时图像编辑都建立在 Flow Matching 之上。

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
