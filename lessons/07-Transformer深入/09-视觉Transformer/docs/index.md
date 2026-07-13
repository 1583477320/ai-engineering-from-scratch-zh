# 视觉 Transformer（ViT）

> 把图片切成小块，展平成序列，喂给 Transformer——ViT 证明了 Transformer 不仅处理文字，还处理图像。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 07 · 05（完整 Transformer）| **时间：** ~45 分钟

---

## 🎯 学习目标

- [ ] 理解 ViT 的核心思想——将图像分割为固定大小的 patch，每个 patch 作为一个"词元"
- [ ] 比较 CNN 和 ViT 的归纳偏置差异——CNN 假设局部性，ViT 假设什么都没有
- [ ] 说明 ViT 为什么需要大量数据——缺乏 CNN 的平移不变性归纳偏置

---

## 1. 问题

CNN 是计算机视觉的默认架构——局部卷积核、池化层、平移不变性。2020 年 ViT 问了一个简单的问题：**如果把图像的每个 patch 当作一个词元，直接用 Transformer 处理会怎样？**

答案：在大数据集（JFT-300M）上，ViT 超越了 CNN。在小数据集上，CNN 因为更强的归纳偏置仍然赢。这验证了一个核心权衡：**归纳偏置 vs 数据效率。**

---

## 2. 概念

### 2.1 ViT 架构

```
输入图像 (H×W×3)
    ↓ 切成 patch（如 16×16）
    ↓ 展平每个 patch → (P×D) 的向量序列
    ↓ 加上可学习的位置嵌入
    ↓ 送入标准 Transformer 编码器
    ↓ [CLS] token 的输出 → 分类头
```

| 参数 | 典型值 |
|---|---|
| Patch 大小 | 16×16 |
| 嵌入维度 d_model | 768 |
| Transformer 层数 | 12 |
| 多头数 | 12 |
| 总参数 | ~86M |

### 2.2 CNN vs ViT 的权衡

| | CNN | ViT |
|---|---|---|
| 归纳偏置 | 局部性、平移不变性 | 无先验 |
| 数据需求 | 较低（归纳偏置补偿） | 较高（需要学习局部结构） |
| 长距离依赖 | 池化层限制 | 全局注意力，无限制 |
| 2026 选择 | 仍然用于边缘设备 | 大数据集上的主流 |

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| Patch | 图像的固定大小子块（如 16×16），展平后作为一个"词元"输入 |
| 位置嵌入 | ViT 用可学习的位置嵌入——图像中 patch 的位置编码 |
| 归纳偏置 | 架构隐式假设数据的形状；CNN 假设局部性，ViT 假设没有 |

---

## 📚 小结

ViT 将图像切成 patch 作为词元序列输入 Transformer。它没有 CNN 的局部性归纳偏置——需要更多数据来学习图像的局部结构，但在大数据集上超越了 CNN。DINOv2、SAM 3 都建立在 ViT 之上——同一个块，不同的输入。

---

## ✏️ 练习

1. 画出 ViT 的完整架构图——从输入图像到分类输出的每一步
2. 对比 ViT-Base（86M）和 ResNet-50（25M）在 CIFAR-10 上的表现——谁赢了？为什么？

---

## 📖 参考资料

1. [论文] Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT). 2021.
2. [论文] Caron et al. "Emerging Properties in Self-Supervised Vision Transformers" (DINOv2). 2021.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
