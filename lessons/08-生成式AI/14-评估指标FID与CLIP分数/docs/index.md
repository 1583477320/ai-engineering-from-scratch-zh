# 评估指标——FID 与 CLIP Score

> 生成模型的评估不能只靠人类看——FID 衡量"生成分布与真实分布的距离"，CLIP Score 衡量"生成图像与文本描述的对齐度"。两个指标共同定义了生成质量的量化基准。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 08 · 01（生成模型分类）| **时间：** ~45 分钟

---

## 🎯 学习目标

- [ ] 从零实现 FID（Fréchet Inception Distance）——理解均值向量和协方差矩阵的作用
- [ ] 解释 CLIP Score 如何评估文本-图像对齐——对比学习的推理时应用
- [ ] 说明 2026 年生成评估的最佳实践——FID + CLIP Score + LPIPS + 人类评估

---

## 1. 问题

如何客观评估生成图像的质量？FID 衡量"生成分布与真实分布的距离"。CLIP Score 衡量"文本描述与生成图像的对齐度"。两者互补——FID 测质量，CLIP 测语义对应。

---

## 2. 概念

### 2.1 FID——Fréchet Inception Distance

$$\text{FID} = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

- $\mu_r, \Sigma_r$：真实图像在 Inception-v3 特征空间的均值和协方差
- $\mu_g, \Sigma_g$：生成图像的均值和协方差
- **越低越好**——FID=0 表示分布完全相同

### 2.2 CLIP Score

CLIP 将图像和文本编码到同一嵌入空间。CLIP Score = 图像嵌入与文本嵌入的余弦相似度。**衡量文本描述与生成图像的语义对齐度。**

### 2.3 2026 年的评估组合

| 指标 | 衡量什么 | 越高/越低越好 |
|---|---|---|
| FID | 生成分布与真实分布的距离 | 越低越好 |
| CLIP Score | 文本-图像语义对齐 | 越高越好 |
| LPIPS | 感知相似度（人类感知） | 越低越好 |
| 人类偏好 | 最终裁判 | 越高越好 |

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| FID | Fréchet Inception Distance——生成分布与真实分布的距离 |
| CLIP Score | CLIP 嵌入的余弦相似度——文本与图像的语义对齐度 |
| LPIPS | 基于 VGG 感知特征的图像相似度 |

---

## 📚 小结

FID 衡量分布距离——越低越好。CLIP Score 衡量语义对齐——越高越好。LPIPS 衡量人类感知相似度。2026 年的最佳实践：FID + CLIP Score + LPIPS + 人类偏好四维度组合评估。

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
