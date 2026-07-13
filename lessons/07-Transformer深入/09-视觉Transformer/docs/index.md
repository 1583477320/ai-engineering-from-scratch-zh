# 视觉 Transformer（ViT）

> 把图片切成小块，展平成序列，喂给 Transformer——ViT 证明了 Transformer 不仅处理文字，还处理图像。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 07 · 05（完整 Transformer）
**时间：** ~45 分钟
**所处阶段：** Tier 2
**关联课程：** 第 7 阶段 · 10（Whisper）— 对比 Transformer 在视觉和音频领域的应用

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

## 3. 从零实现

将图像 patch 作为词元序列处理：

```python
import numpy as np

def image_to_patches(image, patch_size=16):
    """将图像分割为 patch。"""
    H, W, C = image.shape
    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch.flatten())
    return np.array(patches)

# 32×32 图像被分割为 4×4=16 个 8×8 patch
image = np.random.randn(32, 32, 3)
patches = image_to_patches(image, 8)
print(f"Patch 形状: {patches.shape}")  # (16, 192)
# 16 个 patch，每个展平后 8×8×3=192 维
```

完整代码见 `code/main.py`——纯 NumPy，可立即运行。

---

## 4. 工业工具

### 4.1 HuggingFace ViT

```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# 加载 ViT 模型
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# 加载图像
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 预处理 + 预测
inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)
pred_class = outputs.logits.argmax(-1).item()
print(f"预测类别: {model.config.id2label[pred_class]}")
```

### 4.2 性能对比

| 模型 | 参数量 | Patch 大小 | ImageNet 准确率 |
|---|---|---|---|
| ViT-Base | 86M | 16×16 | 77.9% |
| ViT-Large | 307M | 16×16 | 76.5%（更多数据） |
| ViT-Huge | 632M | 14×14 | 81.0% |
| ResNet-50 | 25M | — | 76.1% |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

ViT 奠定了多模态大语言模型的基础。GPT-4V、Gemini、LLaVA 等模型都使用 ViT 作为视觉编码器——将图像转换为词元序列，然后送入大语言模型。

DINOv2 是 Meta 基于 ViT 的自监督学习模型，广泛用于图像检索、语义分割、目标检测。不依赖标注数据，可以从 Web 图像中学习通用视觉表示。

### 5.2 LLM 时代什么变了？

**从独立模型到多模态。** ViT 最初是独立的图像分类模型。现在它是多模态大语言模型的视觉编码器——将图像转换为大语言模型能理解的词元序列。

**从有监督到自监督。** DINOv2 证明了 ViT 可以在无标注数据上学习通用视觉表示——用自监督学习替代 ImageNet 标注。

### 5.3 什么没变？

**核心架构没变。** ViT 的核心思想——将图像分割为 patch 作为词元序列——没有改变。

**归纳偏置的权衡没变。** ViT 仍然需要更多数据才能超越 CNN。在小数据上 CNN 仍然赢。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你在 ChatGPT 上传一张图片时，模型使用 ViT 风格的编码器将图像转换为词元序列，然后与大语言模型一起处理。

这就是多模态大语言模型的工作原理：视觉编码器（如 ViT）将图像转换为词元序列，大语言模型（如 GPT-4）处理文本和图像词元的混合序列。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 图像分类 | ViT-Base / ResNet-50 | 大数据 ViT，小数据 CNN |
| 语义分割 | SETR / DINOv2 | ViT 做分割 |
| 目标检测 | DETR | Transformer 版检测器 |
| 多模态 | ViT + LLaMA | 视觉编码器 + 大语言模型 |

### 6.2 中文场景特别建议

- 中文 OCR 任务使用 ViT + 中文大语言模型

### 6.3 踩坑经验

- ViT 在小数据集上表现不如 CNN——如果数据不足，使用 CNN 或数据增强
- Patch 大小影响模型性能——16×16 是平衡，8×8 更好但计算量大
- ViT 对正则化敏感——Dropout、Stochastic Depth 需要仔细调参

---

## 7. 常见错误

### 错误 1：在小数据集上直接使用 ViT

**现象：** ViT 在 CIFAR-10 上表现不如 ResNet-50。

**原因：** ViT 没有 CNN 的归纳偏置——需要大量数据学习局部结构。小数据集上 CNN 因为更强的归纳偏置仍然赢。

**修复：**
```python
# ❌ 小数据集上使用 ViT
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# ✓ 小数据集上使用 CNN（或 ViT + 强数据增强）
model = ResNet(num_classes=10)  # 归纳偏置补偿数据不足
```

### 错误 2：Patch 大小选择不当

**现象：** 模型在细粒度分类任务上表现差。

**原因：** Patch 太大（如 32×32）丢失细节信息。Patch 太小（如 4×4）计算量太大。

**修复：**
```python
# ❌ 32×32 patch 丢失细节
patch_size = 32

# ✓ 16×16 patch 是平衡选择
patch_size = 16
```

### 错误 3：忘记位置嵌入

**现象：** 模型无法区分不同的图像区域。

**原因：** ViT 使用可学习位置嵌入——patch 的顺序信息不是通过卷积隐含的，而是通过位置嵌入显式注入的。

---

## 8. 面试考点

### Q1：ViT 的核心思想是什么？（难度：⭐⭐）

**参考答案：**
ViT 将图像分割为固定大小的 patch（如 16×16），展平每个 patch 作为"词元"，加上位置嵌入后送入标准 Transformer 编码器。[CLS] token 的输出用于分类。

### Q2：CNN 和 ViT 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
CNN 有局部性归纳偏置——假设相邻像素相关，使用小卷积核。ViT 没有归纳偏置——使用全局注意力，所有 patch 直接交互。

### Q3：为什么 ViT 需要比 CNN 更多的数据？（难度：⭐⭐⭐）

**参考答案：**
CNN 的卷积核天然编码了空间局部性和平移不变性——这是硬编码的先验知识。ViT 的自注意力是全局的，没有这种先验——必须从数据中学习图像的局部结构。

### Q4：ViT 的位置嵌入为什么是可学习的而不是正弦编码？（难度：⭐⭐⭐）

**参考答案：**
图像 patch 的位置编码与文本词元的位置编码意义不同。文本中位置有明确的顺序（第 1 个词元、第 2 个词元），而图像中 patch 的排列是二维的——需要学习二维位置关系。可学习编码的灵活性更适合二维空间。

### Q5：DINOv2 如何改进 ViT？（难度：⭐⭐⭐）

**参考答案：**
DINOv2 是 Meta 基于 ViT 的自监督学习方法：1）无需标注数据——从 Web 图像学习；2）学习通用视觉表示——适应多种任务；3）强大的特征——适用于分类、分割、检索、检测。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Patch | "图像小块" | 图像的固定大小子块（如 16×16），展平后作为"词元"输入 |
| 位置嵌入 | "位置向量" | ViT 用可学习的位置嵌入——图像中 patch 的位置编码 |
| 归纳偏置 | "先验假设" | 架构隐式假设数据的形状；CNN 假设局部性，ViT 假设没有 |
| Patch 投影 | "将 patch 映射为向量" | 将展平的 patch 通过线性层投影到 d_model 维 |
| [CLS] token | "分类 token" | 第一个特殊词元，其输出用于整个图像的分类 |
| 多模态嵌入 | "文本+图像" | 将 ViT 的图像词元与大语言模型的文本词元拼接 |

---

## 📚 小结

ViT 将图像切成 patch 作为词元序列输入 Transformer。它没有 CNN 的局部性归纳偏置——需要更多数据来学习图像的局部结构，但在大数据集上超越了 CNN。DINOv2、SAM 3 都建立在 ViT 之上——同一个块，不同的输入。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释 ViT 如何将图像当作文本处理。写 200 字以内的说明。

2. **【实现】** 从零实现 ViT 的 patch 分割和线性投影——验证 patch 数量和投影维度的计算。

3. **【实验】** 对比 ViT-Base（86M）和 ResNet-50（25M）在 CIFAR-10 上的表现——谁赢了？为什么？

4. **【实现】** 画出 ViT 的完整架构图——从输入图像到分类输出的每一步。

5. **【思考】** 阅读 DINOv2 论文的摘要，用你自己的话解释自监督学习如何让 ViT 学得更好。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| ViT 完整实现 | `code/main.py` | Patch 分割、ViT 模型、分类头 |
| CNN vs ViT 对比指南 | `outputs/vit-cnn-comparison.md` | 两种视觉架构的详细对比 |

---

## 📖 参考资料

1. [论文] Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT). 2021.
2. [论文] Caron et al. "Emerging Properties in Self-Supervised Vision Transformers" (DINOv2). 2021.
3. [论文] Kirillov et al. "Segment Anything" (SAM). 2023.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
