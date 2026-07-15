# 任意分辨率视觉：图块打包与 NaFlex

> 真实图像不是 224×224 的正方形。收据是 9:16，图表是 16:9，医学扫描可能是 4096×4096。2024 年之前的 VLM 将所有图像调整为固定正方形——丢弃了 OCR、文档理解和高分辨率场景解析所需的信号。NaViT（Google，2023）展示了如何将可变分辨率图块打包进一个 Transformer 批次并使用块对角掩码。Qwen2-VL 的 M-RoPE（2024）完全丢弃了绝对位置表。LLaVA-NeXT 的 AnyRes 将高分辨率图像平铺为基础+子图像。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 12 · 01（ViT 图块）、05（LLaVA）| **时间：** ~120 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 将可变分辨率图像的图块打包进一个序列并构建块对角注意力掩码
- [ ] 理解 NaViT 的打包策略——如何在一个批次中处理不同分辨率
- [ ] 解释 M-RoPE（多分辨率旋转位置编码）如何支持任意分辨率
- [ ] 说明 AnyRes 的平铺策略——基础+子图像处理高分辨率

---

## 1. 问题

ViT 要求固定输入分辨率——所有图像必须调整到 224×224。但现实中的图像差异巨大：
- 手机截图 9:19.5 → 调整后扭曲
- 医学扫描 4096×4096 → 压缩到 224×224 丢失细节
- 文档扫描 9:16 → 横向拉伸

**核心矛盾：** 高分辨率需要更多图块 → 更多计算 → Transformer 的二次复杂度。

解决方案：
- **NaViT**：在一个批次中打包不同分辨率的图块，用块对角掩码防止跨图像注意力
- **M-RoPE**：用多分辨率旋转位置编码替代固定位置表
- **AnyRes**：将高分辨率图像平铺为基础分辨率+子区域

---

## 2. 概念

### 2.1 NaViT——打包不同分辨率

传统 ViT：所有图像调整到相同分辨率 → 每张图相同数量的图块。

NaViT：每张图按原始分辨率切分图块 → 不同图像不同数量的图块 → 用掩码防止跨图像注意力。

```
图像 A (224×224): 196 个图块
图像 B (448×224): 392 个图块
图像 C (224×448): 392 个图块

打包到一个序列: [图块A1...196] [图块B1...392] [图块C1...392]
块对角掩码: 图像之间不互相注意
```

### 2.2 块对角掩码

```
图像A的图块  → 只看图像A的图块（不看B和C）
图像B的图块  → 只看图像B的图块（不看A和C）
图像C的图块  → 只看图像C的图块（不看A和B）
```

### 2.3 M-RoPE（多分辨率旋转位置编码）

Qwen2-VL 的方案——不使用固定的位置嵌入表，而是将每个维度的位置信息旋转编码。这样可以处理任意分辨率——因为位置编码是连续的，不需要预定义最大长度。

### 2.4 AnyRes（LLaVA-NeXT）

将高分辨率图像拆分为基础图像（低分辨率）+ 多个子区域（高分辨率）：
- 基础图像提供全局上下文
- 子区域提供局部细节

---

## 3. 从零实现

### Step 1：可变分辨率图块打包

```python
import torch

def patchify_and_pack(images, patch_size=16):
    """
    将一批可变分辨率图像打包为一个序列。
    Args:
        images: list of (C, H_i, W_i) tensors
    Returns:
        patches: (1, total_patches, patch_dim)
        boundaries: 每张图像的起止位置
    """
    all_patches = []
    boundaries = []
    offset = 0

    for img in images:
        C, H, W = img.shape
        # 确保能被 patch_size 整除
        H_pad = (H + patch_size - 1) // patch_size * patch_size
        W_pad = (W + patch_size - 1) // patch_size * patch_size
        padded = torch.nn.functional.pad(img, (0, W_pad - W, 0, H_pad - C + C))

        # 切分图块
        patches = padded.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        _, _, Hp, Wp, _, _ = patches.shape
        patches = patches.reshape(1, Hp * Wp, C * patch_size * patch_size)

        num_patches = Hp * Wp
        boundaries.append((offset, offset + num_patches))
        all_patches.append(patches)
        offset += num_patches

    packed = torch.cat(all_patches, dim=1)  # (1, total, patch_dim)
    return packed, boundaries


def create_block_diagonal_mask(boundaries, seq_len):
    """创建块对角注意力掩码——图块之间不互相注意。"""
    mask = torch.ones(seq_len, seq_len) * float("-inf")
    for start, end in boundaries:
        mask[start:end, start:end] = 0.0  # 同一图像内的图块可以注意
    return mask
```

### Step 2：任意分辨率推理

```python
def naive_var_resolution(images, model, patch_size=16):
    """朴素方法：全部调整到相同分辨率。"""
    resized = [torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(224, 224), mode="bilinear"
    ).squeeze(0) for img in images]
    return torch.stack(resized)

def navit_style(images, model, patch_size=16):
    """NaViT 方法：保持原始分辨率，打包后处理。"""
    packed, boundaries = patchify_and_pack(images, patch_size)
    seq_len = packed.shape[1]
    mask = create_block_diagonal_mask(boundaries, seq_len)
    return packed, mask
```

---

## 4. 工具

### 4.1 LLaVA-NeXT（AnyRes）

```python
from transformers import LlavaNextForConditionalGeneration
# LLaVA-NeXT 自动处理任意分辨率
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf"
)
```

### 4.2 Qwen2-VL（M-RoPE）

```python
from transformers import Qwen2VLForConditionalGeneration
# Qwen2-VL 使用 M-RoPE 处理任意分辨率
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B")
```

### 4.3 工具对比

| 方法 | 分辨率支持 | 实现复杂度 | 推理速度 |
|------|-----------|-----------|---------|
| 固定调整 | 仅 224×224 | 低 | 快 |
| NaViT | 任意 | 中 | 中 |
| AnyRes | 高分辨率 | 中 | 快 |
| M-RoPE | 任意 | 低 | 快 |

---

## 6. 工程最佳实践

### 6.1 分辨率选择策略

| 场景 | 推荐分辨率 | 方法 |
|------|-----------|------|
| 通用对话 | 336×336 | LLaVA 基础 |
| OCR/文档 | 高分辨率（AnyRes） | 平铺子区域 |
| 医学图像 | 512×512+ | NaViT |
| 视频帧 | 固定低分辨率 | 时间压缩 |

### 6.2 踩坑经验

- **内存爆炸**：高分辨率图块数 N 增大 → 注意力 O(N²) 暴增
- **位置编码超出范围**：训练时位置嵌入只覆盖 224×224 → 推理时需要插值
- **掩码维度错误**：打包时忘记更新注意力掩码

---

## 7. 常见错误

### 错误 1：强制所有图像到相同分辨率

**现象：** 纵向收据被横向拉伸——OCR 准确率骤降。

**修复：** 使用 NaViT 保持原始分辨率，用块对角掩码处理可变长度序列。

### 错误 2：位置编码超出训练范围

**现象：** 高分辨率推理时位置信息错误。

**修复：** 使用相对位置编码（RoPE/M-RoPE）——支持任意长度外推。

---

## 8. 面试考点

### Q1：NaViT 的块对角掩码解决了什么问题？（难度：⭐⭐）

**参考答案：**
当不同分辨率的图块打包在同一个序列中时，需要确保每张图像的图块只关注同一图像内的其他图块——不会跨图像注意力。块对角掩码实现了这一点：对角线块内可以互相注意，非对角块被设为负无穷（softmax 后为零）。

### Q2：AnyRes 和 NaViT 的区别是什么？（难度：⭐⭐⭐）

**参考答案：**
NaViT 保持原始分辨率，将不同分辨率的图块打包在同一个序列中。AnyRes（LLaVA-NeXT）将高分辨率图像拆分为"基础+子区域"——基础图像提供全局上下文，子区域提供高分辨率细节。区别：NaViT 更灵活（任意分辨率），AnyRes 更简单（固定几种分辨率组合）。实践中 AnyRes 在 OCR 和文档理解任务上效果更好。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| NaViT | "可变分辨率 ViT" | 在一个批次中打包不同分辨率图块，用块对角掩码防止跨图像注意力 |
| 块对角掩码 | "图像间不注意" | 掩码矩阵的非对角块设为负无穷——同一图像内的图块互相注意，不同图像间不注意 |
| M-RoPE | "多分辨率位置编码" | Qwen2-VL 的旋转位置编码变体——支持任意分辨率输入 |
| AnyRes | "任意分辨率" | LLaVA-NeXT 的平铺策略——基础图像+子区域处理高分辨率 |

---

## 📚 小结

现实图像需要任意分辨率处理。NaViT 通过可变分辨率打包+块对角掩码实现。M-RoPE 用旋转位置编码支持任意长度。AnyRes 用平铺策略处理高分辨率。2026 年的 VLM 默认支持多种分辨率——不再强制调整到固定大小。

---

## ✏️ 练习

1. **【实现】** 实现可变分辨率图块打包——将 3 张不同分辨率图像打包为一个序列
2. **【对比】** 对比 NaViT 和固定调整方法在不同分辨率下的处理速度

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 可变分辨率打包 | `code/main.py` | 图块打包 + 块对角掩码构建 |

---

## 📖 参考资料

1. [论文] Dehghani et al. "NaViT: Native Resolution Vision Transformer". arXiv, 2023.
2. [论文] Wang et al. "Qwen2-VL: Enhancing Vision-Language Model". arXiv, 2024.
3. [论文] Liu et al. "LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge". 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
