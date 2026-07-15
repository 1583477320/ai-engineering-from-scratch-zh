# 视觉 Transformer 与图块词元

> 在多模态之前，图像必须先变成 Transformer 能"吃"的词元序列。2020 年的 ViT 论文用 16×16 像素图块、线性投影和位置嵌入回答了这个问题。2026 年所有前沿模型仍然以这个原语开始——编码器从 ViT 变成 DINOv2 和 SigLIP 2，加入了注册词元，位置编码变成了 2D-RoPE，但核心不变。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 07（Transformer）、阶段 04（计算机视觉）| **时间：** ~120 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 将 H×W×3 的图像转换为带正确位置编码的图块词元序列
- [ ] 计算给定（图块大小、分辨率、隐藏维度、深度）的 ViT 序列长度、参数量和 FLOPs
- [ ] 说出 ViT 从 2020 年研究到 2026 年产品的三个关键升级
- [ ] 在下游任务中选择合适的池化方式——CLS 池化、平均池化、注册词元

---

## 1. 问题

Transformer 处理向量序列。文本已经是序列（字节或词元）。图像是具有三个颜色通道的 2D 像素网格——不是序列。如果把每个像素展平，一张 224×224 的 RGB 图像变成 150,528 个词元——自注意力在这个长度上是不可行的。

2020 年之前的做法是在前端接一个 CNN 特征提取器——ResNet 产生 7×7 的 2048 维特征图，将这 49 个词元送入 Transformer。这可行但继承了 CNN 的偏差。

Dosovitskiy 等人（2020）问了一个直白的问题：如果我们跳过 CNN 会怎样？把图像切成固定大小的图块（如 16×16 像素），将每个图块线性投影为一个向量，添加位置嵌入，将序列送入原生 Transformer。当时这是异端——没有卷积的视觉。

到 2026 年，ViT 原语已经毫无疑问地成为基础。每个开源 VLM 的视觉塔都是它的某种后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是"是否使用图块"，而是"用多大图块、什么分辨率策略、什么预训练目标、什么位置编码"。

---

## 2. 概念

### 2.1 图块即词元

一张 H×W 的 RGB 图像被切分为大小为 P×P 的图块。每个图块有 3×P×P 个像素值。通过线性投影映射到隐藏维度 d：

```
图像 (3, H, W) → 切分为 (N, 3, P, P) 的图块
    ↓ 线性投影
图块嵌入 (N, d)  ← N = (H/P) × (W/P)
```

例如：224×224 的图像，P=16：N = (224/16)² = 196 个图块词元。

### 2.2 位置编码

ViT 2020 使用可学习的一维位置嵌入。后来发现二维 RoPE 更好——将高度和宽度的位置信息旋转编码到注意力计算中。

### 2.3 从 ViT 到 2026 年的 VLM 视觉塔

| 升级 | ViT 2020 | 2026 VLM 视觉塔 |
|------|----------|----------------|
| 预训练 | ImageNet 分类 | DINOv2 自监督 / SigLIP 2 对比学习 |
| 位置编码 | 可学习 1D | 2D-RoPE / 条件位置编码 |
| 注册词元 | 无 | 4-8 个可学习的非空间词元 |
| 分辨率 | 224 固定 | 原生任意分辨率（patch n' pack） |

---

## 3. 从零实现

### Step 1：图像切分为图块

```python
import torch
import torch.nn as nn

def image_to_patches(image, patch_size=16):
    """将图像切分为图块序列。"""
    B, C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0
    # 使用 unfold 切分
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # 展平为 (B, num_patches, C*P*P)
    B, C, H_p, W_p, P1, P2 = patches.shape
    patches = patches.reshape(B, C, H_p*W_p, P*P).permute(0, 2, 1, 3).reshape(B, H_p*W_p, C*P*P)
    return patches  # (B, num_patches, patch_dim)
```

### Step 2：ViT 前向传播

```python
class SimpleViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_layers=12, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 图块投影
        self.patch_proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim)
        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        # CLS 词元
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 分类头
        self.head = nn.Linear(embed_dim, 1000)

    def forward(self, x):
        B = x.shape[0]
        # 切分图块并投影
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size)
        B, C, Hp, Wp, P1, P2 = patches.shape
        patches = patches.reshape(B, C, Hp*Wp, P1*P2).permute(0, 2, 1, 3)
        patches = patches.reshape(B, Hp*Wp, -1)
        patch_emb = self.patch_proj(patches)

        # 添加 CLS 和位置
        cls = self.cls_token.expand(B, -1, -1)
        patch_emb = torch.cat([cls, patch_emb], dim=1)
        patch_emb = patch_emb + self.pos_embed

        # Transformer 编码
        out = self.transformer(patch_emb)

        # 分类：CLS 词元
        cls_out = out[:, 0]
        return self.head(cls_out)
```

### Step 3：参数量和 FLOPs 计算

```python
def vit_stats(image_size, patch_size, embed_dim, num_layers, num_heads):
    """计算 ViT 的统计量。"""
    num_patches = (image_size // patch_size) ** 2
    patch_dim = 3 * patch_size * patch_size
    # 参数量
    patch_proj = patch_dim * embed_dim
    per_layer = 4 * embed_dim * embed_dim + 2 * embed_dim * embed_dim  # 注意力 + FFN
    total_params = patch_proj + num_layers * per_layer + embed_dim  # CLS + 位置
    # FLOPs（近似）
    attn_flops = num_layers * num_patches * num_patches * embed_dim * 2
    ffn_flops = num_layers * num_patches * embed_dim * embed_dim * 8

    return {
        "num_patches": num_patches,
        "patch_dim": patch_dim,
        "total_params_M": total_params / 1e6,
        "attn_flops_G": attn_flops / 1e9,
        "ffn_flops_G": ffn_flops / 1e9,
    }
```

---

## 4. 工具

### 4.1 timm 库（PyTorch Image Models）

```python
import timm
model = timm.create_model("vit_base_patch16_224", pretrained=True)
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
```

### 4.2 HuggingFace Transformers

```python
from transformers import ViTModel
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
```

### 4.3 工具对比

| 工具 | 特点 | 适用场景 |
|------|------|---------|
| timm | 预训练模型库 | 快速实验、迁移学习 |
| HuggingFace | 完整生态 | 推理、微调、部署 |
| 手写（本课） | 教学理解 | 从零理解 ViT |

---

## 6. 工程最佳实践

### 6.1 图块大小选择

| 图块大小 | 序列长度 | 细节 | 速度 | 推荐场景 |
|---------|---------|------|------|---------|
| 8×8 | 784 | 极精细 | 慢 | 高精度需求 |
| 16×16 | 196 | 平衡 | 中 | 通用（推荐） |
| 32×32 | 49 | 粗糙 | 快 | 快速原型 |

### 6.2 中文场景

- ViT 本身对中文无影响——但下游 VLM 的文本编码器需要中文支持
- 中文文档图像（扫描件、表格）需要更小的图块来保留细节

### 6.3 踩坑经验

- **分辨率不匹配**：ViT 的位置嵌入与训练分辨率绑定——推理时需要插值
- **位置嵌入形状错误**：忘记加 CLS 词元导致维度不匹配
- **未冻结视觉塔**：从预训练 ViT 微调时，初始几层通常冻结

---

## 7. 常见错误

### 错误 1：忘记 CLS 词元

**现象：** 分类头输出维度错误——因为位置嵌入形状不匹配。

**原因：** 切分图块时没有预留 CLS 词元的位置。

**修复：** 切分为 N 个图块后，添加 1 个 CLS 词元 → 总长度 N+1。

### 错误 2：图像尺寸不是图块大小的整数倍

**现象：** unfold 操作报错或结果形状错误。

**原因：** 224÷16=14 是整数，但其他尺寸可能不是。

**修复：** 使用 padding 将图像填充到图块大小的整数倍，或使用自适应池化。

---

## 8. 面试考点

### Q1：ViT 的 CLS 池化和平均池化有什么区别？（难度：⭐⭐）

**参考答案：**
CLS 池化取序列中 CLS 词元的输出作为全局表示——它通过自注意力学习聚合全局信息。平均池化取所有图块输出的平均值。CLS 池化的优点是更灵活（可以学习关注哪些位置），缺点是需要额外的可学习 CLS 词元。平均池化更简单，对固定分辨率图像效果相当。2026 年 ViT-DINOv2 等模型使用 CLS 池化为主。

### Q2：ViT 为什么能从足够数据中超越 CNN？（难度：⭐⭐⭐）

**参考答案：**
CNN 有强归纳偏置（局部感受野、平移等变性），在小数据上泛化好但天花板有限。ViT 几乎没有归纳偏置——它对所有图块之间的所有关系做全局注意力。这意味着 ViT 的参数可以被数据更充分地"利用"——只要数据足够（JFT-300M → LAION），ViT 的更大容量就能超越 CNN。归纳偏置少 = 需要更多数据 = 但上限更高。

### Q3：DINOv2 预训练与 CLIP 预训练有什么不同？（难度：⭐⭐⭐）

**参考答案：**
CLIP 是**对比学习**——将图像和文本嵌入到同一向量空间，最大化匹配对的相似度，最小化非匹配对。DINOv2 是**自监督学习**——不需要文本，只用图像。它通过学生-教师网络和自蒸馏学习视觉特征。关键区别：CLIP 的特征对文本描述有语义对齐（能做零样本分类），DINOv2 的特征更适合密集预测（分割、深度估计）。2026 年的 VLM 常常将两者结合。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| ViT | "视觉 Transformer" | 将图像切分为固定大小的图块，线性投影为词元，送入 Transformer 编码器 |
| 图块 (Patch) | "图像的小块" | P×P 像素的图像块——ViT 的基本输入单元 |
| CLS 词元 | "分类 token" | 可学习的全局表示词元——放在序列开头，输出作为图像嵌入 |
| 2D-RoPE | "2D 旋转位置编码" | 将高度和宽度位置信息旋转编码——支持任意分辨率输入 |
| DINOv2 | "自监督视觉编码器" | 自监督预训练的 ViT——不需要图像标签，特征适合下游任务 |
| SigLIP 2 | "Sigmoid CLIP" | 用 sigmoid 替代 softmax 的对比学习——更高效的视觉编码器 |

---

## 📚 小结

ViT 是多模态的基石——将图像转换为 Transformer 可处理的图块词元序列。2020 年的异端在 2026 年成为标准。核心流程：图像切分为 P×P 图块 → 线性投影 → 添加位置嵌入 → 送入 Transformer。2026 年的 VLM 视觉塔（DINOv2、SigLIP 2）仍以此为基础——只是编码器更强、位置编码更好、分辨率更灵活。

---

## ✏️ 练习

1. **【实现】** 用 PyTorch 实现 `image_to_patches` 函数——将 224×224 图像切分为 16×16 图块，验证输出形状
2. **【计算】** 计算 ViT-Base (patch=16, dim=768, layers=12) 的参数量和 FLOPs
3. **【实验】** 用 timm 加载预训练 ViT-Base，对比不同图块大小（8, 16, 32）在 ImageNet 上的精度

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| ViT 从零实现 | `code/main.py` | 图块切分 + 位置编码 + Transformer 编码 + 分类头 |

---

## 📖 参考资料

1. [论文] Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". ICLR, 2021. https://arxiv.org/abs/2010.11929
2. [论文] Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision". TMLR, 2024. https://arxiv.org/abs/2304.07193
3. [论文] Zhai et al. "Sigmoid Loss for Language Image Pre-Training". ICCV, 2023. https://arxiv.org/abs/2303.15343
4. [GitHub] timm: https://github.com/huggingface/pytorch-image-models

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
