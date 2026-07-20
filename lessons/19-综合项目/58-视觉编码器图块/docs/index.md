# 综合项目58——视觉编码器图块（Vision Encoder Patches）

> 读取像素的视觉模型需要像素的分词器。图块嵌入就是这个分词器。将图像切割成方形网格，展平每个方块，通过一个线性层投影，然后添加 2D 位置信号，使 Transformer 知道每个方块在原图中的位置。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 将图像词元化为固定长度的图块嵌入序列
- 实现基于 `Conv2d` 的图块投影，匹配 unfold-then-linear 的数学等价性
- 构建确定性 2D 正弦位置嵌入，使词元顺序编码空间位置
- 验证图块数量、嵌入形状和 `Conv2d`/unfold 等价性

---

## 1. 问题

Transformer 处理的是向量序列。图像是一个 3 通道网格。将每个像素读作一个词元会使序列长度爆炸：一张 224x224 的 RGB 图像有 150,528 个词元，12 层 Transformer 在注意力上无法承受。将图像作为单个巨大扁平向量读取则丢失了注意力无法恢复的局部性。编码器前端的作用是将像素网格压缩为几百个词元，每个词元总结一个方形区域。

图块嵌入用一个线性投影解决这个问题。224x224 图像切分为 16x16 图块，产生 14x14 网格共 196 个图块。每个图块从 `(3, 16, 16) = 768` 个像素值展平为一个向量，然后线性层将其映射到模型的隐藏维度。Transformer 看到 196 个维度为 `hidden`（通常是 768）的词元加上一个 CLS 词元。

---

## 2. 核心概念

### 2.1 为什么是图块而不是像素

注意力在序列长度上是二次的。196 个词元产生 `196 × 196 = 38,416` 个注意力分数/头/层；150,528 个词元产生 `150,528 × 150,528 = 226 亿`。图块带来了 590,000 倍的注意力计算缩减，而单个 16x16 区域承载了足够的信号用于高层视觉任务。

### 2.2 `Conv2d` 技巧

`Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)` 的每个输出位置将图块像素与一个滤波器做点积——这与 unfold-then-linear 在数值上相同。大多数生产代码库使用卷积，因为在 GPU 上更快且少一次 reshape。

### 2.3 位置嵌入

词元从投影输出时不携带顺序。2D 正弦嵌入给每个词元一个编码 `(row, col)` 位置的固定信号：

```text
PE(row, col, 2i)   = sin(row / 10000^(4i/d))
PE(row, col, 2i+1) = cos(row / 10000^(4i/d))
PE(row, col, d+2i) = sin(col / 10000^(4i/d))
PE(row, col, d+2i+1) = cos(col / 10000^(4i/d))
```

| 组件 | 形状 | 参数 |
|------|------|------|
| 图块投影（Conv2d） | `(hidden, 3, patch, patch)` | `3 × P² × hidden + hidden` |
| 位置嵌入（固定） | `(num_patches, hidden)` | 0 |
| CLS 词元（可学习） | `(1, hidden)` | `hidden` |

---

## 3. 从零实现

```python
"""视觉编码器图块——Conv2d 投影+2D 正弦位置。"""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F


def sinusoidal_2d(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    """构建 2D 正弦位置嵌入表，形状 (grid_h * grid_w, dim)。"""
    assert dim % 4 == 0, "dim 必须是 4 的倍数"
    pe = torch.zeros(grid_h * grid_w, dim)
    d = dim // 2
    for pos_row in range(grid_h):
        for pos_col in range(grid_w):
            idx = pos_row * grid_w + pos_col
            div = 10000 ** (torch.arange(0, d, 2) / d)
            pe[idx, 0:d:2] = torch.sin(torch.tensor(pos_row / div))
            pe[idx, 1:d:2] = torch.cos(torch.tensor(pos_row / div))
            pe[idx, d:dim:2] = torch.sin(torch.tensor(pos_col / div))
            pe[idx, d+1:dim:2] = torch.cos(torch.tensor(pos_col / div))
    return pe


class PatchEmbed(nn.Module):
    """Conv2d 图块投影。"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size; self.patch_size = patch_size
        self.grid_h = img_size // patch_size; self.grid_w = img_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) → (B, 768, 14, 14) → (B, 196, 768)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class VisionFrontEnd(nn.Module):
    """图块嵌入 + CLS + 位置信号。"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        pe = sinusoidal_2d(self.patch_embed.grid_h, self.patch_embed.grid_w, embed_dim)
        self.register_buffer("pos_embed", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, 196, 768)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)  # (B, 197, 768)
        x[:, 1:, :] += self.pos_embed  # CLS 不接收位置信号
        return x


def main():
    model = VisionFrontEnd()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")  # (2, 197, 768)
    print(f"图块数量: {model.patch_embed.num_patches}")

    # Conv2d 与 unfold 等价性检查
    conv = model.patch_embed.proj
    unfold = nn.Unfold(kernel_size=16, stride=16)
    linear = nn.Linear(16 * 16 * 3, 768)
    linear.weight.data = conv.weight.data.view(768, -1)
    linear.bias.data = conv.bias.data
    x_small = torch.randn(1, 3, 32, 32)
    conv_out = model.patch_embed(x_small)
    unfold_out = unfold(x_small).transpose(1, 2)
    unfold_out = linear(unfold_out)
    diff = (conv_out - unfold_out).abs().max().item()
    print(f"Conv2d vs Unfold 最大差异: {diff:.6f} (应为 < 1e-5)")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| 图块 | 图像的方形子区域，通常 14×14 或 16×16 |
| 图块嵌入 | 将展平的图块通过线性投影到隐藏维度 |
| 序列长度 | 图块词元化后的词元数（通常 +1 个 CLS） |
| 正弦位置 | 编码 2D 网格坐标的固定 sin/cos 信号 |
| CLS 词元 | 作为池化头的可学习向量，预置到序列前 |

---

## 5. 工程最佳实践

- **选择合适的图块大小**：14×14 → 256 图块，16×16 → 196 图块。更大的图块减少序列长度但丢失更多细节。
- **CLS 不接收位置信号**：这是设计决定的——CLS 作为全局汇聚词元，不应有特定位置。
- **中文场景建议**：AI 绘画模型也使用类似的结构——Stable Diffusion 的 VAE 编码器输出图块化的潜在表示。

---

## 6. 常见错误

- **Conv2d 输出 reshape 错误**：`flatten(2)` 从通道维度开始展开，错误使用 `flatten(1)` 会破坏批次维度。
- **位置嵌入维度对齐错误**：正弦位置要求 `dim % 4 == 0`，否则行列编码不对齐。
- **CLS 接收到位置信号**：CLS 应放在位置嵌入之前，并在切片中排除 `x[:, 1:, :] += pos_embed`。

---

## 📖 参考资料

1. [论文] Dosovitskiy et al. "An Image is Worth 16x16 Words". ICLR 2021. https://arxiv.org/abs/2010.11929
2. [论文] Vaswani et al. "Attention Is All You Need". NeurIPS 2017. https://arxiv.org/abs/1706.03762
