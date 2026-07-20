# 综合项目59——视觉 Transformer 编码器（Vision Transformer Encoder）

> 图块本身看不见。一个 12 层 pre-LN Transformer 将图块词元序列转化为上下文词元序列，CLS 词元在其最终隐藏状态中汇聚整张图像的特征。这节课是每个现代视觉语言模型的引擎室。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 实现带有预层归一化（pre-LN）的 Transformer 块：多头自注意力 + 前馈子层
- 堆叠 12 个块、12 个头，构建 ViT-Base 编码器
- 将第 58 节的图块前端接入编码器并运行前向传播
- 验证 CLS 词元从每个图块汇聚信息

---

## 1. 问题

图块嵌入产生 197 个词元的序列，每个向量对其它图块一无所知。一张猫的图片需要每个图块知道哪些块包含胡须、哪些是背景、哪些是眼睛。Transformer 就是构建这种意识的机制——一层一层注意力地构建。没有它，图块前端就是一个聪明但无理解的分词器。

标准配方是 12 层深、12 头宽、pre-LayerNorm 放置、GELU 激活、4x 前馈网络扩展。这个配方是 CLIP ViT-L、SigLIP、DINOv2、Qwen-VL 系列、InternVL 以及 2025-2026 年每个开源视觉编码器的骨架。

---

## 2. 核心概念

### 2.1 Pre-LN vs Post-LN

原始 Transformer 将 LayerNorm 放在残差之后。Pre-LN（在每个子层之前归一化）是现代模型使用的版本——训练稳定，无需学习率预热技巧。

```
Post-LN: x = LayerNorm(x + SubLayer(x))    # 原始
Pre-LN:  x = x + SubLayer(LayerNorm(x))    # 现代
```

### 2.2 4x 前馈网络扩展

FFN 从 `hidden → 4×hidden → hidden`，中间使用 GELU。因子 4 是经验性的，自 2017 年以来在语言和视觉 Transformer 中保持不变。2x 欠拟合，8x 在固定数据下过拟合。

### 2.3 各组件参数量（ViT-Base）

| 组件 | 参数量 |
|------|--------|
| qkv 投影（每块） | `3 × 768 × 768 = 1.77M` |
| 输出投影（每块） | `768 × 768 = 590K` |
| FFN（4x 扩展） | `2 × 768 × 4 × 768 = 4.72M` |
| 每块合计 | ≈ 7.1M |
| 12 块合计 | ≈ 85M |
| 加前端 | ≈ 86M |

---

## 3. 从零实现

```python
"""视觉 Transformer 编码器——Pre-LN 块+12 层堆叠。"""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads; self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-LN Transformer 块。"""
    def __init__(self, dim: int, heads: int, expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ViT(nn.Module):
    """ViT-Base 编码器：12 层 pre-LN 块堆叠。"""
    def __init__(self, dim: int = 768, heads: int = 12, depth: int = 12, expansion: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads, expansion) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class VisionEncoder(nn.Module):
    """完整 ViT 编码器：前端 + Transformer。"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, dim=768, heads=12, depth=12):
        super().__init__()
        from pathlib import Path
        import sys
        # 复用第 58 节的 VisionFrontEnd
        sys.path.insert(0, str(Path(__file__).parent.parent / "58-视觉编码器图块" / "code"))
        try:
            from frontend import VisionFrontEnd
            self.front_end = VisionFrontEnd(img_size, patch_size, in_chans, dim)
        except ImportError:
            # 内联实现
            self.front_end = self._make_frontend(img_size, patch_size, in_chans, dim)
        self.vit = ViT(dim, heads, depth)

    def _make_frontend(self, img_size, patch_size, in_chans, dim):
        class PatchEmbed(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
                self.num_patches = (img_size // patch_size) ** 2
                self.grid_h = self.grid_w = img_size // patch_size
            def forward(self, x):
                return self.proj(x).flatten(2).transpose(1, 2)
        class FrontEnd(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = PatchEmbed()
                self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
                pe = torch.zeros(self.patch_embed.num_patches, dim)
                for r in range(self.patch_embed.grid_h):
                    for c in range(self.patch_embed.grid_w):
                        i = r * self.patch_embed.grid_w + c
                        d_half = dim // 2
                        for j in range(0, d_half, 2):
                            div = 10000 ** (j / d_half)
                            pe[i, j] = math.sin(r / div)
                            pe[i, j+1] = math.cos(r / div)
                            pe[i, d_half+j] = math.sin(c / div)
                            pe[i, d_half+j+1] = math.cos(c / div)
                self.register_buffer("pe", pe)
            def forward(self, x):
                x = self.patch_embed(x)
                cls = self.cls.expand(x.shape[0], -1, -1)
                x = torch.cat([cls, x], dim=1)
                x[:, 1:, :] += self.pe
                return x
        return FrontEnd()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.front_end(x)  # (B, 197, 768)
        x = self.vit(x)        # (B, 197, 768)
        return x

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """返回 CLS 词元作为图像嵌入。"""
        x = self.forward(x)
        return x[:, 0, :]  # (B, 768)


def main():
    model = VisionEncoder()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    cls = model.encode_image(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"输出形状: {out.shape}")      # (2, 197, 768)
    print(f"CLS 形状: {cls.shape}")      # (2, 768)
    print(f"参数量: {params/1e6:.2f}M")  # ≈ 86M
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| Pre-LN | LayerNorm 在每个子层之前应用 |
| 自注意力 | 每个词元关注同一序列中的所有其他词元 |
| 多头 | 隐藏维度在 H 个独立注意力头之间分割 |
| FFN 扩展 | 前馈层在收缩前扩展到 4×hidden |
| CLS 汇聚 | 使用第一个词元的最终隐藏状态作为图像概要 |

---

## 5. 工程最佳实践

- **ViT 不是唯一的视觉编码器架构**：SigLIP 使用平均汇聚而非 CLS，DINOv2 使用额外注册词元。核心块结构不变。
- **ViT 需要更多数据**：与 CNN 不同，ViT 没有内置的平移不变性。在小型数据集上从头训练 ViT 会欠拟合。
- **中文场景建议**：中文 OCR 任务中，ViT 的标准 16×16 图块可能太大，无法捕捉汉字笔画细节——考虑使用 8×8 或 14×14。

---

## 6. 常见错误

- **Post-LN 与 Pre-LN 混淆**：Post-LN 在大型模型上训练不稳定。始终使用 Pre-LN。
- **CLS 汇聚位置错误**：`x[:, 0, :]` 取第一个词元，必须在最终 LayerNorm 之后。
- **注意力掩码设置错误**：视觉 Transformer 不使用因果掩码——所有词元可以双向注意。

---

## 📖 参考资料

1. [论文] Dosovitskiy et al. "An Image is Worth 16x16 Words". ICLR 2021. https://arxiv.org/abs/2010.11929
2. [论文] Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision". 2023.
3. [论文] Zhai et al. "SigLIP: Sigmoid Loss for Language Image Pre-Training". 2023.
