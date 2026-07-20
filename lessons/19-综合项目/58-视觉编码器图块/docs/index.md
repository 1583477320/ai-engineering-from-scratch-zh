# 综合项目58——视觉编码器图块（Vision Encoder Patches）

> 读取像素的视觉模型需要像素的分词器。图块嵌入就是这个分词器。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 将图像词元化为固定长度的图块嵌入序列
- 实现基于 Conv2d 的图块投影
- 构建确定性 2D 正弦位置嵌入
- 验证图块数量、形状和 Conv2d/unfold 等价性

---

## 1. 问题

Transformer 处理向量序列。图像是 3 通道网格。每像素一个词元→150,528 词元（注意力爆炸）。展平为单个向量→丢失局部性。图块嵌入将图像切分为 16×16 方块，每块展平后线性投影。

---

## 2. 核心概念

### 2.1 为什么是图块而非像素

注意力是序列长度的二次函数。196 图块 = 38,416 注意力分数/层；150,528 像素 = 226 亿。图块带来 590,000 倍缩减。

### 2.2 Conv2d 技巧

`Conv2d(3, hidden, 16, stride=16)` 输出与 unfold-then-linear 数值等价——GPU 更快且少一次 reshape。

### 2.3 2D 正弦位置

```
PE(row, col, 2i)   = sin(row / 10000^(2i/d))
PE(row, col, d+2i) = sin(col / 10000^(2i/d))
```

| 组件 | 参数量 |
|:----|:------|
| 图块投影 Conv2d | `3 × 16² × 768 + 768 ≈ 590K` |
| 2D 正弦位置 | 0（固定） |
| CLS 词元 | 768 |

---

## 3. 从零实现

```python
"""视觉编码器图块——Conv2d 投影+2D 正弦位置。"""
from __future__ import annotations
import math, torch, torch.nn as nn


def sinusoidal_2d(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    assert dim % 4 == 0
    pe = torch.zeros(grid_h * grid_w, dim)
    for r in range(grid_h):
        for c in range(grid_w):
            idx = r * grid_w + c
            d = dim // 2
            div = 10000 ** (torch.arange(0, d, 2) / d)
            pe[idx, 0:d:2] = torch.sin(torch.tensor(r/div))
            pe[idx, 1:d:2] = torch.cos(torch.tensor(r/div))
            pe[idx, d::2]  = torch.sin(torch.tensor(c/div))
            pe[idx, d+1::2]= torch.cos(torch.tensor(c/div))
    return pe


class PatchEmbed(nn.Module):
    def __init__(self, img=224, patch=16, ch=3, dim=768):
        super().__init__()
        self.patch = patch; self.g = img // patch; self.n = self.g ** 2
        self.proj = nn.Conv2d(ch, dim, patch, stride=patch)
    def forward(self, x): return self.proj(x).flatten(2).transpose(1,2)


class VisionFrontEnd(nn.Module):
    def __init__(self, img=224, patch=16, ch=3, dim=768):
        super().__init__()
        self.pe = PatchEmbed(img, patch, ch, dim)
        self.cls = nn.Parameter(torch.randn(1,1,dim)*0.02)
        self.register_buffer("pe_2d", sinusoidal_2d(self.pe.g, self.pe.g, dim))
    def forward(self, x):
        B = x.shape[0]; x = self.pe(x)
        return torch.cat([self.cls.expand(B,-1,-1), x], dim=1) + \
               torch.cat([torch.zeros(1,1,self.pe_2d.shape[1]), self.pe_2d.unsqueeze(0)], dim=1)


def main():
    m = VisionFrontEnd()
    out = m(torch.randn(2,3,224,224))
    print(f"输出: {out.shape}")  # (2, 197, 768)
    # Conv2d vs Unfold 等价
    conv = m.pe.proj; unfold = nn.Unfold(16, stride=16); lin = nn.Linear(768,768)
    lin.weight.data = conv.weight.data.view(768,-1); lin.bias.data = conv.bias.data
    diff = (m(torch.randn(1,3,32,32))[:,:,:] - m(torch.randn(1,3,32,32))[:,:,:]).abs().max().item()
    print(f"图块数量: {m.pe.n}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 模型 | 图块大小 | 投影方式 | 位置嵌入 |
|:----|:--------|:--------|:--------|
| ViT | 16×16 | Conv2d | 可学习 1D |
| SigLIP | 14×14 | Conv2d | 正弦 |
| CLIP ViT | 14×14 | Conv2d | 可学习 1D |
| DINOv2 | 14×14 | Conv2d | 2D 正弦 |

---

## 5. 工程最佳实践

- 选择合适图块大小：14×14（256 图块）vs 16×16（196 图块）
- CLS 不接收位置信号——设计决定
- **中文场景建议**：AI 绘画模型也使用类似结构——Stable Diffusion VAE 编码器

---

## 6. 常见错误

- **Conv2d reshape 错误**：`flatten(2)` 而非 `flatten(1)`
- **CLS 接收位置信号**：`x[:,1:,:] += pe` 排除 CLS
- **dim 不被 4 整除**：正弦 2D 要求 dim % 4 == 0

---

## 7. 面试考点

**Q1：为什么图块大小 16×16 而非更小？**（难度：⭐⭐）

**参考答案：** 更小的图块增加序列长度（二次注意力成本），但捕捉更细粒度的细节。16×16 是 224×224 输入下的精度-成本平衡点。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 图块嵌入 | 展平图块 → 线性投影到隐藏维度 |
| 正弦位置 | 固定 sin/cos 编码 2D 网格坐标 |
| CLS 词元 | 可学习向量，汇聚全局信息 |

---

## 📚 小结

图块嵌入将图像压缩为 Transformer 可处理的序列。你实现了 Conv2d 投影和 2D 正弦位置，验证了与 unfold 的等价性。下一节构建 12 层 ViT 编码器。

---

## ✏️ 练习

1. 【实现】将正弦位置替换为可学习 `nn.Parameter`，比较收敛速度
2. 【实验】支持非正方形图块（如 32×16），验证位置表正确

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 图块嵌入 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Dosovitskiy et al. "An Image is Worth 16x16 Words". ICLR 2021.
2. [论文] Vaswani et al. "Attention Is All You Need". NeurIPS 2017.
