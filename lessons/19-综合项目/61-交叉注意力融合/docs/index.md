# 综合项目61——交叉注意力融合（Cross-Attention Fusion）

> 投影层将一个图像向量与一个标题对齐。真正的视觉语言解码器需要每个文本词元关注每个图块词元，使模型能在区域上定位每个词。交叉注意力就是这种定位的方式。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 实现查询流来自文本、键值流来自图像的多头交叉注意力
- 组合解码器块：因果自注意力 + 交叉注意力 + 前馈网络
- 正确处理掩码形状：自注意力用因果掩码，交叉注意力无掩码
- 在批量文本词元和固定图像池上运行前向传播

---

## 1. 问题

将图像词元和文本词元拼接成一个序列是一种融合选项（早期融合）。交叉注意力是另一种（晚期融合，Flamingo 提出）。在晚期融合中，文本解码器在文本词元上运行，通过每层的交叉注意力伸入图像流。

晚期融合有两个优势。第一，文本流保持干净，模型保留纯文本能力。第二，图像流每张图片只计算一次，在每个解码步骤复用——即使长标题生成也很廉价。

---

## 2. 核心概念

### 2.1 掩码形状

| 注意力 | 查询长度 | 键长度 | 掩码 | 原因 |
|:------|:--------|:------|:-----|:-----|
| 自注意力 | `Nt`（文本） | `Nt`（文本） | 因果：下三角 `(Nt, Nt)` | 自回归中不能前瞻 |
| 交叉注意力 | `Nt`（文本） | `Nv`（图像） | 无掩码 | 图像完全可见 |

### 2.2 为什么交叉注意力不掩码

图像在文本生成前完全观测到。标题的任何位置可以关注图像的任何图块——图像图块没有时间顺序。

### 2.3 KV 缓存

图像键和值在解码开始时计算一次并缓存。每个新文本词元复用缓存——ViT 只运行一次，使推理快速。

### 2.4 块组合

```text
text = text + self_attn(ln1(text))
text = text + cross_attn(ln2(text), image_tokens)
text = text + ffn(ln3(text))
```

---

## 3. 从零实现

```python
"""交叉注意力融合——多头交叉+因果自注意力+解码器块。"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads; self.d = dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)

    def forward(self, text, memory, mask=None):
        B, Nt, D = text.shape
        q = self.q(text).view(B, Nt, self.h, self.d).transpose(1, 2)
        kv = self.kv(memory)
        Nv = kv.shape[1]
        k = kv[:, :, :D].view(B, Nv, self.h, self.d).transpose(1, 2)
        v = kv[:, :, D:].view(B, Nv, self.h, self.d).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.d ** -0.5
        if mask is not None: scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, Nt, D))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads; self.d = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-2, -1)) * self.d ** -0.5
        if mask is not None: scores = scores + mask
        return self.out((scores.softmax(-1) @ v).transpose(1, 2).reshape(B, N, D))


def causal_mask(length, device="cpu"):
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device)) * float("-inf")


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim); self.ln3 = nn.LayerNorm(dim)
        self.self_attn = CausalSelfAttention(dim, heads)
        self.cross_attn = CrossAttention(dim, heads)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, text, memory):
        m = causal_mask(text.shape[1], text.device)
        text = text + self.self_attn(self.ln1(text), m)
        text = text + self.cross_attn(self.ln2(text), memory)
        text = text + self.ffn(self.ln3(text))
        return text


class VisionLanguageDecoder(nn.Module):
    def __init__(self, dim=768, heads=12, depth=4, text_vocab=256):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, text_vocab)

    def forward(self, text_ids, memory, embed):
        x = embed(text_ids)
        for layer in self.layers: x = layer(x, memory)
        return self.head(self.norm(x))


def main():
    dim = 64; heads = 4; vocab = 256
    dec = VisionLanguageDecoder(dim=dim, heads=heads, depth=2, text_vocab=vocab)
    embed = nn.Embedding(vocab, dim)
    B, Nt, Nv = 2, 10, 197
    text_ids = torch.randint(0, vocab, (B, Nt))
    memory = torch.randn(B, Nv, dim)
    logits = dec(text_ids, memory, embed)
    print(f"输出: {logits.shape}")  # (2, 10, 256)
    print(f"因果掩码: {causal_mask(Nt).shape}")  # (10, 10)
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 模型 | 融合方式 | 交叉注意力 | 特点 |
|:----|:--------|:----------|:-----|
| Flamingo | 晚期融合 | 每 K 层 | 带 tanh 门控 |
| BLIP-2 Q-Former | 学习查询池 | 32 个查询 | 压缩图像特征 |
| LLaVA | 早期融合 | 无 | 拼接图像和文本 |
| Qwen-VL | 交叉注意力 | 多层 | 多图支持 |

---

## 5. 工程最佳实践

- 因果掩码用 `triu` 上三角 + `-inf` 构建
- 交叉注意力掩码全 None（全连接可见）
- KV 缓存：计算一次图像键值，解码时复用
- **中文场景建议**：中文标题生成中，图像的局部区域定位更关键——中文描述通常比英文更简洁

---

## 6. 常见错误

- **自注意力和交叉注意力掩码混淆**：自注意力需要因果掩码，交叉注意力不需要
- **KV 缓存未冻结**：缓存的图像键值不应随文本梯度更新
- **维度不匹配**：文本 `Nt` 和图像 `Nv` 长度不同——交叉注意力处理了这一点

---

## 7. 面试考点

**Q1：为什么交叉注意力不需要掩码？**（难度：⭐⭐）

**参考答案：** 图像在文本生成前已完全观测到，没有时间顺序。每个文本位置都可以自由关注所有图像图块。掩码只用于自回归自注意力中防止文本词元看到未来。

**Q2：KV 缓存如何使推理更快？**（难度：⭐⭐⭐）

**参考答案：** 图像的键和值在第一轮解码时计算一次。后续每个词元生成时，直接使用缓存的键值与新文本查询做交叉注意力，避免重复计算图像投影。这将图像处理成本从 O(序列长度) 降到 O(1)。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 晚期融合 | 文本和视觉保持独立流，通过交叉注意力在每层桥接 |
| 交叉注意力 | 查询来自一个流，键值来自另一个流 |
| KV 缓存 | 图像键值计算一次后复用 |
| 因果掩码 | 下三角布尔掩码，防止自回归中看到未来 |

---

## 📚 小结

交叉注意力是连接视觉和语言模态的核心机制。你实现了多头交叉注意力、因果自注意力和完整的解码器块。下一节将训练这些组件进行视觉语言预训练。

---

## ✏️ 练习

1. 【实现】添加 Flamingo 风格的 tanh 门控交叉注意力
2. 【实验】对比有 KV 缓存和无缓存的推理速度差异
3. 【理解】解释为什么交叉注意力的成本是 `Nt × Nv`，在高分辨率图像下可能成为瓶颈

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 交叉注意力融合 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Alayrac et al. "Flamingo: a Visual Language Model for Few-Shot Learning". NeurIPS 2022.
2. [论文] Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training". NeurIPS 2023.
