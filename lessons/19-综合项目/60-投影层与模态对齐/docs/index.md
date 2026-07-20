# 综合项目60——投影层与模态对齐（Projection Layer for Modality Alignment）

> 视觉编码器产生图像词元，文本解码器消费文本词元。两者在不同向量空间中。一个小型两层 MLP 将它们对齐——这是视觉语言模型中最小也最关键的部分。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第58-59节
**预计时间：** 90分钟

---

## 学习目标

- 构建将图像特征映射到文本嵌入空间的 MLP 投影器
- 构建模拟文本嵌入表
- 计算余弦对齐损失
- 冻结编码器和文本表，单独训练投影层

---

## 1. 问题

视觉编码器产生 `vision_hidden=768` 的特征。文本解码器期望 `text_hidden=512`。两者基不同——线性层不够（曲率不匹配），需要两层 MLP + GELU。

```
图像 [768] → Linear → GELU → Linear → [512] → 与标题嵌入对齐
```

---

## 2. 核心概念

### 2.1 池化前投影

视觉编码器输出 197 词元。需要 1 个图像级向量。CLS 取第 0 词元，平均取全部。

### 2.2 两层 vs 单层

GELU 给非线性弯折——足以对齐 CLIP 特征到语言嵌入。约 1.3M 参数。

### 2.3 余弦对齐

```
loss = 1 - cos_sim(image_emb, text_emb)  ∈ [0, 2]
```

### 2.4 冻结策略

86M 视觉编码器 + 文本表冻结，1.3M 投影器是唯一训练参数。

---

## 3. 从零实现

```python
"""投影层与模态对齐——MLP+余弦损失。"""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F


class MLPProjector(nn.Module):
    def __init__(self, in_dim=768, h=1024, out=512):
        super().__init__()
        self.fc1=nn.Linear(in_dim,h); self.fc2=nn.Linear(h,out)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(x)))


class MockTextEmbedding(nn.Module):
    def __init__(self, vocab=128, dim=512, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.table=nn.Embedding(vocab,dim)
        for p in self.parameters(): p.requires_grad=False
    def forward(self,ids): return self.table(ids)


def cosine_alignment_loss(img_emb, txt_emb):
    return (1 - F.cosine_similarity(F.normalize(img_emb,-1), F.normalize(txt_emb,-1))).mean()


def make_pair(seed, vocab=128, cap_len=8, vdim=768, tdim=512):
    torch.manual_seed(seed)
    return torch.randn(vdim), torch.randint(2,vocab,(cap_len,))


def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    proj=MLPProjector().to(device)
    te=MockTextEmbedding().to(device)
    opt=torch.optim.Adam(proj.parameters(),lr=1e-3)
    pairs=[make_pair(i) for i in range(32)]

    print(f"训练投影器 ({sum(p.numel() for p in proj.parameters()):,} 参数)")
    for step in range(200):
        total=0
        for feat,cap in pairs:
            f=feat.unsqueeze(0).to(device); c=cap.unsqueeze(0).to(device)
            ip=proj(f); tc=te(c).mean(1)
            loss=cosine_alignment_loss(ip,tc)
            opt.zero_grad(); loss.backward(); opt.step()
            total+=loss.item()
        if step%50==0 or step==199:
            print(f"  步 {step:3d}: 损失={total/32:.4f}")

    with torch.no_grad():
        f,c=pairs[0]
        cos=F.cosine_similarity(proj(f.unsqueeze(0).to(device)), te(c.unsqueeze(0).to(device)).mean(1)).item()
    print(f"\n最终余弦相似度: {cos:.4f}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 模型 | 投影器 | 参数 | 对齐方法 |
|:----|:------|:----|:--------|
| LLaVA 1.5 | 2 层 MLP (GELU) | 1.3M | 对比损失 |
| BLIP-2 | Q-Former | 188M | 对比损失 |
| Qwen-VL | 交叉注意力 | 200M | 对比损失 |
| MiniGPT-4 | 单层线性 | ~4M | 无 |

---

## 5. 工程最佳实践

- **LLaVA 两阶段**：先冻结投影器训练，再解冻 LLM LoRA 微调
- **CLS vs 平均池化**：分类用 CLS，细粒度识别用平均
- **中文场景建议**：中文视觉语言模型用相同投影架构，后端替换为中文优化 LLM

---

## 6. 常见错误

- **未冻结视觉编码器**：反向传播 86M 参数浪费时间且效果差
- **余弦损失缺温度参数**：可学习温度改善收敛
- **投影器初始化不当**：Xavier 或正态小随机初始化比零初始化好

---

## 7. 面试考点

**Q1：为什么两层 MLP 而非单层？**（难度：⭐⭐）

**参考答案：** 单层只能旋转和缩放。如果两空间有曲率不匹配（语义维度方向不同），无法对齐。GELU 提供非线性弯折——经验上足以对齐视觉-语言特征。

**Q2：为什么冻结编码器时损失仍能下降？**（难度：⭐⭐⭐）

**参考答案：** 投影器将编码器输出映射到一个与文本空间相近的中间空间。即使编码器特征固定，投影器学到的映射已足以改善对齐。这是典型的适配器模式——冻结主干，只训练轻量桥接层。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 模态对齐 | 使图像和文本嵌入在联合空间可比较 |
| 投影头 | 连接两个空间的小模块，通常 2 层 MLP |
| 余弦损失 | `1 - cos_sim`，值越小对齐越好 |
| 冻结编码器 | 参数 `requires_grad=False` |

---

## 📚 小结

投影层将冻结的视觉编码器输出对齐到语言空间——这是 LLaVA、BLIP-2 等视觉语言模型的核心组件。你实现了 MLP 投影器和余弦对齐损失，在合成数据上演示了训练收敛。

---

## ✏️ 练习

1. 【实验】对比 CLS 池化和平均池化在 200 步后的最终损失
2. 【实现】添加可学习温度参数 `log_tau` 到余弦损失
3. 【思考】BLIP-2 的 Q-Former 相比简单 MLP 有什么优势？为什么不直接用 MLP？

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 投影层 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Liu et al. "LLaVA: Large Language and Vision Assistant". NeurIPS 2023.
2. [论文] Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training". NeurIPS 2023.
3. [论文] Bai et al. "Qwen-VL: A Versatile Vision-Language Model". 2023.
