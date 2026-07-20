# 综合项目59——视觉 Transformer 编码器（Vision Transformer Encoder）

> 图块本身看不见。12 层 pre-LN Transformer 将图块序列转化为上下文词元序列，CLS 词元汇聚整张图像特征。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第58节
**预计时间：** 90分钟

---

## 学习目标

- 实现 pre-LN Transformer 块：多头自注意力 + FFN
- 堆叠 12 块 12 头构建 ViT-Base
- 将图块前端接入编码器运行前向传播
- 验证 CLS 词元从所有图块汇聚信息

---

## 1. 问题

图块嵌入产生 197 个词元序列，每个向量对其它块一无所知。Transformer 构建图块间意识——一层注意力地构建。没有它，前端就是聪明但无理解的分词器。

标准配方：12 层深、12 头宽、pre-LN、GELU、4x FFN。这是 CLIP ViT-L、SigLIP、DINOv2 的骨架。

---

## 2. 核心概念

### 2.1 Pre-LN vs Post-LN

```
Post-LN: x = LN(x + SubLayer(x))   # 不稳定
Pre-LN:  x = x + SubLayer(LN(x))   # 稳定，现代标准
```

### 2.2 4x FFN 扩展

`hidden → 4×hidden → hidden` + GELU。4x 是经验最优：2x 欠拟合，8x 过拟合。

### 2.3 ViT-Base 参数量

| 组件 | 参数 |
|:----|:----|
| qkv（每块） | 1.77M |
| 输出投影（每块） | 590K |
| FFN（每块） | 4.72M |
| 12 块合计 | ≈ 85M |
| 加前端 | ≈ 86M |

---

## 3. 从零实现

```python
"""视觉 Transformer 编码器——Pre-LN + 12 层。"""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
import sys

class MHSA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads; self.d = dim // heads
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B,N,D = x.shape
        q,k,v = self.qkv(x).reshape(B,N,3,self.h,self.d).permute(2,0,3,1,4)
        a = (q@k.transpose(-2,-1))*self.d**-0.5; a = a.softmax(-1)
        return self.out((a@v).transpose(1,2).reshape(B,N,D))

class FFN(nn.Module):
    def __init__(self, dim, exp=4):
        super().__init__()
        self.fc1=nn.Linear(dim,dim*exp); self.fc2=nn.Linear(dim*exp,dim)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1=nn.LayerNorm(dim); self.attn=MHSA(dim,heads)
        self.ln2=nn.LayerNorm(dim); self.ffn=FFN(dim)
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); return x+self.ffn(self.ln2(x))

class ViT(nn.Module):
    def __init__(self, dim=768, heads=12, depth=12):
        super().__init__()
        self.blocks=nn.ModuleList([Block(dim,heads) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim)
    def forward(self, x):
        for b in self.blocks: x=b(x)
        return self.norm(x)

class VisionFrontEnd(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.proj=nn.Conv2d(3,dim,16,stride=16)
        self.cls=nn.Parameter(torch.randn(1,1,dim)*0.02)
        g=14; self.register_buffer("pe",
            torch.cat([torch.zeros(1,g*g,dim)]+
            [torch.cat([torch.sin(torch.arange(g*g).float()/10000**(2*i/dim).expand(1,g*g)),
                        torch.cos(torch.arange(g*g).float()/10000**(2*i/dim).expand(1,g*g)),
                        torch.sin(torch.arange(g*g).float()/10000**(2*(i+dim//2)/dim).expand(1,g*g)),
                        torch.cos(torch.arange(g*g).float()/10000**(2*(i+dim//2)/dim).expand(1,g*g))],0) 
            for i in range(dim//4)],0)) if False else torch.zeros(1,196,dim))
    def forward(self,x):
        B=x.shape[0]; x=self.proj(x).flatten(2).transpose(1,2)
        return torch.cat([self.cls.expand(B,-1,-1),x],dim=1)

class VisionEncoder(nn.Module):
    def __init__(self, dim=768, heads=12, depth=12):
        super().__init__()
        self.fe=VisionFrontEnd(dim); self.vit=ViT(dim,heads,depth)
    def forward(self,x): return self.vit(self.fe(x))
    def encode_image(self,x): return self.forward(x)[:,0,:]

def main():
    m=VisionEncoder(); x=torch.randn(2,3,224,224)
    out=m(x); cls=m.encode_image(x)
    params=sum(p.numel() for p in m.parameters())
    print(f"输出: {out.shape} CLS: {cls.shape} 参数: {params/1e6:.2f}M")
    return 0

if __name__=="__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 模型 | 深度 | 宽度 | 参数量 | CLS vs 平均 |
|:----|:----|:----|:------|:-----------|
| ViT-Base | 12 | 768 | 86M | CLS |
| ViT-Large | 24 | 1024 | 304M | CLS |
| SigLIP-So400M | 27 | 1152 | 400M | 平均 |
| DINOv2-ViT-L | 24 | 1024 | 300M | CLS + 注册 |

---

## 5. 工程最佳实践

- **始终用 Pre-LN**：Post-LN 在大型模型上不稳定
- **ViT 需要更多数据**：与 CNN 不同，ViT 无内置平移不变性
- **中文场景建议**：OCR 中 16×16 图块可能太大，考虑 8×8 或 14×14

---

## 6. 常见错误

- **Post-LN 误用**：大型模型必须 Pre-LN
- **CLS 汇聚位置**：`x[:,0,:]` 在最终 LayerNorm 之后取
- **因果掩码误用**：ViT 是编码器，不使用因果掩码

---

## 7. 面试考点

**Q1：为什么 FFN 使用 4x 扩展？**（难度：⭐⭐）

**参考答案：** 经验最优：2x 容量不足，8x 在固定数据下过拟合。MLP 是存储模型知识的主要位置——更宽的中间层容纳更多知识。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| Pre-LN | 每个子层前 LayerNorm |
| 多头注意力 | 隐藏维度分 H 个独立头 |
| FFN 扩展 | 4×hidden 中间层 |
| CLS 池化 | 用第一个词元最终状态作图像摘要 |

---

## 📚 小结

ViT 编码器将图块序列转化为语境化的图像表示。你构建了 pre-LN 块和 12 层堆叠。下一节构建投影层将视觉特征对齐到语言空间。

---

## ✏️ 练习

1. 【实现】添加 4 个注册词元，比较注意力熵
2. 【实验】用 `torch.profiler` 分析哪层（MLP vs 注意力）是瓶颈

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| ViT 编码器 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Dosovitskiy et al. "An Image is Worth 16x16 Words". ICLR 2021.
2. [论文] Oquab et al. "DINOv2". 2023.
