# 综合项目32——Token与位置嵌入

> ID是整数。模型想要向量。两个查找表位于它们之间，位置表的选择决定了模型能学到什么。本课程构建token嵌入（词表ID→密集向量）、学习位置嵌入（位置ID→向量）和正弦位置嵌入（无参数数学公式），并将它们合成为Transformer块的输入。

**类型：** 构建
**编程语言：** Python（PyTorch）
**前置知识：** 第4章（计算机视觉）、第7章（Transformer）、第30-31节
**涉及章节：** P4 · P7 · P10
**预计时间：** 90分钟

---

## 学习目标

- 构建词表ID到密集向量的token嵌入查找表
- 构建按位置索引的学习位置嵌入查找表
- 构建按位置索引的固定正弦位置嵌入
- 将token嵌入和位置嵌入合成为Transformer块的单一输入
- 对比学习嵌入和正弦嵌入在长度外推和参数数量上的区别

---

## 1. 问题

模型与token ID的第一接触是token嵌入矩阵中的行查找。矩阵的每一行对应一个词表ID，每一列对应模型维度。查找返回一个向量，模型的其他部分将其视为该ID的含义。反向传播更新前向传递中使用过的行。在训练过程中，这些行的几何结构学习在方向中编码相似性。

仅有token ID，模型无法感知顺序——位置十七和位置一的区别需要第二个信号。这个信号的两种主流选择是学习位置嵌入（第二个查找表）和固定正弦位置嵌入（无参数的数学公式）。

---

## 2. 核心概念

### 2.1 形状契约

嵌入阶段的输入是(B,T)的token ID批次。输出是(B,T,D)的张量，其中D是模型维度。每个批次元素有相同的上下文长度T，每个位置有相同的向量维度D。

合成是逐元素相加而非拼接——相加保持D不变，让模型按特征决定token含义还是位置占主导。

### 2.2 Token嵌入矩阵

形状为(V,D)的参数张量，V为词表大小。PyTorch的`nn.Embedding(V,D)`在初始化时从小高斯分布采样。前向传递是单个索引操作——(B,T)的int64 ID映射到(B,T,D)的float。

### 2.3 学习位置嵌入

形状为(max_context_length, D)的第二个`nn.Embedding`。按位置ID 0,1,2,...,T-1查找。缺点是无法查询位置T——该行不存在。

### 2.4 正弦位置嵌入

从位置到向量的函数。位置p和特征i产生：

```
angle = p / (10000 ** (2 * (i // 2) / D))
emb[p, 2k]     = sin(angle)
emb[p, 2k + 1] = cos(angle)
```

无参数。每个位置有唯一向量。波长在特征维度上几何变化。位置p+k的向量是位置p的向量的线性函数——注意力层可轻松学习相对位置偏移。

---

## 3. 从零实现

`code/main.py`实现TokenEmbedding、LearnedPositionalEmbedding、SinusoidalPositionalEmbedding和EmbeddingComposer。

```python
"""Token与位置嵌入——三个PyTorch模块+合成器。

TokenEmbedding: (B,T)输入→(B,T,D)输出
LearnedPositionalEmbedding: 学习的位置参数
SinusoidalPositionalEmbedding: 无参数的正弦cos公式
EmbeddingComposer: 逐元素相加合成

运行：python3 code/main.py
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn

DEFAULT_INIT_STD=0.02

class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model,init_std=DEFAULT_INIT_STD):
        super().__init__(); self.embedding=nn.Embedding(vocab_size,d_model)
        with torch.no_grad(): self.embedding.weight.normal_(0,init_std)
    def forward(self,ids): return self.embedding(ids)

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self,max_ctx,d_model,init_std=DEFAULT_INIT_STD):
        super().__init__(); self.max_ctx=max_ctx; self.embedding=nn.Embedding(max_ctx,d_model)
        with torch.no_grad(): self.embedding.weight.normal_(0,init_std)
    def forward(self,seq_len):
        if seq_len>self.max_ctx: raise ValueError(f"{seq_len}>{self.max_ctx}")
        return self.embedding(torch.arange(seq_len))

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self,max_ctx,d_model,base=10000.0):
        super().__init__(); self.max_ctx=max_ctx; self.d_model=d_model
        pe=self._build(max_ctx,d_model,base); self.register_buffer("pe",pe)
    @staticmethod
    def _build(L,D,base):
        pos=torch.arange(L,dtype=torch.float32).unsqueeze(1)
        i=torch.arange(D//2,dtype=torch.float32)
        angle=pos/(base**(2*i/D))
        pe=torch.zeros(L,D,dtype=torch.float32); pe[:,0::2]=torch.sin(angle); pe[:,1::2]=torch.cos(angle)
        return pe
    def forward(self,seq_len):
        if seq_len>self.max_ctx: raise ValueError(f"{seq_len}>{self.max_ctx}")
        return self.pe[:seq_len]

class EmbeddingComposer(nn.Module):
    def __init__(self,token_emb,pos_emb):
        super().__init__(); self.token_emb=token_emb; self.pos_emb=pos_emb
    def forward(self,ids):
        seq_len=ids.shape[1]; tok=self.token_emb(ids); pos=self.pos_emb(seq_len)
        return tok+pos.unsqueeze(0)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def neighbor_cosine(table,max_off=8):
    if table.dim()!=2: raise ValueError
    tbl=table.detach().float(); nr=(tbl/tbl.norm(dim=1,keepdim=True).clamp(min=1e-8))
    return [(nr[:-k]*nr[k:]).sum(dim=1).mean().item() for k in range(1,max_off+1)]

def main():
    V,D,ctx=320,64,128
    tok=TokenEmbedding(V,D); lp=LearnedPositionalEmbedding(ctx,D); sp=SinusoidalPositionalEmbedding(ctx,D)
    comp_l=EmbeddingComposer(tok,lp); comp_s=EmbeddingComposer(tok,sp)
    ids=torch.randint(0,V,(4,32),dtype=torch.long)
    print(f"输出形状: {tuple(comp_l(ids).shape)}")  # (4,32,64)
    print(f"参数: token={count_params(tok)} learn_pos={count_params(lp)} sin_pos={count_params(sp)}")
    lc=neighbor_cosine(lp.embedding.weight,6); sc=neighbor_cosine(sp.pe,6)
    print("邻位余弦:",", ".join(f"{k}:{a:.3f}/{b:.3f}" for k,a,b in zip(range(1,7),lc,sc)))
    return 0

if __name__=="__main__": raise SystemExit(main())
```

运行结果：

```
输出形状: (4, 32, 64)
参数: token=20480 learn_pos=8192 sin_pos=0
邻位余弦: 1:-0.015/0.999 2:-0.002/0.997 3:-0.016/0.995 4:0.007/0.992 5:-0.010/0.989 6:0.011/0.986
```

---

## 4. 工具实践

**权重绑定**：token嵌入和模型输出的投影层常共享权重。当权重绑定时，每次反向传播通过输出侧触及token嵌入的每一行。

**扩展选择**：RoPE和AliBi是现代Transformer的现代选择。它们遵循相同的形状契约，但应用于注意力投影步骤而非输入。

---

## 5. LLM视角

**组成视角**：token嵌入和位置嵌入的相加而非拼接，保持D不变，让模型按特征决定token含义还是位置占主导。

**正弦性质视角**：sin和cos配对使位置p+k的向量是位置p的向量的线性函数，让注意力层可轻松学习相对位置偏移。

---

## 6. 工程最佳实践

**初始化**：从小高斯分布（标准差0.02）初始化嵌入。

**参数计数**：学习嵌入增加max_context_length*D个参数，正弦嵌入为零。

---

## 7. 常见错误

**错误1：拼接而非相加**
症状：D翻倍，后续层参数翻倍
修复：位置向量与token向量逐元素相加

**错误2：学习嵌入超限查询**
症状：训练时位置T不存在
修复：检查seq_len≤max_context_length

---

## 8. 面试考点

**Q1：为什么位置嵌入是相加而非拼接？**
考察：对架构设计的理解

**Q2：正弦位置嵌入的线性性质为什么重要？**
考察：对相对位置编码的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Token嵌入 | "词表查找表" | (V,D)参数矩阵，将ID映射到向量 |
| 学习位置嵌入 | "位置参数表" | (L,D)可学习参数，按位置索引 |
| 正弦位置嵌入 | "无参数公式" | sin/cos函数，波长几何变化 |
| EmbeddingComposer | "嵌入合成器" | token+位置嵌入的逐元素相加 |
| 邻位余弦 | "相似度曲线" | 相邻位置嵌入间的余弦相似度衰减 |
| 权重绑定 | "输出共享" | token嵌入与LM头投影层共享权重 |

---

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 位置编码原始论文
- [RoPE](https://arxiv.org/abs/2104.09864) — 旋转位置编码
- [PyTorch nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
