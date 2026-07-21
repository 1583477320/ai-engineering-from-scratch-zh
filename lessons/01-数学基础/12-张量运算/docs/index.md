# 张量运算——数据与深度学习的通用语言

> 张量是数据和深度学习之间的通用语言。每张图像、每个句子、每个梯度都流经张量。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第 01 阶段 · 01-02
**预计时间：** 90 分钟
**所处阶段：** Tier 1
**关联课程：** 第 07 阶段 · 02（多头注意力）— 注意力中的每一步都是张量运算

---

## 🎯 学习目标

- [ ] 实现从零的张量类：形状、步幅、重塑、转置和逐元素运算
- [ ] 应用广播规则对不同形状的张量运算而不复制数据
- [ ] 编写 einsum 表达式用于点积、矩阵乘法、外积和批处理操作
- [ ] 追踪多头注意力每步的精确张量形状

---

## 1. 问题

构建 Transformer。前向传播看起来清晰。运行得到：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)`。形状错误是深度学习代码中最常见的 bug——每个操作都有形状合约，但错误会级联。

矩阵处理两组事物之间的成对关系。真实数据不拟合两个维度。一批 32 张 RGB 图像是 4D 张量：`(32, 3, 224, 224)`。12 头自注意力也是 4D。你需要一个能推广到任意维度的数据结构——那就是张量。

---

## 2. 核心概念

### 2.1 张量形状

```
视觉: (B, C, H, W) = (32, 3, 224, 224)
NLP:  (B, T, D) = (16, 128, 768)
注意力: (B, H, T, D) = (16, 12, 128, 64)
权重: 线性 (out, in), 卷积 (out_c, in_c, kH, kW), 嵌入 (vocab, dim)
```

### 2.2 广播规则

从右对齐形状。两个维度相等或其中一个为 1 时兼容。少的维度在左边补 1。

### 2.3 Einsum 万能张量操作

```
i,i->       点积
ij,jk->ik   矩阵乘法
i,j->ij     外积
bij,bjk->bik  批处理矩阵乘法
bhtd,bhsd->bhts  注意力分数
```

### 2.4 注意力中的形状追踪

```
输入: (B, T, E) → Q/K/V投影: (B, T, E) → 拆头: (B, H, T, D)
→ 注意力分数: (B, H, T, T) → softmax → 加权求和: (B, H, T, D)
→ 合并: (B, T, E) → 输出投影: (B, T, E)
```

---

## 3. 从零实现

```python
"""张量运算——广播+einsum+注意力形状。"""
import numpy as np

def demo_broadcasting():
    acts = np.random.randn(4, 3)
    bias = np.array([0.1, 0.2, 0.3])
    print(f"激活 {acts.shape} + 偏置 {bias.shape} = {(acts+bias).shape}")
    a = np.array([1,2,3]).reshape(-1,1)
    b = np.array([10,20,30,40]).reshape(1,-1)
    print(f"外积 {a.shape} × {b.shape} = {(a*b).shape}")

def demo_einsum():
    A = np.random.randn(3, 4); B = np.random.randn(4, 5)
    print(f"矩阵乘法: {np.einsum('ik,kj->ij',A,B).shape}")
    ba = np.random.randn(4, 3, 5); bb = np.random.randn(4, 5, 2)
    print(f"批处理: {np.einsum('bij,bjk->bik',ba,bb).shape}")

def demo_attention():
    B, H, T, D = 2, 4, 8, 16
    X = np.random.randn(B, T, H*D)
    Wq = np.random.randn(H*D, H*D) * 0.02
    Q = np.einsum("bte,ek->btk", X, Wq).reshape(B, T, H, D).transpose(0,2,1,3)
    print(f"Q 形状: {Q.shape}")
    scores = np.einsum("bhtd,bhsd->bhts", Q, Q) / np.sqrt(D)
    print(f"注意力分数: {scores.shape}")

def main():
    demo_broadcasting(); demo_einsum(); demo_attention()
    return 0

if __name__ == "__main__": import sys; sys.exit(main())
