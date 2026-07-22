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
```

---

## 4. 工业工具

| 工具 | 用途 |
|:-----|:-----|
| NumPy `np.einsum` | CPU 上的 einsum |
| PyTorch `torch.einsum` | GPU 上的 einsum |
| einops | 更易读的张量重塑 |

---

## 5. 知识连线

- **第 07 阶段 · 02（自注意力）**：注意力中的每一步都是张量操作
- **第 10 阶段 · 04（嵌入）**：Token 嵌入和位置嵌入是张量重塑的直接应用

---

## 6. 工程最佳实践

- **形状调试**：在代码中添加 `assert` 验证形状，尽早暴露问题
- **广播陷阱**：隐式广播可能产生意外结果，显式 `expand` 更安全
- **中文场景建议**：中文文本的张量形状通常比英文更大（词元数更多），内存管理更重要

---

## 7. 常见错误

### 错误 1：视图和副本混淆

**现象：** 修改张量 A，张量 B 也变了。

**原因：** NumPy/PyTorch 的 `view`/`reshape` 可能共享内存。

**修复：** 使用 `copy()` 创建独立副本。

### 错误 2：einsum 形状不匹配

**现象：** `ValueError: operands could not be broadcast`。

**原因：** einsum 的轴索引不对应。

**修复：** 先用注释写出每个张量的形状，再写 einsum 表达式。

---

## 8. 面试考点

### Q1：广播规则是什么？为什么它对 AI 很重要？（难度：⭐⭐）

**参考答案：** 广播允许对不同形状的张量运算——从右对齐维度，缺失维度在左侧填 1。它是 PyTorch/NumPy 中最常见的形状错误来源，因为隐式广播可能在错误维度上产生结果。

### Q2：einsum 相比矩阵乘法的优势？（难度：⭐⭐）

**参考答案：** einsum 用字符串表达式一次定义任意张量缩并，比链式矩阵乘法更可读。例如 `bhtd,bhsd->bhts` 直接表达了注意力分数的计算，而等价的矩阵乘法需要多次 `reshape` 和 `transpose`。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| 张量 | 多维数组的推广——形状 (D₁, D₂, ..., Dₙ) |
| 广播 | 从右对齐维度，缺失填 1 的隐式扩展 |
| einsum | 用轴索引字符串表达张量缩并的语法 |
| 转置 | 交换两个维度——不修改数据，只改变视图 |
| 梯度形状 | 自动微分中张量形状与前向一致 |

---

## 📚 小结

张量运算是深度学习的语言——每个模型都是张量操作的序列。你从零实现了张量的创建、重塑、转置、广播和 einsum，理解了注意力机制中每一步的形状变化。这些技能在调试形状错误时直接可用。

---

## ✏️ 练习

1. 【实现】用 einsum 实现注意力：`(bhtd,bhsd)->bhts`，然后 `softmax`，再 `(bhts,bhsd)->bhtd`
2. 【实验】构造形状不匹配的广播，观察隐式扩展的行为
3. 【理解】在调试模式下为模型每层添加形状断言

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|:-----|:-----|:-----|
| 张量运算 | `code/tensors.py` | 从零张量类 + einsum |
| 提示词 | `outputs/prompt-tensor-debugger.md` | 调试张量形状错误 |

---

## 📖 参考资料

1. [官方文档] PyTorch 张量视图. https://pytorch.org/docs/stable/tensor_view.html
2. [官方文档] einops. https://github.com/arogozhnikov/einops
3. [官方文档] NumPy einsum. https://numpy.org/doc/stable/reference/generated/numpy.einsum.html

---

## 4. 工业工具

| 工具 | 用途 |
|:-----|:-----|
| NumPy `np.einsum` | CPU einsum 运算 |
| PyTorch `torch.einsum` | GPU einsum 运算 |
| einops | 高级张量重塑库 |

---

## 5. 知识连线

- **第 07 阶段 · 02（自注意力）**：注意力中的每一步都是张量操作
- **第 10 阶段 · 04（嵌入）**：Token 嵌入和位置嵌入是张量重塑的直接应用

---

## 6. 工程最佳实践

- **形状调试**：在代码中添加断言验证张量形状，尽早发现错误
- **广播要显式**：显式 `expand` 比隐式广播更安全，避免静默错误
- **中文场景建议**：中文词表更大，嵌入矩阵更大——注意内存管理

---

## 7. 常见错误

### 错误 1：视图和副本混淆

**现象：** 修改张量 A 后张量 B 也变了。

**修复：** NumPy/PyTorch 的 view/reshape 可能共享内存。使用 `copy()` 创建独立副本。

### 错误 2：einsum 形状不匹配

**修复：** 先用注释写出每个张量的形状，再写 einsum 表达式。

---

## 8. 面试考点

### Q1：广播规则是什么？为什么对 AI 很重要？（难度：⭐⭐）

**参考答案：** 广播允许对不同形状张量运算——从右对齐维度，缺失填 1。它是形状错误的最常见来源，因为隐式扩展可能在错误维度产生结果。

### Q2：einsum 相比矩阵乘法的优势？（难度：⭐⭐）

**参考答案：** einsum 用字符串表达式一次定义任意张量缩并，比链式矩阵乘法更可读。`bhtd,bhsd->bhts` 直接表达注意力分数计算，而等价的矩阵乘法需要多次 reshape 和 transpose。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| 张量 | 多维数组，形状 (D₁, D₂, ..., Dₙ) |
| 广播 | 从右对齐维度，缺失填 1 的隐式扩展 |
| einsum | 用轴索引字符串表达张量缩并的语法 |
| 步幅 | 从多维索引到扁平内存位置的映射 |
| 转置 | 交换两个维度——不修改数据，只改变视图 |

---

## 📚 小结

张量运算是深度学习的语言——每个模型都是张量操作的序列。你从零实现了张量的创建、重塑、转置、广播和 einsum，理解了注意力机制中每一步的形状变化。

---

## ✏️ 练习

1. 【实现】用 einsum 实现注意力：`(bhtd,bhsd)->bhts`，然后 `softmax`，再 `(bhts,bhsd)->bhtd`
2. 【实验】构造形状不匹配的广播，观察隐式扩展的行为
3. 【理解】在调试模式下为模型每层添加形状断言

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|:-----|:-----|:-----|
| 张量运算 | `code/tensors.py` | 从零张量类 + einsum |

---

## 📖 参考资料

1. [官方文档] PyTorch 张量视图. https://pytorch.org/docs/stable/tensor_view.html
2. [官方文档] einops. https://github.com/arogozhnikov/einops
3. [官方文档] NumPy einsum. https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
