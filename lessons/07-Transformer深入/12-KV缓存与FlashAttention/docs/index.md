# KV 缓存与 Flash Attention

> KV 缓存让自回归生成不用每步重算；Flash Attention 让 O(N²) 的注意力在硬件上跑出接近 O(N) 的性能。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 07 · 05（完整 Transformer）| **时间：** ~45 分钟

---

## 🎯 学习目标

- [ ] 理解 KV 缓存——为什么自回归解码时不需要重新计算所有历史位置的 K 和 V
- [ ] 解释 Flash Attention 的核心思想——分块计算 + 在线 softmax 代替 O(N) 归约
- [ ] 说明 KV 缓存如何将 Transformer 的推理从 O(N²) 降低到每步 O(1)

---

## 1. 问题

### KV 缓存——解码时的巨大浪费

在自回归生成时，每生成一个新词元，模型需要重新计算**所有历史位置**的 K 和 V。但历史位置的 K、V 不会改变——它们的输入已经固定。**每步重算是浪费。**

解决方案：**缓存**。计算一次 K 和 V，存储在 GPU 内存中。下一步只计算新词元的 K 和 V，与缓存拼接。

```python
# 无缓存：每步重算所有位置（O(N²) 时间 × N 步）
# 有缓存：每步只算 1 个新位置（O(N) 时间 × N 步）
```

### Flash Attention——O(N²) 内存但 O(N) 时间

标准注意力的内存瓶颈：`Q @ K^T` 需要 O(N²) 内存存储完整的分数矩阵。

Flash Attention 的技巧：**分块计算**。将 Q/K/V 分成小块，每块计算注意力后立即丢弃分数矩阵——只保留加权求和的结果。内存从 O(N²) 降到 O(N)，但计算仍是 O(N²)——只是不存储中间结果。

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| KV 缓存 | 缓存历史位置的 Key 和 Value，避免自回归解码时重复计算 |
| Flash Attention | 分块注意力计算——不存储 O(N²) 的分数矩阵，内存 O(N) |
| 推理加速 | KV 缓存将每步推理从 O(N) 降到 O(1)；Flash Attention 将内存从 O(N²) 降到 O(N) |

---

## 📚 小结

KV 缓存：解码时只计算新词元的 K/V，与历史缓存拼接——推理速度从 O(N²) 降到每步 O(N)。Flash Attention：分块计算注意力，不存储中间分数矩阵——内存从 O(N²) 降到 O(N)，硬件效率提升 2-4 倍。两者结合使 128K 上下文的 Transformer 成为可能。

---

## ✏️ 练习

1. 手动推导 KV 缓存的内存增长：128K 上下文 × 32 层 × 1280 维 × 2 字节 = ? GB
2. 对比有/无 Flash Attention 的 PyTorch Transformer 在 4K 序列上的 GPU 内存占用

---

## 📖 参考资料

1. [论文] Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". NeurIPS, 2022.
2. [论文] Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning". 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
