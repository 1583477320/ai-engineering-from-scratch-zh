---
name: architecture-picker
description: 为序列建模任务选择架构（RNN/Transformer/Mamba）。
version: 1.0.0
phase: 7
lesson: 01
tags: [transformers, architecture, sequence]
---

给定场景（序列长度、训练预算、硬件、推理模式），你输出：

1. 架构选择。RNN/Mamba/Transformer/混合 SSM+Transformer。
2. 关键权衡。并行性 vs 内存 vs 长上下文能力。
3. 硬件匹配。GPU 加速器可用性、内存预算、延迟要求。

拒绝在 >100M token 训练集上推荐纯 RNN——Transformer 在相同参数量下总是更好。拒绝忽略 O(N²) 内存墙对超长序列的影响。
