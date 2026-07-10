---
name: attention-shapes
description: 调试注意力实现中的形状 bug。
phase: 5
lesson: 10
---

给定一个损坏的注意力实现，你识别形状不匹配。输出：

1. 哪个矩阵的 shape 错了。命名张量。
2. 它的 shape 应该是什么。从 (d_s, d_h, d_attn, T_enc, T_dec, batch) 推导。
3. 一行修复。transpose、reshape 或 project。
4. 一个回归测试。`assert output.shape == (batch, T_dec, d_h)`，`assert weights.shape == (batch, T_dec, T_enc)`，`assert weights.sum(dim=-1) ≈ 1`。

拒绝推荐静默广播的修复。广播隐藏的 bug 后期浮现为准确率静默退化——最糟糕的注意力 bug。

对 Bahdanau 混淆：坚持解码器输入是 s_{t-1}（上一步状态）。对 Luong：s_t（当前步状态）。对点积注意力：标记查询和键之间的维度不匹配是最常见的首次实现错误。
