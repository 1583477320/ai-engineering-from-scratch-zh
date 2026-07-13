# KV 缓存与 FlashAttention 优化指南

> 两种关键技术如何加速长上下文 Transformer 推理。

## 性能对比

| 技术 | 优化前 | 优化后 | 加速比 |
|------|--------|--------|--------|
| KV 缓存 | O(n²) 时间 | O(n) 时间 | ~n/2x |
| FlashAttention | O(n²) 内存 | O(n) 内存 | 2-4x 硬件效率 |

## KV 缓存内存估算

对于 n=128K, d_k=128, n_layers=32, fp16:

- 每层的 KV 缓存大小：2 × n × d_k = 2 × 128K × 128 = 32M
- 全部 32 层：32 × 32M = 1GB
- 考虑 batch_size=4：4GB

## FlashAttention 版本

| 版本 | 改进 | 加速比 |
|------|------|--------|
| FlashAttention v1 | 分块计算 + online softmax | 2x |
| FlashAttention v2 | 更好的并行 + 减少非矩阵运算 | 3x |
| FlashAttention v3 | FP8 支持 + 硬件协同设计 | 4x+ |
