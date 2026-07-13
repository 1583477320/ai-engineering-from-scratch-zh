# 注意力变体参考指南

> 本文档对比所有主流注意力变体的特性、复杂度和适用场景，供快速查阅。

---

## 一、变体总览

| 变体 | 核心思想 | 时间复杂度 | 空间复杂度 | KV 缓存 | 精确度 | 代表模型 |
|---|---|---|---|---|---|---|
| 缩放点积注意力 (MHA) | 每个头独立 Q/K/V | O(N²d) | O(N²) | 完整 | 精确 | 原始 Transformer |
| 线性注意力 | 核函数替代 Softmax | O(Nd²) | O(Nd) | 无 | 近似 | Performer, Linear Transformer |
| 滑动窗口注意力 | 每个位置只关注 W 个邻居 | O(NWd) | O(NW) | 窗口内 | 窗口内精确 | Mistral, Longformer |
| 多查询注意力 (MQA) | 所有 Q 头共享一对 K/V | O(N²d) | O(Nd) | 1 对 | 精确 | PaLM |
| 分组查询注意力 (GQA) | 多个 Q 头共享一组 K/V | O(N²d) | O(Nd/G) | G 组 | 精确 | LLaMA 3, Qwen2.5, Mistral |
| 稀疏注意力 | 只计算最重要的 N×M 分数 | O(NMd) | O(NM) | 视实现 | 近似 | Longformer, BigBird |
| Flash Attention | 分块计算 + IO 感知 | O(N²d) | O(Nd) | 兼容 | 精确 | 几乎所有 2024+ 模型 |

> **注：** N = 序列长度，d = 注意力维度，W = 滑动窗口大小，G = KV 组数，M = 每位置稀疏度。

---

## 二、按场景选型

### 训练场景

| 场景 | 推荐方案 | 原因 |
|---|---|---|
| 短序列 (< 8K) | 标准注意力 + Flash Attention | Flash Attention 减少内存，不影响速度 |
| 长序列 (8K-128K) | Flash Attention-2 + 滑动窗口 | 窗口内精确，长距离通过层级传播 |
| 超长序列 (> 128K) | 线性注意力 + 局部窗口 | 线性复杂度是唯一可行方案 |

### 推理场景

| 场景 | 推荐方案 | 原因 |
|---|---|---|
| 标准推理 | GQA + Flash Attention + KV 缓存 | 质量与速度的最佳平衡 |
| 长上下文推理 | GQA + 滑动窗口 + PagedAttention | 内存管理 + 窗口注意力 |
| 边缘设备推理 | MQA + 量化 | 最小 KV 缓存 + 最小模型 |

### 批处理推理

| 场景 | 推荐方案 | 原因 |
|---|---|---|
| 高吞吐量 | vLLM PagedAttention | KV 缓存分页，避免显存碎片 |
| 低延迟 | GQA + Flash Attention-2 | 最小化 KV 缓存和计算延迟 |

---

## 三、KV 缓存内存估算

KV 缓存大小 = 2 × n_layers × seq_len × d_k × n_kv_groups × bytes_per_param

| 模型 | n_layers | d_k | n_kv_groups | seq_len | bytes | 缓存大小 |
|---|---|---|---|---|---|---|
| LLaMA 2 7B (MHA) | 32 | 128 | 32 | 4K | 2 | 1 GB |
| LLaMA 3 8B (GQA) | 32 | 128 | 8 | 4K | 2 | 0.25 GB |
| LLaMA 3 8B (GQA) | 32 | 128 | 8 | 128K | 2 | 8 GB |
| Qwen2.5 72B (GQA) | 80 | 128 | 8 | 128K | 2 | 20 GB |

> **结论：** GQA 将 KV 缓存降低 4-8 倍，使长上下文推理在单 GPU 上成为可能。

---

## 四、PyTorch 快速参考

### 标准注意力

```python
import torch.nn.functional as F

# PyTorch 2.0+ 内置 Flash Attention 优化
output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
# 自动选择 FlashAttention / 内存高效注意力 / 数学后端
```

### 多头注意力

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
output, attn_weights = mha(query, key, value)
```

### 分组查询注意力（HuggingFace）

```python
from transformers import AutoModelForCausalLM

# LLaMA 3 使用 GQA，自动配置
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
# model.config.num_key_value_heads = 8 (GQA，不是 32)
```

---

## 五、复杂度对比图

```
序列长度 N:   1K    4K    16K    64K    128K
             │      │      │       │       │
MHA (O(N²))  │ 1M   │ 16M  │ 256M  │ 4.1G  │ 16.4G  ← 内存爆炸
线性 (O(N))  │ 1K   │ 4K   │ 16K   │ 64K   │ 128K   ← 线性增长
滑动 (O(NW)) │ 128K │ 512K │ 2M    │ 8M    │ 16M    ← W=128 时
GQA          │ 1M   │ 16M  │ 256M  │ 4.1G  │ 16.4G  ← 但 KV 缓存 4x
```

> **注意：** 以上是理论内存占用。实际内存还取决于模型参数和优化策略。

---

## 六、常见误区

1. **线性注意力总是更好** -- 线性注意力在理论上是 O(N)，但在短序列上不如标准注意力精确，且 kernel 近似会引入误差。实践中只在长序列场景使用。

2. **滑动窗口丢失了长距离信息** -- 滑动窗口只在单层内丢失长距离信息。通过多层堆叠，信息可以逐层传播。一个位置的信息在 L 层内可以传播 L×W 个位置。

3. **MQA 一定比 GQA 差** -- MQA 在边缘设备上更实用，因为 KV 缓存最小。质量损失在很多任务上可以接受。

4. **Flash Attention 改变了注意力的结果** -- Flash Attention 是精确注意力——计算结果与标准注意力完全一致，只是计算方式不同（分块 vs 全量）。

5. **稀疏注意力需要手动选择关注位置** -- 现代稀疏注意力（如 Longformer）使用可学习的注意力模式，不需要手动指定。

---

## 七、面试速查

| 问题 | 关键点 |
|---|---|
| 为什么需要注意力变体？ | 标准 O(N²) 限制了上下文长度和推理效率 |
| GQA 的核心思想？ | 多个 Q 头共享一组 KV，减少 KV 缓存 4-8 倍 |
| 线性注意力为什么是 O(N)？ | 利用结合律先算 K^TV（固定大小），再乘 Q |
| Flash Attention 是近似吗？ | 不是——精确计算，但分块执行减少内存 |
| 滑动窗口为什么有效？ | 多层堆叠使信息逐层传播，L 层可覆盖 L×W |
