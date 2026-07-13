# 注意力变体

> 标准注意力是 O(N²) 的精确计算，但不是唯一选择——线性注意力用核方法把复杂度压到 O(N)，滑动窗口让超长序列成为可能，GQA 在保持质量的同时把 KV 缓存砍掉 4 倍。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 07 · 05（完整 Transformer）
**时间：** ~60 分钟
**所处阶段：** Tier 2
**关联课程：** 第 7 阶段 · 12（KV 缓存与 Flash Attention）— KV 缓存内存问题推动了 GQA 的产生

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 比较标准注意力、线性注意力、滑动窗口注意力的计算复杂度和精度权衡
- [ ] 解释分组查询注意力（GQA）——如何在不损失质量的前提下将 KV 缓存降低 4-8 倍
- [ ] 说明 2026 年长上下文模型使用的注意力策略组合（GQA + Flash Attention + 滑动窗口）
- [ ] 从零实现滑动窗口注意力和稀疏注意力
- [ ] 诊断"注意力变体选型错误"导致的性能问题

---

## 1. 问题

你的 Transformer 能处理 4K 上下文——这在 2020 年已经很好了。但 2026 年，用户期望模型能一次读完一本 300 页的文档（约 200K 词元）。

标准注意力是 O(N²)。200K 上下文需要计算 200K × 200K = 400 亿个分数——单精度下约 160GB 内存。一张 A100 只有 80GB。**硬件根本装不下。**

你需要的不是更快的标准注意力——你需要一种完全不同的注意力计算方式。线性注意力把 O(N²) 变成 O(N)。滑动窗口让每个位置只关注附近的邻居。GQA 让多个注意力头共享 KV 缓存——内存占用降低 4 倍，质量几乎不变。

没有一种变体完美解决所有问题。实际部署中，这些变体被组合使用：GQA 减少 KV 缓存，Flash Attention 优化内存访问，滑动窗口处理超长序列。**理解每种变体的权衡，才能做出正确的工程选择。**

---

## 2. 概念

### 2.1 六种主要注意力变体

```
标准注意力 (MHA):
  每个头独立计算 Q、K、V
  复杂度 O(N²)        内存 O(N²)
  ┌─────┐
  │Q₁K₁V₁│  ← 头 1
  │Q₂K₂V₂│  ← 头 2
  │ ...  │
  │Q₈K₈V₈│  ← 头 8
  └─────┘

线性注意力:
  用核函数 φ 替代 Softmax
  先算 φ(K)ᵀV (固定大小)，再乘 φ(Q)
  复杂度 O(Nd²)       内存 O(Nd)
  ┌─────┐
  │φ(Q)·(φ(K)ᵀV) │  ← 不需要 N×N 矩阵
  └─────┘

滑动窗口:
  每个位置只关注 W 个邻居
  复杂度 O(NW)        内存 O(NW)
  ┌─────────────┐
  │  ···███···  │  ← 窗口 W=128
  └─────────────┘

GQA:
  多个 Q 头共享一组 KV
  KV 缓存从 n_heads 降到 n_kv_groups
  ┌─────┐
  │Q₁ Q₂ Q₃ Q₄│ → 共享 K₁V₁  ← 组 1
  │Q₅ Q₆ Q₇ Q₈│ → 共享 K₂V₂  ← 组 2
  └─────┘
```

| 变体 | 核心思想 | 复杂度 | KV 缓存 | 精确度 | 使用场景 |
|---|---|---|---|---|---|
| **缩放点积 (MHA)** | 标准注意力 | O(N²) | 完整 | 精确 | 短序列，基线 |
| **线性注意力** | 核函数替代 Softmax | O(N) | 无 | 近似 | 长序列生成 |
| **滑动窗口** | 每个位置只关注 W 个邻居 | O(NW) | 窗口内 | 窗口内精确 | 超长文本 |
| **分组查询注意力 (GQA)** | 多个 Q 共享 K/V | O(N²) | 减少 | 精确 | LLaMA 3, Mistral, Qwen2.5 |
| **多查询注意力 (MQA)** | 所有 Q 共享一对 K/V | O(N²) | 极小 | 精确 | 边缘推理 |
| **稀疏注意力** | 只计算最重要的 N×M 个分数 | O(NM) | 视实现 | 近似 | 长文档 |

### 2.2 GQA——2026 年的主流选择

GQA（分组查询注意力）的核心思想：将多个注意力头分成若干组，每组共享一对 K/V 投影。

```
标准 MHA: 8 个头 × 8 对 KV = 8 个 KV 缓存
GQA:      8 个头 × 2 组 KV = 2 个 KV 缓存  ← 内存减少 4 倍
MQA:      8 个头 × 1 对 KV = 1 个 KV 缓存  ← 内存减少 8 倍
```

**为什么 GQA 是主流？** 因为它在质量和内存之间取得了最佳平衡——KV 缓存降低 4 倍，但质量损失极小（通常 < 1% 的困惑度差异）。LLaMA 3、Mistral、Qwen2.5 都选择 GQA。

GQA 是 MQA 和 MHA 的泛化：
- `GQA(n_kv_groups = n_heads)` = 标准 MHA
- `GQA(n_kv_groups = 1)` = MQA
- 通常取 `n_kv_groups = 4` 或 `8`

### 2.3 滑动窗口——多层传播原理

滑动窗口在单层内丢失长距离信息，但多层堆叠后信息可以逐层传播：

```
层 1: 位置 100 的信息 → 传播到 99~101
层 2: 位置 99 的信息  → 传播到 98~100
...
层 L: 位置 100 的信息 → 传播到 100-L×W ~ 100+L×W

W=128, L=24 层时:
信息可以传播 24 × 128 = 3072 个位置
```

### 2.4 线性注意力的核方法

标准注意力的 Softmax 无法分解——必须先算 Q×K^T（N×N），再乘 V。线性注意力用核函数 φ 替代 Softmax：

$$
\text{Attention}(Q, K, V) = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})}
$$

关键：利用矩阵乘法的结合律，先算 $\phi(K)^T V$（固定大小 d×d），再乘 $\phi(Q)$。**不需要 N×N 矩阵。**

核函数选择：φ(x) = ReLU(x) + ε（ε 为小常数，防止除零）。

---

## 3. 从零实现

完整代码见 `code/main.py`——纯 NumPy，实现了 5 种注意力变体。

### 第 1 步：线性注意力

```python
def linear_attention(Q, K, V):
    """线性注意力——利用结合律避免 N×N 矩阵。

    标准: (Q @ K.T) @ V  → 需要 N×N 矩阵
    线性: Q @ (K.T @ V)  → K.T @ V 是 d×d 矩阵，不随 N 增长
    """
    phi_Q = np.maximum(Q, 0) + 0.01   # φ(x) = ReLU(x) + ε
    phi_K = np.maximum(K, 0) + 0.01
    # 先算 φ(K)ᵀV：形状 (d_k, d_v)，不随序列长度增长
    KV = phi_K.T @ V
    output = phi_Q @ KV
    denom = phi_Q @ phi_K.sum(axis=0)
    return output / denom[:, np.newaxis]
```

### 第 2 步：滑动窗口注意力

```python
def sliding_window_attention(Q, K, V, window_size):
    """滑动窗口——每个位置只关注 W 个邻居，复杂度 O(NW)。"""
    n = Q.shape[0]
    output = np.zeros_like(V)
    half = window_size // 2
    for i in range(n):
        start, end = max(0, i - half), min(n, i + half + 1)
        scores = Q[i] @ K[start:end].T / np.sqrt(Q.shape[-1])
        output[i] = softmax(scores) @ V[start:end]
    return output
```

### 第 3 步：分组查询注意力（GQA）

```python
class GroupedQueryAttention:
    """GQA：多个 Q 头共享同一组 KV。
    GQA(n_kv_groups=1) = MQA，GQA(n_kv_groups=n_heads) = MHA
    """
    def __init__(self, d_model, n_heads, n_kv_groups, seed=42):
        self.group_size = n_heads // n_kv_groups
        # 每个头独立 Q，每组共享 K/V
        self.Wq = [...]   # n_heads 个投影
        self.Wk = [...]   # n_kv_groups 个投影
        self.Wv = [...]   # n_kv_groups 个投影

    def forward(self, X):
        for i in range(self.n_heads):
            g = i // self.group_size
            Q = X @ self.Wq[i]
            K = X @ self.Wk[g]   # 同组共享 K
            V = X @ self.Wv[g]   # 同组共享 V
            # ... 计算注意力
```

### 第 4 步：稀疏注意力

```python
def sparse_attention(Q, K, V, top_k):
    """稀疏注意力——每个位置只关注分数最高的 top_k 个位置。"""
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    output = np.zeros_like(V)
    for i in range(Q.shape[0]):
        top_idx = np.argsort(scores[i])[-top_k:]
        output[i] = softmax(scores[i, top_idx]) @ V[top_idx]
    return output
```

---

## 4. 工业工具

### 4.1 PyTorch SDPA（内置优化）

```python
import torch.nn.functional as F

# PyTorch 2.0+ 内置 Flash Attention 优化
output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
# 自动选择 FlashAttention / 内存高效注意力 / 数学后端
```

### 4.2 HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM

# LLaMA 3 使用 GQA，自动配置
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
# model.config.num_key_value_heads = 8（GQA，不是 32）
# 自动使用 KV 缓存 + Flash Attention
```

### 4.3 vLLM PagedAttention

```python
from vllm import LLM, SamplingParams

# vLLM 自动使用 PagedAttention + GQA
llm = LLM(model="meta-llama/Llama-3-8B")
output = llm.generate(prompts, SamplingParams(temperature=0.7))
```

### 4.4 性能对比

| 实现方式 | 速度 | 内存 | KV 缓存 | 适用场景 |
|---|---|---|---|---|
| 标准注意力 (MHA) | 基线 | 高 | 完整 | 学习理解 |
| 线性注意力 | 快 (O(N)) | 低 | 无 | 长序列生成 |
| 滑动窗口 | 快 (O(NW)) | 低 | 窗口内 | 超长文本 |
| GQA + Flash Attention | 极快 | 低 | 4-8x 减少 | 生产环境 |
| vLLM PagedAttention | 极快 | 极低 | 分页管理 | 大规模推理 |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

2026 年的主流大语言模型几乎都使用 GQA：

| 模型 | 注意力类型 | KV 头数 | Q 头数 | KV 缓存比例 |
|---|---|---|---|---|
| GPT-4 | 未公开 | 未公开 | 未公开 | — |
| Claude 3.5 | 未公开 | 未公开 | 未公开 | — |
| LLaMA 3 8B | GQA | 8 | 32 | 25% |
| LLaMA 3 405B | GQA | 8 | 128 | 6.25% |
| Qwen2.5 72B | GQA | 8 | 64 | 12.5% |
| Mistral 7B | GQA | 8 | 32 | 25% |

GQA 是 2024-2026 年的事实标准——几乎所有开源大语言模型都采用它。

### 5.2 LLM 时代什么变了？

**注意力变体从"学术研究"变成"工程必需"。** 标准 MHA 在 2K 上下文时没有问题，但 128K 上下文让 KV 缓存成为瓶颈。GQA 不是"更好的 MHA"——它是在有限显存下运行长上下文模型的工程妥协。

**Flash Attention 从"优化技巧"变成"标准配置"。** PyTorch 2.0+ 默认使用 Flash Attention 后端，HuggingFace Transformers 自动启用。开发者不需要手动配置——但如果理解原理，可以在出问题时快速诊断。

### 5.3 什么没变？

**注意力的核心语义没变。** 无论用 GQA 还是 MQA，注意力的计算结果仍然是"每个位置对其他位置的加权求和"。变体改变的是实现方式，不是数学定义。

**O(N²) 的理论复杂度没变。** GQA 只减少了 KV 缓存，计算仍然是 O(N²)。真正降低复杂度的是线性注意力和滑动窗口——但它们有精度损失。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你让 Claude 处理一篇 100K 词元的文档时，GQA 使 KV 缓存可控——如果没有 GQA，100K 上下文的 KV 缓存可能需要 40GB 显存，超出单 GPU 限制。Flash Attention 让注意力计算在有限显存内完成——否则 100K × 100K 的注意力矩阵需要 40GB。

**你能观察到的：** 当对话变长时，模型响应变慢——这是因为 KV 缓存增长，每步推理需要处理更多历史信息。GQA 和 PagedAttention 缓解了这个问题，但无法完全消除。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 学习/实验 | PyTorch `nn.MultiheadAttention` | 开箱即用 |
| 训练（< 8K 上下文）| PyTorch SDPA | 自动选取最优实现 |
| 训练（长上下文）| FlashAttention-2 + GQA | IO 感知 + KV 缓存优化 |
| LLM 推理 | GQA + PagedAttention | vLLM 默认配置 |
| 边缘设备 | MQA + 量化 | 最小 KV 缓存 |

### 6.2 中文场景特别建议

- 中文+代码混合场景，优先使用 Llama 3 或 Qwen2.5 的分词器——它们的注意力变体在多语言上经过优化
- 长中文文本处理时，确保 Flash Attention 已启用——中文文本的 token 密度高于英文，更容易触发 O(N²) 内存问题
- 使用 vLLM 部署中文大语言模型时，PagedAttention 可以显著降低长对话的显存消耗

### 6.3 踩坑经验

- GQA 的 KV 头数必须能整除 Q 头数——否则报错。常见配置：8 个 Q 头配 2 个 KV 头
- 线性注意力在短序列（< 1K）上通常不如标准注意力——核近似的误差在短序列上更明显
- 滑动窗口大小 W 不是越大越好——W 增加意味着更多的计算，需要在覆盖范围和效率之间权衡
- Flash Attention 需要 Ampere（A100）或更新的 GPU 架构——旧 GPU 会自动回退到标准实现
- 不要手动实现 GQA——使用 HuggingFace Transformers 或 vLLM 的内置实现

---

## 7. 常见错误

### 错误 1：在短序列上使用线性注意力

**现象：** 模型质量下降——困惑度比标准注意力高 10-20%。

**原因：** 线性注意力的核近似在短序列上引入较大误差。标准注意力在短序列上没有内存瓶颈，不需要近似。

**修复：**
```python
# ❌ 对所有序列使用线性注意力
output = linear_attention(Q, K, V)

# ✓ 只在长序列上使用
if seq_len > 4096:
    output = linear_attention(Q, K, V)
else:
    output = standard_attention(Q, K, V)
```

### 错误 2：GQA 的 KV 头数不能整除 Q 头数

**现象：** 训练启动时报错——`n_heads must be divisible by n_kv_heads`。

**原因：** GQA 要求每个 KV 头服务整数个 Q 头。如果 n_heads=32, n_kv_heads=5，32/5 不是整数。

**修复：**
```python
# ❌ 不整除
n_kv_heads = 5  # 32 / 5 = 6.4 → 报错

# ✓ 选择能整除的值
n_kv_heads = 8  # 32 / 8 = 4 → 每组 4 个 Q 头
```

### 错误 3：滑动窗口过小导致信息丢失

**现象：** 模型在需要长距离依赖的任务上表现差——如文档摘要、长文问答。

**原因：** 窗口 W 太小时，单层内信息传播距离有限。虽然多层可以弥补，但层数不够时信息仍然丢失。

**修复：**
```python
# ❌ W=32，24 层只能传播 768 个位置
output = sliding_window_attention(Q, K, V, window_size=32)

# ✓ W=128，24 层可以传播 3072 个位置
output = sliding_window_attention(Q, K, V, window_size=128)
```

### 错误 4：忘记 Flash Attention 需要特定 GPU

**现象：** 在旧 GPU（如 V100）上使用 Flash Attention 报错。

**原因：** Flash Attention 需要 Ampere（A100）或更新的 GPU 架构。

**修复：**
```python
# ❌ 直接使用，不检查 GPU
output = F.scaled_dot_product_attention(Q, K, V)

# ✓ 检查 GPU 兼容性
import torch
if torch.cuda.is_device_available() and torch.cuda.get_device_capability()[0] >= 8:
    output = F.scaled_dot_product_attention(Q, K, V)  # Flash Attention
else:
    output = Q @ K.T / np.sqrt(K.shape[-1]) @ V  # 标准实现
```

---

## 8. 面试考点

### Q1：比较 GQA 和 MQA 的优劣？（难度：⭐⭐）

**参考答案：**
GQA（分组查询注意力）将 Q 头分成若干组，每组共享一对 K/V。MQA（多查询注意力）是 GQA 的特例——所有 Q 头共享一对 K/V。

GQA 的优势：质量损失更小。MQA 将所有信息压缩到一对 K/V 中，信息瓶颈更严重。GQA 在多个组内保留更多信息。

MQA 的优势：KV 缓存最小（1 对），适合边缘设备和低内存场景。推理时 KV 缓存只有 GQA 的 1/G 倍。

**实践选择：** 大语言模型（LLaMA 3, Qwen2.5）用 GQA（通常 8 个 KV 头），边缘推理用 MQA。

### Q2：线性注意力为什么是 O(N)？它有什么缺点？（难度：⭐⭐⭐）

**参考答案：**
线性注意力用核函数 φ 替代 Softmax，利用矩阵乘法的结合律：$\phi(Q)(\phi(K)^TV)$。先算 $\phi(K)^TV$（固定大小 d×d 矩阵），再乘 $\phi(Q)$——不需要 N×N 矩阵。

缺点：
1. **核近似误差**：ReLU 核函数与 Softmax 的行为不同，短序列上质量下降
2. **无法使用因果掩码**：标准因果掩码需要对 N×N 矩阵操作，线性注意力无法直接使用
3. **训练不稳定**：核函数可能导致梯度消失或爆炸

### Q3：为什么说 Flash Attention 是精确注意力？（难度：⭐⭐）

**参考答案：**
Flash Attention 的计算结果与标准注意力完全一致——它只是改变了计算方式。标准注意力一次性计算 N×N 矩阵；Flash Attention 将 Q/K/V 分成小块，逐块计算注意力并累加结果。

加速来自 GPU 内存层次优化：减少 HBM（主内存）访问，尽量使用 SRAM（高速缓存）。内存占用从 O(N²) 降到 O(N)，但计算仍是 O(N²)——只是不存储中间结果。

### Q4：设计题——如何为一个 200K 上下文的大语言模型选择注意力策略？（难度：⭐⭐⭐）

**参考答案：**
需要考虑三个维度：质量、内存、速度。

1. **质量优先**：GQA（8 个 KV 头）+ Flash Attention-2。GQA 保持 MHA 级别的质量，Flash Attention 优化内存。
2. **内存优先**：GQA + 滑动窗口（W=256）+ PagedAttention。滑动窗口在单层内限制注意力范围，多层传播保持长距离依赖。
3. **速度优先**：GQA + Flash Attention + KV 缓存量化（INT8/INT4）。量化 KV 缓存可以将内存再降低 2-4 倍。

实际部署通常组合使用：GQA + Flash Attention + PagedAttention 是 2026 年的标准配置。

### Q5：滑动窗口的窗口大小 W 如何选择？（难度：⭐⭐）

**参考答案：**
W 的选择取决于任务和模型深度：

- **W 太小**（如 32）：单层传播距离有限，需要更多层来弥补。L=24 层只能传播 768 个位置。
- **W 太大**（如 1024）：接近标准注意力，失去了 O(NW) 的效率优势。
- **经验值**：W=128 到 W=256 是常见选择。L=24 层时，W=128 可以传播 3072 个位置，覆盖大多数任务。

Mistral 7B 使用 W=4096（接近全注意力），但它的模型较浅（32 层），需要更大的窗口来保证长距离依赖。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 线性注意力 | "O(N) 的注意力" | 用核函数替代 Softmax，利用结合律避免 N×N 矩阵——但引入核近似误差 |
| 滑动窗口注意力 | "局部注意力" | 每个位置只关注 W 个邻居——单层内丢失长距离信息，多层传播弥补 |
| GQA | "共享 KV 的注意力" | 分组查询注意力——多个 Q 头共享一组 K/V 投影，减少 KV 缓存 4-8 倍 |
| MQA | "只有一个 KV 的注意力" | 多查询注意力——所有 Q 头共享一对 K/V，KV 缓存最小 |
| 稀疏注意力 | "只关注重要的位置" | 每个位置只计算分数最高的 N×M 个分数——Longformer、BigBird |
| Flash Attention | "更快的注意力" | 分块注意力计算——精确结果，但通过 IO 感知减少 HBM 访问，内存 O(N) |
| KV 缓存 | "缓存键值对" | 缓存历史位置的 Key 和 Value，避免自回归解码时重复计算 |
| 核函数 | "替代 Softmax 的函数" | 满足 k(x,y) = φ(x)·φ(y) 的函数——将 Softmax 的非线性分解为线性操作 |

---

## 📚 小结

标准注意力 O(N²) 限制了上下文长度——128K 上下文需要约 64GB 内存。六种注意力变体从不同角度解决这个问题：线性注意力降低复杂度，滑动窗口限制注意力范围，GQA 减少 KV 缓存，Flash Attention 优化内存访问。没有一种变体完美——实际部署中组合使用（GQA + Flash Attention + PagedAttention 是 2026 年的标准配置）。理解每种变体的权衡，才能在具体场景中做出正确的工程选择。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释线性注意力为什么能是 O(N)——关键在于哪个数学性质？写 200 字以内的说明，让一个没有 ML 背景的程序员也能听懂。

2. **【实现】** 修改 `sliding_window_attention` 函数，加入因果掩码——让每个位置只能看到窗口内它之前的位置。在 16 个词元的序列上测试。

3. **【实验】** 对比 GQA 和 MHA 的 KV 缓存内存：假设模型有 32 层、8 个 Q 头、128 维、序列长度 4K，分别计算 MHA（32 个 KV 头）和 GQA（8 个 KV 头）的 KV 缓存内存。

4. **【思考】** 为什么 Mistral 7B 选择滑动窗口 W=4096（接近全注意力），而不是像 Longformer 那样用 W=256？考虑模型深度和任务需求。

5. **【实验】** 运行 `code/main.py`，对比线性注意力和标准注意力的输出差异。思考：在什么场景下这个差异是可以接受的？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 注意力变体实现 | `code/main.py` | 5 种注意力变体的纯 NumPy 实现 |
| 变体参考指南 | `outputs/attention-variants-guide.md` | 对比所有变体的特性、复杂度和适用场景 |

---

## 📖 参考资料

1. [论文] Katharopoulos et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention". ICML, 2020. https://arxiv.org/abs/2006.16236
2. [论文] Beltagy et al. "Longformer: The Long-Document Transformer". 2020. https://arxiv.org/abs/2004.05150
3. [论文] Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints". EMNLP, 2023. https://arxiv.org/abs/2305.13245
4. [论文] Shazeer. "Fast Transformer Decoding: One Write-Head is All You Need". 2019. https://arxiv.org/abs/1911.02150
5. [论文] Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". NeurIPS, 2022. https://arxiv.org/abs/2205.14135

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
