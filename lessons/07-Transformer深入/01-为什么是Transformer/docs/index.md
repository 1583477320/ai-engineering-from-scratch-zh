# 为什么是 Transformer——RNN 的问题

> RNN 一个词一个词地处理序列。Transformer 一次性处理所有词。这个单一架构选择在 2017 年后改变了深度学习的每个缩放曲线。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 3（深度学习核心）、阶段 05 · 09（序列到序列）、阶段 05 · 10（注意力机制）
**时间：** ~45 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 RNN 的三个致命弱点——串行计算、梯度消失、固定宽度隐藏状态
- [ ] 从数值上对比 RNN 和 Transformer 的串行深度——理解为什么 GPU 并行性使 Transformer 训练快 5-10 倍
- [ ] 说明 Transformer 的代价——O(N²) 注意力内存，以及 Mamba/SSM 如何弥合这个差距

---

## 1. 问题

2017 年之前，每个序列模型——语言、翻译、语音——都是 RNN。LSTM 和 GRU 半个世纪内主导了翻译基准。它们是当时唯一的工具。

但它们有三个致命弱点：

1. **串行计算。** 词元 t+1 需要词元 t 的隐藏状态——无法并行化。1024 词元序列 = 1024 串行步，在可以并行 100 万次浮点运算的 GPU 上，99% 的算力被浪费
2. **梯度消失。** 50 个词元前的信息被 50 个非线性变换压缩。长距离依赖——"我去年夏天在京都的飞机上读的那本书是……"——经常失败
3. **固定宽度隐藏状态。** 编码器将整个源序列挤压成一个向量——无论源序列是 5 个词还是 500 个词，瓶颈形状相同

2017 年的论文"Attention Is All You Need"提出了一个激进方案：**完全丢掉递归。** 让每个位置并行地关注其他所有位置。用一个大的矩阵乘法代替 1024 个串行步骤。

结果：到 2026 年，Transformer 主导所有模态。语言（GPT-5、Claude 4、Llama 4）、视觉（ViT、DINOv2）、音频（Whisper）、生物（AlphaFold 3）。同一个块，不同的输入。

---

## 2. 概念

### 2.1 递归作为瓶颈

RNN 计算 `h_t = f(h_{t-1}, x_t)`。每一步依赖前一步。你不能在 `h_4` 之前计算 `h_5`。在现代 GPU 的 10,000+ 并行核心上，这在长序列上浪费了 99% 的硅片。

### 2.2 注意力作为广播

自注意力计算 `output_i = Σ_j(a_ij × v_j)`——所有 (i, j) 对同时计算。整个 N×N 注意力矩阵在一次批矩阵乘法中填充。没有步骤依赖另一个。GPU 很喜欢它。

### 2.3 速度差不是常数

RNN 的串行深度 = O(N)。Transformer 的串行深度 = O(1)（并行扫描）或 O(log N)（树规约）。在实践中，Transformer 在 N=512 时每 epoch 训练快 5-10 倍，且差距随序列长度扩大——直到你碰到注意力 O(N²) 的内存墙（Flash Attention 在第 12 课修复了它）。

### 2.4 Transformer 的代价

注意力内存随序列长度 O(N²) 扩展。2K 上下文：没问题。128K 上下文：你需要滑动窗口、RoPE 外推、Flash Attention 分块、或线性注意力变体。递归在时间和内存上都是 O(N)；Transformer 用内存换时间，然后通过并行性赢回时间。

### 2.5 归纳偏置的转变

RNN 假设局部性和近因性。**Transformer 假设什么都没有——每一对都是注意力的候选。** 这就是为什么 Transformer 需要更多数据来训练，但一旦拥有数据就能缩放更远。Chinchilla（2022）形式化了这一点：给定足够的 token，Transformer 总是击败同参数量的 RNN。

---

## 3. 从零实现

### Step 1：串行深度对比

```python
def rnn_style(xs):
    """RNN 风格：每一步依赖前一步。串行深度 = N。"""
    h = 0.0
    for x in xs:
        h = 0.9 * h + x  # h 依赖前一个 h——无法并行
    return h

def attention_style(xs):
    """Attention 风格：每一步独立，可以并行。深度 = 1。"""
    return sum(xs) / len(xs)  # 所有 x 互相独立
```

**关键洞察：** 两个算法做同样数量的加法。区别是依赖深度——RNN 深度 = N，Attention 深度 = 1。深度决定墙钟时间，而非操作总数。

### Step 2：实测缩放曲线

```python
import time

for n in [100, 1_000, 10_000, 100_000]:
    xs = [1.0] * n
    t0 = time.perf_counter()
    rnn_style(xs)
    rnn_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    attention_style(xs)
    attn_time = time.perf_counter() - t0
    print(f"N={n:>7d}: RNN {rnn_time:.4f}s, Attn {attn_time:.4f}s, "
          f"加速 {rnn_time/attn_time:.0f}x")
```

在 2026 年的笔记本上：< 1,000 的序列太快来不及测量。100,000 的序列清晰地显示了线性扫描。将这个缩放到 16,384 token 的 Transformer + 12 层 LSTM 等效物——你就理解了 2016 年训练墙钟为何是瓶颈。

### Step 3：理论操作计数

两者都做 N 次加法。差异是依赖深度——多少操作必须在下一个操作开始前串行完成。RNN 深度 = N。Attention 深度 = log(N)（树规约）或 1（并行扫描）。**深度决定 GPU 时间，而非操作计数。**

完整代码见 `code/main.py`。

---

## 4. 工业工具——什么时候在 2026 年仍然选择 RNN

| 场景 | 选择 |
|---|---|
| 流式推理、一个 token 一次、恒定内存 | RNN 或状态空间模型（Mamba、RWKV） |
| 极长序列（>1M token），注意力内存爆炸 | 线性注意力、Mamba 2、Hyena |
| 无 matmul 加速器的边缘设备 | 深度可分离 RNN 在 FLOPs/瓦特上仍然赢 |
| 其他所有场景（训练、批量推理、128K 上下文） | Transformer |

**状态空间模型（SSM）** 如 Mamba 本质上是参数化结构的 RNN——兼具 O(N) 扫描内存和并行训练（通过选择性扫描）。它们恢复了 Transformer 90% 的质量，同时在长上下文缩放上更好。2026 年大多数前沿实验室训练混合 SSM+Transformer 模型（如 Jamba、Samba）——递归没有消亡，它是组件。

---

## 5. 面试考点

### Q1：为什么 Transformer 的训练速度比 RNN 快 5-10 倍？（难度：⭐⭐）

**参考答案：**
根本原因是串行深度。RNN 中计算第 t 个位置需要等待第 t-1 个位置完成——1024 token 的序列必须走 1024 个串行步。Transformer 中每个位置同时关注所有其他位置——串行深度 = 1（矩阵乘法是并行的）。在 GPU 上，操作数相同，但并行度从 O(N) 降到 O(1)。N=512 时，这个差距在硬件上是 5-10 倍。

### Q2：Mamba 为什么不是"又一个 RNN"？（难度：⭐⭐⭐）

**参考答案：**
Mamba 将 RNN 的递归参数化为**状态空间模型**——用结构化的状态转移矩阵替代了朴素的 `h_t = tanh(W·x_t + U·h_{t-1})`。这个结构化表示有两个关键性质：(1) 选择性扫描允许并行训练——消除了 RNN 的串行瓶颈；(2) 线性扫描的计算量是 O(N)——解决了 Transformer O(N²) 的内存墙。Mamba 恢复了 RNN 的 O(N) 记忆，但恢复了 Transformer 的并行训练能力——这就是为什么它能恢复 Transformer 90% 的质量。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 递归 | "RNN 是串行的" | 第 t 步的计算依赖第 t-1 步，强制沿时间轴串行执行 |
| 串行深度 | "图有多深" | 最长依赖操作链；即使有无限硬件也限制墙钟时间 |
| 注意力 | "让词互相看" | 加权求和 Σ a_ij v_j，其中 a_ij 来自位置 i 和 j 的相似度得分 |
| 上下文窗口 | "模型能看到多少" | 注意力层可以作为输入的位置数；二次内存开销在此扩展 |
| 归纳偏置 | "架构里烤进去的假设" | 对数据外观的先验；CNN 假设平移不变性，RNN 假设近因性 |
| 状态空间模型 | "有代数支撑的 RNN" | 通过结构化状态空间矩阵参数化的递归，支持并行训练 |
| 二次瓶颈 | "为什么上下文成本这么高" | 注意力内存 = O(N²)；Flash Attention 隐藏了常数，但没有隐藏缩放 |

---

## 📚 小结

RNN 有三个致命弱点：串行计算（GPU 利用率 1%）、梯度消失（50 词元后信息归零）、固定宽度隐藏状态（瓶颈形状不随序列长度变化）。Transformer 丢掉了递归——让每个位置并行关注所有其他位置——训练快 5-10 倍。代价是 O(N²) 的注意力内存（Flash Attention 修复了常数因子）。状态空间模型（Mamba）恢复了 RNN 的 O(N) 内存但保留了 Transformer 的并行性——2026 年的前沿实验室训练混合 SSM+Transformer。

---

## ✏️ 练习

1. 【实现】将 `rnn_style` 的标量隐藏状态替换为长度 64 的向量。重新测量——串行开销如何随维度增长？
2. 【实验】在 PyTorch 上实现并行前缀和（Hillis-Steele scan），与串行版本在长度 1024 上对比数值输出。计算深度。
3. 【实验】在 GPU 上对序列长度 64 到 65,536 进行注意力和 RNN 的实测。绘制并解释曲线形状。

---

## 📖 参考资料

1. [论文] Vaswani et al. "Attention Is All You Need". NeurIPS, 2017. https://arxiv.org/abs/1706.03762
2. [论文] Gu and Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces". 2023.
3. [论文] Gu et al. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality". 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
