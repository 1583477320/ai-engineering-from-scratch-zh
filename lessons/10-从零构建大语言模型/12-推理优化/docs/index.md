# 推理优化

> LLM 推理是自回归的——每生成一个 token 需要一次前向传播。KV 缓存 + FlashAttention + 推测解码是加速的三大支柱。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 10 · 04（预训练）、05（分布式训练）| **时间：** ~45 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 11（量化）— 量化减少内存占用 | 阶段 07 · 12（KV 缓存）— 推理优化的基础

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释推理延迟的三个来源——预填充 + KV 生成 + 解码
- [ ] 说明 vLLM 的 PagedAttention——如何解决 KV 缓存碎片化
- [ ] 说明推测解码如何用小模型加速大模型

---

## 1. 问题

LLM 推理的核心问题：**每生成一个 token 需要一次完整的前向传播。** 生成 1000 个 token = 1000 次前向传播。这比图像生成（通常只需要 20-50 步）慢得多。

推理延迟的三个来源：
1. **预填充（Prefill）**：处理输入 token，构建初始 KV 缓存——计算密集但可并行
2. **KV 生成**：每步只处理一个新 token，但需要整个 KV 缓存——显存密集
3. **解码（Decoding）**：自回归生成，每步一次前向传播——延迟密集

优化目标：降低每 token 的延迟（Time-to-First-Token 和 Per-Token-Latency）。

---

## 2. 概念

### 2.1 KV 缓存

自回归生成时，每步只需计算新 token 的 Q/K/V，但注意力需要访问所有历史 token 的 K/V。KV 缓存将历史的 K/V 缓存起来——避免重复计算。

### 2.2 PagedAttention（vLLM）

vLLM 的核心创新：**将 KV 缓存分页管理**——类似操作系统的虚拟内存。解决 KV 缓存的内存碎片化问题。

传统方法：为每个请求预分配最大长度的 KV 缓存——浪费大量显存。
PagedAttention：动态分配固定大小的页——按需分配，按需释放。

### 2.3 FlashAttention

IO 感知的注意力实现——减少 GPU HBM 的读写次数。标准注意力需要 O(n²) 的 HBM 读写，FlashAttention 通过分块计算将其减少到 O(n)。

### 2.4 推测解码（Speculative Decoding）

用一个小模型（草稿模型）快速生成多个候选 token，然后用大模型一次性验证。如果草稿正确——直接接受；如果错误——用大模型重新生成。

```
草稿模型: 快速生成 5 个 token（50ms）
大模型: 并行验证 5 个 token（20ms）
加速: 50ms → 20ms（2.5x 加速）
```

### 2.5 连续批处理（Continuous Batching）

传统批处理：等待当前批次完成后才添加新请求。
连续批处理：在推理过程中动态添加/移除请求——最大化 GPU 利用率。

### 2.6 优化技术汇总

| 技术 | 效果 | 适用场景 |
|------|------|---------|
| KV 缓存 | 避免重复计算 | 所有自回归模型 |
| FlashAttention | 减少 HBM 读写 | 长序列 |
| PagedAttention | 消除 KV 缓存碎片化 | 服务多个用户 |
| 连续批处理 | 满 GPU 利用率 | 在线服务 |
| 推测解码 | 2-3x 加速 | 大模型 |
| 量化 | 减少显存 | 所有场景 |

---

## 3. 工具

### 3.1 vLLM

```python
from vllm import LLM, SamplingParams

# 初始化 vLLM 引擎
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

# 批量推理
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
outputs = llm.generate(["Hello", "What is AI?"], sampling_params)
```

### 3.2 TensorRT-LLM

```python
# NVIDIA 的推理优化——在 H100 上最快
# 需要构建引擎，但推理速度最优
```

### 3.3 llama.cpp

```bash
# CPU/边缘设备推理
./main -m model-q4_k_m.gguf -p "Hello" -n 100
```

### 3.4 工具对比

| 工具 | 特点 | 适用场景 |
|------|------|---------|
| vLLM | PagedAttention + 连续批处理 | GPU 在线服务（推荐） |
| TensorRT-LLM | NVIDIA 优化，最快 | H100/A100 生产部署 |
| llama.cpp | CPU/边缘推理 | 无 GPU 场景 |
| Triton Inference Server | 企业级 | 大规模部署 |

---

## 4. LLM 视角

### 4.1 推理优化与大语言模型的关系

- **TTFT（Time-to-First-Token）**：预填充时间——决定用户感知的"响应速度"
- **TPS（Tokens-per-Second）**：每秒生成 token 数——决定回答的流式输出速度
- **成本**：每 token 的推理成本——直接关系到 API 定价

### 4.2 推测解码在 LLM 中的应用

- **大模型 + 小草稿模型**：如 70B + 1.5B 草稿——2-3x 加速
- **投机验证**：大模型并行验证草稿模型的输出——只需 1 次前向传播
- **2026 年趋势**：几乎所有生产 LLM 都支持推测解码

---

## 5. 工程最佳实践

### 5.1 选择推理引擎

| 场景 | 推荐引擎 |
|------|---------|
| GPU 在线服务 | vLLM |
| 极致速度（H100） | TensorRT-LLM |
| CPU/边缘 | llama.cpp |
| 企业级 | Triton + vLLM |

### 5.2 踩坑经验

- **KV 缓存 OOM**：减少 max_model_len，或增加 gpu_memory_utilization
- **首 token 延迟高**：预填充时间长——减少输入长度或使用 FlashAttention
- **推测解码不加速**：草稿模型太大——选择更小的草稿模型

---

## 6. 常见错误

### 错误 1：忽略 KV 缓存的显存占用

**现象：** 模型本身能放入显存，但推理时 OOM。

**原因：** KV 缓存需要额外显存——对于 8K 上下文的 7B 模型，KV 缓存可能需要 2-4GB。

### 错误 2：没有使用连续批处理

**现象：** GPU 利用率只有 30-40%。

**原因：** 传统批处理等待整个批次完成——短请求等长请求。

---

## 7. 面试考点

### Q1：vLLM 的 PagedAttention 解决了什么问题？（难度：⭐⭐）

**参考答案：**
传统推理为每个请求预分配最大长度的 KV 缓存——导致严重内存浪费和碎片化。PagedAttention 将 KV 缓存分页管理，按需分配和释放，类似操作系统的虚拟内存。这使 vLLM 的吞吐量比 naive 实现高 2-4 倍——特别是在长序列和多用户场景。

### Q2：推测解码的加速原理是什么？（难度：⭐⭐⭐）

**参考答案：**
推测解码利用了"验证比生成便宜"的原理。草稿模型（小而快）并行生成 N 个候选 token（1 次前向传播），大模型一次性验证这 N 个 token（1 次前向传播，但处理 N 个 token）。如果草稿正确率是 80%，平均加速约 2x。关键：大模型的验证是并行的——1 次前向传播处理 N 个 token，与处理 1 个 token 的延迟几乎相同。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| KV 缓存 | "保存历史" | 缓存注意力中 K/V 的历史值——避免重复计算 |
| PagedAttention | "分页内存" | vLLM 的 KV 缓存管理——按需分配/释放，消除碎片化 |
| FlashAttention | "快速注意力" | IO 感知的注意力实现——减少 HBM 读写 |
| 推测解码 | "小模型预生成" | 小模型快速生成候选，大模型并行验证 |
| 连续批处理 | "动态批处理" | 推理时动态添加/移除请求——最大化 GPU 利用率 |
| TTFT | "首 token 延迟" | 处理输入到第一个输出 token 的时间 |
| TPS | "生成速度" | 每秒生成的 token 数 |

---

## 📚 小结

LLM 推理优化的核心：KV 缓存避免重复计算、FlashAttention 减少 HBM 读写、PagedAttention 消除显存碎片、连续批处理最大化 GPU 利用率、推测解码用小模型加速大模型。vLLM 是 2026 年 GPU 在线服务的标准选择。下一课我们看完整的 LLM 流水线——从预处理到部署。

---

## ✏️ 练习

1. **【实验】** 用 vLLM 部署 Llama 3 8B——测量不同并发请求数下的吞吐量。
2. **【实验】** 对比有/无 KV 缓存的推理速度——差异有多大？

---

## 📖 参考资料

1. [论文] Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention". SOSP, 2023.
2. [GitHub] vLLM: https://github.com/vllm-project/vllm
3. [论文] Leviathan et al. "Fast Inference from Transformers via Speculative Decoding". ICML, 2023. https://arxiv.org/abs/2211.17192

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
