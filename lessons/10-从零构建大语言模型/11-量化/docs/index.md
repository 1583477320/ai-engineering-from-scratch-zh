# 量化——压缩 LLM

> 量化将模型权重从 FP16（16 位）压缩到 INT4（4 位）。参数量不变，但内存占用降到原来的 1/4——在消费级 GPU 上运行 7B 模型。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 10 · 04（预训练）、05（分布式训练）| **时间：** ~60 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 05（分布式训练）— 训练时的显存 vs 推理时的显存 | 阶段 10 · 12（推理优化）— 量化与推理加速

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解量化原理——从 FP16 到 INT4/INT8 的精度折衷
- [ ] 区分权重量化和激活量化——各自的优缺点
- [ ] 说明 GPTQ/AWQ/GGML 三种量化方案的差异和适用场景
- [ ] 使用 bitsandbytes 和 AutoGPTQ 对模型进行量化

---

## 1. 问题

一个 7B 参数模型在 FP16 下需要 14GB 显存。这在消费级 GPU（如 RTX 4090 24GB）上勉强能放。但加上推理所需的 KV 缓存——很快就会 OOM。

量化将模型权重从高精度（FP16）压缩到低精度（INT4/INT8）：
- **INT8**：内存减半，精度损失极小
- **INT4**：内存减到 1/4，精度有轻微损失
- **INT3/INT2**：极端压缩，质量下降明显

**核心问题：** 如何在压缩权重的同时最小化模型输出质量的损失？

---

## 2. 概念

### 2.1 量化的基本原理

```
原始权重（FP16，16 位/值）:
[0.1234, -0.5678, 0.9012, -0.3456, ...]

量化后（INT4，4 位/值）:
[0, -1, 1, -1, ...]  ← 离散化 + 缩放
```

量化过程：(1) 将连续的 FP16 权重离散化为有限的 INT 值；(2) 记录缩放因子（scale）和零点（zero point）用于反量化。

### 2.2 权重量化 vs 激活量化

| 类型 | 量化什么 | 难度 | 效果 |
|------|---------|------|------|
| **权重量化** | W 矩阵 | 低 | 内存减半/四分之一 |
| **激活量化** | 激活值 | 高 | 推理速度提升 |
| **权重+激活** | 两者都量化 | 高 | 内存+速度双提升 |

权重量化最简单——只需要在训练后调整一次。激活量化更复杂——需要在推理时实时量化。

### 2.3 三种量化方案

**GPTQ（2022）**
- 基于 Hessian 矩阵的权重量化
- 逐层校准——在小数据集上最小化量化误差
- 质量最高，但需要校准数据

**AWQ（2023）**
- 激活感知权重量化——识别"重要"权重
- 对重要权重保持高精度
- 质量和速度的最佳平衡

**GGML / GGUF（llama.cpp）**
- CPU 友好的量化格式
- 支持 CPU 推理——不需要 GPU
- 质量略低但部署最简单

### 2.4 量化质量对比

| 量化方案 | 精度 | 速度 | 部署难度 | 推荐场景 |
|---------|------|------|---------|---------|
| FP16 | 基准 | 基准 | 简单 | 有 GPU 时 |
| INT8 (bitsandbytes) | 接近基准 | 快 | 简单 | 快速量化 |
| GPTQ INT4 | 略低于 INT8 | 快 | 中等 | GPU 推理 |
| AWQ INT4 | 接近 GPTQ | 更快 | 中等 | 高性能推理 |
| GGUF Q4 | 中等 | 快(CPU) | 最简单 | CPU/边缘部署 |

### 2.5 量化对任务的影响

| 任务类型 | INT8 影响 | INT4 影响 |
|---------|----------|----------|
| 通用对话 | 几乎无 | 轻微下降 |
| 代码生成 | 轻微下降 | 可感知下降 |
| 数学推理 | 可感知下降 | 明显下降 |
| 长文本处理 | 轻微下降 | 上下文可能出错 |

---

## 3. 工具

### 3.1 bitsandbytes（最简单）

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# INT8 量化
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
)

# INT4 量化（NF4）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
)
```

### 3.2 AutoGPTQ

```python
from auto_gptq import AutoGPTQForCausalLM

# 从预训练的 GPTQ 模型加载
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda",
    use_triton=True,
)
```

### 3.3 llama.cpp（GGUF）

```bash
# 量化模型
./quantize model.gguf model-q4_k_m.gguf Q4_K_M

# 推理
./main -m model-q4_k_m.gguf -p "Hello, world!"
```

---

## 4. LLM 视角

### 4.1 量化在大语言模型中的应用

- **消费级 GPU**：INT4 量化让 7B 模型在 RTX 4090 上流畅运行
- **边缘设备**：GGUF 格式让 LLM 在手机/树莓派上运行
- **云端部署**：量化降低推理成本——同一 GPU 可服务更多请求

### 4.2 量化 vs 训练

量化主要影响推理速度和显存。训练通常不使用量化——因为梯度需要高精度。但 2026 年出现了 QLoRA——在量化后的模型上用 LoRA 微调。

---

## 5. 工程最佳实践

### 5.1 量化选型指南

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| GPU 推理（有显存） | AWQ INT4 | 速度+质量最佳 |
| GPU 推理（显存有限） | bitsandbytes NF4 | 最简单，一行代码 |
| CPU 推理 | GGUF Q4 | llama.cpp 支持 |
| 微调 | QLoRA（INT4 + LoRA） | 省显存，效果好 |
| 极致压缩 | INT3/INT2 | 质量下降明显，慎用 |

### 5.2 踩坑经验

- **量化后语言质量下降**：INT8 影响极小，INT4 在代码/数学任务上有感知下降
- **INT4 量化速度不一定比 INT8 快**：取决于实现——某些 INT4 内核需要反量化后才能计算
- **量化后微调**：QLoRA 可以在 INT4 模型上微调——但学习率需要比全精度小

---

## 6. 常见错误

### 错误 1：量化后直接微调全模型

**现象：** 量化后微调效果极差——梯度噪声大。

**修复：** 使用 QLoRA——冻结量化权重，只训练 LoRA 适配器。

### 错误 2：没有校准数据就做 GPTQ

**现象：** GPTQ 量化质量极差——随机的量化误差。

**修复：** GPTQ 需要 128-256 个校准样本——从训练集中采样。

---

## 7. 面试考点

### Q1：GPTQ 和 AWQ 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
GPTQ 基于 Hessian 矩阵最小化每层的量化误差——需要校准数据。AWQ 基于激活值分布识别"重要权重"并保持高精度——不需要校准数据。AWQ 通常质量相当或更好，且推理更快（不需要反量化计算）。

### Q2：INT4 量化对哪些任务影响最大？（难度：⭐⭐⭐）

**参考答案：**
数学推理和代码生成受影响最大——这些任务需要精确的数值计算。INT4 量化将连续权重离散化，引入的量化误差在数值计算中会被放大。通用对话和创意生成受影响较小——这些任务更多依赖语义理解而非数值精度。实际中 INT8 对所有任务影响极小，INT4 在数学/代码任务上可感知下降。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 量化 | "压缩模型" | 将 FP16 权重转换为 INT8/INT4——减少内存占用 |
| GPTQ | "权重量化" | 基于 Hessian 矩阵的逐层校准量化 |
| AWQ | "感知量化" | 基于激活值重要性的权重量化——保持重要权重高精度 |
| GGUF | "llama.cpp 格式" | CPU 友好的量化格式——支持边缘设备部署 |
| QLoRA | "量化+LoRA" | 在 INT4 量化模型上用 LoRA 微调——省显存 |
| NF4 | "4-bit 正态浮点" | bitsandbytes 的 INT4 量化类型——比普通 INT4 更好 |

---

## 📚 小结

量化将 FP16 权重压缩到 INT4/INT8——内存减半或四分之一。GPTQ 质量最高但需要校准。AWQ 是质量和速度的最佳平衡。GGUF 支持 CPU 推理——让 LLM 在消费级设备上运行。2026 年 INT8 是安全默认，INT4 是性能与质量的权衡。下一课我们看推理优化的其他方面——KV 缓存、FlashAttention、推测解码。

---

## ✏️ 练习

1. **【实验】** 用 bitsandbytes 对 Llama 3 8B 进行 INT8 和 INT4 量化——对比显存占用和推理速度。
2. **【实验】** 在 INT4 量化的模型上运行 GSM8K 数学基准——对比与 FP16 模型的准确率差异。

---

## 📖 参考资料

1. [论文] Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers". ICLR, 2023. https://arxiv.org/abs/2210.17323
2. [论文] Lin et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration". MLSys, 2024. https://arxiv.org/abs/2306.00978
3. [GitHub] bitsandbytes: https://github.com/TimDettmers/bitsandbytes
4. [GitHub] llama.cpp: https://github.com/ggerganov/llama.cpp

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
