# 生产量化——AWQ、GPTQ、GGUF K-quants、FP8、NVFP4

> 量化格式不是通用选择——它是硬件、推理引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 占据 CPU 和边缘场景，通过 llama.cpp 和 Ollama 交付。GPTQ 在 vLLM 中当你需要同一基础模型上的多 LoRA 时胜出。AWQ + Marlin 内核在 7B 类模型上达到约 741 tok/s 且 INT4 下 Pass@1 最高——2026 年数据中心生产的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上保持中间路线——近无损且广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）是激进的，需要逐块验证。两个陷阱：校准数据集必须匹配部署领域，KV 缓存与权重量化是分开的。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 10 · 13（量化基础）、阶段 17 · 04（vLLM 内部原理）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 说出 2026 年的六种生产量化格式及其最佳应用场景
- [ ] 根据硬件（CPU vs GPU、Hopper vs Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（常规聊天、推理、多 LoRA）选择格式
- [ ] 计算选定格式节省的权重内存，并说明 KV 缓存不受影响
- [ ] 命名导致量化模型在领域流量上质量退化的校准数据集陷阱

---

## 1. 问题

量化减少了内存和 HBM 带宽——这恰恰是 decode 所需要的。FP16 的 70B 模型权重是 140GB。将权重量化到 INT4（AWQ 或 GPTQ）后模型只有 35GB——一张 H100 装得下权重加 KV 缓存。128 并发序列、2K 上下文时，KV 缓存本身就需要 20-30GB。

但量化不是免费的。激进量化会降低质量，尤其在推理密集型任务上。不同格式与不同引擎配合。不同硬件原生支持不同精度。2026 年的格式动物园是真实的——你不能照搬别人的选择，必须根据你的技术栈来选。

---

## 2. 概念

### 2.1 六种格式

| 格式 | 位数 | 最佳场景 | 引擎 |
|---|---|---|---|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM、TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM（Marlin-AWQ）、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### 2.2 GGUF——CPU/边缘默认

GGUF 是文件格式而非量化方案——它将 K-quant 变体（Q2_K 到 Q8_0）打包在一个容器中。Q4_K_M 和 Q5_K_M 是生产默认——在 4-5 位下接近 BF16 质量。CPU 或边缘部署的首选，因为 llama.cpp 是目前为止最快的 CPU 推理引擎。

vLLM 中的吞吐量惩罚：7B 上约 93 tok/s——格式没有针对 GPU 内核优化。部署目标是 CPU/边缘时用 GGUF。否则不用。

### 2.3 GPTQ——vLLM 中的多 LoRA

GPTQ 是带校准阶段的后训练量化算法。Marlin 内核让它在 GPU 上很快（比非 Marlin GPTQ 快 2.6 倍）。7B 上约 712 tok/s。

独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA 适配器。如果你需要在一个基础模型上服务 10-50 个微调变体（每个是 LoRA），GPTQ 是你的路径。截至 2026 年初，NVFP4 还不支持 LoRA。

### 2.4 AWQ——数据中心 GPU 默认

激活感知权重量化（Activation-aware Weight Quantization）。在量化过程中保护约 1% 最显著的权重。Marlin-AWQ 内核：比朴素快 10.9 倍。7B 上约 741 tok/s，INT4 格式中 Pass@1 最高。

除非你需要多 LoRA（GPTQ）或激进的 Blackwell FP4（NVFP4），否则选择 AWQ。

### 2.5 FP8——可靠的中间路线

8 位浮点。近无损。广泛支持。Hopper Tensor Core 原生加速 FP8。Blackwell 继承。FP8 是 2026 年质量不可妥协时的安全默认（推理、医疗、代码生成）。内存节省是 INT4 的一半，但质量风险低得多。

### 2.6 NVFP4 / MXFP4——Blackwell 激进方案

微缩放 FP4。每块权重有自己的缩放因子。激进但在 Blackwell Tensor Core 上有硬件加速。与 FP8 相比，每词元的字节数减半——这就是第 17 · 07 课中的经济优势。

注意事项：
- 截至 2026 年初不支持 LoRA
- 推理密集型工作负载上质量下降可见
- 每个模型都需要在评估集上验证

### 2.7 校准陷阱

AWQ 和 GPTQ 需要校准数据集——通常是 C4 或 WikiText。对于领域模型（代码、医疗、法律），在通用网络文本上校准会让算法对哪些权重需要保护做出错误决策。HumanEval 上的 Pass@1 可能下降几个点。

修复：在领域内数据上校准。几百个领域样本通常足够。上线前在评估集上测试。

### 2.8 KV 缓存陷阱

AWQ 将权重量化到 4 位。KV 缓存是独立的，保持 FP16/FP8。70B 模型 + AWQ：

- 权重：约 35GB（INT4，从 140GB 压缩）
- KV 缓存：128 并发 × 2K 上下文 = 约 20GB
- 激活：约 5GB
- 总计：约 60GB——装得下 H100 80GB

天真地"我把模型量化到了 4GB"忘了其他 30-50GB。**整体预算 HBM。**

### 2.9 2026 年选型指南

- CPU/边缘服务：GGUF Q4_K_M。结束。
- GPU 服务，常规聊天，无 LoRA：AWQ。
- GPU 服务，多 LoRA：GPTQ + Marlin。
- 推理密集工作负载：FP8。
- Blackwell 数据中心，已验证质量：NVFP4 + FP8 KV。
- 不确定：对每个候选格式运行 1000 条评估。

---

## 3. 从零实现

### 第 1 步：HBM 占用对比计算器

```python
def memory_footprint(params_b, bits, kv_concurrent=128, kv_context=2048,
                     kv_bits=16, n_layers=80, n_heads=8, head_dim=128):
    """计算权重 + KV 缓存的 HBM 占用。"""
    # 权重
    weights_gb = params_b * 1e9 * bits / 8 / 1e9

    # KV 缓存（独立于权重量化）
    kv_per_seq = 2 * n_layers * n_heads * head_dim * kv_bits / 8
    kv_total = kv_per_seq * kv_concurrent / 1e9

    # 激活（估算）
    activations_gb = params_b * 0.05  # 粗略估算

    return {
        "weights": weights_gb,
        "kv": kv_total,
        "activations": activations_gb,
        "total": weights_gb + kv_total + activations_gb,
    }


# 对比六种格式
model_b = 70
for name, bits in [("BF16", 16), ("FP8", 8), ("AWQ INT4", 4),
                    ("GPTQ INT4", 4), ("GGUF Q4", 4), ("NVFP4", 4)]:
    r = memory_footprint(model_b, bits)
    fits_h100 = "✓" if r["total"] <= 80 else "✗"
    print(f"{name:12s} 权重={r['weights']:5.1f}GB  KV={r['kv']:5.1f}GB  "
          f"总计={r['total']:5.1f}GB  H100={fits_h100}")
```

### 第 2 步：量化格式吞吐量对比

```python
THROUGHPUT = {
    "BF16":     {"7B": 600,  "70B": 30},
    "FP8":      {"7B": 1200, "70B": 60},
    "AWQ INT4": {"7B": 741,  "70B": 80},
    "GPTQ INT4":{"7B": 712,  "70B": 75},
    "GGUF Q4":  {"7B": 93,   "70B": 5},
    "NVFP4":    {"7B": 1500, "70B": 100},
}


def throughput_table():
    print(f"{'格式':12s} {'7B tok/s':>10} {'70B tok/s':>10} {'70B vs BF16':>12}")
    for name, t in THROUGHPUT.items():
        ratio = t["70B"] / THROUGHPUT["BF16"]["70B"]
        print(f"{name:12s} {t['7B']:10d} {t['70B']:10d} {ratio:11.1f}x")


throughput_table()
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 量化格式对照

| 格式 | 权重大小（70B） | 吞吐量倍数 | 质量损失 | 支持 LoRA |
|---|---|---|---|---|
| BF16 | 140GB | 1x | 无 | 是 |
| FP8 | 70GB | 2-3x | 近无损 | 是 |
| AWQ INT4 | 35GB | 5-8x | 轻微 | 是 |
| GPTQ INT4 | 35GB | 5-7x | 轻微 | **是（vLLM 中最佳）** |
| GGUF Q4 | 35GB | 0.1x（GPU） | 轻微 | 否 |
| NVFP4 | 17.5GB | 10-15x | 可见退化 | 否（2026 初） |

### 4.2 选型决策树

```
部署环境？
├── CPU/边缘 → GGUF Q4_K_M
└── GPU
    ├── 需要多 LoRA？ → GPTQ + Marlin
    ├── 推理密集？ → FP8
    ├── Blackwell + 已验证质量？ → NVFP4 + FP8 KV
    └── 默认 → AWQ + Marlin
```

---

## 5. 工程最佳实践

### 5.1 AWQ 是数据中心 GPU 的安全默认

除非你有明确的理由选择其他格式，AWQ + Marlin 是 2026 年数据中心 GPU 服务的默认。INT4 下 Pass@1 最高，吞吐量好，支持 LoRA。

### 5.2 校准数据必须匹配领域

AWQ 和 GPTQ 的校准数据决定了哪些权重被保护。在通用数据上校准领域模型会导致质量退化。几百个领域样本足够。

### 5.3 整体预算 HBM

权重量化到 4 位后模型只有 35GB——但 KV 缓存在生产并发下是 20-30GB。激活还有几 GB。永远整体预算 HBM。

### 5.4 中文场景特别建议

- **中文模型的量化策略。** Qwen2.5、GLM-4 等中文模型在 AWQ INT4 下通常表现良好。但中文的词表更大（150K+ vs 英文 32K），嵌入层的量化需要额外注意
- **国内 GPU 的量化支持。** 华为昇腾 910B 对 FP8 的支持有限。国内部署时可能需要使用 INT4 或 BF16
- **GGUF 在国内边缘场景的使用。** 国内的边缘设备（如智能音箱、车载设备）常用 GGUF + llama.cpp 方案。Q4_K_M 是最佳选择

---

## 6. 常见错误

### 错误 1：通用数据校准领域模型

**现象：** 在通用文本上校准 AWQ 后，代码模型的 HumanEval Pass@1 下降 5 点。

**原因：** 校准数据决定了哪些权重被保护。通用数据让算法保护了通用权重，而不是代码相关的权重。

**修复：** 在领域数据上校准。几百个代码样本足够。

### 错误 2：只看权重量化效果

**现象：** "模型量化到 4 位只要 35GB。" 部署后在高并发下 OOM。

**原因：** KV 缓存是独立的，保持 FP16/FP8。128 并发 × 2K 上下文 = 20GB KV + 5GB 激活 = 60GB。

**修复：** 整体预算 HBM：权重 + KV 缓存 + 激活。

---

## 7. 面试考点

### Q1：AWQ 和 GPTQ 的核心区别是什么？什么时候选哪个？（难度：⭐⭐）

**参考答案：**
AWQ 通过激活感知选择性保护最显著的权重——在 INT4 下 Pass@1 最高。GPTQ 是标准的后训练量化算法，但在 vLLM 中支持 LoRA 适配器。选 AWQ 作为数据中心 GPU 默认。选 GPTQ 当你需要在同一基础模型上服务多个微调变体（多 LoRA）时。

### Q2：KV 缓存量化和权重量化有什么不同？（难度：⭐⭐⭐）

**参考答案：**
权重量化减少模型权重的内存占用——这是一次性的固定节省。KV 缓存量化减少每个请求、每个词元的 KV 缓存内存——这随并发数和上下文长度线性增长。KV 缓存量化直接影响注意力计算的精度——键值的量化误差会传播到注意力分数，影响模型输出质量。权重量化可以做到 INT4 几乎无损，KV 缓存量化通常只敢做到 FP8。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| GGUF | "llama.cpp 格式" | 打包 K-quant 变体的文件格式；CPU/边缘默认 |
| Q4_K_M | "Q4 K M" | 4 位 K-quant 中等；GGUF 的生产默认 |
| GPTQ | "后训练 INT4" | 带校准的后训练 INT4；vLLM 中支持 LoRA |
| AWQ | "激活感知量化" | 激活感知 INT4；Marlin 内核；INT4 下 Pass@1 最高 |
| Marlin 内核 | "快速 INT4 内核" | Hopper 上 INT4 的自定义 CUDA 内核；10 倍加速 |
| FP8 | "8 位浮点" | Hopper/Ada/Blackwell 上的安全精度默认 |
| NVFP4 | "微缩放 4 位" | Blackwell 4 位浮点，带逐块缩放因子 |
| 校准数据集 | "校准数据" | 用于选择量化参数的输入文本；必须匹配领域 |

---

## 📚 小结

2026 年有六种生产量化格式，没有一种是通用选择——GGUF 用于 CPU/边缘、GPTQ 用于多 LoRA、AWQ 用于数据中心 GPU、FP8 用于质量敏感场景、NVFP4 用于 Blackwell 激进方案。AWQ + Marlin 是数据中心 GPU 的安全默认。校准数据必须匹配领域——否则质量退化。KV 缓存与权重量化是独立的选择——永远整体预算 HBM。

---

## ✏️ 练习

1. 运行 `code/main.py`。70B 模型在 128 并发、2K 上下文下，每种格式需要多少 HBM？哪种格式装得下一张 H100 80GB？
2. 你有一个 7B 代码模型。选择一种格式并说明理由。如果你的质量判断有误，恢复路径是什么？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 内存对比计算器 | `code/main.py` | 六种格式的 HBM 占用和吞吐量对比 |
| 量化选型建议 | `outputs/skill-quantization-picker.md` | 根据硬件和工作负载推荐量化格式 |

---

## 📖 参考资料

1. [论文] Lin, J. et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration". arXiv:2306.00978 — AWQ 原始论文
2. [论文] Frantar, E. et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers". arXiv:2210.17323 — GPTQ 原始论文
3. [官方文档] vLLM — Quantization. https://docs.vllm.ai/en/latest/features/quantization/index.html
4. [报告] VRLA Tech — LLM Quantization 2026. https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/
