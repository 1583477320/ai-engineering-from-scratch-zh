# 边缘推理——Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 边缘的核心约束是内存带宽，不是计算。移动 DRAM 是 50-90GB/s；数据中心 HBM3 是 2-3TB/s——30-50 倍差距。Decode 是内存带宽密集型的，这个差距是决定性的。2026 年的格局分四路：Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson。每个都有不同的量化路径和吞吐量天花板。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 04（vLLM 内部原理）、阶段 17 · 09（生产量化）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么移动端 LLM 推理是内存带宽密集型的，计算是次要的
- [ ] 列举四个边缘目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）并将每个匹配到用例
- [ ] 说出 2026 年 WebGPU 覆盖率缺口和 Safari iOS 26 的时间线
- [ ] 为每个目标选择量化格式（Core ML INT4、QNN INT8/INT4、WebGPU Q4、NVFP4）

---

## 1. 问题

客户想要一个端侧聊天机器人：语音优先、默认隐私、离线可用。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 达到约 55 tok/s——可以。在 iPhone 16 Pro 上，同一模型只有 3 tok/s——不可以。在骁龙 8 Gen 3 的中端 Android 上，7 tok/s。在 Chrome Android v121+ 的 WebGPU 浏览器上，4-8 tok/s。

吞吐量方差不是移植问题——它是带宽差距 × 量化格式 × NPU 是否可从用户空间访问的结果。2026 年的边缘推理是四个不同的问题，四个不同的解决方案。

---

## 2. 概念

### 2.1 带宽是真正的天花板

Decode 每个词元读取完整权重集。一个 Q4 的 7B 模型是 3.5GB。以 50GB/s 读取 3.5GB 需要 70ms——理论天花板约 14 tok/s。在 90GB/s（高端移动 DRAM）下天花板是约 25 tok/s。再多的计算也无法突破这个数字。

数据中心 HBM3 以 3TB/s 在 1.2ms 内读取同样的 3.5GB——天花板是 830 tok/s。同一个模型、同一组权重。不同的内存子系统。

### 2.2 Apple Neural Engine（M4 / A18）

- 最高 38 TOPS。统一内存（CPU 和 ANE 共享同一池）——无拷贝开销
- 通过 Core ML + `.mlmodel` 编译模型访问，或通过 Metal Performance Shaders（MPS）经 PyTorch 访问
- 2026 年 iOS 应用的最佳路径：Core ML + INT4 权重 + FP16 激活

### 2.3 Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 最高 45 TOPS。与 SoC 中的 CPU 和 GPU 集成但内存域独立
- QNN SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换
- 聊天模板、Llama 3.2、Phi-3 都在 AI Hub 上作为一等产物发布

### 2.4 WebGPU + WebLLM

- 在浏览器中通过 WebGPU 计算着色器运行模型；无需安装
- Llama 3.1 8B Q4 在 M3 Max 上约 41 tok/s——通过同一后端达到原生的约 70-80%
- 17.6k GitHub stars；OpenAI 兼容 JS API；Apache 2.0
- 2026 覆盖：Chrome Android v121+、Safari iOS 26 GA、Firefox Android 追赶中。总体约 70-75% 移动覆盖率

### 2.5 NVIDIA Jetson 家族

- Orin Nano Super (8GB)：适配 Llama 3.2 3B、Phi-3
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 约 40 tok/s
- Thor / T4000（JetPack 7.1）：AGX Orin 2 倍性能，支持 EAGLE-3 和 NVFP4
- TensorRT Edge-LLM（2026）支持 EAGLE-3 投机解码、NVFP4 权重、Chunked Prefill——数据中心优化移植到边缘

### 2.6 量化格式对照

| 目标 | 格式 | 备注 |
|---|---|---|
| Apple ANE | INT4 权重 + FP16 激活 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8/INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | `mlc_llm convert_weight` + 编译的 `.wasm` |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存带宽受限 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 2.7 边缘长上下文的陷阱

Llama 3.1 的 128K 上下文是数据中心特性。在 8GB 内存的手机上，4GB 模型 + 2GB KV 缓存（32K 词元）+ OS 开销 = OOM。边缘部署保持上下文在 4K-8K，除非接受激进的 KV 量化（Q4 KV）。

### 2.8 语音是杀手级应用

语音智能体对延迟敏感（首词元 < 500ms）。本地推理完全消除网络延迟。结合 Whisper Turbo 变体（在边缘运行）的语音转文本，边缘推理成为生产质量的语音循环。

### 2.9 你应该记住的数字

- Apple M4 / A18 ANE：38 TOPS
- Qualcomm Hexagon SD X Elite：45 TOPS
- WebLLM M3 Max：Llama 3.1 8B Q4 约 41 tok/s
- AGX Orin：gpt-oss-20b 约 40 tok/s
- 数据中心-边缘带宽差距：30-50 倍
- WebGPU 移动覆盖率：约 70-75%（Firefox Android 落后）

---

## 3. 从零实现

### 第 1 步：带宽受限的 Decode 天花板计算

```python
def bandwidth_decode_ceiling(model_params_b, bits, bandwidth_gb_s):
    """计算内存带宽受限的 decode 吞吐量天花板。"""
    weight_bytes = model_params_b * 1e9 * bits / 8
    tok_per_sec = bandwidth_gb_s * 1e9 / weight_bytes
    return tok_per_sec


# 对比数据中心和边缘
print("=== Decode 天花板对比 ===")
print(f"{'目标':25s} {'带宽GB/s':>10} {'天花板tok/s':>12}")
print("-" * 50)

targets = [
    ("iPhone 16 Pro (A18)", 50, 4, 0.25),
    ("骁龙 8 Gen 3", 77, 4, 0.30),
    ("M3 Max (WebGPU)", 100, 4, 0.35),
    ("H100 数据中心", 3350, 4, 0.70),
    ("B200 数据中心", 8000, 4, 0.80),
]

for name, bw, bits, params in targets:
    ceiling = bandwidth_decode_ceiling(params, bits, bw)
    print(f"{name:25s} {bw:10d} {ceiling:12.0f}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 边缘推理平台对照

| 平台 | 算力 | 最佳场景 | 量化格式 |
|---|---|---|---|
| Apple ANE | 38 TOPS | iOS/macOS 应用 | Core ML INT4 + FP16 |
| Qualcomm Hexagon | 45 TOPS | Android 应用 | QNN INT8/INT4 |
| WebGPU/WebLLM | 浏览器 | 跨平台、无安装 | Q4 MLC |
| Jetson Orin Nano | 40 TOPS | 嵌入式/机器人 | Q4 GGUF / TRT-LLM INT4 |
| Jetson AGX/Thor | 80+ TOPS | 边缘服务器 | NVFP4 + FP8 KV |

---

## 5. 工程最佳实践

### 5.1 带宽优先于算力

在边缘，decode 是内存带宽密集型的。更高的 TOPS 不一定带来更好的 tok/s——带宽才是瓶颈。选择设备时优先看 DRAM 带宽。

### 5.2 保持上下文短

边缘设备内存有限。4K-8K 上下文是安全的。128K 上下文需要 KV 量化或服务器端处理。

### 5.3 WebGPU 是最灵活的方案

WebGPU 覆盖 Chrome、Safari、Firefox 三大浏览器（2026 年约 70-75% 移动覆盖率）。无需安装、无需转换路径。适合快速原型和跨平台部署。

### 5.4 中文场景特别建议

- **Apple 在中国市场份额高。** iOS 上的 LLM 推理（Core ML + INT4）是一个大市场。Llama 3.1 8B Q4 在 iPhone 16 Pro 上约 3 tok/s——太慢。考虑使用更小的模型（如 Qwen2.5 1.5B）或将部分推理卸载到服务器
- **国内 Android 芯片的 NPU。** 联发科天玑 9300 的 APU 约 36 TOPS，华为麒麟 9000S 的 NPU 约 16 TOPS。量化格式和 SDK 各不相同
- **WebGPU 在国内浏览器的支持。** Chrome 国内版支持 WebGPU。微信浏览器（基于 Chromium）的 WebGPU 支持可能滞后

---

## 6. 常见错误

### 错误 1：在边缘追求高 TOPS

**现象：** 选择了 45 TOPS 的 Qualcomm Hexagon，但 tok/s 仍然很低。

**原因：** TOPS 衡量的是计算能力，不是内存带宽。Decode 是内存带宽密集型的——45 TOPS 不够，需要 50-90GB/s 的 DRAM 带宽。

**修复：** 选择设备时优先看 DRAM 带宽，不是 TOPS。

### 错误 2：在边缘使用 128K 上下文

**现象：** 8GB 内存的设备 OOM。4GB 模型 + 128K 的 KV 缓存 = 10GB+。

**原因：** 128K 上下文是数据中心特性。边缘设备内存有限。

**修复：** 保持上下文在 4K-8K。需要更长上下文时用 KV 量化或服务器端处理。

---

## 7. 面试考点

### Q1：为什么移动端 LLM 推理的瓶颈是带宽而不是计算？（难度：⭐⭐）

**参考答案：**
Decode 每个词元需要读取完整的模型权重集。一个 Q4 的 7B 模型是 3.5GB。移动端 DRAM 带宽是 50-90GB/s，读取 3.5GB 需要 40-70ms——这就是每个词元的最小时间。再多的 TOPS（计算能力）也无法突破这个带宽限制。数据中心 HBM3 是 3TB/s，同样的权重只需 1.2ms——30-50 倍差距。

### Q2：WebGPU 2026 年的覆盖率缺口在哪里？（难度：⭐⭐）

**参考答案：**
Chrome Android v121+ 支持 WebGPU，Safari iOS 26 GA 支持。但 Firefox Android 仍在追赶——WebGPU 支持尚未稳定。总体移动覆盖率约 70-75%。这意味着如果你的边缘 LLM 需要覆盖所有移动用户，仍需要一个服务端回退路径。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| ANE | "Apple 神经引擎" | M 系列和 A 系列的片上 NPU；统一内存 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU；QNN SDK 访问 |
| WebGPU | "浏览器 GPU" | W3C 标准化的浏览器 GPU API；Chrome/Safari 2026 |
| WebLLM | "浏览器 LLM 运行时" | MLC-LLM 项目；Apache 2.0；OpenAI 兼容 JS API |
| Jetson | "NVIDIA 边缘" | Orin Nano / AGX / Thor / T4000 家族 |
| TRT Edge-LLM | "边缘 TensorRT" | 2026 年边缘 TensorRT-LLM 移植 |
| 带宽受限 | "内存限制" | Decode 受限于读取权重的字节/秒速度 |
| Core ML | "Apple 转换" | ANE 原生模型的 Apple 框架 |
| QNN | "Qualcomm 栈" | Qualcomm Neural Network SDK |

---

## 📚 小结

边缘推理的核心约束是内存带宽——移动端 DRAM 50-90GB/s vs 数据中心 HBM3 2-3TB/s，30-50 倍差距。Decode 每个词元读取完整权重集，所以带宽决定了 tok/s 天花板。四个边缘目标各有不同的量化路径：Apple ANE 用 Core ML INT4、Qualcomm Hexagon 用 QNN、WebGPU 用 Q4 MLC、Jetson 用 NVFP4。WebGPU 是最灵活的方案但覆盖率约 70-75%。语音是杀手级应用——本地推理消除网络延迟。

---

## ✏️ 练习

1. 运行 `code/main.py`。7B 模型 Q4 在骁龙 8 Gen 3（约 77GB/s 带宽）上的 decode 天花板是多少？对比实际的 6-8 tok/s——运行时效率如何？
2. 你的 iOS 应用需要 4K 上下文流式输出。哪种模型/格式组合能让 iPhone 16 的活跃内存保持在 4GB 以下？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 带宽天花板计算器 | `code/main.py` | 各边缘目标的 decode 吞吐量天花板 |
| 边缘目标选型 | `outputs/skill-edge-target-picker.md` | 根据平台和预算选择量化格式 |

---

## 📖 参考资料

1. [报告] On-Device LLMs State of the Union 2026. https://v-chandra.github.io/on-device-llms/
2. [NVIDIA] Jetson Edge AI. https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/
3. [论文] WebLLM (arXiv:2412.15803). https://arxiv.org/html/2412.15803v2
4. [官方文档] Apple Core ML. https://developer.apple.com/documentation/coreml
