# 自托管推理引擎选型——llama.cpp、Ollama、TGI、vLLM、SGLang

> 四个引擎主导 2026 年的自托管推理。根据硬件、规模和工作负载选择。llama.cpp 在 CPU 上最快——最广的模型支持。Ollama 是开发者笔记本电脑的一键安装，生产负载下吞吐量差约 3 倍。TGI 于 2025 年 12 月 11 日进入维护模式——新项目应默认 vLLM 或 SGLang。vLLM 是通用生产默认。SGLang 是前缀密集型/智能体多轮工作负载的专家。2026 年流水线模式：开发 Ollama → 暂存 llama.cpp → 生产 vLLM 或 SGLang。全程相同 GGUF/HF 权重。

**类型：** 概念课
**语言：** Python
**前置知识：** 第 17 章全部引擎课（04、06、07、09、18）
**预计时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 根据硬件（CPU/AMD/NVIDIA Hopper/Blackwell）、规模（1 用户/100/10000）和工作负载选择引擎
- [ ] 命名 2026 年 TGI 维护模式状态（2025 年 12 月 11 日）以及为什么新项目偏向 vLLM 或 SGLang
- [ ] 描述使用相同 GGUF 或 HF 权重的开发/暂存/生产流水线
- [ ] 解释为什么"仅 CPU"强制 llama.cpp，"AMD"排除 TRT-LLM

---

## 1. 问题

你的团队开始一个自托管 LLM 项目。一个工程师说 Ollama，另一个说 vLLM，第三个说"TGI 不是开箱即用吗？"在各自不同的上下文中三者都是对的。但没有一个是全对的。

在 2026 年，选择树很重要：硬件第一、规模第二、工作负载第三。一个特定的事件——TGI 于 2025 年 12 月 11 日进入维护模式——改变了新项目的默认选择。

---

## 2. 概念

### 2.1 五个引擎

| 引擎 | 最佳场景 | 备注 |
|---|---|---|
| **llama.cpp** | CPU/边缘/最小依赖/最广模型支持 | CPU 上最快，完全控制 |
| **Ollama** | 开发笔记本/单用户/一键安装 | 比 llama.cpp 慢 15-30%；生产吞吐量差 3 倍 |
| **TGI** | HF 生态、受监管行业 | **2025 年 12 月 11 日进入维护模式** |
| **vLLM** | 通用生产、100+ 用户 | 广谱生产默认；v0.15.1 2026 年 2 月 |
| **SGLang** | 智能体多轮、前缀密集型 | 40 万+ GPU 在生产中 |

### 2.2 硬件优先决策

**仅 CPU** → llama.cpp。Ollama 也工作但更慢。没有其他引擎在 CPU 上有竞争力。

**AMD GPU** → vLLM（AMD ROCm 支持）。SGLang 也行。**TRT-LLM 是 NVIDIA 锁定的**——不能用。

**NVIDIA Hopper（H100/H200）** → vLLM 或 SGLang 或 TRT-LLM。三者都顶级。

**NVIDIA Blackwell（B200/GB200）** → TRT-LLM 是吞吐量领先者。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M 系列）** → llama.cpp（Metal）。Ollama 封装了它。

### 2.3 规模第二决策

- **1 用户/本地开发** → Ollama。一条命令，秒出第一个词元
- **10-100 用户/小团队** → vLLM 单 GPU
- **100-10000 用户/生产** → vLLM production-stack 或 SGLang
- **10000+ 用户/企业** → vLLM production-stack + 分离式 + LMCache

### 2.4 工作负载第三决策

- **通用聊天/Q&A** → vLLM 在广谱默认上胜出
- **智能体多轮（工具、规划、记忆）** → SGLang 的 RadixAttention 主导
- **RAG 前缀密集** → SGLang
- **代码生成** → vLLM 正常；SGLang 缓存略优
- **长上下文（128K+）** → vLLM + Chunked Prefill；SGLang + 分层 KV

### 2.5 TGI 维护模式陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式——仅修复 bug。历史上：顶级可观测性，HF 生态集成最好，原始吞吐量略低于 vLLM。

对于 2026 年的新项目：默认避开 TGI。现有 TGI 部署可以继续但应最终迁移。SGLang 和 vLLM 是更安全的默认选择。

### 2.6 流水线模式

开发（Ollama）→ 暂存（llama.cpp）→ 生产（vLLM）。全程相同 GGUF 或 HF 权重。工程师在笔记本上快速迭代；暂存镜像生产量化；生产是服务目标。

---

## 3. 从零实现

### 第 1 步：引擎选择决策树

```python
def pick_engine(hardware, scale, workload):
    """基于硬件+规模+工作负载的引擎选择。"""
    if hardware == "cpu":
        return "llama.cpp（仅 CPU 选项）"
    if hardware == "amd":
        return "vLLM（AMD ROCm 支持，TRT-LLM 不可用）"
    if hardware == "apple":
        return "llama.cpp（Metal）或 Ollama"

    # NVIDIA GPU
    if workload == "agentic" or workload == "prefix_heavy":
        return "SGLang（RadixAttention）"
    if workload == "chat" or workload == "code":
        return "vLLM"

    if scale > 1000:
        return "vLLM production-stack + 分离式 + LMCache"
    if scale > 100:
        return "vLLM production-stack 或 SGLang"
    return "vLLM（单 GPU）"


# 演示
cases = [
    ("cpu", 1, "chat"),
    ("nvidia", 5000, "agentic"),
    ("amd", 500, "chat"),
    ("nvidia", 2000, "prefix_heavy"),
]
for hw, scale, wl in cases:
    engine = pick_engine(hw, scale, wl)
    print(f"硬件={hw:8s} 规模={scale:5d} 工作负载={wl:15s} → {engine}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 引擎对照

| 引擎 | 硬件 | 规模 | 工作负载 | 2026 状态 |
|---|---|---|---|---|
| llama.cpp | CPU/Apple/边缘 | <10 | 通用 | 主动开发 |
| Ollama | 全平台 | <5 | 开发 | 主动开发 |
| TGI | NVIDIA | <100 | 通用 | **维护模式** |
| vLLM | NVIDIA/AMD | 1-10000+ | 通用默认 | 主动开发 |
| SGLang | NVIDIA | 100-10000+ | 前缀密集/智能体 | 主动开发 |

---

## 5. 工程最佳实践

### 5.1 硬件第一、规模第二、工作负载第三

不要跳步。硬件决定了哪些引擎可用（AMD 不能用 TRT-LLM）。规模决定了单 GPU vs 集群。工作负载决定了 SGLang 是否有优势。

### 5.2 2026 年新项目默认避开 TGI

TGI 于 2025 年 12 月 11 日进入维护模式。新项目选 vLLM（通用）或 SGLang（前缀密集）。现有 TGI 部署应规划迁移。

### 5.3 中文场景特别建议

- **国内 GPU 与引擎匹配。** 华为昇腾 910B 需要 CANN 而非 CUDA，vLLM 通过 CANN 后端支持。但目前不支持 TRT-LLM 和 SGLang。国产推理引擎（MindIE、PLLama）的性能差异需要在选择前验证
- **国内开发者推荐。** 开发阶段用 Ollama（一键安装国内模型如 Qwen2、GLM-4）。生产阶段用国内硬件厂商支持的引擎或用 vLLM + 昇腾适配版
- **国内模型格式。** 中文模型（通义千问、GLM-4、DeepSeek 等）在 HuggingFace 上发布为 HF 格式，GGUF 版本通常在社区中。vLLM 直接支持 HF，llama.cpp 需要 GGUF

---

## 6. 常见错误

### 错误 1：开发和生产用不同引擎不测试量化差异

**现象：** 开发用 Ollama（FP16）没问题，生产用 vLLM（INT4）质量下降 3 点。

**原因：** 开发和生产的量化格式不同——FP16 vs INT4。质量差异没有被捕获。

**修复：** 全程使用相同格式的权重。开发用 GGUF Q4_K_M，暂存和用 Q4_K_M，生产也用 Q4_K_M。

### 错误 2：新手项目选 TGI

**现象：** 2026 年新项目选 TGI。TGI 进入维护模式后，功能不再更新，性能落后于 vLLM。

**原因：** 选择了不再主动开发的引擎。

**修复：** 2026 年新项目默认选 vLLM（通用）或 SGLang（前缀密集）。

---

## 7. 面试考点

### Q1：2026 年自托管推理引擎选型的决策树是什么？（难度：⭐⭐）

**参考答案：**
三层递进：硬件第一（CPU→llama.cpp，AMD→vLLM，NVIDIA→vLLM/SGLang/TRT-LLM，Apple→llama.cpp），规模第二（1 用户→Ollama，10-100→vLLM 单 GPU，100-10000→vLLM production-stack 或 SGLang），工作负载第三（通用聊天→vLLM，前缀密集/智能体→SGLang，长上下文→vLLM+Chunked Prefill）。

### Q2：TGI 进入维护模式对 2026 年新项目的影响是什么？（难度：⭐⭐）

**参考答案：**
TGI 于 2025 年 12 月 11 日进入维护模式——仅修复 bug。新项目应该默认 vLLM（通用生产）或 SGLang（前缀密集/智能体）。TGI 的优势是 HF 生态集成和可观测性——这些已被 vLLM 和 SGLang 追赶。现有 TGI 部署可以继续但应规划迁移。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| llama.cpp | "CPU 那个" | 最广模型支持，CPU 上最快 |
| Ollama | "笔记本那个" | 一键安装，开发级吞吐量 |
| TGI | "HF 的服务" | 2025 年 12 月起维护模式 |
| vLLM | "默认那个" | 2026 年广谱生产基线 |
| SGLang | "智能体那个" | 前缀密集型，RadixAttention |
| TRT-LLM | "NVIDIA 锁定" | Blackwell 吞吐量领先者，仅 NVIDIA |
| 流水线模式 | "开发→暂存→生产" | Ollama→llama.cpp→vLLM，相同权重 |

---

## 📚 小结

2026 年自托管推理引擎选型：硬件第一（CPU→llama.cpp，AMD→vLLM）、规模第二（Ollama→vLLM→production-stack）、工作负载第三（通用→vLLM，前缀密集→SGLang）。TGI 进入维护模式，新项目默认 vLLM 或 SGLang。流水线模式：开发用 Ollama、暂存用 llama.cpp、生产用 vLLM——全程相同权重。

**第 17 章到此结束。**

---

## ✏️ 练习

1. 运行 `code/main.py` 用你的硬件/规模/工作负载。输出是否与你的直觉一致？
2. 你的基础设施是 12 块 H100 + 8 块 MI300X AMD。用什么引擎？为什么 TRT-LLM 不在考虑范围内？
3. 开发 Ollama 到生产 vLLM：量化、配置、可观测性上需要什么变化？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 引擎选择决策树 | `code/main.py` | 三层递进决策树 |
| 引擎选型建议 | `outputs/skill-engine-picker.md` | 根据约束选择引擎和迁移路径 |

---

## 📖 参考资料

1. [博客] AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026. https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/
2. [GitHub] vLLM. https://github.com/vllm-project/vllm
3. [GitHub] SGLang. https://github.com/sgl-project/sglang
4. [GitHub] TGI Announcement. https://github.com/huggingface/text-generation-inference
