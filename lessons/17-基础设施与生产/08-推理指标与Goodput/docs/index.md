# 推理指标——TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定一个推理部署是否在工作。TTFT 是 prefill 加排队加网络。TPOT（等同 ITL）是每个词元的内存带宽受限解码成本。端到端延迟是 TTFT 加 TPOT 乘以输出长度。吞吐量是整个集群每秒的词元总数。但对产品真正重要的是 **goodput**——同时满足所有 SLO 的请求比例。高吞吐量 + 低 goodput = 你在处理永远无法及时送达用户的词元。2026 年始终报告 P50/P90/P99——永远不要只报告均值。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 04（vLLM 内部原理）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 精确定义 TTFT、TPOT、ITL、E2E、吞吐量和 goodput，并说出每个测量的是什么组件
- [ ] 解释为什么均值对 LLM 服务是错误的统计量，以及如何读 P50/P90/P99
- [ ] 构建多约束 SLO（如 TTFT < 500ms 且 TPOT < 15ms 且 E2E < 2s）并计算 goodput
- [ ] 命名两个在同一运行中对 TPOT 给出不同结果的基准测试工具，并解释原因

---

## 1. 问题

"我们的吞吐量是 15,000 tok/s。"所以呢？如果 40% 的请求端到端超过 2 秒，用户就放弃了。仅凭吞吐量无法告诉你产品是否工作。

推理有多个延迟轴，每个都以不同方式失败。Prefill 是计算密集型的，随提示长度缩放。解码是内存带宽密集型的，随批大小缩放。排队延迟是运维问题。网络是物理距离问题。你需要为每个轴定义独立的指标，你需要百分位数，你需要一个复合指标告诉你"用户是否得到了预期的体验"——那就是 goodput。

---

## 2. 概念

### 2.1 TTFT——首个词元延迟

`TTFT = 排队时间 + 网络请求时间 + prefill 时间`

长提示时 prefill 主导。在 Llama-3.3-70B FP8 + H100 上，32K 提示需要约 800ms 纯 prefill。排队时间是调度器在负载下的行为。网络请求是包括 TLS 在内的线路时间。TTFT 是用户在看到第一个词元之前的等待时间。

### 2.2 TPOT / ITL——词元间延迟

多个名字指同一个量。`TPOT`（每个输出词元的时间）、`ITL`（词元间延迟）、`decode latency per token`——都是一样的。它是首个词元之后连续流式词元之间的时间。

`TPOT = (解码前向时间 + 调度器开销) / 产出的词元数`

在 Llama-3.3-70B H100 + Chunked Prefill 下，TPOT 均值约 7ms。无 Chunked Prefill 时，邻居序列的长 prefill 期间 TPOT 可飙升到 50ms。**看 P99，不要看均值。**

### 2.3 端到端延迟

`E2E = TTFT + TPOT × 输出词元数 + 网络响应时间`

长输出（>500 词元）时 E2E 由 TPOT 主导。长提示短输出时 E2E 由 TTFT 主导。报告按输出长度分层的 E2E。

### 2.4 吞吐量

`吞吐量 = 总输出词元数 / 经过时间`

聚合指标。告诉你集群效率。不告诉你单个请求的健康状况。

### 2.5 Goodput——你真正关心的指标

`goodput = 满足 (TTFT ≤ a) 且 (TPOT ≤ b) 且 (E2E ≤ c) 的请求比例`

SLO 是多约束的。一个请求只有在所有约束都满足时才算"好"。Goodput 是满足所有约束的比例。60% goodput 下的高吞吐量是失败。99% goodput 下较低的吞吐量才是目标。

2026 年，goodput 是 MLPerf Inference v6.0 提交和 AI 平台提供商内部 SLA 跟踪中使用的指标。

### 2.6 为什么均值是错误的统计量

LLM 延迟分布是右偏的。一个 decode 批次中有一个长 prefill 邻居时，可以发送 500 个词元 TPOT 约 7ms，20 个词元 TPOT 约 60ms。均值 TPOT 是 9ms。P99 TPOT 是 65ms。用户会频繁碰到 P99——这就是他们离开的原因。

**始终报告三元组 (P50, P90, P99)。** 用户体验优化的是 P99。

### 2.7 基准数字——Llama-3.1-8B-Instruct + TRT-LLM（2026）

- 均值 TTFT：162ms
- 均值 TPOT：7.33ms
- 均值 E2E：1093ms
- P99 TPOT：因 Chunked Prefill 配置不同在 10-25ms 之间

### 2.8 度量陷阱

2026 年两个最常用的基准测试工具对同一运行的 TPOT 给出不同结果：

- **NVIDIA GenAI-Perf：** 将 TTFT 从 ITL 计算中排除。ITL 从词元 2 开始
- **LLMPerf：** 包含 TTFT。ITL 从词元 1 开始

TTFT 500ms + 100 输出词元在 700ms 内完成解码时，GenAI-Perf 报告 `ITL = 700/99 = 7.07ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00ms`。工具选择改变了数字。

**始终声明使用哪个工具。始终发布定义。**

### 2.9 构建 SLO

2026 年 70B 聊天模型的合理面向消费者 SLO：

- TTFT P99 ≤ 800ms
- TPOT P99 ≤ 25ms
- E2E P99 ≤ 3s（<300 词元输出）
- Goodput 目标 ≥ 99%

企业 SLO 收紧 TTFT（200-400ms），放松 E2E。关键是写下来，测量所有三个指标，跟踪 goodput 作为单一复合指标。

### 2.10 如何测量

- 运行真实流量或逼真的合成负载（LLMPerf 参数：`--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）
- 基准测试运行时目标并发量为峰值的 2 倍
- 运行 30-50 次迭代，取合并样本的百分位数
- 发布时附带工具名称、版本、模型、硬件、并发量、提示分布

---

## 3. 从零实现

### 第 1 步：Goodput 计算器

```python
import random


def compute_goodput(ttftps: list, tpots: list, e2es: list,
                    ttft_slo: float, tpot_slo: float, e2e_slo: float):
    """计算满足多约束 SLO 的请求比例。"""
    total = len(ttftps)
    good = sum(
        1 for t, p, e in zip(ttftps, tpots, e2es)
        if t <= ttft_slo and p <= tpot_slo and e <= e2e_slo
    )
    return good / total if total > 0 else 0


# 模拟延迟分布
random.seed(42)
n = 1000
ttfts = [random.gauss(200, 80) for _ in range(n)]
tpots = [random.gauss(8, 3) for _ in range(n)]
e2es = [t + p * random.randint(50, 300) for t, p in zip(ttfts, tpots)]

# 不同 SLO 下的 goodput
for ttft_slo, tpot_slo, e2e_slo in [(800, 25, 3000), (500, 15, 2000), (300, 10, 1000)]:
    g = compute_goodput(ttfts, tpots, e2es, ttft_slo, tpot_slo, e2e_slo)
    print(f"TTFT≤{ttft_slo}ms TPOT≤{tpot_slo}ms E2E≤{e2e_slo}ms → Goodput={g:.1%}")
```

### 第 2 步：P99 尾部分析

```python
def percentile(data, p):
    """计算第 p 百分位数。"""
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def latency_summary(latencies, name=""):
    """延迟分布摘要。"""
    print(f"{name:10s}  P50={percentile(latencies, 50):.1f}  "
          f"P90={percentile(latencies, 90):.1f}  "
          f"P99={percentile(latencies, 99):.1f}  "
          f"均值={sum(latencies)/len(latencies):.1f}")


latency_summary(tpots, "TPOT")
latency_summary(ttfts, "TTFT")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 基准测试工具对照

| 工具 | TPOT 定义 | 输出 | 适用场景 |
|---|---|---|---|
| GenAI-Perf (NVIDIA) | 不含 TTFT | 延迟 + 吞吐 | NVIDIA 硬件优化 |
| LLMPerf (Ray) | 包含 TTFT | 延迟 + 吞吐 | 开源、通用 |
| vLLM Bench | 可配置 | 吞吐 + 延迟 | vLLM 专用 |

### 4.2 SLO 建议

| 场景 | TTFT P99 | TPOT P99 | E2E P99 | Goodput 目标 |
|---|---|---|---|---|
| 聊天（70B） | ≤ 800ms | ≤ 25ms | ≤ 3s | ≥ 99% |
| 企业（70B） | ≤ 400ms | ≤ 15ms | ≤ 5s | ≥ 99.5% |
| 代码补全 | ≤ 200ms | ≤ 20ms | ≤ 2s | ≥ 99% |
| 语音助手 | ≤ 200ms | ≤ 15ms | N/A | ≥ 99% |

---

## 5. 工程最佳实践

### 5.1 Goodput 是唯一的复合指标

吞吐量和延迟分别看没有意义——它们是权衡关系。Goodput 将所有 SLO 约束合并为一个数字：满足所有约束的请求比例。这是你跟踪的唯一指标。

### 5.2 始终报告 P50/P90/P99

只报告均值是一个陷阱。右偏分布中均值可能在 P90 以下——但用户体验的是 P99。

### 5.3 始终声明工具和定义

GenAI-Perf 和 LLMPerf 对 TPOT 的定义不同。不在报告中声明工具和定义的数字不可比较。

### 5.4 中文场景特别建议

- **中文 TTFT 通常更高。** 同样的模型，中文提示词的词元密度更高——同样长度的语义信息需要更多词元。TTFT 中的 prefill 部分更长
- **中文聊天的 TPOT 略高。** 中文词元的解码需要更多计算（词表更大、嵌入维度可能不同）。在设置 SLO 时，中文场景的 TPOT P99 阈值应该比英文宽松 20-30%
- **国内 LLMPerf 部署。** LLMPerf 是开源的，可以在国内任何环境运行。GenAI-Perf 需要 NVIDIA Triton，部署成本更高

---

## 6. 常见错误

### 错误 1：只报告吞吐量

**现象：** "吞吐量 15,000 tok/s。" 用户投诉延迟高。40% 的请求 E2E > 2s。

**原因：** 吞吐量是聚合指标，不反映单个请求的健康状况。高吞吐量 + 低 goodput = 在处理永远无法及时送达用户的词元。

**修复：** 报告 goodput（满足所有 SLO 的请求比例），不只是吞吐量。

### 错误 2：只看均值

**现象：** "均值 TPOT 只有 8ms。" 但 P99 是 65ms，用户体验很差。

**原因：** LLM 延迟分布是右偏的——均值被大多数快速请求拉低，掩盖了尾部。

**修复：** 报告 (P50, P90, P99) 三元组。优化 P99。

---

## 7. 面试考点

### Q1：为什么 goodput 比吞吐量和延迟分开看更重要？（难度：⭐⭐）

**参考答案：**
吞吐量和延迟是权衡关系——增加并发可以提高吞吐量但增加延迟。单独看任何一个都可能误导。Goodput 将所有 SLO 约束合并为一个数字：同时满足 TTFT ≤ X 且 TPOT ≤ Y 且 E2E ≤ Z 的请求比例。60% goodput 下的高吞吐量意味着 40% 的用户体验差——这是失败。Goodput 直接衡量"用户体验是否达标"。

### Q2：GenAI-Perf 和 LLMPerf 对 TPOT 的定义为什么不同？（难度：⭐⭐）

**参考答案：**
GenAI-Perf 将 ITL（TPOT）定义为从词元 2 开始的词元间延迟——不包含 TTFT。LLMPerf 将 ITL 定义为从词元 1 开始的平均延迟——包含 TTFT。对于 TTFT 500ms + 100 词元在 700ms 内解码的情况，GenAI-Perf 报 7.07ms（不含 TTFT），LLMPerf 报 12.00ms（含 TTFT）。差异来自定义，不是测量。始终声明使用哪个工具。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| TTFT | "首词元延迟" | 排队 + 网络 + prefill；长提示时被 prefill 主导 |
| TPOT | "每词元时间" | 首词元之后内存带宽受限的 decode 成本 |
| ITL | "词元间延迟" | 多数工具中与 TPOT 相同（但 GenAI-Perf 不同） |
| E2E | "端到端" | TTFT + TPOT × 输出长度；外加响应侧网络 |
| Goodput | "SLO 达标率" | 同时满足所有 SLO 约束的请求比例 |
| P99 | "尾部" | 百分之一的最差延迟；用户体验指标 |
| SLO 多约束 | "联合约束" | 所有延迟边界的 AND；任一约束违反即失败 |

---

## 📚 小结

推理指标有四个维度：TTFT（首词元延迟）、TPOT（词元间延迟）、E2E（端到端延迟）、吞吐量。但真正决定产品成败的是 **goodput**——同时满足所有 SLO 约束的请求比例。高吞吐量 + 低 goodput = 在处理无法及时送达的词元。始终报告 P50/P90/P99，永远不要只报告均值。注意 GenAI-Perf 和 LLMPerf 对 TPOT 的定义差异——始终声明工具和定义。

---

## ✏️ 练习

1. 运行 `code/main.py`。生成一个 1% 尾部尖峰的分布。将 P99 TPOT 从 30ms 收紧到 15ms 时，goodput 如何变化？
2. 一个供应商报价"15,000 tok/s on Llama 3.3 70B H100"。在信任这个数字之前，问三个问题。
3. 为什么 Chunked Prefill 保护 P99 TPOT 但不保护均值 TPOT？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| Goodput 计算器 | `code/main.py` | 生成延迟分布、应用 SLO、计算 goodput |
| SLO 门控方案 | `outputs/skill-slo-goodput-gate.md` | CI/CD 就绪的基准测试方案，用 goodput 门控部署 |

---

## 📖 参考资料

1. [官方文档] NVIDIA NIM — LLM Benchmarking Metrics. https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html
2. [官方文档] Anyscale — LLM Serving Benchmarking Metrics. https://docs.anyscale.com/llm/serving/benchmarking/metrics
3. [GitHub] LLMPerf. https://github.com/ray-project/llmperf
4. [NVIDIA] GenAI-Perf. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html
5. [官方文档] MLPerf Inference. https://mlcommons.org/benchmarks/inference-datacenter/
