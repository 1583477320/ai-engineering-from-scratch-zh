# LLM API 负载测试——为什么 k6 和 Locust 在说谎

> 传统负载测试器不是为流式响应、可变输出长度、词元级指标或 GPU 饱和设计的。两个陷阱咬住大多数团队。GIL 陷阱：Locust 的词元级测量在 Python GIL 下运行分词，与请求生成竞争——分词积压膨胀了报告的词元间延迟。提示词均匀性陷阱：循环中的相同提示词测试了词元分布上的一个点。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 修复了这个问题。2026 年的工具映射：LLM 专用（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于词元级精度；k6 + k6 Operator 用于 CI/CD 门控。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 08（推理指标）、阶段 17 · 03（GPU 扩缩容）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释两个反模式（GIL 陷阱、提示词均匀性陷阱）为什么让通用负载测试器对 LLM API 说谎
- [ ] 根据目的选择工具：LLMPerf（基准运行）、k6 + 流扩展（CI 门控）、guidellm（大规模合成）、GenAI-Perf（NVIDIA 参考）
- [ ] 设计四种负载模式（稳态、爬坡、尖峰、浸泡）并说出每种捕获的故障模式
- [ ] 使用输入词元的均值+标准差而非固定长度构建真实的提示词分布

---

## 1. 问题

你用 k6 测试了 LLM 端点在 500 并发用户下——它撑住了。你发布了。在生产中 200 个真实用户时服务就垮了——P99 TTFT 爆炸，GPU 钉死。

两个问题发生了。第一，k6 发送了 500 个相同的提示词——你的请求合并和前缀缓存让它看起来像在处理 500 个并发 decode，实际上只处理了一个。第二，k6 没有按眼睛体验的方式追踪流式响应的词元间延迟。

**LLM 的负载测试是自己的学科。**

---

## 2. 概念

### 2.1 GIL 陷阱（Locust）

Locust 使用 Python，在 GIL 下运行客户端分词。高并发下分词器排在请求生成后面。报告的词元间延迟包含了客户端分词积压。你以为服务器慢了——是测试工具的瓶颈。

修复：LLM-Locust 扩展将分词移到独立进程，或使用编译语言工具（k6、LLMPerf 使用 tokenizers.rs）。

### 2.2 提示词均匀性陷阱

所有已知的负载测试器都让你配置一个提示词。10000 次迭代的循环中，每次发送完全相同的提示词。服务器每次看到相同的前缀——前缀缓存命中接近 100%，吞吐量看起来很好。

修复：从提示词分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——多样化的长度和内容。

### 2.3 四种负载模式

1. **稳态** — 恒定 RPS 持续 30-60 分钟。捕获：基线性能回归
2. **爬坡** — 15 分钟内从 0 线性增长到目标 RPS。捕获：容量断点、预热异常
3. **尖峰** — 突然 3-10 倍 RPS 持续 2 分钟然后恢复。捕获：扩缩容延迟、队列饱和、冷启动影响
4. **浸泡** — 稳态持续 4-8 小时。捕获：内存泄漏、连接池漂移、可观测性溢出

### 2.4 2026 年工具映射

| 工具 | 语言 | 最佳场景 |
|---|---|---|
| LLMPerf | Python + Rust 分词 | 性能基准运行 |
| GenAI-Perf | Python | NVIDIA 参考基准 |
| LLM-Locust | Python（Locust 扩展） | 熟悉 Locust DSL + 流式指标 |
| k6 + k6 Operator | Go | CI/CD 门控 + K8s 分布式 |
| Vegeta | Go | 网关/速率限制测试 |

### 2.5 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月
- 典型 LLMPerf 运行：100-1000 次请求，并发 X
- 典型 CI 门控：每次 PR 30-50 次迭代
- 四种模式：稳态、爬坡、尖峰、浸泡

---

## 3. 从零实现

### 第 1 步：真实提示词分布生成器

```python
import random


def generate_prompt_distribution(mean_tokens=500, std_tokens=150, n_prompts=100):
    """从正态分布采样真实提示词长度。"""
    lengths = [max(10, int(random.gauss(mean_tokens, std_tokens)))
               for _ in range(n_prompts)]
    return lengths


def compare_uniform_vs_realistic(n=1000):
    """对比均匀提示词 vs 真实分布的缓存命中率。"""
    # 均匀提示词——全部相同
    uniform_cache_hits = n  # 100% 命中（全部相同前缀）

    # 真实分布——不同长度和内容
    realistic_lengths = generate_prompt_distribution(n_prompts=n)
    unique_prefixes = len(set(l // 100 for l in realistic_lengths))
    realistic_cache_hits = max(1, unique_prefixes)  # 近似命中数

    print(f"均匀提示词缓存命中: {uniform_cache_hits}/{n} (100%)")
    print(f"真实分布缓存命中: 约 {min(10, unique_prefixes)}/{n} ({unique_prefixes/n:.1%})")
    print(f"差距: {uniform_cache_hits - realistic_cache_hits} 次虚假命中")


if __name__ == "__main__":
    compare_uniform_vs_realistic(1000)
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 工具选型对照

| 工具 | 最佳场景 | GIL 陷阱 | 流式感知 |
|---|---|---|---|
| LLMPerf | 性能基准 | 无（Rust 分词） | 是 |
| GenAI-Perf | NVIDIA 参考 | 无 | 是 |
| LLM-Locust | Locust 生态 | 修复 | 是 |
| k6 + Operator | CI/CD 门控 | 无（Go） | 是 |
| Vegeta | 网关测试 | 无（Go） | 否 |

---

## 5. 工程最佳实践

### 5.1 永远不要用相同提示词负载测试

这是 LLM 负载测试的第一准则。相同提示词 → 100% 缓存命中 → 吞吐量虚高。使用提示词分布采样。

### 5.2 CI 门控用 k6

每次 PR 运行 30-50 次迭代，门控在 P95 TTFT、5xx 率、TPOT。破坏构建时阻断发布。

### 5.3 中文场景特别建议

- **中文提示词的分词开销。** 中文分词的计算开销比英文大——Locust 的 GIL 陷阱在中文上更明显
- **中文负载测试工具。** LLMPerf 和 k6 都是语言无关的，可以用于中文负载测试
- **国内 LLM 的速率限制。** 国内 LLM API 的 QPS 限制通常更低——负载测试时要模拟真实限流

---

## 6. 常见错误

### 错误 1：用相同提示词负载测试

**现象：** 负载测试显示 500 并发没问题。上线后 200 并发就垮了。

**原因：** 相同提示词 → 100% 缓存命中 → 吞吐量虚高。生产中的提示词各不相同。

**修复：** 使用 `--mean-input-tokens 500 --stddev-input-tokens 150` 采样真实分布。

### 错误 2：用 Locust 做 LLM 负载测试

**现象：** Locust 报告的词元间延迟很高——但服务器日志显示正常。

**原因：** Locust 的 GIL 陷阱——分词在 Python GIL 下运行，与请求生成竞争。

**修复：** 使用 LLM-Locust 扩展（移分词到独立进程）或使用 k6/LLMPerf。

---

## 7. 面试考点

### Q1：LLM 负载测试的两个反模式是什么？（难度：⭐⭐）

**参考答案：**
GIL 陷阱：Locust 在 Python GIL 下运行分词，与请求生成竞争——报告的延迟包含了客户端分词积压，不是服务器延迟。提示词均匀性陷阱：循环中的相同提示词测试了缓存命中的一个点——生产中的提示词各不相同，缓存命中率远低于测试。修复：使用 LLM 专用工具（LLMPerf、k6）+ 真实提示词分布采样。

### Q2：四种负载模式分别捕获什么故障？（难度：⭐⭐）

**参考答案：**
稳态（30-60 分钟恒定 RPS）：基线性能回归、长时间运行下的缓慢退化。爬坡（15 分钟从 0 到目标）：容量断点、扩缩容触发点、预热异常。尖峰（突然 3-10 倍持续 2 分钟）：扩缩容延迟、队列饱和、冷启动影响。浸泡（4-8 小时稳态）：内存泄漏、连接池漂移、可观测性溢出。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| GIL 陷阱 | "Python 客户端开销" | 分词积压膨胀了报告的延迟 |
| 提示词均匀性陷阱 | "单一提示词的谎言" | 相同提示词循环命中缓存，膨胀吞吐量 |
| LLMPerf | "LLM 测试工具" | Anyscale 的流式感知基准工具 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 参考基准工具 |
| k6 Operator | "K8s k6" | 基于 CRD 的分布式 k6 |
| 稳态 | "恒定负载" | 固定 RPS 持续 N 分钟 |
| 爬坡 | "线性上升" | 0 到目标持续一段时间 |
| 尖峰 | "爆发测试" | 突然倍数然后恢复 |
| 浸泡 | "长时间测试" | 数小时用于检测泄漏 |

---

## 📚 小结

LLM 负载测试有两个核心陷阱：GIL 陷阱（客户端分词膨胀延迟）和提示词均匀性陷阱（相同提示词命中缓存）。四种负载模式：稳态、爬坡、尖峰、浸泡，每种捕获不同的故障。LLMPerf 和 k6 是 2026 年的最佳选择——前者用于性能基准，后者用于 CI/CD 门控。永远不要用相同提示词负载测试。

---

## ✏️ 练习

1. 运行 `code/main.py`。对比均匀 vs 真实分布——差距在哪里？
2. 写一个 k6 脚本用于 CI 门控：100 并发下 P95 TTFT < 800ms，运行 5 分钟。
3. GenAI-Perf 报告 TPOT=6ms，LLMPerf 报告 TPOT=11ms（同一服务器）。解释原因。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 负载测试模拟器 | `code/main.py` | 真实提示词分布 + 统一提示词对比 |
| 负载测试方案 | `outputs/skill-load-test-plan.md` | 根据工作负载和 SLA 选择工具和负载模式 |

---

## 📖 参考资料

1. [GitHub] LLMPerf. https://github.com/ray-project/llmperf
2. [GitHub] k6 Operator. https://github.com/grafana/k6-operator
3. [文档] NVIDIA NIM — LLM Inference Benchmarking. https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html
4. [博客] TrueFoundry — LLM-Locust. https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance
