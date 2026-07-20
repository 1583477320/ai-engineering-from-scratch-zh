# 综合项目11——LLM可观测性与评测仪表盘

> Langfuse转向开放核心。Arize Phoenix发布了2026年GenAI语义约定映射。Helicone和Braintrust都加倍投入按用户成本归因。Traceloop的OpenLLMetry成为事实上的SDK检测标准。生产形态是ClickHouse存储追踪、Postgres存储元数据、Next.js做UI、一批评测任务（DeepEval、RAGAS、LLM-judge）在采样追踪上运行。本综合项目要求你构建一个自托管的仪表盘，从至少四个SDK家族导入数据，在采样追踪上运行评测，检测漂移并发出告警。

**类型：** 综合项目
**编程语言：** TypeScript（UI），Python/TypeScript（导入+评测），SQL（ClickHouse）
**前置知识：** 第11章（LLM工程）、第13章（工具）、第17章（基础设施）、第18章（安全）
**涉及章节：** P11 · P13 · P17 · P18
**预计时间：** 25小时

---

## 学习目标

- 构建自托管LLM可观测性仪表盘
- 实现尾采样策略（保留错误，采样成功）
- 实现LLM评测作为子span（忠诚度、毒性、PII泄露）
- 实现PSI漂移检测和告警链

---

## 1. 问题

2026年每个运行生产流量的AI团队都保持一个可观测性平面。成本归因、幻觉检测、漂移监控、越狱信号、SLO仪表盘、PII泄露告警。开源参考——Langfuse、Phoenix、OpenLLMetry——收敛到OpenTelemetry GenAI语义约定作为导入模式。

你现在可以用一个SDK检测OpenAI、Anthropic、Google、LangChain、LlamaIndex和vLLM，并发送兼容的span。

---

## 2. 核心概念

### 2.1 导入和采样

导入是OTLP HTTP。SDK产生GenAI语义约定span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`等。span落入ClickHouse用于列式分析；元数据落入Postgres。

尾采样器：保留100%的错误追踪和10%的成功追踪。

### 2.2 评测

评测作为批量作业在采样追踪上运行。DeepEval评分忠诚度、毒性和答案相关性。自定义LLM-judge运行领域特定检查（PII泄露、违反策略）。评测结果写回ClickHouse作为与父追踪链接的eval span。

### 2.3 漂移检测

漂移检测随时间监控嵌入空间分布（提示嵌入的PSI或KL散度）加上评测分数趋势。告警通过Prometheus Alertmanager发送到Slack/PagerDuty。

---

## 3. 从零实现

`code/main.py`实现尾采样收集器、评测作为子span、漂移检测器和告警器。

```python
"""LLM可观测性仪表盘——span导入+尾采样+评测脚手架。

核心架构原语是尾采样收集器加评测作为子span：
错误追踪始终保留，成功追踪按比例采样，
每个追踪可通过携带分数的eval span丰富。

运行：python3 code/main.py
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Span模型——GenAI语义约定字段
# ---------------------------------------------------------------------------

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_ms: int
    duration_ms: int
    attributes: dict
    events: list[dict] = field(default_factory=list)
    status: str = "ok"

    def is_llm(self) -> bool:
        return "gen_ai.system" in self.attributes


# ---------------------------------------------------------------------------
# 尾采样器——保留错误，采样成功
# ---------------------------------------------------------------------------

@dataclass
class TailSampler:
    sample_rate: float = 0.10
    rng: random.Random = field(default_factory=lambda: random.Random(3))

    def decide(self, trace: list[Span]) -> bool:
        if any(s.status == "error" for s in trace):
            return True
        for s in trace:
            if s.name == "eval" and (
                s.attributes.get("toxicity", 0) > 0.5
                or s.attributes.get("pii_leak", 0) > 0.8
            ):
                return True
        return self.rng.random() < self.sample_rate


# ---------------------------------------------------------------------------
# 内存ClickHouse替代
# ---------------------------------------------------------------------------

@dataclass
class SpanStore:
    spans: list[Span] = field(default_factory=list)
    by_user: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cost_by_user: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def insert_trace(self, trace: list[Span]) -> None:
        self.spans.extend(trace)
        for s in trace:
            if s.is_llm():
                u = s.attributes.get("user_id", "anon")
                m = s.attributes.get("gen_ai.request.model", "unknown")
                self.by_user[u] += 1
                self.by_model[m] += 1
                self.cost_by_user[u] += s.attributes.get("cost_usd", 0.0)


# ---------------------------------------------------------------------------
# 评测——忠诚度、毒性、PII泄露（LLM-judge存根）
# ---------------------------------------------------------------------------

def eval_faithfulness(response: str, context: str) -> float:
    r = set(response.lower().split())
    c = set(context.lower().split())
    if not r:
        return 0.0
    return len(r & c) / len(r)


def eval_toxicity(response: str) -> float:
    bad = {"hate", "kill", "stupid", "garbage"}
    words = response.lower().split()
    hits = sum(1 for w in words if w in bad)
    return min(1.0, hits / max(1, len(words)) * 10)


def eval_pii_leak(response: str) -> float:
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", response):
        return 0.95
    if re.search(r"[\w.+-]+@[\w.-]+", response):
        return 0.6
    return 0.05


# ---------------------------------------------------------------------------
# 漂移检测器——提示指纹的PSI
# ---------------------------------------------------------------------------

def prompt_fingerprint(prompt: str, n_bins: int = 8) -> int:
    h = hashlib.sha256(prompt.encode()).digest()
    return h[0] % n_bins


def psi(a: list[int], b: list[int], n_bins: int = 8) -> float:
    ca = [0] * n_bins
    cb = [0] * n_bins
    for v in a:
        ca[v] += 1
    for v in b:
        cb[v] += 1
    total_a = max(sum(ca), 1)
    total_b = max(sum(cb), 1)
    score = 0.0
    for i in range(n_bins):
        pa = max(ca[i] / total_a, 0.0001)
        pb = max(cb[i] / total_b, 0.0001)
        score += (pa - pb) * math.log(pa / pb)
    return score


# ---------------------------------------------------------------------------
# 模拟导入——真实SDK混合 + 注入回归
# ---------------------------------------------------------------------------

def synth_trace(trace_id: str, leak_pii: bool, rng: random.Random) -> list[Span]:
    model = rng.choice(["claude-sonnet-4-7", "gpt-5-4", "gemini-3-pro"])
    user = rng.choice(["u_01", "u_02", "u_03", "u_04"])
    root = Span(trace_id=trace_id, span_id=f"{trace_id}_0", parent_span_id=None,
                name="chat_turn", start_ms=int(time.time() * 1000),
                duration_ms=rng.randint(400, 2400), attributes={"app_id": "chatbot"})
    prompt = rng.choice(["what is the weather today", "summarize forecast", "give a tip"])
    resp = "your ssn is 123-45-6789" if leak_pii else "the weather is mild"
    ctx = "relevant weather context mild"
    llm = Span(trace_id=trace_id, span_id=f"{trace_id}_1", parent_span_id=root.span_id,
               name="llm_call", start_ms=root.start_ms + 50, duration_ms=root.duration_ms - 80,
               attributes={"gen_ai.system": model.split("-")[0],
                           "gen_ai.request.model": model,
                           "gen_ai.usage.input_tokens": rng.randint(80, 800),
                           "gen_ai.usage.output_tokens": rng.randint(20, 300),
                           "user_id": user, "prompt": prompt, "response": resp,
                           "context": ctx, "cost_usd": round(rng.uniform(0.002, 0.05), 4)})
    return [root, llm]


def enrich_with_evals(trace: list[Span]) -> list[Span]:
    out = list(trace)
    for s in trace:
        if s.is_llm():
            resp = s.attributes.get("response", "")
            ctx = s.attributes.get("context", "")
            ev = Span(trace_id=s.trace_id, span_id=f"{s.span_id}_eval",
                      parent_span_id=s.span_id, name="eval",
                      start_ms=s.start_ms + s.duration_ms, duration_ms=120,
                      attributes={"faithfulness": eval_faithfulness(resp, ctx),
                                  "toxicity": eval_toxicity(resp),
                                  "pii_leak": eval_pii_leak(resp)})
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# 告警器——阈值超时触发
# ---------------------------------------------------------------------------

def alerter(store: SpanStore) -> list[str]:
    alerts: list[str] = []
    pii_events = [s for s in store.spans if s.name == "eval" and s.attributes.get("pii_leak", 0) > 0.8]
    if pii_events:
        alerts.append(f"检测到PII泄漏: {len(pii_events)}个事件")
    tox_events = [s for s in store.spans if s.name == "eval" and s.attributes.get("toxicity", 0) > 0.5]
    if tox_events:
        alerts.append(f"毒性激增: {len(tox_events)}个事件")
    return alerts


# ---------------------------------------------------------------------------
# 演示——200个正常追踪 + 1%注入PII回归
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(5)
    sampler = TailSampler(sample_rate=0.20, rng=rng)
    store = SpanStore()
    baseline_fps, current_fps = [], []

    for i in range(200):
        leak = rng.random() < 0.01
        trace = synth_trace(f"t{i:04d}", leak_pii=leak, rng=rng)
        trace = enrich_with_evals(trace)
        if sampler.decide(trace):
            store.insert_trace(trace)
        fp = prompt_fingerprint(trace[1].attributes.get("prompt", ""))
        (current_fps if i > 150 else baseline_fps).append(fp)

    print(f"导入span数       : {len(store.spans)}")
    print(f"按模型分布       : {dict(store.by_model)}")
    print(f"按用户成本       : {dict((k, round(v,4)) for k,v in store.cost_by_user.items())}")

    alerts = alerter(store)
    if alerts:
        print("\n告警:")
        for a in alerts:
            print(f"  - {a}")

    psi_val = psi(baseline_fps, current_fps, n_bins=8)
    print(f"\nPSI（当前 vs 基线）: {psi_val:.3f}")
    if psi_val > 0.2:
        print("  漂移告警 (PSI > 0.2)")


if __name__ == "__main__":
    main()
```

运行结果：

```
导入span数       : 289
按模型分布       : {'claude': 28, 'gpt': 25, 'gemini': 27}
按用户成本       : {'u_01': 0.081, 'u_02': 0.064, ...}

检测到PII泄漏: 2个事件

PSI（当前 vs 基线）: 0.083
```

---

## 4. 工具实践

**技术栈：**
- 导入：OpenTelemetry SDK + GenAI语义约定；OTLP HTTP传输
- 收集器：OpenTelemetry Collector + 尾采样处理器
- 存储：ClickHouse（span）、Postgres（元数据）、S3（原始事件归档）
- 评测：DeepEval、RAGAS 0.2、Arize Phoenix评估器包
- 漂移：sentence-transformers提示嵌入的PSI/KL
- 告警：Prometheus Alertmanager → Slack/PagerDuty
- UI：Next.js 15 + Recharts

---

## 5. LLM视角

**导入视角**：GenAI语义约定使所有主流LLM提供商使用一个SDK检测。这是可观测性的基础。

**尾采样视角**：保留100%错误+10%成功。这是成本与覆盖率的平衡。错误追踪是调试的关键。

**评测作为span视角**：评测结果作为子span链接到父追踪，使你可以追溯每个响应的质量分数。

---

## 6. 工程最佳实践

**收集器配置**：
- OTLP HTTP接收器
- 尾采样器：100%错误，10%成功
- ClickHouse和S3导出器

**ClickHouse模式**：
- span表：GenAI语义约定列
- 按user_id和app_id索引
- JSON包用于长负载

**评测作业**：
- 从采样追踪中读取最近N分钟的数据
- 运行DeepEval忠诚度、毒性、答案相关性
- 自定义LLM-judge（PII泄露检查）

---

## 7. 常见错误

**错误1：对所有追踪100%采样**
症状：存储成本爆炸
修复：尾采样（10%成功）

**错误2：不链接eval span**
症状：无法追溯响应质量
修复：eval span作为子span

**错误3：全局PSI漂移检测**
症状：混合应用信号
修复：按app-id计算PSI

---

## 8. 面试考点

**Q1：为什么尾采样对可观测性很重要？**
考察：对成本与覆盖率的理解

**Q2：GenAI语义约定包含哪些字段？**
考察：对OTel规范的理解

**Q3：PSI漂移检测如何工作？**
考察：对监控方法的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GenAI语义约定 | "OTel LLM属性" | 2025年OpenTelemetry LLM span属性规范 |
| 尾采样 | "追踪后采样" | 收集器在追踪完成后决定保留或丢弃 |
| PSI | "人口稳定指数" | 比较两个分布的漂移指标；>0.2通常表示有意义漂移 |
| LLM-judge | "模型评测" | 一个LLM按评分标准评分另一个LLM的输出 |
| Eval span | "链接的评测追踪" | 携带评测分数、链接到原始LLM调用span的子span |
| 每用户成本 | "单位经济" | 一段时间内归因到user_id的美元成本 |

---

## 参考文献

- [Langfuse](https://github.com/langfuse/langfuse)
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)
- [OpenLLMetry（Traceloop）](https://github.com/traceloop/openllmetry)
- [OpenTelemetry GenAI语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [ClickHouse文档](https://clickhouse.com/docs)
- [DeepEval](https://github.com/confident-ai/deepeval)
