# 综合项目28——OTel 可观测性与 Prometheus 指标（Observability with OTel Spans & Prometheus Metrics）

> 没有可观测性的智能体框架是一个烧钱的黑盒。本节手写一个符合 OpenTelemetry GenAI 语义约定的 span 构建器，写入 JSONL 文件，并以 Prometheus 文本格式暴露计数器和直方图。

**类型：** 构建
**语言：** Python（标准库）
**前置知识：** 第19章第25-27节
**预计时间：** 90分钟

---

## 学习目标

- 构建符合 OTel GenAI 语义约定的 span 数据类
- 实现 JSONL 导出器
- 构建带标签的计数器和直方图，输出 Prometheus 文本格式
- 用上下文管理器包装任意可调用对象，记录持续时间和异常

---

## 1. 问题

生产编码智能体每次运行产生三类产物：模型调用、工具执行和验证门决策。没有结构化遥测，这些都不够用。

第一种失败模式是缺失追踪——周二出了问题但只有 500 行聊天日志。第二种是不可解析的追踪——自定义字段名，Grafana/Honeycomb 无法读取。第三种是未聚合的指标——能看到一次慢调用但无法回答"过去一小时 p95 延迟是多少"。

OTel GenAI 语义约定正是为此存在。

---

## 2. 核心概念

### 2.1 Span 结构

```text
GenAISpan
  trace_id: str        # 16字节十六进制（整个智能体调用）
  span_id: str         # 16字节十六进制（单次操作）
  name: str            # gen_ai.chat, gen_ai.tool.execution
  attributes: dict     # gen_ai.system, gen_ai.request.model, ...
  start/end: int       # 纳秒时间戳
  status: str          # OK | ERROR
```

### 2.2 GenAI 属性约定

```text
gen_ai.system               # 提供商 (anthropic, openai)
gen_ai.request.model        # 模型 ID
gen_ai.usage.input_tokens   # 输入词元数
gen_ai.usage.output_tokens  # 输出词元数
gen_ai.tool.name            # 工具名称
```

### 2.3 指标注册表

- **计数器**：`tools_called_total{tool="read_file"}` — 每次调用递增
- **直方图**：`tool_latency_ms{tool="read_file"}` — 记录延迟分布

### 2.4 Prometheus 文本格式

```text
# HELP tools_called_total Total tool calls
# TYPE tools_called_total counter
tools_called_total{tool="read_file"} 5

# HELP tool_latency_ms Tool call latency in ms
# TYPE tool_latency_ms histogram
tool_latency_ms_bucket{le="50"} 3
tool_latency_ms_bucket{le="+Inf"} 5
tool_latency_ms_sum 120.5
tool_latency_ms_count 5
```

---

## 3. 从零实现

```python
"""OTel 可观测性——span + JSONL 导出 + Prometheus 指标。"""
import json, os, time, uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable


@dataclass
class GenAISpan:
    trace_id: str; span_id: str; name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    start_ns: int = 0; end_ns: int = 0
    status: str = "OK"; status_message: str = ""


class SpanBuilder:
    def __init__(self, exporter=None):
        self.exporter = exporter or JSONLExporter()
        self._trace_id = uuid.uuid4().hex[:32]

    def span(self, name: str, attributes: Dict[str, Any] = None):
        return SpanContext(self, name, attributes or {})


class SpanContext:
    def __init__(self, builder: SpanBuilder, name: str, attrs: Dict[str, Any]):
        self._builder = builder; self._name = name; self._attrs = attrs
        self._span = GenAISpan(
            trace_id=builder._trace_id,
            span_id=uuid.uuid4().hex[:32],
            name=name, attributes=dict(attrs),
            start_ns=time.time_ns(),
        )

    def __enter__(self): return self._span
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._span.end_ns = time.time_ns()
        if exc_type:
            self._span.status = "ERROR"
            self._span.status_message = f"{exc_type.__name__}: {exc_val}"
        self._builder.exporter.export(self._span)


class JSONLExporter:
    def __init__(self):
        self.spans: List[Dict] = []
    def export(self, span: GenAISpan):
        self.spans.append({
            "trace_id": span.trace_id, "span_id": span.span_id, "name": span.name,
            "attributes": span.attributes,
            "start_ns": span.start_ns, "end_ns": span.end_ns,
            "duration_ms": (span.end_ns - span.start_ns) / 1e6,
            "status": span.status, "status_message": span.status_message,
        })


class MetricsRegistry:
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.histograms: Dict[str, List[float]] = {}

    def inc_counter(self, name: str, labels: Dict[str, str], value: int = 1):
        key = f"{name}{{{','.join(f'{k}={v}' for k,v in labels.items())}}}"
        self.counters[key] = self.counters.get(key, 0) + value

    def observe(self, name: str, labels: Dict[str, str], value: float):
        key = f"{name}{{{','.join(f'{k}={v}' for k,v in labels.items())}}}"
        if key not in self.histograms: self.histograms[key] = []
        self.histograms[key].append(value)

    def exposition(self) -> str:
        lines = []
        for key, count in self.counters.items():
            metric = key.split("{")[0]
            lines.append(f"# HELP {metric} Counter")
            lines.append(f"# TYPE {metric} counter")
            lines.append(f"{key} {count}")
            lines.append("")
        buckets = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        for key, values in self.histograms.items():
            metric = key.split("{")[0]
            lines.append(f"# HELP {metric} Histogram")
            lines.append(f"# TYPE {metric} histogram")
            for b in buckets:
                count = sum(1 for v in values if v <= b)
                lines.append(f'{metric}_bucket{{le="{b}",{key.split("{")[1]}}} {count}')
            lines.append(f'{metric}_bucket{{le="+Inf",{key.split("{")[1]}}} {len(values)}')
            lines.append(f'{metric}_sum{"{" + key.split("{")[1]} {sum(values):.2f}')
            lines.append(f'{metric}_count{"{" + key.split("{")[1]} {len(values)}')
            lines.append("")
        return "\n".join(lines)


def main():
    builder = SpanBuilder()
    registry = MetricsRegistry()

    with builder.span("gen_ai.chat", {"gen_ai.system": "anthropic", "gen_ai.request.model": "claude-3"}):
        for tool in ["read_file", "write_file", "run_tests"]:
            t0 = time.time_ns()
            with builder.span(f"gen_ai.tool.{tool}", {"gen_ai.tool.name": tool}):
                time.sleep(0.001)
            registry.inc_counter("tools_called_total", {"tool": tool})
            registry.observe("tool_latency_ms", {"tool": tool}, (time.time_ns() - t0) / 1e6)

    print(f"Span 数: {len(builder.exporter.spans)}")
    for s in builder.exporter.spans[:3]:
        print(f"  {s['name']}: {s['duration_ms']:.2f}ms status={s['status']}")
    print(f"\nPrometheus 指标:\n{registry.exposition()[:500]}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 格式 | 特点 |
|:----|:-----|:-----|
| OTel Python SDK | OTLP gRPC | 完整生产方案 |
| Jaeger | OTLP | 分布式追踪 |
| Prometheus | 文本格式 | 拉取式指标 |
| 本课 JSONL | JSONL | 离线可审计 |

---

## 5. 工程最佳实践

- JSONL 导出器是离线等效物——生产中替换为 OTLP gRPC
- span ID 和 trace ID 用 W3C 标准十六进制格式
- **中文场景建议**：GenAI 属性约定不翻译，`gen_ai.system` 等字段名保持英文

---

## 6. 常见错误

- **span 未关闭**：上下文管理器必须在 finally 中关闭 span
- **指标标签顺序不一致**：排序以确保相同标签组合产生相同键
- **直方图桶边界遗漏**：OTel 默认延迟桶为 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000

---

## 7. 面试考点

**Q1：OTel GenAI 语义约定的作用是什么？**（难度：⭐⭐）

**参考答案：** 它标准化了 LLM 应用的 span 属性名称（`gen_ai.system`、`gen_ai.request.model` 等），使不同框架生成的追踪可以被同一个 OTel 后端消费。没有约定，每个框架都有自己的字段名，工具链无法复用。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| Span | 一次操作的追踪记录 |
| Trace | 完整智能体调用的 span 树 |
| GenAI 约定 | LLM 应用的标准属性键集 |
| Prometheus 格式 | 拉取式指标的文本格式 |

---

## 📚 小结

可观测性是智能体从玩具到生产的分水岭。你实现了 OTel span、JSONL 导出器和 Prometheus 指标。下一节将所有组件组合为端到端编码智能体。

---

## ✏️ 练习

1. 【实现】添加 `trace_id` 在根 span 和子 span 之间传播
2. 【实验】用 10 次工具调用验证计数器和直方图的正确性

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| OTel 可观测性 | `code/main.py` |

---

## 📖 参考资料

1. [官方文档] OpenTelemetry GenAI 语义约定. https://opentelemetry.io/docs/specs/semconv/gen-ai/
2. [官方文档] Prometheus 文本格式. https://prometheus.io/docs/instrumenting/exposition_formats/
