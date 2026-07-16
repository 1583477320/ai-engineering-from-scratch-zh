# OpenTelemetry GenAI——端到端追踪工具调用

> 一个智能体调用五个工具、三个 MCP Server 和两个子智能体。你需要一个跨所有这些的 trace。OpenTelemetry GenAI 语义约定（v1.37 中的稳定属性）是 2026 年的标准，原生支持 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps。本课命名必需属性，走一遍 span 层次结构（智能体→LLM→工具），并提供一个可插入任何 OTel exporter 的 stdlib span 发射器。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）、08（MCP Client）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 命名 LLM span 和工具执行 span 的必需 OTel GenAI 属性
- [ ] 实现 span 发射器——记录工具调用的延迟、输入/输出大小、错误
- [ ] 理解 span 层次结构——智能体→LLM→工具的嵌套关系
- [ ] 设计生产级可观测性管道

---

## 1. 问题

一个智能体调用五个工具、三个 MCP Server 和两个子智能体。出了问题——但你不知道是哪个环节慢、哪个工具失败、哪个子智能体卡住了。没有追踪，调试就是猜测。

OpenTelemetry GenAI 提供了标准化的追踪方案——一个 trace 跨越所有组件。

---

## 2. 概念

### 2.1 Span 层次结构

```
[Agent Span]
  ├── [LLM Span: 推理]
  ├── [Tool Span: get_weather]
  ├── [LLM Span: 决策]
  ├── [MCP Server A: 查询]
  ├── [Tool Span: search_docs]
  └── [LLM Span: 生成回答]
```

### 2.2 必需的 OTel GenAI 属性

| Span 类型 | 必需属性 | 说明 |
|-----------|---------|------|
| **LLM Span** | `gen_ai.system`, `gen_ai.request.model` | 模型信息 |
| **LLM Span** | `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | Token 使用量 |
| **Tool Span** | `mcp.tool.name`, `mcp.tool.arguments` | 工具信息 |
| **Tool Span** | `mcp.tool.duration_ms` | 工具执行时间 |

### 2.3 2026 年的可观测性平台

| 平台 | 特点 |
|------|------|
| Langfuse | 开源，LLM 专用 |
| Datadog GenAI | 企业级，集成广泛 |
| Arize Phoenix | 开源，模型评估 |
| OpenLLMetry | 开源，LLM 专用 |

---

## 3. 从零实现

### Step 1：简化版 Span 发射器

```python
import time

class SpanEmitter:
    """简化版 OTel Span 发射器。"""
    def __init__(self):
        self.spans = []

    def start_span(self, name, attributes=None):
        span = {
            "name": name,
            "start_time": time.time(),
            "end_time": None,
            "attributes": attributes or {},
            "status": "ok",
        }
        return span

    def end_span(self, span, status="ok", error=None):
        span["end_time"] = time.time()
        span["duration_ms"] = (span["end_time"] - span["start_time"]) * 1000
        span["status"] = status
        if error:
            span["error"] = str(error)
        self.spans.append(span)
        return span

    def emit_llm_span(self, model, input_tokens, output_tokens, duration_ms):
        """发射 LLM span。"""
        span = self.start_span("llm.generate", {
            "gen_ai.system": "openai",
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "gen_ai.response.duration_ms": duration_ms,
        })
        return self.end_span(span)

    def emit_tool_span(self, tool_name, duration_ms, success=True):
        """发射工具执行 span。"""
        return self.end_span(
            self.start_span(f"tool.{tool_name}", {"tool.name": tool_name}),
            status="ok" if success else "error",
        )


if __name__ == "__main__":
    print("OpenTelemetry GenAI 追踪演示\n")
    emitter = SpanEmitter()

    # 模拟智能体调用
    llm_span = emitter.emit_llm_span("gpt-4o", 150, 50, 1200)
    print(f"LLM span: {llm_span['name']}, {llm_span['duration_ms']:.0f}ms")

    tool_span = emitter.emit_tool_span("get_weather", 200)
    print(f"Tool span: {tool_span['name']}, {tool_span['duration_ms']:.0f}ms")

    llm_span2 = emitter.emit_llm_span("gpt-4o", 200, 150, 2500)
    print(f"LLM span: {llm_span2['name']}, {llm_span2['duration_ms']:.0f}ms")

    total = sum(s["duration_ms"] for s in emitter.spans)
    print(f"\n总追踪: {len(emitter.spans)} 个 span, 总耗时 {total:.0f}ms")
