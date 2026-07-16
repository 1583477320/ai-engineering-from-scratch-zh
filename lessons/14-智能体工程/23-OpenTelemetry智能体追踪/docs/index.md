# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了智能体遥测的标准模式。Span 名称、属性和内容捕获规则在 Datadog、Grafana、Jaeger 和 Honeycomb 等供应商间趋同，使智能体追踪在不同平台中具有相同含义。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 14 · 13（LangGraph）、24（可观测性平台）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 命名 GenAI span 类别——模型/客户端、智能体、工具
- [ ] 实现一个简单的 span 发射器
- [ ] 理解 span 层次结构——智能体→LLM→工具的嵌套关系
- [ ] 设计生产级可观测性管道

---

## 1. 问题

一个智能体调用五个工具、三个 MCP Server 和两个子智能体。出了问题——但你不知道是哪个环节慢、哪个工具失败、哪个子智能体卡住了。没有追踪，调试就是猜测。

OpenTelemetry GenAI 语义约定是 2026 年的追踪标准——一个 trace 跨越所有组件。

---

## 2. 概念

### 2.1 GenAI Span 类别

| 类别 | 必需属性 | 说明 |
|------|---------|------|
| **模型/客户端** | `gen_ai.system`, `gen_ai.request.model` | 模型信息 |
| **智能体** | `agent.id`, `agent.session_id` | 智能体标识 |
| **工具** | `tool.name`, `tool.arguments` | 工具调用信息 |

### 2.2 Span 层次结构

```
[Agent Span: 用户请求]
  ├── [LLM Span: 推理]
  ├── [Tool Span: get_weather]
  ├── [LLM Span: 决策]
  └── [MCP Span: 服务调用]
```

---

## 3. 从零实现

### Step 1：简化版 GenAI Span 发射器

```python
import time

class GenAISpanEmitter:
    def __init__(self):
        self.spans = []

    def start_agent_span(self, agent_id, query):
        span = self._start(f"agent.{agent_id}", {"agent.id": agent_id, "input": query[:50]})
        return span

    def start_llm_span(self, model, input_tokens, output_tokens):
        span = self._start("llm.generate", {"gen_ai.system": "openai", "gen_ai.model": model})
        span["tokens"] = {"input": input_tokens, "output": output_tokens}
        return span

    def start_tool_span(self, tool_name, arguments):
        return self._start(f"tool.{tool_name}", {"tool.name": tool_name})

    def _start(self, name, attrs):
        span = {"name": name, "start": time.time(), "attrs": attrs}
        return span

    def end(self, span):
        span["duration_ms"] = (time.time() - span["start"]) * 1000
        self.spans.append(span)
```

---

## 4. 工具

### 4.1 OTel GenAI 库

| 库 | 特点 |
|------|------|
| OpenLLMetry | 开源 LLM 专用 |
| Langfuse | 开源可观测性 |
| AgentOps | 智能体专用 |

---

## 7. 面试考点

### Q1：OpenTelemetry GenAI 的核心价值是什么？（难度：⭐⭐）

**参考答案：**
跨平台标准化——同样的 span 格式在 Datadog、Grafana、Jaeger、Honeycomb 中含义相同。这意味着你可以换可观测性平台而不重写追踪代码。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| GenAI Semantic Conventions | "GenAI 追踪标准" | OpenTelemetry GenAI SIG 定义的 span 名称和属性规范 |
| Span | "追踪片段" | 操作的时间区间——包含名称、属性、状态 |

---

## 📚 小结

OpenTelemetry GenAI 是 2026 年智能体追踪的标准。三类 span：模型/客户端、智能体、工具。跨供应商标准化使追踪可移植。

---

## ✏️ 练习

1. **【实现】** 构建一个 GenAI span 发射器——追踪 LLM 调用和工具执行
2. **【设计】** 为一个多智能体系统设计追踪方案——需要哪些 span？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| GenAI 追踪 | `code/main.py` | Span 发射器 + 层次追踪 |

---

## 📖 参考资料

1. [文档] OpenTelemetry GenAI SIG
2. [文档] GenAI Semantic Conventions
3. [文档] OpenLLMetry
