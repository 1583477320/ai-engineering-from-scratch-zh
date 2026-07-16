# OpenAI 智能体 SDK：交接、护栏、追踪

> OpenAI Agents SDK 是基于 Responses API 的轻量多智能体框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。Handoff 是名为 `transfer_to_<agent>` 的工具。Guardrail 在输入或输出时触发。Tracing 默认开启。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、06（工具使用）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 命名 OpenAI Agents SDK 的五个原语——Agent、Handoff、Guardrail、Session、Tracing
- [ ] 实现智能体之间的交接（Handoff）
- [ ] 设计护栏规则——在输入/输出时触发
- [ ] 理解 Session 管理和 Tracing 的作用

---

## 1. 问题

OpenAI 的 Responses API 是原始接口——但多智能体协作需要更高级的抽象：智能体之间的交接、输入/输出检查、状态持久化。OpenAI Agents SDK 在 Responses API 之上提供了这些能力。

---

## 2. 概念

### 2.1 五个原语

| 原语 | 说明 |
|------|------|
| **Agent** | 执行任务的智能体 |
| **Handoff** | 智能体间的交接——`transfer_to_<agent>` |
| **Guardrail** | 输入/输出检查 |
| **Session** | 会话状态管理 |
| **Tracing** | 执行追踪（默认开启） |

### 2.2 Handoff 机制

```python
# 工具命名约定：transfer_to_<agent_name>
tools = [
    {"name": "transfer_to_support", "description": "转接到支持团队"},
    {"name": "transfer_to_billing", "description": "转接到计费团队"},
]
```

### 2.3 Guardrail 类型

| 类型 | 触发时机 | 作用 |
|------|---------|------|
| 输入 Guardrail | 用户输入时 | 过滤恶意/不适当输入 |
| 输出 Guardrail | 模型输出时 | 检查输出是否安全 |

---

## 3. 从零实现

### Step 1：Agent 和 Handoff

```python
class SimpleAgent:
    """简化版 OpenAI Agent。"""
    def __init__(self, name, instructions):
        self.name = name
        self.instructions = instructions
        self.handoffs = {}

    def add_handoff(self, agent):
        self.handoffs[f"transfer_to_{agent.name}"] = agent

    def run(self, input_text):
        if any(h in input_text for h in self.handoffs):
            for h, agent in self.handoffs.items():
                if h in input_text:
                    return {"action": "handoff", "target": agent.name}
        return {"action": "respond", "content": f"[{self.name}] 处理: {input_text[:30]}"}
```

### Step 2：Guardrail

```python
def input_guardrail(input_text):
    """输入检查。"""
    dangerous_patterns = ["注入", "系统提示", "忽略规则"]
    for p in dangerous_patterns:
        if p in input_text:
            return False, f"检测到潜在危险输入: {p}"
    return True, "输入安全"


def output_guardrail(response):
    """输出检查。"""
    unsafe_patterns = ["密码", "密钥", "token"]
    for p in unsafe_patterns:
        if p in response:
            return False, f"输出包含敏感信息: {p}"
    return True, "输出安全"
```

---

## 4. 工具

### 4.1 OpenAI Agents SDK

```python
from agents import Agent, Handoff

support_agent = Agent(name="support", instructions="处理技术支持问题")
agent = Agent(
    name="router",
    instructions="根据用户意图路由到对应团队",
    handoffs=[support_agent],
)
```

### 4.2 工具对比

| SDK | 特点 |
|-----|------|
| OpenAI Agents | 轻量、原生 Responses API |
| Claude Agent | 内置工具、子智能体、会话存储 |
| LangGraph | 灵活状态图 |

---

## 5. 工程最佳实践

### 5.1 Handoff 设计

- 每个 Handoff 工具名以 `transfer_to_<agent>` 格式
- Handoff 描述要清晰——帮助 LLM 判断何时交接
- 每个 Agent 应该有明确的职责边界

### 5.2 踩坑经验

- **Handoff 循环**：两个 Agent 互相交接→无限循环。设置最大交接次数
- **Guardrail 误拦截**：正常请求被拦截→调整阈值

---

## 6. 常见错误

### 错误 1：Agent 职责重叠

**现象：** 多个 Agent 争夺相同任务——效率低下。

**修复：** 每个 Agent 负责一个明确的领域。

---

## 7. 面试考点

### Q1：OpenAI Agents SDK 的五个原语是什么？（难度：⭐⭐）

**参考答案：** Agent（智能体）、Handoff（交接）、Guardrail（护栏）、Session（会话）、Tracing（追踪）。Handoff 是特殊的工具——命名为 `transfer_to_<agent>`。Guardrail 在输入和输出时检查安全性。Tracing 默认开启，记录所有执行步骤。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Handoff | "交接" | 智能体间的任务转移——通过特殊工具名实现 |
| Guardrail | "护栏" | 输入/输出检查——防止不安全内容通过 |
| Session | "会话管理" | 智能体的会话状态——支持多轮对话 |
| Tracing | "执行追踪" | 记录每个步骤的输入输出——用于调试和监控 |

---

## 📚 小结

OpenAI Agents SDK 五个原语：Agent、Handoff、Guardrail、Session、Tracing。Handoff 通过特殊工具名实现智能体间交接。Guardrail 在输入/输出时检查。Tracing 默认开启。

---

## ✏️ 练习

1. **【实现】** 构建两个 Agent 并实现 Handoff——路由器→支持团队
2. **【设计】** 为一个客服系统设计 Guardrail 规则——输入过滤+输出检查

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| Agent + Handoff | `code/main.py` | 交接机制 + Guardrail |

---

## 📖 参考资料

1. [文档] OpenAI Agents SDK
2. [文档] Responses API

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
