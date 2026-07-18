# 多智能体原语模型

> 2026 年每个发布的多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是四维设计空间中的一个点。四个原语，仅此而已：智能体、移交、共享状态、编排者。本课从零构建它们，在所有三种编排类型上运行玩具系统，然后将每个主要框架映射到相同轴上，让你用一段话读懂任何新发布。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 14（智能体工程）、阶段 16 · 01（为什么多智能体）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 识别多智能体的四个原语——智能体、移交、共享状态、编排者——并解释每个原语在设计空间中的角色
- [ ] 将每个主要框架（LangGraph、CrewAI、AutoGen、OpenAI Swarm）映射到四个原语上
- [ ] 实现三种编排者变体——静态图、移交驱动、LLM 选择——并对比它们的权衡
- [ ] 理解无状态洞察：除共享状态外，所有原语都是无状态的

---

## 1. 问题

每六个月就有一个新的多智能体框架发布。AutoGen 在 2023 年。CrewAI 在 2024 年。LangGraph 和 OpenAI Swarm 在 2024 年。Google ADK 在 2025 年 4 月。Microsoft Agent Framework RC 在 2026 年 2 月。每个新闻稿都声称是"正确的抽象"。

如果你试图逐个学习它们，你会精疲力竭。API 看起来不同。文档对"智能体"是什么意见不一。一个框架把共享内存称为"黑板"，另一个称为"消息池"，第三个称为"StateGraph"。

它不是。营销之下，四个原语是稳定的。学一次，一段话读懂每个新框架。

---

## 2. 概念

### 2.1 四个原语

```
┌─────────────────────────────────────────────────┐
│                编排者                            │
│          (决定谁下一个发言)                       │
│     ┌──────────┬──────────┬──────────┐          │
│     │ 智能体A  │ 智能体B  │ 智能体C  │          │
│     │(提示+工具)│(提示+工具)│(提示+工具)│          │
│     └────┬─────┘ └────┬─────┘ └────┬─────┘     │
│          │             │             │           │
│          └───── 移交 ──┴──── 移交 ───┘           │
│                    ↓                              │
│              共享状态                             │
│         (消息池/黑板/KV存储)                       │
└─────────────────────────────────────────────────┘
```

1. **智能体** — 系统提示词 + 工具列表。无状态；每次运行从系统提示词和当前消息历史开始
2. **移交** — 从一个智能体到另一个的结构化控制转移
3. **共享状态** — 多个智能体可读（有时可写）的任何数据结构
4. **编排者** — 决定谁下一个发言的组件

### 2.2 每个 2026 框架如何映射

| 框架 | 智能体 | 移交 | 共享状态 | 编排者 |
|------|--------|------|---------|--------|
| OpenAI Swarm | `Agent(instructions, tools)` | 工具返回 Agent | 调用者问题 | LLM 的下一个移交调用 |
| AutoGen v0.4 | `ConversableAgent` | GroupChat 上的说话者选择器 | 消息池 | 选择函数 |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential/Hierarchical` | Task 输出链 | 管理者 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` reducer | 图，确定性 |
| Microsoft Agent Framework | agent + 编排模式 | 模式特定 | thread / context | 模式特定 |

表面差异看起来很大。底下：相同的四个旋钮。

### 2.3 为什么这很重要

一旦你看到原语，框架对比就变成了一个简短的检查清单：

- 编排者信任 LLM 路由（Swarm）还是在代码中固定路由（LangGraph）？
- 共享状态是全历史（GroupChat）还是投影的（StateGraph reducer）？
- 智能体能修改彼此的提示词（CrewAI manager）还是只能移交（Swarm）？

这三个问题回答了 80% 的框架适配问题。

### 2.4 无状态洞察

除共享状态外，每个原语都是无状态的。智能体是（提示词，工具）的函数。移交是函数调用。编排者是调度器。**系统中唯一有状态的东西是共享状态。** 所有有趣的 bug 都在那里：记忆投毒、消息排序、版本控制、写入竞争。

### 2.5 原语解剖

**智能体：** `Agent = (system_prompt, tools, model, optional_name)`。无记忆、无状态。两个具有相同提示词和工具的智能体是可互换的。

**移交：** 三种实现——函数返回（Swarm 模式）、图边（LangGraph）、说话者选择（AutoGen GroupChat）。

**共享状态：** `SharedState = { messages: [], artifacts: {}, context: {} }`。两种拓扑：全池（每个智能体看到每条消息）和投影（角色限定视图）。

**编排者：** 四种风格——静态图（LangGraph）、LLM 选择（AutoGen）、移交驱动（Swarm）、队列驱动（群体架构）。

---

## 3. 从零实现

### 第 1 步：定义四个原语

```python
import threading
from dataclasses import dataclass, field

@dataclass
class SharedState:
    messages: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, msg):
        with self._lock:
            self.messages.append(msg)

    def snapshot(self):
        with self._lock:
            return list(self.messages)

@dataclass
class Agent:
    name: str
    system_prompt: str
    policy: callable  # (SharedState) -> Message

    def run(self, state):
        msg = self.policy(state)
        msg.setdefault("from", self.name)
        return msg
```

### 第 2 步：实现三种编排者

```python
class StaticOrchestrator:
    """固定顺序，LangGraph 式确定性。"""
    def __init__(self, order): self.order = order
    def run(self, team, state, max_steps=10):
        for name in self.order[:max_steps]:
            msg = team[name].run(state)
            state.append(msg)

class HandoffOrchestrator:
    """OpenAI Swarm 式：当前智能体返回移交目标。"""
    def __init__(self, start): self.start = start
    def run(self, team, state, max_steps=10):
        current = self.start
        for _ in range(max_steps):
            if current not in team: return
            msg = team[current].run(state)
            state.append(msg)
            nxt = msg.get("handoff", "done")
            if nxt == "done": return
            current = nxt

class LLMSelectorOrchestrator:
    """AutoGen GroupChat 式说话者选择。"""
    def __init__(self, start, selector):
        self.start = start
        self.selector = selector
    def run(self, team, state, max_steps=10):
        current = self.start
        for _ in range(max_steps):
            if current not in team: return
            msg = team[current].run(state)
            state.append(msg)
            current = self.selector(state, team)
```

### 第 3 步：运行三种对比

```python
def main():
    # 固定顺序（LangGraph 式）
    state_a = SharedState()
    StaticOrchestrator(["researcher", "writer", "reviewer"]).run(make_team(), state_a)
    render_pool("静态（LangGraph 式）", state_a)

    # 移交驱动（Swarm 式）
    state_b = SharedState()
    HandoffOrchestrator("researcher").run(make_team(), state_b)
    render_pool("移交驱动（Swarm 式）", state_b)

    # LLM 选择（AutoGen 式）
    state_c = SharedState()
    LLMSelectorOrchestrator("researcher", round_robin_selector).run(make_team(), state_c)
    render_pool("LLM 选择（AutoGen 式）", state_c)
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 框架原语对照

| 框架 | 智能体 | 移交 | 共享状态 | 编排者 |
|------|--------|------|---------|--------|
| OpenAI Swarm | `Agent(instructions, tools)` | 工具返回 Agent | 调用者问题 | LLM 的下一个移交调用 |
| AutoGen v0.4 | `ConversableAgent` | GroupChat 说话者选择器 | 消息池 | 选择函数 |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential/Hierarchical` | Task 输出链 | 管理者 LLM |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` reducer | 图，确定性 |

### 4.2 选择框架的三个问题

| 问题 | 选项 | 对应框架 |
|------|------|---------|
| 编排者信任 LLM 路由还是固定在代码中？ | LLM 路由 → Swarm；代码固定 → LangGraph | 智能协调 vs 确定性 |
| 共享状态是全历史还是投影？ | 全历史 → GroupChat；投影 → StateGraph | 简单性 vs 可扩展性 |
| 智能体能修改彼此提示词？ | 能 → CrewAI；不能 → Swarm | 灵活性 vs 安全性 |

---

## 5. 工程最佳实践

### 5.1 原语设计原则

| 原则 | 说明 |
|------|------|
| 唯一有状态的是共享状态 | 所有 bug 都在那里——记忆投毒、消息排序、写入竞争 |
| 无状态智能体可互换 | 相同提示词+工具的两个智能体是等价的 |
| 移交是函数调用 | 不是消息传递——是控制转移 |
| 编排者决定谁发言 | 四种风格：静态、LLM、移交、队列 |

---

## 6. 常见错误

### 错误 1：每个框架重新学习

**现象：** 学完 LangGraph 又学 CrewAI，API 完全不同。

**修复：** 映射四个原语。LangGraph 的图边就是移交；CrewAI 的 Process 就是编排者。底下是相同的四个旋钮。

### 错误 2：忽视共享状态的无状态性

**现象：** 认为智能体有状态。实际上只有共享状态有状态——智能体是函数，每次调用从头开始。

**修复：** 智能体 = （提示词 + 工具）。共享状态 = （消息池 + 工件 + 上下文）。所有 bug 在共享状态中。

### 错误 3：选择框架时只看 API 表面

**现象：** CrewAI 的 `Agent(role, goal, backstory)` 看起来比 LangGraph 的图节点更友好。

**修复：** 问三个问题：编排者如何路由？共享状态是全历史还是投影？智能体能修改彼此提示词吗？

---

## 7. 面试考点

### Q1：多智能体的四个原语是什么？（难度：⭐）

**参考答案：**
智能体（提示词+工具，无状态）、移交（从一个到另一个的结构化控制转移）、共享状态（多智能体可读的数据结构）、编排者（决定谁发言的组件）。

### Q2：四个原语中哪个有状态？（难度：⭐⭐）

**参考答案：**
只有共享状态有状态。智能体是函数（提示词+工具），移交是函数调用，编排者是调度器。所有有趣的 bug——记忆投毒、消息排序、版本控制——都在共享状态中。

### Q3：框架对比时应该问哪三个问题？（难度：⭐⭐⭐）

**参考答案：**
（1）编排者信任 LLM 路由（Swarm）还是固定在代码中（LangGraph）？
（2）共享状态是全历史（GroupChat）还是投影（StateGraph reducer）？
（3）智能体能修改彼此提示词（CrewAI）还是只能移交（Swarm）？

这三个问题回答了 80% 的框架适配。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 智能体 | "带工具的 LLM" | `(系统提示词, 工具, 模型)` 三元组。无状态 |
| 移交 | "控制转移" | 命名下一个智能体的结构化调用 |
| 共享状态 | "记忆" / "上下文" | 多智能体系统中唯一有状态的部分 |
| 编排者 | "协调者" | 决定谁下一个运行。静态图、LLM 选择器、移交驱动、队列驱动 |
| 投影状态 | "限定视图" | 角色特定的共享状态视图 |

---

## 📚 小结

每个 2026 年多智能体框架都是四个原语的不同参数化：智能体、移交、共享状态、编排者。学一次原语，一段话读懂每个新框架。唯一有状态的是共享状态——所有 bug 都在那里。三个框架对比问题覆盖 80% 的适配决策。

下一课：监督者-编排者模式——领智能体计划、委派、合成。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py` 三次，使用不同智能体策略。观察编排者选择如何改变哪些智能体运行。

2. **【实现】** 添加第四种编排者：队列驱动——智能体从共享状态轮询工作。什么死锁可能发生？

3. **【阅读】** 阅读 LangGraph 快速入门。将其重写为四个原语。哪些 LangGraph 抽象映射 1:1，哪些是便利包装器？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 四原语演示 | `code/main.py` | 三种编排者类型对比 |
| 技能提示词 | `outputs/skill-primitive-mapper.md` | 读取任何框架并返回原语映射 |

---

## 📖 参考资料

1. [博客] OpenAI Cookbook. "Orchestrating Agents: Routines and Handoffs". https://developers.openai.com/cookbook/examples/orchestrating_agents
2. [文档] AutoGen. https://microsoft.github.io/autogen/stable/
3. [文档] LangGraph. https://docs.langchain.com/oss/python/langgraph/workflows-agents
4. [文档] CrewAI. https://docs.crewai.com/en/introduction

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
