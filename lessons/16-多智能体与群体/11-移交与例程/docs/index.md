# 移交与例程——无状态编排

> OpenAI 的 Swarm（2024 年 10 月）将多智能体编排精炼为两个原语：**例程**（指令+工具作为系统提示词）和**移交**（返回另一个 Agent 的工具）。没有状态机、没有分支 DSL——LLM 通过调用正确的移交工具来路由。OpenAI Agents SDK（2025 年 3 月）是生产后继。Swarm 本身仍是最干净的概念参考——其整个源代码只需几百行。这个模式是病毒式传播的，因为 API 表面大致是"智能体 = 提示词 + 工具；移交 = 返回智能体的函数"。局限：无状态，所以内存是调用者的问题。

**类型：** 概念课 + 实现课
**语言：** Python
**前置知识：** 阶段 16 · 04（原语模型）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 Swarm 的两个原语——例程（提示词+工具）和移交（返回 Agent 的工具）
- [ ] 实现一个无状态移交驱动的编排——智能体通过调用移交工具来切换活跃智能体
- [ ] 理解 Swarm 的无状态权衡——内存是调用者的问题
- [ ] 对比 Swarm 移交和 GroupChat 选择器——谁决定下一个发言

---

## 1. 问题

每个多智能体框架都让你学习它的 DSL：LangGraph 的节点和边、CrewAI 的团队和任务、AutoGen 的 GroupChat 和管理者。DSL 是真实的抽象，但它们让事情感觉比需要的更重。

Swarm 推向相反方向：使用模型已经有的工具调用能力。移交变成工具调用。编排者是当前持有对话的智能体。状态机隐含在智能体的系统提示词中。

---

## 2. 概念

### 2.1 两个原语

**例程 (Routine)：** 定义智能体角色和可用工具的系统提示词。

**移交 (Handoff)：** 智能体可以调用的工具，它返回一个新的 Agent 对象。Swarm 运行时检测 Agent 返回值并在下一轮切换活跃智能体。

```python
def transfer_to_refunds():
    return refund_agent  # Swarm 看到 Agent 返回 → 切换活跃智能体

triage_agent = Agent(
    name="triage",
    instructions="路由用户到正确的专家。",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

### 2.2 为什么病毒式传播

- **小 API**——两个概念要学
- **使用模型已经做的**——工具调用跨提供者已经是生产级的
- **无状态机负担**——你不需要描述图；智能体的提示词描述它们移交给谁

### 2.3 无状态权衡

Swarm 明确地在运行之间无状态。框架在运行期间保持消息历史，但不持久化任何东西。内存、连续性、长期任务——都是调用者的问题。

生产中（OpenAI Agents SDK，2025 年 3 月）这是主要改变点：SDK 在移交原语之上添加了内置会话管理、护栏和追踪。

### 2.4 Swarm 何时合适/失败

**合适：** 分诊模式、基于技能的移交、短对话

**失败：** 长会话+共享内存（移交重置对话状态）、并行执行（移交是逐一的）、审计和重放（无状态运行难以精确重放）

### 2.5 Swarm vs GroupChat

两者都使用 LLM 驱动的路由，但在**谁决定下一个发言**上不同：

- GroupChat：选择器（函数或 LLM）从外部选择下一个发言者
- Swarm：当前智能体通过调用移交工具选择其继任者

Swarm 是"智能体决定下一步"；GroupChat 是"管理者决定下一步"。

---

## 3. 从零实现

### 第 1 步：定义智能体和移交

```python
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

@dataclass
class Agent:
    name: str
    instructions: str
    functions: list[Callable] = field(default_factory=list)

def triage_agent_factory():
    def transfer_to_refunds(): return refund_agent
    def transfer_to_sales(): return sales_agent
    def transfer_to_support(): return support_agent
    return Agent(
        name="triage",
        instructions="路由用户到：退款、销售或支持。",
        functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
    )
```

### 第 2 步：实现 Swarm 运行循环

```python
def run_swarm(start_agent, user_messages):
    """运行 Swarm：检测 Agent 返回并切换活跃智能体。"""
    history = []
    active = start_agent
    for user in user_messages:
        history.append(Msg(role="user", content=user))
        out = scripted_router(active, user)
        if isinstance(out, Agent):
            history.append(Msg(role="assistant", content=f"(移交到 {out.name})", sender=active.name))
            active = out
            out = scripted_router(active, user)
        history.append(Msg(role="assistant", content=str(out), sender=active.name))
    return history
```

### 第 3 步：运行演示

```python
def main():
    scenarios = [
        ("退款流程", ["I need a refund on order 77"]),
        ("销售流程", ["I want to buy the enterprise plan. what's the price?"]),
        ("支持流程", ["my dashboard is broken"]),
        ("模糊", ["hello"]),
    ]
    for label, msgs in scenarios:
        print(f"\n=== {label} ===")
        history = run_swarm(triage_agent, msgs)
        render(history)
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 OpenAI Agents SDK 的改进

| 特性 | 说明 |
|------|------|
| 会话状态 | 跨运行持久化 |
| 护栏 | 输入/输出验证钩子 |
| 追踪 | 每个工具调用和移交都记录 |
| 移交过滤器 | 控制移交时传输什么上下文 |

### 4.2 移交设计检查清单

| 检查项 | 说明 |
|--------|------|
| 移交日志 | 每次移交写入追踪事件 |
| 上下文传输规则 | 移交时传输什么：完整历史、最后 N 条消息、或摘要 |
| 移交护栏 | 移交给不同工具权限的专家必须认证 |
| 循环检测 | 两个智能体来回移交——用简单环检测 |
| 回退智能体 | 如果移交目标不存在，回退到安全默认 |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 移交是函数调用 | 不是消息传递——是控制转移 |
| 无状态 = 调用者的问题 | 内存、连续性、长期任务需要调用者管理 |
| 最小 API | Swarm 证明两个概念足够 |
| 移交过滤器 | 控制移交时传输什么上下文 |

---

## 6. 常见错误

### 错误 1：移交重置对话状态

**现象：** 移交给退款智能体后，它不知道之前的对话历史。

**原因：** Swarm 是无状态的。移交将活跃智能体切换到新智能体的提示词+历史。

**修复：** 生产中使用 OpenAI Agents SDK（添加会话管理）或调用者管理内存。

### 错误 2：移交循环

**现象：** 两个智能体来回移交，永远不会终止。

**修复：** 环检测——如果相同的两个智能体连续移交 3 次，强制退出。

### 错误 3：移交过滤器缺失

**现象：** 移交时所有上下文都传输，导致信息泄露。

**修复：** 使用 SDK 的移交过滤器——只传输必要的上下文。

---

## 7. 面试考点

### Q1：Swarm 的两个原语是什么？（难度：⭐）

**参考答案：**
**例程**：定义智能体角色和可用工具的系统提示词。
**移交**：智能体可以调用的工具，返回新的 Agent 对象。运行时检测 Agent 返回值并切换活跃智能体。

### Q2：Swarm 的无状态权衡意味着什么？（难度：⭐⭐）

**参考答案：**
Swarm 在运行之间不持久化任何东西。内存、连续性、长期任务都是调用者的问题。

OpenAI Agents SDK 添加了会话状态（持久化线程）来解决这个问题。Swarm 本身保持无状态——这是它的概念清晰性的代价。

### Q3：Swarm 移交和 GroupChat 选择器的区别是什么？（难度：⭐⭐⭐）

**参考答案：**
**Swarm**：当前智能体通过调用移交工具选择继任者。决策在活跃智能体的工具调用中。

**GroupChat**：选择器（函数或 LLM）从外部选择下一个发言者。决策在 GroupChatManager 中。

Swarm 是"智能体决定下一步"；GroupChat 是"管理者决定下一步"。Swarm 更简单但更难审计。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 例程 | "智能体提示词" | 系统提示词 + 工具列表。定义角色和可用移交 |
| 移交 | "转移到另一个智能体" | 活跃智能体可以调用的工具，返回新 Agent。运行时切换活跃智能体 |
| 无状态 | "运行间无内存" | Swarm 不持久化任何东西；内存是调用者的问题 |
| 活跃智能体 | "谁在发言" | 当前持有对话的智能体。移交改变这个 |
| 上下文传输 | "移交时传输什么" | 入站智能体看到的完整历史策略 |
| 移交循环 | "智能体乒乓" | 两个智能体来回移交的失败模式 |

---

## 📚 小结

Swarm 将多智能体编排精炼为两个原语：例程（提示词+工具）和移交（返回 Agent 的工具）。最小 API、使用模型已有的工具调用能力。无状态——内存是调用者的问题。Swarm 适合分诊、技能移交、短对话。不适合长会话、并行执行、审计。Swarm vs GroupChat：谁决定下一步？Swarm 是智能体，GroupChat 是管理者。

下一课：A2A 协议——智能体间的通用线路协议。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`，分流到退款智能体。确认第二轮的活跃智能体是退款。

2. **【实现】** 添加循环检测规则：如果相同的两个智能体连续移交 3 次，强制退出。设计回退。

3. **【阅读】** 阅读 OpenAI Agents SDK 文档中的移交过滤器。实现"移交时摘要"版本：出站智能体在入站智能体接管前压缩上下文为要点摘要。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| Swarm 演示 | `code/main.py` | Agent + 移交 + 无状态循环 |
| 技能提示词 | `outputs/skill-handoff-designer.md` | 为任务设计移交拓扑 |

---

## 📖 参考资料

1. [博客] OpenAI Cookbook. "Orchestrating Agents: Routines and Handoffs". https://developers.openai.com/cookbook/examples/orchestrating_agents
2. [GitHub] OpenAI Swarm. https://github.com/openai/swarm — 原始实现，概念参考
3. [文档] OpenAI Agents SDK. https://openai.github.io/openai-agents-python/ — 生产后继

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
