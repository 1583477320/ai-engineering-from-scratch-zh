# LangGraph 与状态机

> LLM 是函数，不是状态机。但大多数 AI 应用需要状态——多轮对话、工具调用链、人工审批。LangGraph 将 LLM 应用建模为状态图——状态、节点、边、条件路由。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 11 · 01-09 | **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解 LangGraph 的核心概念——StateGraph、节点、边、条件路由
- [ ] 实现一个多轮对话状态机——带工具调用和人工审批
- [ ] 理解 LangGraph 的检查点和持久化机制
- [ ] 对比 LangGraph 和 LangChain Agent 的区别

---

## 1. 问题

你有一个多步骤的客服 AI：(1) 理解问题 → (2) 查知识库 → (3) 尝试回答 → (4) 需要时转人工。用 LangChain Agent 实现——但 Agent 是"一问一答"的黑盒。你需要一个有状态的、可调试的、可中断的流水线。

LangGraph 将 LLM 应用建模为**有向图**——每个节点是一个 LLM 调用或工具，每条边是有条件的转移。

---

## 2. 概念

### 2.1 LangGraph 核心组件

| 组件 | 说明 | 类比 |
|------|------|------|
| **StateGraph** | 图的容器 | 状态机图 |
| **节点** | 执行单元（LLM 调用/工具） | 任务处理器 |
| **边** | 节点间连接（有条件/无条件） | 流程箭头 |
| **State** | 在节点间传递的数据 | 消息总线 |
| **检查点** | 状态的持久化快照 | 断点续传 |

### 2.2 基本状态机

```
[开始] → [理解问题] → [查知识库] → [生成回答] → [结束]
                                    ↓ (需要人工)
                              [转人工审批] → [结束]
```

### 2.3 条件路由

```python
def should_transfer(state):
    """决定是否转人工。"""
    if state.get("confidence", 0) < 0.5:
        return "transfer"
    return "respond"
```

---

## 3. 从零实现

### Step 1：简单状态图

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: list
    confidence: float

def understand_query(state):
    """节点：理解用户问题。"""
    return {"messages": state["messages"] + ["理解了你的问题"]}

def retrieve_knowledge(state):
    """节点：检索知识库。"""
    return {"messages": state["messages"] + ["检索到相关文档"]}

def generate_response(state):
    """节点：生成回答。"""
    return {"messages": state["messages"] + ["基于文档的回答"], "confidence": 0.8}

def transfer_to_human(state):
    """节点：转人工。"""
    return {"messages": state["messages"] + ["正在转接人工客服"]}

def route_after_retrieve(state):
    """条件边：决定去生成回答还是转人工。"""
    if state.get("confidence", 0) < 0.5:
        return "transfer"
    return "respond"

# 构建图
graph = StateGraph(AgentState)
graph.add_node("understand", understand_query)
graph.add_node("retrieve", retrieve_knowledge)
graph.add_node("respond", generate_response)
graph.add_node("transfer", transfer_to_human)

graph.add_edge("understand", "retrieve")
graph.add_conditional_edges("retrieve", route_after_retrieve, {"respond": "respond", "transfer": "transfer"})
graph.add_edge("respond", END)
graph.add_edge("transfer", END)

app = graph.compile()
```

### Step 2：运行状态机

```python
result = app.invoke({"messages": ["退货政策是什么？"], "confidence": 0.8})
print(f"最终消息: {result['messages']}")
```

---

## 4. 工具

### 4.1 LangGraph

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 持久化
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# 带线程的调用
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(input_state, config)
```

### 4.2 LangGraph vs LangChain Agent

| 方面 | LangChain Agent | LangGraph |
|------|----------------|-----------|
| 范式 | 黑盒循环 | 有向图 |
| 可调试性 | 困难 | 每个节点可调试 |
| 检查点 | 手动实现 | 内置 |
| 人工审批 | 需要额外工作 | 原生支持 |

---

## 6. 工程最佳实践

### 6.1 状态设计原则

- **最小化状态**：只保留必要信息，避免状态膨胀
- **不可变性**：节点不修改状态，返回新状态
- **类型安全**：用 TypedDict 定义状态结构

### 6.2 踩坑经验

- **状态无限增长**：长对话消息列表不断累积 → 定期截断或摘要
- **条件边遗漏**：某些状态没有默认出口 → 添加 fallback 路由
- **并行节点竞争**：同时修改同一状态字段 → 用 Annotated 类型解决

---

## 7. 常见错误

### 错误 1：状态无限增长

**现象：** 运行时间越长，状态对象越大——内存和 token 消耗持续上升。

**原因：** 消息列表不断追加新的消息——没有上限或清理机制。

**修复：** 在状态中添加 `max_messages` 限制——超过时保留最近 N 条 + 历史摘要。

### 错误 2：条件边遗漏默认出口

**现象：** 某些状态没有条件边也没有无条件边——图执行卡死。

**原因：** 在设计状态图时遗漏了某些转移路径。

**修复：** 为每个有分支的节点添加 fallback 路由——如果条件不满足就走默认路径。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 状态机实现 | `code/main.py` | 客服状态机：问题分类→检索→回答/转人工 |

---

## 📚 小结

### Q1：LangGraph 和 LangChain Agent 有什么本质区别？（难度：⭐⭐）

**参考答案：**
LangChain Agent 是"一问一答"的黑盒——它决定下一步做什么，但你看不到内部决策过程。LangGraph 将流程建模为显式状态图——每个节点是一个 LLM 调用或工具，每条边是条件转移。区别：(1) LangGraph 可调试——每个节点的输入输出清晰可见；(2) LangGraph 支持检查点——可以在任何节点暂停/恢复；(3) LangGraph 支持人工审批——可以在特定节点插入人工检查。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| LangGraph | "LLM 状态机框架" | 将 LLM 应用建模为有向状态图的框架 |
| StateGraph | "图容器" | LangGraph 的核心图定义类 |
| 节点 (Node) | "处理步骤" | 图中的一个执行单元——LLM 调用或工具 |
| 条件边 | "条件跳转" | 根据状态决定下一步走哪条边 |
| 检查点 (Checkpoint) | "断点续传" | 在图中保存状态快照——支持暂停和恢复 |

---

---

## 8. 面试考点

### Q1：LangGraph 的检查点机制是如何工作的？（难度：⭐⭐）

**参考答案：**
检查点（Checkpoint）在每个节点执行后保存状态快照——包括消息列表、对话轮次、当前步骤。当应用崩溃或需要暂停时，可以从最近的检查点恢复——重新加载状态并从断点继续执行。这使得人工审批成为可能：在某个节点暂停，让用户确认或修改，然后继续。

### Q2：LangGraph 的条件边和普通边有什么区别？（难度：⭐⭐）

**参考答案：**
普通边是无条件转移——从节点 A 直接到节点 B。条件边根据当前状态决定下一步走哪条边——需要一个路由函数，输入状态，输出下一个节点的名称。例如：在"检索"节点后，根据置信度决定是"回答"还是"转人工"。

---

## 📚 小结

---

## ✏️ 练习

1. **【实现】** 用 LangGraph 构建一个客服状态机——问题分类→知识库检索→回答/转人工
2. **【实验】** 添加检查点——在某个节点暂停，然后从同一位置恢复

---

## 📖 参考资料

1. [文档] LangGraph: https://langchain-ai.github.io/langgraph/
2. [GitHub] LangGraph: https://github.com/langchain-ai/langgraph
3. [教程] LangGraph Tutorial: https://langchain-ai.github.io/langgraph/tutorials/
