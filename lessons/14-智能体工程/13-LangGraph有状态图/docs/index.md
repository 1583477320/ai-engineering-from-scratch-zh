# LangGraph：有状态图与持久化执行

> LangGraph 是 2026 年底层有状态编排的参考框架。智能体是一个状态机；节点是函数；边是转移；状态是不可变的——每步都检查点。从任何故障点精确恢复。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、12（工作流模式）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述 LangGraph 的核心模型：不可变状态、函数节点、条件边、步骤后检查点
- [ ] 实现 LangGraph 的状态图——定义节点、边和状态
- [ ] 理解 LangGraph 的持久化执行——从失败点恢复
- [ ] 对比 LangGraph 和 LangChain Agent 的优劣

---

## 1. 问题

LangChain Agent 是"黑盒"——你不知道它在每一步做什么。LangGraph 将执行过程建模为**有向图**——每个节点是一个函数，每条边是条件转移，状态是不可变的——每步都保存快照。

**关键优势：** 可调试、可恢复、可中断、可审计。

---

## 2. 概念

### 2.1 LangGraph 核心模型

| 组件 | 类比 | 说明 |
|------|------|------|
| **State** | 状态机状态 | 不可变数据——在节点间传递 |
| **节点** | 状态机动作 | 函数——处理状态并返回新状态 |
| **边** | 状态转移 | 条件或无条件——决定下一步 |
| **检查点** | 断点 | 每步后保存状态快照 |

### 2.2 LangGraph vs LangChain Agent

| 方面 | LangChain Agent | LangGraph |
|------|----------------|-----------|
| 可调试性 | 困难（黑盒） | 每个节点可调试 |
| 持久化 | 手动实现 | 内置检查点 |
| 恢复能力 | 不支持 | 从失败点恢复 |
| 并行 | 有限 | 原生支持 |
| 复杂度 | 简单 | 中等 |

### 2.3 状态设计原则

```python
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: list
    current_step: str
    confidence: float
    results: dict
```

- **不可变性**：节点不修改状态——返回新状态
- **最小化**：只保留必要信息
- **类型安全**：用 TypedDict 定义结构

---

## 3. 从零实现

### Step 1：简化版 LangGraph

```python
class StateGraph:
    """简化版状态图。"""
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.conditional_edges = {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, from_node, to_node):
        self.edges.append((from_node, to_node))

    def add_conditional_edge(self, from_node, condition_fn, routes):
        self.conditional_edges[from_node] = (condition_fn, routes)

    def compile(self):
        return StateGraphExecutor(self.nodes, self.edges, self.conditional_edges)


class StateGraphExecutor:
    """状态图执行器。"""
    def __init__(self, nodes, edges, conditional_edges):
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges

    def invoke(self, state, max_steps=10):
        current = list(self.nodes.keys())[0]
        for _ in range(max_steps):
            if current not in self.nodes:
                break
            result = self.nodes[current](state)
            state.update(result)
            if current in self.conditional_edges:
                cond_fn, routes = self.conditional_edges[current]
                current = routes[cond_fn(state)]
            else:
                next_nodes = [to for fr, to in self.edges if fr == current]
                current = next_nodes[0] if next_nodes else None
                if current is None:
                    break
        return state


def demo():
    """演示 LangGraph 风格的状态图。"""
    graph = StateGraph()

    def understand(state):
        return {"confidence": 0.9, "step": "理解完成"}

    def retrieve(state):
        return {"confidence": 0.95, "step": "检索完成", "retrieved": True}

    def respond(state):
        return {"step": "已回答", "final": True}

    graph.add_node("understand", understand)
    graph.add_node("retrieve", retrieve)
    graph.add_node("respond", respond)

    graph.add_edge("understand", "retrieve")
    graph.add_conditional_edge("retrieve",
        lambda s: "respond" if s.get("confidence", 0) > 0.8 else "fail",
        {"respond": "respond", "fail": None}
    )

    executor = graph.compile()
    result = executor.invoke({"messages": ["你好"]})
    print(f"状态: {result}")


if __name__ == "__main__":
    print("LangGraph 有状态图演示\n")
    demo()

---

## 4. 工具

### 4.1 LangGraph

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
graph.add_node("analyze", analyze_fn)
graph.add_node("retrieve", retrieve_fn)
graph.add_edge("analyze", "retrieve")
app = graph.compile()
```

### 4.2 框架对比

| 框架 | 特点 | 适用 |
|------|------|------|
| LangGraph | 有状态图、检查点 | 复杂多步流程 |
| LangChain Agent | 简单黑盒 | 快速原型 |
| AutoGen | 异步消息 | 多智能体协作 |

---

## 5. 工程最佳实践

### 5.1 状态设计

- **最小化**：只保留必要信息
- **不可变**：节点返回新状态
- **类型安全**：用 TypedDict 定义

### 5.2 检查点策略

- **每步检查点**：支持完整恢复
- **增量检查点**：只保存变化的状态
- **持久化**：使用 SQLite/PostgreSQL 存储

---

## 6. 常见错误

### 错误 1：节点修改状态

**现象：** 状态意外改变——恢复时数据错误。

**修复：** 节点不修改原状态——返回新状态副本。

### 错误 2：条件边缺少默认路由

**现象：** 图执行卡死——某些状态没有出口。

**修复：** 每个条件边都应有默认出口。

---

## 7. 面试考点

### Q1：LangGraph 的不可变状态为什么重要？（难度：⭐⭐）

**参考答案：**
不可变状态确保：(1) 每个节点的输入是确定的——不受其他节点影响；(2) 检查点保存的是干净的状态——恢复时不会带入错误；(3) 并发安全——多个节点可以并行执行而不冲突。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| StateGraph | "状态图容器" | LangGraph 的核心——定义节点、边和状态 |
| 检查点 | "断点" | 每步后保存状态快照——支持恢复 |
| 不可变状态 | "不改状态" | 节点返回新状态——不修改原状态 |

---

## 📚 小结

LangGraph 用状态图编排 LLM 应用——不可变状态、函数节点、条件边、步骤检查点。支持从任意故障点恢复。比 LangChain Agent 更可调试、更可靠。

---

## ✏️ 练习

1. **【实现】** 用 LangGraph 构建一个客服状态机——问题分类→检索→回答/转人工
2. **【实验】** 添加检查点——模拟故障后恢复

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 状态图 | `code/main.py` | 节点+边+条件+检查点 |

---

## 📖 参考资料

1. [GitHub] LangGraph: https://github.com/langchain-ai/langgraph
2. [文档] LangGraph 状态图: https://langchain-ai.github.io/langgraph/
3. [教程] LangGraph 教程

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
