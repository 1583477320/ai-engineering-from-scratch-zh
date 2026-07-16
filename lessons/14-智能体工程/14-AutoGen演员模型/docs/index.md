# AutoGen v0.4：演员模型与智能体框架

> AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕演员模型重新设计了智能体编排。异步消息交换、事件驱动智能体、故障隔离、自然并发。该框架现处于维护模式，Microsoft Agent Framework（2025 年 10 月公开预览）将成为继任者。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、12（工作流模式）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述演员模型：智能体作为演员，消息作为唯一 IPC，每个演员的故障隔离
- [ ] 对比 AutoGen 和 LangGraph 的架构差异
- [ ] 理解 AutoGen v0.4 的异步消息传递模式
- [ ] 实现一个简单的演员模型智能体

---

## 1. 问题

传统智能体编排（如 LangGraph）是同步的、集中式的——每一步都等待前一步完成。多智能体系统需要：异步通信、事件驱动、故障隔离。AutoGen v0.4 基于演员模型——每个智能体是独立演员，通过消息异步通信。

---

## 2. 概念

### 2.1 演员模型

```
演员 A → 消息 → 演员 B → 消息 → 演员 C
         ↕              ↕
    异步、无共享状态、故障隔离
```

| 概念 | 说明 |
|------|------|
| **演员** | 独立执行单元——有自己的状态和逻辑 |
| **消息** | 演员间的唯一通信方式 |
| **邮箱** | 消息队列——异步处理 |
| **故障隔离** | 一个演员失败不影响其他演员 |

### 2.2 AutoGen vs LangGraph

| 方面 | LangGraph | AutoGen |
|------|-----------|---------|
| 通信 | 同步函数调用 | 异步消息 |
| 状态 | 集中式状态 | 每个演员独立状态 |
| 故障 | 全局故障 | 演员级故障隔离 |
| 并发 | 有限 | 原生并发 |

---

## 3. 从零实现

### Step 1：演员模型

```python
import asyncio

class Actor:
    """简化版演员模型。"""
    def __init__(self, name, handler):
        self.name = name
        self.handler = handler
        self.mailbox = asyncio.Queue()

    async def send(self, message):
        await self.mailbox.put(message)

    async def run(self):
        while True:
            message = await self.mailbox.get()
            if message.get("type") == "stop":
                break
            response = await self.handler(message)
            if response:
                print(f"  [{self.name}]: {response}")


async def demo():
    coder = Actor("Coder", lambda msg: f"生成代码: {msg['task']}")
    reviewer = Actor("Reviewer", lambda msg: f"审查代码: {msg['task']}")

    await coder.send({"task": "实现排序函数"})
    await coder.send({"type": "stop"})
    await reviewer.send({"type": "stop"})
```

---

## 4. 工具

### 4.1 AutoGen

```python
# AutoGen v0.4 (archived - maintenance mode)
# 新项目推荐：Microsoft Agent Framework
```

---

## 5. 工程最佳实践

### 5.1 演员模型设计

- **故障隔离**：每个演员独立运行——失败不影响其他
- **异步通信**：消息队列解耦——演员可以自定节奏处理

---

## 6. 常见错误

### 错误 1：演员之间共享状态

**现象：** 演员模型退化为全局状态。

**修复：** 每个演员维持自己的独立状态——消息是唯一通信方式。

---

## 7. 面试考点

### Q1：演员模型和状态图的区别是什么？（难度：⭐⭐）

**参考答案：**
LangGraph 使用集中式状态图——所有节点共享一个状态对象。AutoGen 使用演员模型——每个演员有独立状态，通过异步消息通信。演员模型更灵活（异步、故障隔离），状态图更简单（可调试、可审计）。选择取决于需求的复杂度。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 演员模型 | "独立消息传递" | 每个智能体是独立演员，通过异步消息通信——故障隔离的自然并发 |
| 邮箱 (Mailbox) | "消息队列" | 每个演员的消息队列——异步处理输入 |
| 故障隔离 | "一个失败不影响其他" | 演员模型中一个演员崩溃不会影响其他演员 |

---

## 📚 小结

AutoGen v0.4 围绕演员模型——异步消息、事件驱动、故障隔离。演员模型适合复杂的多智能体系统。AutoGen 已进入维护模式——新项目推荐 Microsoft Agent Framework。

---

## ✏️ 练习

1. **【实现】** 用演员模型实现一个代码审查系统——Coder 演员 + Reviewer 演员 + 异步消息
2. **【对比】** 对比演员模型和状态图在相同任务上的实现差异

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 演员模型 | `code/main.py` | 异步消息 + 事件驱动 |

---

## 📖 参考资料

1. [论文] AutoGen v0.4: https://github.com/microsoft/autogen
2. [文档] Microsoft Agent Framework
3. [书籍] Hewitt et al. "Actor Model". 1973.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
