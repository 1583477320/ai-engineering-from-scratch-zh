# Claude 智能体 SDK：子智能体与会话存储

> Claude Agent SDK 是 Claude Code harness 的库形式。内置工具、用于上下文隔离的子智能体、钩子、W3C 追踪传播、会话存储对等。Claude Managed Agents 是长时间运行异步工作的托管替代。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、10（技能库）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分 Anthropic Client SDK（原始 API）和 Claude Agent SDK（harness 形式）
- [ ] 理解子智能体的上下文隔离——为什么需要子智能体
- [ ] 实现 Claude Agent SDK 的会话存储和钩子机制
- [ ] 对比自托管和 Managed Agents 的权衡

---

## 1. 问题

Claude Code 的 harness 提供了完整的智能体运行环境——工具、追踪、会话管理。Claude Agent SDK 将这个 harness 库化——开发者可以直接调用。但有些场景需要子智能体——上下文隔离、并行处理、专用任务。

---

## 2. 概念

### 2.1 两个 SDK

| SDK | 定位 | 特点 |
|-----|------|------|
| **Anthropic Client SDK** | 原始 API | 直接调用 Messages API |
| **Claude Agent SDK** | Harness 库 | 内置工具、子智能体、会话存储、追踪 |

### 2.2 子智能体

子智能体 = 独立的 Claude 实例——有自己的上下文窗口、工具和状态。

```
主智能体（大上下文）
    ├── 子智能体 A（专用上下文：代码审查）
    ├── 子智能体 B（专用上下文：文档生成）
    └── 子智能体 C（专用上下文：测试编写）
```

### 2.3 会话存储

Claude Agent SDK 的会话存储支持：
- 消息历史持久化
- 跨会话的上下文连续性
- 工具调用结果缓存

### 2.4 Managed Agents

Claude Managed Agents 是托管服务——长时间运行的异步工作，不需要用户保持会话打开。

---

## 3. 从零实现

### Step 1：Claude Agent SDK 概念

```python
class ClaudeAgentSDK:
    """简化版 Claude Agent SDK。"""
    def __init__(self, model="claude-sonnet-5"):
        self.model = model
        self.tools = []
        self.sub_agents = {}
        self.hooks = {"pre_call": [], "post_call": []}

    def add_tool(self, name, handler):
        self.tools.append({"name": name, "handler": handler})

    def add_hook(self, event, handler):
        self.hooks[event].append(handler)

    def add_subagent(self, name, agent):
        self.sub_agents[name] = agent

    def run(self, prompt):
        """运行智能体。"""
        # 触发预钩子
        for hook in self.hooks["pre_call"]:
            prompt = hook(prompt)

        # 调用 Claude
        response = f"[Claude] 处理: {prompt[:50]}..."

        # 触发后钩子
        for hook in self.hooks["post_call"]:
            response = hook(response)

        return response
```

### Step 2：子智能体隔离

```python
class SubAgent:
    """子智能体——独立上下文。"""
    def __init__(self, name, instructions):
        self.name = name
        self.instructions = instructions
        self.history = []

    def run(self, task):
        self.history.append({"task": task})
        result = f"[{self.name}] 完成: {task[:30]}"
        return result
```

---

## 4. 工具

### 4.1 Claude Agent SDK

```python
from anthropic import Anthropic

# 原始 Client SDK
client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "你好"}],
)
```

### 4.2 对比

| 方面 | Client SDK | Agent SDK |
|------|-----------|-----------|
| 工具 | 手动实现 | 内置 |
| 子智能体 | 不支持 | 支持 |
| 会话存储 | 手动 | 内置 |
| 钩子 | 手动 | 内置 |

---

## 5. 工程最佳实践

### 5.1 子智能体设计

- **上下文隔离**：每个子智能体有独立的上下文——避免信息污染
- **专用工具**：子智能体只访问所需的工具
- **结果聚合**：主智能体汇总子智能体的结果

### 5.2 踩坑经验

- **子智能体上下文溢出**：长任务用子智能体分担上下文
- **钩子执行顺序**：预钩子和后钩子的执行顺序需要控制

---

## 6. 常见错误

### 错误 1：子智能体共享上下文

**现象：** 多个子智能体互相干扰。

**修复：** 每个子智能体必须有独立的上下文窗口。

### 错误 2：钩子中执行重操作

**现象：** 钩子执行时间过长——阻塞主智能体。

**修复：** 钩子只做轻量检查——重操作放回主流程。

---

## 7. 面试考点

### Q1：Claude Agent SDK 和 Anthropic Client SDK 有什么区别？（难度：⭐⭐）

**参考答案：**
Client SDK 是原始 API——直接调用 Messages 接口，开发者需要自己实现工具、会话管理和追踪。Agent SDK 是 harness 库——内置工具、子智能体、会话存储、追踪和钩子。Agent SDK 更适合构建完整的智能体应用，Client SDK 更适合简单场景。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Agent SDK | "Claude 智能体库" | Anthropic 的智能体运行 harness——内置工具、子智能体、会话存储 |
| 子智能体 | "独立 Claude 实例" | 有独立上下文窗口的 Claude 实例——用于上下文隔离 |
| Managed Agents | "托管智能体" | Anthropic 的长时间运行异步工作托管服务 |
| 钩子 | "预处理/后处理" | 智能体调用前后的回调——用于日志、验证、修改 |

---

## 📚 小结

Claude Agent SDK 是 Claude Code harness 的库形式——内置工具、子智能体、会话存储、追踪。子智能体提供上下文隔离。Managed Agents 是长时间运行工作的托管替代。

---

## ✏️ 练习

1. **【对比】** 对比 Client SDK 和 Agent SDK 在构建聊天机器人时的代码量差异
2. **【设计】** 设计一个使用子智能体的代码审查系统——主智能体调度，子智能体审查不同方面

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| Agent SDK 演示 | `code/main.py` | Agent + 子智能体 + 钩子 |

---

## 📖 参考资料

1. [文档] Claude Agent SDK
2. [文档] Anthropic Client SDK
3. [文档] Claude Managed Agents

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
