# Agno 与 Mastra：生产运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年的生产运行时配对。Agno 追求微秒级智能体实例化和无状态 FastAPI 后端。Mastra 提供智能体、工具、工作流、统一模型路由和 Vercel AI SDK 基础上的组合存储。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、13（LangGraph）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 识别 Agno 的性能目标及其适用场景
- [ ] 对比 Agno 和 Mastra 的架构差异
- [ ] 理解生产运行时与开发框架的区别
- [ ] 选择适合项目需求的运行时

---

## 1. 问题

开发阶段用 LangGraph/CrewAI 构建原型——但生产环境需要更低的延迟、更高的吞吐量、更好的故障隔离。Agno 和 Mastra 是 2026 年的生产级解决方案。

---

## 2. 概念

### 2.1 Agno vs Mastra

| 方面 | Agno | Mastra |
|------|------|--------|
| 语言 | Python | TypeScript |
| 核心目标 | 微秒级实例化 | 工具+工作流+存储 |
| 架构 | 无状态 FastAPI | Vercel AI SDK 基础 |
| 适用 | 高性能后端 | Web 应用集成 |

### 2.2 Agno 的核心优势

- **微秒级实例化**：智能体创建和销毁极快
- **无状态**：每次请求独立——无共享状态导致的竞态条件
- **FastAPI 集成**：原生支持 Web 服务
- **内存效率**：轻量级内存占用

### 2.3 Mastra 的核心优势

- **统一框架**：智能体 + 工具 + 工作流
- **模型路由**：自动选择最优模型
- **组合存储**：向量+文档+图的统一存储
- **Vercel 集成**：一键部署到 Vercel

---

## 3. 从零实现

### Step 1：Agno 风格的无状态智能体

```python
class StatelessAgent:
    """Agno 风格——无状态、微秒级实例化。"""
    def __init__(self, model_fn, tools):
        self.model_fn = model_fn
        self.tools = tools

    def run(self, query, context=None):
        """无状态运行——每次调用独立。"""
        context = context or []
        return self.model_fn(query, context=context, tools=self.tools)
```

### Step 2：Mastra 风格的统一存储

```python
class UnifiedStorage:
    """Mastra 风格——组合存储。"""
    def __init__(self):
        self.vector_store = {}
        self.document_store = {}

    def store(self, key, value, embedding=None):
        self.document_store[key] = value
        if embedding is not None:
            self.vector_store[key] = embedding

    def search(self, query):
        results = []
        for key, val in self.document_store.items():
            if any(w in key.lower() for w in query.lower().split()[:3]):
                results.append((key, val))
        return results
```

---

## 4. 工具

### 4.1 Agno

```python
# https://github.com/agno-agi/agno
# pip install agno
```

### 4.2 Mastra

```bash
# https://github.com/mastra-ai/mastra
npm install mastra
```

---

## 5. 工程最佳实践

### 5.1 运行时选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| Python 后端 | Agno | 微秒级实例化、FastAPI 集成 |
| Web 应用 | Mastra | Vercel 一键部署 |
| 高吞吐 | Agno | 无状态、内存高效 |

---

## 7. 面试考点

### Q1：Agno 和 LangGraph 有什么区别？（难度：⭐⭐）

**参考答案：**
LangGraph 是开发框架——提供状态图、检查点、调试。Agno 是生产运行时——提供微秒级实例化、无状态执行、FastAPI 集成。LangGraph 适合开发和调试，Agno 适合高吞吐生产部署。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Agno | "Python 生产运行时" | 微秒级智能体实例化、无状态、FastAPI 集成 |
| Mastra | "TypeScript 生产运行时" | 统一框架、Vercel 集成、组合存储 |
| 无状态运行时 | "无共享状态" | 每次请求独立——无竞态条件 |

---

## 📚 小结

Agno 追求性能（微秒级、无状态），Mastra 追求集成（统一框架、Vercel 部署）。选择取决于：Python 后端用 Agno，Web 应用用 Mastra。

---

## ✏️ 练习

1. **【对比】** 对比 Agno 和 LangGraph 在相同任务上的延迟和吞吐量
2. **【设计】** 为一个电商客服系统设计 Agno 运行时架构

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 无状态智能体 | `code/main.py` | Agno 风格的无状态运行时 |

---

## 📖 参考资料

1. [GitHub] Agno: https://github.com/agno-agi/agno
2. [GitHub] Mastra: https://github.com/mastra-ai/mastra
3. [文档] Agno 文档
