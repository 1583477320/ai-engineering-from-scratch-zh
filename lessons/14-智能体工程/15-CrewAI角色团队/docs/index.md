# CrewAI：基于角色的团队与流程

> CrewAI 是 2026 年基于角色的多智能体框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crews（自主的、基于角色的协作）和 Flows（事件驱动、确定性）。文档直白地说："对于任何生产就绪的应用，从 Flow 开始。"

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 12（工作流模式）、14（Actor 模型）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 命名 CrewAI 的四个原语——Agent、Task、Crew、Process——以及各自负责什么
- [ ] 区分 Crews 和 Flows 的适用场景
- [ ] 实现一个 CrewAI 风格的多智能体团队
- [ ] 设计角色分工——让不同智能体专注于不同任务

---

## 1. 问题

多智能体系统需要：角色分工、任务调度、自主协作。CrewAI 提供了最简单的实现——四个原语、两种形态。对于生产应用，Flows（确定性、事件驱动）比 Crews（自主、基于角色）更可靠。

---

## 2. 概念

### 2.1 CrewAI 四原语

| 原语 | 说明 | 类比 |
|------|------|------|
| **Agent** | 执行任务的智能体 | 员工 |
| **Task** | 需要完成的任务 | 工单 |
| **Crew** | 协作的智能体团队 | 项目组 |
| **Process** | 任务执行的顺序 | 工作流程 |

### 2.2 Crews vs Flows

| 方面 | Crews | Flows |
|------|-------|-------|
| 协作方式 | 自主、角色扮演 | 确定性、事件驱动 |
| 适用场景 | 开放式探索 | 生产级应用 |
| 可靠性 | 中 | 高 |
| 生产推荐 | 否 | 是（文档推荐） |

### 2.3 角色设计

```python
# CrewAI 角色定义
researcher = Agent(role="研究分析师", goal="收集和分析数据", 
                   backstory="你是数据分析专家")
writer = Agent(role="技术撰稿人", goal="撰写技术报告", 
               backstory="你是一位资深技术作家")
reviewer = Agent(role="审查员", goal="审查报告质量", 
                 backstory="你有10年技术写作审查经验")
```

---

## 3. 从零实现

### Step 1：简化版 Crew

```python
class SimpleCrew:
    """简化版 CrewAI。"""
    def __init__(self, agents):
        self.agents = agents

    def execute(self, tasks):
        """顺序执行任务。"""
        results = []
        for task in tasks:
            agent = task.get("assignee", self.agents[0])
            print(f"  {agent['role']} 执行: {task['description']}")
            result = f"{agent['role']} 完成了: {task['description']}"
            results.append(result)
        return results
```

---

## 4. 工具

### 4.1 CrewAI

```python
from crewai import Agent, Task, Crew

researcher = Agent(role="研究分析师", goal="收集数据")
writer = Agent(role="撰稿人", goal="撰写报告")
task = Task(description="写一份市场报告", agent=researcher)
crew = Crew(agents=[researcher, writer], tasks=[task])
result = crew.kickoff()
```

---

## 5. 工程最佳实践

### 5.1 Crew vs Flow 选择

- **原型/探索**：Crew（角色协作更灵活）
- **生产部署**：Flow（确定性、可审计）
- **简单任务**：直接 API 调用

---

## 6. 常见错误

### 错误 1：生产环境用 Crew 而非 Flow

**现象：** Crew 的自主行为导致不可预测的结果。

**修复：** 生产应用用 Flow——确定性、可调试。

---

## 7. 面试考点

### Q1：CrewAI 的四个原语是什么？（难度：⭐⭐）

**参考答案：** Agent（智能体）、Task（任务）、Crew（团队）、Process（执行顺序）。Crews 适合探索性任务，Flows 适合生产级应用。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| CrewAI | "角色扮演框架" | 基于角色的多智能体框架——Agent+Task+Crew+Process |
| Crews | "自主团队" | 基于角色的自主协作——适合探索 |
| Flows | "确定性流程" | 事件驱动、确定性执行——适合生产 |

---

## 📚 小结

CrewAI 四原语：Agent、Task、Crew、Process。两种形态：Crews（自主）和 Flows（确定性）。文档推荐生产应用从 Flow 开始。

---

## ✏️ 练习

1. **【实现】** 用 CrewAI 定义一个研究团队——研究员、撰稿员、审查员
2. **【对比】** 对比 Crews 和 Flows 在同一个任务上的表现

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| CrewAI 团队 | `code/main.py` | 四原语 + Crew/Flow 对比 |

---

## 📖 参考资料

1. [GitHub] CrewAI: https://github.com/joaomdmoura/crewAI
2. [文档] CrewAI Flows

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
