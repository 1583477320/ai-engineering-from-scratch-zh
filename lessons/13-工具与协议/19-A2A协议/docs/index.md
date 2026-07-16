# A2A——智能体对智能体协议

> MCP 是智能体对工具。A2A（Agent2Agent）是智能体对智能体——一个让不同框架构建的不透明智能体协作的开放协议。2025 年 4 月由 Google 发布，6 月捐赠给 Linux 基金会，2026 年 4 月达到 v1.0，支持者超过 150 家包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。本课介绍 Agent Card、Task 生命周期和两种传输绑定。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 06（MCP 基础）、08（MCP Client）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分智能体对工具（MCP）和智能体对智能体（A2A）的使用场景
- [ ] 实现 Agent Card 的发布和发现
- [ ] 理解 Task 生命周期——创建、执行、完成
- [ ] 对比 A2A 和 MCP 的互补关系

---

## 1. 问题

MCP 解决了智能体与工具的交互——但当两个智能体需要协作时，MCP 不够。智能体 A 想请求智能体 B 执行任务，但：

- 智能体 B 可能是闭源的——内部状态不可见
- 两个智能体可能基于不同框架
- 需要标准化的任务传递和状态管理

A2A（Agent-to-Agent）协议解决了这些问题——让不透明智能体之间可以协作。

---

## 2. 概念

### 2.1 A2A vs MCP

| 方面 | MCP | A2A |
|------|-----|-----|
| 端点 | 智能体 → 工具 | 智能体 → 智能体 |
| 可见性 | 工具接口暴露 | 智能体内部状态隐藏 |
| 传输 | JSON-RPC 2.0 | HTTP + SSE |
| 状态 | 无状态工具 | 有状态任务 |
| 适用场景 | API 调用、数据查询 | 任务委托、多智能体协作 |

### 2.2 Agent Card

智能体的"简历"——描述自身能力和限制：

```json
{
  "name": "Research Assistant",
  "description": "帮你搜索和总结学术论文",
  "url": "https://assistant.example.com/a2a",
  "skills": ["search", "summarize", "cite"],
  "authentication": "oauth2"
}
```

### 2.3 Task 生命周期

```
创建（created）→ 进行中（working）→ 完成（completed）/ 失败（failed）
```

### 2.4 A2A vs MCP 互补

| 场景 | MCP | A2A |
|------|-----|-----|
| 调用 API | ✅ | ❌ |
| 查询数据库 | ✅ | ❌ |
| 委托研究任务 | ❌ | ✅ |
| 协作编写报告 | ❌ | ✅ |

---

## 3. 从零实现

### Step 1：Agent Card

```python
class AgentCard:
    """A2A Agent Card——智能体的"简历"。"""
    def __init__(self, name, description, url, skills):
        self.name = name
        self.description = description
        self.url = url
        self.skills = skills

    def to_json(self):
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "skills": self.skills,
            "authentication": "oauth2",
        }

    def match(self, query):
        """检查智能体是否能处理查询。"""
        return any(skill in query.lower() for skill in self.skills)


# 注册智能体
agents = [
    AgentCard("Research Assistant", "帮你搜索学术论文", "https://research.example.com/a2a", ["search", "summarize"]),
    AgentCard("Code Assistant", "帮你写代码", "https://code.example.com/a2a", ["code", "debug"]),
    AgentCard("Writing Assistant", "帮你写文案", "https://writing.example.com/a2a", ["write", "edit"]),
]

def discover_agents(query):
    """发现能处理查询的智能体。"""
    return [agent for agent in agents if agent.match(query)]
```

### Step 2：Task 生命周期

```python
class TaskManager:
    """简化版 A2A Task 生命周期。"""
    def __init__(self):
        self.tasks = {}

    def create_task(self, agent_url, description):
        task_id = f"task-{len(self.tasks) + 1}"
        self.tasks[task_id] = {"status": "created", "agent": agent_url, "description": description}
        return task_id

    def start_task(self, task_id):
        self.tasks[task_id]["status"] = "working"

    def complete_task(self, task_id, result):
        self.tasks[task_id]["status"] = "completed"
        self.tasks[task_id]["result"] = result

    def get_status(self, task_id):
        return self.tasks.get(task_id, {"status": "not_found"})
```

---

## 4. 工具

### 4.1 A2A Python SDK

```python
# A2A 通过 HTTP/JSON-RPC 通信
# Server 发布 Agent Card
# Client 查询 Agent Card 并委托任务
```

### 4.2 A2A vs MCP 对比

| 方面 | MCP | A2A |
|------|-----|-----|
| 通信 | JSON-RPC 2.0 | HTTP + JSON-RPC |
| 状态 | 无状态工具 | 有状态任务 |
| 发现 | Client 调用 Server | 智能体发现并委托 |
| 传输 | stdio / HTTP | HTTP + SSE |

---

## 5. 工程最佳实践

### 5.1 A2A vs MCP 选择

| 场景 | 选择 | 原因 |
|------|------|------|
| 调用外部 API | MCP | 工具接口简单 |
| 委托复杂任务 | A2A | 有状态任务管理 |
| 多智能体协作 | A2A | 智能体间通信 |
| 单智能体 + 工具 | MCP | 足够简单 |

### 5.2 踩坑经验

- **Agent Card 未更新**：智能体能力变化后 Card 未刷新——Client 缓存了旧的 Card
- **任务状态不一致**：多智能体场景下状态同步困难——需要中心化的状态管理

---

## 6. 常见错误

### 错误 1：将 A2A 用于工具调用

**现象：** 过度复杂化——工具调用不需要 Task 生命周期。

**修复：** 工具调用用 MCP，任务委托用 A2A。

### 错误 2：忽略 Task 状态管理

**现象：** Task 创建后无人追踪——"丢失"的任务消耗资源。

**修复：** 实现 Task 状态监控和超时清理。

---

## 7. 面试考点

### Q1：A2A 和 MCP 的本质区别是什么？（难度：⭐⭐）

**参考答案：**
MCP 是智能体对工具——工具接口暴露，执行确定，无状态。A2A 是智能体对智能体——智能体内部隐藏，任务可能需要多步骤、需要状态管理、需要错误处理。MCP 适合 API 调用，A2A 适合任务委托。两者互补而非替代。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| A2A | "智能体对智能体" | 让不同框架构建的智能体协作的开放协议 |
| Agent Card | "智能体名片" | 描述智能体能力、限制和端点的 JSON 文档 |
| Task | "任务" | A2A 中的工作单元——有创建、执行、完成的生命周期 |
| A2A vs MCP | "协作 vs 工具调用" | A2A 处理智能体间协作，MCP 处理智能体与工具交互 |

---

## 📚 小结

A2A 是智能体对智能体的开放协议——让不同框架的智能体协作。Agent Card 描述能力，Task 管理生命周期。A2A 和 MCP 互补——MCP 处理工具调用，A2A 处理任务委托。

---

## ✏️ 练习

1. **【设计】** 设计一个研究助手的 Agent Card——定义能力、端点和认证方式
2. **【对比】** 对比 MCP 和 A2A 在"多智能体研究"场景中的分工

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| Agent Card + Task | `code/main.py` | Agent Card 发布 + Task 生命周期 |

---

## 📖 参考资料

1. [文档] A2A 规范: https://google.github.io/A2A/
2. [GitHub] A2A: https://github.com/google/A2A
3. [文档] ACP: https://github.com/IBM/ACP

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
