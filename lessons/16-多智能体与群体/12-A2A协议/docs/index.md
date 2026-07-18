# A2A 协议——智能体到智能体的通用线路协议

> Google 于 2025 年 4 月宣布 A2A；到 2026 年 4 月规范位于 https://a2a-protocol.org/latest/specification/ 且 150+ 组织支持。A2A 是 MCP 的水平补充（第 13 课）：MCP 是垂直的（智能体↔工具），A2A 是点对点的（智能体↔智能体）。它定义了 Agent Card（发现）、带工件的任务（文本、结构化数据、视频）、不透明的任务生命周期和认证。生产系统越来越多地将 MCP 与 A2A 配对。Google Cloud 在 2025-2026 年间将 A2A 支持集成到 Vertex AI Agent Builder。

**类型：** 概念课 + 实现课
**语言：** Python（标准库，`http.server`、`json`）
**前置知识：** 阶段 16 · 04（原语模型）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述 A2A 的四个核心元素——Agent Card、任务、工件、不透明生命周期
- [ ] 实现一个 A2A 最小服务器和客户端——发现、提交任务、轮询、获取工件
- [ ] 对比 MCP（垂直：智能体↔工具）和 A2A（水平：智能体↔智能体）——哪个解决哪个问题
- [ ] 理解 A2A 在 150+ 组织采用中的生态位

---

## 1. 问题

你的智能体需要调用另一个系统上的另一个智能体。怎么做？你可以暴露一个 HTTP 端点、定义一个自定义 JSON Schema，然后希望另一边说同一种语言。每一对智能体都变成一个定制集成。

A2A 是那个调用的通用线路协议。标准化发现、标准化任务模型、标准化传输、标准化工件。像 HTTP+REST 但智能体是一等公民。

---

## 2. 概念

### 2.1 四个核心元素

**Agent Card：** `/.well-known/agent.json` 上的 JSON 文档，描述智能体：名称、技能、端点、支持的模态、认证要求。通过读取卡片进行发现。

**任务：** 工作单元。异步、有状态的对象，生命周期：`submitted → working → completed / failed / canceled`。客户端发送任务、轮询或订阅更新。

**工件：** 任务产生的结果类型。文本、结构化工件、图像、视频、音频。工件是类型化的。

**不透明生命周期：** A2A 不规定远程智能体如何解决任务。客户端看到状态转换和工件；实现可以自由使用任何框架。

### 2.2 MCP vs A2A

```
MCP（第 13 课）：智能体 ↔ 工具（垂直）
A2A：智能体 ↔ 智能体（水平）
```

生产多智能体系统两者都用。A2A 对等体在其侧面调用 MCP 工具。拆分保持两个关注点清洁。

### 2.3 发现流程

```
客户端                     智能体服务器
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

### 2.4 认证模式

A2A 支持三种模式：Bearer 令牌（OAuth2）、mTLS（双向 TLS）、签名请求（HMAC）。认证在 Agent Card 中声明；客户端发现并遵守。

### 2.5 A2A 适用/不适用

**适用：** 跨组织调用、异构框架、类型化工件、长时间运行任务

**不适用：** 延迟敏感微调用（异步生命周期太重）、紧耦合进程内智能体（HTTP 往返过量）、小团队（规范开销真实）

---

## 3. 从零实现

### 第 3 步：实现 A2A 服务器

```python
import json, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from uuid import uuid4

AGENT_CARD = {
    "name": "code-review-agent",
    "skills": ["review-python"],
    "endpoints": {"tasks": "http://localhost:8765/tasks"},
    "protocol_version": "a2a-0.3",
}

class TaskStore:
    def __init__(self):
        self.tasks = {}
        self._lock = threading.Lock()

    def create(self, skill, payload):
        tid = str(uuid4())[:8]
        with self._lock:
            self.tasks[tid] = {"id": tid, "skill": skill, "payload": payload,
                               "state": "submitted", "artifact": None}
        threading.Thread(target=self._run, args=(tid,), daemon=True).start()
        return tid

    def _run(self, tid):
        with self._lock:
            self.tasks[tid]["state"] = "working"
        time.sleep(0.2)
        with self._lock:
            t = self.tasks[tid]
            if t["skill"] == "review-python":
                code = t["payload"].get("code", "")
                issues = []
                if "return" not in code: issues.append("no return statement")
                if "def " not in code: issues.append("no function definition")
                t["artifact"] = {"type": "structured", "data": {"issues": issues}}
                t["state"] = "completed"
            else:
                t["state"] = "failed"
                t["artifact"] = {"type": "text", "data": f"unknown skill '{t['skill']}'"}
```

### 第 4 步：运行演示

```python
def main():
    # 启动服务器
    server = HTTPServer(("localhost", 8765), A2AHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    # 客户端：发现 → 提交 → 轮询 → 获取工件
    card = http_json("GET", "http://localhost:8765/.well-known/agent.json")
    resp = http_json("POST", card["endpoints"]["tasks"],
                     {"skill": "review-python", "payload": {"code": "x = 1\nprint(x)\n"}})
    tid = resp["task_id"]

    for i in range(10):
        task = http_json("GET", f"http://localhost:8765/tasks/{tid}")
        if task["state"] in ("completed", "failed"):
            print(f"工件: {task['artifact']}")
            break
        time.sleep(0.1)
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 A2A 生态（2026 年 4 月）

| 组件 | 说明 |
|------|------|
| 150+ 组织支持 | 企业采用驱动规模化 |
| Google Cloud Vertex AI | A2A 支持集成 |
| Microsoft Agent Framework | 支持 A2A |
| LangGraph/CrewAI/AutoGen | A2A 适配器 |

### 4.2 A2A vs 相关规范

| 规范 | 范围 | 采用度 |
|------|------|--------|
| A2A | 智能体到智能体 | 最高（150+ 组织）|
| ACP | 企业审计 | 合并到 A2A |
| ANP | 去中心化信任 | 活跃开发 |
| NLIP | 自然语言内容 | 标准化 2025 年 12 月 |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 固定规范版本 | A2A 仍在演进；Agent Card 应声明协议版本 |
| 幂等任务创建 | 重复提交（网络重试）应产生一个任务 |
| 工件 Schema | 声明智能体返回什么形状；消费者应验证 |
| 速率限制 + 认证 | A2A 是面向公网的；应用标准 Web 安全 |
| 失败任务死信 | 随时间检查模式以发现重复失败类型 |

---

## 6. 常见错误

### 错误 1：不固定规范版本

**现象：** A2A 规范更新后，旧客户端与新服务器不兼容。

**修复：** Agent Card 声明 `protocol_version`。消费者在解析前检查版本。

### 错误 2：不处理幂等性

**现象：** 网络重试导致同一任务被创建两次。

**修复：** 幂等任务创建——重复提交产生一个任务。

### 错误 3：忽视认证

**现象：** A2A 端点无认证，任何人都可以提交任务。

**修复：** Agent Card 声明认证模式。生产中使用 Bearer 或 mTLS。

---

## 7. 面试考点

### Q1：A2A 的四个核心元素是什么？（难度：⭐）

**参考答案：**
Agent Card（发现）、任务（异步有状态对象）、工件（类型化结果：文本/JSON/图像/视频）、不透明生命周期（实现自由）。

### Q2：MCP 和 A2A 的区别是什么？（难度：⭐⭐）

**参考答案：**
MCP 是垂直的——智能体↔工具（客户端-服务器）。A2A 是水平的——智能体↔智能体（点对点）。生产系统两者都用：A2A 对等体在其侧面调用 MCP 工具。

### Q3：A2A 的不透明生命周期意味着什么？（难度：⭐⭐⭐）

**参考答案：**
客户端看到状态转换（submitted → working → completed）和工件。它不看到远程智能体如何解决任务——实现可以自由使用任何框架。这使异构框架可以互操作：LangGraph 智能体调用 CrewAI 智能体，通过 A2A 规范化。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| A2A | "智能体到智能体" | 跨系统智能体调用的点对点协议 |
| Agent Card | "智能体的名片" | `/.well-known/agent.json` 上的 JSON：技能、端点、认证 |
| 任务 | "工作单元" | 异步有状态对象，带生命周期；完成时产生工件 |
| 工件 | "结果" | 类型化输出：文本、JSON、图像、视频、音频 |
| 不透明生命周期 | "如何解决是智能体的事" | 客户端看到状态转换；实现自由选择框架 |
| MCP vs A2A | "工具 vs 对等" | MCP：垂直智能体↔工具。A2A：水平智能体↔智能体 |

---

## 📚 小结

A2A 是智能体到智能体的通用线路协议——标准化发现、任务、工件和认证。与 MCP 互补：MCP 垂直（工具），A2A 水平（对等）。150+ 组织在 2026 年 4 月支持。不透明生命周期使异构框架互操作。延迟敏感微调用和紧耦合进程内智能体不适合 A2A。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。确认客户端发现服务器并收到正确工件。

2. **【实现】** 在服务器上添加第二个技能（如"摘要"）。更新 Agent Card。写一个根据任务类型选择技能的客户端。

3. **【实现】** 实现 SSE 流式端点：`/tasks/{id}/events` 发出状态变更。客户端需要做什么不同？

4. **【阅读】** 阅读 A2A 规范。识别三个规范强制要求但演示未实现的东西。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| A2A 最小实现 | `code/main.py` | 服务器 + 客户端 + 发现 + 任务生命周期 |
| 技能提示词 | `outputs/skill-a2a-integrator.md` | 设计 A2A 集成 |

---

## 📖 参考资料

1. [规范] A2A. https://a2a-protocol.org/latest/specification/
2. [博客] Google Developers Blog. "A2A: A New Era of Agent Interoperability". https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
3. [GitHub] A2A. https://github.com/a2aproject/A2A
4. [论文] Liu 等人. "A Survey of Agent Interoperability Protocols". https://arxiv.org/html/2505.0227v1

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
