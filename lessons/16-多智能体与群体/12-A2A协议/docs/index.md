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
- [ ] 对比 MCP（垂直：智能体↔工具）和 A2A（水平：智能体↔智能体）
- [ ] 理解 A2A 在 150+ 组织采用中的生态位和与 ACP/ANP/NLIP 的关系

---

## 1. 问题

你的智能体需要调用另一个系统上的另一个智能体。怎么做？你可以暴露一个 HTTP 端点、定义一个自定义 JSON Schema，然后希望另一边说同一种语言。每一对智能体都变成一个定制集成。

A2A 是那个调用的通用线路协议。标准化发现、标准化任务模型、标准化传输、标准化工件。像 HTTP+REST 但智能体是一等公民。

---

## 2. 概念

### 2.1 四个核心元素

| 元素 | 说明 | 示例 |
|------|------|------|
| **Agent Card** | `/.well-known/agent.json` 上的 JSON，描述名称、技能、端点、认证 | 发现机制 |
| **任务** | 异步有状态对象，生命周期：`submitted → working → completed / failed / canceled` | 工作单元 |
| **工件** | 类型化结果：文本、JSON、图像、视频、音频 | 任务产出 |
| **不透明生命周期** | 客户端看到状态转换；实现自由选择框架 | 异构互操作 |

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

A2A 支持三种认证模式：Bearer 令牌（OAuth2）、mTLS（双向 TLS）、签名请求（HMAC）。认证在 Agent Card 中声明；客户端发现并遵守。

### 2.5 A2A vs 相关规范

| 规范 | 范围 | 采用度（2026 年 4 月） |
|------|------|----------------------|
| A2A | 智能体↔智能体 | 150+ 组织，最高 |
| ACP | 企业审计（轨迹元数据） | 合并到 A2A |
| ANP | 去中心化信任（DID） | 活跃开发 |
| NLIP | 自然语言内容 | 2025 年 12 月标准化 |

### 2.6 A2A 适用/不适用

**适用：** 跨组织调用、异构框架（LangGraph↔CrewAI↔自定义）、类型化工件、长时间运行任务。

**不适用：** 延迟敏感微调用（异步生命周期太重）、紧耦合进程内智能体（HTTP 往返过量）、小团队（规范开销真实）。

---

## 3. 从零实现

### 第 1 步：定义 Agent Card 和任务存储

```python
AGENT_CARD = {
    "name": "code-review-agent",
    "version": "0.1.0",
    "skills": ["review-python"],
    "endpoints": {"tasks": "http://localhost:8765/tasks"},
    "auth": {"type": "none"},
    "modalities": ["text", "structured"],
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
```

### 第 2 步：实现 HTTP 服务器

```python
class A2AHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/.well-known/agent.json":
            self._send_json(200, AGENT_CARD)
        elif self.path.startswith("/tasks/"):
            tid = self.path.split("/tasks/", 1)[1]
            task = STORE.get(tid)
            if task:
                self._send_json(200, task)
            else:
                self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/tasks":
            body = json.loads(self.rfile.read(...))
            tid = STORE.create(body.get("skill", ""), body.get("payload", {}))
            self._send_json(201, {"task_id": tid, "state": "submitted"})
```

### 第 3 步：实现客户端

```python
def run_client():
    # 发现
    card = http_json("GET", "http://localhost:8765/.well-known/agent.json")
    print(f"名称={card['name']}, 技能={card['skills']}")

    # 提交任务
    resp = http_json("POST", card["endpoints"]["tasks"],
                     {"skill": "review-python", "payload": {"code": "x = 1\n"}})
    tid = resp["task_id"]

    # 轮询直到完成
    for i in range(10):
        task = http_json("GET", f"http://localhost:8765/tasks/{tid}")
        if task["state"] in ("completed", "failed"):
            print(f"工件: {task['artifact']}")
            break
        time.sleep(0.1)
```

### 第 4 步：运行演示

```python
def main():
    server = run_server()
    time.sleep(0.1)
    try:
        run_client()
    finally:
        server.shutdown()
    print("要点: 发现 + 任务生命周期 + 类型化工件 + 认证 = A2A 表面。")
    print("MCP 是智能体 ↔ 工具（垂直）；A2A 是智能体 ↔ 智能体（水平）。生产两者都用。")
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

### 4.2 A2A vs 相关规范对比

| 规范 | 核心特性 | 最适合 |
|------|---------|--------|
| A2A | Agent Card + 任务 + 工件 | 跨组织智能体协作 |
| ACP | 轨迹元数据 + 审计 | 受监管行业 |
| ANP | DID + E2EE + 元协议 | 去中心化信任 |
| NLIP | 自然语言内容 | 人类-智能体交互 |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 固定规范版本 | A2A 仍在演进；Agent Card 应声明协议版本 |
| 幂等任务创建 | 重复提交产生一个任务 |
| 工件 Schema | 声明智能体返回的形状；消费者应验证 |
| 速率限制 + 认证 | A2A 面向公网；应用标准 Web 安全 |
| 失败任务死信 | 随时间检查重复失败类型 |

### 5.1 中文场景特别建议

- **A2A 的 Agent Card 用中文描述**——技能名、描述用中文，但 tags 和 MIME 类型保持英文
- **认证在中文云环境中同样可用**——阿里云、腾讯云的 OAuth 服务兼容 A2A 的认证模式
- **A2A 在中文企业中的应用**——跨部门智能体协作、跨公司数据交换

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

### Q4：A2A 的认证模式有哪些？（难度：⭐⭐）

**参考答案：**
Bearer 令牌（OAuth2）、mTLS（双向 TLS）、签名请求（HMAC）。认证在 Agent Card 中声明；客户端发现并遵守。

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

5. **【思考】** 对比 A2A（Agent Card 发现）和 MCP（`listTools` 能力列举）。智能体描述和能力探测之间的权衡是什么？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| A2A 最小实现 | `code/main.py` | 服务器 + 客户端 + 发现 + 任务生命周期 |
| 技能提示词 | `outputs/skill-a2a-integrator.md` | 设计 A2A 集成 |

---

## 📖 参考资料

1. [规范] A2A. https://a2a-protocol.org/latest/specification/ — 规范版本
2. [博客] Google Developers Blog. "A2A: A New Era of Agent Interoperability". https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/ — 2025 年 4 月启动文章
3. [GitHub] A2A. https://github.com/a2aproject/A2A — 参考实现和 SDK
4. [论文] Liu 等人. "A Survey of Agent Interoperability Protocols". https://arxiv.org/html/2505.0227v1 — MCP、ACP、A2A、ANP 对比

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
