# 通信协议：MCP、A2A、ACP、ANP

> 不能说同一种语言的智能体不是团队——它们是对着虚空喊叫的陌生人。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 14（智能体工程）、阶段 16 · 01（为什么多智能体）
**预计时间：** ~120 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现 MCP 工具发现和调用——智能体可以使用外部服务器暴露的工具
- [ ] 构建 A2A Agent Card 和任务端点——允许一个智能体通过 HTTP 委派工作
- [ ] 比较 MCP、A2A、ACP 和 ANP 并解释哪个协议解决哪个问题
- [ ] 在单一系统中连接多个协议——智能体通过 MCP 发现工具、通过 A2A 委派任务

---

## 1. 问题

你将系统拆分为多个智能体。研究员、编码者、审查者。它们各自擅长本职工作。但现在你需要它们真正互相交流。

"只传字符串"突然崩了——编码者误解研究摘要，或两个智能体死锁互相等待，或不同团队构建的智能体无法协作。

通信协议问题：没有共享的信息交换契约，多智能体系统是脆弱的、不可审计的、无法扩展的。

AI 生态系统用四个协议回应：
- **MCP** 用于工具访问
- **A2A** 用于智能体间协作
- **ACP** 用于企业审计
- **ANP** 用于去中心化信任

---

## 2. 概念

### 2.1 四层协议景观

```
ANP — 智能体如何信任陌生人？DID + E2EE + 元协议
A2A — 智能体如何协作？Agent Card + 任务生命周期 + 流式传输
ACP — 如何在可审计系统中通信？轨迹元数据 + 会话连续性
MCP — 如何使用工具？工具发现 + 执行 + 上下文共享
```

它们不是竞争关系。解决不同层级的不同问题。

### 2.2 MCP 回顾（阶段 13）

MCP 标准化了 LLM 如何连接外部工具和数据源。客户端-服务器协议——智能体（客户端）发现和调用服务器暴露的工具。MCP 是**智能体到工具**的通信，不帮助智能体互相交谈。

### 2.3 A2A（Agent2Agent 协议）

**创建者：** Google（Linux Foundation `lf.a2a.v1`）
**问题：** 自主智能体如何协作、协商、委派任务？

A2A 是**点对点智能体协作**协议。每个智能体在 `/.well-known/agent-card.json` 发布 Agent Card，其他智能体发现、协商、委派任务。

Agent Card 包含：名称、描述、技能（ID + 标签 + MIME 类型）、能力（流式传输、推送通知）、安全方案。

#### 任务生命周期（9 状态）

| 状态 | 终态？ | 含义 |
|------|--------|------|
| `submitted` | 否 | 已确认，尚未处理 |
| `working` | 否 | 正在处理 |
| `input-required` | 否 | 需要更多信息 |
| `auth-required` | 否 | 需要认证 |
| `completed` | 是 | 成功完成 |
| `failed` | 是 | 出错完成 |
| `canceled` | 是 | 完成前取消 |
| `rejected` | 是 | 智能体拒绝任务 |

终态不可变。后续在同一 `contextId` 内创建新任务。

### 2.4 ACP（Agent Communication Protocol）

**创建者：** IBM / BeeAI | **状态：** 正在合并到 A2A

ACP 的关键差异化：**TrajectoryMetadata**——每个响应携带推理步骤和工具调用的完整日志。对受监管行业这是审计金矿。

运行生命周期 7 状态：created → in_progress → completed/failed/awaiting/cancelling → cancelled。

### 2.5 ANP（Agent Network Protocol）

**创建者：** 开源社区 | **三层架构：**

1. 身份和安全通信（DID `did:wba` + HPKE E2EE）
2. 元协议协商（自然语言格式协商，最多 10 轮）
3. 应用协议（Agent 描述文档）

信任来自三源：域名级 TLS、DID 密码学签名、最小信任原则。

### 2.6 四协议对比

| | MCP | A2A | ACP | ANP |
|---|---|---|---|---|
| **主要用途** | 智能体到工具 | 智能体到智能体 | 智能体到智能体（可审计） | 智能体到智能体（去中心化） |
| **发现** | 工具列表 | Agent Card | `GET /agents` | DID 解析 |
| **身份** | 隐式 | OAuth/mTLS | 服务器级 | W3C DID + E2EE |
| **审计轨迹** | 无 | 基础 | TrajectoryMetadata | 未形式化 |
| **状态机** | 无 | 9 任务状态 | 7 运行状态 | 无 |
| **流式传输** | 无 | SSE | SSE | 传输无关 |
| **独特功能** | 工具 Schema | Agent Card + 技能 | 轨迹审计 | 元协议协商 |
| **最适合** | 工具和数据 | 动态协作 | 受监管行业 | 跨组织信任 |

### 2.7 协同工作

```
组织内：
  研究智能体 ←A2A→ 编码智能体
  研究智能体 →MCP→ 搜索服务器
  编码智能体 →MCP→ GitHub 服务器
  所有智能体响应携带 ACP 轨迹元数据

外部（通过 ANP 验证 DID）：
  研究智能体 ←ANP+A2A→ 外部智能体
```

### 2.8 常见生产失败

| 失败模式 | 说明 | 修复 |
|---------|------|------|
| Schema 漂移 | Agent Card 的 JSON Schema 版本间变化 | 版本化技能和输出 Schema |
| 状态机违规 | 终态后仍尝试更新 | 产出前检查终态 |
| 信任解析失败 | DID 域名不可达 | ANP 推荐 fail closed |
| 轨迹膨胀 | 智能体 200 次工具调用产生巨大审计条目 | 可配置详细级别 |
| 发现惊群 | 50 个智能体同时 `GET /agents` | 缓存 Agent Card + TTL + 错开发现间隔 |

---

## 3. 从零实现

### 第 1 步：核心消息类型

```python
@dataclass
class AgentMessage:
    role: str
    parts: list[dict]
    trajectory: list[dict] = field(default_factory=list)
    reply_to: str = ""

def text_message(role, text):
    return AgentMessage(role=role, parts=[{"kind": "text", "text": text}])
```

### 第 2 步：A2A Agent Card 和注册表

```python
@dataclass
class AgentCard:
    name: str
    description: str
    skills: list[dict]

class AgentRegistry:
    def __init__(self):
        self.cards = {}

    def register(self, card):
        self.cards[card.name] = card

    def discover_by_skill_tag(self, tag):
        return [c for c in self.cards.values()
                if any(tag in s.get("tags", []) for s in c.skills)]

    def resolve(self, name):
        return self.cards.get(name)
```

### 第 3 步：A2A 任务管理器

```python
TERMINAL_STATES = {"completed", "failed", "canceled", "rejected"}

@dataclass
class Task:
    id: str
    status: str = "submitted"
    artifacts: list = field(default_factory=list)

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.handlers = {}

    def register_handler(self, name, handler):
        self.handlers[name] = handler

    async def send_message(self, agent_name, message, context_id=None):
        handler = self.handlers.get(agent_name)
        if not handler:
            return Task(id="error", status="rejected")
        task = Task(id=f"task-{len(self.tasks)}")
        task.status = "working"
        try:
            result = await handler(task, message)
            task.status = "completed"
            task.artifacts = result
        except Exception as e:
            task.status = "failed"
        return task
```

### 第 4 步：协议网关

```python
class ProtocolGateway:
    def __init__(self, registry, task_manager):
        self.registry = registry
        self.task_manager = task_manager

    async def delegate_task(self, target_agent, message, session_id=None):
        card = self.registry.resolve(target_agent)
        if not card:
            return {"error": f"Agent {target_agent} not found"}
        task = await self.task_manager.send_message(target_agent, message)
        return {"task": task}
```

### 第 5 步：运行演示

```python
async def main():
    registry = AgentRegistry()
    registry.register(AgentCard(name="researcher", description="搜索并总结",
                                 skills=[{"id": "web-research", "tags": ["research"]}]))
    registry.register(AgentCard(name="coder", description="写代码",
                                 skills=[{"id": "code-gen", "tags": ["coding"]}]))
    task_manager = TaskManager()

    agents = registry.discover_by_skill_tag("research")
    print(f"发现 {len(agents)} 个智能体: {[a.name for a in agents]}")

    message = text_message("user", "研究 React 19 编译器特性")
    result = await task_manager.send_message("researcher", message)
    print(f"任务状态: {result.status}")

asyncio.run(main())
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 协议选择决策树

```
智能体需要使用工具？ → MCP
智能体需要互相通信？
  → 需要审计轨迹？ → A2A + ACP 轨迹模式
  → 同一组织？ → A2A（Agent Card + 任务）
  → 跨组织？ → ANP + A2A（DID 验证）
```

### 4.2 生产模式

| 协议 | 状态 | 最适合 |
|------|------|--------|
| MCP | 稳定 | 工具和数据 |
| A2A | 稳定（v1.0） | 动态协作 |
| ACP | 合并到 A2A | 受监管行业 |
| ANP | 活跃开发 | 跨组织信任 |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 四协议解决不同问题 | MCP 工具、A2A 协作、ACP 审计、ANP 信任 |
| 结构原语保留 | 谁→谁、意图、内容、关联 ID |
| 轨迹元数据用于审计 | 每个响应携带推理链和工具调用 |
| DID 用于跨组织信任 | 去中心化身份 + E2EE |
| 协议网关统一四者 | 一个入口处理验证→发现→审计→任务 |

---

## 6. 常见错误

### 错误 1：只用一个协议

**修复：** 四协议解决不同问题。真实系统使用多个：MCP 工具 + A2A 协作 + ACP 审计 + ANP 信任。

### 错误 2：忽略轨迹元数据

**修复：** 每个响应都包裹在轨迹元数据中——工具名称、输入、输出、推理步骤。

### 错误 3：不检查状态机约束

**修复：** 在产出事件之前检查终态。A2A 的 9 状态机有终态不可变约束。

---

## 7. 面试考点

### Q1：MCP、A2A、ACP、ANP 分别解决什么问题？（难度：⭐）

**参考答案：**
MCP：智能体到工具的通信。A2A：智能体间协作（Agent Card + 任务生命周期）。ACP：可审计通信（TrajectoryMetadata）。ANP：跨组织信任（DID + E2EE + 元协议协商）。

### Q2：A2A 的 Agent Card 包含什么？任务生命周期有哪些状态？（难度：⭐⭐）

**参考答案：**
Agent Card：名称、描述、版本、技能列表（ID、标签、MIME 类型）、能力、安全方案。任务 9 状态：submitted → working → input-required/auth-required → completed/failed/canceled/rejected。终态不可变。

### Q3：ACP 的 TrajectoryMetadata 为什么重要？（难度：⭐⭐）

**参考答案：**
每个响应携带推理步骤和工具调用的完整日志——哪个工具被调用、输入是什么、输出是什么、推理链。对受监管行业（金融、医疗）这是金矿：每个回答都有可证明的推理链。没有黑箱。

### Q4：ANP 如何实现跨组织信任？（难度：⭐⭐⭐）

**参考答案：**
W3C DID（`did:wba`）+ 密码学签名验证 + 域名级 TLS + 最小信任原则。没有基于八卦的信任传播。元协议协商是真正的新颖特性：从未见过对方的智能体用自然语言协商通信格式。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MCP | "AI 工具协议" | 智能体发现和使用工具的客户端-服务器协议 |
| A2A | "Google 的智能体协议" | Linux Foundation 下的点对点协作：Agent Card、9 状态任务、SSE |
| ACP | "企业智能体通信" | IBM/BeeAI REST API，TrajectoryMetadata 携带推理链 |
| ANP | "去中心化智能体身份" | DID + E2EE + 元协议协商 |
| Agent Card | "智能体的名片" | `/.well-known/agent-card.json` 上的能力描述 |
| TrajectoryMetadata | "审计收据" | ACP 为每个响应附加推理步骤和工具调用 |

---

## 📚 小结

四个协议解决不同层级的问题：MCP（工具访问）、A2A（智能体间协作）、ACP（企业审计）、ANP（去中心化信任）。它们不是竞争关系——真实系统使用多个。A2A 最成熟，ACP 的轨迹元数据被吸收进 A2A，ANP 的元协议协商是真正的新颖特性。通信协议不是新问题——FIPA-ACL 在 2000 年就解决了大部分结构原语。

---

## ✏️ 练习

1. **【实现】** 扩展 `TaskManager` 使处理器可以将子任务委派给其他智能体。研究者接收任务，委派子任务给两个专家，等待两者完成，然后合并结果。

2. **【实现】** 添加 DID 轮转：智能体发布包含更新密钥的新 DID 文档，维持 `previousDid` 引用。验证者在宽限期内接受当前和先前密钥。

3. **【设计】** 实现 ANP 元协议概念。两个智能体交换 `protocolNegotiation` 消息。最多 3 轮后达成格式一致或超时。

4. **【思考】** 添加速率限制发现：`RateLimitedRegistry` 缓存 Agent Card 并限制发现查询。模拟 100 个智能体同时启动时的惊群效应。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 四协议演示 | `code/main.py` | A2A Agent Card + 任务 + 注册表 + 网关 |
| 技能提示词 | `outputs/prompt-protocol-selector.md` | 为你的系统选择协议的提示词 |

---

## 📖 参考资料

1. [官方] Google A2A 规范. https://github.com/google/A2A — Linux Foundation v1.0.0
2. [官方] IBM/BeeAI ACP. https://github.com/i-am-bee/acp — OpenAPI 3.1 规范
3. [GitHub] Agent Network Protocol. https://github.com/agent-network-protocol/AgentNetworkProtocol — DID + E2EE + 元协议
4. [文档] Model Context Protocol. https://modelcontextprotocol.io/ — Anthropic MCP
5. [标准] W3C DID. https://www.w3.org/TR/did-core/ — ANP 的身份基础

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
