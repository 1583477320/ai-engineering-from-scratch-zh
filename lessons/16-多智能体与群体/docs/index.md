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
- [ ] 构建 A2A Agent Card 和任务端点——允许一个智能体通过 HTTP 将工作委派给另一个
- [ ] 比较 MCP（工具访问）、A2A（智能体间）、ACP（企业审计）和 ANP（去中心化信任）并解释哪个协议解决哪个问题
- [ ] 在单一系统中连接多个协议——智能体通过 MCP 发现工具、通过 A2A 委派任务

---

## 1. 问题

你将系统拆分为多个智能体。一个研究员、一个编码者、一个审查者。它们各自擅长本职工作。但现在你需要它们真正互相交流。

你的第一次尝试很明显：传递字符串。研究员返回一段文本，编码者以自己的方式解析它。直到编码者误解了研究摘要，或两个智能体死锁互相等待，或你需要不同团队构建的智能体协作。"只传字符串"突然崩了。

这是通信协议问题。没有共享的信息交换契约，多智能体系统是脆弱的、不可审计的、无法扩展到你个人编写的少数智能体之外的。

AI 生态系统用四个协议回应了这个问题，每个解决不同的切片：
- **MCP** 用于工具访问
- **A2A** 用于智能体间协作
- **ACP** 用于企业审计
- **ANP** 用于去中心化身份和信任

---

## 2. 概念

### 2.1 协议景观——四层模型

```
ANP — 智能体如何信任陌生人？去中心化身份（DID）、E2EE、元协议
A2A — 智能体如何协作完成目标？Agent Card、任务生命周期、流式传输、协商
ACP — 智能体如何在可审计系统中通信？运行、轨迹元数据、会话连续性
MCP — 智能体如何使用工具？工具发现、执行、上下文共享
```

它们不是竞争关系。它们解决不同层级的不同问题。

### 2.2 MCP 回顾

MCP（阶段 13 已深入涵盖）标准化了 LLM 如何连接外部工具和数据源。它是**客户端-服务器**协议——智能体（客户端）发现和调用服务器暴露的工具。

MCP 是**智能体到工具**的通信。它不帮助智能体互相交谈。

### 2.3 A2A（Agent2Agent 协议）

**创建者：** Google（现在在 Linux Foundation 下为 `lf.a2a.v1`）
**问题：** 自主智能体如何协作、协商、委派任务？

A2A 是**点对点智能体协作**的协议。每个智能体在知名 URL 发布 **Agent Card**，其他智能体发现、协商、委派任务。

#### Agent Card 示例

```json
{
  "name": "研究智能体",
  "description": "搜索文档并总结发现",
  "skills": [
    {
      "id": "web-research",
      "name": "Web Research",
      "tags": ["research", "search", "summarization"],
      "inputModes": ["text/plain"],
      "outputModes": ["application/json"]
    }
  ],
  "capabilities": { "streaming": true, "pushNotifications": false },
  "defaultInputModes": ["text/plain", "application/json"],
  "defaultOutputModes": ["text/plain", "application/json"]
}
```

#### 任务生命周期（9 状态）

| 状态 | 终态？ | 含义 |
|------|--------|------|
| `submitted` | 否 | 已确认，尚未处理 |
| `working` | 否 | 正在处理 |
| `input-required` | 否 | 需要更多信息 |
| `completed` | 是 | 成功完成 |
| `failed` | 是 | 出错完成 |
| `canceled` | 是 | 完成前取消 |
| `rejected` | 是 | 智能体拒绝任务 |

终态不可变。后续在同一 `contextId` 内创建新任务。

### 2.4 ACP（Agent Communication Protocol）

**创建者：** IBM / BeeAI
**状态：** 正在合并到 A2A
**问题：** 智能体如何在可审计系统中通信？

ACP 的关键差异化：**TrajectoryMetadata**——每个响应可以携带推理步骤和工具调用的完整日志。对受监管行业来说这是金矿：每个回答都有可证明的推理链。

### 2.5 ANP（Agent Network Protocol）

**创建者：** 开源社区
**问题：** 不同组织的智能体如何相互信任？

ANP 使用 W3C 去中心化标识符（DID）和端到端加密建立信任。三个层次：身份和安全通信（DID + HPKE E2EE）、元协议协商（自然语言格式协商）、应用协议（Agent 描述文档）。

### 2.6 四协议对比

| | MCP | A2A | ACP | ANP |
|---|---|---|---|---|
| **主要用途** | 智能体到工具 | 智能体到智能体 | 智能体到智能体（可审计） | 智能体到智能体（去中心化） |
| **发现** | 工具列表 | `/.well-known/agent-card.json` | `GET /agents` | DID 解析 |
| **身份** | 隐式 | OAuth/mTLS | 服务器级 | W3C DID + E2EE |
| **审计轨迹** | 无 | 基础（任务历史） | TrajectoryMetadata（工具调用、推理） | 未形式化 |
| **状态机** | 无 | 9 任务状态 | 7 运行状态 | 无 |
| **流式传输** | 无 | SSE | SSE | 传输无关 |
| **独特功能** | 工具 Schema | Agent Card + 技能 | 轨迹审计轨迹 | 元协议协商 |
| **最适合** | 工具和数据 | 动态协作 | 受监管行业 | 跨组织信任 |

### 2.7 协议如何协同工作

```
你的组织内：
  研究智能体 ←A2A→ 编码智能体
  研究智能体 →MCP→ 搜索服务器
  编码智能体 →MCP→ GitHub 服务器
  所有智能体响应携带 ACP 轨迹元数据

外部（通过 ANP 验证 DID）：
  研究智能体 ←ANP+A2A→ 外部智能体
  编码智能体 ←ANP+A2A→ 合作伙伴智能体
```

- **MCP** 连接每个智能体到它的工具
- **A2A** 处理智能体间的协作（内部和外部）
- **ACP** 将响应包裹在轨迹元数据中用于审计
- **ANP** 为你不控制的智能体提供身份验证

---

## 3. 从零实现

### 第 3 步：核心消息类型

```python
@dataclass
class AgentMessage:
    role: str        # "user" | "agent"
    parts: list[dict]  # 多模态内容
    trajectory: list[dict] = field(default_factory=list)
    reply_to: str = ""

def text_message(role: str, text: str) -> AgentMessage:
    return AgentMessage(role=role, parts=[{"kind": "text", "text": text}])
```

### 第 4 步：A2A Agent Card 和注册表

```python
@dataclass
class AgentCard:
    name: str
    description: str
    skills: list[dict]

class AgentRegistry:
    def __init__(self):
        self.cards = {}

    def register(self, card: AgentCard):
        self.cards[card.name] = card

    def discover_by_skill_tag(self, tag: str) -> list:
        return [c for c in self.cards.values()
                if any(s.get("tags", []) and tag in s.get("tags", []) for s in c.skills)]

    def resolve(self, name: str):
        return self.cards.get(name)
```

### 第 5 步：A2A 任务管理器

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

### 第 6 步：协议网关

```python
class ProtocolGateway:
    """连接四个协议的统一网关。"""
    def __init__(self, registry, task_manager, audit_runner=None):
        self.registry = registry
        self.task_manager = task_manager
        self.audit_runner = audit_runner

    async def delegate_task(self, target_agent, message, session_id=None):
        # A2A: 发现目标智能体
        card = self.registry.resolve(target_agent)
        if not card:
            return {"error": f"Agent {target_agent} not found"}

        # ACP: 包裹审计轨迹
        # A2A: 创建任务
        task = await self.task_manager.send_message(target_agent, message)
        return {"task": task}
```

### 第 7 步：运行演示

```python
async def main():
    registry = AgentRegistry()
    registry.register(AgentCard(name="researcher", description="搜索并总结",
                                 skills=[{"id": "web-research", "tags": ["research"]}]))
    registry.register(AgentCard(name="coder", description="写代码",
                                 skills=[{"id": "code-gen", "tags": ["coding"]}]))

    task_manager = TaskManager()

    # Agent Card 发现
    agents = registry.discover_by_skill_tag("research")
    print(f"发现 {len(agents)} 个智能体: {[a.name for a in agents]}")

    # 任务委派
    message = text_message("user", "研究 React 19 编译器特性")
    result = await task_manager.send_message("researcher", message)
    print(f"任务状态: {result.status}")

    # 完整审计日志
    print("协议演示完成")

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
  → 所有智能体在同一组织？ → A2A（Agent Card + 任务）
  → 共享基础设施？ → A2A + 消息代理
  → 跨组织？ → ANP + A2A（DID 验证）
```

### 4.2 生产模式

| 模式 | 说明 |
|------|------|
| A2A 最成熟 | Google 开源，Linux Foundation，Python/TypeScript SDK |
| ACP 正在合并到 A2A | 轨迹元数据概念被吸收 |
| ANP 最实验性 | 元协议协商是真正的新颖特性 |
| MCP 是工具标准 | 阶段 13 已涵盖 |

---

## 5. 工程最佳实践

### 5.1 协议设计原则

| 原则 | 说明 |
|------|------|
| 四协议解决不同问题 | MCP 工具、A2A 协作、ACP 审计、ANP 信任 |
| 结构原语保留 | 谁→谁、意图、内容、关联 ID |
| 轨迹元数据用于审计 | 每个响应携带推理链和工具调用 |
| DID 用于跨组织信任 | 去中心化身份 + E2EE |

### 5.2 中文场景特别建议

- **A2A 的 Agent Card 用中文描述**——技能名、描述用中文，但 tags 和 MIME 类型保持英文
- **ANP 的 DID 方法在中文环境中可用**——`did:wba` 基于域名，不依赖特定地理位置
- **ACP 的轨迹元数据对中文合规同样重要**——金融、医疗等受监管行业需要推理链审计

---

## 6. 常见错误

### 错误 1：只用一个协议

**现象：** "我们只用 MCP。"但智能体需要协作、审计、跨组织信任。

**原因：** MCP 解决智能体到工具的问题。智能体到智能体、审计、信任需要其他协议。

**修复：** 四协议解决不同问题。真实系统使用多个：MCP 工具 + A2A 协作 + ACP 审计 + ANP 信任。

### 错误 2：忽略轨迹元数据

**现象：** 智能体返回结果但没有记录推理链。审计时无法追溯"为什么"。

**原因：** ACP 的 TrajectoryMetadata 是可选的。生产系统需要它。

**修复：** 每个响应都包裹在轨迹元数据中——工具名称、输入、输出、推理步骤。

### 错误 3：不检查状态机约束

**现象：** 任务达到终态后仍然尝试更新。

**原因：** A2A 的 9 状态机有终态不可变约束。代码不检查就违反协议。

**修复：** 在产出事件之前检查终态。`TaskManager` 在终态后 break。

---

## 7. 面试考点

### Q1：MCP、A2A、ACP、ANP 分别解决什么问题？（难度：⭐）

**参考答案：**
- **MCP**：智能体到工具的通信——工具发现和调用
- **A2A**：智能体到智能体的协作——Agent Card 发现、任务生命周期、流式传输
- **ACP**：可审计通信——TrajectoryMetadata 记录推理链和工具调用
- **ANP**：跨组织信任——DID 身份验证、E2EE、元协议协商

### Q2：A2A 的 Agent Card 包含什么？任务生命周期有哪些状态？（难度：⭐⭐）

**参考答案：**
Agent Card：名称、描述、版本、技能列表（ID、标签、输入/输出 MIME）、能力（流式传输、推送通知）、安全方案。

任务生命周期 9 状态：submitted → working → input-required/auth-required → completed/failed/canceled/rejected。终态不可变。

### Q3：ACP 的 TrajectoryMetadata 为什么重要？（难度：⭐⭐）

**参考答案：**
每个响应携带推理步骤和工具调用的完整日志——哪个工具被调用、输入是什么、输出是什么、推理链是什么。

对受监管行业（金融、医疗、法律）这是金矿：每个回答都有可证明的推理链。没有黑箱。

### Q4：ANP 如何实现跨组织信任？（难度：⭐⭐⭐）

**参考答案：**
ANP 使用 W3C DID（`did:wba`）实现加密身份验证。每个智能体有 DID 文档，包含认证密钥和密钥协商密钥。

信任来自三源：域名级 TLS 验证 DID 文档主机、DID 密码学签名验证智能体身份、最小信任原则只授予最小权限。没有基于八卦的信任传播或 PageRank 评分。

元协议协商是真正的新颖特性：两个来自不同生态系统的智能体可以在没有预定义共享 Schema 的情况下用自然语言协商通信格式。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MCP | "AI 工具协议" | 智能体发现和使用工具的客户端-服务器协议 |
| A2A | "Google 的智能体协议" | Linux Foundation 下的点对点协作协议：Agent Card、9 状态任务、SSE 流式 |
| ACP | "企业智能体通信" | IBM/BeeAI 的 REST API，TrajectoryMetadata 携带推理链和工具调用 |
| ANP | "去中心化智能体身份" | DID 身份、E2EE、元协议协商 |
| Agent Card | "智能体的名片" | `/.well-known/agent-card.json` 上的 JSON 文档 |
| DID | "去中心化 ID" | W3C 密码学可验证身份标准 |
| TrajectoryMetadata | "审计收据" | ACP 为每个响应附加推理步骤和工具调用 |
| 元协议 | "智能体协商如何通信" | ANP 的自然语言格式协商 |

---

## 📚 小结

四个协议解决不同层级的问题：MCP（工具访问）、A2A（智能体间协作）、ACP（企业审计）、ANP（去中心化信任）。它们不是竞争关系——真实系统使用多个。A2A 最成熟（Linux Foundation），ACP 的轨迹元数据被吸收进 A2A，ANP 的元协议协商是真正的新颖特性。通信协议问题不是新问题——FIPA-ACL 在 2000 年就解决了大部分结构原语。

---

## ✏️ 练习

1. **【实现】** 多跳任务委派：扩展 `TaskManager` 使智能体处理器可以将子任务委派给其他智能体。研究者接收任务，委派"搜索"和"总结"子任务给两个专家智能体，等待两者完成，然后将结果合并到自己的工件中。

2. **【实现】** 添加 DID 轮转：智能体应能发布包含更新密钥的新 DID 文档，同时维持 `previousDid` 引用。验证者在宽限期内应接受来自当前和先前密钥的签名。

3. **【设计】** 实现 ANP 的元协议概念。两个智能体交换 `protocolNegotiation` 消息（候选格式："我能说 JSON-RPC"vs"我偏好 REST"）。最多 3 轮后达成格式一致或超时。商定的格式决定使用哪个 `TaskManager`。

4. **【思考】** 添加速率限制发现：`RateLimitedRegistry` 包装器缓存 Agent Card 查找并限制每个智能体每秒的发现查询。模拟 100 个智能体同时启动时的惊群效应并测量差异。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 四协议演示 | `code/main.py` | MCP/A2A/ACP/ANP 集成演示 |
| 技能提示词 | `outputs/prompt-protocol-selector.md` | 为你的系统选择协议的提示词 |

---

## 📖 参考资料

1. [官方] Google A2A 规范. https://github.com/google/A2A — 开源规范和 SDK（v1.0.0，Linux Foundation）
2. [官方] IBM/BeeAI ACP 规范. https://github.com/i-am-bee/acp — OpenAPI 3.1 规范
3. [GitHub] Agent Network Protocol. https://github.com/agent-network-protocol/AgentNetworkProtocol — DID 身份、E2EE、元协议协商
4. [文档] Model Context Protocol. https://modelcontextprotocol.io/ — Anthropic MCP 规范
5. [标准] W3C 去中心化标识符. https://www.w3.org/TR/did-core/ — ANP 的身份基础

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
