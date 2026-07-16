# 综合实践——构建完整的工具生态系统

> 第 13 章教授了每一块积木。这个综合实践将它们连接成一个生产级系统：一个暴露工具、资源、提示词和任务的 MCP Server，边缘 OAuth 2.1 认证，RBAC 网关，多服务器 Client，A2A 子智能体调用，OTel 追踪到收集器，CI 中的工具投毒检测，以及 AGENTS.md + SKILL.md 套件。完成时你能为每个架构决策辩护。

**类型：** 综合实践 | **语言：** Python | **前置知识：** 阶段 13 · 01-21 | **时间：** ~120 分钟
**所�阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 组合 MCP Server（工具 + 资源 + 提示词 + 任务 + UI）
- [ ] 集成 OAuth 2.1 认证 + RBAC 网关
- [ ] 构建多服务器 Client + A2A 子智能体调用
- [ ] 为每个架构决策辩护——为什么选择这个方案

---

## 1. 问题

第 13 章的前 21 课教了每一块积木。但积木不是系统。这个综合实践将所有积木连接成一个完整的生产级工具生态系统——你能说清楚为什么每个组件存在、为什么这样排列。

---

## 2. 概念

### 2.1 生产级工具生态系统的组件

```
用户
  ↓
[认证层] → OAuth 2.1 + PKCE
  ↓
[RABC 网关] → 权限控制 + 审计
  ↓
[MCP Client] → 多服务器合并
  ├── [MCP Server A] → 工具 + 资源 + 提示词
  ├── [MCP Server B] → 工具 + 任务
  └── [A2A 子智能体] → 复杂任务委托
  ↓
[OTel 追踪] → 端到端可观测性
  ↓
[CI 安全] → 工具投毒检测
  ↓
[AGENTS.md + SKILL.md] → 项目上下文 + 任务方法论
```

### 2.2 架构决策矩阵

| 决策 | 选择 | 理由 |
|------|------|------|
| 传输层 | Streamable HTTP | 远程部署 |
| 认证 | OAuth 2.1 + PKCE | 安全性 |
| 网关 | RBAC + 审计 | 企业控制 |
| 路由 | 多服务器合并 | 统一接口 |
| 追踪 | OTel GenAI | 可观测性 |
| 安全 | 哈希固定 + 投毒检测 | 多层防御 |

### 2.3 端到端数据流

```
用户查询 → Client → OAuth 令牌 → 网关 RBAC → 路由器
  ↓
MCP Server A (天气) + MCP Server B (笔记) + A2A (研究)
  ↓
OTel 追踪：每个 span 记录延迟、输入输出
  ↓
结果返回给用户
```

---

## 3. 从零实现

### Step 1：组合完整 MCP Server

```python
class ProductionMCPServer:
    """生产级 MCP Server——工具 + 资源 + 提示词 + 任务。"""
    def __init__(self, name):
        self.name = name
        self.tools = {}
        self.resources = {}
        self.prompts = {}

    def add_tool(self, name, description, schema, executor):
        self.tools[name] = {"description": description, "schema": schema, "executor": executor}

    def add_resource(self, uri, name, mime_type, reader):
        self.resources[uri] = {"uri": uri, "name": name, "mimeType": mime_type, "reader": reader}

    def add_prompt(self, name, description, arguments, template_fn):
        self.prompts[name] = {"name": name, "description": description,
                               "arguments": arguments, "template": template_fn}
```

### Step 2：网关集成

```python
class ProductionGateway:
    """生产级网关——认证 + RBAC + 审计。"""
    def __init__(self):
        self.servers = {}
        self.audit_log = []

    def add_server(self, name, server):
        self.servers[name] = server

    def call(self, user_role, tool_name, arguments):
        """带 RBAC 的工具调用。"""
        self.audit_log.append({"user": user_role, "tool": tool_name, "args": arguments})
        for server in self.servers.values():
            if tool_name in server.tools:
                return server.tools[tool_name]["executor"](**arguments)
        return {"error": "工具不存在"}
```

### Step 3：架构决策文档

```python
architecture_decisions = {
    "传输层": "Streamable HTTP — 支持远程部署和会话连续性",
    "认证": "OAuth 2.1 + PKCE — 临时令牌 + 资源限制",
    "网关": "RBAC + 审计 — 企业级权限控制",
    "工具投毒防御": "哈希固定 + 模式检测 + 长度限制",
    "追踪": "OTel GenAI — 端到端可观测性",
}
```

---

## 4. 工具

### 4.1 端到端组件清单

| 组件 | 课时 | 功能 |
|------|------|------|
| MCP Server | 07-08 | 工具、资源、提示词 |
| OAuth 2.1 | 16 | 认证和授权 |
| 网关 | 17 | RBAC、审计、速率限制 |
| A2A | 19 | 智能体间协作 |
| OTel | 20 | 追踪和监控 |
| 安全 | 15 | 投毒检测 |
| Skills/AGENTS.md | 22 | 项目上下文和方法论 |

### 4.2 架构决策辩护

每个组件必须有清晰的理由：

| 组件 | 为什么需要 | 如果去掉会怎样 |
|------|-----------|---------------|
| OAuth 2.1 | 安全性 | 任何人可以调用任何工具 |
| 网关 | 企业控制 | 无法管理权限和审计 |
| OTel 追踪 | 可观测性 | 出问题时无法调试 |
| 投毒检测 | 安全性 | 恶意 Server 可以攻击 |

---

## 5. 工程最佳实践

### 5.1 部署清单

- [ ] MCP Server 暴露工具 + 资源 + 提示词
- [ ] OAuth 2.1 认证已配置
- [ ] RBAC 网关已部署
- [ ] OTel 追踪已连接
- [ ] 工具投毒检测已集成到 CI
- [ ] AGENTS.md 和 SKILL.md 已创建

### 5.2 踩坑经验

- **组件间依赖**：网关依赖 OAuth——必须按正确顺序部署
- **配置一致性**：多个组件共享配置——需要集中管理
- **监控覆盖**：每个组件都需要自己的监控指标

---

## 7. 面试考点

### Q1：如果从零构建一个生产级 LLM 工具系统，你会如何设计架构？（难度：⭐⭐⭐）

**参考答案：**
(1) MCP Server 暴露工具/资源/提示词——标准化接口；(2) OAuth 2.1 认证——安全的授权流程；(3) RBAC 网关——企业级权限控制；(4) 多服务器 Client——统一工具命名空间；(5) OTel 追踪——端到端可观测性；(6) AGENTS.md + SKILL.md——智能体上下文和方法论；(7) CI 安全——工具投毒检测。每个组件都有明确的职责和存在的理由。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 端到端工具生态系统 | "完整工具栈" | MCP + OAuth + 网关 + A2A + OTel + 安全 + Skills 的完整组合 |
| 架构决策 | "为什么这样设计" | 每个组件必须有清晰的理由——不是随意堆砌 |
| 生产级 | "可以上线" | 有认证、授权、审计、追踪、安全的完整系统 |

---

## 📚 小结

第 13 章的综合实践：将所有积木连接成一个生产级工具生态系统。每个组件——MCP Server、OAuth、网关、A2A、OTel、安全、Skills——都有明确的职责和理由。理解每个决策的原因比知道如何实现更重要。

---

## ✏️ 练习

1. **【设计】** 为一个"智能客服"场景设计完整的工具生态系统架构图
2. **【实现】** 构建一个最小生产级 MCP Server——包含认证、追踪、投毒检测
3. **【辩护】** 为每个架构决策写出理由——为什么选择 Streamable HTTP 而不是 stdio？为什么用 OAuth 而不是 API Key？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 完整工具生态系统 | `code/main.py` | 端到端 MCP Server + 网关 + 认证 |
| 架构决策文档 | `outputs/architecture-decisions.md` | 每个组件的选择理由 |

---

## 📖 参考资料

1. [文档] MCP 规范: https://spec.modelcontextprotocol.io
2. [文档] A2A 规范: https://google.github.io/A2A/
3. [文档] OTel GenAI: https://opentelemetry.io/docs/specs/gen-ai/
4. [文档] Anthropic Agent Skills: https://docs.anthropic.com/en/docs/agents/skills

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
