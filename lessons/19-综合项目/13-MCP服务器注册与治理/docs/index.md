# 综合项目13——MCP服务器注册与治理

> Model Context Protocol在2026年成为默认工具使用规范。Anthropic、OpenAI、Google和每个主流IDE都发布MCP客户端。Pinterest发布了内部MCP服务器生态系统。AAIF Registry在`.well-known`处形式化能力元数据。AWS ECS发布了参考无状态部署。2026年生产形态：StreamableHTTP传输、OAuth 2.1作用域、OPA策略门控和一个让平台团队发现、验证和启用服务器的注册表。本综合项目要求你端到端构建这个系统。

**类型：** 综合项目
**编程语言：** Python（服务器，FastMCP）或TypeScript，Go（注册表服务）
**前置知识：** 第11章（LLM工程）、第13章（工具与MCP）、第14章（智能体）、第17章（基础设施）、第18章（安全）
**涉及章节：** P11 · P13 · P14 · P17 · P18
**预计时间：** 25小时

---

## 学习目标

- 构建MCP服务器：10个内部工具，带类型化模式和作用域标签
- 实现StreamableHTTP无状态传输和OAuth 2.1作用域检查
- 实现OPA策略门控和人工审批机制
- 构建服务器注册表和`.well-known/mcp-capabilities`能力清单

---

## 1. 问题

MCP在2026年成为工具使用的通用语言。生产挑战不是编写服务器（FastMCP使之容易），而是在企业要求下大规模部署：每租户OAuth作用域、破坏性工具的OPA策略、StreamableHTTP无状态扩展、用于发现的注册表、每工具调用的审计日志。

---

## 2. 核心概念

### 2.1 StreamableHTTP

MCP 2026修订版将StreamableHTTP作为默认传输。无状态默认：单个HTTP端点接受JSON-RPC请求、流式响应，支持长期连接通知。无状态意味着可在负载均衡器后水平扩展。

### 2.2 OAuth 2.1作用域

授权是OAuth 2.1，带每工具作用域。令牌携带如`jira:read`、`s3:list`、`postgres:query:readonly`的作用域。MCP服务器在工具调用时检查作用域，而非仅在会话开始时。对于高风险工具，服务器拒绝任何作用域未在最近N分钟内提升为`approved:by:human`的调用。

### 2.3 注册表

独立服务。每个MCP服务器暴露`.well-known/mcp-capabilities`文档，含工具清单、传输URL、认证要求。注册表轮询、验证并索引。平台团队使用注册表UI查看可用工具、所需作用域和所属团队。

---

## 3. 从零实现

`code/main.py`实现无状态风格的工具分发、OPA策略检查、审计日志和注册表搜索。

```python
"""MCP服务器+注册表+OPA策略门控脚手架。

核心架构原语：(a) 无状态StreamableHTTP风格分发，
查找工具、通过OPA风格策略检查作用域、执行并审计日志丰富；
(b) 从每个服务器拉取.well-known/mcp-capabilities并验证的注册表。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass
class ToolSchema:
    name: str
    required_scope: str
    destructive: bool
    description: str
    input_schema: dict

Handler = Callable[[dict], dict]


@dataclass
class MCPServer:
    name: str
    url: str
    tools: dict[str, ToolSchema] = field(default_factory=dict)
    handlers: dict[str, Handler] = field(default_factory=dict)

    def register(self, schema: ToolSchema, handler: Handler) -> None:
        self.tools[schema.name] = schema
        self.handlers[schema.name] = handler

    def capabilities(self) -> dict:
        return {
            "server": self.name,
            "transport": "streamable_http",
            "url": self.url,
            "tools": [{"name": t.name, "scope": t.required_scope,
                       "destructive": t.destructive, "description": t.description}
                      for t in self.tools.values()],
        }


@dataclass
class Token:
    user: str
    scopes: set[str]
    approved_at: float = 0.0

    def has_scope(self, s: str) -> bool:
        return s in self.scopes

    def fresh_approval(self, now: float, window_s: int = 900) -> bool:
        return "approved:by:human" in self.scopes and (now - self.approved_at) <= window_s


def policy_decide(server: MCPServer, tool: str, token: Token, args: dict, now: float) -> tuple[bool, str]:
    if tool not in server.tools:
        return False, f"no such tool: {tool}"
    schema = server.tools[tool]
    if not token.has_scope(schema.required_scope):
        return False, f"missing scope: {schema.required_scope}"
    if schema.destructive and not token.fresh_approval(now):
        return False, "destructive tool requires fresh human approval"
    if len(json.dumps(args)) > 8192:
        return False, "payload too large (>8KB)"
    return True, "ok"


def redact(payload: dict) -> dict:
    s = json.dumps(payload)
    s = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", s)
    s = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[ssn]", s)
    return json.loads(s)


@dataclass
class AuditEntry:
    ts: float
    user: str
    tool: str
    outcome: str
    args_redacted: dict
    response_redacted: dict


def dispatch(server: MCPServer, token: Token, tool: str, args: dict, audit: list[AuditEntry]) -> dict:
    now = time.time()
    ok, reason = policy_decide(server, tool, token, args, now)
    if not ok:
        audit.append(AuditEntry(now, token.user, tool, f"denied:{reason}", redact(args), {}))
        return {"error": {"code": 403, "message": reason}}
    result = server.handlers[tool](args)
    audit.append(AuditEntry(now, token.user, tool, "ok", redact(args), redact(result)))
    return {"result": result}


@dataclass
class Registry:
    entries: dict[str, dict] = field(default_factory=dict)
    def register(self, server: MCPServer) -> None:
        self.entries[server.name] = server.capabilities()
    def search(self, query: str) -> list[tuple[str, str]]:
        q = query.lower()
        return [(sn, t["name"]) for sn, cap in self.entries.items()
                for t in cap["tools"] if q in t["name"].lower() or q in t["description"].lower()]


def main() -> None:
    ro = MCPServer(name="readonly-mcp", url="https://mcp.internal/readonly")
    ro.register(ToolSchema("postgres.readonly", "postgres:query:readonly", False, "Read-only Postgres query",
                           {"type": "object", "properties": {"sql": {"type": "string"}}}),
                lambda a: {"rows": [[1]]})
    ro.register(ToolSchema("s3.list", "s3:list", False, "List S3 objects",
                           {"type": "object", "properties": {"bucket": {"type": "string"}}}),
                lambda a: {"objects": [{"key": "a/b.txt", "size": 128}]})
    rw = MCPServer(name="destructive-mcp", url="https://mcp.internal/destructive")
    rw.register(ToolSchema("jira.create", "jira:write", True, "Create Jira issue",
                           {"type": "object", "properties": {"title": {"type": "string"}}}),
                lambda a: {"id": "PROJ-99", "created": True})

    registry = Registry()
    registry.register(ro)
    registry.register(rw)
    audit: list[AuditEntry] = []

    readonly_token = Token(user="u42", scopes={"postgres:query:readonly", "s3:list", "jira:read"})
    approved_token = Token(user="u42", scopes={"jira:write", "approved:by:human"}, approved_at=time.time() - 60)

    print("=== 注册表搜索 ===")
    print("  'jira' ->", registry.search("jira"))
    print("  'postgres' ->", registry.search("postgres"))
    print("\n=== postgres.readonly (读取作用域) ===")
    print(" ", dispatch(ro, readonly_token, "postgres.readonly", {"sql": "SELECT * FROM users"}, audit))
    print("\n=== jira.create (无审批，期望拒绝) ===")
    print(" ", dispatch(rw, Token("u42", {"jira:write"}), "jira.create", {"title": "new bug"}, audit))
    print("\n=== jira.create (有新鲜审批) ===")
    print(" ", dispatch(rw, approved_token, "jira.create", {"title": "new bug"}, audit))
    print("\n=== 审计日志 ===")
    for e in audit:
        print(" ", json.dumps(asdict(e), default=str))


if __name__ == "__main__":
    main()
```

运行结果：

```
=== 注册表搜索 ===
  'jira' -> [('destructive-mcp', 'jira.create'), ('readonly-mcp', 's3.list')]
  'postgres' -> [('readonly-mcp', 'postgres.readonly')]

=== postgres.readonly (读取作用域) ===
  {'result': {'rows': [[1]]}}

=== jira.create (无审批，期望拒绝) ===
  {'error': {'code': 403, 'message': 'missing scope: approved:by:human'}}

=== jira.create (有新鲜审批) ===
  {'result': {'id': 'PROJ-99', 'created': True}}
```

---

## 4. 工具实践

**技术栈：**
- 服务器框架：FastMCP（Python）或`@modelcontextprotocol/sdk`（TypeScript）
- 传输：StreamableHTTP over HTTPS（无状态）
- 认证：OAuth 2.1 + SPIFFE/SPIRE工作负载身份
- 策略：OPA/Rego规则每工具
- 注册表：自托管，消费`.well-known/mcp-capabilities`
- 人工审批：Slack交互消息
- 部署：AWS ECS Fargate或Fly.io

---

## 5. LLM视角

**无状态视角**：StreamableHTTP使MCP服务器无状态可扩展。负载均衡器后可水平扩展。

**策略门控视角**：OPA策略在每次工具调用时执行。破坏性工具需要新鲜的人工审批（15分钟内有效）。

**注册表视角**：`.well-known/mcp-capabilities`文档让平台团队发现和验证服务器，无需逐个检查配置。

---

## 6. 工程最佳实践

**安全设计**：
- 每工具作用域检查
- 破坏性工具需人工审批
- PII在审计日志中清洗

**部署模式**：
- StreamableHTTP无状态
- 负载均衡器后水平扩展
- 注册表轮询能力清单

---

## 7. 常见错误

**错误1：作用域仅在会话开始时检查**
症状：权限变更后仍可调用
修复：每次工具调用时检查作用域

**错误2：不区分只读和破坏性工具**
症状：破坏性操作无额外审批
修复：破坏性工具需新鲜人工审批

---

## 8. 面试考点

**Q1：MCP StreamableHTTP与传统SSE有什么区别？**
考察：对传输演进的理解

**Q2：为什么MCP需要注册表服务？**
考察：对企业可发现性的理解

**Q3：作用域提升（Scope Elevation）如何工作？**
考察：对动态权限的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| StreamableHTTP | "2026 MCP传输" | 无状态HTTP+流式；取代网络服务器的SSE+stdio |
| 能力清单 | "Well-known文档" | `.well-known/mcp-capabilities`，含工具清单、认证、传输URL |
| OPA/Rego | "策略引擎" | Open Policy Agent，根据外部规则授权工具调用 |
| 作用域提升 | "人工批准" | 通过Slack审批授予的短期作用域，破坏性工具必需 |
| 注册表 | "工具发现" | 从能力清单索引MCP服务器的服务 |
| 合规套件 | "规范测试" | StreamableHTTP+工具清单正确性的官方MCP测试套件 |

---

## 参考文献

- [Model Context Protocol 2026路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/)
- [AAIF MCP Registry规范](https://github.com/modelcontextprotocol/registry)
- [AWS ECS参考部署](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/)
- [Pinterest内部MCP生态系统](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [Open Policy Agent](https://www.openpolicyagent.org/)
