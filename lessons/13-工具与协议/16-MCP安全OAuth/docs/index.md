# MCP 安全 II——OAuth 2.1、资源指示器、增量权限

> 远程 MCP Server 需要授权而不仅仅是认证。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示器（RFC 8707）+ 受保护资源元数据（RFC 9728）对齐。SEP-835 在 403 WWW-Authenticate 上添加了增量权限同意和渐进授权。本课将渐进授权流程实现为状态机，以便看到每一步。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 09（传输层）、15（安全 I）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分资源服务器和授权服务器的职责
- [ ] 实现 MCP 的 OAuth 2.1 + PKCE 授权流程
- [ ] 理解资源指示器（RFC 8707）如何限制 OAuth scope
- [ ] 实现渐进授权——在 403 响应时请求额外权限

---

## 1. 问题

远程 MCP Server 需要授权——但标准 OAuth 不理解 MCP 的工具/资源原语。SEP-835 与 OAuth 2.1 对齐，添加了资源指示器和渐进授权。

**核心问题：** Server 可以在用户未明确同意的情况下请求特定工具的访问权限——需要一个机制让用户逐步授权。

---

## 2. 概念

### 2.1 MCP OAuth 架构

```
Client → 授权服务器（OAuth 2.1）→ 授权码
    ↓
Client → 资源服务器（MCP Server）→ 携带令牌访问工具
    ↓
MCP Server → 验证令牌 + 检查 scope
```

### 2.2 OAuth 2.1 + PKCE

MCP 使用 OAuth 2.1 的 PKCE（Proof Key for Code Exchange）扩展——防止授权码劫持：

```
1. Client 生成 code_verifier 和 code_challenge
2. Client 重定向到授权服务器，携带 code_challenge
3. 用户授权后，服务器返回授权码
4. Client 用 code_verifier + code_code 交换令牌
```

### 2.3 资源指示器（RFC 8707）

OAuth scope 中指定资源范围——限制令牌只适用于特定 MCP Server：

```
scope = "tools:read tools:call resources:read"
resource = "https://my-mcp-server.com/mcp"
```

### 2.4 渐进授权（SEP-835）

当 Server 返回 403（权限不足）时，可以请求额外授权：

```
Server → Client: 403 + WWW-Authenticate: Bearer resource_metadata="..."
    ↓
Client → 授权服务器：请求扩展 scope
    ↓
用户授权新权限
    ↓
Client → Server：携带更新的令牌访问
```

---

## 3. 从零实现

### Step 1：OAuth 状态机

```python
class MCPAuthState:
    """MCP OAuth 2.1 状态机。"""
    STATES = ["unauthenticated", "authorization_required", "authorized", "error"]

    def __init__(self):
        self.state = "unauthenticated"
        self.tokens = {}
        self.code_verifier = None

    def start_authorization(self):
        """启动 PKCE 授权流程。"""
        self.code_verifier = secrets.token_urlsafe(32)
        code_challenge = hashlib.sha256(self.code_verifier.encode()).hexdigest()
        self.state = "authorization_required"
        return {"authorization_url": f"https://auth.example.com/authorize?code_challenge={code_challenge}"}

    def handle_callback(self, code):
        """处理授权回调。"""
        self.state = "authorized"
        self.tokens = {"access_token": f"token_{code[:8]}"}
        return self.tokens
```

### Step 2：资源指示器

```python
def make_oauth_request(token, resource_uri, scope):
    """构建带资源指示器的 OAuth 请求。"""
    return {
        "headers": {
            "Authorization": f"Bearer {token}",
            "Resource": resource_uri,
        },
        "scope": scope,
    }
```

---

## 4. 工具

### 4.1 MCP 规范

| 规范 | 内容 |
|------|------|
| OAuth 2.1 | MCP 的授权框架 |
| RFC 8707 | 资源指示器——限制 OAuth scope 到特定资源 |
| RFC 9728 | 受保护资源元数据 |
| SEP-835 | 渐进授权——403 时请求额外权限 |

### 4.2 最佳实践

- **最小权限**：只请求必要的 scope
- **渐进授权**：首次请求基础权限，需要时再扩展
- **令牌刷新**：实现自动刷新——避免用户体验中断

---

## 5. 工程最佳实践

### 5.1 授权流程设计

```
1. Client 检测 Server 需要 OAuth
2. Client 重定向到授权服务器（PKCE）
3. 用户授权（最小权限）
4. Client 获取令牌
5. Client 访问 MCP Server（携带令牌）
6. 如果 403 → 请求扩展 scope → 重复
```

### 5.2 踩坑经验

- **Scope 过大**：请求了不需要的权限——用户不信任
- **Scope 过小**：频繁触发 403——用户体验差
- **令牌过期**：实现自动刷新——否则用户需要重新授权

---

## 6. 常见错误

### 错误 1：没有实现资源指示器

**现象：** Server 请求了所有工具的访问权限——用户不信任。

**修复：** 使用 RFC 8707 资源指示器——只请求必要工具的权限。

### 错误 2：忽略 403 响应中的渐进授权

**现象：** Server 返回 403 但 Client 没有请求额外权限。

**修复：** Client 解析 `WWW-Authenticate` 头中的资源元数据，请求扩展 scope。

---

## 7. 面试考点

### Q1：MCP 为什么使用 OAuth 2.1 + PKCE 而不是简单的 API Key？（难度：⭐⭐）

**参考答案：**
API Key 是静态的——如果泄露，攻击者可以无限期访问。OAuth 2.1 + PKCE 提供：(1) 临时令牌——过期后需要重新授权；(2) Scope 限制——令牌只适用于特定工具/资源；(3) 资源指示器——限制令牌到特定 MCP Server；(4) 渐进授权——用户可以逐步授予权限。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| OAuth 2.1 | "MCP 的授权标准" | MCP 远程传输使用的授权框架——基于 OAuth 2.1 |
| PKCE | "授权码保护" | 防止授权码劫持的扩展——Client 证明自己拥有 code_verifier |
| 资源指示器 | "限制权限范围" | RFC 8707——OAuth scope 指定特定资源（如特定 MCP Server） |
| 渐进授权 | "逐步获取权限" | 403 响应时请求额外 scope——用户逐步授予更多权限 |

---

## 📚 小结

MCP 远程部署需要 OAuth 2.1 + PKCE 授权。资源指示器（RFC 8707）限制令牌到特定 Server。渐进授权（SEP-835）让 Server 在 403 时请求额外权限——用户逐步授予。关键是：最小权限、渐进授权、令牌刷新。

---

## ✏️ 练习

1. **【实现】** 模拟 MCP OAuth 2.1 + PKCE 授权流程
2. **【设计】** 为一个多 Server 架构设计渐进授权策略

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| OAuth 状态机 | `code/main.py` | PKCE + 资源指示器 + 渐进授权 |

---

## 📖 参考资料

1. [文档] OAuth 2.1: https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1-07
2. [文档] RFC 8707 资源指示器: https://datatracker.ietf.org/doc/html/rfc8707
3. [文档] SEP-835: https://github.com/modelcontextprotocol/modelcontextprotocol/issues/835

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
