# MCP 生产认证——注册、JWKS 刷新、受众绑定令牌

> 第 16 课在内存中搭建了 OAuth 2.1 状态机。到 2026 年，你部署到真实组织的每个 MCP Server 都在生产认证之后：支持无界客户端数量的客户端注册（Client ID Metadata Documents 作为推荐默认，动态客户端注册作为向后兼容的后备）、授权服务器元数据发现（RFC 8414 或 OpenID Connect Discovery）、不中断凌晨 3 点令牌验证的 JWKS 缓存刷新、以及拒绝跨资源重放的受众绑定令牌。本课用三个角色建模完整表面——授权服务器、资源服务器（MCP Server）、客户端。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 16（OAuth 2.1）、17（网关）| **时间：** ~90 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现 JWKS（JSON Web Key Set）缓存和定期刷新
- [ ] 设计 Client ID Metadata Documents（CIMD）注册机制
- [ ] 理解受众绑定令牌如何防止跨资源重放
- [ ] 构建完整的 MCP 生产认证流水线

---

## 1. 问题

第 16 课搭建了 OAuth 2.1 状态机——但那是内存中的原型。生产环境需要：无界客户端注册、授权服务器发现、JWKS 缓存刷新、受众绑定令牌。这些是"凌晨 3 点还在运行"的可靠性保障。

---

## 2. 概念

### 2.1 生产认证架构

```
Client → [元数据发现] → 授权服务器
    ↓
Client → [CIMD/DCR] → 客户端注册
    ↓
Client → [授权码 + PKCE] → 令牌
    ↓
Client → [资源服务器] → 携带令牌访问
    ↓
资源服务器 → [JWKS 验证] → 确认令牌有效
```

### 2.2 三种注册机制

| 机制 | 说明 | 规范状态 |
|------|------|---------|
| **CIMD** | Client ID Metadata Documents | 推荐（2025-11-25） |
| **DCR** | Dynamic Client Registration | 向后兼容（MAY） |
| **静态注册** | 手动配置 | 企业内部 |

### 2.3 JWKS 缓存

JSON Web Key Set 包含授权服务器的公钥——用于验证 JWT 令牌的签名。缓存策略：

- 首次加载：GET 请求获取 JWKS
- 缓存：设置合理的 TTL（如 1 小时）
- 刷新：令牌验证失败时重新获取（`Cache-Control: no-cache`）

### 2.4 受众绑定

JWT 令牌的 `aud` 声明绑定到特定 MCP Server——防止令牌被重放到其他 Server：

```json
{
  "aud": "https://my-mcp-server.com/mcp",
  "scope": "tools:call resources:read"
}
```

---

## 3. 从零实现

### Step 1：JWKS 缓存

```python
import time
import hashlib

class JWKSProvider:
    """JWKS 缓存和刷新。"""
    def __init__(self):
        self.jwks = {}
        self.last_fetch = 0
        self.ttl = 3600  # 1 小时

    def get_key(self, kid):
        """获取签名密钥。"""
        if time.time() - self.last_fetch > self.ttl:
            self._refresh()
        return self.jwks.get(kid)

    def _refresh(self):
        """刷新 JWKS。"""
        self.jwks = {"key-1": {"kty": "RSA", "kid": "key-1", "use": "sig"}}
        self.last_fetch = time.time()
        print("JWKS 已刷新")


def verify_jwt(token, jwks):
    """验证 JWT 令牌（简化版）。"""
    # 实际中使用 JWT 解码库
    return {"valid": True, "message": "令牌验证通过"}
```

### Step 2：受众绑定验证

```python
def verify_audience(token_aud, server_aud):
    """验证 JWT 令牌的受众。"""
    if token_aud != server_aud:
        return False, f"受众不匹配: 令牌={token_aud}, 服务器={server_aud}"
    return True, "受众匹配"
```

### Step 3：完整认证流水线

```python
def mcp_auth_flow(client, auth_server, resource_server):
    """MCP 生产认证流程。"""
    # 1. 元数据发现
    discovery = auth_server.get_metadata()
    print(f"授权服务器: {discovery['issuer']}")

    # 2. 客户端注册
    client_info = client.register(auth_server)
    print(f"客户端 ID: {client_info['client_id']}")

    # 3. 授权码 + PKCE
    auth_result = client.authorize(auth_server)
    print(f"授权码: {auth_result['code'][:20]}...")

    # 4. 获取令牌
    token = client.exchange_token(auth_server, auth_result["code"])
    print(f"令牌: {token['access_token'][:20]}...")

    # 5. 访问资源服务器
    response = resource_server.access(token)
    print(f"响应: {response['status']}")
```

---

## 4. 工具

### 4.1 JWT 验证库

```python
# PyJWT
import jwt
decoded = jwt.decode(token, key, algorithms=["RS256"])
print(decoded["aud"])  # 受众
print(decoded["scope"])  # 权限范围
```

### 4.2 JWKS 客户端

```python
import requests

def fetch_jwks(jwks_url):
    """获取 JWKS 公钥集。"""
    response = requests.get(jwks_url)
    return response.json()["keys"]
```

---

## 5. 工程最佳实践

### 5.1 JWKS 缓存策略

- **TTL**：1 小时——平衡新鲜度和延迟
- **故障恢复**：令牌验证失败时自动刷新
- **多密钥支持**：支持密钥轮换——旧密钥在宽限期内仍有效

### 5.2 踩坑经验

- **CIMD 未配置**：Client 无法自动发现授权服务器——使用 DCR 作为后备
- **JWKS 刷新失败**：凌晨 3 点的密钥轮换导致验证失败——实现重试和降级
- **受众不匹配**：令牌被重放到其他 Server——总是验证 `aud` 声明

---

## 6. 常见错误

### 错误 1：JWKS 缓存 TTL 过长

**现象：** 密钥轮换后旧令牌仍然有效——安全漏洞。

**修复：** TTL 设为 1 小时，令牌验证失败时立即刷新。

### 错误 2：未验证受众绑定

**现象：** 令牌被重放到其他 MCP Server——权限提升。

**修复：** 总是验证 JWT 的 `aud` 声明与当前 Server 匹配。

---

## 7. 面试考点

### Q1：JWKS 缓存为什么需要在令牌验证失败时立即刷新？（难度：⭐⭐）

**参考答案：**
密钥轮换时，新签名的令牌需要新密钥验证。如果 JWKS 缓存还没过期，旧密钥无法验证新令牌——所有合法请求都会失败。实现"失败时立即刷新"确保密钥轮换不会中断服务。这是"凌晨 3 点还在运行"的关键可靠性保障。

### Q2：CIMD 和 DCR 有什么区别？（难度：⭐⭐⭐）

**参考答案：**
CIMD（Client ID Metadata Documents）是 2025-11-25 规范推荐的默认注册机制——Client 在首次连接时发送元数据文档，Server 验证并返回 Client ID。DCR（Dynamic Client Registration）是向后兼容的后备——Client 通过 OAuth 端点动态注册。CIMD 更安全（Server 可以预先审核），DCR 更灵活（支持无预先配置的 Client）。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| JWKS | "公钥集" | JSON Web Key Set——包含授权服务器的公钥，用于验证 JWT 签名 |
| CIMD | "客户端注册文档" | Client ID Metadata Documents——MCP 的推荐客户端注册机制 |
| 受众绑定 | "令牌锁定资源" | JWT 的 `aud` 声明绑定到特定 MCP Server——防止跨资源重放 |
| 元数据发现 | "找到授权服务器" | Client 自动发现授权服务器的 URL 和支持的能力 |

---

## 📚 小结

MCP 生产认证 = Client 注册（CIMD/DCR）+ 授权服务器发现 + JWKS 缓存刷新 + 受众绑定令牌。CIMD 是推荐的注册机制，JWKS 需要定期刷新，受众绑定防止令牌重放。这些是"凌晨 3 点还在运行"的可靠性保障。

---

## ✏️ 练习

1. **【实现】** 实现 JWKS 缓存——支持 TTL 刷新和失败时自动重试
2. **【设计】** 为多 Server 架构设计令牌验证流程——如何确保令牌不被重放

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| JWKS 缓存 | `code/main.py` | JWKS 缓存 + 受众绑定验证 |

---

## 📖 参考资料

1. [文档] MCP 认证规范 2025-11-25: https://spec.modelcontextprotocol.io
2. [RFC] RFC 8414 授权服务器元数据: https://datatracker.ietf.org/doc/html/rfc8414
3. [RFC] RFC 7517 JWK: https://datatracker.ietf.org/doc/html/rfc7517

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
