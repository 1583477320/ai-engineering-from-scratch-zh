# MCP 传输层——stdio vs Streamable HTTP vs SSE 迁移

> stdio 只在本地工作。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输在 2026 年中被弃用并移除。选错传输意味着需要迁移；选对传输则买到一个支持会话连续性和 DNS 重绑定保护的远程托管 MCP Server。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 13 · 07-08（MCP Server 和 Client）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 根据部署形态（本地 vs 远程、单进程 vs 集群）在 stdio 和 Streamable HTTP 之间选择
- [ ] 实现 Streamable HTTP 端点——处理请求、响应和会话管理
- [ ] 理解 SSE 传输的弃用原因和迁移路径
- [ ] 设计 MCP Server 的传输层架构

---

## 1. 问题

MCP 的传输层决定了 Client 和 Server 如何通信。三种选择：stdio（本地进程通信）、SSE（旧的远程标准，2026 年弃用）、Streamable HTTP（新的远程标准）。

选错传输意味着重新架构；选对传输意味着轻松扩展到远程部署。

---

## 2. 概念

### 2.1 三种传输对比

| 传输 | 通信方式 | 适用场景 | 2026 状态 |
|------|---------|---------|----------|
| **stdio** | 进程间管道 | 本地 Client | ✅ 活跃 |
| **SSE** | HTTP + Server-Sent Events | 远程（旧） | ⚠️ 弃用中 |
| **Streamable HTTP** | HTTP + 可选 SSE | 远程（新） | ✅ 推荐 |

### 2.2 选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| 本地应用（单机） | stdio | 最简单，无网络延迟 |
| 远程服务（单机） | Streamable HTTP | 支持会话连续性 |
| 云端部署（集群） | Streamable HTTP | DNS 安全、可扩展 |
| 已有 SSE 服务 | Streamable HTTP | SSE 即将弃用 |

### 2.3 Streamable HTTP 架构

```
Client → HTTP POST → Server
         ↕
    SSE 流（可选）
```

关键特性：
- **会话连续性**：通过 `Mcp-Session-Id` 头维护会话
- **DNS 重绑定保护**：`Mcp-Session-Id` 防止跨域请求
- **幂等性**：相同请求 ID 返回相同结果

---

## 3. 从零实现

### Step 1：Streamable HTTP 端点（简化版）

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/mcp", methods=["POST"])
def handle_mcp():
    """MCP Streamable HTTP 端点。"""
    data = request.json
    method = data.get("method")
    params = data.get("params", {})
    req_id = data.get("id")

    # 会话验证
    session_id = request.headers.get("Mcp-Session-Id")
    if not session_id:
        return jsonify({"error": "缺少 Mcp-Session-Id"}), 401

    # 处理请求
    if method == "initialize":
        return jsonify({
            "jsonrpc": "2.0",
            "result": {"protocolVersion": "2025-11-25", "capabilities": {}},
            "id": req_id
        })

    return jsonify({"jsonrpc": "2.0", "result": {"status": "ok"}, "id": req_id})
```

### Step 2：SSE 传输（弃用）

```python
# 旧的 SSE 传输——2026 年中弃用
@app.route("/sse", methods=["GET"])
def handle_sse():
    """SSE 端点——已弃用。"""
    return "SSE 传输已弃用，请迁移到 Streamable HTTP", 410
```

---

## 4. 工具

### 4.1 MCP Python SDK

```python
# stdio 传输（本地）
from mcp.client import stdio_client

# Streamable HTTP 传输（远程）
from mcp.client import streamable_http_client
```

### 4.2 传输选择

| SDK | stdio | Streamable HTTP | SSE |
|-----|-------|-----------------|-----|
| Python SDK | ✅ | ✅ | ❌ 弃用 |
| TypeScript SDK | ✅ | ✅ | ❌ 弃用 |

---

## 5. 工程最佳实践

### 5.1 传输层设计

- **本地优先用 stdio**：最简单、无网络延迟
- **远程用 Streamable HTTP**：会话连续性、DNS 安全
- **不要用 SSE**：2026 年中弃用，迁移到 Streamable HTTP

### 5.2 会话管理

- 使用 `Mcp-Session-Id` 头维护会话状态
- 会话超时后自动清理资源
- 支持多客户端并发访问同一 Server

### 5.3 踩坑经验

- **SSE 弃用**：2026 年中移除——不要在新项目中使用
- **DNS 重绑定**：Streamable HTTP 的 `Mcp-Session-Id` 防止跨域攻击
- **长连接超时**：Streamable HTTP 连接可能被代理超时——设置合理的 keep-alive

---

## 6. 常见错误

### 错误 1：在新项目中使用 SSE 传输

**现象：** 项目完成后发现 SSE 被弃用——需要重写传输层。

**修复：** 始终使用 Streamable HTTP——它是 2026 年的远程标准。

### 错误 2：忽略会话连续性

**现象：** 多轮对话中状态丢失——每个请求都是独立的。

**修复：** 使用 `Mcp-Session-Id` 头维护会话状态。

---

## 7. 面试考点

### Q1：stdio 和 Streamable HTTP 有什么区别？（难度：⭐⭐）

**参考答案：**
stdio 通过进程间管道通信——只支持本地 Client，无网络延迟。Streamable HTTP 通过 HTTP 请求通信——支持远程 Client，支持会话连续性（Mcp-Session-Id），支持 DNS 重绑定保护。选择：本地应用用 stdio，远程/云端部署用 Streamable HTTP。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| stdio | "进程通信" | 通过标准输入/输出管道的 MCP 传输——只支持本地 |
| Streamable HTTP | "远程传输" | 基于 HTTP 的 MCP 传输——支持会话连续性和 DNS 安全 |
| SSE | "旧远程标准" | Server-Sent Events——2026 年中弃用 |
| Mcp-Session-Id | "会话标识" | Streamable HTTP 中维护会话状态的 HTTP 头 |

---

## 📚 小结

MCP 传输层有三种：stdio（本地）、SSE（远程，弃用）、Streamable HTTP（远程，推荐）。选对传输决定扩展性——stdio 最简单但只支持本地，Streamable HTTP 支持远程部署和会话连续性。2026 年不要使用 SSE——迁移到 Streamable HTTP。

---

## ✏️ 练习

1. **【对比】** 本地 stdio 和远程 Streamable HTTP 的延迟差异——在同一台机器上测量
2. **【设计】** 为一个生产级 MCP Server 设计传输层——需要支持 100+ 并发 Client

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| Streamable HTTP 端点 | `code/main.py` | HTTP 端点 + 会话管理 |

---

## 📖 参考资料

1. [文档] MCP 传输层规范: https://spec.modelcontextprotocol.io
2. [文档] Streamable HTTP: https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/
3. [文档] MCP SSE 弃用公告: https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1577

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
