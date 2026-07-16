# MCP 基础——原语、生命周期、JSON-RPC 基础

> MCP 之前的每个集成都是一次性的。模型上下文协议（Model Context Protocol）——2024 年 11 月由 Anthropic 首次发布，现由 Linux 基金会的 Agentic AI Foundation 管理——标准化了发现和调用，使任何 Client 都能与任何 Server 对话。2025-11-25 规范命名了六个原语（三个服务器端、三个客户端）、三阶段生命周期和 JSON-RPC 2.0 线格式。学会这些，MCP 章节的其余内容就变成了阅读。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 13 · 01-05（工具接口与函数调用）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 命名所有六个 MCP 原语（服务器端三个、客户端三个）并给出一个用例
- [ ] 描述 MCP 的三阶段生命周期——初始化、能力协商、操作
- [ ] 解释 JSON-RPC 2.0 线格式在 MCP 中的作用
- [ ] 区分 MCP 和直接 API 调用——MCP 的额外价值是什么

---

## 1. 问题

每个 LLM 应用的工具集成都是独立的——为 OpenAI 写的工具代码不能直接给 Anthropic 用。MCP 解决了这个问题：**一个标准协议，任何 Client 都能与任何 Server 对话。**

---

## 2. 概念

### 2.1 六个 MCP 原语

**服务器端原语：**

| 原语 | 说明 | 用例 |
|------|------|------|
| **Tools** | 可执行的函数 | `get_weather(city)` |
| **Resources** | 可读取的数据 | 文件内容、API 文档 |
| **Prompts** | 预定义的提示词模板 | 代码审查模板 |

**客户端原语：**

| 原语 | 说明 | 用例 |
|------|------|------|
| **Roots** | 文件系统根目录 | 代码库路径 |
| **Sampling** | 请求 LLM 生成 | 代码审查时的生成请求 |
| **Elicitation** | 请求用户输入 | 需要确认的操作 |

### 2.2 三阶段生命周期

```
1. 初始化: Client 发送 initialize，Server 返回能力列表
2. 能力协商: 双方确认支持的原语和版本
3. 操作: Client 发送 tools/call、resources/read 等请求
```

### 2.3 JSON-RPC 2.0 线格式

MCP 使用 JSON-RPC 2.0 作为底层通信协议：

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {"city": "北京"}
  },
  "id": 1
}
```

---

## 3. 从零实现

### Step 1：JSON-RPC 解析器

```python
import json

def parse_jsonrpc(message):
    """解析 JSON-RPC 2.0 消息。"""
    msg = json.loads(message)
    return {
        "method": msg.get("method"),
        "params": msg.get("params", {}),
        "id": msg.get("id"),
    }

def make_jsonrpc_response(result, request_id):
    """构建 JSON-RPC 响应。"""
    return json.dumps({
        "jsonrpc": "2.0",
        "result": result,
        "id": request_id,
    })
```

### Step 2：MCP 初始化

```python
def handle_initialize(params):
    """处理 MCP 初始化请求。"""
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {},
            "resources": {"listChanged": True},
            "prompts": {"listChanged": True},
        },
        "serverInfo": {
            "name": "my-server",
            "version": "1.0.0",
        },
    }
```

---

## 4. 工具

### 4.1 MCP Python SDK

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def get_weather(city: str) -> str:
    """获取天气。"""
    return f"{city}：晴天，22°C"
```

### 4.2 MCP TypeScript SDK

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";

const server = new Server({ name: "my-server", version: "1.0.0" });
```

---

## 5. 工程最佳实践

### 5.1 传输层选择

| 传输 | 适用场景 |
|------|---------|
| stdio | 本地进程通信——最简单 |
| HTTP (SSE) | 远程服务器——最灵活 |
| Streamable HTTP | 实时流——最新规范 |

### 5.2 踩坑经验

- **Server 崩溃**：Client 需要重连逻辑——自动重启 Server
- **能力协商失败**：版本不兼容——始终发送 `protocolVersion`

---

## 6. 常见错误

### 错误 1：没有发送 `notifications/initialized`

**现象：** Server 不响应后续请求。

**修复：** Client 在收到 `initialize` 响应后必须发送 `notifications/initialized`。

---

## 7. 面试考点

### Q1：MCP 和直接 API 调用有什么区别？（难度：⭐⭐）

**参考答案：**
直接 API 调用是为每个集成写专用代码——为 OpenAI 写的工具不能给 Anthropic 用。MCP 是一个标准协议——工具提供方写一次 MCP Server，所有支持 MCP 的 Client 都可以使用。MCP 提供：(1) 标准化发现——Client 运行时查询可用工具；(2) 标准化调用——JSON-RPC 2.0 格式；(3) 生命周期管理——初始化、协商、重连。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| MCP | "LLM 的 USB-C" | 模型上下文协议——标准化 LLM 与工具/数据源之间的通信 |
| 原语 (Primitive) | "MCP 的基本操作" | 服务器端（Tools/Resources/Prompts）和客户端（Roots/Sampling/Elicitation） |
| JSON-RPC 2.0 | "消息格式" | MCP 的底层通信协议——请求-响应模式 |
| 能力协商 | "握手" | 初始化时双方确认支持的原语和版本 |

---

## 📚 小结

MCP 是 LLM 工具集成的标准协议。六个原语、三阶段生命周期、JSON-RPC 2.0 线格式。MCP 让任何 Client 都能与任何 Server 对话——降低了工具生态的维护成本。

---

## ✏️ 练习

1. **【实现】** 用 JSON-RPC 2.0 格式实现 `initialize` 和 `tools/call` 两个方法
2. **【对比】** 对比 MCP 和直接 OpenAI Function Calling 的优劣

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| JSON-RPC 解析器 | `code/main.py` | MCP 消息解析和构建 |

---

## 📖 参考资料

1. [文档] MCP 规范 2025-11-25: https://spec.modelcontextprotocol.io
2. [GitHub] MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
3. [文档] JSON-RPC 2.0: https://www.jsonrpc.org/specification

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
