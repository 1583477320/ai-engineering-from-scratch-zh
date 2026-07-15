# 模型上下文协议（MCP）

> MCP 是 LLM 应用的"USB-C 标准"——一个协议，让你的工具、数据和提示词在所有模型和平台之间无缝切换。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 11 · 09（函数调用与工具使用）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 MCP 的架构——Client/Server 模式如何工作
- [ ] 实现一个简单的 MCP 服务器
- [ ] 理解 MCP 与函数调用的区别和互补关系
- [ ] 选择适合的 MCP 工具生态

---

## 1. 问题

你用 LangChain 构建了一个 LLM 应用。今天 OpenAI 发布了新 API——你的代码需要重写。明天 Anthropic 的工具定义格式变了——你又得改。每个 LLM 平台的工具接口都不一样。这增加了维护成本。

MCP（Model Context Protocol）解决了这个问题——**一个标准协议，所有工具和数据源都可以接入，所有 LLM 都可以使用。**

---

## 2. 概念

### 2.1 MCP 架构

```
LLM 应用（Client）
    ↓  MCP 协议
工具服务器 A（天气API）    工具服务器 B（数据库）    工具服务器 C（代码执行）
```

**Client**：你的 LLM 应用，发起请求
**Server**：工具提供方，暴露工具/资源/Prompt
**Transport**：通信协议（stdio / HTTP）

### 2.2 MCP 提供的三种能力

| 能力 | 说明 | 示例 |
|------|------|------|
| **Tools** | 可执行的函数 | `get_weather(city)` |
| **Resources** | 可读取的数据 | `file:///docs/api.md` |
| **Prompts** | 预定义的提示词模板 | `code_review` 模板 |

### 2.3 MCP vs 函数调用

| 方面 | 函数调用 | MCP |
|------|---------|-----|
| 范围 | 单个应用 | 跨平台生态系统 |
| 定义方式 | API 参数 | 协议规范 |
| 工具发现 | 静态列表 | 动态发现 |
| 可组合性 | 手动集成 | 标准接口 |

---

## 3. 从零实现

### Step 1：简单的 MCP 服务器

```python
# 使用 mcp 库创建服务器
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-tools")

@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}：晴天，22°C"

@mcp.tool()
def calculate(expression: str) -> float:
    """计算数学表达式。"""
    return eval(expression)

@mcp.resource("docs://api-reference")
def api_reference() -> str:
    """返回 API 参考文档。"""
    return "# API 参考\n\nGET /users - 获取用户列表"

if __name__ == "__main__":
    mcp.run()
```

### Step 2：MCP Client

```python
from mcp import ClientSession
from mcp.client import stdio_client

async def run_client():
    async with stdio_client("python", "my_server.py") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"可用工具: {[t.name for t in tools.tools]}")
            result = await session.call_tool("get_weather", {"city": "北京"})
            print(f"结果: {result.content[0].text}")
```

---

## 6. 工程最佳实践

### 6.1 MCP vs 函数调用选择

| 场景 | 推荐 |
|------|------|
| 简单应用 | 函数调用（API 原生） |
| 多工具/跨平台 | MCP（生态丰富） |
| 已有 LangChain/LlamaIndex | 函数调用（集成好） |
| 企业级工具管理 | MCP（集中管理） |

### 6.2 踩坑经验

- **MCP 服务器崩溃**：Client 需要重连逻辑——自动重启服务器
- **工具数量过多**：>30 个工具时 LLM 选择困难——分组或按需加载

---

## 7. 常见错误

### 错误 1：MCP 服务器无重连机制

**现象：** MCP 服务器崩溃后 Client 挂起——无法恢复。

**原因：** 没有实现重连逻辑——Client 等待已死进程的响应。

**修复：** 实现指数退避重连——服务器崩溃时自动重启，最多重试 3 次。

### 错误 2：MCP 工具描述不清晰

**现象：** LLM 选错了工具或参数格式错误。

**原因：** 工具描述太笼统——没有明确说明输入格式和使用场景。

**修复：** 每个工具描述包含：输入参数类型、示例调用、常见错误提示。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| MCP 服务器实现 | `code/main.py` | 工具注册 + 资源 + 调用的完整示例 |

---

## 📚 小结

### Q1：MCP 解决了什么问题？（难度：⭐⭐）

**参考答案：**
MCP 解决了 LLM 工具集成的碎片化问题。每个 LLM 平台（OpenAI、Anthropic、Google）的工具定义格式不同——每换一个平台就要重写工具代码。MCP 提供一个标准协议：工具提供方写一次 MCP Server，所有支持 MCP 的 Client 都可以使用。这降低了工具生态的维护成本，使 LLM 应用更容易跨平台迁移。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| MCP | "LLM 的 USB-C" | 模型上下文协议——标准化 LLM 与工具/数据源之间的通信接口 |
| MCP Server | "工具服务器" | 暴露工具/资源/Prompt 的服务进程 |
| MCP Client | "工具客户端" | 发起 MCP 请求的 LLM 应用 |
| Transport | "通信方式" | MCP 传输层（stdio 进程间通信或 HTTP） |

---

---

## 8. 面试考点

### Q1：MCP 和 LangChain 的 Tool 有什么本质区别？（难度：⭐⭐）

**参考答案：**
LangChain 的 Tool 是应用层抽象——它将 Python 函数包装为 LLM 可调用的工具。MCP 是协议层标准——定义了 Client 和 Server 之间的通信规范。区别：(1) MCP 是跨平台的——同一个 MCP Server 可以被 LangChain、AutoGen、自定义 Client 使用；(2) MCP 支持动态工具发现——Client 可以运行时查询可用工具；(3) MCP 有标准化的资源和提示词机制。

### Q2：MCP 的三种能力（Tools/Resources/Prompts）各有什么用？（难度：⭐⭐）

**参考答案：**
Tools：可执行的函数——LLM 调用后获得结果（如 `get_weather`）。Resources：只读的数据源——LLM 可以读取但不修改（如文件内容、API 文档）。Prompts：预定义的提示词模板——存储在 Server 端，Client 可以按需加载。三者的关系：Tools 执行操作，Resources 提供上下文，Prompts 优化交互。

---

## 📚 小结

---

## ✏️ 练习

1. **【实验】** 用 FastMCP 创建一个简单的工具服务器——暴露 `add(a, b)` 和 `get_time()` 两个工具
2. **【思考】** MCP 和 LangChain 的 Tool 有什么本质区别？什么场景下必须用 MCP？

---

## 📖 参考资料

1. [文档] MCP 规范: https://spec.modelcontextprotocol.io
2. [GitHub] FastMCP: https://github.com/jlowin/fastmcp
3. [GitHub] Claude Code MCP: https://github.com/anthropics/claude-code
