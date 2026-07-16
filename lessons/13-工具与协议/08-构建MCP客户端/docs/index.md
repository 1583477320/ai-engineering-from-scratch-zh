# 构建 MCP 客户端——发现、调用、会话管理

> 大多数 MCP 内容只发服务器教程，对客户端一笔带过。Client 代码才是困难编排所在：进程生成、能力协商、跨多个服务器的工具列表合并、采样回调、重连和命名空间冲突解决。本课构建一个将三个不同 MCP 服务器提升为一个平面工具命名空间的多服务器客户端。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 07（构建 MCP 服务器）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 生成 MCP 服务器子进程，完成 `initialize`，发送 `notifications/initialized`
- [ ] 合并多个 MCP 服务器的工具到统一命名空间
- [ ] 处理会话管理和重连逻辑
- [ ] 解决命名空间冲突——不同服务器的同名工具

---

## 1. 问题

MCP Server 很好构建——但 Client 才是困难的部分。一个生产级 Client 需要：

- 生成和管理多个 Server 进程
- 合并来自不同 Server 的工具列表
- 处理 Server 崩溃和重连
- 解决命名空间冲突

---

## 2. 概念

### 2.1 MCP Client 架构

```
LLM 应用
    ↓
[MCP Client]
    ├── Server A (天气工具)
    ├── Server B (笔记工具)
    └── Server C (代码工具)
         ↓
统一工具命名空间 → LLM 使用
```

### 2.2 会话管理

```
1. 生成 Server 进程 (子进程)
2. 发送 initialize 请求
3. 收到能力响应
4. 发送 notifications/initialized
5. 合并工具/资源/提示词列表
6. 处理后续调用
7. 监控 Server 健康状态
```

### 2.3 命名空间冲突

两个 Server 都有一个名为 `search` 的工具——需要加前缀区分：

```python
# 合并后
weather_server.search → "weather_server__search"
code_server.search → "code_server__search"
```

---

## 3. 从零实现

### Step 1：多服务器 Client

```python
class MCPClient:
    """简化版 MCP Client——管理多个 Server。"""
    def __init__(self):
        self.servers = {}
        self.unified_tools = {}

    def add_server(self, name, server):
        """添加一个 MCP Server。"""
        self.servers[name] = server
        # 合并工具到统一命名空间
        for tool_name, tool_info in server.tools.items():
            prefixed_name = f"{name}__{tool_name}"
            self.unified_tools[prefixed_name] = {
                **tool_info,
                "server": name,
                "original_name": tool_name,
            }

    def call_tool(self, name, arguments):
        """调用工具——自动路由到正确的 Server。"""
        if name not in self.unified_tools:
            return {"error": f"工具 {name} 不存在"}
        server_name = self.unified_tools[name]["server"]
        original_name = self.unified_tools[name]["original_name"]
        return self.servers[server_name].execute(original_name, arguments)

    def list_tools(self):
        """返回合并后的工具列表。"""
        return [{"name": name, "description": info["description"]}
                for name, info in self.unified_tools.items()]


if __name__ == "__main__":
    print("MCP 多服务器客户端演示\n")

    # 创建两个虚拟服务器
    class WeatherServer:
        def __init__(self):
            self.tools = {
                "search": {"description": "搜索天气", "executor": lambda query: f"天气搜索结果: {query}"},
                "current": {"description": "获取当前天气", "executor": lambda city: f"{city}: 晴天"},
            }
        def execute(self, name, args):
            return self.tools[name]["executor"](**args)

    class NoteServer:
        def __init__(self):
            self.tools = {
                "search": {"description": "搜索笔记", "executor": lambda query: f"笔记搜索结果: {query}"},
                "create": {"description": "创建笔记", "executor": lambda title: f"已创建: {title}"},
            }
        def execute(self, name, args):
            return self.tools[name]["executor"](**args)

    # 构建 Client
    client = MCPClient()
    client.add_server("weather", WeatherServer())
    client.add_server("notes", NoteServer())

    # 合并后的工具列表
    tools = client.list_tools()
    print(f"可用工具 ({len(tools)}):")
    for t in tools:
        print(f"  {t['name']}: {t['description']}")

    # 调用
    print(f"\n调用 weather__search: {client.call_tool('weather__search', {'query': '北京'})}")
    print(f"调用 notes__create: {client.call_tool('notes__create', {'title': '新笔记'})}")
```

---

## 4. 工具

### 4.1 MCP Python SDK Client

```python
from mcp import ClientSession
from mcp.client import stdio_client

async def connect_to_server(server_script):
    """连接 MCP 服务器。"""
    async with stdio_client("python", server_script) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            # 调用工具
            result = await session.call_tool("get_weather", {"city": "北京"})
```

### 4.2 多服务器管理

```python
class MultiServerManager:
    """管理多个 MCP Server 连接。"""
    def __init__(self):
        self.sessions = {}

    async def connect(self, name, server_script):
        """连接到一个 MCP Server。"""
        # ... 连接逻辑
        self.sessions[name] = session

    async def call(self, server_name, tool_name, args):
        """调用指定 Server 的工具。"""
        return await self.sessions[server_name].call_tool(tool_name, args)
```

---

## 5. 工程最佳实践

### 5.1 多服务器管理

- **命名空间合并**：用前缀避免同名工具冲突
- **健康检查**：定期 ping 每个 Server
- **重连逻辑**：Server 崩溃时自动重启

### 5.2 踩坑经验

- **Server 进程泄漏**：Client 退出时必须清理子进程
- **能力协商不完整**：忘记发送 `notifications/initialized`
- **命名空间冲突**：不同 Server 的同名工具导致调用混乱

---

## 6. 常见错误

### 错误 1：Server 进程未清理

**现象：** Client 退出后 Server 进程仍在运行。

**修复：** 用 `atexit` 注册清理函数，或在 Client 析构时杀掉子进程。

### 错误 2：忽略 Server 健康检查

**现象：** Server 崩溃后 Client 继续调用——得到超时错误。

**修复：** 定期发送 ping，检测超时后自动重连。

---

## 7. 面试考点

### Q1：MCP Client 的命名空间合并如何工作？（难度：⭐⭐）

**参考答案：**
Client 将每个 Server 的工具加上前缀（如 `weather__search`、`notes__search`）合并到统一工具列表。LLM 看到的是合并后的工具列表，调用时通过前缀路由到正确的 Server。这样即使多个 Server 有同名工具，也能正确区分。

### Q2：MCP Client 如何处理 Server 崩溃？（难度：⭐⭐⭐）

**参考答案：**
(1) 健康检查——定期发送 ping 检测 Server 是否存活；(2) 超时检测——设置合理的超时时间；(3) 自动重连——崩溃时自动重启 Server 子进程并重新完成 initialize 握手；(4) 工具列表刷新——重连后重新获取可用工具列表。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 命名空间合并 | "统一工具列表" | 将多个 Server 的工具加前缀后合并为一个列表 |
| 命名空间冲突 | "同名工具" | 不同 Server 有相同工具名——需前缀区分 |
| 健康检查 | "Server 存活检测" | 定期 ping Server，检测崩溃并自动重连 |

---

## 📚 小结

MCP Client 管理多个 Server——命名空间合并、工具路由、健康检查、重连。前缀合并解决同名工具冲突。自动重连处理 Server 崩溃。Client 是多服务器编排的核心。

---

## ✏️ 练习

1. **【实现】** 构建一个管理 2 个 MCP Server 的 Client——支持自动重连
2. **【实验】** 模拟 Server 崩溃——验证 Client 的重连逻辑

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| MCP Client | `code/main.py` | 多服务器 Client + 命名空间合并 |

---

## 📖 参考资料

1. [文档] MCP 规范: https://spec.modelcontextprotocol.io
2. [GitHub] MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
3. [GitHub] MCP TypeScript SDK: https://github.com/modelcontextprotocol/typescript-sdk

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。