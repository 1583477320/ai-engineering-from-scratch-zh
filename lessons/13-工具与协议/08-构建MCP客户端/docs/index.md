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
