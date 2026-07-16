# MCP 多服务器客户端


class MCPClient:
    """简化版 MCP Client——管理多个 Server，合并工具命名空间。"""
    def __init__(self):
        self.servers = {}
        self.unified_tools = {}

    def add_server(self, name, server):
        """添加 MCP Server 并合并工具。"""
        self.servers[name] = server
        for tool_name, tool_info in server.tools.items():
            prefixed = f"{name}__{tool_name}"
            self.unified_tools[prefixed] = {**tool_info, "server": name, "original_name": tool_name}

    def call_tool(self, name, arguments):
        """调用工具——自动路由到正确的 Server。"""
        if name not in self.unified_tools:
            return {"error": f"工具 {name} 不存在"}
        server_name = self.unified_tools[name]["server"]
        original_name = self.unified_tools[name]["original_name"]
        return self.servers[server_name].execute(original_name, arguments)

    def list_tools(self):
        return [{"name": n, "description": t["description"]} for n, t in self.unified_tools.items()]


if __name__ == "__main__":
    print("MCP 多服务器客户端演示\n")

    class WeatherServer:
        def __init__(self):
            self.tools = {"get_weather": {"description": "获取天气", "executor": lambda city: f"{city}: 晴天 22°C"},
                          "forecast": {"description": "天气预报", "executor": lambda city: f"{city} 明天多云"}}
        def execute(self, name, args):
            return self.tools[name]["executor"](**args)

    class NoteServer:
        def __init__(self):
            self.tools = {"search": {"description": "搜索笔记", "executor": lambda q: f"搜索结果: {q}"},
                          "create": {"description": "创建笔记", "executor": lambda title: f"已创建: {title}"}}
        def execute(self, name, args):
            return self.tools[name]["executor"](**args)

    client = MCPClient()
    client.add_server("weather", WeatherServer())
    client.add_server("notes", NoteServer())

    print("合并后工具列表:")
    for t in client.list_tools():
        print(f"  {t['name']}: {t['description']}")

    print(f"\n调用 weather__get_weather: {client.call_tool('weather__get_weather', {'city': '北京'})}")
    print(f"调用 notes__search: {client.call_tool('notes__search', {'q': 'MCP'})}")
