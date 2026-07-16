# MCP 最小网关


class MinimalGateway:
    """最小 MCP 网关——认证 + 速率限制 + 工具合并。"""
    def __init__(self, rate_limit=100):
        self.servers = {}
        self.call_counts = {}
        self.rate_limit = rate_limit

    def register_server(self, name, server):
        self.servers[name] = server

    def call_tool(self, tool_name, arguments, user_id="default"):
        """调用工具——带速率限制。"""
        self.call_counts[user_id] = self.call_counts.get(user_id, 0) + 1
        if self.call_counts[user_id] > self.rate_limit:
            return {"error": "速率限制", "retry_after": 60}

        for server_name, server in self.servers.items():
            if tool_name in server.tools:
                return server.execute(tool_name, arguments)

        return {"error": f"工具 {tool_name} 不存在"}

    def list_tools(self):
        """合并所有 Server 的工具。"""
        tools = []
        for server_name, server in self.servers.items():
            for tool_name, tool_info in server.tools.items():
                tools.append({"name": f"{server_name}__{tool_name}", "description": tool_info["description"]})
        return tools


if __name__ == "__main__":
    print("MCP 最小网关演示\n")

    class MockServer:
        def __init__(self, tools):
            self.tools = tools
        def execute(self, name, args):
            return f"执行 {name}: {args}"

    gateway = MinimalGateway(rate_limit=5)
    gateway.register_server("weather", MockServer({"get_weather": {"description": "获取天气"}}))
    gateway.register_server("notes", MockServer({"search": {"description": "搜索笔记"}}))

    print("合并后工具:")
    for t in gateway.list_tools():
        print(f"  {t['name']}: {t['description']}")

    # 正常调用
    print(f"\n调用: {gateway.call_tool('weather__get_weather', {'city': '北京'})}")

    # 速率限制测试
    for i in range(6):
        result = gateway.call_tool("notes__search", {"q": "test"}, user_id="user1")
        status = "成功" if "error" not in result else "限制"
        print(f"  调用 {i+1}: {status}")
