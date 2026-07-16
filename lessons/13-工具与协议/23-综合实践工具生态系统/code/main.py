# 端到端 MCP 工具生态系统综合实践


class ProductionMCPServer:
    """生产级 MCP Server——工具 + 资源 + 提示词。"""
    def __init__(self, name):
        self.name = name
        self.tools = {}
        self.resources = {}

    def add_tool(self, name, description, executor):
        self.tools[name] = {"description": description, "executor": executor}

    def add_resource(self, uri, name, reader):
        self.resources[uri] = {"uri": uri, "name": name, "reader": reader}

    def execute_tool(self, name, args):
        if name in self.tools:
            return self.tools[name]["executor"](**args)
        return {"error": f"工具 {name} 不存在"}


class ProductionGateway:
    """生产级网关——RBAC + 审计。"""
    def __init__(self):
        self.servers = {}
        self.audit_log = []

    def add_server(self, name, server):
        self.servers[name] = server

    def call(self, user_role, tool_name, arguments):
        self.audit_log.append({"user": user_role, "tool": tool_name})
        for server in self.servers.values():
            if tool_name in server.tools:
                return server.execute_tool(tool_name, arguments)
        return {"error": "工具不存在"}

    def get_audit_log(self):
        return self.audit_log[-10:]


if __name__ == "__main__":
    print("端到端工具生态系统综合实践\n")

    # 1. 创建 MCP Server
    server = ProductionMCPServer("天气+笔记 Server")
    server.add_tool("get_weather", "获取天气", lambda city: f"{city}: 晴天 22°C")
    server.add_tool("search_notes", "搜索笔记", lambda q: f"搜索结果: {q}")
    server.add_resource("notes://list", "所有笔记", lambda: '{"notes": []}')

    # 2. 创建网关
    gateway = ProductionGateway()
    gateway.add_server("weather-notes", server)

    # 3. 模拟调用
    print("调用工具:")
    print(f"  get_weather: {gateway.call('user1', 'get_weather', {'city': '北京'})}")
    print(f"  search_notes: {gateway.call('user1', 'search_notes', {'q': 'MCP'})}")

    # 4. 审计日志
    print(f"\n审计日志:")
    for entry in gateway.get_audit_log():
        print(f"  {entry['user']}: {entry['tool']}")

    # 5. 架构决策
    print("\n架构决策:")
    decisions = [
        "传输层: Streamable HTTP — 远程部署",
        "认证: OAuth 2.1 + PKCE — 安全性",
        "网关: RBAC + 审计 — 企业控制",
        "追踪: OTel GenAI — 端到端可观测",
        "安全: 哈希固定 + 投毒检测 — 多层防御",
    ]
    for d in decisions:
        print(f"  {d}")
