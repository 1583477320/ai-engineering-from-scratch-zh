# MCP 服务器示例（使用 FastMCP 模式）

# 实际使用: from mcp.server.fastmcp import FastMCP
# 这里提供简化版概念演示


class SimpleMCPServer:
    """简化版 MCP 服务器——演示核心概念。"""
    def __init__(self, name):
        self.name = name
        self.tools = {}
        self.resources = {}

    def tool(self, name=None, description=""):
        """注册工具。"""
        def decorator(func):
            tool_name = name or func.__name__
            self.tools[tool_name] = {"function": func, "description": description}
            return func
        return decorator

    def resource(self, uri):
        """注册资源。"""
        def decorator(func):
            self.resources[uri] = func
            return func
        return decorator

    def call_tool(self, name, args):
        if name not in self.tools:
            return {"error": f"工具 {name} 不存在"}
        try:
            result = self.tools[name]["function"](**args)
            return {"result": result, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    def list_tools(self):
        return [{"name": k, "description": v["description"]} for k, v in self.tools.items()]


# 注册工具
server = SimpleMCPServer("my-tools")

@server.tool(name="get_weather", description="获取指定城市天气")
def get_weather(city: str) -> str:
    return f"{city}：晴天，22°C，湿度 45%"

@server.tool(name="calculate", description="执行数学表达式")
def calculate(expression: str) -> float:
    allowed = {"abs": abs, "round": round, "min": min, "max": max}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return result
    except Exception:
        return "表达式错误"

@server.resource("docs://api")
def api_docs():
    return "# API 文档\n\nget_weather(city) - 获取天气\ncalculate(expr) - 计算表达式"


if __name__ == "__main__":
    print("MCP 服务器演示\n")

    # 列出工具
    tools = server.list_tools()
    print("可用工具:")
    for t in tools:
        print(f"  {t['name']}: {t['description']}")

    # 调用工具
    result = server.call_tool("get_weather", {"city": "北京"})
    print(f"\n调用 get_weather: {result}")

    result = server.call_tool("calculate", {"expression": "3 * 7 + 2"})
    print(f"调用 calculate: {result}")

    # 读取资源
    docs = server.resources["docs://api"]()
    print(f"\n资源 docs://api: {docs[:30]}...")
