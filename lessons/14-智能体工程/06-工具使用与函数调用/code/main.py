# 工具管理器：注册、选择、执行


class ToolManager:
    """工具管理器——注册、选择、执行、错误处理。"""
    def __init__(self):
        self.tools = {}

    def register(self, name, description, executor):
        self.tools[name] = {"description": description, "executor": executor}

    def select_tool(self, query):
        """根据查询选择最合适的工具。"""
        for name in self.tools:
            keywords = name.lower().replace("_", " ").split()
            if any(kw in query.lower() for kw in keywords):
                return name
        return None

    def execute(self, tool_name, args):
        if tool_name not in self.tools:
            return {"error": f"工具 {tool_name} 不存在"}
        try:
            return {"result": self.tools[tool_name]["executor"](**args), "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}


if __name__ == "__main__":
    print("工具管理器演示\n")

    tm = ToolManager()
    tm.register("get_weather", "获取天气", lambda city="北京": f"{city}: 晴天, 22°C")
    tm.register("search", "搜索信息", lambda query="": f"搜索结果: {query}")
    tm.register("calculator", "数学计算", lambda expr="1+1": f"结果: {eval(expr)}")

    for q in ["北京天气", "搜索 Python", "计算 3+5"]:
        tool = tm.select_tool(q)
        if tool:
            result = tm.execute(tool, {"city": "北京", "query": q, "expr": "1+1"})
            print(f"  '{q}' -> {tool}: {result['result']}")
