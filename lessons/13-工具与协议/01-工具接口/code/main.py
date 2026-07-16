# 工具接口：四步循环 + 工具注册表


class ToolRegistry:
    """工具注册表——管理可用工具。"""
    def __init__(self):
        self.tools = {}

    def register(self, name, description, schema, executor):
        self.tools[name] = {
            "description": description,
            "schema": schema,
            "executor": executor,
        }

    def execute(self, name, arguments):
        if name not in self.tools:
            return {"error": f"工具 {name} 不存在"}
        try:
            return self.tools[name]["executor"](**arguments)
        except Exception as e:
            return {"error": str(e)}

    def get_tools_for_llm(self):
        return [
            {"type": "function", "function": {
                "name": name, "description": info["description"],
                "parameters": info["schema"],
            }}
            for name, info in self.tools.items()
        ]


if __name__ == "__main__":
    print("工具接口演示\n")
    registry = ToolRegistry()

    # 注册工具
    registry.register("get_weather", "获取天气",
        {"type": "object", "properties": {"city": {"type": "string"}}},
        lambda city: f"{city}：晴天，22°C")

    registry.register("calculate", "计算表达式",
        {"type": "object", "properties": {"expr": {"type": "string"}}},
        lambda expr: eval(expr))

    # 执行
    print(f"get_weather: {registry.execute('get_weather', {'city': '北京'})}")
    print(f"calculate: {registry.execute('calculate', {'expr': '2+3'})}")

    # LLM 格式
    tools = registry.get_tools_for_llm()
    print(f"\nLLM 可用工具: {[t['function']['name'] for t in tools]}")
