# 函数调用深入：三提供商格式转换


def convert_tool_to_all_formats(name, description, schema):
    """将工具声明转换为三种提供商格式。"""
    openai_tool = {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}
    anthropic_tool = {"name": name, "description": description, "input_schema": schema}
    gemini_tool = {"name": name, "description": description, "parameters": schema}
    return {"openai": openai_tool, "anthropic": anthropic_tool, "gemini": gemini_tool}


class UnifiedToolExecutor:
    """统一工具执行器。"""
    def __init__(self):
        self.tools = {}

    def register(self, name, executor, description="", schema=None):
        self.tools[name] = {"executor": executor, "description": description, "schema": schema}

    def execute(self, name, args):
        return self.tools[name]["executor"](**args)

    def get_tools(self):
        return [{"name": n, "description": t["description"], "schema": t["schema"]} for n, t in self.tools.items()]


if __name__ == "__main__":
    print("函数调用三提供商格式转换演示\n")

    # 工具声明转换
    formats = convert_tool_to_all_formats(
        "get_weather", "获取天气",
        {"type": "object", "properties": {"city": {"type": "string"}}}
    )
    for fmt, tool in formats.items():
        print(f"{fmt}: {list(tool.keys())}")

    # 统一执行器
    executor = UnifiedToolExecutor()
    executor.register("get_weather", lambda city: f"{city}：晴天", "获取天气")
    print(f"\n执行: {executor.execute('get_weather', {'city': '北京'})}")
    print(f"可用工具: {[t['name'] for t in executor.get_tools()]}")
