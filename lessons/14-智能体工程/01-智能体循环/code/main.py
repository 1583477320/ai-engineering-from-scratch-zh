# 智能体循环：ReAct 模式实现


class Agent:
    """简化版智能体——感知-推理-行动循环。"""
    def __init__(self, llm_fn, tools):
        self.llm_fn = llm_fn
        self.tools = tools
        self.memory = []

    def run(self, user_query, max_steps=5):
        """运行智能体循环。"""
        observation = user_query
        self.memory.append({"role": "user", "content": observation})

        for step in range(max_steps):
            # 1. 推理
            action = self.llm_fn(observation, tools=self.tools)

            # 2. 检查是否结束
            if action["type"] == "respond":
                self.memory.append({"role": "assistant", "content": action["content"]})
                return action["content"]

            # 3. 执行工具
            tool_result = self.execute_tool(action)

            # 4. 反馈
            self.memory.append({"role": "tool", "content": str(tool_result)})
            observation = f"工具 {action['tool']} 返回: {tool_result}"

        return "超过最大步数限制"

    def execute_tool(self, action):
        tool_name = action.get("tool", "")
        args = action.get("args", {})
        if tool_name in self.tools:
            return self.tools[tool_name](**args)
        return f"工具 {tool_name} 不存在"


if __name__ == "__main__":
    print("智能体循环演示\n")

    TOOLS = {
        "get_weather": lambda city="北京": f"{city}: 晴天, 22°C",
        "search": lambda query="": f"搜索结果: 关于'{query}'的3个相关网页",
        "calculator": lambda expression="1+1": f"计算结果: {eval(expression)}",
    }

    def mock_llm(observation, tools=None):
        if "天气" in observation:
            return {"type": "tool_call", "tool": "get_weather", "args": {"city": "北京"}}
        elif "搜索" in observation:
            return {"type": "tool_call", "tool": "search", "args": {"query": "天气预报"}}
        else:
            return {"type": "respond", "content": f"北京今天天气很好，22°C晴天。"}

    agent = Agent(llm_fn=mock_llm, tools=TOOLS)
    result = agent.run("北京今天天气怎么样？")
    print(f"\n最终回答: {result}")
    print(f"记忆长度: {len(agent.memory)} 步")
