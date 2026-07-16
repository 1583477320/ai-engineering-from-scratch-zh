# ReWOO 计划-执行分离模式


class ReWOOPlanner:
    """ReWOO 规划器——生成完整的行动计划。"""
    def __init__(self, tools):
        self.tools = list(tools.keys())

    def plan(self, query):
        """生成执行计划（简化：模拟 LLM 规划）。"""
        # 实际中用 LLM 根据 query 生成计划
        if "天气" in query:
            return [
                {"step": 1, "tool": "get_weather", "args": {"city": "北京"}},
                {"step": 2, "tool": "search", "args": {"query": "北京明天天气"}},
            ]
        elif "计算" in query:
            return [
                {"step": 1, "tool": "calculator", "args": {"expression": "2+3"}},
            ]
        else:
            return [
                {"step": 1, "tool": "search", "args": {"query": query[:30]}},
            ]


class ReWOOExecutor:
    """ReWOO 执行器——逐步执行计划。"""
    def __init__(self, tools):
        self.tools = tools
        self.results = {}

    def execute_plan(self, plan):
        for step in plan:
            tool_name = step["tool"]
            args = step["args"]
            # 替换前一步的引用
            for k, v in list(args.items()):
                if isinstance(v, str) and v.startswith("$E"):
                    ref_step = int(v[2:])
                    args[k] = self.results.get(ref_step, v)
            # 执行
            result = self.tools.get(tool_name, lambda **k: "未知工具")(**args)
            self.results[step["step"]] = result
            print(f"  E{step['step']}: {tool_name}({args}) -> {str(result)[:50]}")
        return self.results


if __name__ == "__main__":
    print("ReWOO 计划-执行模式演示\n")

    TOOLS = {
        "get_weather": lambda city="北京": f"{city}: 晴天, 22°C",
        "search": lambda query="": f"搜索结果: 关于'{query}'的相关网页",
        "calculator": lambda expression="1+1": f"计算结果: {eval(expression)}",
    }

    planner = ReWOOPlanner(tools)
    executor = ReWOOExecutor(TOOLS)

    # ReAct 方式：每步 LLM 推理
    print("ReAct 方式 (3次LLM调用):")
    print("  LLM推理 → get_weather → LLM推理 → search → LLM推理 → 回答")

    # ReWOO 方式：一次 LLM 规划
    print("\nReWOO 方式 (1次LLM调用):")
    plan = planner.plan("北京明天天气怎么样？")
    print(f"  计划: {[s['tool'] for s in plan]}")
    results = executor.execute_plan(plan)
    print(f"  执行结果: {list(results.values())}")
