# 智能体工作台


class AgentWorkbench:
    def __init__(self, instructions, scope):
        self.instructions = instructions
        self.state = {}
        self.scope = scope
        self.history = []

    def validate(self, task):
        return any(s in task.lower() for s in self.scope)

    def execute(self, task, llm_fn):
        if not self.validate(task):
            return {"error": "超出任务范围"}
        self.history.append({"task": task, "status": "started"})
        result = llm_fn(task, context=self.history)
        self.state[task] = result
        self.history[-1]["status"] = "completed"
        return {"result": result, "status": "success"}


if __name__ == "__main__":
    print("智能体工作台演示\n")
    wb = AgentWorkbench("数据分析助手", ["数据", "分析", "查询"])
    result = wb.execute("数据查询", lambda t, **kw: f"结果: {t}")
    print(f"  {result}")
    result = wb.execute("帮我订外卖", lambda t, **kw: "订餐")
    print(f"  {result} (应在范围内)")
