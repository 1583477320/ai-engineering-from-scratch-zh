# 最小工作台——三文件


class MinimalWorkbench:
    def __init__(self, instructions, state, tasks):
        self.instructions = instructions
        self.state = state
        self.tasks = tasks

    def get_context(self):
        return {"instructions": self.instructions, "state": self.state, "tasks": self.tasks}

    def complete_task(self, task):
        if task in self.tasks:
            self.tasks.remove(task)
        self.state[f"done_{task}"] = True
        return {"remaining": len(self.tasks), "done": list(self.state.keys())}


if __name__ == "__main__":
    print("最小工作台演示\n")
    INSTRUCTIONS = {"role": "分析助手", "rules": ["不超范围"]}
    STATE = {}
    TASKS = ["分析数据", "生成报告"]
    wb = MinimalWorkbench(INSTRUCTIONS, STATE, TASKS)
    print(f"上下文: {list(wb.get_context().keys())}")
    result = wb.complete_task("分析数据")
    print(f"完成任务后: {result}")
