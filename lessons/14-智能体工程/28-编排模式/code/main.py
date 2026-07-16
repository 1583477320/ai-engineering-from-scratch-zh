# 监督者-工作者编排模式


class SupervisorAgent:
    """监督者——调度工作者。"""
    def __init__(self, workers):
        self.workers = workers

    def dispatch(self, task):
        for worker in self.workers:
            if worker.can_handle(task):
                return worker.execute(task)
        return "无可用工作者"


class WorkerAgent:
    def __init__(self, name, skills):
        self.name = name
        self.skills = skills

    def can_handle(self, task):
        return any(skill in task.lower() for skill in self.skills)

    def execute(self, task):
        return f"[{self.name}] 完成: {task[:30]}"


if __name__ == "__main__":
    print("监督者-工作者编排演示\n")
    workers = [
        WorkerAgent("研究员", ["搜索", "查询", "分析"]),
        WorkerAgent("撰稿人", ["写作", "翻译", "总结"]),
    ]
    supervisor = SupervisorAgent(workers)

    for task in ["搜索 Python 教程", "翻译这段话", "写一份报告", "部署到生产"]:
        result = supervisor.dispatch(task)
        print(f"  '{task}' → {result}")
