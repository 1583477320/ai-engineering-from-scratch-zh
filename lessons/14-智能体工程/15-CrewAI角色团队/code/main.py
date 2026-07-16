# CrewAI 角色团队实现


class SimpleCrew:
    """简化版 CrewAI——角色团队。"""
    def __init__(self, agents):
        self.agents = agents

    def execute(self, tasks):
        results = []
        for task in tasks:
            agent = task.get("assignee", self.agents[0])
            print(f"  {agent['role']} 执行: {task['description']}")
            results.append(f"{agent['role']} 完成: {task['description']}")
        return results


if __name__ == "__main__":
    print("CrewAI 角色团队演示\n")
    agents = [
        {"role": "研究分析师", "goal": "收集数据"},
        {"role": "技术撰稿人", "goal": "撰写报告"},
        {"role": "审查员", "goal": "审查质量"},
    ]
    crew = SimpleCrew(agents)
    tasks = [
        {"description": "收集市场数据", "assignee": agents[0]},
        {"description": "撰写分析报告", "assignee": agents[1]},
        {"description": "审查报告质量", "assignee": agents[2]},
    ]
    results = crew.execute(tasks)
    for r in results:
        print(f"  {r}")
