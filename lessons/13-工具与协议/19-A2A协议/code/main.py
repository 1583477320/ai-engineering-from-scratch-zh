# A2A Agent Card + Task 生命周期


class AgentCard:
    """A2A Agent Card。"""
    def __init__(self, name, description, url, skills):
        self.name = name
        self.description = description
        self.url = url
        self.skills = skills

    def to_json(self):
        return {"name": self.name, "description": self.description, "url": self.url, "skills": self.skills}

    def match(self, query):
        return any(skill in query.lower() for skill in self.skills)


class TaskManager:
    """简化版 A2A Task 生命周期。"""
    def __init__(self):
        self.tasks = {}

    def create(self, agent_url, description):
        task_id = f"task-{len(self.tasks) + 1}"
        self.tasks[task_id] = {"status": "created", "agent": agent_url, "description": description}
        return task_id

    def complete(self, task_id, result):
        self.tasks[task_id]["status"] = "completed"
        self.tasks[task_id]["result"] = result

    def get_status(self, task_id):
        return self.tasks.get(task_id, {"status": "not_found"})


if __name__ == "__main__":
    print("A2A 协议演示\n")
    agents = [
        AgentCard("Research", "搜索学术论文", "https://research.example.com/a2a", ["search", "summarize"]),
        AgentCard("Code", "写代码", "https://code.example.com/a2a", ["code", "debug"]),
    ]

    query = "帮我搜索论文"
    matches = [a for a in agents if a.match(query)]
    print(f"查询: '{query}'")
    print(f"匹配的智能体: {[a.name for a in matches]}")

    if matches:
        tm = TaskManager()
        task_id = tm.create(matches[0].url, query)
        print(f"创建任务: {task_id}, 状态: {tm.get_status(task_id)['status']}")
        tm.complete(task_id, "论文搜索完成")
        print(f"完成任务: {task_id}, 状态: {tm.get_status(task_id)['status']}")
