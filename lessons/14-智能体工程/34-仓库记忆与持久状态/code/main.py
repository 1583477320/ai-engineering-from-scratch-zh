# 仓库记忆管理器

import json


class RepoMemory:
    def __init__(self, repo_path="."):
        self.repo_path = repo_path
        self.memory = {}

    def store(self, key, value):
        self.memory[key] = value

    def retrieve(self, key):
        return self.memory.get(key)

    def persist(self, key):
        with open(f"{self.repo_path}/{key}.json", "w") as f:
            json.dump(self.memory.get(key, {}), f, indent=2)


if __name__ == "__main__":
    print("仓库记忆演示\n")
    mem = RepoMemory()
    mem.store("user_preference", {"language": "Python"})
    mem.store("completed_tasks", ["搭建管道"])
    print(f"用户偏好: {mem.retrieve('user_preference')}")
    print(f"已完成: {mem.retrieve('completed_tasks')}")
    print(f"未找到: {mem.retrieve('nonexistent')} (应为 None)")
