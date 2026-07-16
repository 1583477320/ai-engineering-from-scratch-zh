# Voyager 技能库实现


class SkillLibrary:
    """技能库——存储、检索、管理技能。"""
    def __init__(self):
        self.skills = []

    def add(self, name, description, code):
        skill = {"name": name, "description": description, "code": code, "success": 0}
        self.skills.append(skill)
        return skill

    def search(self, task):
        results = []
        for skill in self.skills:
            score = sum(1 for w in task.split() if w in skill["description"])
            if score > 0:
                results.append((skill, score))
        return sorted(results, key=lambda x: -x[1])

    def get_top(self, task):
        results = self.search(task)
        return results[0][0] if results else None

    def update_success(self, skill):
        skill["success"] = skill.get("success", 0) + 1

    def status(self):
        return f"技能库: {len(self.skills)} 个技能"


if __name__ == "__main__":
    print("技能库演示\n")
    lib = SkillLibrary()
    lib.add("迷宫导航", "在网格世界中找到出口", "return go_right()")
    lib.add("收集物品", "收集环境中的金币", "return collect_coin()")
    lib.add("战斗", "与敌人战斗", "return attack()")

    print(lib.status())
    results = lib.search("迷宫")
    print(f"搜索 '迷宫': {[s['name'] for s, _ in results]}")
    print(f"获取技能: {lib.get_top('收集物品')['name']}")
