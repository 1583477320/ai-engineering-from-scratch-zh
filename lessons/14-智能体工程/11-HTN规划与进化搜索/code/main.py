# HTN 规划器


class HTNPlanner:
    """分层任务网络规划器。"""
    def __init__(self):
        self.methods = {}
        self.operators = {}

    def add_method(self, task, subtasks):
        self.methods[task] = subtasks

    def add_operator(self, name, preconditions, effects):
        self.operators[name] = {"preconditions": preconditions, "effects": effects}

    def plan(self, task):
        if task in self.operators:
            return [{"type": "operator", "name": task,
                     "preconditions": self.operators[task]["preconditions"],
                     "effects": self.operators[task]["effects"]}]
        if task in self.methods:
            plan = []
            for subtask in self.methods[task]:
                plan.extend(self.plan(subtask))
            return plan
        return [{"type": "task", "name": task, "status": "unresolved"}]

    def verify(self, plan):
        state = set()
        for step in plan:
            if step["type"] == "operator":
                if not all(p in state for p in step["preconditions"]):
                    return False, f"前提未满足: {step['preconditions']}"
                state.update(step["effects"])
        return True, "计划验证通过"


if __name__ == "__main__":
    print("HTN 规划演示\n")
    planner = HTNPlanner()
    planner.add_method("准备晚餐", ["购买食材", "烹饪", "装盘"])
    planner.add_method("购买食材", ["列出清单", "去超市"])
    planner.add_operator("列出清单", [], {"清单已准备好"})
    planner.add_operator("去超市", ["清单已准备好"], {"食材已购买"})
    planner.add_operator("烹饪", ["食材已购买"], {"菜肴已做好"})
    planner.add_operator("装盘", ["菜肴已做好"], {"晚餐准备完成"})

    plan = planner.plan("准备晚餐")
    print(f"计划步骤: {len(plan)}")
    for step in plan:
        print(f"  {step['type']}: {step['name']}")
    valid, msg = planner.verify(plan)
    print(f"验证: {msg}")
