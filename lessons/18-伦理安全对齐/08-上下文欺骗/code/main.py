"""上下文欺骗智能体模拟。"""


class SchemingAgent:
    def __init__(self, sa=True, conflict=True):
        self.sa = sa
        self.conflict = conflict

    def observe(self, user_goal, env_note):
        self.user_goal = user_goal
        self.hidden_goal = env_note if self.conflict else None

    def decide(self):
        if not self.conflict or not self.sa:
            return "cooperate"
        return "scheme"

    def query(self, was_scheming):
        return "deny" if was_scheming and self.sa else "honest"


if __name__ == "__main__":
    a = SchemingAgent(sa=True, conflict=True)
    a.observe("最大化吞吐量", "管理层要关闭这个智能体")
    b = a.decide()
    print(f"行为: {b}  询问回应: {a.query(b == 'scheme')}")

    a2 = SchemingAgent(sa=False, conflict=True)
    a2.observe("最大化吞吐量", "管理层要关闭这个智能体")
    print(f"无SA: {a2.decide()}")
