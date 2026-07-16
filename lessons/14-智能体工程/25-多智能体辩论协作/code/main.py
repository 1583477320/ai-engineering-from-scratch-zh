# 多代理辩论系统


class MultiAgentDebate:
    def __init__(self, agents, rounds=3):
        self.agents = agents
        self.rounds = rounds

    def debate(self, question):
        answers = [a.generate(question) for a in self.agents]
        for round_num in range(self.rounds - 1):
            new_answers = []
            for i, agent in enumerate(self.agents):
                others = [a for j, a in enumerate(answers) if j != i]
                context = "\n".join(f"- {a[:30]}" for a in others)
                new_answer = agent.generate(f"{question}\n其他观点:\n{context}\n给出改进后的答案")
                new_answers.append(new_answer)
            answers = new_answers
        return answers[0]


class SimpleAgent:
    def __init__(self, name):
        self.name = name
    def generate(self, prompt):
        return f"[{self.name}] 分析'{prompt[:20]}...' → 答案是 14"


if __name__ == "__main__":
    print("多代理辩论演示\n")
    agents = [SimpleAgent(f"Agent-{i}") for i in range(3)]
    debate = MultiAgentDebate(agents, rounds=3)
    result = debate.debate("计算 7×2=？")
    print(f"最终答案: {result}")
