"""多智能体辩论（Du 等人 2023 风格）。

3 个智能体，每个从不同的（可能错误的）答案开始。
每轮每个智能体读取其他智能体的答案并朝加权平均修改。
收敛逐轮记录。智能体策略是脚本化的，不是 LLM 驱动的。

运行：python3 code/main.py
"""

import random
from dataclasses import dataclass, field

TRUE_ANSWER = 42.0


@dataclass
class DebateAgent:
    name: str
    answer: float
    confidence: float
    history: list[float] = field(default_factory=list)

    def initial(self) -> None:
        self.history.append(self.answer)

    def revise(self, others: list["DebateAgent"]) -> None:
        """自己和其他人答案的加权平均。"""
        weights = [self.confidence] + [o.confidence for o in others]
        values = [self.answer] + [o.answer for o in others]
        total_w = sum(weights)
        self.answer = sum(w * v for w, v in zip(weights, values)) / total_w
        self.confidence = min(self.confidence * 1.05, 1.0)
        self.history.append(self.answer)


def agreement_score(agents, tol=0.1):
    mean = sum(a.answer for a in agents) / len(agents)
    agree = sum(1 for a in agents if abs(a.answer - mean) <= tol)
    return agree / len(agents)


def error_vs_truth(agents):
    mean = sum(a.answer for a in agents) / len(agents)
    return abs(mean - TRUE_ANSWER)


def run_debate(agents, rounds, label):
    print(f"\n=== {label} ({rounds} 轮) ===")
    for a in agents:
        a.initial()
    hdr = " ".join(f"{a.name:>6s}" for a in agents)
    print(f"  轮次    {hdr}    同意    误差")
    print(f"    0     {' '.join(f'{a.answer:6.2f}' for a in agents)}    {agreement_score(agents):4.2f}     {error_vs_truth(agents):5.2f}")
    for r in range(1, rounds + 1):
        updates = [(a, [o for o in agents if o is not a]) for a in agents]
        for a, others in updates:
            a.revise(others)
        print(f"    {r}     {' '.join(f'{a.answer:6.2f}' for a in agents)}    {agreement_score(agents):4.2f}     {error_vs_truth(agents):5.2f}")


def fresh_team(seed):
    random.seed(seed)
    return [DebateAgent("A", 38.0, 0.6), DebateAgent("B", 42.5, 0.8), DebateAgent("C", 51.0, 0.4)]


def main():
    print("多智能体辩论（Du 等人 2023 风格）")
    print("-" * 46)
    print(f"真实答案: {TRUE_ANSWER}")

    baseline = fresh_team(seed=1)
    for a in baseline:
        a.initial()
    control_mean = sum(a.answer for a in baseline) / len(baseline)
    print(f"\n控制（第 0 轮均值）: {control_mean:.2f}")
    print(f"误差: {abs(control_mean - TRUE_ANSWER):.2f}")

    team3 = fresh_team(seed=1)
    run_debate(team3, rounds=3, label="辩论 3 智能体, 3 轮")

    team5 = fresh_team(seed=2)
    run_debate(team5, rounds=5, label="辩论 3 智能体, 5 轮（边际收益递减）")

    print("\n要点:")
    print("  - 1 轮交换最大幅度削减误差")
    print("  - 第 2-3 轮复合")
    print("  - 第 3 轮后每轮增益缩小（Du 等人平台期）")
    print("  - 成本按 N * R 次 LLM 调用增长，上下文增长")


if __name__ == "__main__":
    main()
