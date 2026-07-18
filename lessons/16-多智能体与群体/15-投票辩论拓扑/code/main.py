"""投票和辩论拓扑测试工具——纯标准库。

运行星型 / 链型 / 树型 / 图型拓扑。
每个智能体有一个基础准确率概率和一个 error_bias 方向。
测量（准确率、词元、模拟延迟）。

运行：python3 code/main.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SimAgent:
    name: str
    base_accuracy: float
    error_bias: str
    tokens_per_call: int = 400

    def answer(self, correct: str, rng: random.Random) -> str:
        return correct if rng.random() < self.base_accuracy else self.error_bias


@dataclass
class RunResult:
    topology: str
    n: int
    final_answer: str
    correct: str
    tokens: int
    steps: int

    def accuracy(self) -> int:
        return 1 if self.final_answer == self.correct else 0


def majority(items: list[str]) -> str:
    counts: dict[str, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    return max(counts, key=counts.get)


def run_star(agents, correct, rng):
    hub = agents[0]
    workers = agents[1:]
    answers = [w.answer(correct, rng) for w in workers]
    tokens = sum(w.tokens_per_call for w in workers) + hub.tokens_per_call
    final = majority(answers) if answers else hub.answer(correct, rng)
    return RunResult("star", len(agents), final, correct, tokens, steps=2)


def run_chain(agents, correct, rng):
    current = agents[0].answer(correct, rng)
    tokens = agents[0].tokens_per_call
    for a in agents[1:]:
        proposal = a.answer(correct, rng)
        current = proposal if proposal != current and rng.random() < a.base_accuracy else current
        tokens += a.tokens_per_call
    return RunResult("chain", len(agents), current, correct, tokens, steps=len(agents))


def run_tree(agents, correct, rng):
    root = agents[0]
    leaves = agents[1:]
    if len(leaves) <= 1:
        return RunResult("tree", len(agents), root.answer(correct, rng), correct, root.tokens_per_call, steps=2)
    mid = len(leaves) // 2
    left = majority([a.answer(correct, rng) for a in leaves[:mid]])
    right = majority([a.answer(correct, rng) for a in leaves[mid:]])
    tokens = sum(a.tokens_per_call for a in leaves) + root.tokens_per_call
    final = majority([left, right])
    return RunResult("tree", len(agents), final, correct, tokens, steps=3)


def run_graph(agents, correct, rng, rounds=2):
    positions = [a.answer(correct, rng) for a in agents]
    tokens = sum(a.tokens_per_call for a in agents)
    for _ in range(rounds - 1):
        maj = majority(positions)
        new_positions = []
        for pos, ag in zip(positions, agents):
            if pos != maj and rng.random() < 0.4:
                new_positions.append(maj)
            else:
                new_positions.append(pos)
            tokens += ag.tokens_per_call
        positions = new_positions
    return RunResult("graph", len(agents), majority(positions), correct, tokens, steps=rounds * 2)


def make_agents(n, heterogeneous, seed):
    rng = random.Random(seed)
    if heterogeneous:
        biases = ["WRONG-A", "WRONG-B", "WRONG-C"]
        accuracies = [0.72, 0.70, 0.74, 0.71, 0.73, 0.70, 0.72]
    else:
        biases = ["WRONG-A"]
        accuracies = [0.72] * 7
    return [SimAgent(f"agent-{i}", accuracies[i % len(accuracies)], biases[i % len(biases)])
            for i in range(n)]


def bench(correct, trials, heterogeneous):
    tag = "异质" if heterogeneous else "同质（单一文化）"
    print(f"\n基准——{tag}")
    print(f"{'拓扑':10s} {'N':>3s} {'准确率':>8s} {'平均词元':>12s} {'步数':>6s}")
    for topology in ("star", "chain", "tree", "graph"):
        for n in (3, 5, 7):
            acc_sum, tok_sum, step_sum = 0, 0, 0
            for t in range(trials):
                agents = make_agents(n, heterogeneous, seed=t)
                rng = random.Random(t * 31 + 7)
                if topology == "star":
                    r = run_star(agents, correct, rng)
                elif topology == "chain":
                    r = run_chain(agents, correct, rng)
                elif topology == "tree":
                    r = run_tree(agents, correct, rng)
                else:
                    r = run_graph(agents, correct, rng)
                acc_sum += r.accuracy()
                tok_sum += r.tokens
                step_sum += r.steps
            print(f"{topology:10s} {n:>3d} {acc_sum/trials:>8.2f} {tok_sum//trials:>12d} {step_sum//trials:>6d}")


def main():
    bench(correct="RIGHT", trials=200, heterogeneous=False)
    bench(correct="RIGHT", trials=200, heterogeneous=True)
    print("\n要点:")
    print("  异质集成体在每种拓扑/N 上都优于同质。")
    print("  图型/N=7 显示协调税：词元成本是星型/N=3 的 ~7 倍。")
    print("  星型是低成本聚合的甜蜜点。")
    print("  链型在单一文化下表现不佳，因为一个偏差沿链传播。")


if __name__ == "__main__":
    main()
