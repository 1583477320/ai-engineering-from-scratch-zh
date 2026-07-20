"""代码迁移智能体——确定性配方+智能体循环回退脚手架。

核心架构原语是两层结构：先确定性配方通行（快速、可审计、安全），
然后智能体循环处理剩余失败，含硬预算和失败分类步骤。

运行：python3 code/main.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


FAILURE_CLASSES = [
    "dep_upgrade_required", "build_tool_drift", "custom_annotation",
    "test_flake", "syntax_edge_case", "budget_exhausted", "coverage_regression",
]


@dataclass
class Repo:
    name: str
    loc: int
    lang: str
    hardness: float


@dataclass
class Attempt:
    repo: Repo
    recipe_applied: int = 0
    agent_turns: int = 0
    cost_usd: float = 0.0
    wall_min: float = 0.0
    status: str = "pending"
    failure_class: str | None = None
    coverage_base: float = 80.0
    coverage_final: float = 80.0


def run_recipes(repo: Repo) -> int:
    base = 20 + int(repo.loc / 500)
    return int(base * (1 - 0.2 * repo.hardness))


BUDGET_MIN = 30.0
BUDGET_USD = 8.0
BUDGET_TURNS = 20


def agent_loop(attempt: Attempt, rng: random.Random) -> None:
    per_turn_min = 2.8 + attempt.repo.hardness * 2.0
    per_turn_usd = 0.45 + attempt.repo.hardness * 0.65
    turn_pass_p = max(0.02, 0.22 * (1 - attempt.repo.hardness * 0.95))
    while True:
        if attempt.agent_turns >= BUDGET_TURNS or attempt.wall_min >= BUDGET_MIN or attempt.cost_usd >= BUDGET_USD:
            attempt.status = "fail"
            attempt.failure_class = "budget_exhausted"
            return
        attempt.agent_turns += 1
        attempt.wall_min += per_turn_min
        attempt.cost_usd += per_turn_usd
        if rng.random() < turn_pass_p:
            delta = rng.gauss(0.0, 0.6)
            attempt.coverage_final = attempt.coverage_base + delta
            if attempt.coverage_final < attempt.coverage_base - 2.0:
                attempt.status, attempt.failure_class = "fail", "coverage_regression"
                return
            attempt.status = "pass"
            return


def classify_failure(rng: random.Random) -> str:
    weights = {"dep_upgrade_required": 0.30, "build_tool_drift": 0.20, "custom_annotation": 0.18, "test_flake": 0.15, "syntax_edge_case": 0.17}
    r = rng.random()
    acc = 0.0
    for cls, w in weights.items():
        acc += w
        if r <= acc:
            return cls
    return "syntax_edge_case"


def migrate(repo: Repo, rng: random.Random) -> Attempt:
    attempt = Attempt(repo=repo)
    attempt.recipe_applied = run_recipes(repo)
    straight_through_p = 0.55 * (1 - repo.hardness)
    if rng.random() < straight_through_p:
        delta = rng.gauss(0.0, 0.4)
        attempt.coverage_final = attempt.coverage_base + delta
        attempt.status = "pass"
        attempt.wall_min, attempt.cost_usd = 3.0 + rng.random() * 4, 0.30
        return attempt
    agent_loop(attempt, rng)
    if attempt.status == "fail" and attempt.failure_class == "budget_exhausted" and rng.random() < 0.75:
        attempt.failure_class = classify_failure(rng)
    return attempt


def synth_bench(rng: random.Random) -> list[Repo]:
    return [Repo(name=f"repo-{i:02d}-{'java' if rng.random()<0.6 else 'python'}",
                 loc=rng.randint(800, 40000), lang=lang := "", hardness=min(0.95, max(0.05, rng.gauss(0.65, 0.18)))) for i in range(50)]


def main() -> None:
    rng = random.Random(19)
    bench = synth_bench(rng)
    results = [migrate(repo, rng) for repo in bench]
    passed = [a for a in results if a.status == "pass"]
    failed = [a for a in results if a.status == "fail"]
    print(f"=== 迁移基准运行（50个仓库）===")
    print(f"通过: {len(passed)}/{50} ({len(passed)/50:.1%})  失败: {len(failed)}")
    taxonomy = {}
    for a in failed:
        taxonomy[a.failure_class or "unknown"] = taxonomy.get(a.failure_class or "unknown", 0) + 1
    print("失败分类法:", {k: v for k, v in sorted(taxonomy.items(), key=lambda x: -x[1])})
    if passed:
        print(f"平均\$/仓库: \${sum(a.cost_usd for a in passed)/len(passed):.2f}")


if __name__ == "__main__":
    main()
