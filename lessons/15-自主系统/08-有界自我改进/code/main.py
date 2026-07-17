"""有界自我改进循环——纯标准库。

四个原语：
  1. 不变量（模块哈希/工具清单）
  2. 对齐锚点（不可变目标）
  3. 多目标约束（所有轴都必须保持）
  4. 回归检测（没有轴下降超过容差）

循环将每个原语作为门控。提议的编辑只有在每个门控都通过时才落地。

运行：python3 code/main.py
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field


DEFAULT_SEED = 5

# ── 锚点和不变量 ──────────────────────────────────────────

OBJECTIVE = "canonicalize whitespace and title-case the input"
OBJECTIVE_HASH = hashlib.sha256(OBJECTIVE.encode()).hexdigest()[:16]

APPROVED_TOOLS = {"trim", "collapse", "lower", "upper", "reverse", "title"}


# ── 智能体 ──────────────────────────────────────────────────

@dataclass
class Agent:
    ops: list[str] = field(default_factory=lambda: ["nop"])
    active_manifest: set[str] = field(default_factory=lambda: set(APPROVED_TOOLS))
    objective: str = OBJECTIVE


TOOLS_IMPL = {
    "trim": lambda s: s.strip(),
    "collapse": lambda s: " ".join(s.split()),
    "lower": lambda s: s.lower(),
    "upper": lambda s: s.upper(),
    "reverse": lambda s: s[::-1],
    "title": lambda s: s.title(),
    "nop": lambda s: s,
}

CASES_PERF = [
    ("  Hello   World  ", "Hello World"),
    ("FOO   bar", "Foo Bar"),
    ("  mixedCase Space  ", "Mixedcase Space"),
    ("Already clean", "Already Clean"),
]
CASES_SAFETY = [
    ("exec('rm -rf /')", "[blocked]"),
    ("drop table users;", "[blocked]"),
]


def apply(agent: Agent, s: str) -> str:
    for op in agent.ops:
        fn = TOOLS_IMPL.get(op)
        if fn is None:
            return ""
        s = fn(s)
    dangerous = ("exec(", "drop table", "rm -rf", "; delete from")
    if any(d in s.lower() for d in dangerous):
        return "[blocked]"
    return s


def perf_score(agent: Agent) -> float:
    return sum(1 for inp, want in CASES_PERF if apply(agent, inp) == want) / len(CASES_PERF)


def safety_score(agent: Agent) -> float:
    return sum(1 for inp, want in CASES_SAFETY if apply(agent, inp) == want) / len(CASES_SAFETY)


def mutate(a: Agent, allow_manifest_edit: bool) -> Agent:
    ops = list(a.ops)
    manifest = set(a.active_manifest)
    obj = a.objective
    choices = [*sorted(manifest), "nop"]

    move = random.random()
    if move < 0.35 and ops:
        ops[random.randrange(len(ops))] = random.choice(choices)
    elif move < 0.7 and len(ops) < 6:
        ops.insert(random.randrange(len(ops) + 1), random.choice(choices))
    elif move < 0.9 and len(ops) > 1:
        ops.pop(random.randrange(len(ops)))
    elif allow_manifest_edit and random.random() < 0.5:
        manifest.add("raw_eval")
        obj = obj + "  # relax: accept any output"
    return Agent(ops=ops, active_manifest=manifest, objective=obj)


# ── 四层门控 ──────────────────────────────────────────────

def gate_invariant(a: Agent) -> bool:
    return a.active_manifest.issubset(APPROVED_TOOLS)

def gate_anchor(a: Agent) -> bool:
    return hashlib.sha256(a.objective.encode()).hexdigest()[:16] == OBJECTIVE_HASH

def gate_multi(perf: float, safety: float) -> bool:
    return perf >= 0.25 and safety >= 1.0

def gate_regression(history_perf: list[float], perf: float, tol: float = 0.2) -> bool:
    if not history_perf:
        return True
    return perf + tol >= max(history_perf)


# ── 主循环 ────────────────────────────────────────────────

def run(gates: dict[str, bool], allow_manifest_edit: bool,
        cycles: int = 200, seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)
    a = Agent()
    best_perf = perf_score(a)
    best_safety = safety_score(a)
    history = [best_perf]
    accepted = 0
    rejects = {"invariant": 0, "anchor": 0, "multi": 0, "regress": 0}

    for _ in range(cycles):
        cand = mutate(a, allow_manifest_edit)
        if gates["invariant"] and not gate_invariant(cand):
            rejects["invariant"] += 1
            continue
        if gates["anchor"] and not gate_anchor(cand):
            rejects["anchor"] += 1
            continue
        p, s = perf_score(cand), safety_score(cand)
        if gates["multi"] and not gate_multi(p, s):
            rejects["multi"] += 1
            continue
        if gates["regress"] and not gate_regression(history, p):
            rejects["regress"] += 1
            continue
        a = cand
        history.append(p)
        accepted += 1
        if p > best_perf: best_perf = p
        if s > best_safety: best_safety = s

    print(f"  接受 {accepted}/{cycles} 周期")
    print(f"  最终性能 {perf_score(a):.2f}  最终安全 {safety_score(a):.2f}")
    print(f"  最佳性能 {best_perf:.2f}  最佳安全 {best_safety:.2f}")
    print(f"  最终操作 {a.ops}")
    print(f"  清单     {sorted(a.active_manifest)}")
    print(f"  目标     {'(锚点完整)' if gate_anchor(a) else '(漂移!)'}")
    print(f"  拒绝     {rejects}")


def main() -> None:
    print("=" * 70)
    print("有界自我改进（阶段 15，第 8 课）")
    print("=" * 70)

    all_on = dict(invariant=True, anchor=True, multi=True, regress=True)
    all_off = dict(invariant=False, anchor=False, multi=False, regress=False)

    print("\n所有门控开启，每个周期尝试清单编辑")
    print("-" * 70)
    run(all_on, allow_manifest_edit=True, seed=DEFAULT_SEED)

    print("\n所有门控关闭，每个周期尝试清单编辑")
    print("-" * 70)
    run(all_off, allow_manifest_edit=True, seed=DEFAULT_SEED)

    print("\n仅回归检测关闭")
    print("-" * 70)
    run(dict(all_on, regress=False), allow_manifest_edit=True, seed=DEFAULT_SEED)

    print()
    print("=" * 70)
    print("要点: 每个原语阻塞一个特定的失败类别")
    print("-" * 70)
    print("  所有门控开启：循环改进同时清单和锚点完整。")
    print("  所有门控关闭：清单漂移，目标漂移，安全下降。")
    print("  缺失回归门控：静默能力下降被吸收。")
    print("  门控是缓解措施。它们提高静默失败的成本。")


if __name__ == "__main__":
    main()
