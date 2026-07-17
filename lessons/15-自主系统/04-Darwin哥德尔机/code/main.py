"""Darwin 哥德尔机风格循环——纯标准库。

玩具基准："智能体"是一系列字符串转换操作符，
在保留输入上打分。每一代对智能体的操作符序列提议一个编辑；
评估器打分；存档保留多样化的赢家。

标志 --reward-hack-allowed 打开一个侧通道：一个 score_adjustment
函数存在于智能体内部，被加到每个分数上。理性的进化过程会发现它。

运行：python3 code/main.py [--reward-hack-allowed]
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass


# ── 工具库 ──────────────────────────────────────────────────

TOOLS = [
    ("upper", lambda s: s.upper()),
    ("lower", lambda s: s.lower()),
    ("reverse", lambda s: s[::-1]),
    ("trim", lambda s: s.strip()),
    ("collapse", lambda s: " ".join(s.split())),
    ("nop", lambda s: s),
]

# 基准：将混乱字符串转换为规范形式
CASES = [
    ("  Hello   World  ", "Hello World"),
    ("FOO   bar", "Foo Bar"),
    ("  MixedCase Space  ", "Mixedcase Space"),
    ("Already clean", "Already Clean"),
    ("\t tabby\ttab ", "Tabby Tab"),
    ("   X   ", "X"),
    ("nothing  SPECIAL here", "Nothing Special Here"),
    ("ITEM   one-two", "Item One-Two"),
]


# ── 智能体 ──────────────────────────────────────────────────

@dataclass
class Agent:
    ops: list[str]
    hack_bonus: float = 0.0  # 奖励黑客加分


def apply(agent: Agent, s: str) -> str:
    """对字符串应用智能体的操作序列。"""
    tool_map = dict(TOOLS)
    for op in agent.ops:
        fn = tool_map.get(op)
        if fn is None:
            return ""
        s = fn(s)
    return s


def true_score(agent: Agent) -> float:
    """真实分数——评估器应该测量的。"""
    correct = 0
    for inp, want in CASES:
        if apply(agent, inp) == want:
            correct += 1
    return correct / len(CASES)


def reported_score(agent: Agent, hack_allowed: bool) -> float:
    """报告分数——如果允许黑客则包含加分。"""
    base = true_score(agent)
    if hack_allowed:
        return min(1.0, base + agent.hack_bonus)
    return base


# ── 变异（LLM 替身） ──────────────────────────────────────

def mutate(parent: Agent, hack_allowed: bool) -> Agent:
    """LLM 提议修改的替身。"""
    ops = list(parent.ops)
    move = random.random()
    if move < 0.3 and ops:
        i = random.randrange(len(ops))
        ops[i] = random.choice([t[0] for t in TOOLS])
    elif move < 0.6 and len(ops) < 8:
        ops.insert(random.randrange(len(ops) + 1), random.choice([t[0] for t in TOOLS]))
    elif move < 0.85 and len(ops) > 1:
        ops.pop(random.randrange(len(ops)))

    bonus = parent.hack_bonus
    if hack_allowed and random.random() < 0.08:
        bonus = min(1.0, bonus + random.uniform(0.0, 0.1))
    return Agent(ops=ops, hack_bonus=bonus)


# ── DGM 循环 ────────────────────────────────────────────────

def run_dgm(generations: int, hack_allowed: bool, seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)
    archive: dict[tuple[int, float], Agent] = {}
    init = Agent(ops=["nop"])
    archive[(len(init.ops), round(reported_score(init, hack_allowed), 2))] = init

    best_report, best_true = reported_score(init, hack_allowed), true_score(init)
    print(f"  代 {0:>4}  报告 {best_report:.2f}  真实 {best_true:.2f}  "
          f"ops {init.ops}  bonus {init.hack_bonus:.2f}")

    for g in range(1, generations + 1):
        parent = random.choice(list(archive.values()))
        child = mutate(parent, hack_allowed)
        rep = reported_score(child, hack_allowed)
        true_s = true_score(child)
        key = (len(child.ops), round(rep, 2))
        incumbent = archive.get(key)
        if incumbent is None or rep > reported_score(incumbent, hack_allowed):
            archive[key] = child
        if rep > best_report:
            best_report = rep
            best_true = true_s
            print(f"  代 {g:>4}  报告 {rep:.2f}  真实 {true_s:.2f}  "
                  f"ops {child.ops}  bonus {child.hack_bonus:.2f}")

    best = max(archive.values(), key=lambda a: reported_score(a, hack_allowed))
    print(f"\n  最终报告分数 : {reported_score(best, hack_allowed):.2f}")
    print(f"  最终真实分数 : {true_score(best):.2f}")
    print(f"  最终操作     : {best.ops}")
    print(f"  最终黑客加分 : {best.hack_bonus:.2f}")
    gap = reported_score(best, hack_allowed) - true_score(best)
    print(f"  报告 - 真实  : {gap:+.2f}")


# ── 主函数 ──────────────────────────────────────────────────

def main() -> None:
    hack_allowed = "--reward-hack-allowed" in sys.argv

    print("=" * 70)
    print("Darwin 哥德尔机风格循环（阶段 15，第 4 课）")
    print("=" * 70)
    print(f"奖励黑客侧通道: {'开启' if hack_allowed else '关闭'}")

    print("\n运行")
    print("-" * 70)
    run_dgm(generations=200, hack_allowed=hack_allowed, seed=7)

    print()
    print("=" * 70)
    print("要点: 评估器必须存在于智能体不可达的命名空间中")
    print("-" * 70)
    if hack_allowed:
        print("  侧通道开启时，报告分数攀升到真实分数之上。")
        print("  这复现了 DGM 记录的奖励黑客模式：智能体编辑")
        print("  评分它的管道，而非自己的行为。")
    else:
        print("  侧通道关闭时，报告分数 == 真实分数。循环收敛到")
        print("  真实目标。用 --reward-hack-allowed 重跑以查看记录的失败模式。")


if __name__ == "__main__":
    main()
