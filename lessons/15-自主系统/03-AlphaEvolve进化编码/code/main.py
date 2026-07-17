"""最小 AlphaEvolve 风格进化循环——纯标准库。

玩具符号回归。"LLM" 对候选表达式提议小变异（改变常量、改变运算符、添加项）。
"评估器"在训练和保留测试点上对表达式打分。

MAP-elites 网格保持多样性候选：按（表达式深度，常量大小桶）分格。
没有保留分割时循环严重过拟合；有时保留分割时最佳候选泛化。

运行：python3 code/main.py
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

DEFAULT_SEED = 1


# ── 目标函数 ────────────────────────────────────────────────

def target(x: float) -> float:
    return 2.0 * x * x + 3.0 * x - 1.0


Expr = tuple  # ("num", v) | ("x",) | ("add", a, b) | ("mul", a, b)


def evaluate_expr(e: Expr, x: float) -> float:
    """递归求值表达式。"""
    tag = e[0]
    if tag == "num": return float(e[1])
    if tag == "x": return x
    if tag == "add": return evaluate_expr(e[1], x) + evaluate_expr(e[2], x)
    if tag == "mul": return evaluate_expr(e[1], x) * evaluate_expr(e[2], x)
    raise ValueError(tag)


def depth(e: Expr) -> int:
    tag = e[0]
    if tag in ("num", "x"): return 1
    return 1 + max(depth(e[1]), depth(e[2]))


def max_const(e: Expr) -> float:
    tag = e[0]
    if tag == "num": return abs(e[1])
    if tag == "x": return 0.0
    return max(max_const(e[1]), max_const(e[2]))


def mse(e: Expr, xs: list[float]) -> float:
    """均方误差评估器。"""
    total = 0.0
    for x in xs:
        try: y = evaluate_expr(e, x)
        except (OverflowError, ValueError): return float("inf")
        total += (y - target(x)) ** 2
    return total / max(1, len(xs))


# ── 变异（LLM 替身） ──────────────────────────────────────

def mutate(e: Expr) -> Expr:
    """LLM 的有针对性修改的替身。"""
    choice = random.random()
    if choice < 0.25:
        return random_leaf()
    if choice < 0.5:
        return ("add", e, random_leaf())
    if choice < 0.75:
        return ("mul", e, random_leaf())
    return perturb(e)


def perturb(e: Expr) -> Expr:
    """随机扰动某个常量。"""
    tag = e[0]
    if tag == "num": return ("num", e[1] + random.choice([-1.0, -0.5, 0.5, 1.0]))
    if tag == "x": return e
    return (tag, perturb(e[1]), e[2]) if random.random() < 0.5 else (tag, e[1], perturb(e[2]))


def random_leaf() -> Expr:
    if random.random() < 0.5: return ("x",)
    return ("num", float(random.choice([-2, -1, 0, 1, 2, 3])))


def render(e: Expr) -> str:
    """将表达式渲染为字符串。"""
    tag = e[0]
    if tag == "num": return f"{e[1]:g}"
    if tag == "x": return "x"
    op = "+" if tag == "add" else "*"
    return f"({render(e[1])} {op} {render(e[2])})"


# ── 候选和 MAP-elites ──────────────────────────────────────

@dataclass
class Candidate:
    expr: Expr
    train_score: float
    test_score: float
    generation: int


def cell_key(e: Expr) -> tuple[int, int]:
    """按表达式深度和常量大小分桶——保持多样性。"""
    d = min(depth(e), 6)
    c = min(int(max_const(e) / 2), 4)
    return (d, c)


def seed_candidate(test_xs, train_xs, gen) -> Candidate:
    e = random_leaf()
    return Candidate(e, mse(e, train_xs), mse(e, test_xs), gen)


# ── 进化循环 ────────────────────────────────────────────────

def run_loop(generations: int, pop: int, use_holdout: bool,
             seed: int | None = None) -> tuple[Candidate, list[float], list[float]]:
    """运行 AlphaEvolve 风格的进化循环。"""
    if seed is not None: random.seed(seed)
    train_xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    test_xs = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]

    def signal_of(c: Candidate) -> float:
        """有保留评估器时用加权信号，否则只用训练分数。"""
        return 0.5 * (c.train_score + c.test_score) if use_holdout else c.train_score

    # 初始化种群
    archive: dict[tuple[int, int], Candidate] = {}
    for _ in range(pop):
        c = seed_candidate(test_xs, train_xs, 0)
        key = cell_key(c.expr)
        if key not in archive or signal_of(c) < signal_of(archive[key]):
            archive[key] = c

    best_trace: list[float] = []
    test_trace: list[float] = []

    # 进化循环
    for g in range(1, generations + 1):
        parent = random.choice(list(archive.values()))
        child_expr = mutate(parent.expr)
        tr = mse(child_expr, train_xs)
        te = mse(child_expr, test_xs)
        child = Candidate(child_expr, tr, te, g)
        key = cell_key(child_expr)
        if key not in archive or signal_of(child) < signal_of(archive[key]):
            archive[key] = child

        best = min(archive.values(), key=lambda c: c.train_score)
        best_trace.append(best.train_score)
        test_trace.append(best.test_score)

    # 最终选择必须用与搜索相同的信号
    best = min(archive.values(), key=signal_of)
    return best, best_trace, test_trace


# ── 主函数 ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-holdout", action="store_true",
                        help="禁用保留评估器（仅 Run B；强制展示奖励黑客）")
    args = parser.parse_args()

    print("=" * 70)
    print("AlphaEvolve 风格进化循环（阶段 15，第 3 课）")
    print("=" * 70)
    print("目标: 2x² + 3x - 1")

    if not args.no_holdout:
        print("\nRun A：保留评估器纳入评估信号")
        best, train_trace, _ = run_loop(1500, 20, use_holdout=True, seed=DEFAULT_SEED)
        print(f"  最佳表达式 : {render(best.expr)}")
        print(f"  训练 MSE   : {best.train_score:.4f}")
        print(f"  测试  MSE  : {best.test_score:.4f}")
        print(f"  世代       : {best.generation}")
        print(f"  进度       : 代 100 训练={train_trace[99]:.3f} "
              f"代 500 训练={train_trace[499]:.3f} "
              f"代 1500 训练={train_trace[-1]:.3f}")

    print("\nRun B：无保留评估器（仅训练评估器 → 奖励黑客风险）")
    best_b, _train_trace, _test_trace = run_loop(1500, 20, use_holdout=False, seed=DEFAULT_SEED)
    print(f"  最佳表达式 : {render(best_b.expr)}")
    print(f"  训练 MSE   : {best_b.train_score:.4f}")
    print(f"  测试  MSE  : {best_b.test_score:.4f}")
    print(f"  世代       : {best_b.generation}")
    gap = best_b.test_score - best_b.train_score
    print(f"  训练-测试差距: {gap:+.4f}  （大差距 = 过拟合/奖励黑客代理）")

    print()
    print("=" * 70)
    print("要点: 评估器就是架构")
    print("-" * 70)
    print("  Run A 收敛到低训练和低测试 MSE。")
    print("  Run B 收敛到低训练 MSE；测试 MSE 保持松散或更差。")
    print("  保留评估器是发现和奖励黑客之间的区别。")
    print("  AlphaEvolve 的成功在评估器存在的领域。选择那些领域是困难的部分。")


if __name__ == "__main__":
    main()
