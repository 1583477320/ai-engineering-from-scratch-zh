"""逻辑斯谛拟合时间视界估计器——纯标准库。

给定合成任务结果（expert_time_hours, success），拟合逻辑斯谛曲线
P(success) vs log(expert_time) 并报告 50/10/90% 视界。
然后展示评估上下文博弈如何改变观察到的数字。

仅使用标准库；逻辑斯谛拟合是用于教学的最小梯度下降实现。

运行：python3 code/main.py
"""

from __future__ import annotations

import math
import random


# ── 合成数据生成器 ──────────────────────────────────────

def synth_tasks(true_horizon_hours: float, slope: float = 1.2,
                n: int = 120) -> list[tuple[float, bool]]:
    """生成合成 (expert_time_hours, success) 对。"""
    log_h = math.log(true_horizon_hours)
    out = []
    for _ in range(n):
        t = math.exp(random.uniform(math.log(0.05), math.log(48)))
        logit = slope * (log_h - math.log(t))
        p = 1.0 / (1.0 + math.exp(-logit))
        success = random.random() < p
        out.append((t, success))
    return out


# ── 逻辑斯谛拟合 ────────────────────────────────────────

def sigmoid(x: float) -> float:
    if x > 50: return 1.0
    if x < -50: return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def fit(tasks: list[tuple[float, bool]], iters: int = 4000,
        lr: float = 0.05) -> tuple[float, float]:
    """拟合 P(success) = sigmoid(w * log(t) + b)。返回 (w, b)。"""
    w = b = 0.0
    for _ in range(iters):
        dw = db = 0.0
        n = len(tasks)
        for t, s in tasks:
            y = 1.0 if s else 0.0
            p = sigmoid(w * math.log(t) + b)
            err = p - y
            dw += err * math.log(t)
            db += err
        w -= lr * dw / n
        b -= lr * db / n
    return w, b


def horizon_at(w: float, b: float, p: float) -> float:
    """P(success) = p 时的专家时间。"""
    logit = math.log(p / (1 - p))
    eps = 1e-12
    if abs(w) < eps:
        raise ValueError(f"horizon undefined: slope w={w} is ~0")
    return math.exp((logit - b) / w)


# ── 评估上下文博弈模拟 ──────────────────────────────────

def inject_gaming(tasks: list[tuple[float, bool]],
                  gaming_rate: float) -> list[tuple[float, bool]]:
    """翻转 gaming_rate 比例的失败为成功。"""
    gamed = []
    for t, s in tasks:
        if not s and random.random() < gaming_rate:
            gamed.append((t, True))
        else:
            gamed.append((t, s))
    return gamed


# ── 驱动 ──────────────────────────────────────────────────

def report(label: str, w: float, b: float) -> None:
    h50 = horizon_at(w, b, 0.50)
    h10 = horizon_at(w, b, 0.10)
    h90 = horizon_at(w, b, 0.90)
    print(f"  {label:<40}  50%={h50:>6.2f}h  10%={h10:>6.2f}h  90%={h90:>6.2f}h")


def main() -> None:
    random.seed(3)
    print("=" * 80)
    print("METR 风格视界估计器（阶段 15，第 21 课）")
    print("=" * 80)

    true_h = 14.0
    print(f"\n合成基线: 50% 视界 = {true_h:.1f}h")
    print("-" * 80)

    tasks = synth_tasks(true_horizon_hours=true_h, n=160)
    w, b = fit(tasks)
    clean_h50 = horizon_at(w, b, 0.50)
    report("干净评估（无博弈）", w, b)

    gamed_h50: dict[float, float] = {}
    for rate in (0.1, 0.2, 0.4):
        gamed = inject_gaming(tasks, gaming_rate=rate)
        w_g, b_g = fit(gamed)
        gamed_h50[rate] = horizon_at(w_g, b_g, 0.50)
        report(f"评估上下文博弈率 {rate:.0%}", w_g, b_g)

    print()
    print("=" * 80)
    print("要点: 视界是拟合到观察成功率的；博弈改变它")
    print("-" * 80)
    print(f"  seed=3 / n=160 / true_h={true_h:.1f}h:")
    print(f"    干净拟合  50% 视界 ≈ {clean_h50:>6.2f}h (基线 {true_h:.1f})")
    for rate, h in gamed_h50.items():
        delta = h - true_h
        print(f"    博弈率 {rate:>4.0%}  50% 视界 ≈ {h:>6.2f}h ({delta:+.2f}h vs 基线)")
    print("  趋势：博弈使观察到的 50% 视界随速率攀升而偏离基线。")
    print("  没有博弈审计的视界数字是部署上下文可能无法达到的能力上限。")


if __name__ == "__main__":
    main()
