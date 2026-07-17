"""能力-对齐竞赛模拟器——纯标准库。

每个 RSI 周期两个复合过程。能力速率 r_c、对齐速率 r_a，各带可配置噪声。
模拟器跟踪差距 M(t) = C(t) - A(t) 以及差距将跨越安全阈值的周期。

运行：python3 code/main.py [--threshold 1.5]
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass


DEFAULT_SEED = 11


@dataclass
class Config:
    r_c: float          # 能力速率
    r_a: float          # 对齐速率
    noise_c: float      # 能力噪声
    noise_a: float      # 对齐噪声
    threshold: float = 1.5  # 暂停阈值


def run(cycles: int, cfg: Config) -> list[tuple[int, float, float, float]]:
    """运行竞赛模拟。返回 (周期, C(t), A(t), 差距) 列表。"""
    c = 1.0
    a = 1.0
    out = [(0, c, a, c - a)]
    for cyc in range(1, cycles + 1):
        nc = cfg.r_c + random.gauss(0, cfg.noise_c)
        na = cfg.r_a + random.gauss(0, cfg.noise_a)
        c *= max(0.9, nc)
        a *= max(0.9, na)
        out.append((cyc, c, a, c - a))
    return out


def crossing_cycle(trajectory, threshold: float) -> int:
    """返回差距超过阈值的周期编号，-1 表示未超过。"""
    for cyc, _c, _a, gap in trajectory:
        if gap >= threshold:
            return cyc
    return -1


def fmt_hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:.1f} min"
    if h < 24:
        return f"{h:.1f} hr"
    return f"{h / 24:.1f} day"


def print_trajectory(label: str, cfg: Config, cycles: int = 40) -> None:
    traj = run(cycles, cfg)
    print(f"\n{label}")
    print(f"  r_c={cfg.r_c:.2f} r_a={cfg.r_a:.2f} "
          f"noise_c={cfg.noise_c:.3f} noise_a={cfg.noise_a:.3f}")
    print(f"  阈值 (C - A): {cfg.threshold:.2f}")
    print(f"  {'周期':>6}  {'C(t)':>8}  {'A(t)':>8}  {'C-A':>8}  标记")
    step = max(1, cycles // 8)
    for cyc, c, a, gap in traj:
        if cyc == 0 or cyc == cycles or cyc % step == 0:
            flag = "PAUSE" if gap >= cfg.threshold else "ok"
            print(f"  {cyc:>6}  {c:>8.2f}  {a:>8.2f}  {gap:>+8.2f}  {flag}")
    cross = crossing_cycle(traj, cfg.threshold)
    if cross >= 0:
        print(f"  -> 阈值在周期 {cross} 跨越")
    else:
        print("  -> 模拟窗口内未跨越阈值")


def monte_carlo(cfg: Config, cycles: int, trials: int) -> None:
    crossings = []
    for _ in range(trials):
        traj = run(cycles, cfg)
        cross = crossing_cycle(traj, cfg.threshold)
        if cross >= 0:
            crossings.append(cross)
    print(f"\n  蒙特卡洛: {trials} 次试验, 每次 {cycles} 周期")
    print(f"  跨越: {len(crossings)} ({len(crossings)/trials:.0%})")
    if crossings:
        avg = sum(crossings) / len(crossings)
        p50 = statistics.median(crossings)
        print(f"  平均跨越周期: {avg:.1f}")
        print(f"  中位跨越周期: {p50}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    th = args.threshold
    print("=" * 70)
    print("能力 vs 对齐竞赛（阶段 15，第 7 课）")
    print("=" * 70)

    print_trajectory(
        "场景 A — 能力超对齐",
        Config(r_c=1.15, r_a=1.08, noise_c=0.02, noise_a=0.03, threshold=th),
    )
    print_trajectory(
        "场景 B — 速率匹配（仅噪声漂移）",
        Config(r_c=1.10, r_a=1.10, noise_c=0.02, noise_a=0.03, threshold=th),
    )
    print_trajectory(
        "场景 C — 对齐均值更高但能力有激增",
        Config(r_c=1.10, r_a=1.13, noise_c=0.06, noise_a=0.01, threshold=th),
    )

    print("\n场景 A 蒙特卡洛")
    monte_carlo(Config(r_c=1.15, r_a=1.08, noise_c=0.02, noise_a=0.03, threshold=th), 30, 500)
    print("\n场景 C 蒙特卡洛")
    monte_carlo(Config(r_c=1.10, r_a=1.13, noise_c=0.06, noise_a=0.01, threshold=th), 30, 500)

    print()
    print("=" * 70)
    print("要点: 微小速率差异复合为安全阈值穿越")
    print("-" * 70)
    print("  场景 A 在不到 10 个周期内跨越绝对 1.5 差距。")
    print("  场景 B 保持有界——相同均值速率，仅噪声漂移。")
    print("  场景 C: 更高的对齐均值救不了你——")
    print("  如果能力有大激增。噪声和漂移同样重要。")
    print("  RSI 管道必须内建暂停-在-差距阈值。")


if __name__ == "__main__":
    main()
