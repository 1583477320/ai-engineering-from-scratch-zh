"""METR 风格时间视界模拟器——纯标准库。

给定翻倍时间和基准视界，投影未来年份的 50% 任务完成视界。
另外展示每步可靠性在轨迹中如何复合：99% 每步可靠的智能体在 70 步时仍然只有硬币抛掷的成功率。

教学用途，非校准。目的是在信任智能体无人值守运行前把这些数字记在脑子里。

运行：python3 code/main.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HorizonConfig:
    baseline_hours: float
    baseline_month: int  # 距纪元的月数（0 = 现在）
    doubling_months: float


def horizon_at(cfg: HorizonConfig, months_from_now: int) -> float:
    """投影给定月份偏移处的 50% 视界。"""
    delta = months_from_now - cfg.baseline_month
    return cfg.baseline_hours * (2 ** (delta / cfg.doubling_months))


def months_to_cross(cfg: HorizonConfig, target_hours: float) -> float:
    """视界达到目标小时数所需的月数。"""
    ratio = target_hours / cfg.baseline_hours
    return cfg.baseline_month + cfg.doubling_months * math.log2(ratio)


def end_to_end_reliability(per_step: float, steps: int) -> float:
    """序列中每步都成功的概率。"""
    return per_step ** steps


def max_steps_for_target(per_step: float, target: float) -> int:
    """使端到端可靠性不低于目标的最大步数。"""
    if per_step >= 1.0:
        return 10**9
    return math.floor(math.log(target) / math.log(per_step))


def fmt_hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:.1f} min"
    if h < 24:
        return f"{h:.1f} hr"
    return f"{h / 24:.1f} day"


def horizon_projection() -> None:
    """用 METR 的拟合斜率投影视界。"""
    cfg = HorizonConfig(baseline_hours=14.0, baseline_month=0, doubling_months=7.0)
    print("\nMETR 风格视界投影")
    print("-" * 70)
    print(f"  基准: {cfg.baseline_hours:.1f}h @ 月 0 (Claude Opus 4.6, 2026年1月)")
    print(f"  翻倍时间: {cfg.doubling_months:.1f} 月")
    print()
    print(f"  {'月份':>8}  {'视界':>12}  {'解读':<30}")
    for m in (0, 6, 12, 18, 24, 30, 36):
        h = horizon_at(cfg, m)
        tag = ""
        if h < 24:
            tag = "工作日规模"
        elif h < 168:
            tag = "多天任务"
        elif h < 720:
            tag = "周规模"
        else:
            tag = "月规模"
        print(f"  {m:>8}  {fmt_hours(h):>12}  {tag:<30}")

    print()
    print("  目标穿越")
    for target in (24, 48, 168, 720):
        m = months_to_cross(cfg, target)
        print(f"    {fmt_hours(target)}: 月 {m:.1f}")


def reliability_compounding() -> None:
    """展示每步可靠性如何在轨迹中衰减。"""
    print("\n每步可靠性 → 端到端可靠性")
    print("-" * 70)
    print(f"  {'每步':>10}  {'步数':>8}  {'端到端':>12}  {'标记':<20}")
    cases = [
        (0.90, 10), (0.90, 50), (0.95, 50), (0.99, 50),
        (0.99, 70), (0.99, 200), (0.995, 200), (0.999, 1000),
    ]
    for per_step, steps in cases:
        p = end_to_end_reliability(per_step, steps)
        flag = ""
        if p < 0.5:
            flag = "硬币抛掷或更差"
        elif p < 0.8:
            flag = "不生产就绪"
        elif p < 0.95:
            flag = "脆弱"
        else:
            flag = "可以"
        print(f"  {per_step:>10.3f}  {steps:>8}  {p:>12.1%}  {flag:<20}")

    print()
    print("  50% 端到端成功率的最大轨迹长度")
    for per_step in (0.90, 0.95, 0.99, 0.995, 0.999):
        n = max_steps_for_target(per_step, 0.50)
        print(f"    每步 {per_step:.3f}: 最多 {n} 步")


def deploy_gap_note() -> None:
    """评估上下文博弈调整。"""
    print("\n评估-部署差距调整")
    print("-" * 70)
    print("  METR 数字假设理想工具、无后果、无评估上下文博弈。")
    print("  Anthropic 2024 年对齐伪装研究发现 Claude 在 12% 基础测试中")
    print("  表现出伪装行为，重新训练尝试后上升到 78%。")
    print()
    for horizon in (14.0, 48.0, 168.0):
        for gap in (0.0, 0.2, 0.4):
            effective = horizon * (1 - gap)
            print(f"  基准 {fmt_hours(horizon):>7}  "
                  f"差距 {gap:.0%}  →  部署 "
                  f"{fmt_hours(effective):>7}")


def main() -> None:
    print("=" * 70)
    print("METR 时间视界与复合可靠性（阶段 15，第 1 课）")
    print("=" * 70)
    horizon_projection()
    reliability_compounding()
    deploy_gap_note()
    print()
    print("=" * 70)
    print("要点: 视界指数增长，可靠性指数复合")
    print("-" * 70)
    print("  七个月翻倍下，多天视界约 1 年后到来。")
    print("  99% 每步下，70 步轨迹已经是硬币抛掷。")
    print("  两个数字同时重要。为两者都设计。")


if __name__ == "__main__":
    main()
