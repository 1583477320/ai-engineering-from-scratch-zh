"""分层成本门控模拟器——纯标准库。

模拟一个在 30 轮后进入轮询循环的智能体。
比较三种配置：
  1. 无上限：无限制支出
  2. 仅月度上限：最终捕获，但先花很多
  3. 分层栈：per-request + iteration + velocity + monthly

指标：执行的轮数、总词元、总美元、触发上限。

运行：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── 模拟运行配置 ──────────────────────────────────────────

NORMAL_TURN_TOKENS = 2_500
LOOP_TURN_TOKENS = 8_000
LOOP_STARTS_AT = 30
DOLLARS_PER_KTOK = 0.003  # Sonnet 类，2026 年中混合价格


def turn_cost(turn: int) -> int:
    return LOOP_TURN_TOKENS if turn >= LOOP_STARTS_AT else NORMAL_TURN_TOKENS


# ── 门控配置 ──────────────────────────────────────────────

@dataclass
class Governor:
    max_tokens_per_request: int = 10_000
    max_turns: int = 200
    max_budget_usd: float = 50.0
    velocity_usd_per_min: float = 5.0
    velocity_window_min: float = 10.0
    monthly_cap_usd: float = 500.0

    enable_request_cap: bool = True
    enable_iter_cap: bool = True
    enable_velocity: bool = True
    enable_session_cap: bool = True
    enable_monthly_cap: bool = True

    seconds_per_turn: float = 30.0


@dataclass
class Run:
    turns: int = 0
    tokens: int = 0
    dollars: float = 0.0
    history: list[tuple[float, float]] = field(default_factory=list)
    stopped_by: str = ""


# ── 速度限制 ──────────────────────────────────────────────

def velocity_exceeded(run: Run, gov: Governor, now_min: float) -> bool:
    """检查过去 velocity_window_min 分钟内的平均消耗率是否超过限速。"""
    if not run.history:
        return False
    cutoff = now_min - gov.velocity_window_min
    window = [(t, d) for (t, d) in run.history if t >= cutoff]
    if not window:
        return False
    start_min, start_dollars = window[0]
    window_dollars = run.dollars - start_dollars
    elapsed = max(now_min - start_min, 1e-9)
    rate = window_dollars / elapsed
    return rate > gov.velocity_usd_per_min


# ── 模拟器 ────────────────────────────────────────────────

def simulate(gov: Governor, label: str) -> Run:
    run = Run()
    now_min = 0.0

    for turn in range(1, 10_001):
        tok = turn_cost(turn)
        if gov.enable_request_cap and tok > gov.max_tokens_per_request:
            tok = gov.max_tokens_per_request
        run.turns = turn
        run.tokens += tok
        run.dollars += (tok / 1000.0) * DOLLARS_PER_KTOK
        now_min += gov.seconds_per_turn / 60.0
        run.history.append((now_min, run.dollars))

        if gov.enable_iter_cap and turn >= gov.max_turns:
            run.stopped_by = "max_turns"
            break
        if gov.enable_session_cap and run.dollars >= gov.max_budget_usd:
            run.stopped_by = "max_budget_usd"
            break
        if gov.enable_velocity and velocity_exceeded(run, gov, now_min):
            run.stopped_by = "velocity_limit"
            break
        if gov.enable_monthly_cap and run.dollars >= gov.monthly_cap_usd:
            run.stopped_by = "monthly_cap"
            break

    if not run.stopped_by:
        run.stopped_by = "ran out of simulated turns"

    print(f"  {label:<24}  turns={run.turns:>5}  tokens={run.tokens:>8,}  "
          f"dollars=${run.dollars:>7.2f}  stopped_by={run.stopped_by}")
    return run


# ── 主函数 ────────────────────────────────────────────────

def main() -> None:
    print("=" * 85)
    print("分层成本门控（阶段 15，第 13 课）")
    print("=" * 85)
    print()
    print("智能体在 30 轮后进入轮询循环。")
    print("-" * 85)

    # 1. 无上限
    g = Governor(
        enable_request_cap=False, enable_iter_cap=False,
        enable_velocity=False, enable_session_cap=False,
        enable_monthly_cap=False,
    )
    g.max_turns = 10_000
    g.enable_iter_cap = True
    simulate(g, "no caps (iter 10k sim)")

    # 2. 仅月度上限
    g = Governor(
        enable_request_cap=False, enable_iter_cap=False,
        enable_velocity=False, enable_session_cap=False,
        enable_monthly_cap=True,
    )
    simulate(g, "monthly cap only")

    # 3. 分层栈
    g = Governor()
    simulate(g, "layered stack")

    print()
    print("=" * 85)
    print("要点: 上限必须分层，因为失败模式在不同时间尺度")
    print("-" * 85)
    print("  月度上限触发晚：钱包已经半空。")
    print("  速度限制 ($5/min) 在几分钟内捕获循环。")
    print("  迭代上限阻止单次运行超过 N 轮。")
    print("  Per-request 上限阻止单次完成无界。")
    print("  会话美元上限 (max_budget_usd) 扣上安全扣。")
    print("  每层覆盖不同的失败（循环、泄漏、激增、发布）。")


if __name__ == "__main__":
    main()
