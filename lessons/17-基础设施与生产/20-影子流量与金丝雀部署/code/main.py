"""金丝雀发布模拟——带注入回归。"""


def simulate_canary(stage_pct, n_requests, canary_error_rate=0.05,
                    baseline_error_rate=0.05, cost_multiplier=1.0):
    canary = int(n_requests * stage_pct)
    baseline = n_requests - canary
    total_errors = int(canary * canary_error_rate) + int(baseline * baseline_error_rate)
    error_rate = total_errors / n_requests
    cost = baseline * 0.01 + canary * 0.01 * cost_multiplier
    return {"stage": stage_pct, "error_rate": error_rate, "cost": cost,
            "passed": error_rate < 0.08}


if __name__ == "__main__":
    for stage in [0.01, 0.10, 0.25, 0.50, 0.75, 1.0]:
        r = simulate_canary(stage, 1000, cost_multiplier=1.3)
        status = "✓" if r["passed"] else "✗"
        print(f"{r['stage']:>4.0%}  错误率={r['error_rate']:.1%}  "
              f"成本=${r['cost']:.2f}  {status}")
