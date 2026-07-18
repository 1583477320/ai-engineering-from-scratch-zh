"""采样策略模拟器——不同保留策略的成本对比。"""


def simulate_retention(total, strategy, cost_per_trace=0.001):
    if strategy == "full":
        retained = total
    elif strategy == "errors_only":
        retained = int(total * 0.05)
    elif strategy == "sampled":
        retained = int(total * 0.05)
    else:  # errors+sampled
        retained = int(total * 0.10)

    cost = retained * cost_per_trace
    return {"retained": retained, "cost": cost, "ratio": retained / total}


if __name__ == "__main__":
    print(f"{'策略':20s} {'保留':>10} {'比例':>6} {'成本':>10}")
    print("-" * 50)
    for s in ["full", "sampled", "errors_only", "errors+sampled"]:
        r = simulate_retention(1_000_000, s)
        print(f"{s:20s} {r['retained']:>10,} {r['ratio']:>6.1%} ${r['cost']:>9,.0f}")
