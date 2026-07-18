"""模型路由模拟——预路由/级联/集成的成本和质量对比。"""
import random


def classify_task(prompt):
    """简单任务分类器。"""
    for kw in ["代码", "code", "debug", "function", "class", "debug"]:
        if kw in prompt.lower():
            return "frontier"
    for kw in ["分析", "为什么", "如何", "explain", "compare", "analyze"]:
        if kw in prompt.lower():
            return "balanced"
    if len(prompt) > 4000:
        return "balanced"
    return "cheap"


def route_cost(n_requests, split, costs):
    """计算路由后的混合成本。"""
    cheap = int(n_requests * split["cheap"])
    balanced = int(n_requests * split["balanced"])
    frontier = n_requests - cheap - balanced
    return (cheap * costs["cheap"] + balanced * costs["balanced"]
            + frontier * costs["frontier"])


if __name__ == "__main__":
    costs = {"cheap": 0.25, "balanced": 2.0, "frontier": 10.0}

    print("=== 路由分割 vs 成本 ===\n")
    print(f"{'便宜占比':>8} {'平衡占比':>8} {'前沿占比':>8} {'总成本':>10} {'vs全前沿':>10}")
    print("-" * 50)
    for cheap_pct in [0.5, 0.6, 0.7, 0.8]:
        split = {"cheap": cheap_pct, "balanced": 0.2, "frontier": 1 - cheap_pct - 0.2}
        cost = route_cost(1000, split, costs)
        full_frontier = route_cost(1000, {"cheap": 0, "balanced": 0, "frontier": 1.0}, costs)
        print(f"{cheap_pct:>8.0%} {0.2:>8.0%} {split['frontier']:>8.0%} "
              f"${cost:>9,.0f} {cost/full_frontier:>9.0%}")
