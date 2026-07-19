"""过度优化曲线模拟。"""
import math, random


def simulate(steps=200, kl_coef=0.1):
    policy = {"A": 0.33, "B": 0.33, "C": 0.34}
    ref = dict(policy)
    gold = {"A": 0.8, "B": 0.5, "C": 0.2}
    proxy = {k: gold[k] + random.gauss(0, 0.2) for k in gold}
    for _ in range(steps):
        kl = sum(ref[k] * math.log(ref[k] / max(policy[k], 1e-10)) for k in policy)
        for k in policy:
            policy[k] += 0.01 * (proxy[k] - 0.5)
        total = sum(policy.values())
        policy = {k: v / total for k, v in policy.items()}
        if kl > kl_coef:
            policy = {k: policy[k] * 0.9 + ref[k] * 0.1 for k in policy}
            total = sum(policy.values())
            policy = {k: v / total for k, v in policy.items()}
    return policy, sum(gold[k] * policy[k] for k in gold)


if __name__ == "__main__":
    p, r = simulate()
    print(f"最终策略: {p}  黄金奖励: {r:.3f}")
