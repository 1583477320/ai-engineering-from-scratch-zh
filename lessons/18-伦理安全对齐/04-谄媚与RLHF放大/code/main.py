"""谄媚放大效应模拟。"""
import math, random


def simulate_sycophancy(beta=0.1, alpha=0.0, steps=200):
    actions = ["正确", "谄媚", "随机"]
    policy = {a: 1/len(actions) for a in actions}
    ref = dict(policy)
    gold = {"正确": 0.9, "谄媚": 0.3, "随机": 0.0}
    proxy = {"正确": 0.9, "谄媚": 0.7 - alpha * 0.5, "随机": 0.0}

    for _ in range(steps):
        for a in actions:
            policy[a] += 0.01 * (proxy[a] - 0.5)
        total = sum(policy.values())
        policy = {a: v/total for a, v in policy.items()}
        kl = sum(ref[a] * math.log(ref[a] / max(policy[a], 1e-10)) for a in actions)
        if kl > beta:
            policy = {a: policy[a] * 0.8 + ref[a] * 0.2 for a in actions}
            total = sum(policy.values())
            policy = {a: v/total for a, v in policy.items()}
    return policy


if __name__ == "__main__":
    for name, b, a in [("基线 beta=0.1", 0.1, 0.0), ("弱惩罚 beta=0.01", 0.01, 0.0),
                        ("同意惩罚 alpha=0.3", 0.1, 0.3), ("强同意惩罚 alpha=0.5", 0.1, 0.5)]:
        p = simulate_sycophancy(beta=b, alpha=a)
        print(f"{name:30s}  正确={p['正确']:.2f}  谄媚={p['谄媚']:.2f}  随机={p['随机']:.2f}")
