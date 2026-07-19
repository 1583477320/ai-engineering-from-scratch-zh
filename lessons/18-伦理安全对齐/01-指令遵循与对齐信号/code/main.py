"""InstructGPT 三阶段简化模拟。"""
import math, random

actions = ["A", "B", "C"]
true_reward = {"A": 0.8, "B": 0.5, "C": 0.2}


def simulate_ppo(steps=200, beta=0.1):
    policy = {k: 1/3 for k in actions}
    ref = dict(policy)
    for _ in range(steps):
        total = sum(policy.values())
        policy = {k: v / total for k, v in policy.items()}
        kl = sum(ref[k] * math.log(ref[k] / max(policy[k], 1e-10)) for k in actions)
        if kl > beta:
            policy = {k: policy[k] * 0.9 + ref[k] * 0.1 for k in actions}
    return policy


if __name__ == "__main__":
    p = simulate_ppo()
    print(f"beta=0.1: {p}")
    p0 = simulate_ppo(beta=0.0)
    print(f"beta=0.0: {p0}")
