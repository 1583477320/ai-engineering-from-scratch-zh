"""顺序测试模拟——固定边界 vs 顺序边界。"""
import random


def sequential_test(p_true=0.05, p_control=0.04, n_trials=200, threshold=1.96):
    successes = 0
    for i in range(1, n_trials + 1):
        if random.random() < p_true:
            successes += 1
        p_hat = successes / i
        se = (p_hat * (1 - p_hat) / i) ** 0.5 if i > 1 else 1
        z = (p_hat - p_control) / se if se > 0 else 0
        if z > threshold:
            return {"decided": True, "n": i, "result": "treatment_wins"}
    return {"decided": False, "n": n_trials, "result": "inconclusive"}


if __name__ == "__main__":
    random.seed(42)
    for p_true in [0.04, 0.045, 0.05, 0.055, 0.06]:
        r = sequential_test(p_true=p_true)
        print(f"p_true={p_true:.3f}  n={r['n']:>4d}  {r['result']}")
