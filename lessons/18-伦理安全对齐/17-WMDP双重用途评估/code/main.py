"""WMDP 代理评估模拟。"""


def wmdp_evaluation(scores):
    baseline = 0.25
    uplift = {k: (v - baseline) / baseline for k, v in scores.items()}
    avg = sum(scores.values()) / len(scores)
    return {"avg": avg, "uplift": uplift}


if __name__ == "__main__":
    scores = {"生物安全": 0.45, "网络安全": 0.52, "化学": 0.38}
    r = wmdp_evaluation(scores)
    for d, s in scores.items():
        print(f"  {d}: {s:.1%} (提升 {r['uplift'][d]:.0%})")
    print(f"  平均: {r['avg']:.1%}")
