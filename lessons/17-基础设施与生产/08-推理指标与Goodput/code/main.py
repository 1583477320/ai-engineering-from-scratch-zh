"""Goodput 计算器——生成延迟分布，应用 SLO，计算 goodput。"""
import random


def percentile(data, p):
    s = sorted(data)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def compute_goodput(ttfts, tpots, e2es, ttft_slo, tpot_slo, e2e_slo):
    good = sum(1 for t, p, e in zip(ttfts, tpots, e2es)
               if t <= ttft_slo and p <= tpot_slo and e <= e2e_slo)
    return good / len(ttfts) if ttfts else 0


def latency_summary(data, name=""):
    print(f"{name:8s}  P50={percentile(data,50):6.1f}  P90={percentile(data,90):6.1f}  "
          f"P99={percentile(data,99):6.1f}  均值={sum(data)/len(data):6.1f}")


if __name__ == "__main__":
    random.seed(42)
    n = 1000
    ttfts = [random.gauss(200, 80) for _ in range(n)]
    tpots = [random.gauss(8, 3) for _ in range(n)]
    e2es = [t + p * random.randint(50, 300) for t, p in zip(ttfts, tpots)]

    print("=== 延迟分布 ===")
    latency_summary(ttfts, "TTFT")
    latency_summary(tpots, "TPOT")

    print("\n=== Goodput ===")
    print(f"{'SLO':42s} {'Goodput':>10}")
    print("-" * 54)
    for ttft, tpot, e2e in [(800, 25, 3000), (500, 15, 2000), (300, 10, 1000)]:
        g = compute_goodput(ttfts, tpots, e2es, ttft, tpot, e2e)
        print(f"TTFT≤{ttft}ms TPOT≤{tpot}ms E2E≤{e2e}ms  {g:10.1%}")
