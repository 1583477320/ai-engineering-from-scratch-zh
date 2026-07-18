"""真实提示词分布 vs 统一提示词的负载测试对比。"""
import random


def generate_prompt_distribution(mean=500, std=150, n=100):
    return [max(10, int(random.gauss(mean, std))) for _ in range(n)]


if __name__ == "__main__":
    n = 1000
    random.seed(42)
    uniform = [500] * n
    realistic = generate_prompt_distribution(n=n)
    uniform_hits = n
    realistic_hits = max(1, len(set(l // 100 for l in realistic)))
    print(f"统一提示词:          缓存命中 {uniform_hits}/{n} (100%)")
    print(f"真实分布:            缓存命中 约{realistic_hits}/{n} ({realistic_hits/n:.1%})")
    print(f"差距: {uniform_hits - realistic_hits} 次虚假命中")
