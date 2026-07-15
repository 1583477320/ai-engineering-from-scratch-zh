# 缓存与成本优化

import hashlib
import time


class ResponseCache:
    """基于查询哈希的响应缓存。"""
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, prompt, model="gpt-4o"):
        key = hashlib.md5((prompt + model).encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, prompt, model, response):
        key = hashlib.md5((prompt + model).encode()).hexdigest()
        self.cache[key] = response

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / max(total, 1)


def calculate_cost(input_tokens, output_tokens, model="gpt-4o"):
    prices = {"gpt-4o": (2.5, 10.0), "gpt-4o-mini": (0.15, 0.6), "claude-sonnet": (3.0, 15.0)}
    in_p, out_p = prices.get(model, (2.5, 10.0))
    return (input_tokens / 1e6 * in_p) + (output_tokens / 1e6 * out_p)


def estimate_monthly_cost(daily_calls, avg_input=500, avg_output=200, cache_rate=0.0):
    monthly = daily_calls * 30
    effective_input = avg_input * (1 - cache_rate) + avg_input * 0.1 * cache_rate
    return monthly * calculate_cost(effective_input, avg_output)


if __name__ == "__main__":
    print("缓存与成本优化演示\n")
    cache = ResponseCache()
    for i in range(100):
        prompt = f"关于主题{i % 10}的问题"  # 每10个重复
        cached = cache.get(prompt)
        if cached is None:
            cache.set(prompt, "gpt-4o", f"回答{i}")
    print(f"缓存命中率: {cache.hit_rate():.0%}")
    print(f"月成本估算 (无缓存): ${estimate_monthly_cost(100000, cache_rate=0):,.2f}")
    print(f"月成本估算 (70%缓存): ${estimate_monthly_cost(100000, cache_rate=0.7):,.2f}")
