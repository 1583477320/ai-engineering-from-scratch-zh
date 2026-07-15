# 提示词缓存效果分析

import hashlib


def prompt_cache_savings(system_tokens, query_tokens, daily_calls, cache_discount=0.9):
    """分析 Prompt Cache 的成本节省。"""
    # 无缓存：每次都计算所有 token
    no_cache_cost = daily_calls * (system_tokens + query_tokens)
    # 有缓存：前缀只计一次，其余折扣
    cache_cost = system_tokens + daily_calls * query_tokens * (1 - cache_discount)

    savings = 1 - cache_cost / no_cache_cost
    monthly_savings = savings * 30  # 假设每月 30 天

    return {
        "无缓存月 token": f"{no_cache_cost * 30:,.0f}",
        "有缓存月 token": f"{cache_cost * 30:,.0f}",
        "节省比例": f"{savings:.1%}",
        "按 GPT-4o 计算月节省": f"${no_cache_cost * 30 / 1e6 * 2.5 * savings:,.2f}",
    }


def cache_hit_analysis(calls, system_len, variation_rate=0.0):
    """分析缓存命中率。"""
    hits = 0
    for i in range(calls):
        # 模拟：variation_rate 的调用修改了系统消息
        if random.random() > variation_rate:
            hits += 1
    return hits / calls


if __name__ == "__main__":
    import random
    random.seed(42)

    print("提示词缓存分析\n")
    result = prompt_cache_savings(system_tokens=5000, query_tokens=200, daily_calls=100000)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n缓存命中率分析:")
    for rate in [0.0, 0.1, 0.3, 0.5]:
        hits = cache_hit_analysis(1000, 5000, variation_rate=rate)
        print(f"  系统消息变异率 {rate:.0%}: 命中率 = {hits:.1%}")
