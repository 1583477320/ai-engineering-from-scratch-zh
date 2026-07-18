"""L1+L2 缓存成本和命中率模拟。"""


def l2_cache_cost(requests, prompt_tokens, cached_pct=0.5,
                  fresh_rate=3.0, cached_rate=0.30, write_premium=1.25):
    """L2 提示词缓存的成本计算。"""
    no_cache = requests * prompt_tokens * fresh_rate / 1_000_000
    fresh = int(requests * (1 - cached_pct))
    cached = int(requests * cached_pct)
    cost = (fresh * prompt_tokens * fresh_rate / 1_000_000
            + cached * prompt_tokens * fresh_rate * write_premium / 1_000_000
            + cached * prompt_tokens * cached_rate / 1_000_000)
    return {"no_cache": no_cache, "with_cache": cost, "savings": 1 - cost / no_cache}


if __name__ == "__main__":
    r = l2_cache_cost(100_000, 4000, cached_pct=0.6)
    print(f"无缓存: ${r['no_cache']:.2f}")
    print(f"有缓存: ${r['with_cache']:.2f}")
    print(f"节省: {r['savings']:.1%}")
