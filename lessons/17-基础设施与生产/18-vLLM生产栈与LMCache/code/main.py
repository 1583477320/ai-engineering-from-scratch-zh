"""KV 溢出模拟器——LMCache 恢复 vs 重新 prefill。"""


def kv_spill_simulation(hbm_gb, kv_per_request_gb, requests_per_sec, duration_s):
    total_kv = kv_per_request_gb * min(requests_per_sec * duration_s, 1000)
    overflow = max(0, total_kv - hbm_gb)
    reprefill_cost = overflow * 2
    lmcache_restore = overflow * 0.1
    speedup = reprefill_cost / max(lmcache_restore, 0.01)
    return {"total_kv_gb": total_kv, "overflow_gb": overflow,
            "reprefill_s": reprefill_cost, "lmcache_s": lmcache_restore,
            "speedup": speedup}


if __name__ == "__main__":
    r = kv_spill_simulation(80, 0.5, 50, 10)
    print(f"总 KV: {r['total_kv_gb']:.0f}GB  溢出: {r['overflow_gb']:.0f}GB")
    print(f"重新 prefill: {r['reprefill_s']:.0f}s  LMCache: {r['lmcache_s']:.1f}s")
    print(f"加速比: {r['speedup']:.1f}x")
