"""Blackwell + TRT-LLM 精度对比——HBM 占用和吞吐量。"""


def hbm_footprint(params_b, bits, kv_concurrent=32, kv_context=2048,
                  kv_bits=16, n_layers=80, n_kv_heads=8, head_dim=128):
    weights_gb = params_b * 1e9 * bits / 8 / 1e9
    kv_per_seq = 2 * n_layers * n_kv_heads * head_dim * kv_bits / 8
    kv_total = kv_per_seq * kv_concurrent / 1e9
    return {"weights": weights_gb, "kv": kv_total, "total": weights_gb + kv_total}


if __name__ == "__main__":
    print(f"{'精度':8s} {'权重GB':>8} {'KV GB':>8} {'总计GB':>8} {'H100':>6}")
    print("-" * 42)
    for name, bits in [("BF16", 16), ("FP8", 8), ("NVFP4", 4)]:
        r = hbm_footprint(70, bits)
        fits = "✓" if r["total"] <= 80 else "✗"
        print(f"{name:8s} {r['weights']:8.1f} {r['kv']:8.1f} {r['total']:8.1f} {fits:>6}")
