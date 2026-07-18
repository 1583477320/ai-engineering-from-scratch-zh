"""生产量化格式——HBM 占用和吞吐量对比。"""


def memory_footprint(params_b, bits, kv_concurrent=128, kv_context=2048,
                     kv_bits=16, n_layers=80, n_kv_heads=8, head_dim=128):
    weights_gb = params_b * 1e9 * bits / 8 / 1e9
    kv_per_seq = 2 * n_layers * n_kv_heads * head_dim * kv_bits / 8
    kv_total = kv_per_seq * kv_concurrent / 1e9
    activations_gb = params_b * 0.05
    return {"weights": weights_gb, "kv": kv_total, "total": weights_gb + kv_total + activations_gb}


THROUGHPUT = {
    "BF16": 30, "FP8": 60, "AWQ INT4": 80, "GPTQ INT4": 75,
    "GGUF Q4": 5, "NVFP4": 100,
}

if __name__ == "__main__":
    print(f"{'格式':12s} {'权重GB':>7} {'KV GB':>7} {'总计GB':>7} {'70B tok/s':>10} {'vs BF16':>8}")
    print("-" * 55)
    for name, bits in [("BF16", 16), ("FP8", 8), ("AWQ INT4", 4),
                        ("GPTQ INT4", 4), ("GGUF Q4", 4), ("NVFP4", 4)]:
        r = memory_footprint(70, bits)
        tp = THROUGHPUT[name]
        ratio = tp / THROUGHPUT["BF16"]
        fits = "✓" if r["total"] <= 80 else "✗"
        print(f"{name:12s} {r['weights']:7.1f} {r['kv']:7.1f} {r['total']:7.1f} "
              f"{tp:10d} {ratio:7.1f}x {fits}")
