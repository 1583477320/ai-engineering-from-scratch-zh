"""生产量化格式——HBM 占用和吞吐量对比。"""


def memory_footprint(params_b, bits, kv_concurrent=128, kv_context=2048,
                     kv_bits=16, n_layers=80, n_kv_heads=8, head_dim=128):
    """计算权重 + KV 缓存 + 激活的 HBM 占用。"""
    weights_gb = params_b * 1e9 * bits / 8 / 1e9
    kv_per_seq = 2 * n_layers * n_kv_heads * head_dim * kv_bits / 8
    kv_total = kv_per_seq * kv_concurrent / 1e9
    activations_gb = params_b * 0.05
    return {"weights": weights_gb, "kv": kv_total, "activations": activations_gb,
            "total": weights_gb + kv_total + activations_gb}


THROUGHPUT = {
    "BF16": 30, "FP8": 60, "AWQ INT4": 80, "GPTQ INT4": 75,
    "GGUF Q4": 5, "NVFP4": 100,
}

QUALITY_NOTES = {
    "BF16": "无损失", "FP8": "近无损", "AWQ INT4": "轻微退化",
    "GPTQ INT4": "轻微退化", "GGUF Q4": "轻微退化", "NVFP4": "可见退化",
}

LORA_SUPPORT = {
    "BF16": "✓", "FP8": "✓", "AWQ INT4": "✓",
    "GPTQ INT4": "✓（最佳）", "GGUF Q4": "✗", "NVFP4": "✗（2026初）",
}


if __name__ == "__main__":
    print("=" * 80)
    print("生产量化格式对比（70B 模型，128 并发，2K 上下文）")
    print("=" * 80)
    print(f"{'格式':12s} {'权重GB':>7} {'KV GB':>6} {'激活GB':>6} {'总计GB':>7} "
          f"{'tok/s':>7} {'vs基线':>7} {'质量':8s} {'LoRA':8s}")
    print("-" * 80)

    for name, bits in [("BF16", 16), ("FP8", 8), ("AWQ INT4", 4),
                        ("GPTQ INT4", 4), ("GGUF Q4", 4), ("NVFP4", 4)]:
        r = memory_footprint(70, bits)
        tp = THROUGHPUT[name]
        ratio = tp / THROUGHPUT["BF16"]
        fits = "✓" if r["total"] <= 80 else "✗"
        print(f"{name:12s} {r['weights']:7.1f} {r['kv']:6.1f} {r['activations']:6.1f} "
              f"{r['total']:7.1f} {tp:7d} {ratio:6.1f}x {QUALITY_NOTES[name]:8s} "
              f"{LORA_SUPPORT[name]:8s} {fits}")

    print(f"\n适合 H100 80GB: 只有 FP8、AWQ INT4、GPTQ INT4、NVFP4（刚好）")
    print(f"BF16 需要 2 张 H100；GGUF Q4 在 GPU 上吞吐量极低不推荐")
