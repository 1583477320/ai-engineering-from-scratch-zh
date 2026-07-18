"""Blackwell + TRT-LLM 精度对比——HBM 占用、吞吐量和成本。"""


def hbm_footprint(params_b, bits, kv_concurrent=32, kv_context=2048,
                  kv_bits=16, n_layers=80, n_kv_heads=8, head_dim=128):
    """计算权重 + KV 缓存的 HBM 占用。"""
    weights_gb = params_b * 1e9 * bits / 8 / 1e9
    kv_per_seq = 2 * n_layers * n_kv_heads * head_dim * kv_bits / 8
    kv_total = kv_per_seq * kv_concurrent / 1e9
    return {"weights": weights_gb, "kv": kv_total, "total": weights_gb + kv_total}


def decode_throughput(params_b, bits, hbm_gb, bandwidth_tb_s):
    """估算内存带宽受限的 decode 吞吐量（tok/s）。"""
    weight_bytes = params_b * 1e9 * bits / 8
    tok_per_sec = bandwidth_tb_s * 1e12 / weight_bytes
    return tok_per_sec


def cost_per_million_tokens(throughput_tok_s, gpu_cost_per_hour):
    """计算每百万词元的成本。"""
    tokens_per_hour = throughput_tok_s * 3600
    return gpu_cost_per_hour / (tokens_per_hour / 1_000_000)


if __name__ == "__main__":
    print("=" * 65)
    print("Blackwell + TRT-LLM 精度对比分析")
    print("=" * 65)

    configs = [
        ("H100 BF16+vLLM",  70, 16, 80, 3.35, 2.5),
        ("H100 FP8+vLLM",   70, 8,  80, 3.35, 2.5),
        ("B200 NVFP4+TRT",  70, 4,  192, 8.0, 3.0),
    ]

    print(f"\n{'配置':20s} {'精度':5s} {'权重GB':>7} {'KV GB':>6} {'总计GB':>7} "
          f"{'带宽TB/s':>9} {'吞吐tok/s':>10} {'$/M tok':>9}")
    print("-" * 78)

    for name, params, bits, hbm, bw, cost in configs:
        r = hbm_footprint(params, bits)
        tp = decode_throughput(params, bits, r["total"], bw)
        cpt = cost_per_million_tokens(tp, cost)
        fits = "✓" if r["total"] <= hbm else "✗"
        print(f"{name:20s} {bits:4d}b {r['weights']:7.1f} {r['kv']:6.1f} {r['total']:7.1f} "
              f"{bw:9.1f} {tp:10.0f} {cpt:9.4f}")

    print(f"\n{'经济差距':20s} H100 FP8 vs B200 NVFP4: ", end="")
    h100_cost = cost_per_million_tokens(
        decode_throughput(70, 8, 70, 3.35), 2.5)
    b200_cost = cost_per_million_tokens(
        decode_throughput(70, 4, 60, 8.0), 3.0)
    print(f"{h100_cost / b200_cost:.1f}x")
