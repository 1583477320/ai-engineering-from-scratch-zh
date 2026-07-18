"""边缘推理带宽天花板计算器。"""


def bandwidth_decode_ceiling(model_params_b, bits, bandwidth_gb_s):
    """计算内存带宽受限的 decode 吞吐量天花板。"""
    weight_bytes = model_params_b * 1e9 * bits / 8
    return bandwidth_gb_s * 1e9 / weight_bytes


if __name__ == "__main__":
    print(f"{'目标':25s} {'带宽GB/s':>10} {'天花板tok/s':>12}")
    print("-" * 50)
    for name, bw, bits, params in [
        ("iPhone 16 Pro (A18)", 50, 4, 0.25),
        ("骁龙 8 Gen 3", 77, 4, 0.30),
        ("M3 Max (WebGPU)", 100, 4, 0.35),
        ("H100 数据中心", 3350, 4, 0.70),
        ("B200 数据中心", 8000, 4, 0.80),
    ]:
        ceiling = bandwidth_decode_ceiling(params, bits, bw)
        print(f"{name:25s} {bw:10d} {ceiling:12.0f}")
