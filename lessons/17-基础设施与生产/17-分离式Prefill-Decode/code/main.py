"""分离式 vs 共置吞吐量对比。"""


def throughput_colocated(prefill_tokens, decode_tokens, gpu_flops=2000, gpu_bandwidth=3):
    prefill_time = prefill_tokens / gpu_flops
    decode_time = decode_tokens / gpu_bandwidth
    return 1 / (prefill_time + decode_time)


def throughput_disaggregated(prefill_tokens, decode_tokens, kv_transfer_ms=50):
    prefill_time = prefill_tokens / 2000
    decode_time = decode_tokens / 3
    total = prefill_time + decode_time + kv_transfer_ms / 1000
    return 1 / total


if __name__ == "__main__":
    print(f"{'提示词':>6} {'共置':>8} {'分离':>8} {'比率':>6}")
    print("-" * 32)
    for pt in [256, 512, 1024, 2048, 4096, 8192]:
        dt = 200
        c = throughput_colocated(pt, dt)
        d = throughput_disaggregated(pt, dt)
        print(f"{pt:>6d} {c:>8.2f} {d:>8.2f} {d/c:>5.2f}x")
