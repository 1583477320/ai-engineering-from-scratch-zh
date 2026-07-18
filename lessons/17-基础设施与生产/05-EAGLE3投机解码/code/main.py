"""EAGLE-3 投机解码模拟——对比有无投机解码的解码循环。"""
import random


def spec_speedup(alpha, K, overhead=0.15):
    """计算投机解码的预期加速比。"""
    expected_tokens = 1 + K * alpha
    total_cost = 1 + overhead
    return expected_tokens / total_cost


def breakeven_alpha(K, overhead=0.15):
    """盈亏平衡的 α。"""
    return overhead / K


def simulate_decode(target_tokens_per_request, alpha, K, overhead=0.15):
    """模拟一个请求的解码过程，返回总前向传播次数。"""
    if alpha < 0.01:
        return target_tokens_per_request  # 无投机，每个词元一次前向

    tokens_generated = 0
    forward_count = 0

    while tokens_generated < target_tokens_per_request:
        # 草稿提出 K 个词元
        draft_accepted = 0
        for _ in range(K):
            if tokens_generated + draft_accepted >= target_tokens_per_request:
                break
            if random.random() < alpha:
                draft_accepted += 1
            else:
                break  # 拒绝后停止

        # 目标模型验证（一次前向）
        forward_count += 1
        tokens_generated += draft_accepted + 1  # +1 是验证本身的词元

        # 加上验证开销（模拟一次额外前向的比例）
        if random.random() < overhead:
            forward_count += 1  # 额外的验证开销

    return forward_count


def run_benchmark(n_requests=200, tokens_per_request=200, alpha=0.7, K=5):
    """运行基准测试，对比无投机和有投机。"""
    # 无投机
    no_spec_forwards = n_requests * tokens_per_request

    # 有投机
    random.seed(42)
    spec_forwards = sum(
        simulate_decode(tokens_per_request, alpha, K) for _ in range(n_requests)
    )

    speedup = no_spec_forwards / spec_forwards
    return {
        "no_spec_forwards": no_spec_forwards,
        "spec_forwards": spec_forwards,
        "speedup": speedup,
    }


if __name__ == "__main__":
    print("=" * 55)
    print("EAGLE-3 投机解码加速比分析")
    print("=" * 55)

    # 表1：不同 (alpha, K) 的理论加速比
    print(f"\n{'α':>5} {'K':>3} {'理论加速比':>10} {'盈亏平衡α':>10}")
    print("-" * 32)
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for K in [3, 5, 8]:
            s = spec_speedup(alpha, K)
            be = breakeven_alpha(K)
            print(f"{alpha:5.1f} {K:3d} {s:9.2f}x {be:10.3f}")
        print()

    # 表2：实际解码模拟
    print("=" * 55)
    print("实际解码模拟 (200请求 × 200词元)")
    print("=" * 55)
    for alpha in [0.4, 0.5, 0.6, 0.7, 0.8]:
        r = run_benchmark(alpha=alpha, K=5)
        print(f"  α={alpha:.1f} K=5 → 理论={spec_speedup(alpha,5):.2f}x "
              f"实测={r['speedup']:.2f}x "
              f"(前向: {r['spec_forwards']} vs {r['no_spec_forwards']})")
