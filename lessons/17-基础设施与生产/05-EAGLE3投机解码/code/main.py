"""EAGLE-3 投机解码加速比模拟。"""


def spec_speedup(alpha, K, overhead=0.15):
    return (1 + K * alpha) / (1 + overhead)


def breakeven_alpha(K, overhead=0.15):
    return overhead / K


if __name__ == "__main__":
    print(f"{'α':>5} {'K':>3} {'加速比':>8} {'盈亏平衡α':>10}")
    print("-" * 30)
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for K in [3, 5, 8]:
            s = spec_speedup(alpha, K)
            be = breakeven_alpha(K)
            print(f"{alpha:5.1f} {K:3d} {s:8.2f}x {be:10.3f}")
        print()
