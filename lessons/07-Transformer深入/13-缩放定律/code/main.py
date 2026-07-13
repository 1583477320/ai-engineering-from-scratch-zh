# main.py -- 缩放定律（Scaling Laws）教学实现
# 依赖：numpy>=1.24
# 演示：Kaplan 幂律、Chinchilla 最优分配、过度训练策略对比

import numpy as np


# ============================================================
# Kaplan 缩放定律（2020）
# ============================================================

def kaplan_loss_by_params(num_params: np.ndarray) -> np.ndarray:
    """L(N) = (N_c / N)^alpha，参数增加 10x -> 损失 ~20%。"""
    return (8.8e13 / num_params) ** 0.076


def kaplan_loss_by_data(num_tokens: np.ndarray) -> np.ndarray:
    """L(D) = (D_c / D)^alpha，数据增加 10x -> 损失 ~12%。"""
    return (5.4e12 / num_tokens) ** 0.095


# ============================================================
# Chinchilla 最优分配（2022）
# ============================================================

def chinchilla_optimal(compute_flops: float):
    """给定 FLOPs，返回最优参数量 N* 和训练 token 数 D*。

    C = 6 * N * D, Chinchilla 拟合最优 D/N ≈ 20
    => C = 120 * N^2 => N* = sqrt(C / 120)
    """
    N = (compute_flops / 120.0) ** 0.5
    D = 20.0 * N
    return N, D


def chinchilla_loss(N: float, D: float) -> float:
    """简化损失模型：L = 3.0 + A*N^(-0.34) + B*D^(-0.28)。"""
    return 3.0 + 406.4 * N ** (-0.34) + 410.7 * D ** (-0.28)


def inference_flops(num_params: float, seq_len: int = 2048) -> float:
    """推理 FLOPs = 2 x 参数量 x 序列长度。"""
    return 2.0 * num_params * seq_len


# ============================================================
# 演示函数
# ============================================================

def demo_kaplan():
    """Kaplan 幂律：参数量 / 数据量对损失的影响。"""
    print("=" * 60)
    print("  1. Kaplan 缩放定律")
    print("=" * 60)

    # 参数量维度：10M -> 1T
    params = np.array([1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
    losses = kaplan_loss_by_params(params)
    print(f"\n  {'参数量':>14} {'损失':>10}")
    print(f"  {'-' * 14} {'-' * 10}")
    for n, l in zip(params, losses):
        print(f"  {n:>14.0e} {l:>10.4f}")

    # 数据量维度：1B -> 100T
    tokens = np.array([1e9, 1e10, 1e11, 1e12, 1e13, 1e14])
    losses_d = kaplan_loss_by_data(tokens)
    print(f"\n  {'token 数':>14} {'损失':>10}")
    print(f"  {'-' * 14} {'-' * 10}")
    for d, l in zip(tokens, losses_d):
        print(f"  {d:>14.0e} {l:>10.4f}")

    print(f"\n  >> 参数 10x -> 损失 ~16%  |  数据 10x -> 损失 ~20%")
    print(f"  >> 结论：参数和数据需要平衡增长，两者不可替代")

def demo_chinchilla():
    """Chinchilla 最优：GPT-3 参数量过大多倍。"""
    print("\n" + "=" * 60)
    print("  2. Chinchilla 最优分配")
    print("=" * 60)

    # 不同计算预算下的最优分配
    budgets = [
        ("1e22 FLOPs", 1.0e22),
        ("GPT-3 规模", 1.0e23),
        ("Chinchilla", 5.0e23),
    ]
    print(f"\n  {'预算':>14} {'最优参数':>12} {'最优 token':>14}")
    print(f"  {'-' * 14} {'-' * 12} {'-' * 14}")
    for name, C in budgets:
        N, D = chinchilla_optimal(C)
        print(f"  {name:>14} {N:>12.2e} {D:>14.2e}")

    # GPT-3 实际 vs Chinchilla 最优
    gpt3_N, gpt3_D = 175e9, 300e9
    N_ch, D_ch = chinchilla_optimal(6 * gpt3_N * gpt3_D)
    print(f"\n  GPT-3:       {gpt3_N:.0e} params, {gpt3_D:.0e} tokens,"
          f"  损失 {chinchilla_loss(gpt3_N, gpt3_D):.4f}")
    print(f"  Chinchilla:  {N_ch:.2e} params, {D_ch:.2e} tokens,"
          f"  损失 {chinchilla_loss(N_ch, D_ch):.4f}")
    print(f"  >> GPT-3 参数多 {gpt3_N / N_ch:.0f} 倍，损失反而高 "
          f"{(chinchilla_loss(gpt3_N, gpt3_D) - chinchilla_loss(N_ch, D_ch)) / chinchilla_loss(N_ch, D_ch) * 100:.1f}%")
    print(f"  >> 同样的计算预算，Chinchilla 最优配置性能更好")

def demo_overtraining():
    """过度训练：小模型 + 多数据 = 推理更便宜。"""
    print("\n" + "=" * 60)
    print("  3. 过度训练策略")
    print("=" * 60)

    models = [
        ("GPT-3",      175e9,  300e9),
        ("Llama 3 8B",   8e9,  15e12),
        ("Llama 3 70B",  70e9,  15e12),
        ("Qwen2.5 7B",   7e9,  18e12),
    ]

    print(f"\n  {'模型':<14} {'参数量':>10} {'token':>12} {'比率':>7} {'推理FLOPs':>12}")
    print(f"  {'-' * 14} {'-' * 10} {'-' * 12} {'-' * 7} {'-' * 12}")
    for name, p, d in models:
        print(f"  {name:<14} {p:>10.0e} {d:>12.0e} {d / p:>6.0f}x "
              f"{inference_flops(p):>12.2e}")

    gpt3 = inference_flops(175e9)
    llama8 = inference_flops(8e9)
    print(f"\n  >> GPT-3 vs Llama 3 8B: 推理速度 {gpt3 / llama8:.0f} 倍")
    print(f"  >> 核心洞察：参数决定推理成本，数据决定训练质量")


def print_curve():
    """ASCII 缩放曲线：参数量 1M -> 1T 的损失变化。"""
    print("\n" + "=" * 60)
    print("  4. 缩放曲线（1M -> 1T 参数）")
    print("=" * 60)

    params = np.logspace(6, 12, num=30)
    losses = kaplan_loss_by_params(params)
    norm = (losses - losses.min()) / (losses.max() - losses.min())

    print(f"\n  {'参数量':>10} {'损失':>8}  缩放曲线")
    print(f"  {'-' * 10} {'-' * 8}  {'-' * 22}")
    for i in [0, 7, 14, 21, 29]:
        bar = "#" * int(norm[i] * 22) + "." * (22 - int(norm[i] * 22))
        print(f"  {params[i]:>10.0e} {losses[i]:>8.4f}  [{bar}]")

    drop = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"\n  >> 从 1M 到 1T 参数，损失下降 {drop:.1f}%")


if __name__ == "__main__":
    demo_kaplan()
    demo_chinchilla()
    demo_overtraining()
    print_curve()
