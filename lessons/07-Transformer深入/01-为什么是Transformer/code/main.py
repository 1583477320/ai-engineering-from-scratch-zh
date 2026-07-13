# Transformer vs RNN：串行深度对比实验
# 纯 Python 标准库实现
# 对应课程：阶段 07 · 01

import time


def rnn_style(xs):
    """RNN 风格：每一步依赖前一步——串行深度 = N。"""
    h = 0.0
    for x in xs:
        h = 0.9 * h + x  # h 依赖前一个 h——无法并行
    return h


def attention_style(xs):
    """Attention 风格：所有步骤独立——深度 = 1。"""
    return sum(xs) / len(xs)  # 所有 x 互相独立


def parallel_prefix_sum(xs):
    """并行前缀和（Hillis-Steele）——对数深度。"""
    result = list(xs)
    step = 1
    while step < len(result):
        for i in range(step, len(result)):
            left = max(0, i - step)
            result[i] = result[left] + result[i]
        step *= 2
    return result


def serial_prefix_sum(xs):
    """串行前缀和——线性深度。"""
    result = [0.0] * len(xs)
    result[0] = xs[0]
    for i in range(1, len(xs)):
        result[i] = result[i - 1] + xs[i]
    return result


def timing_table():
    """测量两种风格在不同序列长度下的执行时间。"""
    print("=== 串行深度对比：RNN vs Attention ===")
    print(f"  {'N':>8}  {'RNN':>10}  {'Attention':>10}  {'加速比':>8}")
    print(f"  {'':->8}  {'(串行)':>10}  {'(并行)':>10}  {'':>8}")
    for n in [100, 1_000, 10_000, 100_000]:
        xs = [1.0] * n
        t0 = time.perf_counter()
        rnn_style(xs)
        rnn_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        attention_style(xs)
        attn_time = time.perf_counter() - t0
        speedup = rnn_time / attn_time if attn_time > 0 else float('inf')
        print(f"  {n:>8}  {rnn_time:.6f}s  {attn_time:.6f}s  {speedup:>7.1f}x")


def count_operations():
    """理论操作计数对比。"""
    print("\n=== 理论操作数对比 ===")
    print(f"  {'N':>8}  {'RNN加法':>10}  {'Attention加法':>12}  {'差异':>8}")
    print(f"  {'':->8}  {'(串行深度=N)':>10}  {'(深度=1)':>12}  {'':>8}")
    for n in [100, 1_000, 10_000]:
        print(f"  {n:>8}  {n:>10}  {n:>12}  {'相同':>8}")
    print(f"  操作数相同，但串行深度决定 GPU 时间。")


def scaling_analysis():
    """缩放分析：为什么在 2016 年 Transformer 改变了训练速度。"""
    print("\n=== 缩放分析 ===")
    print("  序列长度   RNN 串行步   Transformer 串行步   差距")
    for seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
        rnn_steps = seq_len  # RNN: 每个位置一步
        attn_steps = 1      # Attention: 一次矩阵乘法
        gap = rnn_steps / attn_steps
        bar = "█" * min(int(gap / 200), 30)
        print(f"  {seq_len:>7}  {rnn_steps:>10}  {attn_steps:>12}  {gap:>7.0f}x {bar}")
    print()
    print("  在 N=16,384 的 12 层 Transformer vs LSTM 等效物中，")
    print("  训练墙钟时间差距是 2016 年成为瓶颈的根本原因。")


def cost_analysis():
    """Transformer 的代价：O(N²) 内存。"""
    print("\n=== Transformer 的代价：O(N²) 注意力内存 ===")
    print(f"  {'序列长度':>10}  {'注意力矩阵大小':>15}  {'264MB GPU':>10}")
    for n in [512, 1024, 2048, 4096, 8192, 16384]:
        mat_size_mb = (n * n * 2 * 4) / (1024 * 1024)  # float16, 2 bytes
        fits = "✓" if mat_size_mb < 264 else "✗"
        print(f"  {n:>7}  {mat_size_mb:>12.1f} MB  {fits:>10}")
    print("  Flash Attention 解决了常数因子，但 O(N²) 缩放仍在。")


def main():
    timing_table()
    count_operations()
    scaling_analysis()
    cost_analysis()

    print("\n=== 2026 架构选择速查 ===")
    print(f"  {'场景':<25} {'选择':<20}")
    print(f"  {'流式推理（一个token一次）':<25} {'RNN/Mamba/RWKV'}")
    print(f"  {'极长序列（>1M token）':<25} {'Mamba/Hyena/线性注意力'}")
    print(f"  {'无matmul加速器的边缘设备':<25} {'深度可分离RNN'}")
    print(f"  {'其他所有场景':<25} {'Transformer'}")
    print()
    print("  注意：2026年前沿实验室训练混合 SSM+Transformer 模型（Jamba, Samba）")
    print("  递归没有消亡——它是组件。")


if __name__ == "__main__":
    main()
