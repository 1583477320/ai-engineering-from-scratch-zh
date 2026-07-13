# main.py — 位置编码完整实现
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 04（位置编码）

import numpy as np
import math


# === 正弦位置编码 ===

def sinusoidal_position_encoding(max_len, d_model):
    """正弦位置编码——每个位置一个 d_model 维向量。

    Args:
        max_len: 最大序列长度
        d_model: 嵌入维度

    Returns:
        pe: 形状 (max_len, d_model) 的位置编码矩阵
    """
    pe = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            pe[pos, i] = math.sin(angle)
            pe[pos, i + 1] = math.cos(angle)
    return pe


# === RoPE（旋转位置编码）===

def precompute_rope_frequencies(d_model, max_len, theta=10000.0):
    """预计算 RoPE 频率。

    Args:
        d_model: 嵌入维度
        max_len: 最大序列长度
        theta: 频率基数

    Returns:
        freqs: 形状 (max_len, d_model) 的复数频率
    """
    # 计算频率
    freqs = 1.0 / (theta ** (np.arange(0, d_model, 2) / d_model))

    # 计算位置-频率对
    positions = np.arange(max_len)
    angles = np.outer(positions, freqs)

    # 返回复数形式的频率（cos + i*sin）
    return np.exp(1j * angles)


def apply_rope(x, freqs):
    """应用旋转位置编码。

    Args:
        x: 输入张量，形状 (seq_len, d_model)
        freqs: 预计算的频率，形状 (seq_len, d_model//2)

    Returns:
        旋转后的张量，形状 (seq_len, d_model)
    """
    # 将 x 重塑为复数形式
    x_complex = x.view(np.complex128)

    # 应用旋转
    x_rotated = x_complex * freqs

    return x_rotated.view(np.float64)


# === 可学习位置编码 ===

class LearnedPositionEncoding:
    """可学习位置编码。"""

    def __init__(self, max_len, d_model, seed=42):
        """初始化可学习位置编码。

        Args:
            max_len: 最大序列长度
            d_model: 嵌入维度
            seed: 随机种子
        """
        rng = np.random.default_rng(seed)
        self.pe = rng.normal(0, 0.02, (max_len, d_model))

    def forward(self, seq_len):
        """获取位置编码。

        Args:
            seq_len: 序列长度

        Returns:
            位置编码，形状 (seq_len, d_model)
        """
        return self.pe[:seq_len]


# === 演示 ===

def demo():
    """演示位置编码的完整流程。"""
    print("=" * 60)
    print("位置编码 — 演示")
    print("=" * 60)

    max_len = 100
    d_model = 64

    # 正弦位置编码
    print("\n--- 正弦位置编码 ---")
    pe = sinusoidal_position_encoding(max_len, d_model)
    print(f"形状: {pe.shape} (max_len={max_len}, d_model={d_model})")
    print(f"位置 0 前 8 维: {pe[0, :8].round(3)}")
    print(f"位置 10 前 8 维: {pe[10, :8].round(3)}")
    print(f"位置 50 前 8 维: {pe[50, :8].round(3)}")

    # 可视化不同位置的编码
    print("\n--- 位置编码可视化 ---")
    print("位置 0-5 在前 4 维的编码:")
    for pos in range(6):
        print(f"  位置 {pos}: {pe[pos, :4].round(3)}")

    # 正弦编码的性质
    print("\n--- 正弦编码的性质 ---")
    # 相邻位置的编码差异
    diff_0_1 = np.linalg.norm(pe[1] - pe[0])
    diff_0_10 = np.linalg.norm(pe[10] - pe[0])
    diff_0_50 = np.linalg.norm(pe[50] - pe[0])
    print(f"位置 0 和位置 1 的距离: {diff_0_1:.3f}")
    print(f"位置 0 和位置 10 的距离: {diff_0_10:.3f}")
    print(f"位置 0 和位置 50 的距离: {diff_0_50:.3f}")

    # RoPE 频率
    print("\n--- RoPE 频率 ---")
    freqs = precompute_rope_frequencies(d_model, max_len)
    print(f"频率形状: {freqs.shape}")
    print(f"位置 0 的频率前 4 维: {freqs[0, :4]}")
    print(f"位置 10 的频率前 4 维: {freqs[10, :4]}")

    # 可学习位置编码
    print("\n--- 可学习位置编码 ---")
    learned_pe = LearnedPositionEncoding(max_len, d_model, seed=42)
    pe_learned = learned_pe.forward(10)
    print(f"形状: {pe_learned.shape}")
    print(f"位置 0 前 8 维: {pe_learned[0, :8].round(3)}")

    # 比较三种编码
    print("\n--- 三种编码比较 ---")
    print("正弦编码: 无额外参数，可外推到更长序列")
    print("RoPE: 无额外参数，相对位置信息更直接")
    print("可学习编码: 有 L×d_model 个参数，无外推能力")


if __name__ == "__main__":
    demo()
