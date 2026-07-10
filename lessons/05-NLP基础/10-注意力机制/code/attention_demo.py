# attention_demo.py — 注意力机制：Bahdanau加性 + Luong点积 + 数值演示 + QKV过渡
# 依赖：numpy>=1.24
# 安装：pip install numpy
# 对应课程：阶段 05 · 10（注意力机制）

import numpy as np
from typing import Tuple


# ============================================================
# 1. Softmax
# ============================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax——减去最大值保证数值稳定性。"""
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


# ============================================================
# 2. Bahdanau 加性注意力
# ============================================================

def additive_attention(decoder_state: np.ndarray,   # (d_s,)
                       encoder_states: np.ndarray,   # (T_enc, d_h)
                       W_a: np.ndarray,              # (d_attn, d_s)
                       U_a: np.ndarray,              # (d_attn, d_h)
                       v_a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bahdanau 加性注意力。

    得分：e_i = v_a^T · tanh(W_a·s + U_a·h_i)

    v_a 的作用：将 d_attn 维的中间表示投影为单一的标量分数。
    它不是魔法——只是一个学到的线性投影，回答"这个编码器位置有多重要"。
    """
    # (T_enc, d_attn) + (d_attn,) → (T_enc, d_attn)
    projected_dec = W_a @ decoder_state          # (d_attn,)
    projected_enc = encoder_states @ U_a.T       # (T_enc, d_attn)
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a                      # (T_enc,)
    weights = softmax(scores)                    # (T_enc,)
    context = weights @ encoder_states           # (d_h,)
    return context, weights


# ============================================================
# 3. Luong 点积注意力（三行代码）
# ============================================================

def dot_attention(decoder_state: np.ndarray,   # (d_h,)
                  encoder_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Luong 点积注意力——三行，要求 d_s == d_h。"""
    scores = encoder_states @ decoder_state      # (T_enc,)
    weights = softmax(scores)
    return weights @ encoder_states, weights    # context (d_h,), weights (T_enc,)


# ============================================================
# 4. Luong 一般注意力（带可学习投影，解除维度约束）
# ============================================================

def general_attention(decoder_state: np.ndarray,  # (d_s,)
                      encoder_states: np.ndarray,  # (T_enc, d_h)
                      W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Luong 一般注意力——W 矩阵解除 d_s == d_h 的约束。"""
    projected = W.T @ decoder_state              # (d_h,)
    scores = encoder_states @ projected          # (T_enc,)
    weights = softmax(scores)
    return weights @ encoder_states, weights


# ============================================================
# 5. 数值演示——注意力如何对齐
# ============================================================

def demo_numerical():
    """展示注意力权重随查询变化而转移。"""
    print("=" * 60)
    print("注意力数值演示——权重随查询移动")
    print("=" * 60)

    # 三个编码器状态——分别代表 "猫"(0), "坐"(1), "垫子"(2)
    H = np.array([
        [1.0, 0.0, 0.2],   # cat
        [0.5, 0.5, 0.1],   # sat
        [0.1, 0.9, 0.3],   # mat
    ])

    # 查询接近"猫"——注意力应该集中在位置 0
    s_cat = np.array([0.9, 0.1, 0.2])
    ctx, w = dot_attention(s_cat, H)
    print(f"查询接近 '猫' {s_cat}:")
    print(f"  注意力权重: {np.round(w, 3)}")
    print(f"  上下文向量: {np.round(ctx, 3)}")
    print(f"  → 位置 0 (猫) 权重最高 = {w[0]:.1%}")

    # 查询接近"垫子"——注意力应该转移到位置 2
    s_mat = np.array([0.2, 0.8, 0.4])
    ctx2, w2 = dot_attention(s_mat, H)
    print(f"\n查询接近 '垫子' {s_mat}:")
    print(f"  注意力权重: {np.round(w2, 3)}")
    print(f"  → 位置 2 (垫子) 权重 = {w2[2]:.1%}")
    print(f"  → 注意力从位置 0 转移到了位置 2")


# ============================================================
# 6. Q/K/V 过渡——从经典注意力到自注意力
# ============================================================

def demo_qkv_bridge():
    """将经典注意力的语言翻译为 Q/K/V。"""
    print("\n" + "=" * 60)
    print("从经典注意力到 Q/K/V——同样的数学，不同的名字")
    print("=" * 60)

    print("""
    经典注意力 (Bahdanau/Luong):
      解码器状态  ×  编码器状态  →  分数  →  softmax  →  加权和
         ↑              ↑
       "查询"        "被查询的内容"

    自注意力 (Self-Attention):
      查询(Q)  ×  键(K)  →  分数  →  softmax  →  加权求和 值(V)
        ↑          ↑                                  ↑
      "我要找什么"  "我含有什么"                "我提供什么信息"

    关键区别：
      - 经典注意力: K 和 V 是同一组编码器状态（不做区分）
      - 自注意力: K 和 V 可以来自不同的投影（不同的权重矩阵）
      - 缩放因子: 自注意力加了 √d_k 防止大维度点积饱和

    数学本质完全相同——加权平均，权重来自查询和键的相似度。
    """)

    # 用 NumPy 模拟自注意力（Q/K/V 来自同一输入）
    np.random.seed(42)
    seq_len, d_k, d_v = 4, 3, 3
    X = np.random.randn(seq_len, 4)  # 4 个词元，4 维嵌入
    Wq = np.random.randn(4, d_k) * 0.1
    Wk = np.random.randn(4, d_k) * 0.1
    Wv = np.random.randn(4, d_v) * 0.1

    Q = X @ Wq  # (4, 3)
    K = X @ Wk  # (4, 3)
    V = X @ Wv  # (4, 3)

    # 缩放点积注意力
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.array([softmax(row) for row in scores])
    output = weights @ V

    print(f"输入 X shape: {X.shape}")
    print(f"Q/K/V shape:  {Q.shape} / {K.shape} / {V.shape}")
    print(f"注意力权重 (4个词元互相看):")
    print(np.round(weights, 2))
    print(f"\n这就是 Transformer 的注意力层。只是加了多头、堆叠了层数。")
    print(f"从 Bahdanau 到这里——数学没有变，只是应用方式从")
    print(f"'解码器看编码器'变成了'序列自己看自己'。")


# ============================================================
# 演示主程序
# ============================================================

def main():
    demo_numerical()

    # 形状检查
    print("\n--- Bahdanau 加性注意力形状检查 ---")
    T_enc, d_h, d_s, d_attn = 5, 8, 6, 10
    rng = np.random.default_rng(42)
    H = rng.normal(0, 1, (T_enc, d_h))
    s = rng.normal(0, 1, (d_s,))
    W_a = rng.normal(0, 0.1, (d_attn, d_s))
    U_a = rng.normal(0, 0.1, (d_attn, d_h))
    v_a = rng.normal(0, 0.1, (d_attn,))

    ctx, w = additive_attention(s, H, W_a, U_a, v_a)
    print(f"编码器状态: {H.shape}")
    print(f"解码器状态: {s.shape}")
    print(f"上下文向量: {ctx.shape} (应与编码器状态维度一致: {d_h})")
    print(f"注意力权重: {w.shape} (5个编码器位置各一个权重)")
    assert ctx.shape == (d_h,), f"上下文形状错误: {ctx.shape}"
    assert abs(w.sum() - 1.0) < 1e-9, f"权重和≠1: {w.sum()}"
    print("✓ 形状检查通过，权重和为 1.0")

    demo_qkv_bridge()


if __name__ == "__main__":
    main()
