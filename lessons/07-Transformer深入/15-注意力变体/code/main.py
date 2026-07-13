# main.py — 注意力变体教学实现
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 15（注意力变体）

import numpy as np


# === Softmax ===

def softmax(x):
    """数值稳定的 Softmax。"""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# === 1. 标准缩放点积注意力（基线） ===

def standard_attention(Q, K, V, mask=None):
    """标准缩放点积注意力。复杂度 O(N²)。

    Args:
        Q/K/V: 形状 (n, d_k) 或 (n, d_v)，mask 为 True 的位置被屏蔽
    Returns:
        output: (n, d_v)，weights: (n, n)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores)
    return weights @ V, weights


# === 2. 线性注意力（核函数方法） ===

def linear_attention(Q, K, V):
    """线性注意力——用核函数替代 softmax，复杂度 O(N)。

    核心：利用结合律 (φ(Q)φ(K)ᵀ)V = φ(Q)(φ(K)ᵀV)，
    先算 φ(K)ᵀV 得到固定大小矩阵 (d_k, d_v)，再乘 φ(Q)。
    """
    phi_Q = np.maximum(Q, 0) + 0.01   # φ(x) = ReLU(x) + ε
    phi_K = np.maximum(K, 0) + 0.01
    # φ(K)ᵀV 先算：形状 (d_k, d_v)，不随序列长度增长
    KV = phi_K.T @ V
    output = phi_Q @ KV
    # 用 φ(K) 的行和做归一化
    denom = phi_Q @ phi_K.sum(axis=0)
    return output / denom[:, np.newaxis]


# === 3. 滑动窗口注意力 ===

def sliding_window_attention(Q, K, V, window_size):
    """滑动窗口注意力——每个位置只关注 W 个邻居，复杂度 O(NW)。"""
    n = Q.shape[0]
    output = np.zeros_like(V)
    half = window_size // 2
    for i in range(n):
        start, end = max(0, i - half), min(n, i + half + 1)
        scores = Q[i] @ K[start:end].T / np.sqrt(Q.shape[-1])
        output[i] = softmax(scores) @ V[start:end]
    return output


# === 4. 分组查询注意力（GQA） ===

class GroupedQueryAttention:
    """分组查询注意力——多个 Q 头共享同一组 KV。

    GQA(n_kv_groups=1) = MQA（多查询注意力）
    GQA(n_kv_groups=n_heads) = MHA（标准多头注意力）
    """

    def __init__(self, d_model, n_heads, n_kv_groups, seed=42):
        assert n_heads % n_kv_groups == 0
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.group_size = n_heads // n_kv_groups
        self.d_k = d_model // n_heads

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        # 每个头有独立的 Q 投影，每组共享一对 K/V 投影
        self.Wq = [rng.normal(0, scale, (d_model, self.d_k)) for _ in range(n_heads)]
        self.Wk = [rng.normal(0, scale, (d_model, self.d_k)) for _ in range(n_kv_groups)]
        self.Wv = [rng.normal(0, scale, (d_model, self.d_k)) for _ in range(n_kv_groups)]
        self.Wo = rng.normal(0, scale, (n_heads * self.d_k, d_model))

    def forward(self, X):
        """前向传播。X: (n, d_model) -> output: (n, d_model)"""
        head_outputs = []
        for i in range(self.n_heads):
            g = i // self.group_size
            Q = X @ self.Wq[i]
            K = X @ self.Wk[g]       # 同组共享 K
            V = X @ self.Wv[g]       # 同组共享 V
            scores = Q @ K.T / np.sqrt(Q.shape[-1])
            head_outputs.append(softmax(scores) @ V)
        return np.concatenate(head_outputs, axis=-1) @ self.Wo


# === 5. 稀疏注意力 ===

def sparse_attention(Q, K, V, top_k):
    """稀疏注意力——每个位置只关注分数最高的 top_k 个位置。"""
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    output = np.zeros_like(V)
    for i in range(Q.shape[0]):
        top_idx = np.argsort(scores[i])[-top_k:]
        output[i] = softmax(scores[i, top_idx]) @ V[top_idx]
    return output


# === 演示 ===

def demo():
    print("=" * 60)
    print("注意力变体 — 演示")
    print("=" * 60)

    # 公共参数
    seq_len, d_model, d_k = 8, 16, 4
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (seq_len, d_model))
    Q = X @ rng.normal(0, 0.5, (d_model, d_k))
    K = X @ rng.normal(0, 0.5, (d_model, d_k))
    V = X @ rng.normal(0, 0.5, (d_model, d_k))

    print(f"\n序列长度: {seq_len}, d_model: {d_model}, d_k: {d_k}\n")

    # 1. 标准注意力（基线）
    out, weights = standard_attention(Q, K, V)
    print(f"1. 标准注意力 (O(N²))")
    print(f"   输出形状: {out.shape}, 权重形状: {weights.shape}")
    print(f"   权重每行和（应为 1）: {weights.sum(axis=-1)[:3].round(4)}")

    # 2. 线性注意力
    out_lin = linear_attention(Q, K, V)
    print(f"\n2. 线性注意力 (O(N))")
    print(f"   与标准注意力的平均差异: {np.abs(out - out_lin).mean():.4f}")

    # 3. 滑动窗口 + 因果掩码
    out_sw = sliding_window_attention(Q, K, V, window_size=4)
    print(f"\n3. 滑动窗口注意力 (窗口=4, O(NW))")
    print(f"   输出形状: {out_sw.shape}")

    # 因果掩码演示
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    _, w_causal = standard_attention(Q, K, V, mask=mask)
    print(f"   因果掩码示例（前 4 个位置的权重）:")
    for i in range(4):
        row = " ".join(f"{w:.2f}" for w in w_causal[i])
        print(f"     位置 {i}: [{row}]")

    # 4. GQA vs MQA vs MHA
    n_heads = 8
    configs = [("MHA (全量 KV)", 8), ("GQA (分组 KV)", 2), ("MQA (共享 KV)", 1)]
    print(f"\n4. KV 共享策略对比 (n_heads={n_heads})")
    print(f"   {'策略':<18}{'KV 组数':<8}{'KV 缓存':<8}{'KV 投影参数'}")
    print(f"   {'-' * 48}")
    for label, n_kv in configs:
        gqa = GroupedQueryAttention(d_model, n_heads, n_kv, seed=42)
        gqa.forward(X)
        kv_params = n_kv * (d_model * gqa.d_k) * 2
        print(f"   {label:<18}{n_kv:<8}{n_kv:<8}{kv_params:,}")

    # 5. 稀疏注意力
    top_k = 3
    out_sp = sparse_attention(Q, K, V, top_k)
    print(f"\n5. 稀疏注意力 (top_k={top_k})")
    print(f"   每位置只计算 {top_k}/{seq_len} 个分数 ({top_k/seq_len:.0%} 比例)")

    # 复杂度总结
    print(f"\n--- 复杂度总结 ---")
    print(f"{'变体':<20}{'复杂度':<12}{'KV 缓存'}")
    print("-" * 42)
    print(f"{'标准注意力 (MHA)':<16}{'O(N²)':<12}{n_heads}")
    print(f"{'线性注意力':<16}{'O(N)':<12}{n_heads}")
    print(f"{'滑动窗口':<16}{'O(NW)':<12}{n_heads}")
    print(f"{'GQA':<18}{'O(N²)':<12}2")
    print(f"{'MQA':<19}{'O(N²)':<12}1")


if __name__ == "__main__":
    demo()
