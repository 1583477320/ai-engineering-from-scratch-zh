# main.py — KV 缓存与 FlashAttention 演示
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 12（KV 缓存与 FlashAttention）

import numpy as np


# === Softmax ===

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# === 缩放点积注意力（标准版） ===

def standard_attention(Q, K, V, mask=None):
    """标准注意力——计算完整的分数矩阵 O(n²) 内存。"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # (n, n)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores)
    output = weights @ V
    return output, weights


# === KV 缓存版因果自注意力 ===

class CausalSelfAttentionWithCache:
    """带 KV 缓存的自注意力——自回归生成时缓存历史 K/V。"""

    def __init__(self, d_model, d_k, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + d_k))
        self.Wq = rng.normal(0, scale, (d_model, d_k))
        self.Wk = rng.normal(0, scale, (d_model, d_k))
        self.Wv = rng.normal(0, scale, (d_model, d_k))

        # KV 缓存
        self.k_cache = None
        self.v_cache = None

    def clear_cache(self):
        """清空 KV 缓存。"""
        self.k_cache = None
        self.v_cache = None

    def forward_with_cache(self, x):
        """前向传播（带 KV 缓存）。

        Args:
            x: 当前词元的嵌入，形状 (d_model,)

        Returns:
            output: 注意力输出，形状 (d_k,)
        """
        # 计算当前词元的 Q/K/V
        q = x @ self.Wq  # (d_k,)
        k = x @ self.Wk  # (d_k,)
        v = x @ self.Wv  # (d_k,)

        # 更新 KV 缓存
        if self.k_cache is None:
            self.k_cache = k[np.newaxis, :]  # (1, d_k)
            self.v_cache = v[np.newaxis, :]  # (1, d_k)
        else:
            self.k_cache = np.vstack([self.k_cache, k[np.newaxis, :]])
            self.v_cache = np.vstack([self.v_cache, v[np.newaxis, :]])

        # 注意力计算（只看历史 + 当前）
        d_k = self.k_cache.shape[-1]
        scores = q @ self.k_cache.T / np.sqrt(d_k)  # (1, cache_len)
        weights = softmax(scores[np.newaxis, :])[0]  # (1, cache_len)
        output = weights @ self.v_cache  # (d_k,)
        return output

    def forward_no_cache(self, x_seq):
        """前向传播（无 KV 缓存）——每步重算全部。

        Args:
            x_seq: 整个序列的嵌入，形状 (seq_len, d_model)

        Returns:
            output: 注意力输出，形状 (seq_len, d_k)
        """
        Q = x_seq @ self.Wq
        K = x_seq @ self.Wk
        V = x_seq @ self.Wv
        d_k = Q.shape[-1]
        n = Q.shape[0]
        scores = Q @ K.T / np.sqrt(d_k)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
        weights = softmax(scores)
        output = weights @ V
        return output


# === 模拟 FlashAttention（分块计算） ===

def flash_attention_simulated(Q, K, V, block_size=2):
    """模拟 FlashAttention——分块计算注意力，不存储完整分数矩阵。

    Args:
        Q, K, V: 输入矩阵，形状 (n, d_k)
        block_size: 分块大小

    Returns:
        output: 注意力输出
    """
    n = Q.shape[0]
    d_k = Q.shape[-1]
    d_v = V.shape[-1]
    output = np.zeros((n, d_v))

    for i in range(0, n, block_size):
        q_block = Q[i:i+block_size]
        # 分块计算：每个 Q 块与所有 K 计算分数
        scores_block = q_block @ K.T / np.sqrt(d_k)  # (block_size, n)
        weights_block = softmax(scores_block)  # (block_size, n)
        # 加权求和——立即丢弃分数矩阵
        output[i:i+block_size] = weights_block @ V

    return output  # 内存 O(n)，不存储分数矩阵


# === 演示 ===

def demo():
    print("=" * 60)
    print("KV 缓存与 FlashAttention — 演示")
    print("=" * 60)

    d_model = 16
    d_k = 8
    seq_len = 6

    # --- KV 缓存演示 ---
    print("\n--- KV 缓存演示 ---")
    attn = CausalSelfAttentionWithCache(d_model, d_k, seed=42)

    # 生成 5 个词元
    rng = np.random.default_rng(42)
    tokens = [rng.randn(d_model) for _ in range(seq_len)]

    # 无缓存
    print("无 KV 缓存:")
    x_seq = np.array(tokens)
    output_no_cache = attn.forward_no_cache(x_seq)

    # 有缓存（自回归）
    print("有 KV 缓存:")
    attn.clear_cache()
    output_with_cache = []
    for i, t in enumerate(tokens):
        out = attn.forward_with_cache(t)
        output_with_cache.append(out)
    output_with_cache = np.array(output_with_cache)

    # 验证结果一致
    diff = np.abs(output_no_cache - output_with_cache).max()
    print(f"  结果差异: {diff:.6f} (应为 0)")
    print(f"  缓存大小 (生成后): {attn.k_cache.shape}")

    # 计算量对比
    print("\n--- 计算量对比 ---")
    n = 1000
    print(f"序列长度 = {n}")
    print(f"无缓存: O(n²) = {n**2/1e6:.1f}M 次操作")
    print(f"有缓存: O(n) 每步 = 约 {n*2:.1f}K 次操作")
    print(f"加速比: ~{n/2:.0f}x")

    # --- FlashAttention 演示 ---
    print("\n--- FlashAttention 演示 ---")
    np.random.seed(42)
    Q = np.random.randn(8, 4)
    K = np.random.randn(8, 4)
    V = np.random.randn(8, 4)

    output_standard, _ = standard_attention(Q, K, V)
    output_flash = flash_attention_simulated(Q, K, V, block_size=2)

    diff = np.abs(output_standard - output_flash).max()
    print(f"标准输出形状: {output_standard.shape}")
    print(f"FlashAttention 输出形状: {output_flash.shape}")
    print(f"结果差异: {diff:.6f} (应为 0)")

    # 内存对比
    print("\n--- 内存对比 ---")
    print(f"标准注意力: 存储完整 (n, n) 分数矩阵 = O(n²)")
    print(f"FlashAttention: 只存储 (block_size, n) = O(n)")
    print(f"对于 n=128K: 标准需要 {128*1024**2*4/1024**3:.1f}GB 分数矩阵")
    print(f"  FlashAttention 省略分数矩阵存储")


if __name__ == "__main__":
    demo()
