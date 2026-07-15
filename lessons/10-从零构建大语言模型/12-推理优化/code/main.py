# 推理优化——KV缓存、FlashAttention、推测解码

import numpy as np
import time


# ============================================================================
# 第 1 步：标准注意力 vs FlashAttention 模拟
# ============================================================================

def standard_attention(Q, K, V):
    """标准缩放点积注意力——O(n²) HBM 读写。"""
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V


def flash_attention(Q, K, V, block_size=4):
    """模拟 FlashAttention——分块 IO 感知注意力。"""
    n = Q.shape[0]
    d = Q.shape[-1]
    output = np.zeros_like(Q)

    for i in range(0, n, block_size):
        Q_block = Q[i:i+block_size]
        for j in range(0, n, block_size):
            K_block = K[j:j+block_size]
            V_block = V[j:j+block_size]
            scores = Q_block @ K_block.T / np.sqrt(d)
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights = weights / weights.sum(axis=-1, keepdims=True)
            output[i:i+block_size] += weights @ V_block

    return output


def benchmark_attention(seq_len=128, dim=64):
    """对比注意力实现。"""
    Q = np.random.randn(seq_len, dim)
    K = np.random.randn(seq_len, dim)
    V = np.random.randn(seq_len, dim)

    start = time.time()
    out_std = standard_attention(Q, K, V)
    t_std = time.time() - start

    start = time.time()
    out_flash = flash_attention(Q, K, V, block_size=8)
    t_flash = time.time() - start

    err = np.abs(out_std - out_flash).max()
    print(f"  序列长度 {seq_len}: 标准={t_std:.4f}s Flash={t_flash:.4f}s (加速={t_std/t_flash:.1f}x)")


# ============================================================================
# 第 2 步：KV 缓存模拟
# ============================================================================

class KVCache:
    """KV 缓存——避免重复计算历史 K/V。"""
    def __init__(self, dim=64, max_len=1024):
        self.k_cache = np.zeros((max_len, dim))
        self.v_cache = np.zeros((max_len, dim))
        self.pos = 0
        self.max_len = max_len

    def update(self, K, V):
        """缓存新的 K/V。"""
        n = K.shape[0]
        if self.pos + n <= self.max_len:
            self.k_cache[self.pos:self.pos+n] = K
            self.v_cache[self.pos:self.pos+n] = V
            self.pos += n

    def get(self):
        """获取所有缓存的 K/V。"""
        return self.k_cache[:self.pos], self.v_cache[:self.pos]


def simulate_kv_cache(seq_len=128, dim=64):
    """KV 缓存加速模拟。"""
    # 无 KV 缓存：每步重新计算所有历史
    t_no_cache = 0
    for step in range(seq_len):
        q = np.random.randn(1, dim)
        k = np.random.randn(step+1, dim)
        v = np.random.randn(step+1, dim)
        start = time.time()
        attn = q @ k.T
        t_no_cache += time.time() - start

    # 有 KV 缓存：只计算当前步
    cache = KVCache(dim, seq_len)
    t_cache = 0
    for step in range(seq_len):
        q = np.random.randn(1, dim)
        k = np.random.randn(1, dim)
        v = np.random.randn(1, dim)
        cache.update(k, v)
        k_all, v_all = cache.get()
        start = time.time()
        attn = q @ k_all.T
        t_cache += time.time() - start

    print(f"  无 KV 缓存: {t_no_cache:.4f}s, 有 KV 缓存: {t_cache:.4f}s (加速={t_no_cache/t_cache:.1f}x)")


# ============================================================================
# 第 3 步：推测解码模拟
# ============================================================================

def speculative_decoding(draft_model, target_model, prompt_tokens, n_draft=5, max_new=20):
    """模拟推测解码——小模型生成候选，大模型验证。"""
    tokens = list(prompt_tokens)

    while len(tokens) - len(prompt_tokens) < max_new:
        # 草稿模型生成 N 个候选
        draft_tokens = draft_model(tokens[-32:], n_draft)

        # 大模型并行验证
        valid = True
        for n, dt in enumerate(draft_tokens):
            real_next = target_model(tokens[-32:], 1)[0]
            if dt == real_next:
                tokens.append(dt)
            else:
                tokens.append(real_next)
                valid = False
                break

        if valid:
            tokens.extend(draft_tokens)

    return tokens


def draft_generator(context, n_tokens):
    """简化草稿模型——快速生成候选。"""
    return [np.random.randint(0, 100) for _ in range(n_tokens)]


def target_generator(context, n_tokens):
    """简化目标模型——验证候选。"""
    return [np.random.randint(0, 100) for _ in range(n_tokens)]


# ============================================================================
# 第 4 步：连续批处理模拟
# ============================================================================

def continuous_batching(requests, batch_size=4):
    """模拟连续批处理——动态添加/移除请求。"""
    queue = list(requests)
    active = []
    completed = 0
    total_steps = 0

    while queue or active:
        # 添加新请求到活跃集
        while len(active) < batch_size and queue:
            active.append(queue.pop(0))

        # 模拟处理一个批次
        total_steps += 1
        for req in list(active):
            req["progress"] += 1
            if req["progress"] >= req["length"]:
                active.remove(req)
                completed += 1

    return completed, total_steps


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("推理优化演示")
    print("=" * 60)

    # 1. 注意力对比
    print("\n1. 标准注意力 vs FlashAttention")
    for sl in [32, 64, 128]:
        benchmark_attention(seq_len=sl, dim=64)

    # 2. KV 缓存
    print("\n2. KV 缓存效果")
    simulate_kv_cache(seq_len=128, dim=64)

    # 3. 推测解码
    print("\n3. 推测解码模拟")
    tokens = speculative_decoding(draft_generator, target_generator, [1, 2, 3], n_draft=5, max_new=10)
    print(f"  生成 {len(tokens)-3} 个 token")

    # 4. 连续批处理
    print("\n4. 连续批处理")
    requests = [{"id": i, "length": np.random.randint(2, 8), "progress": 0}
                for i in range(10)]
    completed, steps = continuous_batching(requests, batch_size=4)
    print(f"  10 请求, batch=4: {steps} 步完成 (传统批处理需要 {sum(r['length'] for r in requests)//4+1} 步)")

    print("\n完成！")
