# main.py — 多头注意力完整实现
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 03（多头注意力）

import numpy as np


# === Softmax ===

def softmax(x):
    """数值稳定的 Softmax。"""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# === 缩放点积注意力 ===

def scaled_dot_product_attention(Q, K, V, mask=None):
    """计算缩放点积注意力。

    Args:
        Q: 查询矩阵，形状 (n, d_k)
        K: 键矩阵，形状 (n, d_k)
        V: 值矩阵，形状 (n, d_v)
        mask: 可选。掩码矩阵

    Returns:
        output: 注意力输出，形状 (n, d_v)
        weights: 注意力权重，形状 (n, n)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores)
    output = weights @ V
    return output, weights


# === 多头注意力 ===

class MultiHeadSelfAttention:
    """多头自注意力层。

    将 Q/K/V 拆分为 n_heads 个子空间，每个子空间独立计算注意力，
    拼接后通过输出投影矩阵 Wo 映射回原始维度。
    """

    def __init__(self, d_model, n_heads, seed=42):
        """初始化多头注意力层。

        Args:
            d_model: 输入嵌入维度
            n_heads: 注意力头的数量
            seed: 随机种子
        """
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.dk = d_model // n_heads  # 每个头的维度

        # 每个头有独立的 Q/K/V 投影
        self.heads = []
        for i in range(n_heads):
            rng = np.random.default_rng(seed + i)
            scale = np.sqrt(2.0 / (d_model + self.dk))
            Wq = rng.normal(0, scale, (d_model, self.dk))
            Wk = rng.normal(0, scale, (d_model, self.dk))
            Wv = rng.normal(0, scale, (d_model, self.dk))
            self.heads.append((Wq, Wk, Wv))

        # 输出投影矩阵
        rng = np.random.default_rng(seed + n_heads)
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.Wo = rng.normal(0, scale, (n_heads * self.dk, d_model))

    def forward(self, X, mask=None):
        """前向传播。

        Args:
            X: 输入嵌入，形状 (n, d_model)
            mask: 可选。注意力掩码

        Returns:
            output: 注意力输出，形状 (n, d_model)
            all_weights: 每个头的注意力权重列表
        """
        head_outputs = []
        all_weights = []

        for Wq, Wk, Wv in self.heads:
            Q = X @ Wq
            K = X @ Wk
            V = X @ Wv
            output, weights = scaled_dot_product_attention(Q, K, V, mask)
            head_outputs.append(output)
            all_weights.append(weights)

        # 拼接所有头的输出
        concatenated = np.concatenate(head_outputs, axis=-1)  # (n, n_heads * dk)
        # 输出投影
        final_output = concatenated @ self.Wo  # (n, d_model)
        return final_output, all_weights


# === 注意力头分析器 ===

class AttentionHeadAnalyzer:
    """分析每个注意力头的学习模式。"""

    def __init__(self, labels):
        """初始化分析器。

        Args:
            labels: 词元标签列表
        """
        self.labels = labels

    def analyze(self, all_weights):
        """分析每个头的注意力模式。

        Args:
            all_weights: 每个头的注意力权重列表

        Returns:
            analysis: 每个头的分析结果列表
        """
        analysis = []
        for head_idx, weights in enumerate(all_weights):
            # 计算平均注意力范围（每个位置平均关注多少个其他位置）
            avg_range = self._compute_avg_range(weights)

            # 计算最大注意力跨度（最远的关注距离）
            max_span = self._compute_max_span(weights)

            # 计算注意力熵（分布的分散程度）
            entropy = self._compute_entropy(weights)

            analysis.append({
                "head": head_idx,
                "avg_range": avg_range,
                "max_span": max_span,
                "entropy": entropy,
            })
        return analysis

    def _compute_avg_range(self, weights):
        """计算平均注意力范围。"""
        n = weights.shape[0]
        ranges = []
        for i in range(n):
            # 找到权重 > 0.1 的位置
            important = np.where(weights[i] > 0.1)[0]
            if len(important) > 0:
                ranges.append(important[-1] - important[0] + 1)
        return np.mean(ranges) if ranges else 0

    def _compute_max_span(self, weights):
        """计算最大注意力跨度。"""
        n = weights.shape[0]
        max_span = 0
        for i in range(n):
            important = np.where(weights[i] > 0.1)[0]
            if len(important) > 0:
                span = important[-1] - important[0]
                max_span = max(max_span, span)
        return max_span

    def _compute_entropy(self, weights):
        """计算注意力熵。"""
        # 避免 log(0)
        safe_weights = np.clip(weights, 1e-10, 1.0)
        # 熵 = -sum(p * log(p))
        entropy = -np.sum(safe_weights * np.log(safe_weights), axis=-1)
        return np.mean(entropy)


# === 因果掩码 ===

def create_causal_mask(n):
    """创建因果掩码。"""
    return np.triu(np.ones((n, n), dtype=bool), k=1)


# === 演示 ===

def demo():
    """演示多头注意力的完整流程。"""
    print("=" * 60)
    print("多头注意力 — 演示")
    print("=" * 60)

    # 句子：6 个词元，每个 16 维
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    d_model = 16
    n_heads = 4

    # 随机生成嵌入
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (len(sentence), d_model))

    print(f"\n句子: {sentence}")
    print(f"输入形状: {X.shape} (n={len(sentence)}, d_model={d_model})")
    print(f"头数: {n_heads}，每个头的维度: {d_model // n_heads}")

    # 多头自注意力
    print("\n--- 多头自注意力 ---")
    mha = MultiHeadSelfAttention(d_model, n_heads, seed=42)
    output, all_weights = mha.forward(X)
    print(f"输出形状: {output.shape}")

    # 每个头的注意力模式
    print("\n--- 每个头的注意力权重 ---")
    for head_idx, weights in enumerate(all_weights):
        print(f"\n头 {head_idx + 1}:")
        for i, word in enumerate(sentence):
            row = weights[i]
            bars = "".join("█" if w > 0.15 else "▏" for w in row)
            print(f"  {word:>6} |{bars}|")

    # 注意力头分析
    print("\n--- 注意力头分析 ---")
    analyzer = AttentionHeadAnalyzer(sentence)
    analysis = analyzer.analyze(all_weights)
    for a in analysis:
        print(f"头 {a['head'] + 1}: 平均范围={a['avg_range']:.2f}, "
              f"最大跨度={a['max_span']}, 熵={a['entropy']:.3f}")

    # 因果掩码
    print("\n--- 因果掩码（解码器模式）---")
    mask = create_causal_mask(len(sentence))
    output_causal, weights_causal = mha.forward(X, mask=mask)
    print("带因果掩码的注意力权重（头 1）:")
    for i, word in enumerate(sentence):
        row = weights_causal[0][i]
        bars = "".join("█" if w > 0.15 else "▏" for w in row)
        print(f"  {word:>6} |{bars}|")

    # 参数量计算
    print("\n--- 参数量计算 ---")
    print(f"d_model = {d_model}")
    print(f"n_heads = {n_heads}")
    print(f"dk = dv = {d_model // n_heads}")

    # 每头参数
    per_head_params = 3 * (d_model * (d_model // n_heads))
    total_head_params = n_heads * per_head_params
    wo_params = (n_heads * (d_model // n_heads)) * d_model
    total_params = total_head_params + wo_params

    print(f"每头参数 (Wq + Wk + Wv): {per_head_params:,}")
    print(f"八头总计: {total_head_params:,}")
    print(f"输出投影 Wo: {wo_params:,}")
    print(f"多头注意力总参数: {total_params:,}")


if __name__ == "__main__":
    demo()
