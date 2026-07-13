# main.py — 从零实现自注意力（完整版）
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 02（从零实现自注意力）

import numpy as np


# === 第 1 步：Softmax ===

def softmax(x):
    """数值稳定的 Softmax。"""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# === 第 2 步：缩放点积注意力 ===

def scaled_dot_product_attention(Q, K, V, mask=None):
    """计算缩放点积注意力。

    Args:
        Q: 查询矩阵，形状 (n, d_k)
        K: 键矩阵，形状 (n, d_k)
        V: 值矩阵，形状 (n, d_v)
        mask: 可选。掩码矩阵，形状 (n, n)。
              True 的位置将被设为 -inf

    Returns:
        output: 注意力输出，形状 (n, d_v)
        weights: 注意力权重，形状 (n, n)
    """
    d_k = Q.shape[-1]
    # 缩放因子 √d_k 防止点积过大导致 Softmax 梯度消失
    scores = Q @ K.T / np.sqrt(d_k)

    # 应用掩码（因果掩码或填充掩码）
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    weights = softmax(scores)
    output = weights @ V
    return output, weights


# === 第 3 步：带可学习投影的自注意力 ===

class SelfAttention:
    """从零实现的自注意力层。

    每个词元通过三个可学习的权重矩阵投影为 Q、K、V，
    然后计算缩放点积注意力。
    """

    def __init__(self, d_model, dk, dv, seed=42):
        """初始化自注意力层。

        Args:
            d_model: 输入嵌入维度
            dk: 查询和键的维度
            dv: 值的维度
            seed: 随机种子
        """
        rng = np.random.default_rng(seed)
        # Xavier 初始化
        scale_qk = np.sqrt(2.0 / (d_model + dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))

        self.Wq = rng.normal(0, scale_qk, (d_model, dk))
        self.Wk = rng.normal(0, scale_qk, (d_model, dk))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))

    def forward(self, X, mask=None):
        """前向传播。

        Args:
            X: 输入嵌入，形状 (n, d_model)
            mask: 可选。注意力掩码

        Returns:
            output: 注意力输出
            weights: 注意力权重矩阵
        """
        Q = X @ self.Wq  # (n, dk)
        K = X @ self.Wk  # (n, dk)
        V = X @ self.Wv  # (n, dv)
        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        return output, weights


# === 第 4 步：多头注意力 ===

class MultiHeadSelfAttention:
    """多头自注意力。

    将 Q/K/V 拆分为多个头，每个头独立计算注意力，
    拼接后通过输出投影矩阵。
    """

    def __init__(self, d_model, n_heads, seed=42):
        """初始化多头注意力。

        Args:
            d_model: 输入嵌入维度
            n_heads: 注意力头的数量
            seed: 随机种子
        """
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.dk = d_model // n_heads  # 每个头的维度

        # 每个头有独立的 Q/K/V 投影
        self.heads = [
            SelfAttention(d_model, self.dk, self.dk, seed=seed + i)
            for i in range(n_heads)
        ]

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

        for head in self.heads:
            out, weights = head.forward(X, mask)
            head_outputs.append(out)
            all_weights.append(weights)

        # 拼接所有头的输出
        concatenated = np.concatenate(head_outputs, axis=-1)  # (n, n_heads * dk)
        # 输出投影
        output = concatenated @ self.Wo  # (n, d_model)
        return output, all_weights


# === 第 5 步：因果掩码 ===

def create_causal_mask(n):
    """创建因果掩码（上三角矩阵）。

    位置 i 只能看到 0~i，不能看到 i+1~n。

    Args:
        n: 序列长度

    Returns:
        mask: 形状 (n, n) 的布尔掩码，True 表示被遮挡
    """
    # 上三角为 True，对角线及以下为 False
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return mask


# === 演示 ===

def demo():
    """演示自注意力的完整流程。"""
    print("=" * 60)
    print("从零实现自注意力 — 演示")
    print("=" * 60)

    # 句子：6 个词元，每个 16 维
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    d_model = 16
    n_heads = 4

    # 随机生成嵌入（实际中来自词元嵌入层）
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (len(sentence), d_model))

    print(f"\n句子: {sentence}")
    print(f"输入形状: {X.shape} (n={len(sentence)}, d_model={d_model})")

    # 单头自注意力
    print("\n--- 单头自注意力 ---")
    attn = SelfAttention(d_model, dk=8, dv=8, seed=42)
    output, weights = attn.forward(X)
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    print(f"权重每行之和（应为 1.0）: {weights.sum(axis=-1).round(3)}")

    # 多头自注意力
    print("\n--- 多头自注意力 ---")
    mha = MultiHeadSelfAttention(d_model, n_heads, seed=42)
    mha_output, mha_weights = mha.forward(X)
    print(f"输出形状: {mha_output.shape}")
    print(f"头数: {n_heads}，每个头的维度: {d_model // n_heads}")
    print(f"每个头的注意力权重形状: {mha_weights[0].shape}")

    # 因果掩码
    print("\n--- 因果掩码（解码器模式）---")
    mask = create_causal_mask(len(sentence))
    print("掩码矩阵（True = 被遮挡）:")
    print(mask.astype(int))

    # 带因果掩码的自注意力
    output_causal, weights_causal = attn.forward(X, mask=mask)
    print("\n带因果掩码的注意力权重:")
    print("位置 0（The）只关注自己:")
    print(f"  权重: {weights_causal[0].round(3)}")
    print("位置 5（mat）关注所有词:")
    print(f"  权重: {weights_causal[5].round(3)}")

    # 注意力可视化
    print("\n--- 注意力权重可视化 ---")
    print("（每行：该词关注哪些词）")
    for i, word in enumerate(sentence):
        row = weights_causal[i]
        # 用 ASCII 可视化
        bars = "".join("█" if w > 0.15 else "▏" for w in row)
        print(f"  {word:>6} |{bars}| {row.round(2)}")


if __name__ == "__main__":
    demo()
