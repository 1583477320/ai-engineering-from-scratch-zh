# main.py — 完整 Transformer 块实现
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 05（完整 Transformer）

import numpy as np


# === Softmax ===

def softmax(x):
    """数值稳定的 Softmax。"""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# === 缩放点积注意力 ===

def scaled_dot_product_attention(Q, K, V, mask=None):
    """计算缩放点积注意力。"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores)
    output = weights @ V
    return output, weights


# === 多头注意力 ===

class MultiHeadSelfAttention:
    """多头自注意力层。"""

    def __init__(self, d_model, n_heads, seed=42):
        """初始化多头注意力层。"""
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.dk = d_model // n_heads

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
        """前向传播。"""
        head_outputs = []
        for Wq, Wk, Wv in self.heads:
            Q = X @ Wq
            K = X @ Wk
            V = X @ Wv
            output, _ = scaled_dot_product_attention(Q, K, V, mask)
            head_outputs.append(output)

        concatenated = np.concatenate(head_outputs, axis=-1)
        return concatenated @ self.Wo


# === 层归一化 ===

class LayerNorm:
    """层归一化。"""

    def __init__(self, d_model, eps=1e-8):
        """初始化层归一化。"""
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        """前向传播。"""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + self.eps
        return self.gamma * (x - mean) / std + self.beta


# === 前馈网络 ===

class FeedForward:
    """前馈网络：d_model → d_ff → d_model。"""

    def __init__(self, d_model, d_ff, seed=42):
        """初始化前馈网络。"""
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        self.W1 = rng.normal(0, scale1, (d_model, d_ff))
        self.W2 = rng.normal(0, scale2, (d_ff, d_model))

    def forward(self, x):
        """前向传播（GELU 激活）。"""
        # GELU 近似：max(0, x) * (1 + tanh(sqrt(2/π) * (x + 0.044715x³)))
        linear1 = x @ self.W1
        gelu = 0.5 * linear1 * (1 + np.tanh(np.sqrt(2 / np.pi) * (linear1 + 0.044715 * linear1**3)))
        return gelu @ self.W2


# === Transformer 块 ===

class TransformerBlock:
    """完整的 Transformer 块。"""

    def __init__(self, d_model, n_heads, d_ff, seed=42):
        """初始化 Transformer 块。"""
        self.mha = MultiHeadSelfAttention(d_model, n_heads, seed)
        self.ffn = FeedForward(d_model, d_ff, seed + 100)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """前向传播。"""
        # 自注意力 + 残差连接 + 层归一化
        attn_out = self.mha.forward(x, mask)
        x = self.ln1.forward(x + attn_out)

        # 前馈网络 + 残差连接 + 层归一化
        ffn_out = self.ffn.forward(x)
        x = self.ln2.forward(x + ffn_out)

        return x


# === Transformer 编码器 ===

class TransformerEncoder:
    """Transformer 编码器（多层堆叠）。"""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len=512, seed=42):
        """初始化 Transformer 编码器。"""
        self.d_model = d_model

        # 词元嵌入
        rng = np.random.default_rng(seed)
        self.embedding = rng.normal(0, 0.02, (vocab_size, d_model))

        # 正弦位置编码
        self.positional_encoding = self._sinusoidal_encoding(max_len, d_model)

        # Transformer 块堆叠
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff, seed + i * 10)
            for i in range(n_layers)
        ]

    def _sinusoidal_encoding(self, max_len, d_model):
        """生成正弦位置编码。"""
        pe = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i / d_model))
                pe[pos, i] = np.sin(angle)
                pe[pos, i + 1] = np.cos(angle)
        return pe

    def forward(self, input_ids, mask=None):
        """前向传播。"""
        seq_len = input_ids.shape[0]

        # 词元嵌入 + 位置编码
        x = self.embedding[input_ids] + self.positional_encoding[:seq_len]

        # 通过所有层
        for layer in self.layers:
            x = layer.forward(x, mask)

        return x


# === Transformer 解码器 ===

class TransformerDecoder:
    """Transformer 解码器（带因果掩码）。"""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len=512, seed=42):
        """初始化 Transformer 解码器。"""
        self.d_model = d_model

        # 词元嵌入
        rng = np.random.default_rng(seed)
        self.embedding = rng.normal(0, 0.02, (vocab_size, d_model))

        # 正弦位置编码
        self.positional_encoding = self._sinusoidal_encoding(max_len, d_model)

        # Transformer 块堆叠（带因果掩码）
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff, seed + i * 10)
            for i in range(n_layers)
        ]

    def _sinusoidal_encoding(self, max_len, d_model):
        """生成正弦位置编码。"""
        pe = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i / d_model))
                pe[pos, i] = np.sin(angle)
                pe[pos, i + 1] = np.cos(angle)
        return pe

    def _create_causal_mask(self, n):
        """创建因果掩码。"""
        return np.triu(np.ones((n, n), dtype=bool), k=1)

    def forward(self, input_ids):
        """前向传播。"""
        seq_len = input_ids.shape[0]

        # 词元嵌入 + 位置编码
        x = self.embedding[input_ids] + self.positional_encoding[:seq_len]

        # 创建因果掩码
        mask = self._create_causal_mask(seq_len)

        # 通过所有层
        for layer in self.layers:
            x = layer.forward(x, mask)

        return x


# === 演示 ===

def demo():
    """演示完整 Transformer 的完整流程。"""
    print("=" * 60)
    print("完整 Transformer — 演示")
    print("=" * 60)

    # 参数设置
    vocab_size = 1000
    d_model = 64
    n_heads = 4
    d_ff = 128
    n_layers = 2
    seq_len = 10

    print(f"\n参数设置:")
    print(f"  vocab_size = {vocab_size}")
    print(f"  d_model = {d_model}")
    print(f"  n_heads = {n_heads}")
    print(f"  d_ff = {d_ff}")
    print(f"  n_layers = {n_layers}")
    print(f"  seq_len = {seq_len}")

    # Transformer 编码器
    print("\n--- Transformer 编码器 ---")
    encoder = TransformerEncoder(vocab_size, d_model, n_heads, d_ff, n_layers)
    input_ids = np.random.randint(0, vocab_size, seq_len)
    output = encoder.forward(input_ids)
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出均值: {output.mean():.4f}")
    print(f"输出标准差: {output.std():.4f}")

    # Transformer 解码器
    print("\n--- Transformer 解码器 ---")
    decoder = TransformerDecoder(vocab_size, d_model, n_heads, d_ff, n_layers)
    output = decoder.forward(input_ids)
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {output.shape}")

    # 参数量计算
    print("\n--- 参数量计算 ---")
    total_params = 0

    # 词元嵌入
    embed_params = vocab_size * d_model
    total_params += embed_params
    print(f"词元嵌入: {embed_params:,}")

    # 位置编码（正弦，无参数）
    print(f"位置编码: 0（正弦编码，无参数）")

    # 每层参数
    per_layer_params = 0

    # 多头注意力
    mha_params = 4 * d_model * d_model  # Wq, Wk, Wv, Wo
    per_layer_params += mha_params
    print(f"每层多头注意力: {mha_params:,}")

    # FFN
    ffn_params = 2 * d_model * d_ff  # W1, W2
    per_layer_params += ffn_params
    print(f"每层 FFN: {ffn_params:,}")

    # 层归一化（每层 2 个）
    ln_params = 2 * 2 * d_model  # 每个 LN 有 gamma 和 beta
    per_layer_params += ln_params
    print(f"每层层归一化: {ln_params:,}")

    print(f"每层总参数: {per_layer_params:,}")
    print(f"{n_layers} 层总计: {per_layer_params * n_layers:,}")

    total_params += per_layer_params * n_layers
    print(f"\n模型总参数: {total_params:,}")


if __name__ == "__main__":
    demo()
