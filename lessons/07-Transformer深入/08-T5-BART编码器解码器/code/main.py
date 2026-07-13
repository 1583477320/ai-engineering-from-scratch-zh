# main.py — T5/BART 编码器-解码器演示
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 08（T5/BART 编码器-解码器架构）

import numpy as np


# === Softmax ===

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# === 缩放点积注意力 ===

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores)
    output = weights @ V
    return output, weights


# === 多头注意力 ===

class MultiHeadAttention:
    def __init__(self, d_model, n_heads, seed=42):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        self.heads = []
        for i in range(n_heads):
            rng = np.random.default_rng(seed + i)
            scale = np.sqrt(2.0 / (d_model + self.dk))
            Wq = rng.normal(0, scale, (d_model, self.dk))
            Wk = rng.normal(0, scale, (d_model, self.dk))
            Wv = rng.normal(0, scale, (d_model, self.dk))
            self.heads.append((Wq, Wk, Wv))
        rng = np.random.default_rng(seed + n_heads)
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.Wo = rng.normal(0, scale, (n_heads * self.dk, d_model))

    def forward(self, Q, K, V, mask=None):
        head_outputs = []
        for Wq, Wk, Wv in self.heads:
            q = Q @ Wq
            k = K @ Wk
            v = V @ Wv
            output, _ = scaled_dot_product_attention(q, k, v, mask)
            head_outputs.append(output)
        concatenated = np.concatenate(head_outputs, axis=-1)
        return concatenated @ self.Wo


# === 前馈网络 ===

class FeedForward:
    def __init__(self, d_model, d_ff, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / (d_model + d_ff)), (d_model, d_ff))
        self.W2 = rng.normal(0, np.sqrt(2.0 / (d_ff + d_model)), (d_ff, d_model))

    def forward(self, x):
        linear1 = x @ self.W1
        gelu = 0.5 * linear1 * (1 + np.tanh(np.sqrt(2 / np.pi) * (linear1 + 0.044715 * linear1**3)))
        return gelu @ self.W2


# === 层归一化 ===

class LayerNorm:
    def __init__(self, d_model, eps=1e-8):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + self.eps
        return self.gamma * (x - mean) / std + self.beta


# === Transformer 编码器块 ===

class EncoderBlock:
    def __init__(self, d_model, n_heads, d_ff, seed=42):
        self.self_attn = MultiHeadAttention(d_model, n_heads, seed)
        self.ffn = FeedForward(d_model, d_ff, seed + 100)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.self_attn.forward(x, x, x, mask)
        x = self.ln1.forward(x + attn_out)
        ffn_out = self.ffn.forward(x)
        x = self.ln2.forward(x + ffn_out)
        return x


# === Transformer 解码器块（带交叉注意力） ===

class DecoderBlock:
    """解码器块：自注意力 + 交叉注意力 + FFN。"""

    def __init__(self, d_model, n_heads, d_ff, seed=42):
        self.self_attn = MultiHeadAttention(d_model, n_heads, seed)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, seed + 100)
        self.ffn = FeedForward(d_model, d_ff, seed + 200)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)

    def _create_causal_mask(self, n):
        return np.triu(np.ones((n, n), dtype=bool), k=1)

    def forward(self, x, encoder_output, mask=None):
        # 自注意力（因果掩码）
        seq_len = x.shape[0]
        causal_mask = self._create_causal_mask(seq_len)
        # 合并两个掩码：填充掩码 + 因果掩码
        combined_mask = causal_mask if mask is None else (mask | causal_mask)
        attn_out = self.self_attn.forward(x, x, x, combined_mask)
        x = self.ln1.forward(x + attn_out)

        # 交叉注意力（看编码器输出）
        cross_out = self.cross_attn.forward(x, encoder_output, encoder_output)
        x = self.ln2.forward(x + cross_out)

        # FFN
        ffn_out = self.ffn.forward(x)
        x = self.ln3.forward(x + ffn_out)
        return x


# === 编码器-解码器 Transformer ===

class EncoderDecoderTransformer:
    """完整的编码器-解码器架构。"""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_encoder_layers=3, n_decoder_layers=3, max_len=512, seed=42):
        self.d_model = d_model
        rng = np.random.default_rng(seed)
        self.encoder_embedding = rng.normal(0, 0.02, (vocab_size, d_model))
        self.decoder_embedding = rng.normal(0, 0.02, (vocab_size, d_model))
        self.position_embedding = np.random.default_rng(seed + 1).normal(0, 0.02, (max_len, d_model))

        self.encoder_layers = [EncoderBlock(d_model, n_heads, d_ff, seed + i) for i in range(n_encoder_layers)]
        self.decoder_layers = [DecoderBlock(d_model, n_heads, d_ff, seed + 100 + i) for i in range(n_decoder_layers)]

        rng_out = np.random.default_rng(seed + 1000)
        self.output_proj = rng_out.normal(0, 0.02, (d_model, vocab_size))

    def encode(self, input_ids):
        """编码器前向传播。"""
        seq_len = input_ids.shape[0]
        x = self.encoder_embedding[input_ids] + self.position_embedding[:seq_len]
        for layer in self.encoder_layers:
            x = layer.forward(x)
        return x

    def decode(self, decoder_ids, encoder_output):
        """解码器前向传播。"""
        seq_len = decoder_ids.shape[0]
        x = self.decoder_embedding[decoder_ids] + self.position_embedding[:seq_len]
        for layer in self.decoder_layers:
            x = layer.forward(x, encoder_output)
        logits = x @ self.output_proj
        return logits

    def forward(self, input_ids, decoder_ids):
        """完整的前向传播。"""
        encoder_output = self.encode(input_ids)
        logits = self.decode(decoder_ids, encoder_output)
        return logits


# === 演示 ===

def demo():
    print("=" * 60)
    print("编码器-解码器 Transformer — 演示")
    print("=" * 60)

    vocab_size = 50
    # 编码器输入："translate English to French: The cat sat"
    # 解码器输入："Le chat s'est assis"
    encoder_input = np.array([2, 5, 12, 8, 3, 7, 15, 9])
    decoder_input = np.array([4, 6, 14, 11, 10])

    print(f"\n编码器输入形状: {encoder_input.shape}")
    print(f"解码器输入形状: {decoder_input.shape}")

    # 模型
    model = EncoderDecoderTransformer(vocab_size, d_model=16, n_heads=2, d_ff=32)

    # 编码
    print("\n--- 编码器 ---")
    enc_output = model.encode(encoder_input)
    print(f"编码器输出形状: {enc_output.shape}")

    # 解码
    print("\n--- 解码器（带交叉注意力） ---")
    logits = model.decode(decoder_input, enc_output)
    print(f"解码器输出形状: {logits.shape}")

    # 交叉注意力说明
    print("\n--- 交叉注意力说明 ---")
    print("解码器每一步都可以关注编码器的所有位置:")
    print(f"  解码器位置 0 → 关注编码器所有 8 个位置")
    print(f"  解码器位置 3 → 关注编码器所有 8 个位置")
    print("这不同于自注意力——交叉注意力不考虑位置对应关系")

    # 三种架构对比
    print("\n--- 三种 Transformer 变体对比 ---")
    print(f"{'架构':<20} {'组件':<30} {'生成':<8} {'代表模型'}")
    print("-" * 70)
    print(f"{'仅编码器':<20} {'只有 Transformer 编码器':<30} {'❌':<8} {'BERT, RoBERTa'}")
    print(f"{'仅解码器':<20} {'只有 Transformer 解码器':<30} {'✅':<8} {'GPT, LLaMA'}")
    print(f"{'编码器+解码器':<20} {'编码器+解码器+交叉注意力':<30} {'✅':<8} {'T5, BART'}")


if __name__ == "__main__":
    demo()
