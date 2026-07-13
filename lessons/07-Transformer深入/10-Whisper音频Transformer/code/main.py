# main.py — Whisper 音频 Transformer 演示
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 10（Whisper 音频 Transformer）

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
    return weights @ V, weights


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


# === 编码器块 ===

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


# === 解码器块（带交叉注意力） ===

class DecoderBlock:
    def __init__(self, d_model, n_heads, d_ff, seed=42):
        self.self_attn = MultiHeadAttention(d_model, n_heads, seed)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, seed + 50)
        self.ffn = FeedForward(d_model, d_ff, seed + 100)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)

    def _create_causal_mask(self, n):
        return np.triu(np.ones((n, n), dtype=bool), k=1)

    def forward(self, x, encoder_output, mask=None):
        seq_len = x.shape[0]
        causal_mask = self._create_causal_mask(seq_len)
        combined_mask = causal_mask if mask is None else (mask | causal_mask)
        attn_out = self.self_attn.forward(x, x, x, combined_mask)
        x = self.ln1.forward(x + attn_out)
        cross_out = self.cross_attn.forward(x, encoder_output, encoder_output)
        x = self.ln2.forward(x + cross_out)
        ffn_out = self.ffn.forward(x)
        x = self.ln3.forward(x + ffn_out)
        return x


# === Whisper 简化版 ===

class Whisper:
    """简化的 Whisper 音频 Transformer。"""

    def __init__(self, d_model=64, n_heads=4, d_ff=128, n_encoder_layers=3, n_decoder_layers=3,
                 vocab_size=50, max_audio_frames=100, max_text_len=50, seed=42):
        self.d_model = d_model
        rng = np.random.default_rng(seed)
        # 卷积降采样（模拟 2 层 stride-2 卷积）
        self.conv_downsample = rng.normal(0, 0.02, (80, d_model))  # 80 mel → d_model
        # 位置编码
        self.audio_pe = np.random.default_rng(seed + 1).normal(0, 0.02, (max_audio_frames // 4, d_model))
        self.text_pe = np.random.default_rng(seed + 2).normal(0, 0.02, (max_text_len, d_model))
        # 词元嵌入
        self.token_embedding = rng.normal(0, 0.02, (vocab_size, d_model))
        # 编码器 + 解码器
        self.encoder_layers = [EncoderBlock(d_model, n_heads, d_ff, seed + 10 + i) for i in range(n_encoder_layers)]
        self.decoder_layers = [DecoderBlock(d_model, n_heads, d_ff, seed + 100 + i) for i in range(n_decoder_layers)]
        # 输出投影
        self.lm_head = np.random.default_rng(seed + 1000).normal(0, 0.02, (d_model, vocab_size))

    def _audio_to_features(self, mel_spectrogram):
        """将 log-mel 频谱图转换为特征序列（模拟卷积降采样）。"""
        return mel_spectrogram @ self.conv_downsample

    def encode(self, mel_spectrogram):
        """编码器前向传播。"""
        x = self._audio_to_features(mel_spectrogram)
        x = x + self.audio_pe[:x.shape[0]]
        for layer in self.encoder_layers:
            x = layer.forward(x)
        return x

    def decode(self, token_ids, encoder_output):
        """解码器前向传播。"""
        x = self.token_embedding[token_ids] + self.text_pe[:len(token_ids)]
        for layer in self.decoder_layers:
            x = layer.forward(x, encoder_output)
        logits = x @ self.lm_head
        return logits


# === 演示 ===

def demo():
    print("=" * 60)
    print("Whisper 音频 Transformer — 演示")
    print("=" * 60)

    # 模拟 log-mel 频谱图（80 mel × 100 帧）
    np.random.seed(42)
    mel = np.random.randn(100, 80)

    # Whisper 模型
    whisper = Whisper(d_model=32, n_heads=2, d_ff=64, n_encoder_layers=2, n_decoder_layers=2)

    # 编码器
    print("\n--- 编码器 ---")
    enc_output = whisper.encode(mel)
    print(f"输入 mel 形状: (100, 80)")
    print(f"编码器输出形状: {enc_output.shape} (降采样后帧数, d_model)")

    # 解码器
    print("\n--- 解码器 ---")
    token_ids = np.array([2, 5, 12, 8])  # 模拟词元
    logits = whisper.decode(token_ids, enc_output)
    print(f"输入词元: {token_ids}")
    print(f"解码器输出形状: {logits.shape}")

    # 多任务说明
    print("\n--- 多任务设计 ---")
    print(f"{'任务':<20} {'启动 token':<40}")
    print("-" * 60)
    print(f"{'转录 (中文)':<20} {'<|startoftranscript|><|zh|><|transcribe|>':<40}")
    print(f"{'翻译 (中→英)':<20} {'<|startoftranscript|><|zh|><|translate|>':<40}")
    print(f"{'带时间戳转录':<20} {'<|startoftranscript|><|zh|><|transcribe|><|timestamps|>':<40}")


if __name__ == "__main__":
    demo()
