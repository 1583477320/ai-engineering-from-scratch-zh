# main.py — GPT 因果语言建模完整实现
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 07（GPT 因果语言建模）

import numpy as np


# === Softmax ===

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# === 缩放点积注意力（支持掩码） ===

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores)
    output = weights @ V
    return output, weights


# === 多头因果自注意力 ===

class MultiHeadCausalAttention:
    """多头因果自注意力——每个位置只能看前面的位置。"""

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

    def _create_causal_mask(self, n):
        """创建因果掩码：上三角为 True（遮挡），下三角为 False（可见）。"""
        return np.triu(np.ones((n, n), dtype=bool), k=1)

    def forward(self, X):
        seq_len = X.shape[0]
        causal_mask = self._create_causal_mask(seq_len)
        head_outputs = []
        for Wq, Wk, Wv in self.heads:
            Q = X @ Wq
            K = X @ Wk
            V = X @ Wv
            output, _ = scaled_dot_product_attention(Q, K, V, causal_mask)
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


# === GPT 解码器块 ===

class GPTBlock:
    """GPT 解码器块（带因果自注意力）。"""

    def __init__(self, d_model, n_heads, d_ff, seed=42):
        self.attention = MultiHeadCausalAttention(d_model, n_heads, seed)
        self.ffn = FeedForward(d_model, d_ff, seed + 100)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x):
        # 因果自注意力 + 残差 + 层归一化
        attn_out = self.attention.forward(x)
        x = self.ln1.forward(x + attn_out)
        # FFN + 残差 + 层归一化
        ffn_out = self.ffn.forward(x)
        x = self.ln2.forward(x + ffn_out)
        return x


# === GPT 模型 ===

class GPT:
    """简化的 GPT 模型——因果语言模型。"""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len=512, seed=42):
        self.d_model = d_model
        rng = np.random.default_rng(seed)
        self.token_embedding = rng.normal(0, 0.02, (vocab_size, d_model))
        self.position_embedding = np.random.default_rng(seed + 1).normal(0, 0.02, (max_len, d_model))
        self.layers = [GPTBlock(d_model, n_heads, d_ff, seed + i * 10) for i in range(n_layers)]
        self.lm_head = np.random.default_rng(seed + 1000).normal(0, 0.02, (d_model, vocab_size))

    def forward(self, input_ids):
        seq_len = input_ids.shape[0]
        x = self.token_embedding[input_ids] + self.position_embedding[:seq_len]
        for layer in self.layers:
            x = layer.forward(x)
        logits = x @ self.lm_head  # (seq_len, vocab_size)
        return logits

    def generate(self, input_ids, max_new_tokens=10, temperature=1.0):
        """自回归生成。

        Args:
            input_ids: 起始词元 ID 序列
            max_new_tokens: 最多生成的 New 词元数
            temperature: 采样温度。越低越确定，越高越随机

        Returns:
            output_ids: 生成的完整序列
        """
        output_ids = list(input_ids)
        for _ in range(max_new_tokens):
            # 前向传播
            logits = self.forward(np.array(output_ids))
            # 取最后一个位置的 logits
            next_logits = logits[-1] / temperature
            # 采样
            probs = softmax(next_logits)
            next_id = np.random.choice(len(probs), p=probs)
            output_ids.append(int(next_id))
        return output_ids


# === 演示 ===

def demo():
    print("=" * 60)
    print("GPT 因果语言建模 — 演示")
    print("=" * 60)

    vocab_size = 50
    input_ids = [2, 5, 12, 8]

    print(f"\n输入词元: {input_ids}")

    # 因果掩码可视化
    print("\n--- 因果掩码 ---")
    n = 6
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    print("True = 被遮挡，False = 可见:")
    print(mask)
    print("\n位置 2 可以看到位置 0,1,2:")
    print(f"  可见: {~mask[2]}")
    print(f"  遮挡: {mask[2]}")

    # GPT 前向传播
    print("\n--- GPT 前向传播 ---")
    gpt = GPT(vocab_size=vocab_size, d_model=16, n_heads=2, d_ff=32, n_layers=2)
    logits = gpt.forward(np.array(input_ids))
    print(f"输入形状: (4,)")
    print(f"输出形状: {logits.shape} (seq_len=4, vocab_size={vocab_size})")
    print(f"最后一个位置预测: ID={np.argmax(logits[-1])}, 概率={softmax(logits[-1]).max():.3f}")

    # 自回归生成
    print("\n--- 自回归生成 ---")
    prompt = [2, 5]
    generated = gpt.generate(prompt, max_new_tokens=5, temperature=0.8)
    print(f"提示词: {prompt}")
    print(f"生成序列: {generated}")
    print(f"生成词元数: {len(generated) - len(prompt)}")

    # BERT vs GPT 对比
    print("\n--- BERT vs GPT 对比 ---")
    print("BERT: 双向编码器 → 分类 / NER / QA")
    print("GPT:  单向解码器 → 生成 / 对话 / 翻译")
    print("\n训练目标:")
    print("  BERT: MLM — 预测被遮住的词（15% 词元参与训练）")
    print("  GPT:  CLM — 预测下一个词（100% 词元参与训练）")
    print("\n注意力:")
    print("  BERT: 双向 — 每个词元看到左右上下文")
    print("  GPT:  单向 — 每个词元只能看到左边上下文（因果掩码）")


if __name__ == "__main__":
    demo()
