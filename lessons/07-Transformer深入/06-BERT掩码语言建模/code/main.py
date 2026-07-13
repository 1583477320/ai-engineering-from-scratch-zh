# main.py — BERT 掩码语言建模完整实现
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 06（BERT 掩码语言建模）

import numpy as np
import random


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

    def forward(self, X, mask=None):
        head_outputs = []
        for Wq, Wk, Wv in self.heads:
            Q = X @ Wq
            K = X @ Wk
            V = X @ Wv
            output, _ = scaled_dot_product_attention(Q, K, V, mask)
            head_outputs.append(output)
        concatenated = np.concatenate(head_outputs, axis=-1)
        return concatenated @ self.Wo


# === 前馈网络 ===

class FeedForward:
    """前馈网络：d_model → d_ff → d_model。"""

    def __init__(self, d_model, d_ff, seed=42):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        self.W1 = rng.normal(0, scale1, (d_model, d_ff))
        self.W2 = rng.normal(0, scale2, (d_ff, d_model))

    def forward(self, x):
        linear1 = x @ self.W1
        gelu = 0.5 * linear1 * (1 + np.tanh(np.sqrt(2 / np.pi) * (linear1 + 0.044715 * linear1**3)))
        return gelu @ self.W2


# === 层归一化 ===

class LayerNorm:
    """层归一化。"""

    def __init__(self, d_model, eps=1e-8):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + self.eps
        return self.gamma * (x - mean) / std + self.beta


# === Transformer 块 ===

class TransformerBlock:
    """完整的 Transformer 块（Pre-LN）。"""

    def __init__(self, d_model, n_heads, d_ff, seed=42):
        self.mha = MultiHeadSelfAttention(d_model, n_heads, seed)
        self.ffn = FeedForward(d_model, d_ff, seed + 100)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.mha.forward(x, mask)
        x = self.ln1.forward(x + attn_out)
        ffn_out = self.ffn.forward(x)
        x = self.ln2.forward(x + ffn_out)
        return x


# === Transformer 编码器 ===

class TransformerEncoder:
    """Transformer 编码器（多层堆叠）。"""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len=512, seed=42):
        self.d_model = d_model
        rng = np.random.default_rng(seed)
        self.word_embedding = rng.normal(0, 0.02, (vocab_size, d_model))
        self.position_embedding = rng.normal(0, 0.02, (max_len, d_model))
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff, seed + i * 10)
            for i in range(n_layers)
        ]

    def forward(self, input_ids, mask=None):
        seq_len = input_ids.shape[0]
        x = self.word_embedding[input_ids] + self.position_embedding[:seq_len]
        for layer in self.layers:
            x = layer.forward(x, mask)
        return x


# === 正弦位置编码 ===

def sinusoidal_position_encoding(max_len, d_model):
    """正弦位置编码。"""
    pe = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            pe[pos, i] = np.sin(angle)
            pe[pos, i + 1] = np.cos(angle)
    return pe


# === BERT 专用组件 ===

class BERTForMLM:
    """BERT 掩码语言模型（NumPy 教学版）。"""

    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=128, n_layers=2, max_len=128, seed=42):
        """初始化 BERT MLM 模型。"""
        self.d_model = d_model
        rng = np.random.default_rng(seed)

        # 词元嵌入 + 位置嵌入（使用可学习编码）
        self.word_embedding = rng.normal(0, 0.02, (vocab_size, d_model))
        self.position_embedding = np.random.default_rng(seed + 1).normal(0, 0.02, (max_len, d_model))

        # Transformer 编码器
        self.encoder = TransformerEncoder(vocab_size, d_model, n_heads, d_ff, n_layers, max_len, seed)

        # MLM 分类头（将 d_model 映射回 vocab_size）
        rng_head = np.random.default_rng(seed + 1000)
        self.mlm_head_W = rng_head.normal(0, 0.02, (d_model, vocab_size))
        self.mlm_head_b = np.zeros(vocab_size)

    def forward(self, input_ids, mask=None):
        """前向传播，返回每个位置对词表的预测分数。

        Args:
            input_ids: 形状 (seq_len,) 的词元 ID 序列
            mask: 可选。注意力掩码

        Returns:
            logits: 形状 (seq_len, vocab_size) 的预测分数
        """
        # Transformer 编码
        h = self.encoder.forward(input_ids, mask)

        # MLM 分类头
        logits = h @ self.mlm_head_W + self.mlm_head_b
        return logits

    def predict_masked(self, input_ids, mask_token_id=103):
        """预测 [MASK] 位置的词元。

        Args:
            input_ids: 包含 [MASK] 的词元 ID 序列
            mask_token_id: [MASK] 对应的 ID

        Returns:
            predictions: 每个位置的预测词元 ID
        """
        logits = self.forward(input_ids)
        # 找出被遮住的位置
        masked_positions = np.where(input_ids == mask_token_id)[0]
        # 返回每个被遮住位置的预测
        predictions = []
        for pos in masked_positions:
            probs = softmax(logits[pos])
            predicted_id = np.argmax(probs)
            predictions.append((pos, predicted_id, probs[predicted_id]))
        return predictions


# === 掩码策略 ===

def create_mlm_input(tokens, vocab_size, mask_token_id=103, mask_prob=0.15, seed=None):
    """创建 MLM 输入：80% [MASK]，10% 随机替换，10% 不变。

    Args:
        tokens: 原始词元 ID 列表
        vocab_size: 词表大小
        mask_token_id: [MASK] 特殊词元的 ID
        mask_prob: 掩码比例（默认 15%）
        seed: 随机种子

    Returns:
        output: 掩码后的输入序列
        labels: 标签序列（-100 表示不计算损失）
    """
    if seed is not None:
        random.seed(seed)

    labels = [-100] * len(tokens)
    output = list(tokens)

    for i in range(len(tokens)):
        if random.random() < mask_prob:
            labels[i] = tokens[i]  # 保留原始词元作为标签
            r = random.random()
            if r < 0.8:
                output[i] = mask_token_id  # 80% → [MASK]
            elif r < 0.9:
                output[i] = random.randint(0, vocab_size - 1)  # 10% → 随机
            # 10% → 保持不变

    return output, labels


# === 演示 ===

def demo():
    """演示 BERT MLM 的完整流程。"""
    print("=" * 60)
    print("BERT 掩码语言建模 — 演示")
    print("=" * 60)

    # 模拟词表（简单数字词表）
    vocab_size = 100
    # 模拟一个句子："the cat sat on the mat"
    # 词元 ID: [2, 5, 12, 8, 2, 15]
    sentence_tokens = [2, 5, 12, 8, 2, 15]
    words = ["[PAD]", "[CLS]", "the", "[SEP]", "[MASK]", "cat", "dog",
             "on", "the", "ran", "ate", "sat", "on", "the"]

    print(f"\n句子词元: {sentence_tokens}")
    print(f"对应的词: {[words[t] if t < len(words) else '?' for t in sentence_tokens]}")

    # 创建 MLM 输入
    print("\n--- 掩码策略 ---")
    output, labels = create_mlm_input(sentence_tokens, vocab_size, mask_token_id=4, seed=42)
    print(f"原始: {sentence_tokens}")
    print(f"掩码后: {output}")
    print(f"标签: {labels}")
    masked_positions = [i for i, o in enumerate(output) if o == 4]
    print(f"被 [MASK] 替换: 位置 {masked_positions}")

    # BERT 模型
    print("\n--- BERT MLM 前向传播 ---")
    bert = BERTForMLM(vocab_size=vocab_size, d_model=16, n_heads=2, d_ff=32, n_layers=2)
    logits = bert.forward(np.array(output))
    print(f"输入形状: ({len(output)},)")
    print(f"输出形状: {logits.shape} (seq_len={len(output)}, vocab_size={vocab_size})")

    # 预测被遮住的词
    if masked_positions:
        print("\n--- [MASK] 位置预测 ---")
        predictions = bert.predict_masked(np.array(output), mask_token_id=4)
        for pos, pred_id, prob in predictions:
            pred_word = words[pred_id] if pred_id < len(words) else f"<{pred_id}>"
            orig_word = words[sentence_tokens[pos]] if sentence_tokens[pos] < len(words) else "?"
            print(f"位置 {pos}: 原始='{orig_word}', 预测='{pred_word}'(ID={pred_id}), 概率={prob:.3f}")

    # 参数量计算
    print("\n--- 参数量计算 ---")
    print(f"词元嵌入: {vocab_size} × 16 = {vocab_size * 16}")
    print(f"位置嵌入: 128 × 16 = 2048")
    print(f"每层 Transformer: 约 4 × 16² + 2 × 16 × 32 = 2048")
    print(f"2 层总计: 约 4096")
    print(f"MLM 分类头: 16 × 100 = 1600")

    print("\n--- BERT vs GPT 对比 ---")
    print("BERT: 双向编码器 → 分类 / 问答 / 信息提取")
    print("GPT:  单向解码器 → 生成 / 对话 / 翻译")


if __name__ == "__main__":
    demo()
