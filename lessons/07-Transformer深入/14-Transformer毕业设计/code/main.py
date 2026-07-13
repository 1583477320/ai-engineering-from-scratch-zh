# main.py — Transformer 毕业设计：完整 Encoder-Decoder（numpy）
# 依赖：numpy>=1.24 | 对应课程：阶段 07 · 14（Transformer 毕业设计）
# 演示：编码器-解码器架构 + 教师强制训练 + 贪心解码 vs 束搜索

import numpy as np
PAD, BOS, EOS = 0, 1, 2


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sinusoidal_encoding(max_len, d):
    pe = np.zeros((max_len, d))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d, 2) * -(np.log(10000) / d))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


class MultiHeadAttention:
    def __init__(self, d_model, n_heads, seed=42):
        assert d_model % n_heads == 0
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        rng = np.random.default_rng(seed)
        s = np.sqrt(2.0 / d_model)
        self.Wq = rng.normal(0, s, (d_model, d_model))
        self.Wk = rng.normal(0, s, (d_model, d_model))
        self.Wv = rng.normal(0, s, (d_model, d_model))
        self.Wo = rng.normal(0, s, (d_model, d_model))

    def forward(self, q_in, kv_in, mask=None):
        sq, sk = q_in.shape[0], kv_in.shape[0]
        h, dk = self.n_heads, self.d_k
        Q = (q_in @ self.Wq).reshape(sq, h, dk).transpose(1, 0, 2)
        K = (kv_in @ self.Wk).reshape(sk, h, dk).transpose(1, 0, 2)
        V = (kv_in @ self.Wv).reshape(sk, h, dk).transpose(1, 0, 2)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dk)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        weights = softmax(scores)
        out = (weights @ V).transpose(1, 0, 2).reshape(sq, -1)
        return out @ self.Wo


class LayerNorm:
    def __init__(self, d):
        self.gamma, self.beta = np.ones(d), np.zeros(d)

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        return self.gamma * (x - mu) / (x.std(axis=-1, keepdims=True) + 1e-8) + self.beta


class FeedForward:
    def __init__(self, d_model, d_ff, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / d_model), (d_model, d_ff))
        self.W2 = rng.normal(0, np.sqrt(2 / d_ff), (d_ff, d_model))

    def forward(self, x):
        h = x @ self.W1
        h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h**3)))
        return h @ self.W2


class EncoderBlock:
    def __init__(self, d_model, n_heads, d_ff, seed=42):
        self.attn = MultiHeadAttention(d_model, n_heads, seed)
        self.ffn = FeedForward(d_model, d_ff, seed + 100)
        self.ln1, self.ln2 = LayerNorm(d_model), LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.ln1.forward(x + self.attn.forward(x, x, mask))
        return self.ln2.forward(x + self.ffn.forward(x))


class DecoderBlock:
    def __init__(self, d_model, n_heads, d_ff, seed=42):
        self.self_attn = MultiHeadAttention(d_model, n_heads, seed)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, seed + 50)
        self.ffn = FeedForward(d_model, d_ff, seed + 100)
        self.ln1, self.ln2, self.ln3 = (
            LayerNorm(d_model), LayerNorm(d_model), LayerNorm(d_model))

    def forward(self, x, enc_out, causal_mask=None):
        x = self.ln1.forward(x + self.self_attn.forward(x, x, causal_mask))
        x = self.ln2.forward(x + self.cross_attn.forward(x, enc_out))
        return self.ln3.forward(x + self.ffn.forward(x))


class Encoder:
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, seed=42):
        rng = np.random.default_rng(seed)
        self.embed = rng.normal(0, 0.02, (vocab_size, d_model))
        self.pe = sinusoidal_encoding(50, d_model)
        self.layers = [EncoderBlock(d_model, n_heads, d_ff, seed + i * 10)
                       for i in range(n_layers)]

    def forward(self, ids):
        x = self.embed[ids] + self.pe[:len(ids)]
        for layer in self.layers:
            x = layer.forward(x)
        return x


class Decoder:
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, seed=42):
        rng = np.random.default_rng(seed)
        self.embed = rng.normal(0, 0.02, (vocab_size, d_model))
        self.pe = sinusoidal_encoding(50, d_model)
        self.layers = [DecoderBlock(d_model, n_heads, d_ff, seed + i * 10)
                       for i in range(n_layers)]
        self.proj = rng.normal(0, 0.02, (d_model, vocab_size))
        self._hidden = None

    def forward(self, ids, enc_out):
        n = len(ids)
        causal = np.triu(np.ones((n, n), dtype=bool), k=1)
        x = self.embed[ids] + self.pe[:n]
        for layer in self.layers:
            x = layer.forward(x, enc_out, causal)
        self._hidden = x
        return x @ self.proj


class Transformer:
    def __init__(self, src_vocab, tgt_vocab, d_model=32, n_heads=4,
                 d_ff=64, n_layers=1):
        self.encoder = Encoder(src_vocab, d_model, n_heads, d_ff, n_layers)
        self.decoder = Decoder(tgt_vocab, d_model, n_heads, d_ff, n_layers)

    def forward(self, src_ids, tgt_ids):
        return self.decoder.forward(tgt_ids, self.encoder.forward(src_ids))


def build_vocab(sentences):
    vocab = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS}
    for s in sentences:
        for w in s.split():
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def encode(text, vocab):
    return [vocab[w] for w in text.split()]


def decode_tokens(ids, id2tok):
    return " ".join(id2tok.get(i, "?") for i in ids if i > EOS)


def cross_entropy(logits, targets):
    probs = softmax(logits)
    return -np.log(probs[np.arange(len(targets)), targets] + 1e-8).mean()


def train(model, pairs, src_vocab, tgt_vocab, epochs=200, lr=0.01):
    """训练循环——教师强制 + 输出投影梯度下降。"""
    for epoch in range(epochs):
        total_loss = 0
        for src_text, tgt_text in pairs:
            src_ids = np.array(encode(src_text, src_vocab))
            tgt_ids = encode(tgt_text, tgt_vocab)
            dec_in = np.array([BOS] + tgt_ids[:-1])  # 教师强制
            dec_tgt = np.array(tgt_ids)
            logits = model.forward(src_ids, dec_in)
            loss = cross_entropy(logits, dec_tgt)
            total_loss += loss
            # 输出投影梯度：dL/dW = hidden^T @ (softmax - one_hot)
            probs = softmax(logits)
            grad = probs.copy()
            grad[np.arange(len(dec_tgt)), dec_tgt] -= 1
            model.decoder.proj -= lr * model.decoder._hidden.T @ grad
        if (epoch + 1) % 50 == 0:
            print(f"  轮次 {epoch+1:>4d} | 损失: {total_loss / len(pairs):.4f}")


def greedy_decode(model, src_ids, max_len=20):
    """贪心解码——每步取概率最高的词元。"""
    enc_out = model.encoder.forward(src_ids)
    seq = [BOS]
    for _ in range(max_len):
        logits = model.decoder.forward(np.array(seq), enc_out)
        next_id = int(logits[-1].argmax())
        if next_id == EOS:
            break
        seq.append(next_id)
    return seq


def beam_search(model, src_ids, beam_width=3, max_len=20):
    """束搜索——保持 beam_width 个最优候选序列。"""
    enc_out = model.encoder.forward(src_ids)
    beams = [(0.0, [BOS])]
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == EOS:
                candidates.append((score, seq))
                continue
            logits = model.decoder.forward(np.array(seq), enc_out)
            log_probs = np.log(softmax(logits[-1]) + 1e-8)
            for tid in log_probs.argsort()[-beam_width:]:
                candidates.append((score + log_probs[tid], seq + [int(tid)]))
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
    return beams[0][1]


if __name__ == "__main__":
    print("=" * 55)
    print("  Transformer 毕业设计 -- 从零构建完整 Encoder-Decoder 模型")
    print("=" * 55)
    pairs = [
        ("the cat sat .", "le chat assis ."),
        ("the dog ran .", "le chien a couru ."),
        ("i love code .", "j aime le code ."),
        ("she reads books .", "elle lit des livres ."),
        ("he plays music .", "il joue de la musique ."),
    ]
    src_vocab = build_vocab([s for s, _ in pairs])
    tgt_vocab = build_vocab([t for _, t in pairs])
    tgt_id2tok = {v: k for k, v in tgt_vocab.items()}
    print(f"\n源语言词表: {len(src_vocab)} 词元 | 目标语言词表: {len(tgt_vocab)} 词元")
    model = Transformer(len(src_vocab), len(tgt_vocab), d_model=32, n_heads=4, d_ff=64)
    print("模型配置: d_model=32, n_heads=4, d_ff=64, n_layers=1")
    print("\n--- 训练（教师强制 + 交叉熵损失） ---")
    train(model, pairs, src_vocab, tgt_vocab, epochs=200, lr=0.01)
    print("\n--- 贪心解码 ---")
    for src, tgt in pairs[:3]:
        ids = greedy_decode(model, np.array(encode(src, src_vocab)))
        print(f"  {src:<20s} -> {decode_tokens(ids, tgt_id2tok):<20s}  (目标: {tgt})")
    print("\n--- 束搜索 (beam_width=3) ---")
    for src, tgt in pairs[:3]:
        ids = beam_search(model, np.array(encode(src, src_vocab)), beam_width=3)
        print(f"  {src:<20s} -> {decode_tokens(ids, tgt_id2tok):<20s}  (目标: {tgt})")
