# Mini GPT 从零实现（GPT-2 Small 架构）
# 纯 NumPy 实现——理解每一步

import numpy as np


# ============================================================================
# 第 1 步：词嵌入 + 位置嵌入
# ============================================================================

class Embedding:
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, embed_dim) * 0.02

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        tok_emb = self.token_embed[token_ids]
        pos_emb = self.pos_embed[:seq_len]
        return tok_emb + pos_emb


# ============================================================================
# 第 2 步：Layer Norm
# ============================================================================

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


# ============================================================================
# 第 3 步：多头注意力
# ============================================================================

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_out = np.random.randn(embed_dim, embed_dim) * 0.02

    def forward(self, x, mask=None):
        batch, seq_len, d = x.shape
        Q = (x @ self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x @ self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x @ self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        attn_out = weights @ V

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)
        return attn_out @ self.W_out


# ============================================================================
# 第 4 步：前馈网络 + Transformer Block
# ============================================================================

class FeedForward:
    def __init__(self, embed_dim, ff_dim):
        self.W1 = np.random.randn(embed_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

    def forward(self, x):
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)  # ReLU
        return h @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    def forward(self, x, mask=None):
        x = x + self.attn.forward(self.ln1.forward(x), mask)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x


# ============================================================================
# 第 5 步：Mini GPT 模型
# ============================================================================

class MiniGPT:
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072):
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)  # 因果掩码

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)
        logits = x @ self.embedding.token_embed.T
        return logits

    def count_parameters(self):
        total = 0
        total += self.embedding.token_embed.size
        total += self.embedding.pos_embed.size
        for block in self.blocks:
            total += block.attn.W_q.size + block.attn.W_k.size
            total += block.attn.W_v.size + block.attn.W_out.size
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size
            total += block.ln1.gamma.size + block.ln1.beta.size
            total += block.ln2.gamma.size + block.ln2.beta.size
        total += self.ln_f.gamma.size + self.ln_f.beta.size
        return total


# ============================================================================
# 第 6 步：损失函数 + 生成
# ============================================================================

def cross_entropy_loss(logits, targets):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )
    loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
    return loss


def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8):
    """自回归生成。"""
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]
        next_logits = next_logits / temperature
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()
        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(next_token)

    return tokens


# ============================================================================
# 第 7 步：GPT-2 家族参数分析
# ============================================================================

def parameter_breakdown():
    configs = [
        ("GPT-2 Small", 50257, 768, 12, 12, 1024, 3072),
        ("GPT-2 Medium", 50257, 1024, 16, 24, 1024, 4096),
        ("GPT-2 Large", 50257, 1280, 20, 36, 1024, 5120),
        ("GPT-2 XL", 50257, 1600, 25, 48, 1024, 6400),
    ]
    print("GPT-2 家族参数量")
    print("=" * 65)
    print(f"{'模型':<16} {'层数':>6} {'头数':>6} {'维度':>6} {'参数量':>14}")
    print("-" * 65)
    for name, vocab, dim, heads, layers, seq_len, ff in configs:
        token_emb = vocab * dim
        pos_emb = seq_len * dim
        per_block_attn = 4 * dim * dim
        per_block_ff = 2 * dim * ff + dim + ff
        per_block_ln = 4 * dim
        per_block = per_block_attn + per_block_ff + per_block_ln
        final_ln = 2 * dim
        total = token_emb + pos_emb + layers * per_block + final_ln
        print(f"{name:<16} {layers:>6} {heads:>6} {dim:>6} {total:>14,}")


def memory_estimate():
    print("推理显存估算 (FP16)")
    print("=" * 65)
    models = [
        ("GPT-2 Small (124M)", 124e6, 12, 12, 64, 1024),
        ("Llama 3 8B", 8e9, 32, 32, 128, 8192),
        ("Llama 3 70B", 70e9, 80, 64, 128, 8192),
        ("Llama 3 405B", 405e9, 126, 128, 128, 131072),
    ]
    print(f"{'模型':<24} {'权重':>10} {'KV缓存':>12} {'总计':>10}")
    print("-" * 65)
    for name, params, layers, heads, head_dim, max_seq in models:
        weight_bytes = params * 2
        kv_per_token = 2 * layers * heads * head_dim * 2
        kv_full = kv_per_token * max_seq
        total = weight_bytes + kv_full
        def fmt(b):
            return f"{b / 1e9:.1f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB"
        print(f"{name:<24} {fmt(weight_bytes):>10} {fmt(kv_full):>12} {fmt(total):>10}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    parameter_breakdown()
    memory_estimate()

    corpus = """The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Self-attention computes relationships between all pairs of positions in a sequence.
Multi-head attention splits the representation into multiple subspaces.
Each attention head can learn different types of relationships.
The feedforward network provides nonlinear transformations at each position.
Residual connections enable gradient flow through deep networks.
Layer normalization stabilizes training by normalizing activations.
Position embeddings give the model information about token ordering.
The causal mask ensures autoregressive generation during training.
Pre-training on large text corpora teaches the model general language understanding.
Fine-tuning adapts the pre-trained model to specific downstream tasks."""

    print("\n训练 Mini GPT")
    print("=" * 65)
    tokens = np.array(list(corpus.encode("utf-8")[:2048]))
    model = MiniGPT(vocab_size=256, embed_dim=128, num_heads=4, num_layers=4, max_seq_len=64, ff_dim=512)
    print(f"参数量: {model.count_parameters():,}")

    for step in range(200):
        start = np.random.randint(0, max(1, len(tokens) - 65))
        batch = tokens[start:start + 65]
        input_ids = batch[:-1].reshape(1, -1)
        target_ids = batch[1:].reshape(1, -1)

        mask = np.triu(np.full((64, 64), -1e9), k=1)
        x = model.embedding.forward(input_ids)
        for block in model.blocks:
            x = block.forward(x, mask)
        logits = model.ln_f.forward(x) @ model.embedding.token_embed.T
        loss = cross_entropy_loss(logits, target_ids)

        if step % 50 == 0:
            print(f"  Step {step:4d}: loss={loss:.4f}")

    prompt = list("The transformer".encode("utf-8"))
    output = generate(model, prompt, max_new_tokens=100, temperature=0.8)
    print(f"\n提示词: 'The transformer'")
    print(f"生成: {bytes(output).decode('utf-8', errors='replace')}")
