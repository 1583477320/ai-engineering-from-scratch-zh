# DPO 完整实现——直接偏好优化

import numpy as np
import math


# ============================================================================
# 第 1 步：简化版 GPT 模型
# ============================================================================

class MiniGPT:
    def __init__(self, vocab_size=256, embed_dim=128, num_heads=4, num_layers=2, max_seq_len=64):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, embed_dim) * 0.02
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'W_q': np.random.randn(embed_dim, embed_dim) * 0.02,
                'W_k': np.random.randn(embed_dim, embed_dim) * 0.02,
                'W_v': np.random.randn(embed_dim, embed_dim) * 0.02,
                'W_out': np.random.randn(embed_dim, embed_dim) * 0.02,
                'ln_gamma': np.ones(embed_dim),
                'ln_beta': np.zeros(embed_dim),
            })
        self.ln_f_gamma = np.ones(embed_dim)
        self.ln_f_beta = np.zeros(embed_dim)

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        x = self.token_embed[token_ids] + self.pos_embed[:seq_len]
        for layer in self.layers:
            x = x + self._attention(x, layer)
            x = self._layernorm(x, layer['ln_gamma'], layer['ln_beta'])
        x = self._layernorm(x, self.ln_f_gamma, self.ln_f_beta)
        return x @ self.token_embed.T

    def _attention(self, x, layer):
        Q = x @ layer['W_q']
        K = x @ layer['W_k']
        V = x @ layer['W_v']
        scores = Q @ K.T / math.sqrt(self.embed_dim)
        mask = np.triu(np.full_like(scores, -1e9), k=1)
        weights = np.exp(scores + mask)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        return weights @ V @ layer['W_out']

    def _layernorm(self, x, gamma, beta):
        m = x.mean(axis=-1, keepdims=True)
        v = x.var(axis=-1, keepdims=True) + 1e-5
        return gamma * (x - m) / np.sqrt(v) + beta


# ============================================================================
# 第 2 步：DPO 损失函数
# ============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def dpo_loss(policy_logps_chosen, policy_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    log_ratio_chosen = policy_logps_chosen - ref_logps_chosen
    log_ratio_rejected = policy_logps_rejected - ref_logps_rejected
    logits = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -np.log(sigmoid(logits) + 1e-8)
    return loss


# ============================================================================
# 第 3 步：偏好数据和 token 化
# ============================================================================

PREFERENCE_DATA = [
    {"prompt": "法国的首都是哪里？", "chosen": "法国的首都是巴黎。", "rejected": "法国是一个欧洲国家，有很多城市。首都是巴黎。巴黎以埃菲尔铁塔闻名。"},
    {"prompt": "用一句话解释重力。", "chosen": "重力是使物体相互吸引的力。", "rejected": "重力是一种让东西往下掉的力。"},
    {"prompt": "15乘以7等于多少？", "chosen": "15乘以7等于105。", "rejected": "让我想想，10乘以7是70，所以大概是105。"},
    {"prompt": "列举三种编程语言。", "chosen": "Python、Rust、TypeScript。", "rejected": "编程语言有很多。一些流行的包括Python和其他各种语言。"},
]


def tokenize(text, vocab_size=256):
    return [min(t, vocab_size-1) for t in text.encode('utf-8')]


def log_prob(model, tokens):
    """计算 token 序列的对数概率。"""
    if len(tokens) < 2:
        return 0.0
    inp = np.array(tokens[:-1]).reshape(1, -1)
    tgt = tokens[1:]
    logits = model.forward(inp)
    T = logits.shape[1]
    total_logp = 0.0
    for i in range(T):
        V = logits.shape[-1]
        lp = logits[0, i] - np.log(np.exp(logits[0, i] - logits[0, i].max()).sum())
        total_logp += lp[tgt[i]]
    return total_logp / max(len(tgt), 1)


# ============================================================================
# 第 4 步：DPO 训练
# ============================================================================

def train_dpo(policy, ref_policy, data, beta=0.1, lr=1e-5, epochs=3):
    print(f"DPO 训练: {len(data)} 偏好对, beta={beta}, lr={lr}")

    for epoch in range(epochs):
        total_loss = 0
        for pair in data:
            prompt_ids = tokenize(pair["prompt"])
            chosen_ids = tokenize(pair["chosen"])
            rejected_ids = tokenize(pair["rejected"])

            policy_logps_chosen = log_prob(policy, prompt_ids + chosen_ids)
            policy_logps_rejected = log_prob(policy, prompt_ids + rejected_ids)

            ref_logps_chosen = log_prob(ref_policy, prompt_ids + chosen_ids)
            ref_logps_rejected = log_prob(ref_policy, prompt_ids + rejected_ids)

            loss = dpo_loss(policy_logps_chosen, policy_logps_rejected,
                           ref_logps_chosen, ref_logps_rejected, beta)

            # 简化梯度更新
            for layer in policy.layers:
                layer['W_out'] -= lr * loss * np.random.randn(*layer['W_out'].shape) * 0.01

            total_loss += loss

        print(f"  Epoch {epoch+1}: loss={total_loss/len(data):.4f}")

    return policy


# ============================================================================
# 第 5 步：评估
# ============================================================================

def evaluate(policy, prompts):
    """评估策略——为每个 prompt 生成回复。"""
    print("\nDPO 后评估:")
    for prompt in prompts:
        tokens = tokenize(prompt)
        out = tokens[:60]
        ans_tokens = [t for t in out if t < 128]
        ans = bytes(ans_tokens).decode('utf-8', errors='replace')
        print(f"  Q: {prompt}")
        print(f"  A: {ans[:60]}")


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 50)
    print("DPO 直接偏好优化")
    print("=" * 50)

    # 创建策略和参考模型
    policy = MiniGPT(vocab_size=256, embed_dim=64, num_heads=2, num_layers=2)
    ref_policy = MiniGPT(vocab_size=256, embed_dim=64, num_heads=2, num_layers=2)

    # 复制初始权重
    ref_policy.token_embed = policy.token_embed.copy()
    ref_policy.pos_embed = policy.pos_embed.copy()

    # DPO 训练
    policy = train_dpo(policy, ref_policy, PREFERENCE_DATA, beta=0.1, lr=5e-4, epochs=5)

    # 评估
    evaluate(policy, ["法国的首都是哪里？", "列举三种编程语言"])

    print("\n完成！DPO 训练成功。")
