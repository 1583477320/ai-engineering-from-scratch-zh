# RLHF 演示：奖励模型 + PPO + KL 惩罚
# 简化版实现，展示核心概念

import numpy as np
import math


# ============================================================================
# 简化版 GPT 模型（同 SFT 课）
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
        logits = x @ self.token_embed.T
        return logits

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
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True) + 1e-5
        return gamma * (x - mean) / np.sqrt(var) + beta

    def copy_from(self, src):
        self.token_embed = src.token_embed.copy()
        self.pos_embed = src.pos_embed.copy()
        self.ln_f_gamma = src.ln_f_gamma.copy()
        self.ln_f_beta = src.ln_f_beta.copy()
        for i, s in enumerate(src.layers):
            for k in ['W_q','W_k','W_v','W_out','ln_gamma','ln_beta']:
                self.layers[i][k] = s[k].copy()


# ============================================================================
# 第 1 步：奖励模型
# ============================================================================

class RewardModel(nn.Module if __name__ == "__main__" else object):
    """奖励模型——对 prompt+response 序列打一个标量分。"""
    def __init__(self, vocab_size=256, embed_dim=128, num_heads=4, num_layers=2, max_seq_len=64):
        self.gpt = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)
        self.reward_head = np.random.randn(embed_dim) * 0.02

    def score(self, token_ids):
        """给 prompt+response 打分——取最后一个位置的表示。"""
        logits = self.gpt.forward(token_ids)
        hidden = logits[:, -1, :]  # 简化：用 logits 作为表示
        reward = hidden @ self.reward_head
        return reward[0]


# ============================================================================
# 第 2 步：Bradley-Terry 偏好损失
# ============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def bradley_terry_loss(r_preferred, r_rejected):
    """Bradley-Terry 偏好损失：-log(σ(r+ - r-))。"""
    return -np.log(sigmoid(r_preferred - r_rejected) + 1e-8)


# ============================================================================
# 第 3 步：训练奖励模型
# ============================================================================

PREFERENCE_DATA = [
    {"prompt": "法国的首都是哪里？", "preferred": "法国的首都是巴黎。", "rejected": "法国是一个欧洲国家，有很多城市。首都是巴黎。巴黎以埃菲尔铁塔闻名。"},
    {"prompt": "用一句话解释重力。", "preferred": "重力是使物体相互吸引的力。", "rejected": "重力是一种让东西往下掉的力。"},
    {"prompt": "15乘以7等于多少？", "preferred": "15乘以7等于105。", "rejected": "让我想想，15乘以7。嗯，10乘以7是70，5乘以7是35，所以大概是105。"},
    {"prompt": "列举三种编程语言。", "preferred": "Python、Rust、TypeScript。", "rejected": "编程语言有很多。一些流行的包括Python和其他各种语言。"},
    {"prompt": "机器学习是什么？", "preferred": "机器学习是算法从数据中学习模式进行预测的领域。", "rejected": "机器学习是一种AI。AI代表人工智能。机器学习使用数据来学习。"},
]


def tokenize_pair(prompt, response, max_len=64):
    """将 prompt+response 转换为 token 序列。"""
    p_tokens = [min(t, 255) for t in prompt.encode('utf-8')]
    r_tokens = [min(t, 255) for t in response.encode('utf-8')]
    tokens = p_tokens + [0] + r_tokens
    return tokens[:max_len]


def train_reward_model(rm, data, num_epochs=10, lr=1e-4):
    """训练奖励模型——使用 Bradley-Terry 偏好损失。"""
    print(f"训练奖励模型: {len(data)} 个偏好对, {num_epochs} 轮")

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        n = 0

        for pair in data:
            pref_ids = np.array(tokenize_pair(pair["prompt"], pair["preferred"])).reshape(1, -1)
            rej_ids = np.array(tokenize_pair(pair["prompt"], pair["rejected"])).reshape(1, -1)

            r_pref = rm.score(pref_ids)
            r_rej = rm.score(rej_ids)

            loss = bradley_terry_loss(r_pref, r_rej)

            # 梯度更新（简化）
            diff = r_pref - r_rej
            grad = sigmoid(diff) - 1.0
            # 更新奖励头
            hidden = rm.gpt.forward(pref_ids)[:, -1, :]
            rm.reward_head -= lr * grad * hidden.flatten()

            total_loss += loss
            n += 1
            if r_pref > r_rej:
                correct += 1

        avg_loss = total_loss / max(n, 1)
        acc = correct / max(n, 1)
        losses.append(avg_loss)
        accuracies.append(acc)

        if epoch % 2 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.1%}")

    return rm, losses, accuracies


# ============================================================================
# 第 4 步：PPO + KL 惩罚训练
# ============================================================================

def compute_kl(policy_logits, ref_logits):
    """计算 KL 散度。"""
    p = np.exp(policy_logits - policy_logits.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    q = np.exp(ref_logits - ref_logits.max(axis=-1, keepdims=True))
    q = q / q.sum(axis=-1, keepdims=True)
    return np.sum(p * np.log(p / q + 1e-10), axis=-1).mean()


def generate(model, prompt_tokens, max_new=20, temp=0.8, max_len=64):
    """自回归生成。"""
    tokens = list(prompt_tokens)
    for _ in range(max_new):
        ctx = np.array(tokens[-max_len:]).reshape(1, -1)
        logits = model.forward(ctx)[0, -1] / temp
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        tokens.append(np.random.choice(len(probs), p=probs))
    return tokens


def ppo_train(policy, ref_policy, rm, prompts, num_episodes=20, lr=1e-5, kl_coeff=0.02):
    """PPO + KL 惩罚训练。"""
    print(f"PPO 训练: {num_episodes} 轮, KL 系数={kl_coeff}")

    rewards, kls = [], []

    for ep in range(num_episodes):
        prompt = prompts[ep % len(prompts)]
        prompt_tokens = [min(t, 252) for t in prompt.encode('utf-8')]

        # 生成回复
        response = generate(policy, prompt_tokens, max_new=20)
        resp_ids = np.array(response[:64]).reshape(1, -1)

        # 计算奖励
        reward = rm.score(resp_ids)

        # 计算 KL
        policy_logits = policy.forward(resp_ids)
        ref_logits = ref_policy.forward(resp_ids)
        kl = compute_kl(policy_logits, ref_logits)

        # 总奖励 = 奖励 - KL 惩罚
        total_reward = reward - kl_coeff * kl

        # 简化策略更新
        for layer in policy.layers:
            layer['W_out'] += lr * total_reward * np.random.randn(*layer['W_out'].shape) * 0.01

        rewards.append(reward)
        kls.append(kl)

        if ep % 5 == 0:
            avg_r = np.mean(rewards[-5:]) if rewards else 0
            print(f"  Episode {ep}: reward={reward:.4f}, kl={kl:.4f}, avg={avg_r:.4f}")

    return policy, rewards, kls


# ============================================================================
# 第 5 步：评估对比
# ============================================================================

def compare(sft_model, rlhf_model, rm, eval_prompts):
    """对比 SFT 和 RLHF 模型的奖励分数。"""
    print("\nSFT vs RLHF 奖励分数对比:")
    print("-" * 50)
    sft_total, rlhf_total = 0, 0

    for prompt in eval_prompts:
        tokens = [min(t, 252) for t in prompt.encode('utf-8')]
        sft_out = generate(sft_model, tokens, temp=0.6)
        rlhf_out = generate(rlhf_model, tokens, temp=0.6)

        sft_r = rm.score(np.array(sft_out[:64]).reshape(1, -1))
        rlhf_r = rm.score(np.array(rlhf_out[:64]).reshape(1, -1))

        sft_total += sft_r
        rlhf_total += rlhf_r
        print(f"  {prompt[:30]:30s} SFT={sft_r:+.4f} RLHF={rlhf_r:+.4f}")

    n = len(eval_prompts)
    print(f"  {'平均':30s} SFT={sft_total/n:+.4f} RLHF={rlhf_total/n:+.4f}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("RLHF 管道: 奖励模型 + PPO + KL 惩罚")
    print("=" * 60)

    # 阶段 1：SFT 模型（模拟已微调的模型）
    print("\n阶段 1: SFT 模型")
    sft_model = MiniGPT(vocab_size=256, embed_dim=128, num_heads=4, num_layers=2, max_seq_len=64)
    print(f"  参数: {sum(p.size for l in sft_model.layers for p in l.values()):,}")

    # 阶段 2：训练奖励模型
    print("\n阶段 2: 训练奖励模型")
    rm = RewardModel(vocab_size=256, embed_dim=128, num_heads=4, num_layers=2, max_seq_len=64)
    rm, rm_losses, rm_accs = train_reward_model(rm, PREFERENCE_DATA, num_epochs=10, lr=1e-4)

    # 评估奖励模型
    print("\n奖励模型评估:")
    correct = 0
    for pair in PREFERENCE_DATA:
        pref_ids = np.array(tokenize_pair(pair["prompt"], pair["preferred"])).reshape(1, -1)
        rej_ids = np.array(tokenize_pair(pair["prompt"], pair["rejected"])).reshape(1, -1)
        r_pref = rm.score(pref_ids)
        r_rej = rm.score(rejected_ids := rej_ids)
        is_correct = r_pref > r_rej
        correct += int(is_correct)
        print(f"  {pair['preferred'][:30]:30s} r={r_pref:+.4f} vs r={r_rej:+.4f} {'✓' if is_correct else '✗'}")
    print(f"  准确率: {correct}/{len(PREFERENCE_DATA)} = {correct/len(PREFERENCE_DATA):.1%}")

    # 阶段 3：PPO + KL 惩罚
    print("\n阶段 3: PPO + KL 惩罚训练")
    policy = MiniGPT(vocab_size=256, embed_dim=128, num_heads=4, num_layers=2, max_seq_len=64)
    ref_policy = MiniGPT(vocab_size=256, embed_dim=128, num_heads=4, num_layers=2, max_seq_len=64)
    policy.copy_from(sft_model)
    ref_policy.copy_from(sft_model)

    prompts = [p["prompt"] for p in PREFERENCE_DATA]
    policy, rewards, kls = ppo_train(policy, ref_policy, rm, prompts, num_episodes=20, lr=1e-5, kl_coeff=0.02)

    # 对比
    compare(sft_model, policy, rm, prompts[:3])

    # KL 分析
    print(f"\nKL 分析: 初始={kls[0]:.4f}, 最终={kls[-1]:.4f}, 最大={max(kls):.4f}")
    print(f"KL 是否爆炸: {'是' if max(kls) > 0.1 else '否'}")
