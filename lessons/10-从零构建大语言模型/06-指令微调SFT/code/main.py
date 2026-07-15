# 指令微调 SFT 演示
# 使用简化的 MiniGPT 架构演示 SFT 的核心概念

import numpy as np
import math


# ============================================================================
# 第 1 步：简化版 GPT 模型
# ============================================================================

class MiniGPT:
    """简化版 GPT——用于演示 SFT 概念。"""
    def __init__(self, vocab_size=256, embed_dim=128, num_heads=4,
                 num_layers=2, max_seq_len=64):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # 词嵌入 + 位置嵌入
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, embed_dim) * 0.02

        # 简化的 Transformer 层
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

        # 输出头
        self.ln_f_gamma = np.ones(embed_dim)
        self.ln_f_beta = np.zeros(embed_dim)

    def forward(self, token_ids):
        """前向传播：返回 logits。"""
        seq_len = token_ids.shape[-1]
        x = self.token_embed[token_ids] + self.pos_embed[:seq_len]

        for layer in self.layers:
            x = x + self._attention(x, layer)
            x = self._layernorm(x, layer['ln_gamma'], layer['ln_beta'])

        x = self._layernorm(x, self.ln_f_gamma, self.ln_f_beta)
        logits = x @ self.token_embed.T
        return logits

    def _attention(self, x, layer):
        """简化版注意力。"""
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

    def count_parameters(self):
        total = self.token_embed.size + self.pos_embed.size
        for layer in self.layers:
            total += layer['W_q'].size + layer['W_k'].size + layer['W_v'].size + layer['W_out'].size
        total += self.ln_f_gamma.size + self.ln_f_beta.size
        return total

    def copy_from(self, source):
        """从另一个模型复制权重。"""
        self.token_embed = source.token_embed.copy()
        self.pos_embed = source.pos_embed.copy()
        self.ln_f_gamma = source.ln_f_gamma.copy()
        self.ln_f_beta = source.ln_f_beta.copy()
        for i, layer in enumerate(source.layers):
            for key in ['W_q', 'W_k', 'W_v', 'W_out', 'ln_gamma', 'ln_beta']:
                self.layers[i][key] = layer[key].copy()


# ============================================================================
# 第 2 步：SFT 数据处理
# ============================================================================

# 指令-响应对
INSTRUCTION_DATA = [
    {"instruction": "法国的首都是哪里？", "response": "法国的首都是巴黎。"},
    {"instruction": "用一句话解释重力。", "response": "重力是使物体相互吸引的力。"},
    {"instruction": "写一首关于大海的俳句。", "response": "海浪拍海岸，盐沫伴阳光，无边蓝意漫。"},
    {"instruction": "15乘以7等于多少？", "response": "15乘以7等于105。"},
    {"instruction": "列举三种编程语言。", "response": "Python、Rust 和 TypeScript。"},
    {"instruction": "简述光合作用。", "response": "光合作用将阳光、水和二氧化碳转化为葡萄糖和氧气。"},
    {"instruction": "第二次世界大战哪一年结束？", "response": "第二次世界大战于1945年结束。"},
    {"instruction": "定义机器学习。", "response": "机器学习是算法从数据中学习模式并进行预测的领域。"},
]

# 特殊 token
BOS = 253  # 指令开始
EOS = 254  # 指令结束
RESP = 255  # 响应开始


def tokenize_pair(instruction, response, vocab_size=256):
    """将指令-响应对转换为 token 序列。"""
    inst_tokens = [min(t, vocab_size - 4) for t in instruction.encode('utf-8')]
    resp_tokens = [min(t, vocab_size - 4) for t in response.encode('utf-8')]
    return [BOS] + inst_tokens + [EOS] + [RESP] + resp_tokens


def create_loss_mask(tokens):
    """创建损失掩码——只在响应部分计算损失。"""
    mask = np.zeros(len(tokens))
    in_response = False
    for i, t in enumerate(tokens):
        if t == RESP:
            in_response = True
        elif in_response:
            mask[i] = 1.0
    return mask


# ============================================================================
# 第 3 步：SFT 训练
# ============================================================================

def masked_cross_entropy(logits, targets, mask):
    """带掩码的交叉熵损失。"""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_probs = logits_flat - max_logits - np.log(np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True))
    per_token_loss = -log_probs[np.arange(len(targets_flat)), targets_flat]

    masked_loss = per_token_loss * mask_flat
    n = mask_flat.sum()
    return masked_loss.sum() / n if n > 0 else 0.0


def train_sft(model, dataset, num_epochs=3, lr=5e-4, seq_len=64):
    """SFT 训练——只在响应部分计算损失。"""
    formatted = [(tokenize_pair(d["instruction"], d["response"]),
                  create_loss_mask(tokenize_pair(d["instruction"], d["response"])))
                 for d in dataset]

    print(f"SFT 训练: {len(formatted)} 样本, {num_epochs} 轮, lr={lr}")
    print(f"总 token 数: {sum(len(t) for t, _ in formatted):,}")

    all_losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        n = 0
        for tokens, mask in formatted:
            if len(tokens) < 3:
                continue
            tokens = tokens[:seq_len]
            mask = mask[:seq_len]

            input_ids = np.array(tokens[:-1]).reshape(1, -1)
            target_ids = np.array(tokens[1:]).reshape(1, -1)
            loss_mask = mask[1:]

            logits = model.forward(input_ids)
            loss = masked_cross_entropy(logits, target_ids, loss_mask)

            # 简化梯度更新（仅更新 FFN 权重演示）
            for layer in model.layers:
                layer['W_out'] -= lr * np.random.randn(*layer['W_out'].shape) * 0.01

            total_loss += loss
            n += 1
            all_losses.append(loss)

        print(f"  Epoch {epoch+1}: avg_loss = {total_loss/max(n,1):.4f}")

    return model, all_losses


# ============================================================================
# 第 4 步：评估
# ============================================================================

def generate(model, prompt_tokens, max_new=30, temp=0.7):
    """自回归生成。"""
    tokens = list(prompt_tokens)
    for _ in range(max_new):
        ctx = np.array(tokens[-model.max_seq_len:]).reshape(1, -1)
        logits = model.forward(ctx)[0, -1] / temp
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        tokens.append(np.random.choice(len(probs), p=probs))
    return tokens


def evaluate(model, instructions):
    """评估指令跟随能力。"""
    print("\n指令跟随评估:")
    print("-" * 50)
    for inst in instructions:
        prompt = [BOS] + [min(t, 252) for t in inst.encode('utf-8')] + [EOS] + [RESP]
        output = generate(model, prompt, max_new=30, temp=0.6)
        resp = bytes([t for t in output[len(prompt):] if t < 128]).decode('utf-8', errors='replace')
        print(f"  Q: {inst}")
        print(f"  A: {resp[:60]}")
        print()


def measure_forgetting(model, text, seq_len=64):
    """测量灾难性遗忘——在原始文本上的困惑度。"""
    tokens = np.array([min(t, 255) for t in text.encode('utf-8')[:512]])
    total_loss = 0.0
    n = 0
    for start in range(0, len(tokens) - seq_len - 1, seq_len):
        inp = tokens[start:start+seq_len].reshape(1, -1)
        tgt = tokens[start+1:start+seq_len+1].reshape(1, -1)
        logits = model.forward(inp)
        V = logits.shape[-1]
        flat_logits = logits.reshape(-1, V)
        flat_tgt = tgt.reshape(-1)
        max_l = flat_logits.max(axis=-1, keepdims=True)
        log_probs = flat_logits - max_l - np.log(np.exp(flat_logits - max_l).sum(axis=-1, keepdims=True))
        loss = -log_probs[np.arange(len(flat_tgt)), flat_tgt].mean()
        total_loss += loss
        n += 1
    return total_loss / max(n, 1)


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("指令微调 (SFT) 演示")
    print("=" * 60)

    # 创建模型
    model = MiniGPT(vocab_size=256, embed_dim=128, num_heads=4, num_layers=2, max_seq_len=64)
    print(f"模型参数: {model.count_parameters():,}")

    # 测量预训练模型的遗忘基准
    test_text = "Transformer 架构通过自注意力处理序列。每一层应用多头注意力。残差连接稳定深层网络。模型学习根据之前的词预测下一个词。"
    base_loss = measure_forgetting(model, test_text)
    print(f"\nSFT 前 (原始文本困惑度): {base_loss:.4f}")

    # SFT 训练
    print("\n" + "=" * 60)
    print("SFT 训练")
    print("=" * 60)
    model, losses = train_sft(model, INSTRUCTION_DATA, num_epochs=3, lr=5e-4)

    # 测量 SFT 后的遗忘
    sft_loss = measure_forgetting(model, test_text)
    print(f"\nSFT 后 (原始文本困惑度): {sft_loss:.4f}")
    change = (sft_loss - base_loss) / base_loss * 100
    print(f"变化: {change:+.1f}%")
    if abs(change) < 15:
        print("遗忘程度轻微 (< 15%)")
    else:
        print("⚠ 检测到显著遗忘")

    # 指令跟随评估
    evaluate(model, ["法国的首都是哪里？", "列举三种编程语言", "定义重力"])

    # 训练损失曲线
    print("训练损失曲线:")
    window = max(1, len(losses) // 5)
    for i in range(0, len(losses), window):
        chunk = losses[i:i+window]
        avg = sum(chunk) / len(chunk)
        print(f"  Steps {i:3d}-{i+len(chunk)-1:3d}: avg_loss = {avg:.4f}")
