# main.py — 混合专家模型（MoE）演示
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 11（混合专家模型）

import numpy as np


# === Softmax ===

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# === 专家 FFN ===

class ExpertFFN:
    """单个专家——一个小型 FFN。"""

    def __init__(self, d_model, d_ff, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / (d_model + d_ff)), (d_model, d_ff))
        self.W2 = rng.normal(0, np.sqrt(2.0 / (d_ff + d_model)), (d_ff, d_model))

    def forward(self, x):
        gelu = 0.5 * x @ self.W1 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x @ self.W1 + 0.044715 * (x @ self.W1)**3)))
        return gelu @ self.W2


# === 路由器 ===

class Router:
    """路由器——决定每个输入激活哪些专家。"""

    def __init__(self, d_model, n_experts, top_k=2, seed=42):
        self.top_k = top_k
        self.n_experts = n_experts
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.02, (d_model, n_experts))

    def forward(self, x):
        """计算每个专家的得分，返回 top-K 索引和权重。"""
        logits = x @ self.weights
        probs = softmax(logits)

        # 选 top-K
        if len(x.shape) == 1:
            # 单个位置
            indices = np.argsort(probs)[-self.top_k:][::-1]
            weights = probs[indices]
        else:
            # 多个位置
            indices = np.argsort(probs, axis=-1)[:, -self.top_k:][:, ::-1]
            weights = np.take_along_axis(probs, indices, axis=-1)

        return indices, weights


# === MoE 层 ===

class MoELayer:
    """混合专家层——多个专家 + 路由器。"""

    def __init__(self, d_model, d_ff, n_experts=8, top_k=2, seed=42):
        self.top_k = top_k
        self.router = Router(d_model, n_experts, top_k, seed)
        self.experts = [ExpertFFN(d_model, d_ff, seed + i) for i in range(n_experts)]

    def forward(self, x):
        """前向传播——路由器选专家，只激活被选中的专家。"""
        indices, weights = self.router.forward(x)

        if len(x.shape) == 1:
            output = np.zeros_like(x)
            for i, idx in enumerate(indices):
                output += weights[i] * self.experts[idx].forward(x)
            return output
        else:
            output = np.zeros_like(x)
            for i in range(x.shape[0]):
                for j, idx in enumerate(indices[i]):
                    output[i] += weights[i, j] * self.experts[idx].forward(x[i])
            return output

    def compute_auxiliary_loss(self, indices, n_experts):
        """计算辅助负载均衡损失——鼓励专家被均匀使用。

        如果所有专家被均匀使用，损失接近 0。
        如果某些专家从未被使用，损失很大。
        """
        counts = np.zeros(n_experts)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                counts[indices[i, j]] += 1
        total = counts.sum()
        probs = counts / total
        # 熵——越均匀越大
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        # 归一化到 [0, 1]
        max_entropy = np.log(n_experts)
        # 返回负载不均的损失（1 表示完全均匀）
        return 1 - entropy / max_entropy


# === 标准 FFN（对比用） ===

class StandardFFN:
    def __init__(self, d_model, d_ff, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / (d_model + d_ff)), (d_model, d_ff))
        self.W2 = rng.normal(0, np.sqrt(2.0 / (d_ff + d_model)), (d_ff, d_model))

    def forward(self, x):
        gelu = 0.5 * x @ self.W1 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x @ self.W1 + 0.044715 * (x @ self.W1)**3)))
        return gelu @ self.W2


# === 演示 ===

def demo():
    print("=" * 60)
    print("混合专家模型（MoE）— 演示")
    print("=" * 60)

    d_model = 16
    d_ff = 32
    n_experts = 8
    top_k = 2
    seq_len = 6

    # 标准 FFN
    print("\n--- 标准 FFN ---")
    standard_ffn = StandardFFN(d_model, d_ff, seed=42)
    x = np.random.randn(seq_len, d_model)
    output_standard = standard_ffn.forward(x)
    standard_params = d_model * d_ff * 2
    print(f"参数量: {standard_params:,}")
    print(f"输出形状: {output_standard.shape}")

    # MoE FFN
    print(f"\n--- MoE FFN (n_experts={n_experts}, top_k={top_k}) ---")
    moe = MoELayer(d_model, d_ff, n_experts, top_k, seed=42)
    output_moe = moe.forward(x)
    moe_params = n_experts * d_model * d_ff * 2
    activated_params = top_k * d_model * d_ff * 2
    print(f"总参数量: {moe_params:,} ({n_experts}×{d_model}×{d_ff}×2)")
    print(f"激活参数量: {activated_params:,} ({top_k}×{d_model}×{d_ff}×2)")
    print(f"参数量比: {moe_params/standard_params:.1f}x")
    print(f"计算量比: {activated_params/standard_params:.2f}x")

    # 路由分析
    print("\n--- 路由分析 ---")
    indices, weights = moe.router.forward(x)
    print(f"每个位置的专家选择:")
    for i in range(seq_len):
        print(f"  位置 {i}: 专家 {list(indices[i])} 权重 {weights[i].round(3)}")

    # 辅助损失
    aux_loss = moe.compute_auxiliary_loss(indices, n_experts)
    print(f"\n辅助负载均衡损失: {aux_loss:.4f} (0=完全均匀, 1=完全不均衡)")

    # 参数量对比
    print("\n--- 参数量对比 (d_model=512, d_ff=2048) ---")
    print(f"标准 FFN: 2 × 512 × 2048 = {2*512*2048:,}")
    print(f"MoE (8专家, top-2): 8 × 2 × 512 × 2048 = {8*2*512*2048:,} 参数, "
          f"激活 {2*2*512*2048:,}")
    print("MoE 效果: 参数量 8x, 计算量仅 2x")


if __name__ == "__main__":
    demo()
