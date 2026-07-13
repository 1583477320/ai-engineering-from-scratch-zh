# main.py — 视觉 Transformer（ViT）演示
# 依赖：numpy>=1.24
# 对应课程：阶段 07 · 09（视觉 Transformer）

import numpy as np


# === Softmax ===

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# === 缩放点积注意力 ===

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V, weights


# === 多头注意力 ===

class MultiHeadSelfAttention:
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

    def forward(self, X):
        head_outputs = []
        for Wq, Wk, Wv in self.heads:
            Q = X @ Wq
            K = X @ Wk
            V = X @ Wv
            output, _ = scaled_dot_product_attention(Q, K, V)
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

class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, seed=42):
        self.mha = MultiHeadSelfAttention(d_model, n_heads, seed)
        self.ffn = FeedForward(d_model, d_ff, seed + 100)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.mha.forward(x)
        x = self.ln1.forward(x + attn_out)
        ffn_out = self.ffn.forward(x)
        x = self.ln2.forward(x + ffn_out)
        return x


# === ViT 模型 ===

class VisionTransformer:
    """视觉 Transformer（ViT）——将图像作为词元序列处理。"""

    def __init__(self, image_size=224, patch_size=16, d_model=64, n_heads=4, d_ff=128, n_layers=4, num_classes=10, seed=42):
        """初始化 ViT。

        Args:
            image_size: 输入图像尺寸（H=W）
            patch_size: patch 大小
            d_model: 嵌入维度
            n_heads: 注意力头数
            d_ff: 前馈网络维度
            n_layers: Transformer 层数
            num_classes: 分类数
            seed: 随机种子
        """
        self.patch_size = patch_size
        self.d_model = d_model

        # 计算 patch 数量
        num_patches = (image_size // patch_size) ** 2  # H/16 × W/16
        self.num_patches = num_patches

        # Patch 投影（线性层替代 Conv2d）
        rng = np.random.default_rng(seed)
        self.patch_proj = rng.normal(0, 0.02, (patch_size * patch_size * 3, d_model))

        # 可学习位置嵌入
        self.position_embedding = np.random.default_rng(seed + 1).normal(0, 0.02, (num_patches + 1, d_model))

        # [CLS] token
        self.cls_token = np.random.default_rng(seed + 2).normal(0, 0.02, (1, d_model))

        # Transformer 编码器
        self.layers = [TransformerBlock(d_model, n_heads, d_ff, seed + 10 + i) for i in range(n_layers)]

        # 分类头
        self.classifier = np.random.default_rng(seed + 1000).normal(0, 0.02, (d_model, num_classes))

    def _image_to_patches(self, image):
        """将图像分割为 patch。

        Args:
            image: 形状 (H, W, 3) 的图像

        Returns:
            patches: 形状 (num_patches, patch_size * patch_size * 3)
        """
        H, W, C = image.shape
        ps = self.patch_size
        patches = []
        for i in range(0, H, ps):
            for j in range(0, W, ps):
                patch = image[i:i+ps, j:j+ps, :]
                patches.append(patch.flatten())
        return np.array(patches)

    def forward(self, image):
        """前向传播。

        Args:
            image: 形状 (H, W, 3) 的图像

        Returns:
            logits: 形状 (num_classes,) 的分类分数
        """
        # 1. 图像分割为 patch
        patches = self._image_to_patches(image)  # (num_patches, p*p*3)

        # 2. 线性投影
        patch_embeddings = patches @ self.patch_proj  # (num_patches, d_model)

        # 3. 添加 [CLS] token
        cls = self.cls_token  # (1, d_model)
        x = np.concatenate([cls, patch_embeddings], axis=0)  # (num_patches+1, d_model)

        # 4. 添加位置嵌入
        x = x + self.position_embedding

        # 5. Transformer 编码器
        for layer in self.layers:
            x = layer.forward(x)

        # 6. [CLS] token 的输出用于分类
        cls_output = x[0]  # 第一个位置是 [CLS]
        logits = cls_output @ self.classifier
        return logits


# === 演示 ===

def demo():
    print("=" * 60)
    print("视觉 Transformer（ViT）— 演示")
    print("=" * 60)

    # 模拟输入图像（32×32 RGB 图像）
    H, W, C = 32, 32, 3
    patch_size = 8
    num_patches = (H // patch_size) ** 2

    print(f"\n图像尺寸: {H}×{W}×{C}")
    print(f"Patch 大小: {patch_size}×{patch_size}")
    print(f"Patch 数量: {num_patches} (H/{patch_size} × W/{patch_size})")

    # 创建随机图像
    rng = np.random.default_rng(42)
    image = rng.rand(H, W, C)

    # ViT 模型
    print(f"\n--- ViT 模型配置 ---")
    vit = VisionTransformer(
        image_size=H,
        patch_size=patch_size,
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_layers=2,
        num_classes=10,
        seed=42
    )
    print(f"d_model = {vit.d_model}")
    print(f"n_heads = 2")
    print(f"n_layers = 2")
    print(f"num_classes = 10")
    print(f"嵌入维度 = {vit.d_model}")

    # 图像分割
    print("\n--- 图像分割 ---")
    patches = vit._image_to_patches(image)
    print(f"Patch 形状: {patches.shape}")
    print(f"  每个 patch 展平后: {patch_size}×{patch_size}×{C} = {patches.shape[1]} 维")
    print(f"  Total patches: {num_patches}")

    # 前向传播
    print("\n--- ViT 前向传播 ---")
    logits = vit.forward(image)
    pred_class = np.argmax(logits)
    print(f"输出形状: {logits.shape}")
    print(f"预测类别: {pred_class} (概率={softmax(logits)[pred_class]:.3f})")

    # CNN vs ViT 对比
    print("\n--- CNN vs ViT 对比 ---")
    print(f"{'维度':<25} {'CNN':<25} {'ViT'}")
    print("-" * 70)
    print(f"{'归纳偏置':<25} {'局部性、平移不变性':<25} {'无先验'}")
    print(f"{'数据需求':<25} {'较低（归纳偏置补偿）':<25} {'较高（需学习局部结构）'}")
    print(f"{'长距离依赖':<25} {'池化层限制':<25} {'全局注意力，无限制'}")
    print(f"{'参数量':<25} {'25M（ResNet-50）':<25} {'86M（ViT-Base）'}")
    print(f"{'2026 选择':<25} {'边缘设备':<25} {'大数据集上的主流'}")

    # ViT 流程总结
    print("\n--- ViT 流程 ---")
    print("输入图像(H×W×C)")
    print(f"  → 分割为 {num_patches} 个 {patch_size}×{patch_size} patch")
    print(f"  → 展平每个 patch → patch_embeddings ({num_patches}, d_model)")
    print("  + [CLS] token → (n+1, d_model)")
    print("  + 可学习位置嵌入 → (n+1, d_model)")
    print("  → Transformer 编码器 → (n+1, d_model)")
    print("  → [CLS] 输出 → 分类头 → 预测类别")


if __name__ == "__main__":
    demo()
