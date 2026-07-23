# main.py — 玩具视觉语言模型（ToyVLM）教学实现
# 依赖：torch>=2.0
# 对应课程：第 04 阶段 · 第 25 课（视觉语言模型）

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
# 第 1 部分：基础组件
# ===================================================================

class Projector(nn.Module):
    """视觉-语言投影层（MLP）。

    这是 VLM 的核心组件。它负责将视觉编码器输出的特征向量
    映射到大语言模型的嵌入空间，使得两者可以在同一个语义空间中进行交互。
    """

    def __init__(self, vision_dim: int = 1024, language_dim: int = 4096,
                 hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, language_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ToyVLM(nn.Module):
    """简化版视觉语言模型。

    模拟完整的 VLM 管线：视觉 token → 投影层 → 池化 → 分类头。
    注意：这是一个教学玩具模型，不包含真实的 ViT 编码器或 LLM，
    仅用于理解 VLM 数据流和训练流程。
    """

    def __init__(self, vit_dim: int = 1024, llm_dim: int = 4096,
                 num_classes: int = 5):
        super().__init__()
        self.projector = Projector(vision_dim=vit_dim, language_dim=llm_dim)
        self.classifier = nn.Linear(llm_dim, num_classes)

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        # 1) 视觉 token 投影到语言嵌入空间
        projected = self.projector(vision_tokens)       # (B, N, LLM_DIM)
        # 2) 平均池化合并所有 patch 的信息
        pooled = projected.mean(dim=1)                   # (B, LLM_DIM)
        # 3) 分类
        return self.classifier(pooled)                   # (B, NUM_CLASSES)


# ===================================================================
# 第 2 部分：DeepStack 多层特征融合
# ===================================================================

class DeepStackFeatureFuser(nn.Module):
    """DeepStack 多层特征融合器。

    现代 VLM（如 Qwen3-VL）从 ViT 的多个深度采集特征：
    - 浅层（前 1/3）：空间位置、纹理、边缘
    - 中层（中 1/3）：局部结构、部件
    - 深层（后 1/3）：全局语义、类别信息

    将它们拼接再投影，可以让 LLM 同时获得"看到哪里"和"看到了什么"两种信号。
    """

    def __init__(self, per_layer_dim: int = 1024, num_layers: int = 3,
                 language_dim: int = 4096):
        super().__init__()
        stacked_dim = per_layer_dim * num_layers
        self.fusion = nn.Sequential(
            nn.Linear(stacked_dim, language_dim),
            nn.GELU(),
        )

    def forward(self, multi_layer_features: list) -> torch.Tensor:
        """
        Args:
            multi_layer_features: 每层特征的列表，每项形状为 (B, num_patches, d)
        Returns:
            融合后的特征，形状 (B, num_patches, language_dim)
        """
        # 沿通道维度拼接所有层的特征
        stacked = torch.cat(multi_layer_features, dim=-1)  # (B, N, d * num_layers)
        return self.fusion(stacked)                            # (B, N, LLM_DIM)


# ===================================================================
# 第 3 部分：跨模态误差率（CMER）— 诊断 VLM 幻觉
# ===================================================================

def compute_cmer(image_embeddings: torch.Tensor,
                 text_embeddings: torch.Tensor,
                 text_confidence: torch.Tensor,
                 similarity_threshold: float = 0.25,
                 confidence_threshold: float = 0.8) -> float:
    """计算跨模态误差率（CMER）。

    CMER 衡量的是"高置信度描述但图像证据不足"的比率。
    这是工业界监控 VLM 幻觉问题的核心 KPI。

    Args:
        image_embeddings: 图像嵌入，形状 (B, D)
        text_embeddings: 文本嵌入（来自 LLM 的视觉相关部分），形状 (B, D)
        text_confidence: 模型对每个输出的置信度分数，形状 (B,)
        similarity_threshold: 低于此值认为图文不一致
        confidence_threshold: 高于此值认为是"高置信度"

    Returns:
        CMER 值，范围 [0, 1]，越高表示幻觉越严重
    """
    img_norm = F.normalize(image_embeddings, dim=-1)
    txt_norm = F.normalize(text_embeddings, dim=-1)

    # 计算余弦相似度
    sim = (img_norm * txt_norm).sum(dim=-1)

    # 高置信度 + 低相似度 = 幻觉输出
    hallucinated = (text_confidence > confidence_threshold) & (sim < similarity_threshold)

    return hallucinated.float().mean().item()


# ===================================================================
# 第 4 部分：合成数据与训练循环
# ===================================================================

def generate_synthetic_vision_data(num_samples: int = 200, num_classes: int = 5,
                                   num_patches: int = 16, d_vision: int = 32,
                                   seed: int = 42) -> tuple:
    """生成合成视觉 token 数据。

    每个类别有一组原型特征，每张图片是从该原型加噪声生成的 patch 序列。
    这模拟了 ViT 编码真实图像后产生的 token 分布。
    """
    generator = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(num_classes, d_vision, generator=generator)

    X, Y = [], []
    per_class = num_samples // num_classes

    for c in range(num_classes):
        for _ in range(per_class):
            # 每个样本是一组 patch token
            base = prototypes[c].unsqueeze(0).expand(num_patches, -1)
            noise = 0.1 * torch.randn(num_patches, d_vision, generator=generator)
            X.append(base + noise)
            Y.append(c)

    return torch.stack(X), torch.tensor(Y)


def train_vlm(model, X_train, Y_train, X_val, Y_val, lr: float = 1e-3,
              epochs: int = 50, batch_size: int = 32):
    """训练 VLM 主循环。

    模拟两阶段训练流程的第 1 阶段（冻结视觉编码器，只训练投影层 + 分类头）。
    在实际 VLM 训练中，这一步被称为"对齐阶段（Alignment Stage）"。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X_train)

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(epoch))
        total_loss = 0.0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            logits = model(X_train[batch_idx])
            loss = F.cross_entropy(logits, Y_train[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)

        avg_loss = total_loss / n

        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc = (val_logits.argmax(dim=-1) == Y_val).float().mean().item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")


# ===================================================================
# 第 5 部分：主程序
# ===================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 50)
    print("ToyVLM 教学演示")
    print("=" * 50)

    # --- 测试 1：投影层形状变换 ---
    print("\n[测试 1] 投影层形状变换")
    projector = Projector(vision_dim=32, language_dim=64, hidden_dim=64)
    batch_size, num_patches = 4, 16
    dummy_vision = torch.randn(batch_size, num_patches, 32)
    projected = projector(dummy_vision)
    print(f"  输入形状: {dummy_vision.shape}")  # (4, 16, 32)
    print(f"  输出形状: {projected.shape}")      # (4, 16, 64)

    # --- 测试 2：ToyVLM 训练 ---
    print("\n[测试 2] ToyVLM 对齐训练")
    # 使用更大的噪声使数据更有区分度，避免 loss 瞬间归零
    X, Y = generate_synthetic_vision_data(
        num_samples=200, num_classes=5, num_patches=16, d_vision=32, seed=42
    )
    # 增加 patch 数间的差异来模拟真实 ViT token 的多样性
    X = X + 0.5 * torch.randn_like(X)
    split = int(0.85 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]
    print(f"  训练集 {len(X_train)} 样本, 验证集 {len(X_val)} 样本")

    model = ToyVLM(vit_dim=32, llm_dim=64, num_classes=5)
    train_vlm(model, X_train, Y_train, X_val, Y_val, epochs=50)

    # --- 测试 3：DeepStack 特征融合 ---
    print("\n[测试 3] DeepStack 多层特征融合")
    layers = [torch.randn(4, 16, 32) for _ in range(3)]  # 3 个深度的 ViT 输出
    fuser = DeepStackFeatureFuser(per_layer_dim=32, num_layers=3, language_dim=64)
    fused = fuser(layers)
    print(f"  3 层 × (4, 16, 32) → DeepStack 融合 → {tuple(fused.shape)}")

    # --- 测试 4：CMER 幻觉检测 ---
    print("\n[测试 4] CMER 跨模态误差率诊断")
    embed_dim = 32
    n_samples = 8
    image = F.normalize(torch.randn(n_samples, embed_dim), dim=-1)

    # 前 4 条正确（与图像高度一致）：从图像中取前 4 条并加微小噪声
    good_text_base = image[:4] + 0.05 * torch.randn_like(image[:4])
    good_text = F.normalize(good_text_base, dim=-1)
    # 后 4 条幻觉（随机文本，与图像无关）
    bad_text = F.normalize(torch.randn(4, embed_dim), dim=-1)
    combined_text = torch.cat([good_text, bad_text], dim=0)
    confidence = torch.tensor([0.95, 0.90, 0.88, 0.85, 0.92, 0.90, 0.87, 0.91])

    cmer = compute_cmer(image, combined_text, confidence)
    print(f"  CMER = {cmer:.3f}  （期望 ~0.500，因为 4/8 是幻觉）")

    # --- 总结 ---
    print("\n" + "=" * 50)
    print("所有测试完成。")
    print("下一课请尝试用 HuggingFace Transformers 加载真实的 LLaVA 模型。")
    print("=" * 50)
