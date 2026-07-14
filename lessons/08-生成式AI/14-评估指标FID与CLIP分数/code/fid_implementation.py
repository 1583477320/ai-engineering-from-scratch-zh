# FID 从零实现
# 演示 Fréchet Inception Distance 的计算流程

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg


# ============================================================================
# 第 1 步：FID 核心计算
# ============================================================================

def compute_fid(mu_real, sigma_real, mu_gen, sigma_gen, eps=1e-6):
    """
    计算 Fréchet Inception Distance (FID)。

    公式：FID = ||μ_r - μ_g||² + Tr(σ_r + σ_g - 2·sqrt(σ_r·σ_g))

    Args:
        mu_real: 真实图像特征的均值向量 (D,)
        sigma_real: 真实图像特征的协方差矩阵 (D, D)
        mu_gen: 生成图像特征的均值向量 (D,)
        sigma_gen: 生成图像特征的协方差矩阵 (D, D)
        eps: 数值稳定性常数
    Returns:
        fid: FID 值（越低越好）
    """
    # 均值差平方
    diff = mu_real - mu_gen
    diff_sq = torch.dot(diff, diff).item()

    # 协方差矩阵乘积的平方根
    # sqrtm(σ_r · σ_g)
    product = sigma_real.cpu().numpy() @ sigma_gen.cpu().numpy()

    # 数值稳定性：对称化（sqrtm 要求输入对称）
    product = (product + product.T) / 2

    covmean, _ = linalg.sqrtm(product, disp=False)

    # 处理复数结果
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 迹部分：Tr(σ_r) + Tr(σ_g) - 2·Tr(sqrt(σ_r·σ_g))
    trace_part = (
        torch.trace(sigma_real).item()
        + torch.trace(sigma_gen).item()
        - 2 * np.trace(covmean)
    )

    return diff_sq + trace_part


def compute_statistics(features):
    """
    计算特征向量的均值和协方差。
    Args:
        features: 特征矩阵 (N, D)
    Returns:
        mu: 均值向量 (D,)
        sigma: 协方差矩阵 (D, D)
    """
    mu = features.mean(dim=0)
    sigma = torch.cov(features.T)  # (D, D)
    return mu, sigma


def compute_fid_from_features(real_features, gen_features):
    """
    从特征矩阵直接计算 FID。
    Args:
        real_features: 真实图像的特征 (N_r, D)
        gen_features: 生成图像的特征 (N_g, D)
    Returns:
        fid: FID 值
    """
    mu_real, sigma_real = compute_statistics(real_features)
    mu_gen, sigma_gen = compute_statistics(gen_features)
    return compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)


# ============================================================================
# 第 2 步：Inception-v3 特征提取器
# ============================================================================

class InceptionV3FeatureExtractor(nn.Module):
    """
    Inception-v3 特征提取器。
    从倒数第二层（融合池化层前）提取 2048 维特征。
    """

    def __init__(self, device="cpu"):
        super().__init__()
        # 加载预训练的 Inception-v3
        from torchvision.models import inception_v3, Inception_V3_Weights
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

        # 移除分类头，保留到融合池化层
        self.base = nn.Sequential(
            *list(inception.children())[:-1]
        )

        # 自适应平均池化
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to(device)

    def forward(self, x):
        """
        提取 2048 维特征。
        Args:
            x: 图像张量 (B, 3, 299, 299)，归一化到 ImageNet 标准
        Returns:
            features: (B, 2048)
        """
        # Inception 特征提取
        h = self.base(x)
        # 全局池化
        h = self.pool(h)
        # 展平
        return h.view(h.size(0), -1)


def preprocess_for_inception(images):
    """
    Inception-v3 的图像预处理。
    Args:
        images: 图像张量 (B, C, H, W)，值域 [0, 1]
    Returns:
        processed: (B, 3, 299, 299)，ImageNet 归一化
    """
    from torchvision import transforms as T

    # 调整大小（Inception 期望 299×299）
    transform = T.Compose([
        T.Resize(299),
        T.CenterCrop(299),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return transform(images)


# ============================================================================
# 第 3 步：完整评估流水线
# ============================================================================

def evaluate_fid(real_loader, gen_loader, device="cpu", num_features=50000):
    """
    完整 FID 评估流水线。
    Args:
        real_loader: 真实图像 DataLoader (图像, 标签)
        gen_loader: 生成图像 DataLoader (图像,)
        device: 计算设备
        num_features: 用于计算的特征数量
    Returns:
        fid: FID 值
    """
    extractor = InceptionV3FeatureExtractor(device=device)

    real_features = []
    gen_features = []

    # 收集真实图像特征
    with torch.no_grad():
        for batch in real_loader:
            images = batch[0].to(device)
            features = extractor(images)
            real_features.append(features.cpu())
            if len(real_features) * images.size(0) >= num_features:
                break

        # 收集生成图像特征
        for gen_images in gen_loader:
            if isinstance(gen_images, (list, tuple)):
                gen_images = gen_images[0]
            gen_images = gen_images.to(device)
            features = extractor(gen_images)
            gen_features.append(features.cpu())
            if len(gen_features) * gen_images.size(0) >= num_features:
                break

    real_features = torch.cat(real_features, dim=0)[:num_features]
    gen_features = torch.cat(gen_features, dim=0)[:num_features]

    print(f"真实图像特征: {real_features.shape}")
    print(f"生成图像特征: {gen_features.shape}")

    return compute_fid_from_features(real_features, gen_features)


# ============================================================================
# 第 4 步：快速验证——用随机数据测试 FID 实现
# ============================================================================

def test_fid_on_random_data():
    """
    在随机数据上验证 FID 实现。
    - 完全相同的数据 → FID ≈ 0
    - 相似的数据 → FID 较小
    - 不同的数据 → FID 较大
    """
    torch.manual_seed(42)
    D = 2048  # Inception-v3 特征维度

    # Case 1: 完全相同的分布
    mu1 = torch.zeros(D)
    sigma1 = torch.eye(D)
    fid_identical = compute_fid(mu1, sigma1, mu1, sigma1)
    print(f"Case 1 - 完全相同分布: FID = {fid_identical:.6f} (应为 0)")

    # Case 2: 不同分布的随机数据
    N = 10000
    real_features = torch.randn(N, D)
    gen_features = torch.randn(N, D)
    mu_real, sigma_real = compute_statistics(real_features)
    mu_gen, sigma_gen = compute_statistics(gen_features)
    fid_different = compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"Case 2 - 两个独立标准正态分布: FID ≈ {fid_different:.2f}")

    # Case 3: 偏移的分布（均值偏移 1）
    mu_shifted = mu_real + 1.0
    fid_shifted = compute_fid(mu_shifted, sigma_real, mu_gen, sigma_gen)
    print(f"Case 3 - 均值偏移 1: FID ≈ {fid_shifted:.2f}")

    # Case 4: 缩放的分布（方差缩小）
    sigma_scaled = sigma_gen * 0.5
    fid_scaled = compute_fid(mu_real, sigma_real, mu_gen, sigma_scaled)
    print(f"Case 4 - 方差缩小 0.5x: FID ≈ {fid_scaled:.2f}")

    return {
        "identical": fid_identical,
        "different": fid_different,
        "shifted": fid_shifted,
        "scaled": fid_scaled,
    }


if __name__ == "__main__":
    print("=== FID 随机数据测试 ===\n")
    results = test_fid_on_random_data()

    print("\n=== 解释 ===")
    print("FID = 0：完全相同分布")
    print("FID 较小（~1-5）：分布相似（如不同的随机种子）")
    print("FID 中等（~5-15）：分布有可见差异")
    print("FID 较大（>15）：分布明显不同")
    print("\n注意：在真实评估中，FID 需要 ≥ 10000 样本才稳定。")
