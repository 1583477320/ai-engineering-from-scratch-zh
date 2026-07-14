# 3D Gaussian Splatting 概念实现
# 演示 3D 高斯椭球体表示和体素扩散

import torch
import torch.nn as nn
import numpy as np


# ============================================================================
# 第 1 步：3D Gaussian 椭球体
# ============================================================================

def quaternion_to_rotation_matrix(q):
    """
    四元数 (w, x, y, z) 转旋转矩阵 (3x3)。
    四元数是 3D 旋转的紧凑表示——比欧拉角更稳定（无万向锁）。
    """
    w, x, y, z = q
    return torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


class Gaussian3D:
    """单个 3D 高斯椭球体。"""

    def __init__(self, position=None, scale=None, rotation=None,
                 opacity=None, color=None):
        self.position = position if position is not None else torch.zeros(3)
        self.scale = scale if scale is not None else torch.ones(3) * 0.1
        self.rotation = rotation if rotation is not None else torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.opacity = opacity if opacity is not None else torch.tensor(0.5)
        self.color = color if color is not None else torch.tensor([1.0, 0.5, 0.2])

    @property
    def covariance(self):
        """计算 3D 协方差矩阵——决定椭球的形状和方向。"""
        S = torch.diag(self.scale)
        R = quaternion_to_rotation_matrix(self.rotation)
        # 协方差 = R @ S @ S^T @ R^T
        return R @ S @ S.T @ R.T

    def evaluate(self, points):
        """
        计算 3D 点在当前高斯椭球下的值（概率密度）。
        Args:
            points: (N, 3) 的 3D 点坐标
        Returns:
            values: (N,) 每个点的概率密度
        """
        cov = self.covariance + torch.eye(3) * 1e-6  # 加小量防止奇异
        dev = points - self.position
        # 马氏距离平方
        try:
            inv_cov = torch.linalg.inv(cov)
            mahalanobis = (dev @ inv_cov * dev).sum(dim=-1)
            values = self.opacity * torch.exp(-0.5 * mahalanobis)
            return values
        except torch.linalg.LinalgError:
            return torch.zeros(points.size(0))


class Gaussian3DScene(nn.Module):
    """
    3D 高斯场景——一组可训练的高斯椭球体集合。
    这是 3DGS 的核心数据结构。
    """

    def __init__(self, num_gaussians=10000):
        super().__init__()
        self.num_gaussians = num_gaussians

        # 所有参数都是可训练的
        self.positions = nn.Parameter(torch.randn(num_gaussians, 3) * 0.5)
        self.scales = nn.Parameter(torch.randn(num_gaussians, 3))
        self.rotations = nn.Parameter(torch.randn(num_gaussians, 4))
        self.opacities = nn.Parameter(torch.randn(num_gaussians))
        self.colors = nn.Parameter(torch.rand(num_gaussians, 3))

    def forward(self):
        """返回当前场景的所有高斯参数。"""
        return {
            "positions": self.positions,
            "scales": torch.exp(self.scales),  # 正数缩放
            "rotations": self.rotations / self.rotations.norm(dim=-1, keepdim=True),  # 单位四元数
            "opacities": torch.sigmoid(self.opacities),  # [0, 1]
            "colors": self.colors,
        }


# ============================================================================
# 第 2 步：简化版体素扩散
# ============================================================================

class SimpleVoxelDiffusion(nn.Module):
    """简化版体素扩散模型——将 DDPM 扩展到 3D 体素空间。"""

    def __init__(self, in_channels=4, voxel_size=32):
        super().__init__()
        self.voxel_size = voxel_size

        # 3D 卷积编码器
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 4, stride=2, padding=1),  # 32→16
            nn.ReLU(),
        )
        # 瓶颈
        self.mid = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.ReLU(),
        )
        # 3D 解码器
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),  # 16→32
            nn.ReLU(),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        """
        Args:
            x: 带噪体素 (B, C, D, H, W)
            t: 时间步 (B,)
        Returns:
            预测的噪声 (B, C, D, H, W)
        """
        # 简化：没有时间嵌入，直接处理
        h = self.enc(x)
        h = self.mid(h)
        return self.dec(h)


# ============================================================================
# 第 3 步：Shap-E 风格的条件 3D 生成
# ============================================================================

class SimpleShapECondition(nn.Module):
    """
    简化版 Shap-E 条件生成。
    文本/图像编码 → 条件注入 → 生成 3D 潜向量 → 解码为 3D 表示。
    """

    def __init__(self, text_dim=768, latent_dim=512, output_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, latent_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # Transformer 在潜空间上做扩散
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=8, batch_first=True,
            dim_feedforward=latent_dim * 4,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.output_proj = nn.Linear(latent_dim, output_dim)

    def forward(self, text_encoding, timestep):
        """
        Args:
            text_encoding: 文本嵌入 (B, text_dim)
            timestep: 时间步 (B,)
        Returns:
            latent: 生成的 3D 潜向量 (B, output_dim)
        """
        t_emb = self.time_embed(timestep)
        cond = self.text_proj(text_encoding) + t_emb
        cond = cond.unsqueeze(1)

        latent = self.transformer(cond)
        return self.output_proj(latent.squeeze(1))

    @staticmethod
    def sds_loss(rendered_image, text_embed, diffusion_model):
        """
        SDS (Score Distillation Sampling) 损失。
        用 2D 扩散模型评估 3D 渲染图像的质量。
        """
        # 添加噪声到渲染图像
        noise = torch.randn_like(rendered_image)
        t = torch.randint(0, 1000, (rendered_image.size(0),))

        alpha_bar = torch.linspace(0.999, 0.001, 1000, device=rendered_image.device)
        alpha_t = alpha_bar[t].view(-1, 1, 1, 1)
        noisy = alpha_t * rendered_image + (1 - alpha_t).sqrt() * noise

        # 扩散模型预测噪声
        predicted_noise = diffusion_model(noisy, t, text_embed)

        # SDS 损失 = 预测噪声与添加噪声的差异
        loss = (predicted_noise - noise).abs().mean()
        return loss


# ============================================================================
# 主程序：演示
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 1. 测试 3D Gaussian
    print("=== 3D Gaussian 椭球体 ===")
    g = Gaussian3D(
        position=(0.5, 0.0, -0.3),
        scale=(0.2, 0.1, 0.3),
        rotation=(1.0, 0.0, 0.0, 0.0),
        opacity=0.8,
        color=(0.9, 0.2, 0.1),
    )
    test_points = torch.randn(100, 3)
    values = g.evaluate(test_points)
    print(f"形状: {g.position.numpy()}")
    print(f"协方差矩阵:\n{g.covariance.numpy()}")
    print(f"测试点 (N={len(test_points)}) 上的最大值: {values.max().item():.4f}")

    # 2. 测试高斯场景
    print("\n=== 3D Gaussian 场景 ===")
    scene = Gaussian3DScene(num_gaussians=5000).to(device)
    params = scene()
    print(f"高斯数量: {scene.num_gaussians}")
    print(f"参数形状: ", {k: v.shape for k, v in params.items()})
    total_params = sum(p.numel() for p in scene.parameters())
    print(f"总可训练参数: {total_params:,}")

    # 3. 测试体素扩散
    print("\n=== 体素扩散模型 ===")
    voxel_model = SimpleVoxelDiffusion(in_channels=4, voxel_size=32).to(device)
    dummy_voxels = torch.randn(2, 4, 32, 32, 32).to(device)
    dummy_t = torch.randint(0, 1000, (2,), device=device)
    with torch.no_grad():
        noise_pred = voxel_model(dummy_voxels, dummy_t)
    print(f"体素输入: {dummy_voxels.shape}")
    print(f"噪声预测输出: {noise_pred.shape}")

    # 4. 测试 Shap-E 条件生成
    print("\n=== Shap-E 条件生成 ===")
    shap_e = SimpleShapECondition(text_dim=768, latent_dim=256, output_dim=256).to(device)
    dummy_text = torch.randn(2, 768).to(device)
    dummy_t = torch.randint(0, 1000, (2,), device=device)
    with torch.no_grad():
        latent = shap_e(dummy_text, dummy_t)
    print(f"文本嵌入: {dummy_text.shape}")
    print(f"生成的 3D 潜向量: {latent.shape}")

    print("\n完成！")
