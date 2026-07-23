# 3D Gaussian Splatting 从零实现（教学版）
#
# 本代码实现了 2D 高斯泼溅的核心算法：
#   - 2D 高斯密度评估
#   - Alpha 合成体渲染器
#   - 可训练的高斯场景模型
#   - 球谐函数 (SH) 颜色评估
#
# 运行方式：python main.py
# 依赖：torch >= 2.0, numpy


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 第 1 步：2D 高斯密度评估
# =============================================================================
# 给定一组 2D 高斯分布的中心和协方差矩阵，对图像上每个像素求密度值。
# 核心公式：density = exp(-0.5 * d^T Sigma^{-1} d)
# 其中 d = pixel - mean，即像素点到高斯中心的偏移向量。


def eval_2d_gaussian(means, covs, points):
    """
    评估 2D 高斯在多个像素点上的密度值。

    参数:
        means:  (G, 2)  G 个高斯的中心坐标
        covs:   (G, 2, 2) G 个高斯的 2x2 协方差矩阵
        points: (H, W, 2) 图像上每个像素的坐标

    返回:
        (G, H, W) 每个高斯在每个像素处的密度值
    """
    G = means.size(0)
    H, W, _ = points.shape

    # 将像素坐标展平为 (H*W, 2)
    flat = points.view(-1, 2)

    # 计算每个高斯的协方差矩阵的逆
    inv = torch.linalg.inv(covs)

    # diff 形状: (G, H*W, 2) — 每个高斯中心到每个像素的偏移
    diff = flat[None, :, :] - means[:, None, :]

    # 二次型: d^T Sigma^{-1} d，结果形状 (G, H*W)
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)

    # 高斯密度 = exp(-0.5 * d)
    density = torch.exp(-0.5 * d)

    return density.view(G, H, W)


# =============================================================================
# 第 2 步：2D 高斯泼溅渲染器
# =============================================================================
# 核心思想：将所有高斯按深度排序后从前到后做 Alpha 合成。
# 每像素颜色 = sum(alpha_i * transmittance_i * color_i)
# 透射率 T_i = prod_{j < i} (1 - alpha_j)


def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    2D 高斯泼溅渲染器。

    参数:
        means:     (G, 2)      高斯中心坐标
        covs:      (G, 2, 2)   高斯协方差矩阵
        colours:   (G, 3)      每个高斯的 RGB 颜色
        opacities: (G,)        每个高斯的透明度 [0, 1]
        depths:    (G,)        用于排序的深度标量
        image_size: (H, W)      输出图像的尺寸

    返回:
        (H, W, 3) 渲染后的图像
    """
    H, W = image_size
    device = means.device

    # 构建像素网格坐标
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    # 计算每个高斯在每个像素处的密度
    densities = eval_2d_gaussian(means, covs, points)

    # alpha = opacity * density，截断防止数值溢出
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    # 按深度从高到低排序（远到近），从前向后合成
    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    # 初始化透射率（全透明）和输出图像（全黑）
    T = torch.ones(H, W, device=device)
    out = torch.zeros(H, W, 3, device=device)

    # 从前到后迭代所有高斯，进行 Alpha 合成
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)

    return out


# =============================================================================
# 第 3 步：可训练的 2D 高斯场景模型
# =============================================================================
# 这是一个 PyTorch Module，包含了所有可通过梯度下降优化的参数。
# 关键设计模式：使用无约束参数 + 激活函数映射到合法范围。


class Splats2D(nn.Module):
    """
    2D 高斯泼溅场景的可训练模型。

    每个高斯包含以下可学习参数：
        - means:         2D 中心坐标
        - log_scale:     缩放系数（对数空间优化）
        - rot:           旋转角度（2D 中为单个标量）
        - colour_logits:  RGB 颜色的 logits（Sigmoid 映射到 [0, 1]）
        - opacity_logit: 透明度的 logits（Sigmoid 映射到 [0, 1]）
        - depth:         深度标量（用于排序）
    """

    def __init__(self, num_splats=64, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size

        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.full((num_splats, 2), math.log(3.0)))
        self.rot = nn.Parameter(torch.zeros(num_splats))
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.3)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def build_covariances(self):
        """
        从 log_scale 和 rot 构建协方差矩阵。
        3D 中公式为 Sigma = R S S^T R^T。
        2D 中 S = diag(exp(log_scale))，R 为 2D 旋转矩阵。
        """
        s = torch.exp(self.log_scale)  # 实际缩放系数
        c, si = torch.cos(self.rot), torch.sin(self.rot)

        # 2D 旋转矩阵: [[cos, -sin], [sin, cos]]
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)

        # S^2 = diag(s^2)
        S = torch.diag_embed(s ** 2)

        # Sigma = R @ S^2 @ R^T
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.build_covariances()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)


# =============================================================================
# 第 4 步：生成目标图像（带红色圆形和蓝色正方形）
# =============================================================================
def make_target(size=48):
    """
    创建一个合成目标图像，包含红色圆形和蓝色正方形，
    用于验证高斯泼溅能否精确拟合任意形状。
    """
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.ones((size, size, 3), dtype=np.float32)

    # 红色圆形 (center=(15,15), radius=8)
    mask = (xx - 15) ** 2 + (yy - 15) ** 2 < 8 ** 2
    img[mask] = [0.95, 0.2, 0.15]

    # 蓝色正方形 (center=(34,32), half-width=6)
    mask = (np.abs(xx - 34) < 6) & (np.abs(yy - 32) < 6)
    img[mask] = [0.2, 0.35, 0.95]

    return torch.from_numpy(img)


# =============================================================================
# 第 5 步：球谐函数（Spherical Harmonics）评估
# =============================================================================
# 球谐函数是定义在单位球面上的傅里叶基函数。
# 在 3DGS 中，每个高斯的颜色不仅取决于位置，还取决于观察方向。
# SH degree 3 提供 16 个系数 per color channel，足以表示镜面高光
# 和基本的视图相关颜色变化。


def sh_degree_3_basis(dirs):
    """
    计算 SH degree-3 基函数在给定方向上的值。

    参数:
        dirs: (*, 3) 单位方向向量 [x, y, z]

    返回:
        (*, 16) 16 个 SH 基函数的值
    """
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    # SH 归一化常数
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [
        1.092548430592079,
        1.092548430592079,
        0.315391565252520,
        1.092548430592079,
        0.546274215296039,
    ]
    C3 = [
        0.590043589926644,
        2.890611442640554,
        0.457045799464465,
        0.373176332590115,
        0.457045799464465,
        1.445305721320277,
        0.590043589926644,
    ]

    basis = torch.stack([
        # Degree 0 (1 term)
        torch.full_like(x, C0),
        # Degree 1 (3 terms)
        -C1 * y, C1 * z, -C1 * x,
        # Degree 2 (5 terms)
        C2[0] * xy, C2[1] * yz, C2[2] * (2 * z2 - x2 - y2),
        C2[3] * xz, C2[4] * (x2 - y2),
        # Degree 3 (7 terms)
        -C3[0] * y * (3 * x2 - y2),
        C3[1] * xy * z,
        -C3[2] * y * (4 * z2 - x2 - y2),
        C3[3] * z * (2 * z2 - 3 * x2 - 3 * y2),
        -C3[4] * x * (4 * z2 - x2 - y2),
        C3[5] * z * (x2 - y2),
        -C3[6] * x * (x2 - 3 * y2),
    ], dim=-1)

    return basis


def eval_sh_degree_3(sh_coeffs, dirs):
    """
    使用 SH degree-3 基函数计算方向相关的颜色。

    参数:
        sh_coeffs: (*, 16, 3) 每个高斯的 16 个 SH 系数，最后一维是 RGB
        dirs:      (*, 3)     观察方向的单位向量

    返回:
        (*, 3) 对应的 RGB 颜色值
    """
    basis = sh_degree_3_basis(dirs)
    # 基函数与系数的点积 = 最终颜色
    return torch.einsum("...b,...bc->...c", basis, sh_coeffs)


# =============================================================================
# 主程序：训练 2D 高斯拟合目标图像
# =============================================================================
def main():
    # 设置随机种子以确保可复现性
    torch.manual_seed(0)
    device = "cpu"

    # 创建目标图像（红色圆 + 蓝色方）
    target = make_target(48).to(device)

    # 初始化 48 个 2D 高斯模型
    model = Splats2D(num_splats=48, image_size=48).to(device)

    # Adam 优化器 — 同时优化所有参数（means, scale, rot, colour, opacity）
    opt = torch.optim.Adam(model.parameters(), lr=0.08)

    print("=" * 60)
    print("Fitting 48 2D Gaussians to a red circle + blue square...")
    print("=" * 60)

    # 训练循环：300 轮次
    for step in range(300):
        pred = model((48, 48))
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  step {step:3d}  mse {loss.item():.4f}")

    # 打印最终损失
    with torch.no_grad():
        final = F.mse_loss(model((48, 48)), target).item()
    print(f"final mse: {final:.4f}")

    # =========================================================
    # 验证球谐函数：在不同观察方向上评估同一个高斯的颜色
    # =========================================================
    print("\nSpherical harmonics sanity check:")
    sh = torch.randn(1, 16, 3)
    dirs = F.normalize(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        dim=-1,
    )
    rgb = eval_sh_degree_3(sh, dirs)
    print(f"  SH(16, 3) evaluated at 3 directions -> {tuple(rgb.shape)}")

    # 打印三个方向上的颜色差异
    for i, direction in enumerate(["+X axis", "+Y axis", "+Z axis"]):
        r, g, b = rgb[i].tolist()
        print(f"  {direction}: RGB = ({r:.3f}, {g:.3f}, {b:.3f})")


if __name__ == "__main__":
    main()
