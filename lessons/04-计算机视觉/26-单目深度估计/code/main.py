# main.py — 单目深度估计：从零实现核心组件
# 依赖：torch>=2.0, numpy
# 对应课程：第 04 阶段 · 第 26 课（单目深度估计）

import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# 第 1 部分：深度评估指标
# ============================================================================

def abs_rel_error(pred: torch.Tensor, target: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> float:
    """计算绝对相对误差（AbsRel）。

    AbsRel = mean(|d_pred - d_gt| / d_gt)，越低越好。
    生产级模型的典型值在 0.05 ~ 0.10 之间。

    Args:
        pred: 预测深度图，形状 (H, W) 或 (batch, H, W)
        target: 真实深度图，形状与 pred 相同
        mask: 可选的布尔掩码，标记有效像素（True = 有效）

    Returns:
        AbsRel 标量值
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    # clamp 防止除零，最小值为 1e-6
    return (torch.abs(pred - target) / target.clamp(min=1e-6)).mean().item()


def delta_accuracy(pred: torch.Tensor, target: torch.Tensor,
                   threshold: float = 1.25,
                   mask: Optional[torch.Tensor] = None) -> float:
    """计算 delta 准确率。

    delta < t 准确率 = 满足 max(d_pred/d_gt, d_gt/d_pred) < t 的像素比例。
    越高越好。SOTA 模型在 KITTI 上的 delta<1.25 通常超过 0.95。

    Args:
        pred: 预测深度
        target: 真实深度
        threshold: 阈值，默认 1.25
        mask: 可选有效像素掩码

    Returns:
        delta 准确率（0~1 之间的浮点数）
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    ratio = torch.maximum(
        pred / target.clamp(min=1e-6),
        target / pred.clamp(min=1e-6),
    )
    return (ratio < threshold).float().mean().item()


def rmse_log(pred: torch.Tensor, target: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> float:
    """计算对数空间 RMSE。

    sqrt(mean((log(d_pred) - log(d_gt))^2))，用于相对深度评估。
    与 AbsRel 互补——AbsRel 对小深度敏感，RMSE_log 在大深度区间更稳定。

    Args:
        pred: 预测深度
        target: 真实深度
        mask: 可选有效像素掩码

    Returns:
        RMSE Log 标量值
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    log_pred = torch.log(pred.clamp(min=1e-6))
    log_target = torch.log(target.clamp(min=1e-6))
    return torch.sqrt(((log_pred - log_target) ** 2).mean()).item()


# ============================================================================
# 第 2 部分：尺度-偏移对齐（相对深度模型的关键步骤）
# ============================================================================

def align_scale_shift(pred: torch.Tensor, target: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """将相对深度预测对齐到真实深度。

    对于 MiDaS、Depth Anything 等相对深度模型，预测值只有顺序和比例含义，
    没有绝对尺度。在评估之前需要用最小二乘法拟合 a * pred + b = target。

    这步至关重要——不对齐直接算 AbsRel 会完全测不出模型的排序质量。

    Args:
        pred: 预测深度图
        target: 真实深度图
        mask: 可选有效像素掩码

    Returns:
        对齐后的预测深度图
    """
    if mask is not None:
        p = pred[mask]
        t = target[mask]
    else:
        p = pred.flatten()
        t = target.flatten()

    # 构建线性回归设计矩阵 [pred, 1]
    A = torch.stack([p, torch.ones_like(p)], dim=1)
    sol = torch.linalg.lstsq(A, t.unsqueeze(-1))
    a, b = sol.solution[:2, 0]

    return a * pred + b


# ============================================================================
# 第 3 部分：相机内参与深度→点云提升
# ============================================================================

def depth_to_point_cloud(depth: np.ndarray,
                         intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
    """将单通道深度图提升为 3D 点云。

    使用小孔相机模型：给定图像坐标 (u, v)、深度值 d 和相机内参，
    计算像素对应的 3D 空间坐标 (X, Y, Z)。

    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

    Args:
        depth: 深度图，形状 (H, W)，单位为米或其他统一单位
        intrinsics: 相机内参 (fx, fy, cx, cy)
            - fx, fy: 焦距（像素单位）
            - cx, cy: 主点坐标（通常为图像中心）

    Returns:
        点云数组，形状 (H, W, 3)，每个元素为 (X, Y, Z)
    """
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics

    # 生成每个像素的图像坐标网格
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # 小孔相机反投影公式
    z = depth.astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1)


def point_cloud_from_depth_with_intrinsics(depth: np.ndarray,
                                            camera_matrix: np.ndarray,
                                            img_size: Tuple[int, int]) -> np.ndarray:
    """使用 OpenCV 风格相机矩阵提升深度图为点云。

    这是 `depth_to_point_cloud` 的工程化版本，支持任意大小的相机内参矩阵。

    Args:
        depth: 深度图 (H, W)
        camera_matrix: 3x3 相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        img_size: 图像尺寸 (width, height)

    Returns:
        点云 (N, 3)，N = H * W
    """
    H, W = depth.shape
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    z = depth.reshape(-1)

    x = (u.ravel() - cx) * z / fx
    y = (v.ravel() - cy) * z / fy

    points = np.column_stack([x, y, z])
    return points


# ============================================================================
# 第 4 部分：合成深度场景（用于教学演示）
# ============================================================================

def synthetic_depth(size: int = 96, seed: int = 0) -> np.ndarray:
    """生成合成的深度场景，用于测试深度估计管线。

    场景包含：
    - 地面：从近（顶部）到远（底部）的深度渐变
    - 一个盒子：位于场景中间偏下的近距离物体

    Args:
        size: 图像边长（正方形）
        seed: 随机种子

    Returns:
        合成深度图，形状 (size, size)，单位模拟米
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")

    # 地面：线性渐变，从 1m（近）到 5m（远）
    depth = 1.0 + (yy / size) * 4.0

    # 中间有一个箱子，距离相机 2m（比地面前进）
    box_half = size // 6
    box_center_y = int(size * 0.6)
    mask = (np.abs(xx - size / 2) < box_half) & \
           (np.abs(yy - box_center_y) < box_half)
    depth[mask] = 2.0

    return depth.astype(np.float32)


# ============================================================================
# 第 5 部分：简易深度解码器（教学用）
# ============================================================================

class DepthDecoder(nn.Module):
    """简易深度解码器（DPT 风格的简化版）。

    模拟 Depth Anything 架构：冻结的 ViT 编码器 + 可训练的卷积解码器。
    编码器被省略（视为已冻结的特征提取器），这里只展示解码器部分。

    输入是 ViT 输出的密集特征（下采样 14 倍），输出是全分辨率深度图。
    """

    def __init__(self, in_channels: int = 768, hidden_dim: int = 256,
                 out_channels: int = 1):
        super().__init__()

        # 特征升维：768 → 256
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # 逐步上采样：1/14 → 1/7 → 1/4 → 1/2 → 1/1
        self.decoder = nn.Sequential(
            # 第 1 级上采样：通道不变，分辨率翻倍
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            # 第 2 级上采样
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.GELU(),
            # 第 3 级上采样
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.GELU(),
            # 第 4 级上采样：恢复到全分辨率
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim // 8, out_channels, kernel_size=1),
            # Sigmoid 将输出压缩到 [0, 1]（归一化的相对深度）
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: ViT 输出的密集特征，形状 (B, C, H', W')，
                     其中 H' = H/14, W' = W/14

        Returns:
            深度图，形状 (B, 1, H, W)，归一化到 [0, 1]
        """
        x = self.projector(features)
        return self.decoder(x)


# ============================================================================
# 第 6 部分：无监督深度估计的损失函数（Eigen 方法）
# ============================================================================

class PhotometricLoss(nn.Module):
    """光度一致性损失（Eigen 无监督深度估计的核心）。

    无监督深度估计的关键思想：如果我有两张来自不同视角的图片，
    以及对应的深度图和相机运动（位姿），我就可以把一张图片"缝合"
    到另一张的位置上。缝合结果与原图应该一致——这个一致性就是损失信号。

    L = sum over views of || I_target - Warp(I_source, depth, pose) ||
    """

    def __init__(self, gamma: float = 0.85):
        """
        Args:
            gamma: 结构相似度（SSIM）权重。综合损失 = gamma * SSIM_loss
                   + (1-gamma) * 光度损失。SSIM 能捕捉结构信息，
                   纯像素级 MSE 容易在均匀区域产生伪影。
        """
        super().__init__()
        self.gamma = gamma
        # 简单的高斯模糊核用于多尺度 SSIM 计算
        self.register_buffer("gaussian_kernel", self._create_gaussian_kernel())

    @staticmethod
    def _create_gaussian_kernel() -> torch.Tensor:
        """创建 5x5 高斯核，用于多尺度结构相似度计算。"""
        kernel = torch.tensor([
            [1., 4., 6., 4., 1.],
            [4., 16., 24., 16., 4.],
            [6., 24., 36., 24., 6.],
            [4., 16., 24., 16., 4.],
            [1., 4., 6., 4., 1.],
        ], dtype=torch.float32) / 256.0
        return kernel.unsqueeze(0).unsqueeze(0)

    def _ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """简化的结构相似度计算。

        SSIM(x, y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)
                    / (mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)

        这里使用局部均值和方差做近似。
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.functional.conv2d(pred, self.gaussian_kernel, padding=2)
        mu_y = nn.functional.conv2d(target, self.gaussian_kernel, padding=2)
        sigma_x = nn.functional.conv2d(pred ** 2, self.gaussian_kernel, padding=2) - mu_x ** 2
        sigma_y = nn.functional.conv2d(target ** 2, self.gaussian_kernel, padding=2) - mu_y ** 2
        sigma_xy = nn.functional.conv2d(pred * target, self.gaussian_kernel, padding=2) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))
        return ssim_map.mean()

    def forward(self, warp_image: torch.Tensor, reference_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            warp_image: 从源视角"缝合"过来的图像
            reference_image: 目标视角的真实图像

        Returns:
            综合损失 = gamma * SSIM_loss + (1-gamma) * L1_loss
        """
        ssim_loss = 1.0 - self._ssim(warp_image, reference_image)
        photometric_loss = torch.abs(warp_image - reference_image).mean()
        return self.gamma * ssim_loss + (1.0 - self.gamma) * photometric_loss


# ============================================================================
# 第 7 部分：主程序 —— 完整流水线演示
# ============================================================================

def main():
    """运行完整的单目深度估计教学演示。"""
    torch.manual_seed(0)

    print("=" * 60)
    print("第 26 课：单目深度估计 — 完整演示")
    print("=" * 60)

    # ---- 1. 合成数据 ----
    print("\n[1] 生成合成深度场景")
    gt_np = synthetic_depth(size=64)
    gt = torch.from_numpy(gt_np)

    # 模拟一个带尺度和偏移偏差的预测（相对深度模型常见情况）
    pred_noise = 0.3 * torch.randn_like(gt)
    pred = gt + pred_noise
    scaled_pred = 2.5 * pred + 0.8  # 不正确的尺度和偏移

    print(f"  深度范围: [{gt.min():.2f}, {gt.max():.2f}] 米")
    print(f"  预测范围: [{scaled_pred.min():.2f}, {scaled_pred.max():.2f}]")

    # ---- 2. 评估指标对比 ----
    print("\n[2] 评估指标对比（未对齐 vs 对齐）")
    print("-" * 50)

    # 直接使用预测（错误的尺度和偏移）
    absrel_raw = abs_rel_error(scaled_pred, gt)
    delta_raw = delta_accuracy(scaled_pred, gt)
    rmse_log_raw = rmse_log(scaled_pred, gt)

    print(f"  原始预测  |  AbsRel = {absrel_raw:.4f}  "
          f"Delta<1.25 = {delta_raw:.4f}  RMSE_log = {rmse_log_raw:.4f}")

    # 对齐后
    aligned = align_scale_shift(scaled_pred, gt)
    absrel_aligned = abs_rel_error(aligned, gt)
    delta_aligned = delta_accuracy(aligned, gt)
    rmse_log_aligned = rmse_log(aligned, gt)

    print(f"  对齐后    |  AbsRel = {absrel_aligned:.4f}  "
          f"Delta<1.25 = {delta_aligned:.4f}  RMSE_log = {rmse_log_aligned:.4f}")

    improvement = (absrel_raw - absrel_aligned) / absrel_raw * 100
    print(f"\n  对齐带来的 AbsRel 提升: {improvement:.1f}%")

    # 验证对齐是否正确恢复了尺度
    a_est, b_est = align_scale_shift(scaled_pred, gt).unbind() if False else (None, None)
    # 最小二乘恢复
    A_mat = torch.stack([scaled_pred.flatten(), torch.ones_like(scaled_pred.flatten())], dim=1)
    coeffs, *_ = torch.linalg.lstsq(A_mat, gt.flatten().unsqueeze(-1))
    a_recov, b_recov = coeffs.squeeze()
    print(f"  恢复的尺度因子 a = {a_recov:.2f}（真实值 1/2.5 = 0.40）")
    print(f"  恢复的偏移量 b = {b_recov:.2f}（真实值 -0.8/2.5 = -0.32）")

    # ---- 3. 深度转点云 ----
    print("\n[3] 深度图 → 3D 点云")
    cam_matrix = np.array([[64.0, 0.0, 32.0],
                           [0.0, 64.0, 32.0],
                           [0.0, 0.0, 1.0]])
    pc = point_cloud_from_depth_with_intrinsics(gt_np, cam_matrix, (64, 64))
    print(f"  点云数量: {pc.shape[0]} 个顶点")
    print(f"  X 范围: [{pc[:, 0].min():.2f}, {pc[:, 0].max():.2f}]")
    print(f"  Y 范围: [{pc[:, 1].min():.2f}, {pc[:, 1].max():.2f}]")
    print(f"  Z 范围: [{pc[:, 2].min():.2f}, {pc[:, 2].max():.2f}]")

    # ---- 4. 测试深度解码器 ----
    print("\n[4] 测试简易深度解码器")
    batch_size = 2
    feat_h, feat_w = 4, 4  # 假设输入特征下采样了 16 倍（64/4 = 16）
    dummy_features = torch.randn(batch_size, 768, feat_h, feat_w)

    decoder = DepthDecoder(in_channels=768, hidden_dim=128)
    depth_pred = decoder(dummy_features)
    print(f"  输入特征形状: {dummy_features.shape}")
    print(f"  输出深度图形状: {depth_pred.shape}")
    print(f"  输出深度范围: [{depth_pred.min():.3f}, {depth_pred.max():.3f}]")
    print(f"  解码器参数量: {sum(p.numel() for p in decoder.parameters()):,}")

    # ---- 5. 测试无监督光度损失 ----
    print("\n[5] 测试无监督光度一致性损失")
    # 模拟两张图像和一个"缝合"结果
    img_size = (1, 3, 32, 32)  # 批量大小=1，RGB，32x32
    reference_img = torch.rand(*img_size)  # 真实的目标视角图像
    warp_img = reference_img + 0.05 * torch.randn_like(reference_img)  # 轻微形变

    photometric_loss_fn = PhotometricLoss(gamma=0.85)
    loss_val = photometric_loss_fn(warp_img, reference_img)
    print(f"  光度一致性损失: {loss_val:.4f}")

    # 加入更多噪声时损失增大
    noisy_warp = reference_img + 0.3 * torch.randn_like(reference_img)
    noisy_loss = photometric_loss_fn(noisy_warp, reference_img)
    print(f"  加噪后损失:   {noisy_loss:.4f}")
    print(f"  损失增大倍数: {noisy_loss / loss_val:.1f}x（符合预期：形变越大损失越高）")

    # ---- 6. 完整评估流程 ----
    print("\n[6] 完整评估流程演示")
    print("-" * 50)

    # 生成更大尺度的合成数据
    gt_large = synthetic_depth(size=256)
    # 模拟真实的预测：有正确的尺度但存在系统性偏差
    pred_large = gt_large * 0.95 + 0.1 * np.random.randn(*gt_large.shape).astype(np.float32)

    gt_t = torch.from_numpy(gt_large)
    pred_t = torch.from_numpy(pred_large)

    # 未对齐
    eval_unaligned = {
        "absRel": abs_rel_error(pred_t, gt_t),
        "delta<1.25": delta_accuracy(pred_t, gt_t),
        "delta<1.25^2": delta_accuracy(pred_t, gt_t, threshold=1.25**2),
        "delta<1.25^3": delta_accuracy(pred_t, gt_t, threshold=1.25**3),
        "RMSE_log": rmse_log(pred_t, gt_t),
    }

    # 对齐后
    aligned_large = align_scale_shift(pred_t, gt_t)
    eval_aligned = {
        "absRel": abs_rel_error(aligned_large, gt_t),
        "delta<1.25": delta_accuracy(aligned_large, gt_t),
        "delta<1.25^2": delta_accuracy(aligned_large, gt_t, threshold=1.25**2),
        "delta<1.25^3": delta_accuracy(aligned_large, gt_t, threshold=1.25**3),
        "RMSE_log": rmse_log(aligned_large, gt_t),
    }

    print("  评估指标对比（未对齐 → 对齐）：")
    print(f"  {'指标':<12} {'未对齐':>10} {'对齐后':>10} {'变化':>10}")
    print(f"  {'-' * 44}")
    for key in ["absRel", "RMSE_log"]:
        before = eval_unaligned[key]
        after = eval_aligned[key]
        change = (before - after) / before * 100
        print(f"  {key:<12} {before:>10.4f} {after:>10.4f} {change:>+9.1f}%")

    for key in ["delta<1.25", "delta<1.25^2", "delta<1.25^3"]:
        before = eval_unaligned[key]
        after = eval_aligned[key]
        change = (after - before) * 100
        print(f"  {key:<12} {before:>10.4f} {after:>10.4f} {change:>+9.1f}pp")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
