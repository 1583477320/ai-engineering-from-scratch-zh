# Inpainting 掩码工具集
# 创建掩码、准备输入、羽化处理

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter


# ============================================================================
# 第 1 步：掩码创建
# ============================================================================

def create_center_mask(image_size=(512, 512), mask_ratio=0.3):
    """
    创建中心区域掩码——最常用的 inpainting 测试掩码。
    Args:
        image_size: 图像尺寸 (H, W)
        mask_ratio: 掩码区域占图像的比例
    Returns:
        mask: 二进制掩码 (H, W)，1=需要修复，0=保留
    """
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.uint8)

    h_start = int(H * (1 - mask_ratio) / 2)
    h_end = int(H * (1 + mask_ratio) / 2)
    w_start = int(W * (1 - mask_ratio) / 2)
    w_end = int(W * (1 + mask_ratio) / 2)

    mask[h_start:h_end, w_start:w_end] = 1
    return mask


def create_random_masks(image_size=(512, 512), num_masks=None, max_ratio=0.2):
    """
    创建随机位置的多个圆形掩码。
    适合测试 inpainting 的多区域修复能力。
    """
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.uint8)

    if num_masks is None:
        num_masks = np.random.randint(1, 4)

    for _ in range(num_masks):
        cy = np.random.randint(0, H)
        cx = np.random.randint(0, W)
        radius = int(min(H, W) * max_ratio * np.random.uniform(0.5, 1.0))

        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
        mask[dist < radius] = 1

    return mask


def create_manual_mask_from_points(image_size, points, radius=30):
    """
    根据用户点击的点创建圆形掩码。
    适合交互式 inpainting 工具。
    Args:
        image_size: (H, W)
        points: [(y1, x1), (y2, x2), ...] 点坐标列表
        radius: 每个点对应的掩码半径
    """
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.uint8)

    for y, x in points:
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((Y - y)**2 + (X - x)**2)
        mask[dist < radius] = 1

    return mask


# ============================================================================
# 第 2 步：掩码羽化（边缘渐变过渡）
# ============================================================================

def feather_mask(mask, feather_radius=10):
    """
    对掩码进行羽化处理——边缘渐变过渡，避免修复区域与原图的硬边界。
    Args:
        mask: 二进制掩码 (H, W) 或 (1, H, W)
        feather_radius: 羽化半径（像素）
    Returns:
        feathered_mask: 羽化后的掩码，值在 [0, 1] 之间
    """
    if mask.ndim == 3:
        mask = mask[0]

    # 使用高斯模糊进行羽化
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    feathered = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    # 转回 numpy，值域 [0, 1]
    feathered_np = np.array(feathered).astype(np.float32) / 255.0
    return feathered_np


def create_feathered_center_mask(image_size=(512, 512), mask_ratio=0.3,
                                 feather_radius=15):
    """创建带羽化的中心掩码——最常用的高质量修复掩码。"""
    mask = create_center_mask(image_size, mask_ratio)
    return feather_mask(mask, feather_radius)


# ============================================================================
# 第 3 步：Inpainting 输入准备
# ============================================================================

def prepare_inpainting_input(image_tensor, mask, noise_level=1.0):
    """
    准备 inpainting 模型输入。
    Args:
        image_tensor: 原始图像 (B, C, H, W)，值域 [-1, 1]
        mask: 掩码 (H, W) 或 (1, H, W)，1=修复，0=保留
        noise_level: 掩码区域添加的噪声强度（0-1）
    Returns:
        model_input: 模型输入 (B, C+1, H, W) = [带噪图像, 掩码]
    """
    B, C, H, W = image_tensor.shape

    # 处理掩码维度
    if mask.ndim == 2:
        mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = torch.from_numpy(mask).float().unsqueeze(1)

    mask = mask.to(image_tensor.device)
    mask_3ch = mask.expand(B, C, H, W)  # 扩展到与图像相同的通道数

    # 在掩码区域添加噪声
    noise = torch.randn_like(image_tensor)
    noisy_image = (1 - mask_3ch) * image_tensor + mask_3ch * noise * noise_level

    # 模型输入：[带噪图像, 掩码通道]
    model_input = torch.cat([noisy_image, mask], dim=1)  # (B, C+1, H, W)
    return model_input


def prepare_outpainting_canvas(image_tensor, target_size, placement="center"):
    """
    准备 outpainting 画布。
    Args:
        image_tensor: 原始图像 (B, C, H, W)
        target_size: 目标尺寸 (target_H, target_W)
        placement: 原图放置位置 "center" / "top-left" / "bottom-right"
    Returns:
        canvas: 画布 (B, C, target_H, target_W)
        mask: 掩码 (B, 1, target_H, target_W)，1=生成区域
    """
    B, C, H, W = image_tensor.shape
    target_H, target_W = target_size

    # 创建空白画布（噪声初始化）
    canvas = torch.randn(B, C, target_H, target_W, device=image_tensor.device)
    mask = torch.ones(B, 1, target_H, target_W, device=image_tensor.device)

    # 放置原始图像
    if placement == "center":
        offset_h = (target_H - H) // 2
        offset_w = (target_W - W) // 2
    elif placement == "top-left":
        offset_h, offset_w = 0, 0
    else:  # bottom-right
        offset_h = target_H - H
        offset_w = target_W - W

    canvas[:, :, offset_h:offset_h+H, offset_w:offset_w+W] = image_tensor
    mask[:, :, offset_h:offset_h+H, offset_w:offset_w+W] = 0

    return canvas, mask


# ============================================================================
# 第 4 步：可视化工具
# ============================================================================

def visualize_mask_comparison(image, mask, output_path="mask_comparison.png"):
    """
    可视化原始图像、掩码、叠加效果。
    Args:
        image: 原始图像 (H, W, C)，值域 [0, 255]
        mask: 掩码 (H, W)，值域 [0, 1]
        output_path: 输出路径
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("原始图像")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("掩码 (白色=修复区域)")
    axes[1].axis("off")

    # 叠加显示
    overlay = image.copy()
    mask_3ch = np.stack([mask] * 3, axis=-1)
    overlay = np.where(mask_3ch > 0.5, [255, 0, 0], overlay)  # 红色标记修复区域
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("掩码叠加 (红色=修复区域)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"可视化结果已保存到: {output_path}")


# ============================================================================
# 主程序：演示
# ============================================================================

if __name__ == "__main__":
    # 创建测试图像
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # 1. 创建不同类型的掩码
    print("创建掩码...")
    center_mask = create_center_mask((512, 512), mask_ratio=0.3)
    random_mask = create_random_masks((512, 512))
    feathered = create_feathered_center_mask((512, 512), mask_ratio=0.3)

    print(f"  中心掩码: {center_mask.shape}, 修复区域比例: {center_mask.mean():.2%}")
    print(f"  随机掩码: {random_mask.shape}, 修复区域比例: {random_mask.mean():.2%}")
    print(f"  羽化掩码: {feathered.shape}, 值域: [{feathered.min():.2f}, {feathered.max():.2f}]")

    # 2. 准备 inpainting 输入
    image_tensor = torch.randn(1, 3, 512, 512)  # 模拟归一化图像
    model_input = prepare_inpainting_input(image_tensor, center_mask)
    print(f"\n模型输入形状: {model_input.shape}")  # (1, 4, 512, 512)

    # 3. 准备 outpainting 画布
    canvas, mask = prepare_outpainting_canvas(image_tensor, (1024, 1024))
    print(f"画布形状: {canvas.shape}")
    print(f"掩码形状: {mask.shape}")
    print(f"原始图像区域比例: {(1 - mask).mean():.2%}")

    print("\n完成！")
