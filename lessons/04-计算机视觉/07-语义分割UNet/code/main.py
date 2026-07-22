# main.py — 语义分割 UNet：从零实现训练流水线
# 依赖：torch>=2.0, torchvision
# 安装：pip install torch torchvision
# 对应课程：阶段 04 · 07（语义分割 UNet）

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# === DoubleConv：双卷积模块 ===
class DoubleConv(nn.Module):
    """两个连续的 3x3 卷积，每个卷积后接批归一化和 ReLU。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# === Down：下采样模块 ===
class Down(nn.Module):
    """最大池化 + 双卷积模块，用于编码器。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# === Up：上采样 + 跳跃连接模块 ===
class Up(nn.Module):
    """双线性上采样后与编码器特征拼接，再经过双卷积。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 使用上采样而非转置卷积——避免棋盘效应（checkerboard artifacts）
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_encoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # 上采样解码器特征
        x = self.up(x_encoder)
        # 如果尺寸不完全匹配，做插值对齐（处理奇数尺寸等边缘情况）
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        # 拼接：编码器的高分辨率特征 + 解码器的低分辨率特征
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# === U-Net 架构 ===
class UNet(nn.Module):
    """
    U-Net 编码器-解码器架构。

    架构示意：
        ┌─────────────────────────────────────────────────┐
        │                  输入图像                         │
        └──────────────────────┬──────────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────────┐
        │  编码器（收缩路径 / 收缩部分）                     │
        │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐               │
        │  │ Conv│→│Pool│→│ Conv│→│Pool│→│ ...         │
        │  └─────┘ └─────┘ └─────┘ └─────┘               │
        │              瓶颈层（最低分辨率特征）             │
        └──────────────────────┬──────────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────────┐
        │  解码器（扩展路径 / 扩展部分）                     │
        │        ↑ skip      ↑ skip    ↑ skip   ↑ skip     │
        │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
        │  │Conv │←│Up   │←│Conv │←│Up   │←│ ...  │       │
        │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │
        └──────────────────────┬──────────────────────────┘
                               ▼
                          ┌─────────┐
                          │  1×1 conv│  → 像素级预测
                          └─────────┘

    跳跃连接是关键设计：编码器保留空间细节但缺乏语义，解码器拥有语义但丢失了空间信息，
    拼接让两者互补。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
    ):
        super().__init__()
        # 编码器
        self.inc = DoubleConv(in_channels, base_channels)
        self.d1 = Down(base_channels, base_channels * 2)
        self.d2 = Down(base_channels * 2, base_channels * 4)
        self.d3 = Down(base_channels * 4, base_channels * 8)
        self.d4 = Down(base_channels * 8, base_channels * 16)
        # 瓶颈之后的解码器
        self.u1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.u2 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.u3 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.u4 = Up(base_channels * 2 + base_channels, base_channels)
        # 输出层：1×1 卷积将通道数映射到类别数
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码路径（保存跳跃连接的中间输出）
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        # 解码路径（拼接编码器特征）
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)


# === Dice Loss：重叠率损失 ===
def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Dice 损失：衡量预测概率分布与真实掩码之间的重叠程度。
    公式：Dice = 2 * |A ∩ B| / (|A| + |B|)

    交叉熵损失基于像素级独立判断，Dice 损失基于全局区域评估。
    对于医学影像等极度不平衡的场景，Dice 直接优化目标指标（重叠率），
    比像素精度更有意义。
    """
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)
    total = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (total + eps)
    return 1.0 - dice.mean()


# === 组合损失：交叉熵 + Dice ===
def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    lam: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    交叉熵提供像素级分类信号，Dice 提供区域重叠信号，
    二者结合兼顾局部精度和整体结构。
    """
    ce = F.cross_entropy(logits, targets)
    dc = dice_loss(logits, targets, num_classes)
    loss = ce + lam * dc
    return loss, {"ce": ce.detach().item(), "dice": dc.detach().item()}


# === 逐类 IoU 计算 ===
@torch.no_grad()
def iou_per_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """计算每个类别的交并比（IoU）指标。"""
    preds = logits.argmax(dim=1)
    ious = torch.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (targets == c)
        inter = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        ious[c] = (inter / union) if float(union) > 0 else float("nan")
    return ious


# === 合成数据集：几何图形生成器 ===
def synthetic_segmentation(
    num_samples: int = 120,
    size: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成合成分割数据集：在绿色背景上随机放置红色圆形或蓝色正方形。
    每个像素属于三类之一：背景（0）、圆形（1）、正方形（2）。

    合成数据的价值：不需要标注成本即可验证完整的训练-评估流水线，
    让读者先理解"模型学到了什么"，再迁移到真实数据。
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, size, size, 3), dtype=np.float32)
    masks = np.zeros((num_samples, size, size), dtype=np.int64)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    circle_color = np.array([0.9, 0.1, 0.1], dtype=np.float32)
    square_color = np.array([0.1, 0.2, 0.9], dtype=np.float32)
    for i in range(num_samples):
        bg = np.array([0.3, 0.7, 0.3], dtype=np.float32)
        images[i] = bg
        cls = int(rng.integers(1, 3))
        cx, cy = int(rng.integers(14, size - 14)), int(rng.integers(14, size - 14))
        r = int(rng.integers(8, 14))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        if cls == 1:
            # 圆形
            images[i][mask] = circle_color
            masks[i][mask] = 1
        else:
            # 正方形
            mask_sq = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)
            images[i][mask_sq] = square_color
            masks[i][mask_sq] = 2
        images[i] += rng.normal(0, 0.02, images[i].shape)
        images[i] = np.clip(images[i], 0, 1)
    return images, masks


# === 数据集类 ===
class SegDataset(Dataset):
    """从 NumPy 数组构建 PyTorch 数据集。"""

    def __init__(self, images: np.ndarray, masks: np.ndarray):
        self.images = images
        self.masks = masks

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float()
        mask = torch.from_numpy(self.masks[idx]).long()
        return img, mask


# === 主训练循环 ===
def main():
    torch.manual_seed(0)
    images, masks = synthetic_segmentation(num_samples=60, size=64)
    split = int(0.85 * len(images))
    train_ds = SegDataset(images[:split], masks[:split])
    val_ds = SegDataset(images[split:], masks[split:])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 3
    model = UNet(in_channels=3, num_classes=num_classes, base_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"设备: {device}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(8):
        # 训练轮次
        model.train()
        loss_sum, total = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss, _ = combined_loss(logits, y, num_classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item() * x.size(0)
            total += x.size(0)

        # 验证轮次
        model.eval()
        iou_sum = torch.zeros(num_classes)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                iou_sum += iou_per_class(model(x), y, num_classes).nan_to_num(0)
        iou_mean = (iou_sum / len(val_loader)).tolist()
        print(
            f"epoch {epoch}  train_loss {loss_sum / total:.3f}  "
            f"iou {[f'{v:.2f}' for v in iou_mean]}"
        )


if __name__ == "__main__":
    main()
