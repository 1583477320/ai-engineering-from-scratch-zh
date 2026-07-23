# 单目深度估计：从一张图推断三维世界

> 两只眼睛看世界能感知深度，单只眼睛也能——关键在于模型学会了"空间直觉"。

**类型：** 实现课
**语言：** Python
**前置知识：** 第 03 阶段（深度学习核心）、第 14 课（Vision Transformers）
**预计时间：** ~90 分钟
**所处阶段：** Tier 1
**关联课程：** 第 13 课（3D 视觉 NeRF）— 深度图是三维重建的关键输入；第 07 课（语义分割 UNet）— 编码器-解码器架构与深度估计网络共享设计模式

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释单目深度估计的核心挑战——为什么从二维图像恢复三维深度是一个病态问题
- [ ] 实现 KITTI 数据集上的三项标准评估指标：AbsRel、RMSE_log 和 delta 准确率
- [ ] 理解监督方法（Eigen 等人）与无监督方法（Monodepth2）的训练策略差异
- [ ] 构建一个完整的深度解码器模块，理解从 ViT 特征到全分辨率深度图的上采样过程
- [ ] 使用尺度-偏移对齐将相对深度预测转换为绝对深度，完成评估流程

## 1. 问题

你在自动驾驶汽车上安装了一台摄像头。车辆需要知道前方 5 米有一辆卡车、30 米处有行人。但摄像头拍下的只是一张二维图片——深度信息在投影过程中被永久丢失了。

双目立体视觉可以用视差（disparity）来恢复深度。激光雷达（LiDAR）可以精确测量。但它们都意味着额外的硬件成本。一台 LiDAR 要数万美元，双目标定和同步也增加了系统复杂度。

你需要的是：**只用一台普通摄像头，从单张图像中推断每个像素到相机的距离。**

这就是单目深度估计（Monocular Depth Estimation）要解决的问题。如果可行，它意味着任何一台摄像头——手机、监控、无人机——都能变成深度传感器。这个代价从数万美元降到了零元。

但这件事的难度远超直觉。两张二维照片之间丢失的深度信息，理论上是不可恢复的——同一条深度线上的无穷多个三维点，投影到图像上都变成同一个像素。模型必须从场景的统计规律（近大远小、遮挡关系、纹理梯度、地平线）中"学会"空间直觉。

好消息是：2025 年的模型已经做得很好。Depth Anything 等基础模型在 KITTI 基准测试上达到了接近激光雷达的精度。但理解它们的工作原理——从 Eigen 2014 年的奠基工作到 Monodepth2 的无监督训练——仍然需要你掌握几个关键概念。

## 2. 概念

### 2.1 直观理解

单目深度估计本质上是一个逐像素回归任务：给定一张 RGB 图像，输出一张同样大小的深度图，每个像素的值代表该点到相机的距离。

```
输入: RGB 图像 (H, W, 3)        输出: 深度图 (H, W)
┌──────────────────────┐        ┌──────────────────────┐
│  🚗   🚶   🏠        │        │  5m   30m   50m       │
│                      │  ──→   │  ▓▓   ░░   ░░░       │
│  ████  地面  ████    │        │  ██   ▒▒   ▒▒▒       │
└──────────────────────┘        └──────────────────────┘
                                  深 = 近 (5m)  浅 = 远 (50m)
```

有两种根本不同的训练范式：

**监督学习**：用 LiDAR 采集的真实深度图（ground truth）作为标签，直接训练模型预测深度值。Eigen 等人（2014）首次证明了这个思路可行。

**无监督学习**：不需要 LiDAR 标签。只需要一对同一场景、不同视角的图像（如双目相机的左右图或视频连续帧）。模型预测深度和相机运动，然后将一张图"缝合"到另一张图的位置。缝合结果与真实目标图应该一致——这个一致性就是监督信号。

### 2.2 为什么深度估计是病态问题

从三维到二维的投影过程损失了信息：

$$
u = f_x \cdot \frac{X}{Z} + c_x, \quad v = f_y \cdot \frac{Y}{Z} + c_y
$$

其中 $(u, v)$ 是像素坐标，$(X, Y, Z)$ 是三维空间坐标，$(f_x, f_y)$ 是焦距，$(c_x, c_y)$ 是主点。给定一个像素 $(u, v)$，有无穷多个 $(X, Y, Z)$ 组合满足这个方程——只要 $X/Z$ 和 $Y/Z$ 的比值固定就行。

因此，模型必须依赖**先验知识**（视觉线索）来消除歧义：

| 视觉线索 | 工作原理 | 局限性 |
|---|---|---|
| 近大远小 | 同一物体在图像中越小，距离越远 | 未知物体的真实大小时失效 |
| 遮挡 | 被遮挡的物体更远 | 无遮挡时无信号 |
| 纹理梯度 | 地面纹理越密，距离越远 | 纹理均匀区域失效 |
| 地平线位置 | 靠近地平线的物体更远 | 室内场景不适用 |
| 暗示 | 天空最远，地面从近到远 | 过度依赖语义理解 |

### 2.3 监督方法：Eigen 的多尺度架构

Eigen 等人（2014）[1] 提出了深度估计的经典基线。核心设计：

1. **粗尺度网络**：输入整张图像，预测低分辨率的全局深度结构
2. **细尺度网络**：以粗尺度预测为条件，预测高分辨率的局部细节
3. **多尺度损失**：在两个尺度上分别计算损失，兼顾全局一致性与局部精度

$$
\mathcal{L}_{\text{total}} = \frac{1}{n}\sum_{p}\left[\frac{1}{2}\log d_p - \frac{1}{2}\log \hat{d}_p\right]^2 + \lambda \cdot \frac{1}{n}\sum_{p}\left[\frac{1}{2}\log d_p - \frac{1}{2}\log \hat{d}_p\right]^2
$$

其中 $d_p$ 是真实深度，$\hat{d}_p$ 是预测深度，两个求和分别在粗、细两个尺度上进行。取对数是为了让模型对近距离误差更敏感（避免远距离主导损失）。

### 2.4 无监督方法：光度一致性

Monodepth2（Godard 等人，2019）[2] 证明了无需 LiDAR 也能训练高质量的深度模型。核心思想基于几何一致性：

**左图和右图来自同一场景的不同位置。** 如果我知道左图的深度 $D_L$ 和相机运动（位姿 $T_{L \to R}$），我就能将左图"变形"（warp）到右图的视角：

$$
I_R(u, v) \approx I_L\left(\text{warp}(u, v; D_L, T_{L \to R})\right)
$$

缝合图像与真实右图之间的差异就是损失信号：

$$
\mathcal{L}_{\text{photo}} = \frac{1}{N}\sum_{p}\rho\left(I_L(p), I_R(p)\right)
$$

其中 $\rho$ 是光度一致性损失函数，通常结合 L1 距离和结构相似度（SSIM）：

$$
\rho(a, b) = \gamma \cdot (1 - \text{SSIM}(a, b)) + (1 - \gamma) \cdot \|a - b\|_1
$$

$\gamma$ 通常取 0.85。SSIM 捕捉结构信息，L1 捕捉像素级差异——两者结合能避免在纹理均匀区域产生伪影。

**多尺度策略：** Monodepth2 使用三个解码尺度，在每个尺度上独立计算光度损失，并取最小损失（auto-masking）来处理遮挡和动态物体。

```
左图 ──→ [深度网络] ──→ 深度图 D_L
                            ↓
左图 + D_L + 位姿 ──→ [Warp] ──→ 缝合图像
                                       ↓
                               [光度损失] ← 真实右图
```

### 2.5 深度评估指标

KITTI 数据集是深度估计的标准基准。以下是必须掌握的三项核心指标：

**绝对相对误差（AbsRel）：**

$$
\text{AbsRel} = \frac{1}{|V|}\sum_{p \in V}\frac{|d_p - \hat{d}_p|}{d_p}
$$

越低越好。衡量的是预测值与真实值之间的相对偏差。生产级模型的典型值在 0.05 ~ 0.10 之间。

**delta 准确率：**

$$
\delta < t = \frac{1}{|V|}\sum_{p \in V}\mathbb{1}\left[\max\left(\frac{d_p}{\hat{d}_p}, \frac{\hat{d}_p}{d_p}\right) < t\right]
$$

越高越好。统计满足相对误差小于阈值 $t$ 的像素比例。常用阈值为 1.25、$1.25^2$、$1.25^3$。SOTA 模型在 KITTI 上的 $\delta < 1.25$ 通常超过 0.95。

**尺度-偏移对齐（Scale-Shift Alignment）：**

对于 MiDaS、Depth Anything 等相对深度模型，预测值没有绝对尺度意义。评估前必须用最小二乘法对齐：

$$
\hat{d}_{\text{aligned}} = a \cdot \hat{d} + b, \quad \text{where} \quad (a, b) = \arg\min_{a,b}\sum_p(a \cdot \hat{d}_p + b - d_p)^2
$$

不对齐直接算 AbsRel 会完全测不出模型的排序质量。

## 3. 从零实现

### 第 1 步：深度评估指标

```python
# main.py — 单目深度估计：从零实现核心组件
# 依赖：torch>=2.0, numpy
# 对应课程：第 04 阶段 · 第 26 课（单目深度估计）

import numpy as np
import torch


def abs_rel_error(pred, target, mask=None):
    """计算绝对相对误差（AbsRel）。

    AbsRel = mean(|d_pred - d_gt| / d_gt)，越低越好。
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return (torch.abs(pred - target) / target.clamp(min=1e-6)).mean().item()
```

为什么用 `clamp(min=1e-6)` 而不是直接除？因为深度为零的像素（通常来自 LiDAR 无效区域）会导致除零错误。`clamp` 将最小值限制为 1e-6，避免了这个问题。

### 第 2 步：delta 准确率

```python
def delta_accuracy(pred, target, threshold=1.25, mask=None):
    """计算 delta 准确率。

    统计 max(d_pred/d_gt, d_gt/d_pred) < threshold 的像素比例。
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    ratio = torch.maximum(
        pred / target.clamp(min=1e-6),
        target / pred.clamp(min=1e-6),
    )
    return (ratio < threshold).float().mean().item()
```

注意取 `maximum` 而不是绝对值：如果预测 10m 而真实 8m，ratio 是 1.25；如果预测 8m 而真实 10m，ratio 是 1.25。两者对称，说明 delta 对"预测偏高"和"预测偏低"一视同仁。

### 第 3 步：尺度-偏移对齐

相对深度模型的预测只有相对顺序有意义。用最小二乘法恢复真实尺度：

```python
def align_scale_shift(pred, target, mask=None):
    """将相对深度预测对齐到真实深度。

    拟合 a * pred + b = target，使预测值获得正确的尺度和偏移。
    """
    if mask is not None:
        p = pred[mask]
        t = target[mask]
    else:
        p = pred.flatten()
        t = target.flatten()

    A = torch.stack([p, torch.ones_like(p)], dim=1)
    sol = torch.linalg.lstsq(A, t.unsqueeze(-1))
    a, b = sol.solution[:2, 0]
    return a * pred + b
```

### 第 4 步：深度解码器（DPT 风格简化版）

现代深度估计模型（如 Depth Anything）使用 ViT 编码器提取特征，再用卷积解码器恢复全分辨率深度图。这里展示解码器部分：

```python
import torch.nn as nn


class DepthDecoder(nn.Module):
    """简易深度解码器（DPT 风格的简化版）。

    模拟 Depth Anything 架构：冻结的 ViT 编码器 + 可训练的卷积解码器。
    输入是 ViT 输出的密集特征（下采样 14 倍），输出是全分辨率深度图。
    """

    def __init__(self, in_channels=768, hidden_dim=256, out_channels=1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim // 8, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        x = self.projector(features)
        return self.decoder(x)
```

### 第 5 步：无监督光度损失

```python
class PhotometricLoss(nn.Module):
    """光度一致性损失（无监督深度估计的核心）。

    综合损失 = gamma * SSIM_loss + (1-gamma) * L1_loss
    SSIM 捕捉结构信息，避免在均匀区域产生伪影。
    """

    def __init__(self, gamma=0.85):
        super().__init__()
        self.gamma = gamma

    def _ssim(self, pred, target):
        """简化版结构相似度计算。"""
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        kernel = self._create_gaussian_kernel().to(pred.device)
        mu_x = nn.functional.conv2d(pred, kernel, padding=2)
        mu_y = nn.functional.conv2d(target, kernel, padding=2)
        sigma_x = nn.functional.conv2d(pred**2, kernel, padding=2) - mu_x**2
        sigma_y = nn.functional.conv2d(target**2, kernel, padding=2) - mu_y**2
        sigma_xy = nn.functional.conv2d(pred * target, kernel, padding=2) - mu_x * mu_y
        return ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) /
                ((mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2))).mean()

    def forward(self, warp_image, reference_image):
        ssim_loss = 1.0 - self._ssim(warp_image, reference_image)
        l1_loss = torch.abs(warp_image - reference_image).mean()
        return self.gamma * ssim_loss + (1 - self.gamma) * l1_loss
```

### 第 6 步：深度转点云

有了深度图和相机内参，可以将二维图像恢复为三维点云：

```python
def depth_to_point_cloud(depth, intrinsics):
    """将深度图提升为 3D 点云。

    小孔相机反投影公式：
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
    """
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics

    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth.astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1)
```

运行完整流水线可查看所有模块的协同效果：

```python
if __name__ == "__main__":
    torch.manual_seed(0)
    gt = torch.from_numpy(synthetic_depth(size=64))
    pred = 2.5 * (gt + 0.3 * torch.randn_like(gt)) + 0.8

    print(f"AbsRel (对齐前): {abs_rel_error(pred, gt):.4f}")
    aligned = align_scale_shift(pred, gt)
    print(f"AbsRel (对齐后): {abs_rel_error(aligned, gt):.4f}")
    print(f"Delta<1.25 (对齐后): {delta_accuracy(aligned, gt):.4f}")
```

## 4. 工业工具

### 4.1 Depth Anything V2 推理

Depth Anything 是 2024 年发布的通用深度估计基础模型，在 6200 万张图像上无监督训练，然后在 KITTI 等数据集上微调。

```python
from transformers import pipeline

# 使用 HuggingFace Pipeline 加载 Depth Anything V2
depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Large-hf",
    device="cuda",
)

# 推理
result = depth_pipe(image)
depth_map = result["depth"]  # PIL Image 格式的深度图
```

### 4.2 MiDaS 推理

MiDaS 是 Intel 开发的多尺度深度估计模型，擅长处理室内和室外场景。

```python
import torch

# 使用 PyTorch Hub 加载 MiDaS
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# 输入预处理
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
input_batch = transform(image).to("cuda")

with torch.no_grad():
    prediction = midas(input_batch)
    depth = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
```

### 4.3 Monodepth2 训练流程

```python
# Monodepth2 风格的无监督训练伪代码
models = {
    "encoder": Encoder().cuda(),
    "depth_decoder": DepthDecoder().cuda(),
    "pose_net": PoseNet().cuda(),
}

optimizer = torch.optim.Adam(
    list(models["encoder"].parameters()) +
    list(models["depth_decoder"].parameters()) +
    list(models["pose_net"].parameters()),
    lr=1e-4,
)

for batch in dataloader:
    left_img, right_img = batch["left"], batch["right"]

    # 预测深度
    features = models["encoder"](left_img)
    depth = models["depth_decoder"](features)

    # 预测位姿（从左图到右图）
    pose = models["pose_net"](torch.cat([left_img, right_img], dim=1))

    # 计算光度损失
    warpped = warp_image(right_img, depth, pose)
    loss = photometric_loss(warpped, left_img)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 4.4 工业级深度估计选型

| 场景 | 推荐模型 | 特点 |
|---|---|---|
| 通用场景 | Depth Anything V2 | 零样本泛化能力强，训练数据量最大 |
| 自动驾驶 | Monodepth2 / MonoDepthV2 | 在 KITTI 上有系统评测，可复现性好 |
| 室内导航 | ZoeDepth | 专门为室内场景优化，支持度量深度 |
| 边缘部署 | Depth Anything V2-Small | 参数量小，适合移动端推理 |
| 高精度场景 | Depth Anything V2 + KITTI 微调 | 通用预训练 + 监督微调，精度最高 |

## 5. 知识连线

本课学习的单目深度估计是计算机视觉中从二维到三维的关键桥梁：

- **第 13 课（3D 视觉 NeRF）**：NeRF 需要多视角图像和相机位姿来重建三维场景，而单目深度估计可以为每一帧提供粗略的深度先验，大幅提升重建质量。
- **第 07 课（语义分割 UNet）**：深度估计网络普遍采用编码器-解码器架构，与 UNet 的设计思想一脉相承。理解了深度估计中的多尺度融合，你就能更好地理解语义分割中的特征金字塔。
- **后续阶段 12（多模态 AI）**：现代 VLM（如 Qwen-VL）已经能够直接回答"这张图片中物体的距离有多远"这类问题，底层正是依赖深度估计能力。

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 通用深度估计 | Depth Anything V2 Large | 6200 万张图像预训练，零样本泛化强 |
| KITTI 基准评测 | Monodepth2 + ResNet-101 | 标准可复现基线 |
| 室内场景 | ZoeDepth | 专门处理度量深度，室内效果好 |
| 边缘部署 | Depth Anything V2-Small INT8 | 可运行在手机 NPU 上 |
| 高精度工业 | 监督微调 Depth Anything V2 | 在目标域标注少量数据微调 |

### 6.2 中文场景特别建议

- 中文文档中的深度估计术语应遵循术语表：深度图（depth map）、点云（point cloud）、内参矩阵（intrinsic matrix），不要混用英文。
- KITTI 数据集是英文标注的，但国内的 Apollo、nuScenes 数据集也提供深度标注，且更贴近国内路况场景。评测时优先考虑 nuScenes。
- 自动驾驶场景中的深度估计需要处理雨天、雾天、夜间等中文路况常见挑战。推荐在训练数据中加入中国道路场景的图像。

### 6.3 踩坑经验

- **尺度不一致**：不同传感器采集的深度图单位可能不同（米、毫米、归一化）。KITTI 用米，Cityscapes 用米但范围不同。混用会导致评估指标完全不可比。
- **LiDAR 稀疏性**：KITTI 的 LiDAR 点云只有约 5% 的像素有深度标注。评估时务必使用有效掩码（mask），否则未标注像素的深度为 0 会产生巨大误差。
- **相对深度 vs 绝对深度**：Depth Anything 默认输出相对深度，直接和 KITTI 的绝对深度对比会得到荒谬的结果。必须先做尺度-偏移对齐。
- **动态物体问题**：无监督方法假设静态场景。运动中的车辆和行人会违反光度一致性假设，导致深度预测不准。Monodepth2 的 auto-masking 策略可以缓解，但不能完全解决。
- **评估数据泄漏**：微调 Depth Anything 到 KITTI 时，确保训练集和测试集严格划分。KITTI 有官方的 train/val/test 分割，不要自行划分。

## 7. 常见错误

### 错误 1：直接对相对深度计算 AbsRel

**现象：** AbsRel 值远大于 1.0（如 5.0 或更高），而正常模型应该在 0.05 ~ 0.15 之间。

**原因：** Depth Anything、MiDaS 等模型输出的是相对深度（relative depth），预测值与真实值之间存在任意的尺度和偏移。直接比较绝对值毫无意义。

**修复：**

```python
# ❌ 错误：直接计算
absrel = abs_rel_error(depth_anything_output, gt_depth)

# ✓ 正确：先对齐再计算
aligned = align_scale_shift(depth_anything_output, gt_depth)
absrel = abs_rel_error(aligned, gt_depth)
```

### 错误 2：忽略 LiDAR 无效区域

**现象：** AbsRel 看起来很低（如 0.03），但深度图的边缘区域质量很差，实际效果与指标不符。

**原因：** KITTI 的 LiDAR 深度图约 95% 的像素是无效的（值为 0）。如果不使用掩码，评估会只计算少量有效像素的误差，掩盖了边缘区域的失败。

**修复：**

```python
# ❌ 错误：不使用掩码
absrel = abs_rel_error(pred, gt)

# ✓ 正确：只计算有效像素
mask = gt > 0  # LiDAR 有效区域
absrel = abs_rel_error(pred, gt, mask=mask)
```

### 错误 3：训练时未做数据增强

**现象：** 模型在训练集上 AbsRel 为 0.05，但测试集上飙升到 0.25。

**原因：** 自动驾驶场景的光照、天气、时间变化很大。如果训练数据单一（如只用白天、晴天），模型会过拟合到特定条件。

**修复：**

```python
# 在数据加载时加入标准增强
transform = Compose([
    RandomCrop(256, 512),
    RandomHorizontalFlip(0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 错误 4：无监督训练时的尺度漂移

**现象：** 训练初期深度预测值趋近于一个常数，loss 不下降。

**原因：** 无监督训练中，深度网络和位姿网络是联合优化的。如果深度预测趋近常数，位姿网络可以通过调整尺度来最小化损失，导致"零深度"的退化解。

**修复：** 添加深度正则化项，鼓励深度预测的多样性：

```python
depth_reg = torch.mean(1.0 / depth)  # 均匀深度惩罚
total_loss = photo_loss + 0.001 * depth_reg
```

## 8. 面试考点

### Q1：单目深度估计为什么是病态问题？从数学上解释。（难度：⭐⭐）

**参考答案：**

从针孔相机模型可知，一个三维点 $(X, Y, Z)$ 投影到像素 $(u, v)$ 的方程为 $u = f_x X/Z + c_x$，$v = f_y Y/Z + c_y$。给定一个像素 $(u, v)$，有三个未知量（$X, Y, Z$）但只有两个方程，系统是欠定的。深度 $Z$ 可以是任意正值——只要 $X/Z$ 和 $Y/Z$ 的比值不变，投影结果就相同。

这意味着从二维图像恢复深度在数学上需要额外的约束或先验。监督方法通过标注数据学习场景统计规律，无监督方法通过多视角几何约束提供额外信息。

### Q2：无监督深度估计中，为什么需要 SSIM 而不仅仅是 L1/L2 损失？（难度：⭐⭐）

**参考答案：**

纯像素级 L1/L2 损失对所有像素一视同仁。但在纹理均匀的区域（如天空、白墙），即使深度预测有较大误差，像素级差异也可能很小——因为没有纹理可供匹配。这会导致模型在这些区域"偷懒"，产生深度伪影。

SSIM 比较的是局部区域的结构信息（均值、方差、协方差），而不是单个像素的值。即使两个区域的绝对亮度不同，只要结构相似（如边缘方向一致），SSIM 就会给高分。这迫使模型学习更合理的深度结构，而不是只拟合像素值。

实践中，gamma 取 0.85（SSIM 权重 85%，L1 权重 15%）是经过大量实验验证的经验值。

### Q3：KITTI 数据集的深度评估中，为什么 LiDAR 点云只有约 5% 的有效像素？这给评估带来了什么问题？（难度：⭐⭐⭐）

**参考答案：**

KITTI 使用 64 线 LiDAR，垂直方向的角度分辨率约为 0.4 度，水平方向约为 0.08 度。在 1242x375 的图像分辨率下，LiDAR 的扫描点只能覆盖图像中约 5% 的像素位置。图像边缘、天空区域和远处物体上几乎没有 LiDAR 点。

这带来了三个评估问题：（1）模型可能在有 LiDAR 标注的区域表现很好，但在没有标注的区域（天空、图像边缘）完全失败，但评估指标看不到这一点；（2）LiDAR 在远处的点密度急剧下降，导致远距离深度的评估不可靠；（3）未标注像素的深度值为 0，如果不使用掩码直接计算，AbsRel 会因除以零或极小值而失真。

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 单目深度估计 | "一张图算距离" | 从单张 RGB 图像预测每个像素到相机距离的计算机视觉任务 |
| 深度图 | "距离图" | 二维图像，每个像素值代表该点到相机的真实距离（单位通常为米） |
| AbsRel | "相对误差" | 绝对相对误差——|预测深度 - 真实深度| / 真实深度 的均值，越低越好 |
| delta 准确率 | "精度" | 预测值与真实值的比值落在阈值内的像素比例，越高越好 |
| 光度一致性 | "图像匹配" | 无监督训练的核心假设——左图经深度和位姿变换后应与右图一致 |
| 尺度-偏移对齐 | "校准" | 将相对深度预测通过线性变换映射到绝对深度空间的预处理步骤 |
| 自编码器 | "压缩再还原" | 深度估计网络常用的编码器-解码器架构，先提取特征再恢复分辨率 |
| LiDAR | "激光雷达" | 通过激光脉冲测量距离的传感器，用于采集深度真值 |

## 📚 小结

单目深度估计是从二维图像恢复三维信息的核心能力。你掌握了三项 KITTI 标准评估指标（AbsRel、delta 准确率、RMSE_log），理解了监督方法（Eigen 的多尺度架构）和无监督方法（Monodepth2 的光度一致性损失）的原理差异，并实现了完整的深度解码器和评估流水线。尺度-偏移对齐是使用相对深度模型时不可跳过的关键步骤。

下一课我们将探讨多目标跟踪——如何在连续视频帧中追踪多个运动物体，将深度信息与目标检测和轨迹预测结合起来。

## ✏️ 练习

1. 【理解】用自己的话解释为什么无监督深度估计不需要 LiDAR 标签就能训练。用"缝合图像"的类比说明光度一致性损失的原理。写 200 字以内的说明。

2. 【实现】修改 `DeltaAccuracy` 函数，让它同时返回三个阈值（1.25、1.25^2、1.25^3）的结果。用合成数据测试，分析三个阈值之间的数值关系。

3. 【实验】用 Depth Anything V2 对同一张图片推理，分别在有尺度-偏移对齐和无对齐的情况下计算 AbsRel。对比两者的差异，解释为什么相对深度模型的输出必须对齐。

4. 【思考】在无监督训练中，如果输入的左右图像对来自两个不同时间点（而不是同时拍摄的双目），会出现什么问题？Monodepth2 如何处理这种情况？

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 深度评估指标库 | `code/main.py` | AbsRel、delta_accuracy、rmse_log 的完整实现 |
| 尺度-偏移对齐函数 | `code/main.py::align_scale_shift` | 可直接集成到任何深度评估流水线 |
| 深度解码器 | `code/main.py::DepthDecoder` | DPT 风格的卷积解码器，支持任意分辨率输入 |
| 光度一致性损失 | `code/main.py::PhotometricLoss` | 无监督深度估计的核心损失函数 |

## 📖 参考资料

1. [论文] Eigen, Fergus. "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network". NeurIPS, 2014. https://arxiv.org/abs/1406.2283
2. [论文] Godard et al. "Digging into Self-Supervised Monocular Depth Estimation". ICCV, 2019. https://arxiv.org/abs/1806.01260
3. [论文] Yang et al. "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data". CVPR, 2024. https://arxiv.org/abs/2401.10891
4. [论文] Yang et al. "Depth Anything V2". NeurIPS, 2024. https://arxiv.org/abs/2406.09414
5. [论文] Ranftl et al. "Vision Transformers for Dense Prediction". ICCV, 2021. https://arxiv.org/abs/2103.13413
6. [GitHub] LiheYoung/Depth-Anything. https://github.com/LiheYoung/Depth-Anything
7. [官方文档] Hugging Face Transformers - Depth Estimation: https://huggingface.co/docs/transformers/tasks/depth_estimation

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
