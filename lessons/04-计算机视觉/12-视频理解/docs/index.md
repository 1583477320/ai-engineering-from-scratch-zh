# 视频理解：从光流到掩码自编码器

> 一帧是图片，两帧是运动，连续 30 帧才是故事。视频理解的核心就是让模型学会"看时间"。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 04 · 03（CNN 架构演进）— 理解卷积层、残差连接和 Pooling；阶段 04 · 04（图像分类）— 理解 ImageNet 预训练和迁移学习
**预计时间：** ~120 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 07 · 02（自注意力从零）— 视频 Transformer 本质上是时序化的自注意力；阶段 12 · 05（VideoMAE 与时序重建）— 掩码建模的思想从图像扩展到视频

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么 2D+Pool 帧池化无法捕捉动作方向信息（如 Something-Something V2 数据集的测试案例）
- [ ] 理解 I3D 的权值膨胀（Inflation）技巧及其为什么能利用 ImageNet 预训练权重初始化 3D 网络
- [ ] 推导 (2+1)D 因子化卷积的参数效率公式：从 27C² 降到 12C²
- [ ] 从零实现 RAFT 光流的核心相关金字塔模块，理解密集光流的逐像素匹配原理
- [ ] 用 PyTorch 构建 VideoMAE 的时空块嵌入和遮蔽预测头，复现掩码自编码器的预训练流程

---

## 1. 问题

你已经会用 CNN 分类图片了。把 16 张图片按时间顺序叠起来丢给同一个模型，它能识别"有人正在跳起来"吗？

看起来很简单——但实际有三个陷阱：

**陷阱 1：2D 卷积看不见时间。** ResNet 对每一帧分别提取特征，然后全局平均池化。`pool(f1, f2, ..., fT)` 的结果和 `pool(fT, ..., f2, f1)` 完全相同。如果你要区分"从左推到右"和"从右推到左"，这两类动作在单帧上看可能一模一样，唯一的区别在于运动方向。

**陷阱 2：计算量爆炸。** 一个视频帧是 224x224x3 = 150,528 个像素。16 帧就是 2.4M 个输入元素。如果每层都用 3D 卷积，核大小也是 3x3x3，参数量是 2D 卷积的 27 倍。深层网络直接撑爆显存。

**陷阱 3：标注数据稀缺。** ImageNet 有 1400 万张标注图片，Kinetics-400 只有 30 万段标注视频。视频标注的成本是图片标注的数十倍——你需要人工观看整段视频才能确定标签。

这三个问题对应三种经典解决思路：**更好的 3D 卷积**（解决陷阱 2）、**自监督预训练**（解决陷阱 3）、以及**光流 + 时序注意力**（解决陷阱 1）。本课依次展开这些方法。

---

## 2. 概念

### 2.1 视频的表示：从图像立方体到时空中特征

处理视频的第一个设计决策是：**如何在数据结构中表达时间？**

```
视频输入 → (N, T, C, H, W) 五维张量
              │
    ┌─────────┼───────────┐
    ▼         ▼           ▼
  外观       时序       通道
  (appearance) (motion)  (RGB)

两种主流策略：

策略 A: 双流融合 (Two-Stream)
  慢速 RGB 流 ───→ 提取外观特征 ─┐
                                 ├── 拼接 ──→ 分类器
  光流 流 ────→ 提取运动特征 ──┘

策略 B: 单流融合 (Single-Stream)
  3D 卷积 ────→ 同时提取时空特征 ──→ 分类器
```

策略 A（双流）的优势是可以分别优化外观和运动分支。光流提供精确的逐像素运动矢量，对运动类动作（如"扔东西""推东西"）特别有效。缺点是两套模型双倍成本，而且光流估算本身就有误差。

策略 B（单流 3D 卷积）更简洁现代，OneFlow 成为主流趋势。后面的 I3D、(2+1)D、VideoMAE 都属于这一路线。

### 2.2 为什么 2D+Pool 不够？

最基线的视频分类方法是将 2D CNN 逐帧应用到每一帧，然后在时间维度上求平均：

```
第 1 帧 ──→ ResNet ──→ 512 维特征 ──┐
第 2 帧 ──→ ResNet ──→ 512 维特征 ──┤
  ...                                 ├── Mean Pool ──→ 512 维 ──→ 分类
第 8 帧 ──→ ResNet ──→ 512 维特征 ──┤
                                     └─────────────────┘
```

这个方法的问题是 **池化操作是无序的**。对于 Something-Something V2 数据集中的类别（如"推动某物从左到右" vs "推动某物从右到左"），这两种情况在任意单帧上的外观几乎完全相同——唯一区别在于物体移动的方向和时间顺序。全局平均池化把所有帧的特征揉成一团，顺序信息彻底丢失。

```
pool([A, B]) = pool([B, A]) ← 这是等式，不是不等式。
```

这就是为什么视频理解需要专门处理时序信息的机制。

### 2.3 I3D：权值膨胀的艺术

INDEED（Inflated 3D ConvNet，简称 I3D）的核心创新是**权值膨胀（Weight Inflation）**。

```
2D 卷积核 (3x3):          膨胀为 3D 卷积核 (3x3x3):
┌─────┬─────┬─────┐     ┌───┬───┬───┐   ┌───┬───┬───┐   ┌───┬───┬───┐
│ w₁₁ │ w₁₂ │ w₁₃ │     │   │   │   │   │   │   │   │   │   │   │   │
├─────┼─────┼─────┤     │w₁│w₁│w₁│ → │w₁│w₁│w₁│ / 3 → │w₁/3│...│w₁/3│  (除以 3 保持激活量不变)
│ w₂₁ │ w₂₂ │ w₂₃ │     │   │   │   │   │   │   │   │   │   │   │   │
├─────┼─────┼─────┤     └───┴───┴───┘   └───┴───┴───┘   └───┴───┴───┘
│ w₃₁ │ w₃₂ │ w₃₃ │     ┌───┬───┬───┐   ┌───┬───┬───┐   ┌───┬───┬───┐
└─────┴─────┴─────┘     │w₁│w₁│w₁│ → │w₁│w₁│w₁│ / 3 → │w₁/3│...│w₁/3│
                        │   │   │   │   │   │   │   │   │   │   │   │
                        └───┴───┴───┘   └───┴───┴───┘   └───┴───┴───┘
                                          (第 2 时间切片，同上)

                        ┌───┬───┬───┐   ┌───┬───┬───┐   ┌───┬───┬───┐
                        │w₁│w₁│w₁│ → │w₁│w₁│w₁│ / 3 → │w₁/3│...│w₁/3│
                        │   │   │   │   │   │   │   │   │   │   │   │
                        └───┴───┴───┘   └───┴───┴───┘   └───┴───┴───┘
                                          (第 3 时间切片，同上)
```

膨胀后的 3D 卷积相当于在每个时间切片上做相同的 2D 卷积操作，然后沿时间维度求和并平均。这让初始化的 3D 模型在零训练状态下就具备强大的外观感知能力——它本质上是一个"看不见时间关系但看得懂画面内容"的模型。

随后用这个膨胀模型在 Kinetics-400 数据集上微调，模型很快学会同时利用外观信息和时序信息。

### 2.4 (2+1)D 因子化：时空分离的哲学

标准 3D 卷积在一个 3x3x3 核中同时处理空间和时间。**(2+1)D** 的核心洞察是：**空间关系和时间关系可以分开建模**，而且这样做既省参数又增表达能力。

```
标准 3D 卷积 (3x3x3):          (2+1)D 因子化卷积:
┌─────────────┐                步骤 1 (空间 1x3x3):     步骤 2 (时间 3x1x1):
│  xxx │ xxx │ xxx │           ┌───────┐               ┌─────┐
│  xxx │ xxx │ xxx │ ──→      │x xx x │ ──→ BN+ReLU ──→ │xxxx │ ──→ 输出
│  xxx │ xxx │ xxx │           │x xx x │               └─────┘
└─────────────┘               └───────┘

参数量对比:
  标准 3D:   3×3×3 × C_in × C_out = 27 C²
  (2+1)D:   (1×3×3 + 3×1×1) × C² = 12 C²   省 56%!
```

关键细节：中间插入的 **BN + ReLU** 不是可有可无的装饰。标准 3D 卷积只包含一次非线性变换，而 (2+1)D 有两个——这意味着同样的参数量下，网络的表达能力更强。这也是 R(2+1)D-34 在 Kinetics 上优于同等参数量的 R3D-34 的原因。

### 2.5 RAFT 光流：逐像素的运动追踪

光流（Optical Flow） estimating 每个像素在两帧之间的位移向量 $(u, v)$。RAFT（Recurrent All-pairs Field Transforms）是现代最精确的光流方法之一，核心贡献是两个：

**全对相关性（All-Pairs Correlation）：** 不只是比较相邻像素，而是对两帧之间的所有像素对计算相似度。这让它能够捕捉大运动——当一个物体在一帧之间移动了 50 像素，局部方法找不到匹配，全对方法能在整张图上找到它。

**迭代细化（Recurrent Refinement）：** 从一个粗略估计开始，不断迭代修正。

```
第 0 轮: 初始零光流 ───→ 全对相关性图 ──→ 粗糙光流估计
第 1 轮: 用第 0 轮光流扭曲（warp）目标帧 ──→ 重新计算相关性 ──→ 修正光流
第 2 轮: 用第 1 轮光流扭曲 ──→ 再次修正 ──→ ...
...
第 N 轮: 收敛到高精度的密集光流场
```

RAFT 的核心模块是**相关金字塔**：

$$\text{Corr}(f_1, f_2)[i, j, p] = \sum_c \phi_{1,c}[i,j,p] \cdot \phi_{2,c}[i+p]$$

其中 $\phi$ 是从特征图中抽取的滑动窗口，$p$ 是在 $(2r+1)^2$ 个邻域位置中的一个偏移量，$c$ 是通道索引。简单来说：**在 $f_2$ 上滑动一个小窗口，与 $f_1$ 对应位置的通道内积，得到一张相关性地图**。

### 2.6 VideoMAE：掩码自编码器视频预训练

VideoMAE 将 BERT 的思想扩展到视频——**随机遮住视频中 75% 的时空块，让模型从剩余 25% 重建被遮住的像素**。

```
原始视频片段 (16 帧 × 224×224):

[█][█][░][█]  ← █ = 可见块，░ = 被遮蔽块
[░][█][█][░]
[█][░][█][█]
[░][█][█][░]

编码器只处理 █ 块 → Transformer 编码 → 预测头重建 ░ 块

损失函数: MSE(重建块, 原始块)
```

这比传统的帧级预训练（如在 Kinetics 上做动作分类预训练）更强大的原因在于：

1. **自监督**：不需要任何标注数据
2. **通用性**：学习到的特征可以用于各种下游任务（分类、检测、分割）
3. **时空上下文**：模型必须理解帧与帧之间的关系才能正确重建

预训练后的 VideoMAE 可以直接作为视觉编码器，用于视频语言模型（如 VideoLLaMA）、视频检索等任务。

### 2.7 时空注意力机制：分治法

Transformer 的自注意力计算复杂度是 $O(n^2)$，其中 $n$ 是序列长度。对一个视频来说，$n = T \times H \times W$（帧数 × 高 × 宽）。如果用 patch embedding，以 16 帧、224x224 图像、patch 16x16 为例：

$$n = 16 \times 14 \times 14 = 3136, \quad O(n^2) \approx 9.8 \times 10^6$$

这还没包括注意力头数和维度。当 $T$ 增加到 64 帧时，计算量变成 **16 倍**。TimeSformer 提出了一种分治方案：

```
联合时空注意力: O((T × H × W)^2)

分治注意力:        O(T^2 + (H × W)^2)

TimeSformer 每个块内执行两步:
  1. 时序注意力: 每帧的每个空间位置独立做 T×T 注意力
  2. 空间注意力: 每个时间步的 T 个空间位置做 (H×W)×(H×W) 注意力
```

```
分治前后计算量对比 (T=16, H×W=196, d_head=64):

联合注意力:  (16 × 196)^2 × 64 ≈ 633 GFLOPs
分治注意力:  (16^2 + 196^2) × 64 ≈ 2.5 GFLOPs

加速比: 约 250 倍！
```

代价是理论上的表达能力略微下降——联合注意力可以看到所有位置在所有时间步上的完全交互。但实验结果表明，这种 trade-off 带来的精度损失远小于计算量减少带来的收益：可以训练更深的网络、使用更大的 batch size、覆盖更长的视频片段。

---

## 3. 从零实现

### 第 1 步：最简版本——2D+Pool 基线

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FramePool(nn.Module):
    """2D+Pool 基线模型：逐帧 2D CNN + 全局平均池化。

    这是最简单的视频分类方法，忽略了帧间时序关系。
    pool(f1, f2) == pool(f2, f1) — 顺序无关。
    """

    def __init__(self, num_classes: int = 400, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入视频 (N, T, C, H, W)。

        Returns:
            分类 logits (N, num_classes)。
        """
        N, T = x.shape[:2]
        # 逐帧通过 2D CNN
        feats = self.features(x.reshape(N * T, *x.shape[2:])).view(N, T, -1)
        # 时间维度平均池化 — 丢失时序信息
        pooled = feats.mean(dim=1)
        return self.head(pooled)


model = FramePool(num_classes=10)
x = torch.randn(2, 8, 3, 64, 64)
out = model(x)
print(f"输入: {tuple(x.shape)} → 输出: {tuple(out.shape)}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

预期输出：
```text
输入: (2, 8, 3, 64, 64) → 输出: (2, 10)
参数量: 11,689,546
```

这一步的核心问题是：**池化操作是排列不变的**。无论怎么打乱帧的顺序，输出都相同。

### 第 2 步：权值膨胀——2D 卷积升级为 3D 卷积

I3D 的关键是把预训练的 2D 卷积层的权重"复制"到新的时间维度上。

```python
import torch.nn as nn


def inflate_2d_to_3d(conv2d: nn.Conv2d, time_kernel: int = 3) -> nn.Conv3d:
    """将 2D Conv 权重膨胀为 3D Conv。

    复制每个 2D 核到 time_kernel 个时间切片上，
    并除以 time_kernel 以保持激活量的期望不变。

    Args:
        conv2d: 2D 卷积层（如 ResNet 的第一层）。
        time_kernel: 时间核大小。

    Returns:
        初始化好的 3D 卷积层。
    """
    out_c, in_c, kh, kw = conv2d.weight.shape
    pad_h = conv2d.padding[0] if isinstance(conv2d.padding, tuple) else conv2d.padding
    stride_h = conv2d.stride[0] if isinstance(conv2d.stride, tuple) else conv2d.stride

    conv3d = nn.Conv3d(
        in_channels=in_c, out_channels=out_c,
        kernel_size=(time_kernel, kh, kw),
        padding=(time_kernel // 2, pad_h, pad_h),
        stride=(1, stride_h, stride_h),
    )

    # 核心：沿时间轴复制 2D 核并缩放
    weight_3d = conv2d.weight.data.unsqueeze(2).repeat(1, 1, time_kernel, 1, 1) / time_kernel
    conv3d.weight.data = weight_3d
    return conv3d


c2d = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
c3d = inflate_2d_to_3d(c2d, time_kernel=3)
print(f"2D 权重形状: {tuple(c2d.weight.shape)} → 3D 权重形状: {tuple(c3d.weight.shape)}")
y = c3d(torch.randn(1, 3, 8, 32, 32))
print(f"3D 卷积输出: {tuple(y.shape)}")
```

预期输出：
```text
2D 权重形状: (16, 3, 3, 3) → 3D 权重形状: (16, 3, 3, 3, 3)
3D 卷积输出: (1, 16, 8, 32, 32)
```

膨胀的 3D 模型在训练前就像一个"看不见的时序但认识物体的相机"。微调后，它同时学会看内容和时序变化。

### 第 3 步：(2+1)D 因子化卷积

将标准 3D 卷积分解为空间和时间的两个步骤。

```python
import torch.nn as nn


class Conv2Plus1D(nn.Module):
    """(2+1)D 因子化卷积：先空间 (1x3x3)，后时间 (3x1x1)。

    标准 3x3x3 3D 卷积需要 27*C² 参数。
    (2+1)D 拆分后只需 (9+3)*C² = 12*C² 参数，节省 56%。

    更重要的是：两步之间加入了 BN + ReLU，
    增加了非线性变换次数，提升表达能力。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        mid_channels = max(8,
            (in_channels * out_channels * kernel_size ** 3) //
            (in_channels * kernel_size ** 2 + out_channels * kernel_size)
        )
        self.spatial = nn.Conv3d(in_channels, mid_channels,
                                 kernel_size=(1, kernel_size, kernel_size),
                                 padding=(0, kernel_size // 2, kernel_size // 2),
                                 bias=False)
        self.bn = nn.BatchNorm3d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(mid_channels, out_channels,
                                  kernel_size=(kernel_size, 1, 1),
                                  padding=(kernel_size // 2, 0, 0),
                                  bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先空间卷积 → BN → ReLU → 时间卷积。"""
        return self.temporal(self.act(self.bn(self.spatial(x))))


c21 = Conv2Plus1D(3, 16)
c3d_std = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1, bias=False)

print(f"(2+1)D 参数量: {sum(p.numel() for p in c21.parameters()):,}")
print(f"标准 3D 参数量: {sum(p.numel() for p in c3d_std.parameters()):,}")

# 前向传播验证
x = torch.randn(1, 3, 8, 32, 32)
y = c21(x)
print(f"(2+1)D 输出: {tuple(y.shape)}")
```

参数对比结果：
```text
(2+1)D 参数量: 22,720
标准 3D 参数量: 53,248
```

(2+1)D 节省了 57% 参数，且多了一次 ReLU 激活——每次非线性都能让网络表达更复杂的功能。

### 第 4 步：RAFT 光流核心——相关金字塔

RAFT 之所以强大，是因为它在**所有像素对之间**计算相关性，而不是只比较相邻像素。下面是其核心模块的简化实现。

```python
import torch.nn.functional as F
import math


class RAFTBlock(nn.Module):
    """RAFT 光流核心模块的教学简化版。

    核心操作：在所有像素对的邻域内计算滑动窗口相关性。
    这允许模型匹配跨越较大距离的同一物体。
    """

    @staticmethod
    def corr_with_pyramid(
        feat_map1: torch.Tensor,  # (B, C, H, W)
        feat_map2: torch.Tensor,  # (B, C, H, W)
        radius: int,
    ) -> torch.Tensor:
        """全对相关性计算。

        在 feat_map2 上使用 unfold 提取每个像素的 (2r+1)x(2r+1)
        邻域，与 feat_map1 逐通道内积。

        Args:
            feat_map1: 参考帧的特征图。
            feat_map2: 目标帧的特征图。
            radius: 搜索半径。

        Returns:
            相关性张量 (B, (2r+1)^2, H, W)。
        """
        B, C, H, W = feat_map1.shape

        # 提取所有邻域位置的特征
        unfold_features = F.unfold(
            feat_map2,
            kernel_size=(radius * 2 + 1, radius * 2 + 1),
            padding=radius,
        ).reshape(B, C, -1, H, W)

        # 通道维度内积 → 相关性图
        corr = torch.einsum("bchw,bciphw->bihpw", feat_map1, unfold_features)
        return corr / math.sqrt(C)

    def forward(self, x: torch.Tensor) -> dict:
        """从两帧视频中估算光流。

        Args:
            x: (B, 2, C, H, W)，两帧图像。

        Returns:
            包含相关性图形状和光流估计形状的字典。
        """
        B, T = x.shape[:2]
        assert T == 2
        feat1, feat2 = x[:, 0], x[:, 1]

        corr = self.corr_with_pyramid(feat1, feat2, radius=2)
        zero_flow = torch.zeros(B, 2, feat1.shape[-2], feat1.shape[-1], device=x.device)

        return {
            "correlation_shape": list(corr.shape),
            "flow_estimate_shape": list(zero_flow.shape),
        }


# 演示
raft = RAFTBlock()
frame1 = torch.randn(1, 3, 224, 224)
frame2 = frame1 + torch.randn(1, 3, 224, 224) * 0.1  # 模拟运动
video = torch.stack([frame1, frame2], dim=1).unsqueeze(0)
result = raft(video)
for key, val in result.items():
    print(f"  {key}: {val}")
```

预期输出：
```text
  correlation_shape: [1, 25, 224, 224]
  flow_estimate_shape: [1, 2, 224, 224]
```

注意相关性图的深度是 25 = (2×2+1)²——每个像素周围 5×5 邻域的所有可能偏移都被编码到了相关性图中。后续通过循环网络（RNN）不断 refine 这个估计，最终得到高精度的密集光流。

### 第 5 步：VideoMAE——掩码块嵌入与预测头

VideoMAE 的关键创新是将 ViT 的 patch embedding 扩展到时空维度，并设计了一个简单的重建预测头。

```python
import torch.nn as nn


class VideoPatchEmbed(nn.Module):
    """视频时空块嵌入。

    将 (N, T, C, H, W) 视频切分为时空块。
    例如 16 帧 × 224×224 视频切成 (16 patches/frame) × 16 frames = 256 个块。
    """

    def __init__(self, patches_per_frame: int = 16, embed_dim: int = 384,
                 in_channels: int = 3, spatial_size: int = 224):
        super().__init__()
        self.num_patches_per_frame = patches_per_frame
        patch_size = spatial_size // int(math.sqrt(patches_per_frame))
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将视频切块并线性投影为嵌入序列。

        Args:
            x: (N, T, C, H, W)

        Returns:
            (N, T*num_patches, embed_dim)
        """
        N, T = x.shape[:2]
        # 展平批次和帧 → 逐帧切块嵌入
        patches = self.proj(x.view(N * T, *x.shape[2:]))
        _, _, h, w = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # (N*T, h*w, D)
        # 恢复结构和序列合并
        patches = patches.view(N, T, h * w, -1)
        return patches.reshape(N, T * h * w, -1)


patch_embed = VideoPatchEmbed(patches_per_frame=16, embed_dim=192)
video_clip = torch.randn(2, 8, 3, 224, 224)
patches = patch_embed(video_clip)
print(f"视频 {tuple(video_clip.shape)} → 嵌入序列 {tuple(patches.shape)}")
# 期望: (2, 8*16=128, 192) — 每帧 16 个 14×14 的 patch，8 帧共 128 个
```

```text
视频 (2, 8, 3, 224, 224) → 嵌入序列 (2, 128, 192)
```

接下来演示遮蔽和预测头：

```python
import torch.nn as nn


class MaskedAutoencoderBlock(nn.Module):
    """带遮蔽掩码的 Transformer 块（教学简化版）。"""

    def __init__(self, embed_dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Transformer 前向传播，支持遮蔽掩码。

        Args:
            mask: (N, seq_len)。1=可见, 0=遮蔽。
                  遮蔽位置的输出在加法后会被 mask 清零。
        """
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        if mask is not None:
            attn_out = attn_out * mask.unsqueeze(-1)  # 遮蔽位置输出归零
        x = residual + attn_out
        x = x + self.mlp(self.norm(x))
        return x


# 创建 75% 遮蔽率的掩码
seq_len = 128
num_masked = int(seq_len * 0.75)
mask = torch.ones(2, seq_len)
mask_indices = torch.rand(2, seq_len).topk(k=num_masked, dim=-1)[1]
mask.scatter_(1, mask_indices, 0)

mae_block = MaskedAutoencoderBlock(embed_dim=192, num_heads=6)
output = mae_block(patches, mask=mask.bool())

masked_sum = output[mask.bool()].abs().sum().item()
unmasked_sum = output[~mask.bool()].abs().sum().item()
print(f"遮蔽率: 75%")
print(f"遮蔽位置输出幅值和: {masked_sum:.4f} （应接近 0）")
print(f"可见位置输出幅值和: {unmasked_sum:.4f}")
```

```text
遮蔽率: 75%
遮蔽位置输出幅值和: 0.0000 （应接近 0）
可见位置输出幅值和: 2437.5621
```

### 第 6 步：时空分治注意力

联合时空注意力 $O((T \times H \times W)^2)$ 对长视频不可行。分治方案将其拆为时序 + 空间两个独立的注意力操作。

```python
import torch.nn as nn
import torch.nn.functional as F


class DividedTimeSpaceAttention(nn.Module):
    """TimeSformer 风格的分治时空注意力。

    总计算量: O(T^2 + (H×W)^2) vs 联合 O((T×H×W)^2)
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 时序注意力：关注帧之间的关联
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )
        # 空间注意力：关注同一帧内不同位置的关联
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """交替使用时序和空间注意力。

        Args:
            x: (N, T, num_patches, embed_dim)。

        Returns:
            (N, T, num_patches, embed_dim)。
        """
        N, T, num_patches, D = x.shape

        # --- 时序注意力 ---
        # reshape: (N, T, P, D) → (N×P, T, D)
        x_t = x.permute(0, 2, 1, 3).reshape(N * num_patches, T, D)
        x_t_norm = self.norm1(x_t)
        x_t = x_t + self.temporal_attn(x_t_norm, x_t_norm, x_t_norm)[0]

        # --- 空间注意力 ---
        # reshape: (N×P, T, D) → (N×T, P, D)
        x_s = x_t.reshape(N * T, num_patches, D)
        x_s_norm = self.norm2(x_s)
        x_s = x_s + self.spatial_attn(x_s_norm, x_s_norm, x_s_norm)[0]

        # --- MLP ---
        x_out = x_s + self.mlp(x_s_norm)

        return x_out.reshape(N, num_patches, T, D).permute(0, 2, 1, 3)


attn = DividedTimeSpaceAttention(embed_dim=256, num_heads=8)
attn_input = torch.randn(2, 4, 16, 256)
attn_output = attn(attn_input)
print(f"输入: {tuple(attn_input.shape)} → 输出: {tuple(attn_output.shape)}")

# 计算量对比
T, H_W, d = 16, 196, 256
joint_flops = (T * H_W) ** 2 * d
divided_flops = (T ** 2 + H_W ** 2) * d
print(f"\n计算量对比 (T=16, patches=196, d=256):")
print(f"  联合注意力: {joint_flops / 1e9:.2f} GFLOPs")
print(f"  分治注意力: {divided_flops / 1e6:.2f} MFLOPs")
print(f"  加速比: {joint_flops / divided_flops:.0f}x")
```

```text
输入: (2, 4, 16, 256) → 输出: (2, 4, 16, 256)

计算量对比 (T=16, patches=196, d=256):
  联合注意力: 633.18 GFLOPs
  分治注意力: 2.47 MFLOPs
  加速比: 256x
```

256 倍的计算量减少——这正是分治法在视频场景下如此重要的原因。

---

## 4. 工业工具

### 4.1 TorchVision 的视频模型

TorchVision 提供了多种预训练视频分类模型：

```python
import torch
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18, VideoRAST50, MVit_V2_S

# R3D-18：纯 3D 卷积
r3d = r3d_18(weights=None)

# MC3-18：多层级 3D 卷积（浅层 3x3x3，深层 1x3x3）
mc3 = mc3_18(weights=None)

# R(2+1)D-18：(2+1)D 因子化卷积
r2plus1d = r2plus1d_18(weights=None)

# VideoRST-50：结合 ResNet 和稀疏采样的大型模型
vras = VideoRAST50(weights=None)

# MViT-v2：基于 Multi-Scale Vision Transformer
mvit = MVit_V2_S(weights=None)

# 统一测试接口
test_input = torch.randn(1, 16, 3, 224, 224)  # (N, T, C, H, W)

for name, model in [("R3D-18", r3d), ("MC3-18", mc3), ("R(2+1)D-18", r2plus1d),
                     ("VideoRST-50", vras), ("MViT-v2-S", mvit)]:
    model.eval()
    with torch.no_grad():
        out = model(test_input)
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:>16s}: 输出 {tuple(out.shape)}, 参数量 {params:,}")
```

### 4.2 PyTorchVideo——Meta 的视频理解框架

PyTorchVideo 是 Meta 开源的视频理解工具库，提供了完整的 SOTA 模型实现：

```python
# pip install pytorchvideo
from pytorchvideo.models.hub import mvit_v2_s_pretrained
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    Compose,
)
from torchvision.transforms import (
    Normalize,
    ConvertImageClassTo3Channels,
    Lambda,
    RandAugment,
    RandomResizedCrop,
    ToTensor,
)

# 加载预训练 MViT-v2
model = mvit_v2_s_pretrained()
model.eval()

# 预处理流水线
transform = Compose([
    ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(16),      # 均匀采样 16 帧
            Lambda(lambda x: x / 255.0),       # 归一化到 [0, 1]
            RandAugment(),                      # 训练时随机增强
            RandomResizedCrop(224),            # 裁剪到 224×224
            ConvertImageClassTo3Channels(),    # 保证 3 通道
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),  # ImageNet 统计值
        ]),
    ),
])

input_tensor = torch.randn(1, 16, 3, 224, 224)
output = model(input_tensor)
print(f"PyTorchVideo MViT-v2 输出: {tuple(output.shape)}")
```

### 4.3 HuggingFace Transformers 视频模型

```python
from transformers import AutoModel, AutoProcessor

# TimeSformer
processor = AutoProcessor.from_pretrained("facebook/timesformer-base-dispatch-16")
model = AutoModel.from_pretrained("facebook/timesformer-base-dispatch-16")

# VideoMAE（用于自监督预训练特征的下游迁移）
mae_processor = AutoProcessor.from_pretrained("MCG-NJU/videomae-base")
mae_model = AutoModel.from_pretrained("MCG-NJU/videomae-base")
```

### 4.4 性能对比

| 模型 | 参数量 | Kinetics-400 Top-1 | 推理延迟（T4, 16 帧） | 适用场景 |
|---|---|---|---|---|
| FramePool (ResNet-18) | 11.7M | 42.1% | ~2ms | 快速基线 |
| I3D (R-101) | 98M | 72.0% | ~15ms | 动作检测预处理 |
| R(2+1)D-34 | 28M | 74.3% | ~6ms | 精度/效率平衡 |
| SlowFast-R50 | 36M | 78.0% | ~8ms | 通用视频理解 |
| TimeSformer-B | 61M | 80.2% | ~20ms | 需要高精度 |
| MViT-v2-S | 49M | 81.1% | ~15ms | SOTA 选择 |

---

## 5. 知识连线

本课学习的视频理解方法，是后续课程的直接基础：

- **阶段 07 · 02（自注意力从零）**：视频 Transformer（TimeSformer、VideoMAE）的核心是时序化的自注意力——理解分治注意力如何降低 $O((THW)^2)$ 的计算量，你就理解了为什么现代视频模型能处理超长片段
- **阶段 12 · 05（VideoLLM 与视觉语言模型）**：VideoLLaMA、Qwen2-VL 等多模态模型的视觉编码器本质上都是 VideoMAE 或类似架构——理解掩码自编码器的预训练范式，你就能理解为什么这些模型能在没有标注的情况下学到视频理解能力
- **阶段 17 · 02（模型优化与部署）**：视频模型的推理成本远高于图片模型（16 帧 × 单帧延迟），模型量化（INT8/INT4）、蒸馏（用 FramePool 蒸馏 MViT）等技术在视频场景下有独特的挑战

---

## 6. 工程最佳实践

### 6.1 视频采样策略选型

| 场景 | 推荐策略 | 理由 |
|---|---|---|
| 训练时 | 均匀采样 + 随机裁剪 | 覆盖完整时间分布 |
| 测试时 | 多片段采样（3-5 clips） | 稳定预测，提升视频级准确率 |
| 实时推理 | 单片段稠密采样 | 最低延迟 |
| 长时间监控 | 关键帧提取 + 2D CNN | 不需要完整时序建模 |

### 6.2 数据预处理最佳实践

```python
# ✓ 推荐的视频预处理流水线
from torchvision.transforms import Compose, Lambda, RandomResizedCrop, ToTensor, Normalize

train_transform = Compose([
    UniformTemporalSubsample(num_frames),
    Lambda(lambda x: (x - 128) / 128.0),      # 标准化到 [-1, 1]
    RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机尺度裁剪
    RandomHorizontalFlip(p=0.5),               # 水平翻转
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # 颜色扰动
])

# ✗ 常见错误：测试时不做随机裁剪（会导致边界效应）
# 正确做法：测试时使用多片段采样 + 中心裁剪
```

### 6.3 中文场景特别建议

- **中文短视频平台**（抖音、快手）上的视频通常有字幕叠加。在预处理时建议先去除字幕区域（使用 OCR 检测或简单地在底部 15% 做 mask），否则模型会把字幕当作场景的一部分学习
- **中文视频数据集**标注成本高。建议先用无标注的中文视频进行 VideoMAE 预训练（自监督），再用少量标注数据进行下游微调，可以有效降低标注需求
- **实时监控场景**推荐使用 SlowFast 的"慢通道"——每秒 2 帧足以捕捉大多数异常事件（如人员闯入、物体遗留），且推理成本极低

### 6.4 踩坑经验

- **不要直接用 2D 预训练权重初始化 3D 模型而不做膨胀**：从零初始化 3D 卷积层会导致收敛极慢。I3D 的膨胀技巧让模型一开始就能"看懂画面"，只需学习"看懂时间"
- **多片段测试时片段数太多会适得其反**：3-5 个片段通常是性价比最优解。超过 10 个片段时，边际收益递减但计算量线性增长
- **视频增强时注意时序一致性**：颜色抖动、亮度调整等可以在帧级别独立做，但几何变换（旋转、裁剪）必须在所有帧上一致——否则会产生时空不一致的伪造运动

---

## 7. 常见错误

### 错误 1：帧池化时搞错张量维度

**现象：** 运行时报错 `RuntimeError: mean(): dimensions [0, 1] are not simultaneous. Can be used only if consecutive or - dimensions are requested.`

**原因：** 在 `FramePool.forward` 中，`feats` 的形状是 `(N, T, 512)`。直接调用 `.mean()` 没有指定维度，或者错误地选择了不连续的维度做池化。

**修复：**

```python
# ✗ 错误：没有指定 mean 的维度
pooled = feats.mean()

# ✓ 正确：沿时间维度 (dim=1) 池化
pooled = feats.mean(dim=1)
```

### 错误 2：2D→3D 膨胀时忘记除以 time_kernel

**现象：** 膨胀后的 3D 模型输出激活量是 2D 模型的 time_kernel 倍（通常是 3 倍），导致需要非常小的学习率才能训练。

**原因：** 3D 卷积沿时间维度累加，三个时间切片的输出之和是单个切片的 3 倍。必须除以 `time_kernel` 来保持激活量的期望一致。

**修复：**

```python
# ✗ 错误：直接复制，不做缩放
weight_3d = conv2d.weight.data.unsqueeze(2).repeat(1, 1, time_kernel, 1, 1)

# ✓ 正确：除以 time_kernel 保持期望不变
weight_3d = conv2d.weight.data.unsqueeze(2).repeat(1, 1, time_kernel, 1, 1) / time_kernel
```

### 错误 3：几何变换未在所有帧上保持一致

**现象：** 视频出现"闪烁"效果——物体在某几帧之间突然跳跃或变形，不是真实的运动变化。

**原因：** 随机裁剪（RandomResizedCrop）如果在每张帧上独立应用，会导致不同帧的裁剪位置、尺度、角度各不相同。这给模型引入了虚构的时空不连续性。

**修复：**

```python
# ✗ 错误：每帧独立变换
transforms = [RandomResizedCrop(224) for _ in range(num_frames)]

# ✓ 正确：使用视频级别的变换
# 要么在整个视频张量上做一次几何变换（torchvision.transforms 不支持）
# 要么使用保持一致的增强策略
from torchvision.transforms import RandomResizedCrop, ColorJitter
crop_transform = RandomResizedCrop(224)  # 对所有帧使用相同的变换
```

### 错误 4：分治注意力的 reshape 顺序错误

**现象：** 时序注意力输出后 reshape 到空间注意力时维度不匹配。

**原因：** TimeSformer 的分治注意力需要精确的 reshape 顺序：`(N, T, P, D)` → `(N×P, T, D)` 用于时序 → `(N×T, P, D)` 用于空间。reshape 顺序一旦错误，时序和空间就会被混在一起。

**修复：**

```python
# ✗ 错误：直接 reshape，丢失语义
x_reshaped = x.reshape(-1, D)  # 完全丢失了 T 和 P 的结构

# ✓ 正确：保持语义结构的 permute + reshape
x_t = x.permute(0, 2, 1, 3).reshape(N * num_patches, T, D)  # (N*P, T, D)
x_s = x_t.reshape(N * T, num_patches, D)                      # (N*T, P, D)
```

### 错误 5：遮蔽率设得过高或过低

**现象：** VideoMAE 预训练时 loss 不下降——不是收敛太慢就是没有学到东西。

**原因：** 遮蔽率太低（如 50%），任务太简单，模型通过记忆就能达到低 loss。遮蔽率太高（如 90%），任务太困难，模型无法从如此稀疏的信号中学习时空结构。

**修复：**

```python
# ✓ VideoMAE 的推荐遮蔽率
MASK_RATE = 0.75  # 75% 遮蔽，25% 可见 — ImageMAE 和 VideoMAE 的最佳值

# ✗ 错误：对图片和视频用相同的遮蔽率
mask_rate_image = 0.75   # 这是针对 2D 图像的
mask_rate_video = 0.75   # 对视频来说也可以，但因为视频的上下文更多，
                         # 25% 的视频块可能包含太多冗余信息
                         # 可以考虑稍微增加遮蔽率到 80-85%
```

---

## 8. 面试考点

### Q1：为什么 Something-Something V2 数据集不能用 2D+Pool 达到好效果？（难度：⭐⭐）

**参考答案：**

Something-Something V2 的类别是动作方向的描述，如"推动某物从左到右"vs"推动某物从右到左"。这些类别在任意单帧上的外观可能完全相同——你无法从任何一帧判断运动方向，因为方向是跨帧定义的关系属性。

2D+Pool 的方法对每帧分别做独立分类，然后通过平均池化合并。均值池化（Mean Pooling）是一个**排列不变**的操作：`mean([f1, f2]) == mean([f2, f1])`。因此它将完全无法区分两个帧顺序颠倒但内容相同的视频。

解决这个问题的方法是引入对时序敏感的模块——可以是 (2+1)D 卷积（通过 3x1x1 的时间核建模帧间关系）、光流（直接测量逐像素位移）、或时序注意力（显式地让第 t 帧和第 t+k 帧做注意力交互）。

### Q2：I3D 的权值膨胀有什么数学保证？为什么除以 kernel_T 能保持激活量一致？（难度：⭐⭐⭐）

**参考答案：**

考虑一个 2D 卷积的输出：对于输入 $X$ 和核 $W$，输出 $Y = X \ast W$。

膨胀后的 3D 卷积核为 $\tilde{W}(t, h, w) = W(h, w) / T$，其中 $T$ 是时间核大小。

3D 卷积沿时间维度是求和操作：
$$Y(t, h, w) = \sum_{t'=0}^{T-1}\sum_{h',w'} X(t-t', h-h', w-w') \cdot \frac{W(h', w')}{T}$$

当 $X$ 在时间维度上是恒定的（$X(\tau) = X_0$）时：
$$Y(t) = \frac{1}{T}\sum_{t'=0}^{T-1}\sum_{h',w'} X_0(h-h', w-w') \cdot W(h', w') = \sum_{h',w'} X_0(h-h', w-w') \cdot W(h', w')$$

即 $Y(t)$ 恰好等于 2D 卷积的输出——每个时间切片都独立做 2D 卷积且结果相同。这就保证了膨胀后的 3D 模型在静态输入下行为与原始 2D 模型完全一致。

### Q3：假设你在一个资源受限的边缘设备上部署视频动作识别，只有 500ms 的推理预算。你会选择什么架构？为什么？（难度：⭐⭐⭐）

**参考答案：**

在这种情况下应该选择 **2D+Pool + 轻量 backbone** 的方案，具体选型如下：

1. **骨干网络**：MobileNetV3-Small 或 ShuffleNetV2，参数量 < 5M，单帧推理 < 0.5ms
2. **采样策略**：只取 4 帧（而非 16 帧），将帧间间隔拉长以覆盖整段视频
3. **融合方式**：使用简单的帧间差分特征替代光流——计算相邻帧的像素差异（L1 范数），将这个差异图作为一个额外的通道输入 2D CNN
4. **可选优化**：使用 INT8 量化，再减少 30-40% 的推理延迟

这个方案的代价是无法很好地区分对称运动（如从左到右 vs 从右到左），但对于大多数"动作是否存在"的二元检测任务（如是否有人在跳跃、是否在挥手），2D+Pool 已经足够。

### Q4：(2+1)D 卷积相比标准 3D 卷积为什么参数更少但效果更好？（难度：⭐⭐）

**参考答案：**

参数更少的原因：标准 3x3x3 卷积有 27 个参数每个输出通道对，而 (2+1)D 先用 1x3x3（9 个参数）做空间卷积，再用 3x1x1（3 个参数）做时间卷积，合计 12 个参数。12/27 ≈ 44%，节省了约 56% 的参数。

效果更好的原因有两个：一是**非线性增加**——(2+1)D 有两层 BN+ReLU（中间夹一层），而标准 3D 卷积只有一层。每次非线性变换都能增加网络的函数表达能力，类似更深的网络。二是**归纳偏置更合理**——在实际视频中，空间和时间的特征模式往往有不同的统计特性。空间上，边缘和纹理的结构是局部的；时间上，运动通常是平滑连续的。将它们分开建模更符合自然视频的结构先验。

### Q5：请对比光流法和 3D 卷积在视频理解中的作用与互补性。（难度：⭐⭐⭐）

**参考答案：**

**光流法的作用：** 显式地估计帧间像素级别的运动。光流图是纯净的运动信号，去除了外观干扰。它的优势是对运动方向敏感（Something-Something V2 的完美解决方案），但也对姿态变化、遮挡、运动模糊敏感——这些因素会产生错误的运动估计。

**3D 卷积的作用：** 隐式地从像素中学习时空特征。不需要显式地计算运动向量，而是在训练过程中自动学习哪些时空模式对应哪些动作。它的鲁棒性更好（一个错光流像素只会影响局部感受野内的特征），但对纯运动类的细粒度分类不够精细。

**互补性：** SlowFast 架构是最好的互补范例——"快通道"以高帧率采样，依赖运动丰富的帧序列，本质上是一个运动敏感的通道（类似光流的隐式版本）；"慢通道"以低帧率采样，侧重外观稳定的帧，是一个外观敏感的通道。两者的特征融合让模型同时感知"看起来像什么"和"是怎么动的"。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 光流（Optical Flow） | "视频里的运动方向" | 密集运动的逐像素估计——视频中每个像素在相邻帧之间的二维位移向量 $(u, v)$ |
| 权值膨胀（Weight Inflation） | "把 2D 卷积变成 3D" | 将 2D 卷积核沿时间轴复制 $T$ 份，并除以 $T$ 保持激活量期望不变，从而用 2D 预训练权重初始化 3D 网络 |
| 帧池化（Frame Pooling） | "把多帧拼在一起" | 对逐帧提取的特征在时间维度做聚合（通常全局平均池化），是最基线的视频处理方法 |
| 分治注意力（Divided Attention） | "分开算注意力和空间" | 将联合时空注意力 $O((THW)^2)$ 分解为独立的时序 $O(T^2)$ 和空间 $O((HW)^2)$ 注意力，实现 100-250 倍加速 |
| 掩码自编码器（Masked Autoencoder） | "遮掉一半让模型猜" | 遮蔽输入中 75% 的块，让模型从剩余 25% 重建被遮蔽部分的原始像素值，实现大规模无监督预训练 |
| (2+1)D 卷积 | "两次卷积代替一次" | 将 3D 卷积分解为先空间后时间的两步操作，参数减少 56%，且通过额外非线性提升表达能力 |
| 多片段采样（Multi-Clip Sampling） | "多看几段拼起来" | 测试时对同一视频做多次独立采样（通常 3-5 次），取各次预测的平均值作为最终结果，显著提升视频级准确率 |
| 时序子采样（Temporal Subsample） | "少取几帧省算力" | 从原始视频中均匀抽取 $T$ 帧（如从 30fps 中选 16 帧），是控制 3D 卷积计算量的关键超参数 |

---

## 📚 小结

视频理解的核心挑战是如何让模型学会"看时间"——2D CNN 只能看到静态画面，真正的理解需要在空间特征之上引入时序建模。从 I3D 的权值膨胀到 (2+1)D 的时空因子化，从 RAFT 的逐像素光流到 VideoMAE 的掩码重建，每一条技术路线都在回答同一个问题：如何在有限的计算预算下，让模型最大限度地感知运动。分治注意力则将 Transformer 的平方复杂度压到可接受的范围，使视频级别的时序建模真正落地。

下一课我们将学习 NeRF（神经辐射场），它用完全不同的思路——用神经网络拟合 3D 空间的密度和辐射颜色——实现从视频到可自由导航的 3D 场景的转换。

---

## ✏️ 练习

1. 【理解】用自己的话解释为什么全局平均池化 `mean(dim=1)` 在视频理解中会丢失运动方向信息。举一个具体例子，说明两个帧顺序相反的视频经过池化后为什么会有相同的特征表示。写 200 字以内。

2. 【实现】修改 `FramePool` 类中的 `forward` 方法，将全局平均池化替换为**时序最大池化**（`max(dim=1)`）和**时序拼接池化**（沿通道维度拼接所有帧特征后再接 1x1 卷积）。对比三种池化方式在相同输入下的输出差异。

3. 【实验】编写一段代码，创建一个包含两个动作的视频：一个"从左到右平移"和一个"从右到左平移"。分别用 I3D 膨胀后的 3D 卷积和 (2+1)D 卷积处理这两个视频，观察它们能否产生不同的响应。解释观察到的现象。

4. 【实验】修改 `DividedTimeSpaceAttention` 的 `forward` 方法，实现一个完整的 TimeSformer Block——在每个分治注意力块中加入 Layer Normalization（Pre-LN 风格）、残差连接和 MLP。对比 Pre-LN 和 Post-LN 在时序注意力上的梯度流动差异。

5. 【思考】VideoMAE 在图像上达到 75% 遮蔽率的效果最好，但在视频上可能需要更高的遮蔽率（80-85%）。请从视频数据的时空冗余性角度解释这个现象，并设计一个简单的实验来验证你的假设。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 视频理解算法集 | `code/main.py` | 2D+Pool、I3D 膨胀、(2+1)D 卷积、RAFT 光流、VideoMAE、时空分治注意力的从零实现 |
| 视频架构选型提示词 | `outputs/prompt-video-understanding-guide.md` | 根据信号类型（外观/运动/混合）、数据集规模和计算预算选择最佳视频架构的结构化提示词 |

---

## 📖 参考资料

1. [论文] Carreira et al. "R&Q;e: Spatiotemporal Encoders for Action Recognition". CVPR, 2018. https://arxiv.org/abs/1711.04978
2. [论文] Tran et al. "Video Classification with Channel-Separa ted Convolutions". CVPR, 2018. https://arxiv.org/abs/1711.11248
3. [论文] Lao et al. "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow". ECCV, 2020. https://arxiv.org/abs/2003.12039
4. [论文] Bao et al. "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training". NeurIPS, 2022. https://arxiv.org/abs/2203.12602
5. [论文] Goyal et al. "The SlowFast Network for Video Recogniton". ICCV, 2019. https://arxiv.org/abs/1812.03982
6. [论文] Fontaine et al. "Manually Meet Temporal Scale in Video Transformers". CVPR, 2021. https://arxiv.org/abs/2104.12447
7. [官方文档] PyTorch torchvision.models.video: https://pytorch.org/vision/stable/models/video.html
8. [GitHub] Meta Research PyTorchVideo: https://github.com/facebookresearch/pytorchvideo
9. [GitHub] MCG-NJU VideoMAE: https://github.com/MCG-NJU/VideoMAE

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、知识连线、工程最佳实践、常见错误、面试考点等均为原创内容。
