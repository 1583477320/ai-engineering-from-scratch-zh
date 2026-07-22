# 语义分割 UNet：编码器-解码器与跳跃连接

> 像素级预测不是分类的简单堆叠——它要求模型同时理解"这是什么"和"它在哪儿"。语义分割的架构本质上是一个空间推理引擎。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 04 · 03（CNN 架构演进）、阶段 04 · 05（迁移学习）— 理解卷积、池化、BatchNorm 等基本操作，以及预训练模型的微调方法
**预计时间：** ~120 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 04 · 04（图像分类）— 将单张图片的标签映射推广到每个像素的标签；阶段 12 · 01（视觉语言模型）— 医学图像分割是多模态理解的核心组件

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释语义分割与图像分类的本质区别——从"整张图片属于哪类"到"每个像素属于哪类"的问题升级
- [ ] 画出 U-Net 的编码器-解码器架构，说明跳跃连接在空间重建中的作用
- [ ] 从零实现 Dice 损失函数，并解释为什么它比交叉熵更适合处理类别不平衡
- [ ] 理解 FCN 和 Atrous Convolution 如何奠定了现代语义分割的基础
- [ ] 使用 PyTorch 构建完整的语义分割训练流水线——数据生成、模型定义、损失计算、评估指标

---

## 1. 问题

图像分类的标签是**一张图一个答案**。医学影像的标签是**每一张图几百万个答案**。

假设你在读一张肺部 CT 扫描。分类模型会告诉你："这张片子有肿瘤"。但医生需要知道的是——"肿瘤边界在哪里？从第 142 层到第 189 层？它和支气管的距离是多少？手术刀应该避开哪些像素？"分类只能给出一个整数，分割才能给出一个像素级的精确轮廓。

语义分割面临的根本挑战是：**下采样丢失了空间信息。**

卷积网络的每一层池化和步长大于 1 的卷积都会让特征图的空间尺寸减半。经过 5 次下采样后，256×256 的输入变成了 8×8 的特征图。这个 8×8 的表示或许足够判断"这里有肿瘤"（分类），但绝对不够重建肿瘤的像素级轮廓（分割）。

解决方案只有一个：**把丢失的空间信息补回来**。U-Net 的做法是在编码器的每一层保留高分辨率特征图，然后通过跳跃连接直接传递给解码器——用编码器的"空间分辨率"补充解码器的"语义丰富度"。

```
没有跳跃连接的编码器-解码器：

输入 256×256
    │ (5 次 × 减半)
    ▼
  8×8 ←── 这里丢失了全部 254×254 的空间信息
    │ (5 次 × 翻倍)
    ▼
输出 256×256? ←── 8×8 的单点不可能重建 256×256 的细节

有跳跃连接：

输入 256×256                          输出 256×256
    │                                       ▲
    ├────────────── 跳跃连接 ─────────────────┤
    ▼                                       │
  128×128 → 64×64 → 32×32 → 16×16 → 8×8    │
              ↑       ↑       ↑       ↑      │
              └───────┴───────┴───────┘      │
                         拼接恢复空间细节      │
```

不做这件事，你的"分割"就是一个粗糙的彩色斑块——类别边界模糊，小目标完全消失。这是工业界部署分割模型时最常见的失败原因之一。

---

## 2. 概念

### 2.1 语义分割 vs 实例分割 vs 全景分割

语义分割只是图像分割家族的一个成员。理解它的定位很重要：

| 任务类型 | 输出 | 例子 | 区别 |
|---|---|---|---|
| **图像分类** | 整图一个标签 | "猫" | 不关心物体在哪 |
| **目标检测** | 框 + 标签 | 画框标注每只猫 | 知道位置，但不知道精确形状 |
| **语义分割** | 像素级类别标签 | 每个像素标"猫/狗/背景" | 区分类别，但不区分同类的不同个体 |
| **实例分割** | 像素级 + 实例 ID | 左边的猫 = ID-1，右边的猫 = ID-2 | 既分类别又分个体 |
| **全景分割** | 语义 + 实例 | 猫/狗（语义）+ 每张猫/狗的 ID（实例） | 统一两种范式 |

语义分割和图像分类的区别可以这样直观展示：

```
一幅包含三辆车 + 道路 + 行人的图像：

分类标签：[car, car, car, road, pedestrian] → 丢失了所有空间关系

语义分割（像素级标签）：
┌──────────────────────────────────┐
│ ░░░░░░███░░░░░░░░░░░███░░░░░░░░ │ ← ██ = 汽车, ░ = 道路
│ ░░░░░░███░░░░░░░░░░░███░░░░░░░░ │
│ ░░░░░░███░░░▓▓░░░░░░███░░▓▓▓░░ │ ← ▓ = 行人
│ ░░▓▓░░▓▓░░░░░░░░░░░░░░░░▓▓▓░░ │
│ ░░▓▓░░▓▓░░░░░░░░░░░░░░░░▓▓▓░░ │
└──────────────────────────────────┘
```

**像素精度（Pixel Accuracy）陷阱：** 在语义分割中，像素级准确率往往具有欺骗性。假设一张图中 90% 的像素是道路，只有 10% 是汽车——一个"什么都不做"的模型（把所有像素都预测为道路）就能达到 90% 的准确率，但它完全没有检测出任何车辆。这就是为什么分割任务从来不看像素准确率，只看 **IoU（交并比）**。

### 2.2 U-Net 架构——为什么叫 "U"

```mermaid
graph TD
    subgraph ENCODER["编码器（收缩路径）"]
        A[输入 572×572] --> B[DoubleConv 64]
        B --> C[MaxPool → 286×286]
        C --> D[DoubleConv 128]
        D --> E[MaxPool → 143×143]
        E --> F[DoubleConv 256]
        F --> G[MaxPool → 72×72]
        G --> H[DoubleConv 512]
        H --> I[MaxPool → 36×36]
        I --> J[DoubleConv 1024]
        style A fill:#e8f5e9,stroke:#4caf50
        style J fill:#fff3e0,stroke:#ff9800
    end

    subgraph DECODER["解码器（扩展路径）"]
        J --> K[UpSample × 2]
        K --> L[Concat + DoubleConv 512]
        L --> M[UpSample × 2]
        M --> N[Concat + DoubleConv 256]
        N --> O[UpSample × 2]
        O --> P[Concat + DoubleConv 128]
        P --> Q[UpSample × 2]
        Q --> R[Concat + DoubleConv 64]
        R --> S[1×1 Conv]
        style S fill:#e3f2fd,stroke:#2196f3
    end

    I -.skip.--> L
    G -.skip.--> P
    D -.skip.--> N
    B -.skip.--> R
    style I stroke-dasharray:5,5
    style G stroke-dasharray:5,5
    style D stroke-dasharray:5,5
    style B stroke-dasharray:5,5

    classDef encoder fill:#e8f5e9,stroke:#4caf50,color:#333
    classDef decoder fill:#e3f2fd,stroke:#2196f3,color:#333
    classDef skip stroke:#ff9800,stroke-width:3px,fill:none,stroke-dasharray:5,5
```

原始 U-Net 论文发表于 2015 年，作者是为**生物医学图像分割**而设计的。医学影像的极端稀缺催生了这个架构的特殊选择——下面详述。

### 2.3 跳跃连接的设计哲学

跳跃连接本质上是**多尺度特征融合**。它在每一层将两个不同性质的特征矩阵拼接在一起：

```
编码器第 i 层的特征图:    (batch, C_i, H_i, W_i)     → 高分辨率，低语义
解码器第 i 层的特征图:    (batch, C'_i, H'_i, W'_i)   → 低分辨率，高语义

上采样对齐后拼接:
(batch, C_i + C'_i, H_i, W_i)

编码器的特征贡献: 物体的精确边界、纹理细节
解码器的特征贡献: 物体的类别、上下文关系
```

跳跃连接的数量决定了参数的增长速度：

```
层级        通道数（编码器侧）    通道数（解码器侧，拼接前）    拼接后的通道数
L0          3                     3                          —
L1          32                    32                         —
L2          64                    128                      64 + 64 = 128
L3          128                   256                      128 + 128 = 256
L4          256                   512                      256 + 256 = 512
L5（瓶颈）   512                   1024                     512 + 512 = 1024
```

### 2.4 Dice 损失——为什么交叉熵不够

**交叉熵损失的局限：** 像素级交叉熵对每个像素独立做分类，它不关心像素之间的空间关系。在肿瘤分割这种场景中，肿瘤像素可能只占 1%，模型会把所有像素预测为背景，交叉熵依然很低（因为 99% 都是对的）。

**Dice 损失公式：**

$$
\text{Dice}(A, B) = \frac{2|A \cap B|}{|A| + |B|} = \frac{2 \sum p_i g_i}{\sum p_i + \sum g_i}
$$

其中 $p_i$ 是像素 $i$ 的预测概率，$g_i$ 是真实标签（0 或 1）。

```
Dice 评分解读：
Dice = 1.0  → 完美重叠
Dice = 0.5  → 预测区域只有一半覆盖了真实区域
Dice = 0.0  → 完全没有重叠
```

**为什么 Dice 能解决不平衡？** 因为它是基于整体区域的比例，而不是基于单个像素的计数。即使 99% 的像素是背景，只要肿瘤区域的预测重叠率高，Dice 就能反映出来。

### 2.5 FCN 与空洞卷积——U-Net 的前人铺路

在 U-Net（2015）之前，Full Convolutional Networks（FCN，2015）已经证明了"全卷积可以做分割"。但 FCN 有一个核心缺陷：它的解码器通过**池化反向操作**上采样，这种方式只利用了下采样时丢失的空间维度信息，而没有利用编码器中间层的特征。

U-Net 的改进在于：保留了编码器各层的高分辨率特征图，通过跳跃连接传递到解码器。后来 DeeperLab（2020）的实验证明：**跳跃连接比单纯的上采样操作带来的提升大得多**。

**空洞卷积（Atrous / Dilated Convolution）** 则是另一种保持分辨率的手段——在不降低特征图尺寸的前提下扩大感受野：

```
标准 3x3 卷积的感受野：
┌───┬───┬───┐
│ × × × │  3×3 = 9 个位置
└───┴───┴───┘

空洞卷积（dilation_rate=2）的感受野：
┌ × ─ × ─ × ┐
│             │  等效 7×7，但实际只有 9 个参数
└ × ─ × ─ × ┘
        ×

空洞卷积（dilation_rate=4）的感受野：
┌ × ─ ─ ─ × ─ ─ ─ × ┐
│                     │  等效 15×15，依然只有 9 个参数
└ × ─ ─ ─ × ─ ─ ─ × ┘
                ×
```

---

## 3. 从零实现

### 第 1 步：DoubleConv——基本卷积块

一切从最基本的构建块开始。

```python
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """两个连续的 3×3 卷积 + BN + ReLU。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
```

`padding=1` 保证卷积输出的空间尺寸不变。`inplace=True` 节省显存——ReLU 不修改原位值也能得到正确结果，且不需要额外存储输入图的副本。

### 第 2 步：Encoder——编码路径

```python
class Down(nn.Module):
    """下采样 = 最大池化 + 双卷积。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)
```

### 第 3 步：Decoder——解码路径 + 跳跃连接

```python
class Up(nn.Module):
    """上采样 + 跳跃连接拼接 + 双卷积。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # bilinear + conv 是工业界首选，避免转置卷积的棋盘效应
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_encoder, skip):
        x = self.up(x_encoder)
        # 尺寸对齐处理
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        # 通道维拼接（dim=1）
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
```

### 第 4 步：完整 U-Net

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_channels=32):
        super().__init__()
        # 编码器
        self.inc = DoubleConv(in_channels, base_channels)
        self.d1 = Down(base_channels, base_channels * 2)
        self.d2 = Down(base_channels * 2, base_channels * 4)
        self.d3 = Down(base_channels * 4, base_channels * 8)
        self.d4 = Down(base_channels * 8, base_channels * 16)
        # 解码器
        self.u1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.u2 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.u3 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.u4 = Up(base_channels * 2 + base_channels, base_channels)
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)
```

### 第 5 步：验证架构

```python
model = UNet(in_channels=3, num_classes=3, base_channels=16)
dummy = torch.randn(1, 3, 64, 64)
output = model(dummy)

print(f"输出形状: {output.shape}")  # (1, 3, 64, 64) — (batch, classes, height, width)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

输出 `(1, 3, 64, 64)` 的含义：每个像素对应 3 个未归一化的 logits（背景、圆形、正方形各一个）。Softmax 会在损失计算时自动完成归一化。

---

## 4. 工业工具

### 4.1 MONAI——医学影像分割的标准库

在医学影像领域，PyTorch 生态中最广泛使用的分割框架是 **MONAI**（Medical Open Network for AI）。它在 U-Net 的基础上提供了更丰富的功能：

```python
from monai.networks.nets import UNet as MonaiUNet
import torch

# MONAI 的 U-Net 支持更灵活的配置
model = MonaiUNet(
    spatial_dims=2,           # 2D（3D 用于体数据）
    in_channels=3,            # RGB 输入
    out_channels=3,           # 3 个类别
    channels=(16, 32, 64, 128),  # 每层通道数
    strides=(2, 2, 2),            # 每层下采样倍数
)

dummy = torch.randn(2, 3, 64, 64)
output = model(dummy)
print(f"MONAI 输出形状: {output.shape}")  # (2, 3, 64, 64)
print(f"MONAI 参数量: {sum(p.numel() for p in model.parameters()):,}")
```

MONAI 还内置了医学影像专用的数据集加载器、数据增强管道（随机旋转、弹性形变、强度偏移等）和评估指标。如果你在做医学影像项目，MONAI 是最快上手的选择。

### 4.2 torchvision 中的分割模型

```python
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50, lraspp_mobilenet_v3_large

# FCN（残差网络骨干）
fcn = fcn_resnet50(weights="DEFAULT")
print(f"FCN-ResNet50 参数量: {sum(p.numel() for p in fcn.parameters()):,}")

# DeepLabV3（带空洞空间金字塔池化 ASPP）
deeplab = deeplabv3_resnet50(weights="DEFAULT")
print(f"DeepLabV3-ResNet50 参数量: {sum(p.numel() for p in deeplab.parameters()):,}")

# MobileOne-Seg（轻量级，适合移动端部署）
mobilenet_seg = lraspp_mobilenet_v3_large(weights="DEFAULT")
print(f"MobileNetSeg 参数量: {sum(p.numel() for p in mobilenet_seg.parameters()):,}")
```

### 4.3 HuggingFace Segment Anything

Meta 的 SAM（Segment Anything Model）是 2023 年的里程碑式工作。虽然它不是一个传统意义上的 U-Net，但其**提示式分割**的思想正在改变分割任务的范式：

```python
from transformers import AutoImageProcessor, AutoModelForImageSegmentation

# HuggingFace 上的 SAM 适配版
processor = AutoImageProcessor.from_pretrained("facebook/sam-vit-base")
model = AutoModelForImageSegmentation.from_pretrained("facebook/sam-vit-base")

# 任意图像、任意数量的提示点，都可以得到精细的掩码
```

### 4.4 性能对比

| 方案 | 参数量 | 推理速度（RTX 3090） | 适用场景 |
|---|---|---|---|
| 教学版 U-Net（base=32）| ~31M | ~15ms (64×64) | 学习理解 |
| MONAI UNet（channels=16-64-128）| ~10M | ~3ms (64×64) | 医学影像快速原型 |
| FCN ResNet-50 | 38.9M | ~8ms (512×512) | COCO 全景分割 |
| DeepLabV3+ | 35M | ~15ms (512×512) | 高精度分割 |
| MobileOne-Seg | ~5M | ~3ms (256×256) | 手机端 |

---

## 5. 知识连线

本课学习的编码器-解码器架构，是后续多门课程的基础：

- **阶段 07 · 02（自注意力从零）**：Transformer 架构中的编码器-解码器设计直接借鉴了 U-Net 的思路——编码器浓缩信息，解码器逐步展开——但 Transformer 用的是注意力而非卷积来传递信息
- **阶段 12 · 01（视觉语言模型）**：CLIP 和 GPT-4V 等多模态模型的视觉编码器建立在 CNN 骨干网络之上，理解 U-Net 的编码过程有助于理解视觉特征的抽象层次
- **阶段 17 · 02（模型优化与部署）**：分割模型的量化和剪枝面临特殊挑战——输出通道数和空间分辨率的匹配，需要在部署优化时特别注意

---

## 6. 工程最佳实践

### 6.1 上采样方式的选择

| 方法 | 优点 | 缺点 | 推荐场景 |
|---|---|---|---|
| 双线性插值 + 卷积 | 无棋盘效应，速度快 | 需要额外的卷积层做变换 | 绝大多数场景（默认选择） |
| 转置卷积 | 可学习上采样核 | 容易产生命名"checkerboard artifacts" | 仅在仔细调参时使用 |
| PixelShuffle | 无棋盘效应 | 计算量较大 | 超分辨率重建 |

```python
# ✓ 推荐：upsample + conv（无棋盘效应）
up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
conv = nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1)

# ✗ 不推荐：未调参的 TransposedConv2d
up = nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2)
# 结果：周期性棋盘效应，尤其当 kernel_size 不被 stride 整除时
```

### 6.2 医学影像分割特别建议

- **3D 体数据**使用 3D U-Net：将 2D 卷积替换为 `nn.Conv3d`，池化替换为 `nn.MaxPool3d`，输入形状为 $(B, C, D, H, W)$
- **数据增强必须保真**：医学图像的旋转、翻转、弹性形变不能改变病理结构的解剖合理性——水平翻转对面部 CT 可能意味着左右器官位置互换，这在诊断上是错误的
- **使用混合精度训练**：医学体数据通常很大（512×512×500 层以上 FP32 需要数 GB），FP16 可将内存占用减半且几乎不损失精度

### 6.3 中文场景特别建议

- 国内医院的数据格式多样化（DICOM、NIfTI、PNG 序列），推荐使用 `nibabel` 读取 NIfTI 格式（神经影像主流格式）
- 肺结节分割建议使用 LISS（肺癌筛查数据集）或 LIDC-IDRI 数据集作为起步基准
- 皮肤镜图像分割（黑色素瘤检测）推荐使用 ISIC Challenge 数据集，注意这些图像的颜色分布差异大，需要颜色增强

### 6.4 踩坑经验

- **不要只用交叉熵训练分割模型**：在类别极度不平衡时，交叉熵会把大部分权重分配给背景像素。加上 Dice 损失可以把优化目标直接对准分割质量——这是医学影像分割的黄金组合
- **跳层连接时通道数会翻倍**：拼接后通道数变为两倍，记得在上半部分的 `DoubleConv` 中正确设置 `in_channels`，否则维度错误会在运行时报错
- **`align_corners=False` 是默认安全选择**：除非你的上采样尺寸正好是下采样尺寸的精确倍数，否则设为 `False` 可以避免边缘像素的位置偏差

---

## 7. 常见错误

### 错误 1：跳跃连接的通道数搞错

**现象：** 运行时报错 `RuntimeError: size mismatch, m1: [8, 256, 4, 4], m2: [8, 512, 8, 8]`。

**原因：** 解码器的 `Up` 模块接收了编码器某一层的跳过特征（`skip`），但 `DoubleConv` 的输入通道数没有正确设置为 `encoder_channels + decoder_channels`。最常见的问题是通道翻倍后的 `in_channels` 忘记乘以 2。

**修复：**

```python
# ✗ 错误：没有考虑拼接后的通道翻倍
self.u1 = Up(base_channels * 16, base_channels * 8)

# ✓ 正确：拼接后编码器贡献 base_channels * 8，解码器贡献 base_channels * 16
self.u1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8)
```

### 错误 2：用像素准确率评估分割模型

**现象：** 模型声称准确率达到 97%，但肉眼可见的输出完全是错误的。

**原因：** 在肿瘤占 1% 像素的医学图像上，全预测为背景的分类准确率就是 99%。像素准确率对不平衡数据毫无意义。

**修复：**

```python
# ✗ 错误：像素准确率
preds = logits.argmax(dim=1)
accuracy = (preds == targets).float().mean()

# ✓ 正确：平均 IoU + 每类 IoU
from code.main import iou_per_class
ious = iou_per_class(logits, targets, num_classes=3)
mIoU = ious.nan_to_num(0).mean().item()
```

### 错误 3：在医学图像上使用随机的颜色增强

**现象：** 训练集表现优异，但在真实 DICOM 图像上预测结果一团糟。

**原因：** ColorJitter 对 RGB 照片没问题，但 DICOM 灰度图像的像素值是物理量（CT 值的 Hounsfield Unit 或 MRI 的信号强度）。随机改变亮度和对比度会扭曲真实的解剖学特征。

**修复：**

```python
# ✗ 错误：对所有医疗图像使用颜色增强
transform = transforms.ColorJitter(brightness=0.2, contrast=0.2)

# ✓ 正确：对灰度影像仅做几何变换，不做颜色变换
transform = transforms.Compose([
    transforms.RandomRotation(15),           # 允许小幅旋转
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 小范围平移
    # 不加 ColorJitter！
])
```

### 错误 4：损失函数中忘记加 epsilon

**现象：** 训练时 loss 变为 NaN，尤其是小目标分割（如微小结节）。

**原因：** Dice 损失公式的分母是 $\sum p_i + \sum g_i$。如果某个目标完全不出现（例如某类别在批次中全部为 0），分母为零导致梯度爆炸。

**修复：**

```python
# ✗ 错误：没有 epsilon
dice = 2 * intersection / total  # total 可能为 0 → NaN!

# ✓ 正确：加一个小常数
eps = 1e-6
dice = (2.0 * intersection + eps) / (total + eps)
```

---

## 8. 面试考点

### Q1：U-Net 的跳跃连接中为什么使用"拼接"而不是"相加"？（难度：⭐⭐）

**参考答案：**

拼接（concatenation）和相加（addition）的根本区别在于信息的传递方式。

相加要求两个张量维度完全相同——这意味着编码器和对应解码器层的通道数必须一致。但在 U-Net 中，编码器逐层加倍通道数，解码器逐层减半，同一层的通道数本来就不同，无法直接相加。

更重要的是，拼接实现了**信息无损传递**：编码器原始特征和解码器上采样特征在通道维度上各占一半，解码器可以自主决定使用多少来自编码器的信息。相加会将两层特征强制混合，可能导致编码器的空间细节被解码器的低语义特征稀释。

### Q2：Dice 损失和交叉熵损失各有什么优缺点？为什么要结合使用？（难度：⭐⭐⭐）

**参考答案：**

交叉熵的优势是稳定、逐像素独立、有明确的概率解释；缺点是它对类别不平衡不鲁棒，在肿瘤像素仅占 1% 时，模型几乎总是预测为背景即可得分。

Dice 的优势是直接优化重叠率指标，天生处理类别不平衡；缺点是不稳定——在小批量且目标占比极小时梯度噪声大，还可能产生局部最优。

结合使用的直觉：交叉熵提供每个像素的精细分类信号，把全局结构稳定下来；Dice 提供区域级别的全局信号，把目标的整体形状拉向正确方向。这就是为什么 `CE + Dice` 是医学分割的事实标准损失。

### Q3：为什么转置卷积容易产生棋盘效应？怎么避免？（难度：⭐⭐⭐）

**参考答案：**

转置卷积本质上是对输入做填充后滑动卷积——当卷积核大小不能被步长整除时（或更一般地，当感受野在输出网格上分布不均匀时），某些输出像素会接收到更多卷积核的重叠，某些则较少。这种周期性的不均匀覆盖就是棋盘效应。

三种避免方式：
1. 使用 `kernel_size = stride * n`（如 4×4 核 + stride 2），使重叠均匀
2. 用双线性上采样 + 普通卷积替代（最常用）
3. 调整 kernel_size 和 stride 的组合，使输出尺寸满足整除关系

工业界的默认选择是第 2 种——上采样用固定的双线性插值（不含可学习参数），再用 3×3 卷积学习空间变换。

### Q4：给定一个 512×512 的 U-Net（base_channels=32），粗略估算其参数量。推理一张图的显存消耗大约是多少？（难度：⭐⭐⭐）

**参考答案：**

编码器通道：32 → 64 → 128 → 256 → 512。
每一层的 DoubleConv 有 $2 \times (C_{in} \cdot C_{out} \cdot 3^2 + C_{out} \cdot 2) \approx 18 \cdot C_{in} \cdot C_{out}$ 个参数（卷积核 + BN 参数）。

- inc: $18 \times 3 \times 32 = 1,728$
- d1: $18 \times 32 \times 64 = 36,864$
- d2: $18 \times 64 \times 128 = 147,456$
- d3: $18 \times 128 \times 256 = 589,824$
- d4: $18 \times 256 \times 512 = 2,359,296$

解码器同理加上跳跃拼接的维度倍增。总计约 3,200 万参数——即约 128 MB（FP32）。

推理时显存消耗主要来自**激活值缓存**（反向传播需要保留），约数百 MB 到数 GB，取决于 batch size 和输入尺寸。这也说明了为什么分割模型的显存消耗远高于同等深度的分类模型。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 语义分割（Semantic Segmentation） | "给图片里的东西标颜色" | 为图像中每个像素分配一个类别标签。同一类别的不同个体（比如两辆车）不分开标记 |
| 编码器-解码器（Encoder-Decoder） | "先压缩再解压" | 编码器逐步降低空间分辨率、增加通道数以提取高级语义；解码器逐步恢复空间分辨率并生成预测 |
| 跳跃连接（Skip Connection） | "把前面的特征传给后面" | 将编码器的各层特征图直接拼接（而非相加）到解码器的对应层，用于恢复丢失的空间细节 |
| 交并比（IoU / Jaccard） | "重合多少" | 预测区域与真实区域的交集除以并集。$IoU = \frac{|A \cap B|}{|A \cup B|}$，取值 0~1，越接近 1 越好 |
| Dice 系数 | "重叠面积占总面积的几分之几" | $Dice = \frac{2|A \cap B|}{|A| + |B|}$。与 IoU 数学等价，但在形式上对分子分母的归一化方式不同 |
| 空洞卷积（Atrous Convolution） | "带空隙的卷积" | 在标准卷积核之间插入空隙（dilation rate > 1），在不减少特征图分辨率的前提下增大感受野 |
| 转置卷积（Transposed Convolution） | "反着做的卷积" | 也叫"分数步长卷积"，用于可学习的上采样操作。容易产生棋盘效应 |
| 标注（Annotation / Labeling） | "手动涂色" | 在医学影像中由放射科医生逐像素勾勒病灶轮廓。这是分割任务最大的瓶颈——高质量标注需要专业医学人员耗时数十小时 |

---

## 📚 小结

语义分割将图像分类从"一张图一个标签"升级为"每个像素一个标签"，核心挑战是**下采样过程中的空间信息丢失**。U-Net 通过编码器-解码器结构和跳跃连接解决了这一问题——编码器捕获语义，解码器恢复空间分辨率，跳跃连接提供空间细节。Dice 损失配合交叉熵是处理类别不平衡的黄金组合。

下一课我们将转向更加现代的分割范式：实例分割和全景分割，让模型不仅知道"这是什么"，还能区分"这是不同的个体"。

---

## ✏️ 练习

1. 【理解】用自己的话解释"为什么跳��连接用拼接而不用相加"。画一个 4 层的编码器-解码器示意图，标注每一层的通道数变化，说明跳跃连接如何帮助恢复空间信息。

2. 【实现】修改 `code/main.py` 中的 U-Net，将 base_channels 从 16 改为 32，计算新的参数量。比较两者在相同轮次（8 轮）下的收敛速度和最终 IoU 差异。

3. 【实验】分别用纯交叉熵损失、纯 Dice 损失、CE+Dice 组合训练同一个模型，在验证集上画出三条 Loss 曲线和对应的 IoU 曲线，分析三种损失函数的收敛行为和稳定性差异。

4. 【思考】U-Net 是为 2D 图像设计的。如果要把它扩展到 3D 医学影像（如 CT 体数据），需要做哪些改动？（提示：Conv2d → Conv3d、pooling 的维度、内存消耗的变化）

5. 【设计】假设你要为一个车牌分割任务设计 U-Net：输入是 1024×512 的道路监控图像，需要分割出背景、车牌文字、车牌边框三个类别。你会如何调整 U-Net 的网络深度和通道数？为什么？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 完整 U-Net 实现 | `code/main.py` | 编码器-解码器 + 跳跃连接 + Dice 损失 + 合成数据训练流水线 |
| 语义分割指导提示词 | `outputs/prompt-semantic-segmentation-guide.md` | 用于分析和设计任意语义分割任务的提示词模板 |

---

## 📖 参考资料

1. [论文] Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation". MICCAI, 2015. https://arxiv.org/abs/1505.04597
2. [论文] Long et al. "Fully Convolutional Networks for Semantic Segmentation". CVPR, 2015. https://arxiv.org/abs/1411.4038
3. [论文] Chen et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation". ECCV, 2018. https://arxiv.org/abs/1802.02611
4. [官方文档] PyTorch. "torchvision.models.segmentation": https://pytorch.org/vision/stable/models.html#segmentation
5. [GitHub] Project-MONAI/MONAI: https://github.com/Project-MONAI/MONAI
6. [论文] Cheng et al. "DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation". ECCV, 2018. https://arxiv.org/abs/1802.02611
7. [论文] Kirillov et al. "Segment Anything". ICCV, 2023. https://arxiv.org/abs/2304.02643

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、知识连线、工程最佳实践、常见错误、面试考点等均为原创内容。
