# CNN 架构演进

> 深度学习的每一次突破，本质上都是让梯度流过更深的网络——LeNet 到 ResNet 的演进史，就是一部梯度消失的抗争史。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 03 · 01-05（感知机、多层网络、反向传播、激活函数、损失函数）— 理解神经网络的基本结构和训练原理
**预计时间：** ~90 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 03 · 03（反向传播）— 理解梯度消失是本课所有架构演进的根本驱动力；阶段 05 · 02（词嵌入）— CNN 的特征提取思想与 Transformer 的嵌入层有深层联系

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 说出 6 种经典 CNN 架构（LeNet、AlexNet、VGG、GoogLeNet、ResNet、DenseNet）的核心设计思想和参数量
- [ ] 解释残差连接（y = F(x) + x）为什么能让 100+ 层的网络稳定训练，以及捷径路径在维度不匹配时如何处理
- [ ] 对比 VGG 的"统一小卷积核堆叠"与 GoogLeNet 的"并行多尺度卷积"两种设计哲学的权衡
- [ ] 用 PyTorch 实现 BasicBlock 残差块和 Inception 模块，并分析各自的参数分布
- [ ] 根据任务需求和计算预算，选择合适的 CNN 架构作为骨干网络

---

## 1. 问题

你已经知道卷积层可以提取图像特征，全连接层可以做分类。用几个卷积层叠起来，理论上就能做图像识别。

但实际做起来，问题来了：**网络越深，训练越难**。

1998 年的 LeNet-5 只有 5 层，手写数字识别效果不错。2012 年之前的实验尝试把网络加深到 20 层以上，发现了一个反直觉的现象——**训练损失反而上升了**。更深的网络理应有更大的容量，为什么训练效果反而变差？这不是过拟合（训练损失上升，而不是验证损失上升），而是优化器根本无法有效训练深层网络。

问题出在梯度上。反向传播时，梯度从输出层向输入层逐层传递，每经过一层都会乘以该层的导数。当导数小于 1 时，多层连乘后梯度指数级缩小——靠近输入的层几乎收不到任何梯度信号，参数无法更新。这就是**梯度消失问题**。

从 LeNet 到 ResNet 的演进史，本质上就是一部工程师不断找到新方法来解决梯度消失、让更深的网络可以被训练的历史。理解这条演进线索，你就能理解为什么现代视觉模型长成现在这个样子。

---

## 2. 概念

### 2.1 演进时间线

```
1998        2012        2014        2014         2015         2017
 │           │           │           │            │            │
 LeNet    AlexNet      VGG     GoogLeNet      ResNet     DenseNet
 5 层      8 层        19 层     22 层         152 层       121 层
 ~60K     ~61M        ~138M     ~6.8M        ~25.6M       ~8M
 tanh      ReLU       3×3小核   多尺度并行    残差连接     密集连接
```

### 2.2 LeNet-5（1998）— 一切的起点

LeNet-5 是 Yann LeCun 等人提出的第一个在工业界成功应用的卷积神经网络，用于手写邮政编码识别。

**结构特点：**
- 两层卷积（5x5 核）→ 两层全连接
- 使用平均池化（不是最大池化）
- 使用 tanh 激活函数

```
输入 32×32 灰度图
    │
    ▼
卷积层1 (5×5, 6个) → tanh → 平均池化     → 14×14×6
    │
    ▼
卷积层2 (5×5, 16个) → tanh → 平均池化    → 5×5×16
    │
    ▼
全连接 400→120 → tanh
全连接 120→84  → tanh
全连接 84→10   → 输出（10个数字类别）
```

**参数量：** 约 60,000。

**历史意义：** LeNet 证明了一件事——神经网络可以端到端地从像素中学习特征，不需要手工设计特征提取器。但受限于当时的硬件和数据量，更深的网络无法训练。

### 2.3 AlexNet（2012）— GPU 时代的开端

AlexNet 在 2012 年的 ImageNet 竞赛中以压倒性优势获胜（错误率从 26% 降到 16%），标志着深度学习时代的正式开始。

**核心创新：**

| 创新 | 作用 | 影响 |
|---|---|---|
| ReLU 激活函数 | 替代 tanh/sigmoid | 训练速度提升约 6 倍 |
| Dropout | 随机丢弃神经元 | 有效缓解过拟合 |
| GPU 训练 | 用两块 GTX 580 并行训练 | 让大规模训练成为可能 |
| 数据增强 | 随机裁剪、水平翻转 | 提升泛化能力 |
| 局部响应归一化 | 类似生物神经元的侧抑制 | 后被批归一化取代 |

**关键设计选择——为什么 ReLU 如此重要？**

Tanh 的导数最大值只有 1.0（在零点附近），大部分区域导数接近 0。经过多层连乘，梯度迅速消失。ReLU 的导数在正区间恒为 1——梯度要么完全消失（负区间），要么完整传递（正区间）。这个简单的变化让梯度可以穿过更多层，直接解锁了更深网络的训练可能性。

### 2.4 VGG（2014）— 统一的建筑哲学

VGG 的核心思想可以用一句话概括：**整只网络只用一种零件——3x3 卷积**。

**两个 3x3 堆叠 vs 一个 5x5：**

```
感受野对比：

一个 5x5 卷积的感受野：
┌─────────────┐
│             │  5×5 = 25 个权重
│             │  25C² 参数
│             │  1 次非线性变换
└─────────────┘

两个 3x3 卷积的感受野：
┌───────────┐
│           │  3×3 = 9 个权重
│    ┌───┐  │  9+9 = 18C² 参数
│    │ 3 │  │  2 次非线性变换
└────└───┘──┘
```

参数量：18C² vs 25C²（节省 28%），非线性变换多了一次（表达能力更强）。

**VGG 的参数量灾难：** VGG-16 有约 1.38 亿参数，其中大部分来自最后三个全连接层（约 1.03 亿）。这个设计后来被 ResNet 的全局平均池化取代。

### 2.5 GoogLeNet / Inception（2014）— 让网络自己选择

GoogLeNet 引入了 Inception 模块，核心问题是：**卷积核尺寸该怎么选？** 3x3 捕获小尺度特征，5x5 捕获大尺度特征——与其猜，不如让网络自己决定。

**Inception 模块结构：**

```
输入特征图
    │
    ├── 1x1 卷积 (64) ──────────────────────┐
    │                                        │
    ├── 1x1 卷积 (96) → 3x3 卷积 (128) ────┤
    │                                        ├── 拼接（通道维度）
    ├── 1x1 卷积 (16) → 5x5 卷积 (32) ─────┤
    │                                        │
    └── 3x3 最大池化 → 1x1 卷积 (32) ──────┘
```

**1x1 卷积的妙用：** 在大卷积核之前先用 1x1 卷积降维（如 192 通道 → 16 通道），再做 5x5 卷积。这在不损失感受野的前提下，大幅减少了计算量。

**参数量：** GoogLeNet 只有约 680 万参数——比 VGG-16 少了 20 倍，但 ImageNet 上的准确率相当。

### 2.6 ResNet（2015）— 残差连接的革命

ResNet 解决了困扰深度学习社区的核心问题：**为什么更深的网络训练损失反而更高？**

**残差学习：**

传统网络学习目标映射 H(x)。ResNet 让网络学习残差 F(x) = H(x) - x，最终输出：

$$y = F(x) + x$$

```
残差块：

    x ───────────────────────→ ┌───┐
    │                           │ + │ → ReLU → 输出
    │   ┌──────┐  ┌──────┐     └───┘
    └──→│ 3x3  │→ │ 3x3  │─────┘
        │  BN  │  │  BN  │
        └──────┘  └──────┘
          主路径（学习残差）
```

**为什么这解决了退化问题？**

如果最优映射接近恒等变换（即 H(x) ≈ x），传统网络需要学习一个接近 0 的映射——优化器很难做到。残差网络只需要让 F(x) → 0，这是最容易学的函数之一（权重接近零即可）。恒等映射变成了默认行为，网络不需要额外努力。

**捷径路径（Shortcut Connection）：**

当输入和输出维度不匹配时（通道数变化或空间尺寸变化），捷径路径使用 1x1 卷积对齐维度：

```
输入 (32 通道, 32×32)
    │
    ├── 3x3 Conv(stride=2) → BN → ReLU → 3x3 Conv → BN   → (64 通道, 16×16)
    │
    └── 1x1 Conv(stride=2) → BN                             → (64 通道, 16×16)
                                                                         │
                                                                    相加 → ReLU
```

**参数效率对比：**

| 模型 | 参数量 | Top-5 错误率 |
|---|---|---|
| VGG-16 | 1.38 亿 | 7.3% |
| GoogLeNet | 680 万 | 6.7% |
| ResNet-50 | 2560 万 | 5.3% |
| ResNet-152 | 6020 万 | 4.5% |

ResNet 可以训练到 152 层甚至 1000+ 层，且每增加一层都在持续提升性能。

### 2.7 DenseNet（2017）— 密集连接与特征复用

DenseNet 将 ResNet 的思想推向极致：**每一层都与前面所有层直接相连**。

```
ResNet 连接方式（加法）：
层1 → 层2 → 层3 → ...
       │          │
       └──────────┘ (残差加法)

DenseNet 连接方式（拼接）：
层1 → 层2 → 层3 → ...
  │     │     │
  └─────┴─────┘ (特征拼接，每层接收前面所有层的输出)
```

**密集连接的优势：**
- 特征复用：前面层的特征直接传递到后面每一层，无需重新学习
- 梯度直通路径：每一层的梯度可以直接流向任意前面的层
- 参数效率：由于特征复用，每层只需要学习新的少量特征（growth_rate，通常 12-32）

**代价：** 通道数随层数线性增长，内存消耗大。DenseNet 通过过渡层（Transition Layer，1x1 卷积 + 平均池化）来控制通道数增长。

---

## 3. 从零实现

### 第 1 步：LeNet-5——最简单的卷积网络

```python
import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # 灰度输入，6 个 5x5 卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # 6→16 通道
        self.pool = nn.AvgPool2d(2)                     # 2x2 平均池化
        self.fc1 = nn.Linear(16 * 5 * 5, 120)          # 展平后全连接
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))   # tanh 激活（AlexNet 之前的标准）
        x = self.pool(torch.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)                     # 展平为一维向量
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)
```

LeNet 只有约 60K 参数。5 层结构——这就是 1998 年的全部。

### 第 2 步：VGG Block——用 3x3 卷积构建统一模块

```python
import torch.nn.functional as F


class VGGBlock(nn.Module):
    """VGG 的标准构建块：两层 3x3 卷积 + 最大池化。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x)
```

每个 VGGBlock 只做一件事：两层 3x3 卷积 + 池化。VGG 的全部哲学就浓缩在这 15 行代码里。

### 第 3 步：Inception 模块——并行多尺度

```python
class InceptionModule(nn.Module):
    """四条并行路径：1x1、3x3、5x5 卷积 + 池化投影。"""
    def __init__(self, in_ch, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()
        # 1x1 卷积路径
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1), nn.ReLU(inplace=True),
        )
        # 1x1 降维 + 3x3 卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce), nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3), nn.ReLU(inplace=True),
        )
        # 1x1 降维 + 5x5 卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce), nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5), nn.ReLU(inplace=True),
        )
        # 池化 + 1x1 投影
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 四条路径的输出在通道维度拼接
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)
```

关键洞察：1x1 卷积在这里扮演"瓶颈层"的角色——先将通道数从 192 降到 16，再做计算量大的 5x5 卷积，总计算量大幅降低。

### 第 4 步：残差块——深度网络的核心构件

```python
class BasicBlock(nn.Module):
    """ResNet 的基本残差块。"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主路径：两层 3x3 卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径路径：维度不匹配时用 1x1 卷积对齐
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # 残差连接：输出 = 主路径 + 捷径
        return F.relu(out)
```

注意两个关键设计决策：
- `bias=False`：紧跟 BatchNorm 的卷积层不需要偏置，BN 的 beta 已经参数化了偏置
- 残差加法在 ReLU 之前：`out = self.bn2(self.conv2(out)); out = out + shortcut(x); return F.relu(out)`

### 第 5 步：在真实数据上验证

完整的 `code/main.py` 包含了所有 6 种架构的实现，并在 32x32 输入上对比了参数量和前向传播行为。运行 `python code/main.py` 即可查看对比结果。

---

## 4. 工业工具

### 4.1 PyTorch 内置实现

PyTorch 提供了现成的 ResNet 模型，可以直接加载预训练权重：

```python
import torch
from torchvision import models

# 加载预训练的 ResNet-50（ImageNet 上训练）
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# 查看模型结构
print(model)

# 前向传播验证
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(f"输出形状: {output.shape}")  # (1, 1000) — 1000 个 ImageNet 类别
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

### 4.2 timm 库——视觉模型的瑞士军刀

timm（PyTorch Image Models）提供了数百种预训练视觉模型，是计算机视觉工程师的标配工具：

```python
import timm

# 列出所有可用的 ResNet 变体
resnet_models = timm.list_models("resnet*")
print(f"ResNet 变体数量: {len(resnet_models)}")
for name in resnet_models[:10]:
    print(f"  {name}")

# 加载模型（自动下载预训练权重）
model = timm.create_model("resnet50", pretrained=True, num_classes=10)
print(f"分类头输出维度: {model.num_features}")  # 2048

# 也可以用作特征提取器（去掉分类头）
feature_extractor = timm.create_model("resnet50", pretrained=True, num_classes=0)
features = feature_extractor(torch.randn(1, 3, 224, 224))
print(f"特征维度: {features.shape}")  # (1, 2048)
```

### 4.3 HuggingFace Transformers——视觉模型的统一接口

```python
from transformers import AutoModel, AutoImageProcessor

# 使用 ViT（Vision Transformer）——CNN 之后的新一代架构
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224")

# 查看参数量
print(f"ViT-Base 参数量: {sum(p.numel() for p in model.parameters()):,}")
# ViT-Base: ~86M 参数，与 ResNet-50（~25M）在同一量级
```

### 4.4 性能对比

| 模型 | 参数量 | ImageNet Top-1 | 推理延迟（T4） | 适用场景 |
|---|---|---|---|---|
| MobileNetV3-Small | 2.5M | 67.5% | ~1ms | 手机端实时推理 |
| ResNet-18 | 11.7M | 69.8% | ~2ms | 轻量级服务器推理 |
| ResNet-50 | 25.6M | 76.1% | ~4ms | 通用视觉骨干 |
| EfficientNet-B0 | 5.3M | 77.1% | ~3ms | 精度/效率平衡 |
| ConvNeXt-B | 89M | 83.8% | ~12ms | 高精度场景 |
| ViT-B/16 | 86M | 77.9% | ~8ms | 大数据集预训练 |

---

## 5. 知识连线

本课学习的 CNN 架构演进，是后续多门课程的基础：

- **阶段 07 · 02（自注意力从零）**：Transformer 的自注意力可以看作一种"可学习的卷积"——理解卷积的局限性，你就能理解为什么 Transformer 要用全局注意力替代局部卷积
- **阶段 12 · 01（视觉语言模型）**：CLIP、GPT-4V 等视觉语言模型的视觉编码器（ViT、SigLIP）都建立在 CNN 骨干网络的基础上，理解 ResNet 的设计思想对于理解视觉编码器至关重要
- **阶段 17 · 02（模型优化与部署）**：模型量化、蒸馏、剪枝等部署优化技术，首先需要理解各层的计算量分布——VGG 的全连接层参数浪费、ResNet 的参数效率对比都是实际部署中的经典案例

---

## 6. 工程最佳实践

### 6.1 架构选型速查

| 场景 | 推荐架构 | 理由 |
|---|---|---|
| 学习/实验 | ResNet-18 | 参数量小（11.7M），训练快，社区支持好 |
| 工业分类 | ResNet-50 | 预训练权重丰富，精度和速度平衡 |
| 嵌入式/手机 | MobileNetV3 | 深度可分离卷积，参数量 < 5M |
| 高精度需求 | ConvNeXt-B/L | CNN 架构，Transformer 级别的精度 |
| 图像分割 | ResNet-50 + FPN | 多尺度特征提取能力强 |

### 6.2 残差连接的实现规范

```python
# ✓ 标准写法：先加后激活
out = self.conv2(self.bn2(F.relu(self.bn1(self.conv1(x)))))
out = out + self.shortcut(x)  # 加法在 ReLU 之前
return F.relu(out)

# ✗ 错误写法：先激活后加（信息不对称）
out = F.relu(self.conv2(...))  # 主路径已经 ReLU 过了
out = out + self.shortcut(x)   # 捷径还是原始值，没有 ReLU
return out                      # 不对称的激活范围
```

### 6.3 迁移学习最佳实践

```python
import torch.nn as nn
from torchvision import models

# 加载预训练 ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# 冻结所有卷积层（只训练分类头）
for param in model.parameters():
    param.requires_grad = False

# 替换分类头（适配你的任务）
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 只有分类头的参数会被训练
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数: {trainable:,}")
```

### 6.4 中文场景特别建议

- 中文 OCR 任务推荐使用 CRNN（CNN + BiLSTM）架构，而不是纯 CNN——中文字符数量大（6000+ 常用字），序列建模能力很重要
- 医学影像（如 CT、MRI）推荐使用预训练的 ResNet-50 或 EfficientNet 作为骨干，再微调——医学影像数据量通常不大，从零训练效果差
- 工业缺陷检测通常使用 ResNet-18 或 MobileNet 作为骨干，配合异常检测头——实时性要求高，模型不能太大

### 6.5 踩坑经验

- 不要在 ResNet 残差块的捷径路径上使用 `bias=True`——紧接 BN 的卷积不需要偏置，浪费参数且可能影响收敛
- 使用预训练权重时，注意 `num_classes` 必须和预训练时一致——如果你的 `model.fc` 输出维度不是 1000，需要从头训练分类头
- DenseNet 的内存消耗随层数线性增长——121 层 DenseNet 的内存占用可能超过 ResNet-152，部署时需要特别注意显存限制
- `nn.ReLU(inplace=True)` 在残差块的主路径上要小心使用——如果在加法之前使用 in-place ReLU，会覆盖原始值，导致反向传播时梯度计算错误

---

## 7. 常见错误

### 错误 1：残差块捷径路径未对齐维度

**现象：** 运行时报错 `RuntimeError: The size of tensor a (32) must match the size of tensor b (64) at non-singleton dimension 1`。

**原因：** 当 `stride != 1` 或 `in_channels != out_channels` 时，输入和输出的空间尺寸或通道数不同，不能直接做加法。捷径路径需要用 1x1 卷积对齐维度。

**修复：**

```python
# ✗ 错误：始终使用 Identity
self.shortcut = nn.Identity()

# ✓ 正确：维度不匹配时用 1x1 Conv 对齐
self.shortcut = nn.Identity()
if stride != 1 or in_channels != out_channels:
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )
```

### 错误 2：BatchNorm 之后的卷积层保留了 bias

**现象：** 模型收敛变慢，参数量比预期多。虽然不影响最终精度，但浪费了内存和计算。

**原因：** BatchNorm 层有自己的平移参数（beta/bias），紧跟其后的卷积层的 bias 是冗余的。BN 会先减去均值再除以标准差，再加上 beta——卷积层的 bias 在这个过程中完全被抵消。

**修复：**

```python
# ✗ 错误
self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

# ✓ 正确：紧跟 BN 的卷积设 bias=False
self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
self.bn1 = nn.BatchNorm2d(out_channels)
```

### 错误 3：混淆 ResNet 的加法与 DenseNet 的拼接

**现象：** 试图用 `x + self.branch(x)` 实现 DenseNet，但输出通道数不对。

**原因：** ResNet 使用加法（addition），输出通道数等于输入通道数。DenseNet 使用拼接（concatenation），输出通道数 = 输入通道数 + growth_rate。混用会导致维度灾难或信息丢失。

**修复：**

```python
# ResNet 风格：加法
out = self.branch(x) + x       # 通道数不变

# DenseNet 风格：拼接
out = torch.cat([x, self.branch(x)], dim=1)  # 通道数增加
```

### 错误 4：预训练模型的分类头维度不匹配

**现象：** 加载预训练 ResNet 后微调，训练报错 `RuntimeError: mat1 and mat2 shapes cannot be multiplied`。

**原因：** 预训练的 `model.fc` 输出 1000 维（ImageNet 的类别数），但你的任务只有 N 个类别。直接训练会导致矩阵乘法维度不匹配。

**修复：**

```python
# ✗ 错误：直接训练
output = model(images)  # 输出 1000 维，但标签只有 N 个

# ✓ 正确：替换分类头
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 替换为你的类别数
```

---

## 8. 面试考点

### Q1：为什么 VGG-16 有 1.38 亿参数，而 ResNet-50 只有 2560 万，两者 ImageNet 准确率却差不多？（难度：⭐⭐）

**参考答案：**

VGG-16 的参数量被三个全连接层主导。最后一个全连接层将 25088 维特征映射到 4096 维，需要约 1.03 亿参数。ResNet-50 用全局平均池化替代了这三个全连接层——将每个通道的空间平均值直接作为特征，参数量几乎为零。

在特征提取方面，VGG 的 13 层卷积没有残差连接，深层的梯度信号衰减严重，新增层的实际贡献有限。ResNet 的残差连接让每层都能有效学习，同样的参数预算分配得更合理。

### Q2：ResNet 的残差连接让网络可以"直接跳过"某些层。这是否意味着更深的网络不一定更准确？（难度：⭐⭐）

**参考答案：**

残差连接确实让某些层可以"不做什么"（F(x) → 0 时等于恒等映射），但这并不意味着更深不好。实际训练中，优化器会逐渐学会让不重要的层接近零贡献，而有用的层正常工作。ResNet-152 的精度持续优于 ResNet-50，说明更多的层确实提供了更多的特征提取能力。

但有一个实际的边界：超过 1000 层后，模型的提升趋于平缓，而计算开销和内存消耗持续增加。工程中通常选择 ResNet-50 或 ResNet-101 作为精度/效率的平衡点。

### Q3：解释 GoogLeNet 中 1x1 卷积的作用。为什么在 3x3 或 5x5 卷积之前要用 1x1 卷积降维？（难度：⭐⭐⭐）

**参考答案：**

假设输入有 192 通道，直接做 5x5 卷积输出 32 通道，计算量为 192 × 32 × 5 × 5 = 153,600 次乘加运算（每个空间位置）。

用 1x1 卷积先将 192 通道降到 16 通道，再做 5x5 卷积，计算量变为 192 × 16 × 1 × 1 + 16 × 32 × 5 × 5 = 3,072 + 12,800 = 15,872 次乘加运算。

降维后的计算量只有原始的约 1/10，而 1x1 卷积本身也引入了非线性变换。这就是"瓶颈层"（Bottleneck）的核心思想——用最少的计算在低维空间中提取必要信息，再用大卷积核在高维空间中做空间特征提取。

### Q4：DenseNet 的密集连接和 ResNet 的残差连接，各自的优缺点是什么？（难度：⭐⭐⭐）

**参考答案：**

ResNet 的加法连接让网络可以学习"在恒等映射的基础上需要多少修改"，参数效率高，内存可控。缺点是特征只能通过加法融合，信息可能被稀释。

DenseNet 的拼接连接实现了特征复用——每一层都能直接访问前面所有层的原始特征，不需要重新学习。这让 DenseNet 用更少的参数达到相当的准确率。但拼接导致通道数线性增长，内存消耗随深度急剧增加。一个 121 层 DenseNet 的中间激活可能占用数 GB 显存，在生产环境中部署需要特别注意。

### Q5：如果你现在要设计一个新的 CNN 架构用于工业缺陷检测，你会从哪种经典架构开始？为什么？（难度：⭐⭐⭐）

**参考答案：**

从 ResNet-18 开始。原因：

1. **参数量合适**（11.7M），满足工业部署的实时性要求
2. **预训练权重丰富**，ImageNet 预训练 + 少量缺陷数据微调就能达到不错的精度
3. **残差连接让微调更稳定**，即使你的缺陷数据只有几百张，也能有效训练
4. **社区支持好**，PyTorch、ONNX、TensorRT 都有完善的优化路径

如果精度不够，可以逐步升级到 ResNet-34 或 ResNet-50。如果计算资源极其受限（如嵌入式设备），则考虑 MobileNetV3。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 感受野（Receptive Field） | "网络能看多大的区域" | 输出特征图上一个像素对应输入图像上的空间范围。3x3 卷积的感受野是 3x3，两个堆叠的 3x3 卷积感受野是 5x5 |
| 残差连接（Residual Connection） | "让网络跳过某些层" | 输出 = 主路径 F(x) + 恒等映射 x。不是跳过，而是"在原始输入的基础上做增量修改" |
| 退化问题（Degradation Problem） | "深网络过拟合了" | 训练损失随深度增加而上升（不是验证损失）。这和过拟合不同——过拟合是训练好但验证差，退化是训练本身都变差 |
| 全局平均池化（Global Average Pooling） | "对整个特征图取平均" | 将每个通道的所有空间位置取平均值，输出一个标量。替代了 VGG 中参数量巨大的全连接层 |
| 瓶颈层（Bottleneck） | "中间窄两头宽" | 先用 1x1 卷积降维，在低维空间做计算量大的操作，再升维。Inception 和 ResNet-50 都使用了这个技巧 |
| 批归一化（Batch Normalization） | "让每层的输入标准化" | 对一个批次内每个通道分别做归一化（减均值除标准差），再通过可学习的缩放和平移参数恢复表达能力 |
| 特征复用（Feature Reuse） | "每一层都用前面的特征" | DenseNet 的密集连接让每一层都能直接访问前面所有层的原始特征输出，无需重新学习 |
| 深度可分离卷积（Depthwise Separable Convolution） | "分组卷积的极端情况" | 先对每个通道独立做空间卷积（depthwise），再用 1x1 卷积融合通道（pointwise）。计算量只有标准卷积的 1/8 到 1/9 |

---

## 📚 小结

从 LeNet-5 的 5 层到 ResNet-152 的 152 层，CNN 架构演进的核心驱动力是解决梯度消失问题，让更深的网络可以被训练。AlexNet 用 ReLU 打开了大门，VGG 用统一的 3x3 卷积简化了设计，GoogLeNet 用 Inception 模块探索了多尺度特征提取，ResNet 用残差连接彻底解决了退化问题，DenseNet 用密集连接进一步提升了特征复用效率。

理解这条演进线索，你不仅掌握了 6 种经典架构，更掌握了深度学习架构设计的核心思想——**让梯度能够有效流过更多层**。这个思想在后续的 Transformer 架构中以残差连接的形式延续，至今仍是所有深层模型的基石。

---

## ✏️ 练习

1. 【理解】用自己的话解释"退化问题"和"过拟合"的区别。为什么 ResNet 解决的是退化问题而不是过拟合？写 200 字以内的说明，用一个类比让没有 ML 背景的人也能理解。

2. 【实现】修改 `BasicBlock` 类，实现 ResNet-50 中使用的瓶颈残差块（Bottleneck Block）：主路径为 1x1 降维 → 3x3 卷积 → 1x1 升维。对比 BasicBlock 和 Bottleneck 的参数量差异。

3. 【实验】修改 `code/main.py`，添加一个 `PlainNet`（没有残差连接的普通深层网络），对比 `TinyResNet` 和 `PlainNet` 在训练 20 个轮次后的训练损失曲线差异。

4. 【思考】DenseNet 的内存消耗随深度线性增长。如果你要在手机端部署一个 DenseNet，有哪些压缩策略可以考虑？（提示：思考 Growth Rate、Transition Layer 的压缩率、模型剪枝）

5. 【设计】如果你要为一个中文手写汉字识别任务选择 CNN 架构，输入是 128x128 灰度图，可用数据约 5000 张，计算预算为一块 T4 GPU。你会选择哪种架构？为什么？请给出具体的迁移学习方案。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 6 种 CNN 架构实现 | `code/main.py` | LeNet、AlexNet、VGG、Inception、ResNet、DenseNet 的教学实现，含参数量对比和前向传播演示 |
| CNN 架构选型提示词 | `outputs/prompt-cnn-architecture-guide.md` | 根据任务需求和计算预算选择最合适 CNN 骨干网络的结构化提示词 |

---

## 📖 参考资料

1. [论文] LeCun et al. "Gradient-Based Learning Applied to Document Recognition". Proceedings of the IEEE, 1998. https://doi.org/10.1109/5.726791
2. [论文] Krizhevsky et al. "ImageNet Classification with Deep Convolutional Neural Networks". NeurIPS, 2012. https://doi.org/10.1145/3065386
3. [论文] Simonyan and Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition". ICLR, 2015. https://arxiv.org/abs/1409.1556
4. [论文] Szegedy et al. "Going Deeper with Convolutions". CVPR, 2015. https://arxiv.org/abs/1409.4842
5. [论文] He et al. "Deep Residual Learning for Image Recognition". CVPR, 2016. https://arxiv.org/abs/1512.03385
6. [论文] Huang et al. "Densely Connected Convolutional Networks". CVPR, 2017. https://arxiv.org/abs/1608.06993
7. [官方文档] PyTorch. "torchvision.models". https://pytorch.org/vision/stable/models.html
8. [GitHub] huggingface/pytorch-image-models (timm): https://github.com/huggingface/pytorch-image-models

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、知识连线、工程最佳实践、常见错误、面试考点等均为原创内容。
