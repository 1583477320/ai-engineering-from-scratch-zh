# main.py — CNN 架构演进：从 LeNet 到 ResNet
# 依赖：torch>=2.0, torchvision>=0.15
# 安装：pip install torch torchvision
# 对应课程：阶段 04 · 03（CNN 架构演进）

"""
本文件实现了 6 种经典 CNN 架构的教学版本，并对比它们的参数量和前向传播行为。
架构按时间顺序排列：LeNet → AlexNet → VGG → GoogLeNet → ResNet → DenseNet。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ============================================================
# 工具函数
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """统计模型的可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def forward_summary(name: str, model: nn.Module, input_tensor: torch.Tensor):
    """打印模型的输入/输出形状和参数量。"""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    params = count_parameters(model)
    print(f"{name:14s}  输入 {str(tuple(input_tensor.shape)):>18s}  "
          f"-> 输出 {str(tuple(output.shape)):>18s}  "
          f"参数量 {params:>12,}")


# ============================================================
# 1. LeNet-5（1998）
# 手写数字识别的开创性工作。结构简单：两层卷积 + 两层全连接。
# 核心贡献：证明了卷积神经网络可以从像素中直接学习特征。
# ============================================================

class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 卷积层：1 通道输入（灰度图），逐步提取特征
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)       # 1x32x32 -> 6x28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)      # 6x14x14 -> 16x10x10
        self.pool = nn.AvgPool2d(2)                        # 平均池化（LeNet 使用平均池化，不是最大池化）
        # 全连接层：将特征图展平后做分类
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LeNet 使用 tanh 激活函数（ReLU 尚未被广泛采用）
        x = self.pool(torch.tanh(self.conv1(x)))   # -> 6x14x14
        x = self.pool(torch.tanh(self.conv2(x)))   # -> 16x5x5
        x = torch.flatten(x, 1)                     # -> 400
        x = torch.tanh(self.fc1(x))                 # -> 120
        x = torch.tanh(self.fc2(x))                 # -> 84
        return self.fc3(x)                           # -> num_classes


# ============================================================
# 2. AlexNet 简化版（2012）
# ImageNet 竞赛的里程碑。核心创新：
#   - 使用 ReLU 替代 tanh/sigmoid，训练速度提升约 6 倍
#   - 使用 Dropout 防止过拟合
#   - 使用 GPU 进行训练
# ============================================================

class AlexNetSimplified(nn.Module):
    """AlexNet 的简化版本，保留了核心架构思想。"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 五层卷积 + 三层全连接
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # -> 64x55x55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),                              # -> 64x27x27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),           # -> 192x27x27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),                              # -> 192x13x13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),          # -> 384x13x13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # -> 256x13x13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),          # -> 256x13x13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),                              # -> 256x6x6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================
# 3. VGG 简化版（2014）
# 核心思想：用多个 3x3 小卷积核替代大卷积核。
# 两个 3x3 卷积的感受野等于一个 5x5 卷积，但参数更少且多一次非线性变换。
# VGGBlock: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
# ============================================================

class VGGBlock(nn.Module):
    """VGG 的基本构建块：两层 3x3 卷积 + 最大池化。"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class MiniVGG(nn.Module):
    """VGG 的迷你版本，演示"统一 3x3 卷积块"的设计理念。"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 64),       # 32x32 -> 16x16
            VGGBlock(64, 128),     # 16x16 -> 8x8
            VGGBlock(128, 256),    # 8x8   -> 4x4
            VGGBlock(256, 256),    # 4x4   -> 2x2
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 全局平均池化，替代全连接层
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 4. GoogLeNet / Inception 简化版（2014）
# 核心思想：Inception 模块在同一层内并行使用多种卷积核（1x1, 3x3, 5x5），
# 让网络自己决定每层该用哪种尺度的特征。
# 1x1 卷积用于降维，大幅减少计算量。
# ============================================================

class InceptionModule(nn.Module):
    """Inception 模块：四条并行路径，拼接输出。"""
    def __init__(self, in_channels: int, ch1x1: int, ch3x3_reduce: int,
                 ch3x3: int, ch5x5_reduce: int, ch5x5: int, pool_proj: int):
        super().__init__()
        # 路径 1：1x1 卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True),
        )
        # 路径 2：1x1 降维 + 3x3 卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True),
        )
        # 路径 3：1x1 降维 + 5x5 卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True),
        )
        # 路径 4：3x3 最大池化 + 1x1 投影
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 四条路径的输出在通道维度拼接
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class MiniInception(nn.Module):
    """迷你 Inception 网络，演示并行多尺度特征提取。"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 两个 Inception 模块，逐步增加通道数
        self.inception1 = InceptionModule(64, 32, 48, 64, 8, 16, 16)   # -> 128 通道
        self.pool1 = nn.MaxPool2d(2)
        self.inception2 = InceptionModule(128, 64, 64, 96, 16, 32, 32) # -> 224 通道
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(224, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inception1(x)
        x = self.pool1(x)
        x = self.inception2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ============================================================
# 5. ResNet 基本块（2015）
# 核心创新：残差连接（Skip Connection）。
# 传统网络学习 H(x)，残差网络学习 F(x) = H(x) - x。
# 恒等映射成为默认行为——网络只需学习"残差"，大幅降低优化难度。
# 这使得训练 100+ 层的网络成为可能。
# ============================================================

class BasicBlock(nn.Module):
    """ResNet 的基本残差块：两层 3x3 卷积 + 残差连接。"""
    expansion = 1  # 输出通道数 = 中间通道数 × expansion

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # 主路径：两层 3x3 卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径路径：当维度或尺寸不匹配时，用 1x1 卷积对齐
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差连接：输出 = 主路径 + 捷径
        out = out + self.shortcut(x)
        return F.relu(out)


class TinyResNet(nn.Module):
    """迷你 ResNet，演示残差连接如何让深层网络稳定训练。"""
    def __init__(self, block: type = BasicBlock, num_classes: int = 10):
        super().__init__()
        # stem 层：一个 3x3 卷积
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 四个残差组，通道数逐步加倍，空间尺寸逐步减半
        self.layer1 = self._make_group(block, 32, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_group(block, 32, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_group(block, 64, 128, num_blocks=2, stride=2)
        self.layer4 = self._make_group(block, 128, 256, num_blocks=2, stride=2)
        # 分类头：全局平均池化 + 全连接（替代 VGG 的大全连接层）
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def _make_group(self, block, in_c, out_c, num_blocks, stride):
        """构建一个残差组：第一个块做降维/下采样，后续块保持维度不变。"""
        layers = [block(in_c, out_c, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(block(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


# ============================================================
# 6. DenseNet 简化版（2017）
# 核心思想：密集连接——每一层都与前面所有层直接相连。
# 特征复用（Feature Reuse）：前面层的特征直接传递到后面每一层。
# 与 ResNet 的加法不同，DenseNet 使用拼接（concatenation）。
# ============================================================

class DenseLayer(nn.Module):
    """DenseNet 的单层：BN -> ReLU -> 1x1 降维 -> BN -> ReLU -> 3x3 卷积。"""
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 可以是前面所有层输出的拼接
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)  # 拼接而非相加


class DenseBlock(nn.Module):
    """Dense Block：多个 DenseLayer 串联，每层输入包含前面所有层的输出。"""
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate  # 每层增加 growth_rate 个通道

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MiniDenseNet(nn.Module):
    """迷你 DenseNet，演示密集连接和特征复用。"""
    def __init__(self, num_classes: int = 10, growth_rate: int = 12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 2 * growth_rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2 * growth_rate),
            nn.ReLU(inplace=True),
        )
        # 两个 Dense Block
        num_channels = 2 * growth_rate
        self.block1 = DenseBlock(4, num_channels, growth_rate)
        num_channels += 4 * growth_rate  # 4 层 × growth_rate
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, bias=False),
            nn.AvgPool2d(2),
        )
        num_channels = num_channels // 2
        self.block2 = DenseBlock(4, num_channels, growth_rate)
        num_channels += 4 * growth_rate
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ============================================================
# 主程序：对比所有架构的参数量和输出形状
# ============================================================

if __name__ == "__main__":
    # 使用 32x32 彩色输入（适合 CIFAR-10 数据集）
    dummy_input = torch.randn(1, 3, 32, 32)

    # 使用 224x224 彩色输入（适合 ImageNet 数据集）
    dummy_input_224 = torch.randn(1, 3, 224, 224)

    print("=" * 75)
    print("CNN 架构演进对比（输入: 1 x 3 x 32 x 32）")
    print("=" * 75)

    architectures = [
        ("LeNet-5 (1998)",       LeNet5(10)),
        ("AlexNet 简化",         AlexNetSimplified(10)),
        ("MiniVGG (VGG 风格)",   MiniVGG(10)),
        ("MiniInception",        MiniInception(10)),
        ("TinyResNet",           TinyResNet(BasicBlock, 10)),
        ("MiniDenseNet",         MiniDenseNet(10)),
    ]

    # LeNet-5 需要灰度输入
    lenet_input = torch.randn(1, 1, 32, 32)

    for name, model in architectures:
        inp = lenet_input if "LeNet" in name else dummy_input
        forward_summary(name, model, inp)

    print()
    print("=" * 75)
    print("ResNet 残差块参数量分析")
    print("=" * 75)

    # 演示残差块的参数量
    block_no_residual = BasicBlock(32, 64, stride=1)
    block_with_residual = BasicBlock(32, 128, stride=2)
    print(f"BasicBlock(32->64, stride=1):  "
          f"{count_parameters(block_no_residual):>8,} 参数  "
          f"(shortcut = Identity)")
    print(f"BasicBlock(32->128, stride=2): "
          f"{count_parameters(block_with_residual):>8,} 参数  "
          f"(shortcut = 1x1 Conv + BN)")

    print()
    print("=" * 75)
    print("残差连接的效果演示")
    print("=" * 75)

    # 展示残差连接如何缓解梯度消失
    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    block = BasicBlock(32, 32, stride=1)
    out = block(x)
    loss = out.sum()
    loss.backward()
    print(f"输入梯度范数: {x.grad.norm().item():.4f}")
    print("（没有残差连接时，同样深度的网络梯度会小得多）")

    print()
    print("=" * 75)
    print("DenseNet 特征复用演示")
    print("=" * 75)

    dense_layer = DenseLayer(32, growth_rate=12)
    x_demo = torch.randn(1, 32, 8, 8)
    out_demo = dense_layer(x_demo)
    print(f"输入通道数: {x_demo.shape[1]}")
    print(f"输出通道数: {out_demo.shape[1]}")
    print(f"通道增长: +{out_demo.shape[1] - x_demo.shape[1]} (growth_rate = 12)")
    print("每层的输出包含前面所有层的特征——这就是特征复用。")
