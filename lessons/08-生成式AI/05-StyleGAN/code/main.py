# main.py — StyleGAN 映射网络 + AdaIN 概念实现
# 依赖：torch>=2.0, numpy
# 安装：pip install torch numpy
# 对应课程：阶段 08 · 05（StyleGAN）

"""
StyleGAN 核心概念演示：
1. 映射网络（Mapping Network）：将 z 映射到中间空间 w，解纠缠不同因素
2. AdaIN（自适应实例归一化）：将 w 的风格信息注入到每一层
3. 样式混合（Style Mixing）：不同层使用不同的 w 向量

本文件从零实现这两个核心组件，帮助理解 StyleGAN 为什么比标准 GAN 更稳定。
"""

import torch
import torch.nn as nn
import numpy as np


# === 第 1 步：映射网络（Mapping Network） ===

class MappingNetwork(nn.Module):
    """
    映射网络：将噪声 z 映射到中间空间 w。

    标准 GAN 直接把 z 输入生成器，导致 z 的不同分量纠缠在一起——
    改变一个因素（年龄）会影响其他因素（性别、表情）。

    映射网络的作用是将 z "拉伸"到一个更解纠缠的 w 空间，
    使得 w 的不同维度对应不同的语义因素。
    """

    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        # 8 层全连接网络（StyleGAN 原始设计）
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.append(nn.Linear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        """z (batch, z_dim) → w (batch, w_dim)"""
        return self.net(z)


# === 第 2 步：AdaIN（自适应实例归一化） ===

class AdaIN(nn.Module):
    """
    自适应实例归一化（Adaptive Instance Normalization）。

    AdaIN 的核心思想：用 w 向量中的信息来"调制"特征图的风格。

    具体步骤：
    1. 对特征图做实例归一化（去掉每个通道的均值和方差）
    2. 用 w 生成缩放因子 y_scale 和偏移因子 y_bias
    3. 用这两个因子重新调整归一化后的特征图

    效果：y_scale 控制每个通道的"强度"，y_bias 控制"偏移"。
    不同的 w 产生不同的 y_scale 和 y_bias → 不同的风格。
    """

    def __init__(self, style_dim=512, num_features=256):
        super().__init__()
        # 从 w 生成缩放和偏移——这是"风格注入"的核心
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_bias = nn.Linear(style_dim, num_features)

    def forward(self, x, w):
        """
        x: (batch, channels, height, width) — 特征图
        w: (batch, style_dim) — 中间向量
        """
        # 第 1 步：实例归一化——对每个样本、每个通道独立归一化
        # 去掉"内容"的统计量，只保留相对模式
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-8
        normalized = (x - mean) / std

        # 第 2 步：用 w 生成风格参数
        # y_scale 控制每个通道的"对比度"，y_bias 控制"亮度"
        y_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        y_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)

        # 第 3 步：调制——这就是"风格注入"的时刻
        return y_scale * normalized + y_bias


# === 第 3 步：演示 z → w → 风格调制的完整流程 ===

def demo_mapping_network():
    """演示映射网络如何将 z 映射到 w。"""
    print("=" * 60)
    print("演示 1：映射网络 — z → w")
    print("=" * 60)

    z_dim = 512
    batch_size = 4

    # 创建映射网络
    mapping_net = MappingNetwork(z_dim=z_dim, w_dim=z_dim, num_layers=8)

    # 生成随机噪声 z
    z = torch.randn(batch_size, z_dim)
    print(f"\n输入 z 的形状: {z.shape}")  # (4, 512)

    # 通过映射网络得到 w
    w = mapping_net(z)
    print(f"输出 w 的形状: {w.shape}")  # (4, 512)

    # 验证：z 和 w 的分布应该不同
    # z 是标准正态分布，w 经过 8 层非线性变换后分布会改变
    print(f"\nz 的统计量 — 均值: {z.mean():.4f}, 标准差: {z.std():.4f}")
    print(f"w 的统计量 — 均值: {w.mean():.4f}, 标准差: {w.std():.4f}")

    # 检查解纠缠：两个不同的 z 应该映射到不同的 w
    z1 = torch.randn(1, z_dim)
    z2 = torch.randn(1, z_dim)
    w1 = mapping_net(z1)
    w2 = mapping_net(z2)
    distance = torch.norm(w1 - w2).item()
    print(f"\n两个不同 z 映射到 w 后的欧氏距离: {distance:.4f}")
    print("（距离越大，说明映射网络成功将不同 z 分开）")


# === 第 4 步：演示 AdaIN 的风格调制 ===

def demo_adain():
    """演示 AdaIN 如何用 w 调制特征图的风格。"""
    print("\n" + "=" * 60)
    print("演示 2：AdaIN — w → 风格调制")
    print("=" * 60)

    style_dim = 512
    channels = 256
    spatial_size = 8

    # 创建 AdaIN 层
    adain = AdaIN(style_dim=style_dim, num_features=channels)

    # 模拟生成器中间层的特征图（内容特征）
    content = torch.randn(1, channels, spatial_size, spatial_size)
    print(f"\n内容特征图形状: {content.shape}")  # (1, 256, 8, 8)

    # 两个不同的 w 向量——代表两种不同的"风格"
    w_style_a = torch.randn(1, style_dim)  # 风格 A
    w_style_b = torch.randn(1, style_dim)  # 风格 B

    # 用同一个内容特征，注入不同风格
    output_a = adain(content, w_style_a)
    output_b = adain(content, w_style_b)

    print(f"注入风格 A 后的特征图形状: {output_a.shape}")
    print(f"注入风格 B 后的特征图形状: {output_b.shape}")

    # 验证：归一化后的输出均值接近 0，标准差接近 1（被 y_scale 和 y_bias 调制）
    print(f"\n输出 A — 均值: {output_a.mean():.4f}, 标准差: {output_a.std():.4f}")
    print(f"输出 B — 均值: {output_b.mean():.4f}, 标准差: {output_b.std():.4f}")

    # 关键观察：A 和 B 的统计量不同——这就是风格差异
    diff = torch.norm(output_a - output_b).item()
    print(f"\n相同内容 + 不同风格的输出差异: {diff:.4f}")
    print("（差异越大，说明两个 w 产生的风格越不同）")


# === 第 5 步：演示样式混合（Style Mixing） ===

def demo_style_mixing():
    """
    演示样式混合：不同层使用不同的 w 向量。

    StyleGAN 的一个强大能力是"样式混合"——
    粗粒度层（低分辨率）控制全局属性（姿势、脸型），
    细粒度层（高分辨率）控制局部细节（纹理、颜色）。

    通过在不同层注入不同的 w，可以混合两张脸的特征。
    """
    print("\n" + "=" * 60)
    print("演示 3：样式混合 — 不同层使用不同的 w")
    print("=" * 60)

    style_dim = 512
    channels = 64

    # 模拟 4 个生成器层（从粗到细）
    adain_layers = nn.ModuleList([
        AdaIN(style_dim, channels) for _ in range(4)
    ])

    # 模拟特征图（实际中由卷积层生成）
    feature_maps = [torch.randn(1, channels, 8, 8) for _ in range(4)]

    # 两张不同的"脸"的 w 向量
    w_face_a = torch.randn(1, style_dim)  # 脸 A
    w_face_b = torch.randn(1, style_dim)  # 脸 B

    # 混合策略：前两层用脸 A 的风格（粗粒度），后两层用脸 B 的风格（细粒度）
    # 效果：脸 A 的整体结构 + 脸 B 的细节纹理
    mixed_styles = [w_face_a, w_face_a, w_face_b, w_face_b]

    print("\n样式混合方案：")
    print("  层 0（最粗）: 脸 A 的风格 → 控制姿势和脸型")
    print("  层 1（较粗）: 脸 A 的风格 → 控制五官位置")
    print("  层 2（较细）: 脸 B 的风格 → 控制纹理和颜色")
    print("  层 3（最细）: 脸 B 的风格 → 控制细节和光照")

    outputs = []
    for i, (layer, feat, w) in enumerate(zip(adain_layers, feature_maps, mixed_styles)):
        out = layer(feat, w)
        outputs.append(out)
        print(f"\n  层 {i}: 输入 {feat.shape} + w → 输出 {out.shape}")

    print("\n最终：4 层输出拼接后经过上采样 → 生成图像")
    print("混合效果：脸 A 的全局结构 + 脸 B 的局部细节")


# === 第 6 步：对比标准 GAN vs StyleGAN ===

def compare_architectures():
    """对比标准 GAN 和 StyleGAN 的架构差异。"""
    print("\n" + "=" * 60)
    print("对比：标准 GAN vs StyleGAN")
    print("=" * 60)

    z_dim = 512
    w_dim = 512

    print("""
    标准 GAN 生成器：
    ┌─────────────────────────────────────────┐
    │  z (噪声向量)                            │
    │    │                                     │
    │    ▼                                     │
    │  [全连接层] → [卷积层 × N] → 图像        │
    └─────────────────────────────────────────┘
    问题：z 的不同分量纠缠 → 改变年龄时性别也变

    StyleGAN 生成器：
    ┌─────────────────────────────────────────┐
    │  z (噪声向量)                            │
    │    │                                     │
    │    ▼                                     │
    │  [映射网络 8 层 MLP] → w (中间向量)       │
    │                        │                 │
    │              ┌─────────┼─────────┐       │
    │              ▼         ▼         ▼       │
    │           [AdaIN]   [AdaIN]   [AdaIN]   │
    │              │         │         │       │
    │              ▼         ▼         ▼       │
    │  [卷积层] → [卷积层] → ... → 图像       │
    └─────────────────────────────────────────┘
    优势：w 空间更解纠缠 → 可以独立控制不同因素
    """)

    # 参数量对比
    mapping_params = sum(p.numel() for p in MappingNetwork(z_dim, w_dim).parameters())
    print(f"映射网络参数量: {mapping_params:,}（约 {mapping_params / 1e6:.2f}M）")
    print("（这额外的参数换取了更好的可控性和更稳定的训练）")


# === 第 7 步：用 PyTorch 演示完整的前向传播 ===

def demo_full_forward():
    """
    演示一个简化的 StyleGAN 前向传播流程。
    使用极小的尺寸（8x8）来展示数据流。
    """
    print("\n" + "=" * 60)
    print("演示 4：简化 StyleGAN 前向传播")
    print("=" * 60)

    z_dim = 512
    w_dim = 512
    batch_size = 2

    # 映射网络
    mapping = MappingNetwork(z_dim, w_dim, num_layers=8)

    # 简化的生成器：2 层卷积 + AdaIN
    conv1 = nn.Conv2d(3, 16, 3, padding=1)
    conv2 = nn.Conv2d(16, 3, 3, padding=1)
    adain1 = AdaIN(w_dim, 16)
    adain2 = AdaIN(w_dim, 3)
    relu = nn.LeakyReLU(0.2)

    # 输入噪声
    z = torch.randn(batch_size, z_dim)
    print(f"\n1. 输入噪声 z: {z.shape}")

    # 映射网络
    w = mapping(z)
    print(f"2. 映射后 w: {w.shape}")

    # 从常量开始（StyleGAN 的设计：从一个可学习的常量张量开始）
    constant_input = torch.randn(batch_size, 3, 8, 8)
    print(f"3. 常量输入: {constant_input.shape}")

    # 第 1 层：卷积 + AdaIN
    h = conv1(constant_input)
    h = adain1(h, w)
    h = relu(h)
    print(f"4. 第 1 层输出: {h.shape}")

    # 第 2 层：卷积 + AdaIN
    h = conv2(h)
    h = adain2(h, w)
    h = relu(h)
    print(f"5. 第 2 层输出: {h.shape}")

    # 最终输出（用 tanh 压缩到 [-1, 1]）
    output = torch.tanh(h)
    print(f"6. 最终输出: {output.shape}")
    print(f"   输出范围: [{output.min():.4f}, {output.max():.4f}]")


# === 主程序 ===

if __name__ == "__main__":
    print("StyleGAN 核心概念演示")
    print("从零理解映射网络和 AdaIN 的工作原理\n")

    # 设置随机种子以保证可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行所有演示
    demo_mapping_network()
    demo_adain()
    demo_style_mixing()
    compare_architectures()
    demo_full_forward()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("""
    关键收获：
    1. 映射网络将 z 映射到更解纠缠的 w 空间
    2. AdaIN 用 w 的信息调制每一层的特征——控制风格
    3. 样式混合允许在不同层注入不同的 w——混合不同属性
    4. 这些组件共同实现了可控的人脸生成
    """)
