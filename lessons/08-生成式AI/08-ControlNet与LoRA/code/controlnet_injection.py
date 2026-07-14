# ControlNet 简化实现
# 演示零卷积初始化和 ControlNet 特征注入机制

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 第 1 步：零卷积层
# ============================================================================

class ZeroConv2d(nn.Module):
    """
    零卷积层——ControlNet 的核心设计。
    所有权重和偏置初始化为零，确保训练开始时输出恒为零。
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=1, padding=0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding,
        )
        # 关键：将卷积核权重和偏置全部初始化为零
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        """初始时始终返回零张量。"""
        return self.conv(x)


# ============================================================================
# 第 2 步：简化版 ControlNet（克隆 U-Net 下采样分支）
# ============================================================================

class SimpleControlNet(nn.Module):
    """
    简化版 ControlNet——克隆 U-Net 的下采样（编码器）分支。
    学习将控制信号（如边缘图）编码为特征偏移量。
    """

    def __init__(self, in_channels=3, base_channels=64, conditioning_channels=3):
        super().__init__()
        # 下采样分支（克隆 U-Net encoder）
        self.input_conv = nn.Conv2d(conditioning_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList([
            # Block 1: base_channels -> base_channels
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
            ),
            # Block 2: base_channels -> base_channels*2
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            ),
            # Block 3: base_channels*2 -> base_channels*4
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            ),
            # Block 4: base_channels*4 -> base_channels*4
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            ),
        ])

        # 零卷积层——每个下采样块对应一个零卷积
        self.zero_convs = nn.ModuleList([
            ZeroConv2d(base_channels),
            ZeroConv2d(base_channels * 2),
            ZeroConv2d(base_channels * 4),
            ZeroConv2d(base_channels * 4),
        ])

    def forward(self, control_cond):
        """
        将控制信号编码为特征偏移量。
        Args:
            control_cond: 控制信号（如 Canny 边缘图），shape: (B, C, H, W)
        Returns:
            每个下采样块的零卷积输出列表，用于注入到 U-Net 对应层
        """
        h = self.input_conv(control_cond)
        outputs = []

        for i, (down_block, zero_conv) in enumerate(zip(self.down_blocks, self.zero_convs)):
            h = down_block(h)
            # 通过零卷积（初始时输出为零）
            outputs.append(zero_conv(h))

        return outputs


# ============================================================================
# 第 3 步：模拟 ControlNet 注入到 U-Net
# ============================================================================

class SimpleUNetWithControlNet(nn.Module):
    """
    简化版 U-Net + ControlNet 注入——展示 ControlNet 如何工作。
    """

    def __init__(self, base_channels=64):
        super().__init__()
        # ControlNet
        self.controlnet = SimpleControlNet(
            in_channels=3,
            base_channels=base_channels,
            conditioning_channels=3,  # 控制信号通道数
        )

        # 简化 U-Net 下采样块
        self.down_blocks = nn.ModuleList([
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
        ])

        # 简化 U-Net 上采样块
        self.up_blocks = nn.ModuleList([
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.Conv2d(base_channels, 3, 3, padding=1),
        ])

    def forward(self, x, control_cond):
        """
        Args:
            x: 带噪潜向量 (B, C, H, W)
            control_cond: 控制信号 (B, 3, H, W)
        Returns:
            预测的噪声 (B, C, H, W)
        """
        # 获取 ControlNet 的特征偏移量
        control_offsets = self.controlnet(control_cond)

        # U-Net 下采样 + 注入 ControlNet 偏移
        h = x
        for down_block, offset in zip(self.down_blocks, control_offsets):
            h = down_block(h)
            h = h + offset  # 零卷积输出初始为零，所以第一次前向等价于无 ControlNet

        # U-Net 上采样
        for up_block in self.up_blocks:
            h = up_block(h)

        return h


# ============================================================================
# 主程序：演示
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    base_channels = 16  # 小通道数用于测试
    model = SimpleUNetWithControlNet(base_channels=base_channels).to(device)

    batch_size = 2
    dummy_latent = torch.randn(batch_size, base_channels, 32, 32).to(device)
    dummy_control = torch.randn(batch_size, 3, 32, 32).to(device)

    # 第一次前向——ControlNet 输出为零，等价于普通 U-Net
    print("第一次前向传播（零卷积初始状态）:")
    model.eval()
    with torch.no_grad():
        output = model(dummy_latent, dummy_control)
    print(f"  输出形状: {output.shape}")
    print(f"  输出均值: {output.mean().item():.6f} (应接近 0)")
    print(f"  输出标准差: {output.std().item():.4f}")

    # 验证 ControlNet 的零初始化
    print("\n验证零卷积初始化:")
    controlnet = model.controlnet
    control_offsets = controlnet(dummy_control)
    for i, offset in enumerate(control_offsets):
        print(f"  Block {i}: 均值={offset.mean().item():.10f}, 最大值={offset.abs().max().item():.10f}")

    # 总参数量
    total_params = sum(p.numel() for p in model.parameters())
    controlnet_params = sum(p.numel() for p in controlnet.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"ControlNet 参数量: {controlnet_params:,}")
    print(f"ControlNet 占比: {controlnet_params/total_params*100:.1f}%")
