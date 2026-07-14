# DDPM 从零实现
# 演示完整的 DDPM 训练与采样流程
# 使用 MNIST 数据集作为示例

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


# ============================================================================
# 第 1 步：噪声调度（Noise Schedule）
# ============================================================================

class NoiseScheduler:
    """DDPM 噪声调度器——管理 beta、alpha、alpha_bar 序列。"""

    def __init__(self, num_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_steps = num_steps
        # 线性调度：beta 从 beta_start 线性增长到 beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):
        """获取时间步 t 对应的 alpha_bar 累积值。"""
        return self.alpha_bars[t]

    def sample(self, t, shape, device):
        """
        采样噪声系数——从 q(x_t | x_0) 的闭式解中采样。
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        alpha_bar = self.get_alpha_bar(t).to(device)
        # 对每个样本广播 alpha_bar
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)  # (batch, 1, 1, 1)
        noise = torch.randn(shape, device=device)
        sqrt_ab = torch.sqrt(alpha_bar)
        sqrt_one_minus_ab = torch.sqrt(1 - alpha_bar)
        return sqrt_ab * noise, sqrt_one_minus_ab, noise


# ============================================================================
# 第 2 步：简化 U-Net 去噪网络
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码——用于时间步嵌入。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """残差块——U-Net 的基本构建单元。"""

    def __init__(self, channels, emb_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_channels),
            nn.Linear(emb_channels, channels),
            nn.SiLU(),
        )
        self.layers = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x, t):
        """前向传播：x 为输入图像，t 为时间步嵌入。"""
        h = self.layers(x)
        # 将时间嵌入加到特征图上
        t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        h = h + t_emb
        return x + h  # 残差连接


class SimpleUNet(nn.Module):
    """简化版 U-Net——预测噪声残差。"""

    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=256):
        super().__init__()
        self.in_channels = in_channels

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # 编码器（下采样）
        self.down1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.res1_1 = ResBlock(base_channels, time_emb_dim)
        self.res1_2 = ResBlock(base_channels, time_emb_dim)
        self.down_sample1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)

        self.down2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.res2_1 = ResBlock(base_channels * 2, time_emb_dim)
        self.res2_2 = ResBlock(base_channels * 2, time_emb_dim)
        self.down_sample2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ResBlock(base_channels * 2, time_emb_dim),
            ResBlock(base_channels * 2, time_emb_dim),
        )

        # 解码器（上采样）
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_res1_1 = ResBlock(base_channels * 3, time_emb_dim)  # 3x: concat skip
        self.up_res1_2 = ResBlock(base_channels * 3, time_emb_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_res2_1 = ResBlock(base_channels * 2, time_emb_dim)
        self.up_res2_2 = ResBlock(base_channels * 2, time_emb_dim)

        # 输出层
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.output = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t):
        """
        前向传播：预测输入 x 在时间步 t 添加的噪声。
        Args:
            x: 带噪图像 (B, C, H, W)
            t: 时间步 (B,)
        Returns:
            predicted_noise: 预测的噪声 (B, C, H, W)
        """
        # 时间嵌入
        t_emb = self.time_mlp(t)

        # 编码器
        h = self.down1(x)
        h = self.res1_1(h, t_emb)
        h1 = self.res1_2(h)  # 保存跳跃连接
        h = self.down_sample1(h1)

        h = self.down2(h)
        h = self.res2_1(h, t_emb)
        h2 = self.res2_2(h)  # 保存跳跃连接
        h = self.down_sample2(h2)

        # 瓶颈
        h = self.bottleneck(h)

        # 解码器（含跳跃连接）
        h = self.up1(h)
        h = torch.cat([h, h2], dim=1)  # 拼接跳跃连接
        h = self.up_res1_1(h, t_emb)
        h3 = self.up_res1_2(h)

        h = self.up2(h3)
        h = torch.cat([h, h1], dim=1)
        h = self.up_res2_1(h, t_emb)
        h = self.up_res2_2(h)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.output(h)


# ============================================================================
# 第 3 步：前向扩散与训练
# ============================================================================

def forward_diffuse(x_0, scheduler, device):
    """
    前向扩散：随机采样一个时间步 t，向 x_0 添加噪声。
    使用 q(x_t | x_0) 的闭式解，无需逐步加噪。
    """
    batch_size = x_0.size(0)
    # 随机采样时间步
    t = torch.randint(0, scheduler.num_steps, (batch_size,), device=device)
    # 采样噪声
    noise = torch.randn_like(x_0)
    # 闭式解：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    scaled_noise, scale_noise, _ = scheduler.sample(t, x_0.shape, device)
    x_t = scaled_noise + x_0 * (1 - scale_noise)
    return x_t, t, noise


def train_ddpm(model, dataloader, scheduler, num_epochs=10, lr=1e-3):
    """
    训练 DDPM 模型。
    损失函数：预测噪声与实际噪声的 MSE。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        model.train()

        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            x_t, t, noise = forward_diffuse(batch_x, scheduler, device)

            # 预测噪声
            pred_noise = model(x_t, t)

            # 计算 MSE 损失
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    return model


# ============================================================================
# 第 4 步：采样（从噪声生成图像）
# ============================================================================

@torch.no_grad()
def sample_ddpm(model, scheduler, num_samples=16, image_size=28, device="cpu"):
    """
    从纯噪声逐步去噪生成图像。
    从 x_T ~ N(0, I) 开始，迭代 T 步去噪。
    """
    model.eval()
    x_T = torch.randn(num_samples, 1, image_size, image_size, device=device)

    for t in reversed(range(scheduler.num_steps)):
        t_batch = torch.full((num_samples,), t, device=device)

        # 预测噪声
        predicted_noise = model(x_T, t_batch)

        # 计算 x_{t-1} 的均值
        alpha = scheduler.alphas[t].to(device)
        alpha_bar = scheduler.get_alpha_bar(t).to(device)
        alpha_bar_prev = scheduler.get_alpha_bar(max(t - 1, 0)).to(device)

        alpha = alpha.view(-1, 1, 1, 1)
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(-1, 1, 1, 1)

        # DDPM 采样公式
        # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * epsilon_theta)
        #          + sigma_t * z
        coefficient = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        mean = (1 / torch.sqrt(alpha)) * (x_T - coefficient * predicted_noise)

        # 最后一步（t=0）不加随机噪声
        if t > 0:
            sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * scheduler.betas[t])
            z = torch.randn_like(x_T)
            x_T = mean + sigma_t * z
        else:
            x_T = mean

    # 将像素值裁剪到 [-1, 1] 范围并返回
    x_T = torch.clamp(x_T, -1.0, 1.0)
    return x_T


# ============================================================================
# 第 5 步：DDIM 加速采样（减少步数）
# ============================================================================

@torch.no_grad()
def sample_ddim(model, scheduler, num_samples=16, image_size=28,
                device="cpu", num_steps=50):
    """
    DDIM 采样——确定性去噪，可将步数从 1000 压缩到 50。
    与 DDPM 的区别：不使用随机采样，而是确定性地计算 x_{t-1}。
    """
    model.eval()

    # 创建缩减的时间步序列
    stride = scheduler.num_steps // num_steps
    timesteps = list(range(0, scheduler.num_steps, stride))

    x = torch.randn(num_samples, 1, image_size, image_size, device=device)

    for i, t in enumerate(reversed(timesteps)):
        t_batch = torch.full((num_samples,), t, device=device)
        predicted_noise = model(x, t_batch)

        alpha = scheduler.alphas[t].to(device)
        alpha_bar = scheduler.get_alpha_bar(t).to(device)
        alpha_next = scheduler.get_alpha_bar(timesteps[min(i + 1, len(timesteps) - 1)]).to(device) \
            if i + 1 < len(timesteps) else torch.ones_like(alpha_bar)

        alpha = alpha.view(-1, 1, 1, 1)
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        alpha_next = alpha_next.view(-1, 1, 1, 1)

        # DDIM 确定性公式
        # x_{t-1} = sqrt(alpha_bar_next) * (x_t - sqrt(1 - alpha_bar) * epsilon_theta) / sqrt(alpha_bar)
        #          + sqrt(1 - alpha_bar_next) * epsilon_theta
        coefficient = torch.sqrt(alpha_bar)
        pred_original = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / coefficient
        x = torch.sqrt(alpha_next) * pred_original + torch.sqrt(1 - alpha_next) * predicted_noise

    x = torch.clamp(x, -1.0, 1.0)
    return x


# ============================================================================
# 主程序：完整流程演示
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载 MNIST 数据（归一化到 [-1, 1]）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),  # [0, 1] -> [-1, 1]
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    # 初始化模型和调度器
    scheduler = NoiseScheduler(num_steps=100)  # 演示用 100 步而非 1000 步
    model = SimpleUNet(in_channels=1, base_channels=32, time_emb_dim=128).to(device)

    # 训练
    print("\n开始训练...")
    model = train_ddpm(model, dataloader, scheduler, num_epochs=3, lr=1e-3)

    # DDIM 采样（快速验证）
    print("\n开始采样（DDIM, 50 步）...")
    samples = sample_ddim(model, scheduler, num_samples=16, device=device, num_steps=50)

    # 保存一张样本图像
    sample_img = (samples[0].cpu().squeeze() + 1) / 2  # 转回 [0, 1]
    print(f"\n采样完成！输出形状: {samples.shape}")
    print("样本图像像素范围:", sample_img.min().item(), "~", sample_img.max().item())
