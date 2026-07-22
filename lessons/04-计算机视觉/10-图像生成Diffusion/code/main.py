# === 文件头注释 ===
# main.py — 从零实现 DDPM 扩散模型
# 依赖：torch>=2.0, numpy
# 对应课程：阶段 04 · 10（图像生成 Diffusion）

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 噪声调度
# ============================================================

def linear_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    """线性噪声调度：beta_t 从 beta_start 线性增长到 beta_end。"""
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T, s=0.008):
    """余弦噪声调度（Nichol & Dhariwal, 2021）。
    相比线性调度，余弦调度让信号在更早的时间步不会过快衰减，
    使得在较少步数采样时也能保持较高质量的生成效果。
    """
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def precompute_schedule(betas):
    """预处理调度参数，避免训练和采样中重复计算。

    参数:
        betas: 噪声调度序列，形状 (T,)

    返回:
        schedule: 包含所有预计算系数的字典
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
    }


# ============================================================
# 前向加噪（闭合形式）
# ============================================================

def q_sample(x0, t, noise, schedule):
    """前向加噪：根据闭合形式公式直接采样任意时刻 t 的噪声图片。

    利用公式 x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    参数:
        x0: 原始干净图片，形状 (batch, channels, H, W)
        t: 时间步索引，形状 (batch,)
        noise: 从 N(0,I) 采样的噪声，形状与 x0 相同
        schedule: 预计算的调度字典

    返回:
        xt: t 时刻的噪声图片，形状 (batch, channels, H, W)
    """
    sqrt_a = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_one_minus_a * noise


# ============================================================
# 时间步嵌入
# ============================================================

def timestep_embedding(t, dim):
    """正弦时间步编码。

    与 Transformer 中的位置编码相同：用不同频率的正余弦函数
    将标量时间步映射到高维空间。

    参数:
        t: 时间步，形状 (batch,)
        dim: 输出的嵌入维度

    返回:
        emb: 时间嵌入向量，形状 (batch, dim)
    """
    t = t.float()
    half_dim = (dim + 1) // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
    args = t[:, None] * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb[:, :dim]


# ============================================================
# TinyUNet 架构
# ============================================================

class ResidualBlock(nn.Module):
    """残差块：两组卷积 + 时间步条件注入。"""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # 将时间嵌入投影到合适的维度后与特征图逐元素相加
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.conv1(x))
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_proj
        h = self.act(self.conv2(h))
        # 残差连接
        return h + x


class TinyUNet(nn.Module):
    """微型 U-Net，用于教学演示。
    两层编码器-解码器结构，含跳跃连接和时间步条件注入。
    """

    def __init__(self, img_channels=3, base=16, t_dim=64):
        super().__init__()
        self.t_dim = t_dim

        # 时间步 MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, base * 4),
        )

        # 编码器
        self.enc1 = ResidualBlock(img_channels, base, base * 4)
        self.enc2 = ResidualBlock(base, base * 2, base * 4)
        self.downsample = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)

        # 瓶颈
        self.bottleneck = ResidualBlock(base * 2, base * 2, base * 4)

        # 解码器
        self.upsample = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(base * 2, base, base * 4)
        self.output_head = nn.Conv2d(base, img_channels, 1)

    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.t_mlp(timestep_embedding(t, self.t_dim))

        # 编码路径
        h1 = self.enc1(x, t_emb)       # skip connection 1
        h = self.enc2(h1, t_emb)
        h = self.downsample(h)

        # 瓶颈
        h = self.bottleneck(h, t_emb)

        # 解码路径 + 跳跃连接
        h = self.upsample(h)
        h = torch.cat([h, h1], dim=1)  # 拼接 skip 特征
        h = self.dec1(h, t_emb)

        return self.output_head(h)


# ============================================================
# 训练
# ============================================================

def train_one_step(model, batch, schedule, optimizer, device, T):
    """训练单步：随机采样时间步、前向加噪、预测噪声、计算 MSE。

    参数:
        model: U-Net 噪声预测模型
        batch: 一个批次的真实图片，形状 (batch_size, channels, H, W)
        schedule: 预计算的调度字典
        optimizer: 优化器
        device: 运行设备 ("cpu" 或 "cuda")
        T: 总时间步数

    返回:
        loss: 该步的 MSE 损失值
    """
    model.train()
    x0 = batch.to(device)
    bs = x0.size(0)

    # 随机采样时间步
    t = torch.randint(0, T, (bs,), device=device)
    # 采样高斯噪声
    noise = torch.randn_like(x0)
    # 用闭合形式生成带噪图片
    x_t = q_sample(x0, t, noise, schedule)
    # 模型预测噪声
    predicted_noise = model(x_t, t)
    # MSE 损失
    loss = F.mse_loss(predicted_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ============================================================
# 采样器
# ============================================================

@torch.no_grad()
def sample_ddpm(model, schedule, shape, T, device):
    """DDPM 祖先采样：从纯噪声出发，逐步去噪。每步引入随机扰动。

    参数:
        model: 训练好的噪声预测模型
        schedule: 调度字典
        shape: 目标形状 (batch, channels, H, W)
        T: 总时间步数
        device: 运行设备

    返回:
        samples: 生成的图片张量
    """
    model.eval()
    # 从标准高斯分布采样纯噪声
    x = torch.randn(shape, device=device)
    betas = schedule["betas"].to(device)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(device)
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"].to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        # 从噪声预测推导 x_{t-1} 的条件均值
        coef = betas[t] / sqrt_one_minus_a[t]
        mean = sqrt_recip_alphas[t] * (x - coef * eps)
        # 加入随机扰动（祖先采样的标志）
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(betas[t]) * noise
        else:
            x = mean
    return x


@torch.no_grad()
def sample_ddim(model, schedule, shape, steps, T, device, eta=0.0):
    """DDIM 确定性采样：跳过中间时间步，沿确定性路径演化。

    参数:
        eta: 随机性强度。0=完全确定，1=退化为 DDPM。
        steps: 采样步数（跳跃式，不是逐一步）

    返回:
        samples: 生成的图片张量
    """
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    # 均匀间隔采样 steps+1 个时间步
    ts = torch.linspace(T - 1, 0, steps + 1).long()
    for i in range(steps):
        t_curr = int(ts[i])
        t_next = int(ts[i + 1])
        t_batch = torch.full((shape[0],), t_curr, dtype=torch.long, device=device)

        eps = model(x, t_batch)
        a_t = alphas_cumprod[t_curr]
        a_next = alphas_cumprod[t_next]

        # 先反推 x_0（干净图片的估计）
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)

        # DDIM 方向分量（确定性）+ 可能的随机扰动
        sigma = eta * torch.sqrt(
            (1 - a_next) / (1 - a_t).clamp_min(0) * (1 - a_t / a_next).clamp_min(0)
        )
        dir_xt = torch.sqrt((1 - a_next - sigma ** 2).clamp_min(0)) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0
        x = torch.sqrt(a_next) * x0_pred + dir_xt + noise

    return x


# ============================================================
# 合成数据
# ============================================================

def synthetic_circles(num_samples=200, image_size=16, seed=42):
    """生成合成数据：随机大小、随机颜色、随机位置的彩色圆圈。

    这个数据集足够简单，可以在 CPU 上几分钟内收敛训练。
    它验证了扩散模型能否学习到"圆形"这个结构。

    参数:
        num_samples: 样本数量
        image_size: 图像边长
        seed: 随机种子

    返回:
        tensors: 形状为 (num_samples, 3, image_size, image_size) 的张量
    """
    rng = np.random.default_rng(seed)
    imgs = np.full((num_samples, 3, image_size, image_size), -1.0, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing="ij")
    for i in range(num_samples):
        radius = rng.uniform(3, 5)
        cx, cy = rng.uniform(radius, image_size - radius, size=2)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2
        color = rng.uniform(-0.3, 1.0, size=3)
        for c in range(3):
            imgs[i, c][mask] = color[c]
    return torch.from_numpy(imgs)


# ============================================================
# 主程序
# ============================================================

def main():
    """完整的 DDPM 训练与采样入口。"""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 200  # 教学用途减少步数以加快训练

    print("=" * 50)
    print("  从零实现 DDPM 扩散模型")
    print("=" * 50)

    # 1. 构建噪声调度
    betas = linear_beta_schedule(T=T, beta_start=1e-4, beta_end=0.04)
    schedule = precompute_schedule(betas)
    print(f"\n调度配置: T={T}  (线性 beta 1e-4 → {betas[-1]:.4f})")
    print(f"  alpha_bar[0]  = {float(schedule['alphas_cumprod'][0]):.4f}")
    print(f"  alpha_bar[-1] = {float(schedule['alphas_cumprod'][-1]):.4f}")

    # 2. 加载合成数据
    data = synthetic_circles(num_samples=100, image_size=16)
    loader = DataLoader(TensorDataset(data), batch_size=16, shuffle=True)
    print(f"  数据集: {len(data)} 张图片, 每张 {data.shape[2]}x{data.shape[3]}")

    # 3. 初始化模型
    model = TinyUNet(img_channels=3, base=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {param_count:,}")

    # 4. 训练
    print("\n--- 开始训练 ---")
    for epoch in range(3):
        losses = []
        for (batch,) in loader:
            loss = train_one_step(model, batch, schedule, opt, device, T)
            losses.append(loss)
        print(f"  Epoch {epoch + 1}/3 | 平均 MSE: {np.mean(losses):.4f}")

    # 5. 采样
    print("\n--- 采样 ---")
    shape = (4, 3, 16, 16)

    ddpm_samples = sample_ddpm(model, schedule, shape, T=T, device=device)
    print(f"DDPM 输出: 形状 {tuple(ddpm_samples.shape)}, "
          f"值域 [{ddpm_samples.min():.2f}, {ddpm_samples.max():.2f}]")

    ddim_samples = sample_ddim(model, schedule, shape, steps=20, T=T, device=device)
    print(f"DDIM 输出: 形状 {tuple(ddim_samples.shape)}, "
          f"值域 [{ddim_samples.min():.2f}, {ddim_samples.max():.2f}]")

    print(f"\n采样完成！DDPM 运行了 {T} 步，DDIM 运行了 20 步。")


if __name__ == "__main__":
    main()
