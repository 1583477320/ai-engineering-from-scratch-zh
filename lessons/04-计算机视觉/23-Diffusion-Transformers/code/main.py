# main.py — Diffusion Transformer (DiT) + Rectified Flow 从零实现
# 依赖：torch>=2.0, numpy
# 安装：pip install torch numpy
# 对应课程：阶段 04 · 23（Diffusion Transformers）

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# === 第 1 步：时间步嵌入 ===
# DiT 需要知道当前去噪处于哪个时间步 t ∈ [0, 1]
# 使用正弦位置编码（与 Transformer 的位置编码原理相同）

def timestep_embedding(t, dim):
    """将标量时间步 t 编码为 dim 维向量。

    Args:
        t: 时间步张量，形状 (batch,)，值在 [0, 1] 之间
        dim: 输出嵌入维度

    Returns:
        形状 (batch, dim) 的时间步嵌入向量
    """
    half = dim // 2
    # 高频到低频的频率，范围 [1, 10000]
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    # 拼接 sin 和 cos 得到完整嵌入
    return torch.cat([args.sin(), args.cos()], dim=-1)


# === 第 2 步：AdaLN-Zero 条件调制层 ===
# DiT 的核心创新：用自适应层归一化（Adaptive Layer Norm）替代 U-Net 的 FiLM 调制
# 条件向量（时间步）通过 MLP 预测 scale、shift、gate 三个参数
# "Zero" 指 MLP 权重初始化为零——训练初期整个块表现为恒等映射，稳定深层训练

class AdaLNZero(nn.Module):
    """自适应层归一化 + 零初始化门控。

    从条件向量预测三个调制参数：
    - scale：缩放归一化输出
    - shift：偏移归一化输出
    - gate：控制残差连接的强度
    """

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # 输出 3 倍维度：scale + shift + gate
        self.mlp = nn.Linear(cond_dim, dim * 3)
        # 零初始化：训练开始时 block 是恒等映射
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, x, cond):
        scale, shift, gate = self.mlp(cond).chunk(3, dim=-1)
        # 先归一化，再用 scale 和 shift 调制
        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return h, gate.unsqueeze(1)


# === 第 3 步：DiT Block ===
# 一个标准的 Transformer 块，但用 AdaLN 替代了固定的 LayerNorm
# 每个块包含：AdaLN → 多头注意力 → 残差 → AdaLN → FFN → 残差
# 关键区别：gate 控制残差强度，训练初期 gate ≈ 0，梯度可以稳定回传

class DiTBlock(nn.Module):
    """Diffusion Transformer 的基本构建块。

    与标准 Transformer 块的区别：
    1. 用 AdaLN-Zero 替代固定 LayerNorm
    2. 残差连接由门控值控制
    3. 所有条件信息通过 AdaLN 注入（不使用交叉注意力）
    """

    def __init__(self, dim=96, heads=3, mlp_ratio=4, cond_dim=96):
        super().__init__()
        # 第一个 AdaLN：调制进入注意力层的输入
        self.adaln1 = AdaLNZero(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        # 第二个 AdaLN：调制进入 FFN 的输入
        self.adaln2 = AdaLNZero(dim, cond_dim)
        # 前馈网络：先扩展 4 倍，再投影回来
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond):
        # 注意力子层：AdaLN 调制 → 注意力 → 门控残差
        h, gate1 = self.adaln1(x, cond)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1 * a

        # FFN 子层：AdaLN 调制 → FFN → 门控残差
        h, gate2 = self.adaln2(x, cond)
        x = x + gate2 * self.mlp(h)
        return x


# === 第 4 步：完整的 TinyDiT 模型 ===
# 整体架构：
#   1. Patch Embedding：将图像切分为 patch 并线性映射到向量空间
#   2. 可学习位置编码
#   3. 时间步嵌入 → MLP → 条件向量
#   4. N 个 DiT Block（每个都有 AdaLN 条件调制）
#   5. Unpatchify：将向量序列还原为图像张量

class TinyDiT(nn.Module):
    """教学用微型 Diffusion Transformer。

    参数量约 0.5M，适合在 CPU 上快速实验。
    工业级 DiT（如 FLUX 12B）使用相同架构，只是更深更宽。
    """

    def __init__(self, image_size=16, patch_size=2, in_channels=3,
                 dim=96, depth=4, heads=3):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channels = in_channels
        # patch 数量 = (image_size / patch_size) ^ 2
        self.num_patches = (image_size // patch_size) ** 2

        # Patch Embedding：用卷积实现，每个 patch 映射为 dim 维向量
        self.patch = nn.Conv2d(in_channels, dim,
                               kernel_size=patch_size, stride=patch_size)
        # 可学习位置编码
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))

        # 时间步 MLP：将时间步嵌入映射为条件向量
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

        # 堆叠 DiT Block
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, heads, cond_dim=dim) for _ in range(depth)]
        )

        # 输出层：先归一化，再投影为每个 patch 的原始像素
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, patch_size * patch_size * in_channels)

        # 初始化位置编码
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x, t):
        """
        Args:
            x: 噪声图像，形状 (batch, channels, height, width)
            t: 时间步，形状 (batch,)，值在 [0, 1] 之间

        Returns:
            预测的速度场，形状与输入 x 相同
        """
        n = x.size(0)

        # Patch Embedding：(batch, 3, H, W) → (batch, num_patches, dim)
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2) + self.pos

        # 时间步编码 → 条件向量
        t_emb = self.time_mlp(timestep_embedding(t, self.pos.size(-1)))

        # 依次通过每个 DiT Block
        for blk in self.blocks:
            x = blk(x, t_emb)

        # 输出映射
        x = self.norm_out(x)
        x = self.head(x)

        # 还原为图像张量
        return self._unpatchify(x, n)

    def _unpatchify(self, x, n):
        """将 patch 序列还原为图像张量。"""
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        # (batch, num_patches, patch_size^2 * channels) → 图像
        x = x.view(n, h, w, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (batch, channels, h, p, w, p)
        x = x.reshape(n, self.in_channels, h * p, w * p)
        return x


# === 第 5 步：Rectified Flow 训练 ===
# 核心思想：在干净数据 x_0 和噪声 epsilon 之间画一条直线
# x_t = (1 - t) * x_0 + t * epsilon
# 网络学习预测沿直线方向的速度 v = epsilon - x_0
# 这比 DDPM 的弯曲随机路径简单得多，采样只需要 20 步而非 1000 步

def rectified_flow_train_step(model, x0, optimizer, device):
    """执行一步 Rectified Flow 训练。

    训练流程：
    1. 从均匀分布采样时间步 t ~ U(0, 1)
    2. 按直线插值构造 x_t
    3. 目标速度 = epsilon - x_0（从数据指向噪声的方向）
    4. 最小化预测速度与目标速度的均方误差
    """
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)

    # 时间步均匀采样
    t = torch.rand(n, device=device)

    # 随机噪声
    epsilon = torch.randn_like(x0)

    # 直线插值：t=0 时是干净数据，t=1 时是纯噪声
    t_expand = t[:, None, None, None]
    x_t = (1 - t_expand) * x0 + t_expand * epsilon

    # 速度目标：从 x_0 指向 epsilon 的方向
    target_velocity = epsilon - x0

    # 网络预测速度
    pred_velocity = model(x_t, t)

    # 均方误差损失
    loss = F.mse_loss(pred_velocity, target_velocity)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# === 第 6 步：Euler 采样器 ===
# Rectified Flow 的采样就是一个 ODE 积分
# 从 t=1（纯噪声）出发，沿预测速度反向积分到 t=0（干净数据）
# 20 步 Euler 积分就足以生成高质量样本

@torch.no_grad()
def rectified_flow_sample(model, shape, steps=20, device="cpu"):
    """用 Euler 方法从 Rectified Flow 模型采样。

    从纯噪声出发，沿预测速度逐步去噪。
    steps=20 通常就够了，distilled 变体可以降到 1-4 步。

    Args:
        model: 训练好的 DiT 模型
        shape: 输出形状 (batch, channels, height, width)
        steps: Euler 积分步数
        device: 计算设备

    Returns:
        生成的样本张量
    """
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)

    for _ in range(steps):
        v = model(x, t)     # 预测当前时刻的速度
        x = x - dt * v      # 沿反方向前进一步
        t = t - dt           # 时间步递减

    return x


# === 第 7 步：合成数据集 ===
# 为了在 CPU 上快速验证，生成简单的彩色圆形斑点数据

def synthetic_blobs(num=200, size=16, seed=0):
    """生成合成彩色圆形斑点数据集。

    每张图片包含一个随机颜色、随机位置的圆形斑点。
    用于验证 DiT + Rectified Flow 管道是否能正常训练和生成。

    Args:
        num: 生成图片数量
        size: 图片尺寸（size x size）
        seed: 随机种子

    Returns:
        形状 (num, 3, size, size) 的张量，值在 [-1, 1]
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((num, 3, size, size), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")

    for i in range(num):
        # 随机圆心（远离边缘）和半径
        cx, cy = rng.uniform(4, size - 4, size=2)
        r = rng.uniform(2, 4)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2

        # 随机颜色
        colour = rng.uniform(-1, 1, size=3)
        for c in range(3):
            out[i, c][mask] = colour[c]

    return torch.from_numpy(out)


# === 第 8 步：端到端运行 ===
def main():
    torch.manual_seed(0)
    device = "cpu"

    # 生成合成数据
    data = synthetic_blobs(num=128, size=16)
    print(f"数据形状: {tuple(data.shape)}")
    print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")

    # 创建 TinyDiT 模型
    model = TinyDiT(
        image_size=16, patch_size=2, in_channels=3,
        dim=96, depth=4, heads=3
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # 训练 300 步
    batch_size = 32
    print(f"\n开始训练（{batch_size} 步 x {300 // 50} 次报告）...")
    for step in range(300):
        idx = np.random.choice(len(data), batch_size)
        x0 = data[idx]
        loss = rectified_flow_train_step(model, x0, optimizer, device)
        if step % 50 == 0:
            print(f"  step {step:3d}  rectified_flow_mse {loss:.4f}")

    # 采样测试：20 步（标准）
    print("\n[采样] 20 步 Euler（标准质量）")
    s20 = rectified_flow_sample(model, (4, 3, 16, 16), steps=20, device=device)
    print(f"  范围 [{s20.min():.2f}, {s20.max():.2f}]  "
          f"形状 {tuple(s20.shape)}")

    # 采样测试：4 步（schnell/turbo 风格）
    print("[采样] 4 步 Euler（schnell 风格，快速推理）")
    s4 = rectified_flow_sample(model, (4, 3, 16, 16), steps=4, device=device)
    print(f"  范围 [{s4.min():.2f}, {s4.max():.2f}]  "
          f"形状 {tuple(s4.shape)}")


if __name__ == "__main__":
    main()
