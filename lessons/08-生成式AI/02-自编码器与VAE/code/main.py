# main.py -- 从零实现自编码器与 VAE
# 依赖：torch>=2.0, numpy>=1.24
# 安装：pip install torch numpy
# 对应课程：阶段 08 · 02（自编码器与 VAE）

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 第 1 步：自编码器（Autoencoder）
# ============================================================

class Autoencoder(nn.Module):
    """最基础的自编码器——编码器压缩、解码器重建。"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # 编码器：输入 -> 潜在向量
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        # 解码器：潜在向量 -> 重建
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # 输出范围 [0, 1]，与归一化后的像素对齐
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# ============================================================
# 第 2 步：VAE（变分自编码器）
# ============================================================

class VAE(nn.Module):
    """变分自编码器——编码器输出均值和方差，通过重参数化采样。"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # 编码器输出 2 * latent_dim（均值 + 对数方差）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2 * latent_dim),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        """重参数化技巧：z = mu + sigma * epsilon。
        让采样操作可微分——梯度可以流过随机节点。"""
        std = torch.exp(0.5 * logvar)  # logvar -> 标准差
        eps = torch.randn_like(std)     # 从 N(0,1) 采样
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)  # 拆分为均值和对数方差
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


# ============================================================
# 第 3 步：损失函数
# ============================================================

def vae_loss(x_recon, x, mu, logvar):
    """VAE 损失 = 重建损失 + KL 散度正则化。"""
    # 重建损失：衡量重建与原始的差距
    recon_loss = F.mse_loss(x_recon, x, reduction="sum")
    # KL 散度：强制潜在分布接近 N(0, I)
    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def ae_loss(x_recon, x):
    """自编码器损失——只有重建损失。"""
    return F.mse_loss(x_recon, x, reduction="sum")


# ============================================================
# 第 4 步：合成数据——模拟手写数字的特征
# ============================================================

def make_synthetic_mnist(n_samples=2000, input_dim=784, n_classes=10, seed=42):
    """生成合成数据——每个类别是一个随机中心点加噪声。
    模拟 MNIST 的结构（10 个类别，784 维 = 28x28），但不需要下载数据集。"""
    rng = np.random.RandomState(seed)
    # 10 个类别的中心点
    centers = rng.randn(n_classes, input_dim) * 0.3
    labels = rng.randint(0, n_classes, size=n_samples)
    # 每个样本 = 类别中心 + 噪声
    data = centers[labels] + rng.randn(n_samples, input_dim) * 0.15
    # 归一化到 [0, 1]
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels)


# ============================================================
# 第 5 步：训练循环
# ============================================================

def train_model(model, train_loader, loss_fn, optimizer, n_epochs, is_vae=False):
    """通用训练循环，支持自编码器和 VAE。"""
    model.train()
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            if is_vae:
                x_recon, mu, logvar = model(batch_x)
                loss = loss_fn(x_recon, batch_x, mu, logvar)
            else:
                x_recon, z = model(batch_x)
                loss = loss_fn(x_recon, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"  轮次 {epoch+1:>3}/{n_epochs} | 平均损失: {avg_loss:.4f}")
    return losses


# ============================================================
# 第 6 步：生成与可视化
# ============================================================

def ascii_digit(vector, width=7):
    """将 49 维向量可视化为 7x7 ASCII 字符画。"""
    chars = " .:-=+*#%@"
    pixels = (vector[:width*width] * (len(chars) - 1)).clip(0, len(chars) - 1).astype(int)
    lines = []
    for row in range(width):
        line = "".join(chars[pixels[row * width + col]] for col in range(width))
        lines.append(f"  {line}")
    return "\n".join(lines)


def visualize_reconstruction(model, data, labels, is_vae=False, n_examples=5):
    """展示重建结果——对比原始和重建。"""
    model.eval()
    with torch.no_grad():
        if is_vae:
            x_recon, mu, logvar = model(data[:n_examples])
        else:
            x_recon, z = model(data[:n_examples])

    print(f"\n=== 重建对比 ({'VAE' if is_vae else '自编码器'}) ===")
    for i in range(n_examples):
        label = labels[i].item()
        print(f"\n样本 {i+1}（类别 {label}）:")
        print(f"  原始:")
        print(ascii_digit(data[i].numpy()))
        print(f"  重建:")
        print(ascii_digit(x_recon[i].numpy()))


def visualize_latent_space(model, data, labels, is_vae=False, n_samples=500):
    """可视化潜在空间——如果是 2 维，直接绘制散点图（ASCII）。"""
    model.eval()
    with torch.no_grad():
        if is_vae:
            _, mu, _ = model(data[:n_samples])
            z = mu.numpy()  # 用均值代替采样点
        else:
            _, z = model(data[:n_samples])
            z = z.numpy()

    if z.shape[1] == 2:
        print(f"\n=== 潜在空间散点图（{'VAE' if is_vae else '自编码器'}，前 100 个样本）===")
        # ASCII 散点图
        grid_size = 30
        grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]
        markers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        n = min(100, len(z))
        # 映射到网格坐标
        z_min, z_max = z[:n].min(axis=0), z[:n].max(axis=0)
        for i in range(n):
            x = int((z[i, 0] - z_min[0]) / (z_max[0] - z_min[0] + 1e-8) * (grid_size - 1))
            y = int((z[i, 1] - z_min[1]) / (z_max[1] - z_min[1] + 1e-8) * (grid_size - 1))
            row = grid_size - 1 - y
            col = x
            grid[row][col] = markers[labels[i].item()]
        for row in grid:
            print("  " + "".join(row))
        print("  [数字代表不同类别，理想状态是同类聚集、异类分离]")
    else:
        print(f"\n=== 潜在空间统计（维度 = {z.shape[1]}）===")
        for d in range(min(4, z.shape[1])):
            print(f"  维度 {d}: 均值={z[:, d].mean():.3f}, 标准差={z[:, d].std():.3f}")


def generate_from_vae(vae, n_samples=10, latent_dim=2):
    """从标准正态分布采样——VAE 的核心能力。"""
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        generated = vae.decoder(z)
    print(f"\n=== 从 N(0,I) 采样生成 {n_samples} 个新样本 ===")
    for i in range(min(n_samples, 5)):
        print(f"\n生成样本 {i+1}:")
        print(ascii_digit(generated[i].numpy()))


# ============================================================
# 主程序
# ============================================================

def main():
    # 超参数
    INPUT_DIM = 49       # 7x7 图像展平后的维度
    HIDDEN_DIM = 128
    LATENT_DIM = 2       # 2 维便于可视化
    BATCH_SIZE = 64
    N_EPOCHS = 30
    LEARNING_RATE = 1e-3

    print("=" * 55)
    print("  自编码器 vs VAE：从重建到生成")
    print("=" * 55)

    # 生成合成数据
    print("\n[1/6] 生成合成数据...")
    data, labels = make_synthetic_mnist(
        n_samples=2000, input_dim=INPUT_DIM, n_classes=10
    )
    dataset = TensorDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"  数据形状: {data.shape}（{data.shape[0]} 个样本，{INPUT_DIM} 维）")
    print(f"  类别数: {len(torch.unique(labels))}")

    # ---- 训练自编码器 ----
    print(f"\n[2/6] 训练自编码器...")
    ae = Autoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    ae_optim = torch.optim.Adam(ae.parameters(), lr=LEARNING_RATE)
    ae_losses = train_model(ae, train_loader, ae_loss, ae_optim, N_EPOCHS, is_vae=False)

    # ---- 训练 VAE ----
    print(f"\n[3/6] 训练 VAE...")
    vae = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    vae_optim = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    vae_losses = train_model(vae, train_loader, vae_loss, vae_optim, N_EPOCHS, is_vae=True)

    # ---- 重建对比 ----
    print(f"\n[4/6] 重建质量对比...")
    visualize_reconstruction(ae, data, labels, is_vae=False)
    visualize_reconstruction(vae, data, labels, is_vae=True)

    # ---- 潜在空间可视化 ----
    print(f"\n[5/6] 潜在空间可视化...")
    visualize_latent_space(ae, data, labels, is_vae=False)
    visualize_latent_space(vae, data, labels, is_vae=True)

    # ---- 生成新样本 ----
    print(f"\n[6/6] 生成新样本（仅 VAE 可以）...")
    generate_from_vae(vae, n_samples=10, latent_dim=LATENT_DIM)

    # ---- 训练曲线对比 ----
    print(f"\n=== 训练损失对比 ===")
    print(f"  自编码器最终损失: {ae_losses[-1]:.4f}")
    print(f"  VAE 最终损失:     {vae_losses[-1]:.4f}")
    print(f"  VAE 损失更高是正常的——KL 散度项牺牲了部分重建质量以获得生成能力")


if __name__ == "__main__":
    main()
