# GAN——生成器与判别器
# 依赖：torch>=2.0, torchvision>=0.15, numpy
# 安装：pip install torch torchvision numpy
# 对应课程：阶段 08 · 03（GAN 生成器与判别器）

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# === 超参数 ===
LATENT_DIM = 64        # 噪声向量维度
HIDDEN_DIM = 256       # 隐藏层维度
IMAGE_DIM = 28 * 28    # MNIST 展平后的维度
BATCH_SIZE = 128       # 批次大小
G_LR = 2e-4            # 生成器学习率
D_LR = 1e-4            # 判别器学习率（通常比 G 慢 2-5 倍）
NUM_EPOCHS = 30        # 训练轮次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 第 1 步：生成器 —— 从噪声生成图像
# ============================================================

class Generator(nn.Module):
    """生成器：将随机噪声 z 映射为 28x28 图像。

    架构：z → 全连接 → ReLU → 全连接 → Sigmoid
    Sigmoid 将输出压缩到 [0, 1]，与像素值范围匹配。
    """

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )

    def forward(self, z):
        """z 的形状: (batch, latent_dim)"""
        return self.net(z)


# ============================================================
# 第 2 步：判别器 —— 区分真假图像
# ============================================================

class Discriminator(nn.Module):
    """判别器：将图像映射为"真"或"假"的概率。

    架构：x → 全连接 → LeakyReLU → 全连接 → Sigmoid
    输出 [0, 1]——接近 1 表示"真"，接近 0 表示"假"。
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x 的形状: (batch, input_dim)"""
        return self.net(x)


# ============================================================
# 第 3 步：训练循环
# ============================================================

def train_gan():
    """GAN 训练主循环——交替训练 D 和 G。"""

    # 数据加载：MNIST 手写数字
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 缩放到 [-1, 1]
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 初始化模型
    generator = Generator(LATENT_DIM, HIDDEN_DIM, IMAGE_DIM).to(DEVICE)
    discriminator = Discriminator(IMAGE_DIM, HIDDEN_DIM).to(DEVICE)

    # 优化器：判别器学习率比生成器小，避免 D 训练过快导致 G 梯度消失
    opt_g = optim.Adam(generator.parameters(), lr=G_LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=D_LR, betas=(0.5, 0.999))

    # 损失函数：二元交叉熵（BCE）
    criterion = nn.BCELoss()

    # 模式坍塌检测：记录生成样本的多样性
    mode_collapse_history = []

    print(f"开始训练 | 设备: {DEVICE} | 轮次: {NUM_EPOCHS}")
    print(f"生成器参数: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数: {sum(p.numel() for p in discriminator.parameters()):,}")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        total_loss_d = 0.0
        total_loss_g = 0.0
        num_batches = 0

        for real_images, _ in dataloader:
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1).to(DEVICE)  # 展平

            # 真/假标签（带标签平滑：真标签用 0.9 而非 1.0，提升训练稳定性）
            real_labels = torch.ones(batch_size, 1, device=DEVICE) * 0.9
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

            # ---------- 训练判别器 ----------
            # D 的目标：真图像输出接近 1，假图像输出接近 0
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            fake_images = generator(z).detach()  # detach：不回传梯度到 G

            d_real = discriminator(real_images)
            d_fake = discriminator(fake_images)

            loss_d_real = criterion(d_real, real_labels)
            loss_d_fake = criterion(d_fake, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) / 2

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ---------- 训练生成器 ----------
            # G 的目标：让 D 认为假图像是真的
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            fake_images = generator(z)
            d_fake = discriminator(fake_images)

            # 非饱和损失：最大化 log D(G(z)) → 最小化 -log D(G(z))
            # 使用真标签——G 希望 D 输出接近 1
            loss_g = criterion(d_fake, real_labels)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            total_loss_d += loss_d.item()
            total_loss_g += loss_g.item()
            num_batches += 1

        # 打印每轮的平均损失
        avg_d = total_loss_d / num_batches
        avg_g = total_loss_g / num_batches
        print(f"轮次 [{epoch+1:2d}/{NUM_EPOCHS}] "
              f"D 损失: {avg_d:.4f} | G 损失: {avg_g:.4f}")

        # ---------- 模式坍塌检测 ----------
        # 每 5 轮生成一批样本，检查输出的多样性
        if (epoch + 1) % 5 == 0:
            diversity = check_mode_collapse(generator, device=DEVICE)
            mode_collapse_history.append((epoch + 1, diversity))
            if diversity < 0.1:
                print(f"  [!] 警告：模式坍塌——生成多样性指数仅 {diversity:.4f}")

        # 保存生成样本
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                z = torch.randn(64, LATENT_DIM, device=DEVICE)
                samples = generator(z).view(-1, 1, 28, 28)
                save_image(samples, f"samples_epoch_{epoch+1:03d}.png",
                           nrow=8, padding=2)

    # 保存模型
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print(f"\n模型已保存 | 最终 D 损失: {avg_d:.4f} | 最终 G 损失: {avg_g:.4f}")

    return generator, discriminator, mode_collapse_history


# ============================================================
# 第 4 步：模式坍塌检测
# ============================================================

def check_mode_collapse(generator, n_samples=1000, device="cpu", n_bins=10):
    """通过统计生成像素的分布来检测模式坍塌。

    健康的 GAN 生成的像素值应该接近均匀分布（多样性高）。
    模式坍塌时，像素值集中在少数几个值附近（多样性低）。

    返回值：归一化熵——越接近 1 越多样，越接近 0 越坍塌。
    """
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM, device=device)
        fake_images = generator(z).cpu().numpy().flatten()

    # 将像素值离散化到 n_bins 个区间
    hist, _ = torch.histc(torch.tensor(fake_images), bins=n_bins, min=0.0, max=1.0)
    hist = hist / hist.sum()  # 归一化为概率分布

    # 计算归一化熵：H / H_max
    # H = -sum(p * log(p)), H_max = log(n_bins)
    # 熵为 0 表示完全坍塌，熵为 1 表示均匀分布
    nonzero = hist[hist > 0]
    entropy = -(nonzero * torch.log(nonzero)).sum()
    max_entropy = math.log(n_bins)
    normalized_entropy = entropy / max_entropy

    generator.train()
    return normalized_entropy.item()


# ============================================================
# 第 5 步：用训练好的生成器采样
# ============================================================

def generate_samples(generator, n_samples=16):
    """从训练好的生成器中采样新图像。"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
        fake_images = generator(z).view(-1, 1, 28, 28)
        save_image(fake_images, "final_samples.png", nrow=4, padding=2)
        print(f"已生成 {n_samples} 张样本 → final_samples.png")


# ============================================================
# 第 6 步：WGAN 损失（可选实验）
# ============================================================

def train_wgan_loss(generator, discriminator, dataloader,
                    n_critic=5, clip_value=0.01):
    """WGAN 训练——用 Wasserstein 距离替代 BCE。

    WGAN 优势：
    1. 损失与生成质量相关（可作为训练进度的指标）
    2. 不需要模式坍塌检测（Wasserstein 距离天然鼓励多样性）
    3. 训练更稳定

    注意：WGAN 需要裁剪判别器权重（权重裁剪）或使用梯度惩罚（WGAN-GP）。
    """
    opt_g = optim.Adam(generator.parameters(), lr=G_LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=D_LR, betas=(0.5, 0.999))

    for epoch in range(NUM_EPOCHS):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1).to(DEVICE)

            # ---------- 训练判别器（Critic） ----------
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            fake_images = generator(z).detach()

            # WGAN 损失：D 希望 real 高、fake 低
            d_real = discriminator(real_images).mean()
            d_fake = discriminator(fake_images).mean()
            loss_d = d_fake - d_real  # Wasserstein 距离的负数

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # 权重裁剪：将判别器权重限制在 [-clip_value, clip_value]
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # 每 n_critic 步更新一次生成器
            if i % n_critic == 0:
                z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                fake_images = generator(z)
                loss_g = -discriminator(fake_images).mean()

                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

        if (epoch + 1) % 5 == 0:
            print(f"WGAN 轮次 [{epoch+1}/{NUM_EPOCHS}] D: {loss_d.item():.4f} G: {loss_g.item():.4f}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 训练标准 GAN
    generator, discriminator, mode_history = train_gan()

    # 生成最终样本
    generate_samples(generator)

    # 打印模式坍塌检测历史
    print("\n模式坍塌检测历史（归一化熵，越接近 1 越多样）:")
    for epoch, diversity in mode_history:
        bar = "#" * int(diversity * 40)
        print(f"  轮次 {epoch:3d}: {diversity:.4f} |{bar}")

    print("\n提示：尝试以下实验来加深理解：")
    print("  1. 将 D_LR 设为 5 * G_LR，观察 G 损失崩溃")
    print("  2. 将 train_gan() 替换为 WGAN 损失（调用 train_wgan_loss）")
    print("  3. 减小 HIDDEN_DIM 到 64，观察生成质量下降")
