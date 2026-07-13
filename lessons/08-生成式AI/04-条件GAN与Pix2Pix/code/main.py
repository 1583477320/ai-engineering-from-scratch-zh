# 条件 GAN 与 Pix2Pix
# 依赖：torch>=2.0, numpy
# 安装：pip install torch numpy
# 对应课程：阶段 08 · 04（条件 GAN 与 Pix2Pix）

"""
Pix2Pix 风格的条件 GAN 实现。
生成器：U-Net（编码器-解码器 + 跳跃连接）
判别器：PatchGAN（对图像块分类，而非整图）
数据：合成配对数据（边缘→形状），无需真实数据集
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# === 超参数 ===
IMAGE_SIZE = 64          # 图像尺寸（64x64 灰度）
BATCH_SIZE = 32          # 批次大小
EPOCHS = 200             # 训练轮次
G_LR = 2e-4              # 生成器学习率
D_LR = 1e-4              # 判别器学习率（比 G 慢，避免 D 过强）
LAMBDA_L1 = 100          # L1 损失权重——像素级对齐的关键
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 第 1 步：合成数据集——边缘图 → 形状图（配对数据）
# ============================================================

class SyntheticEdgeDataset(Dataset):
    """生成合成的边缘图→形状图配对数据。

    源图像：简单的几何形状（矩形、圆形）的边缘图
    目标图像：对应的填充形状
    这模拟了 Pix2Pix 的核心场景——输入一种表示，输出另一种表示。
    """

    def __init__(self, n_samples=512, image_size=64):
        self.n_samples = n_samples
        self.image_size = image_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        np.random.seed(idx)  # 可复现

        target = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        source = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # 随机选择形状：矩形或圆形
        if np.random.rand() > 0.5:
            # 矩形
            h = np.random.randint(10, 30)
            w = np.random.randint(10, 30)
            y = np.random.randint(5, self.image_size - h - 5)
            x = np.random.randint(5, self.image_size - w - 5)
            target[y:y+h, x:x+w] = 1.0
            # 边缘检测：简单地将内部清零，只保留边界
            source[y:y+1, x:x+w] = 1.0   # 上边
            source[y+h-1:y+h, x:x+w] = 1.0  # 下边
            source[y:y+h, x:x+1] = 1.0   # 左边
            source[y:y+h, x+w-1:x+w] = 1.0  # 右边
        else:
            # 圆形
            r = np.random.randint(8, 25)
            cy = np.random.randint(r+2, self.image_size - r - 2)
            cx = np.random.randint(r+2, self.image_size - r - 2)
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if (i - cy)**2 + (j - cx)**2 <= r**2:
                        target[i, j] = 1.0
                    if r**2 <= (i - cy)**2 + (j - cx)**2 <= (r+1)**2:
                        source[i, j] = 1.0

        # 添加轻微噪声模拟真实场景
        source += np.random.randn(*source.shape).astype(np.float32) * 0.05

        # 增加通道维度：(1, H, W)
        source = np.clip(source, -1, 1)[np.newaxis, :, :]
        target = np.clip(target, -1, 1)[np.newaxis, :, :]

        return torch.tensor(source), torch.tensor(target)


# ============================================================
# 第 2 步：U-Net 生成器——编码器-解码器 + 跳跃连接
# ============================================================

class UNetDown(nn.Module):
    """U-Net 编码器块：Conv → BatchNorm → LeakyReLU。"""

    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    """U-Net 解码器块：ConvTranspose → BatchNorm → Dropout → ReLU。

    跳跃连接：将编码器对应层的输出拼接到解码器输入，
    恢复空间信息。这是 Pix2Pix 保持结构细节的关键。
    """

    def __init__(self, in_ch, out_ch, dropout=True):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))  # 编码器中间层用 Dropout 正则化
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip_input=None):
        """skip_input 来自编码器对应层的输出（可选）。"""
        x = self.block(x)
        if skip_input is not None:
            return torch.cat([x, skip_input], dim=1)  # 通道维度拼接
        return x


class UNetGenerator(nn.Module):
    """Pix2Pix 生成器：U-Net 架构。

    编码器逐步下采样提取特征，解码器逐步上采样恢复分辨率。
    跳跃连接将编码器的浅层特征直接传递到解码器，
    保留边缘和位置信息——这是生成锐利图像的关键。

    结构（64x64 输入）：
    编码器：1 → 64 → 128 → 256 → 512 → 512 → 512
    解码器：512 → 512 → 512 → 256 → 128 → 64 → 1
    """

    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        # 编码器：下采样（输入 → 潜在表示）
        self.down1 = UNetDown(in_ch, 64, normalize=False)  # 64x64 → 32x32
        self.down2 = UNetDown(64, 128)     # 32x32 → 16x16
        self.down3 = UNetDown(128, 256)    # 16x16 → 8x8
        self.down4 = UNetDown(256, 512)    # 8x8 → 4x4
        self.down5 = UNetDown(512, 512)    # 4x4 → 2x2
        self.down6 = UNetDown(512, 512)    # 2x2 → 1x1（瓶颈层）

        # 解码器：上采样 + 跳跃连接
        # 输入通道数 = 上采样输出 + 跳跃连接输入
        self.up1 = UNetUp(512, 512)          # d6(512) → 512, + skip d5(512) = 1024
        self.up2 = UNetUp(1024, 512)         # u1(1024) → 512, + skip d4(512) = 1024
        self.up3 = UNetUp(1024, 512)         # u2(1024) → 512, + skip d3(256) = 768
        self.up4 = UNetUp(768, 256, dropout=False)   # u3(768) → 256, + skip d2(128) = 384
        self.up5 = UNetUp(384, 128, dropout=False)   # u4(384) → 128, + skip d1(64) = 192
        self.up6 = UNetUp(192, 64, dropout=False)    # u5(192) → 64（无跳跃连接）

        # 最终输出层：Tanh 将输出压缩到 [-1, 1]
        self.final = nn.Sequential(
            nn.Conv2d(64, out_ch, 7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        """x: (batch, in_ch, H, W) → (batch, out_ch, H, W)"""
        # 编码器：逐层下采样，保存每层输出供跳跃连接使用
        d1 = self.down1(x)       # (batch, 64, 32, 32)
        d2 = self.down2(d1)      # (batch, 128, 16, 16)
        d3 = self.down3(d2)      # (batch, 256, 8, 8)
        d4 = self.down4(d3)      # (batch, 512, 4, 4)
        d5 = self.down5(d4)      # (batch, 512, 2, 2)
        d6 = self.down6(d5)      # (batch, 512, 1, 1)

        # 解码器：逐层上采样 + 跳跃连接
        u1 = self.up1(d6, d5)    # (batch, 1024, 2, 2)
        u2 = self.up2(u1, d4)    # (batch, 1024, 4, 4)
        u3 = self.up3(u2, d3)    # (batch, 768, 8, 8)
        u4 = self.up4(u3, d2)    # (batch, 384, 16, 16)
        u5 = self.up5(u4, d1)    # (batch, 192, 32, 32)
        u6 = self.up6(u5)        # (batch, 64, 64, 64) — 最后一层不需要跳跃连接

        return self.final(u6)    # (batch, 1, 64, 64)


# ============================================================
# 第 3 步：PatchGAN 判别器——对图像块分类
# ============================================================

class PatchGANDiscriminator(nn.Module):
    """PatchGAN 判别器：对图像的每个 70x70 patch 做真假分类。

    与传统 GAN 的判别器输出一个标量不同，PatchGAN 输出一个空间映射。
    每个输出值代表对应 patch（70x70 区域）是真还是假。

    优势：
    1. 参数更少（不需要全连接层）
    2. 可以处理任意大小的图像
    3. 更关注局部纹理和结构——适合图像翻译任务
    """

    def __init__(self, in_ch=2):
        """in_ch = 条件图像 + 目标图像 = 2（拼接后输入）。"""
        super().__init__()
        self.model = nn.Sequential(
            # 第 1 层：不使用归一化（判别器第一层惯例）
            nn.Conv2d(in_ch, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 第 2-4 层：逐步增加通道数
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 最终输出：每个位置的真假概率
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
            # 不加 Sigmoid——使用 BCEWithLogitsLoss 更稳定
        )

    def forward(self, condition, target):
        """将条件图像和目标图像在通道维度拼接后输入。

        Args:
            condition: 条件图像 (batch, 1, 64, 64)
            target: 目标图像 (batch, 1, 64, 64)
        Returns:
            patch_scores: 每个 patch 的真假分数 (batch, 1, ~7, ~7)
        """
        x = torch.cat([condition, target], dim=1)  # (batch, 2, 64, 64)
        return self.model(x)


# ============================================================
# 第 4 步：训练循环——对抗损失 + L1 损失
# ============================================================

def train_pix2pix():
    """Pix2Pix 训练主循环。

    损失函数 = 对抗损失 + L1 损失 × λ
    - 对抗损失：让生成器骗过判别器（图像真实感）
    - L1 损失：让生成的像素接近目标（结构对齐）
    λ = 100 表示 L1 损失是主导——这是 Pix2Pix 论文的关键发现
    """

    # 数据
    dataset = SyntheticEdgeDataset(n_samples=512, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型
    generator = UNetGenerator(in_ch=1, out_ch=1).to(DEVICE)
    discriminator = PatchGANDiscriminator(in_ch=2).to(DEVICE)

    # 优化器：判别器学习率比生成器小
    opt_g = optim.Adam(generator.parameters(), lr=G_LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=D_LR, betas=(0.5, 0.999))

    # 损失函数
    criterion_adv = nn.BCEWithLogitsLoss()  # 对抗损失
    criterion_l1 = nn.L1Loss()              # L1 损失（像素级对齐）

    print(f"开始训练 | 设备: {DEVICE} | 轮次: {EPOCHS}")
    print(f"生成器参数: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数: {sum(p.numel() for p in discriminator.parameters()):,}")
    print("-" * 60)

    # 计算判别器输出的 patch 映射大小（用于创建匹配的标签）
    with torch.no_grad():
        dummy_cond = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        dummy_tgt = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        patch_shape = discriminator(dummy_cond, dummy_tgt).shape[1:]  # 去掉 batch 维度

    print(f"PatchGAN 输出形状: {patch_shape}（每个位置对应一个图像块）")

    for epoch in range(EPOCHS):
        total_loss_d = 0.0
        total_loss_g = 0.0
        num_batches = 0

        for condition, target in dataloader:
            condition = condition.to(DEVICE)
            target = target.to(DEVICE)
            batch_size = condition.size(0)

            # 真/假标签（全 1 / 全 0）
            real_label = torch.ones(batch_size, *patch_shape, device=DEVICE)
            fake_label = torch.zeros(batch_size, *patch_shape, device=DEVICE)

            # ---------- 训练判别器 ----------
            fake = generator(condition).detach()  # detach 阻止梯度回传

            d_real = discriminator(condition, target)
            d_fake = discriminator(condition, fake)

            loss_d = (criterion_adv(d_real, real_label) +
                      criterion_adv(d_fake, fake_label)) / 2

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ---------- 训练生成器 ----------
            fake = generator(condition)
            d_fake = discriminator(condition, fake)

            # 对抗损失：让 D 认为假的是真的
            loss_g_adv = criterion_adv(d_fake, real_label)
            # L1 损失：像素级对齐——让生成图像在结构上接近目标
            loss_g_l1 = criterion_l1(fake, target) * LAMBDA_L1

            loss_g = loss_g_adv + loss_g_l1

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            total_loss_d += loss_d.item()
            total_loss_g += loss_g.item()
            num_batches += 1

        avg_d = total_loss_d / num_batches
        avg_g = total_loss_g / num_batches
        if (epoch + 1) % 20 == 0:
            print(f"轮次 [{epoch+1:3d}/{EPOCHS}] "
                  f"D: {avg_d:.4f} | G: {avg_g:.4f}")

    # 保存模型
    torch.save(generator.state_dict(), "pix2pix_generator.pth")
    torch.save(discriminator.state_dict(), "pix2pix_discriminator.pth")
    print(f"\n模型已保存")

    return generator


# ============================================================
# 第 5 步：推理——给定条件图像生成输出
# ============================================================

def demo_generate(generator, n_samples=4):
    """从训练好的生成器中推理，展示条件生成的效果。"""
    generator.eval()
    dataset = SyntheticEdgeDataset(n_samples=n_samples, image_size=IMAGE_SIZE)

    with torch.no_grad():
        for i in range(n_samples):
            condition, target = dataset[i]
            condition = condition.unsqueeze(0).to(DEVICE)
            generated = generator(condition)

            # 打印条件图像和生成图像的统计信息
            cond_sum = condition.sum().item()
            gen_sum = generated.sum().item()
            tgt_sum = target.sum().item()
            print(f"样本 {i+1}: 条件像素和={cond_sum:.1f}, "
                  f"生成像素和={gen_sum:.1f}, 目标像素和={tgt_sum:.1f}")

    print("\n提示：像素和越接近目标，说明生成器学会了条件映射")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    generator = train_pix2pix()
    demo_generate(generator)
