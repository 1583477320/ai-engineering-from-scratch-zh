# main.py -- 从零实现 DCGAN（生成对抗网络）
# 依赖：torch>=2.0, torchvision, numpy, matplotlib
# 安装：pip install torch torchvision matplotlib numpy
# 对应课程：阶段 04 * 09（GAN）

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def synthetic_circles(num=800, size=32, seed=0):
    """生成合成圆形数据集。"""
    rng = np.random.default_rng(seed)
    imgs = np.full((num, 3, size, size), -1.0, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    for i in range(num):
        r = rng.uniform(6, 10)
        cx, cy = rng.uniform(r, size - r, size=2)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.uniform(-0.3, 1.0, size=3)
        for c in range(3):
            imgs[i, c][mask] = color[c]
    return torch.from_numpy(imgs)


class Generator(nn.Module):
    """DCGAN 生成器：将噪声向量转换为合成图像。"""

    def __init__(self, z_dim=64, img_channels=3, feat=32):
        super().__init__()
        self.net = nn.Sequential(
            # (z_dim, 1, 1) -> (feat*4, 4, 4)
            nn.ConvTranspose2d(z_dim, feat * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(inplace=True),
            # (feat*4, 4, 4) -> (feat*2, 8, 8)
            nn.ConvTranspose2d(feat * 4, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(inplace=True),
            # (feat*2, 8, 8) -> (feat, 16, 16)
            nn.ConvTranspose2d(feat * 2, feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            # (feat, 16, 16) -> (img_channels, 32, 32)，值域 [-1, 1]
            nn.ConvTranspose2d(feat, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    """DCGAN 判别器：判断输入图像是真实还是伪造。"""

    def __init__(self, img_channels=3, feat=32, use_sn=False):
        super().__init__()
        layers = []

        def conv_layer(in_c, out_c, apply_bn):
            c = nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not apply_bn)
            if use_sn:
                from torch.nn.utils import spectral_norm
                c = spectral_norm(c)
            layers.append(c)
            if apply_bn and not use_sn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        conv_layer(img_channels, feat, apply_bn=False)
        conv_layer(feat, feat * 2, apply_bn=True)
        conv_layer(feat * 2, feat * 4, apply_bn=True)
        last = nn.Conv2d(feat * 4, 1, 4, 1, 0)
        if use_sn:
            from torch.nn.utils import spectral_norm
            last = spectral_norm(last)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


def train_step_bce(G, D, real, z, opt_g, opt_d, device, label_smoothing=1.0):
    """使用原始 BCE 损失的训练步。"""
    bs = real.size(0)
    ones = torch.ones(bs, device=device)
    zeros = torch.zeros(bs, device=device)
    # -- 更新判别器 --
    opt_d.zero_grad()
    d_real = D(real)
    d_fake = D(G(z).detach())
    loss_d_real = F.binary_cross_entropy_with_logits(d_real, ones * label_smoothing)
    loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, zeros)
    loss_d = loss_d_real + loss_d_fake
    loss_d.backward()
    opt_d.step()
    # -- 更新生成器（非饱和损失）--
    opt_g.zero_grad()
    d_fake_gen = D(G(z))
    loss_g = F.binary_cross_entropy_with_logits(d_fake_gen, ones)
    loss_g.backward()
    opt_g.step()
    return loss_d.item(), loss_g.item()


def train_step_wgan_gp(G, D, real, z, opt_g, opt_d, device, gp_lambda=10.0):
    """使用 WGAN-GP 损失的训练步。"""
    bs = real.size(0)
    # -- 更新判别器 --
    opt_d.zero_grad()
    d_real = D(real)
    d_fake = D(G(z).detach())
    loss_d = -(torch.mean(d_real) - torch.mean(d_fake))
    # 梯度惩罚
    alpha = torch.rand(bs, 1, 1, 1, device=device)
    interpolates = alpha * real + (1 - alpha) * G(z).detach()
    interpolates.requires_grad_(True)
    d_interp = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interp, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True,
    )[0]
    grad_norm = gradients.reshape(gradients.size(0), -1).pow(2).sum(dim=1, keepdim=True)
    grad_norm = grad_norm.sqrt()
    gp = gp_lambda * ((grad_norm - 1) ** 2).mean()
    loss_d = loss_d + gp
    loss_d.backward()
    opt_d.step()
    # -- 更新生成器 --
    opt_g.zero_grad()
    d_fake_gen = D(G(z))
    loss_g = -torch.mean(d_fake_gen)
    loss_g.backward()
    opt_g.step()
    return loss_d.item(), loss_g.item()


@torch.no_grad()
def sample_images(G, n=16, z_dim=64, device='cpu'):
    """采样固定数量的合成图像，用于监控训练过程。"""
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    imgs = G(z)
    G.train()
    return ((imgs + 1) / 2).clamp(0, 1)


def plot_training_curves(d_losses, g_losses, save_dir):
    """绘制训练曲线并保存到指定目录。"""
    if not HAS_MPL:
        print('matplotlib 不可用，跳过绘图。')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(d_losses, label='D Loss', color='tab:orange')
    axes[0].set_title('Discriminator Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(g_losses, label='G Loss', color='tab:blue')
    axes[1].set_title('Generator Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'training-curves.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f'训练曲线已保存至 {path}')


def save_sample_grid(samples, epoch, save_dir):
    """将样本保存为网格图。"""
    if not HAS_MPL:
        return
    os.makedirs(save_dir, exist_ok=True)
    from torchvision.utils import make_grid
    from PIL import Image
    grid = make_grid(samples[:16], nrow=4, normalize=True, pad_value=0.1)
    path = os.path.join(save_dir, f'samples_epoch_{epoch}.png')
    grid_np = grid.permute(1, 2, 0).mul(255).byte().cpu().numpy()
    Image.fromarray(grid_np).save(path)
    print(f'样本图已保存至 {path}')


def main():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z_dim = 64
    lr = 2e-4
    num_epochs = 15
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'samples')

    data = synthetic_circles(num=400)
    loader = DataLoader(TensorDataset(data), batch_size=32, shuffle=True)

    G = Generator(z_dim=z_dim, img_channels=3, feat=32).to(device)
    D = Discriminator(img_channels=3, feat=32, use_sn=True).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    print(f'G 参数量: {sum(p.numel() for p in G.parameters()):,}')
    print(f'D 参数量: {sum(p.numel() for p in D.parameters()):,}')
    print(f'设备: {device}')

    d_losses = []
    g_losses = []

    for epoch in range(num_epochs):
        ld_sum, lg_sum, n = 0.0, 0.0, 0
        for (batch,) in loader:
            z = torch.randn(batch.size(0), z_dim, device=device)
            ld, lg = train_step_wgan_gp(G, D, batch, z, opt_g, opt_d, device)
            ld_sum += ld
            lg_sum += lg
            n += 1
        avg_d = ld_sum / n
        avg_g = lg_sum / n
        d_losses.append(avg_d)
        g_losses.append(avg_g)
        print(f'epoch {epoch:>2d}  D {avg_d:.3f}  G {avg_g:.3f}')

        if epoch % 3 == 0 or epoch == num_epochs - 1:
            samples = sample_images(G, n=16, z_dim=z_dim, device=device)
            save_sample_grid(samples, epoch, save_dir)

    plot_training_curves(d_losses, g_losses, save_dir)

    model_dir = os.path.join(save_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(model_dir, 'generator.pt'))
    torch.save(D.state_dict(), os.path.join(model_dir, 'discriminator.pt'))
    print('模型已保存。')


if __name__ == '__main__':
    main()
