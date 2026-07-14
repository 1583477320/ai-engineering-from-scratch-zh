# 视频扩散模型简化实现
# 演示时空 patches + Transformer 的架构

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# 第 1 步：视频数据预处理
# ============================================================================

def preprocess_video_frames(frames, target_size=(64, 64), num_frames=16):
    """
    将视频帧序列预处理为模型输入。
    Args:
        frames: 原始帧列表，每个 (H, W, 3)，值域 [0, 255]
        target_size: 目标尺寸 (H, W)
        num_frames: 采样的帧数
    Returns:
        video_tensor: (C, T, H, W)，值域 [-1, 1]
    """
    # 均匀采样帧
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]

    processed = []
    for frame in frames:
        # 调整尺寸
        if hasattr(frame, 'resize'):
            frame = frame.resize(target_size[::-1])  # PIL resize 使用 (W, H)
            frame_np = np.array(frame).astype(np.float32) / 127.5 - 1.0
        else:
            # numpy 数组
            from PIL import Image
            img = Image.fromarray(frame)
            img = img.resize(target_size[::-1])
            frame_np = np.array(img).astype(np.float32) / 127.5 - 1.0
        processed.append(frame_np)

    # (T, H, W, C) -> (C, T, H, W)
    video = np.stack(processed, axis=0)
    return torch.from_numpy(video.transpose(3, 0, 1, 2))


# ============================================================================
# 第 2 步：时空 Patches 嵌入
# ============================================================================

class SpaceTimePatchEmbedding(nn.Module):
    """将视频切分为时空 patches 并嵌入为向量——Sora 的核心组件。"""

    def __init__(self, in_channels=4, patch_size=2, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        # 3D 卷积：同时在时间和空间维度切分 patch
        # 输入: (B, C, T, H, W)
        # 输出: (B, embed_dim, T//ps, H//ps, W//ps)
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: 视频潜向量 (B, C, T, H, W)
        Returns:
            patches: (B, num_patches, embed_dim)
        """
        # 3D 卷积切分 patches
        x = self.proj(x)  # (B, D, T', H', W')
        B, D = x.shape[0], x.shape[1]
        # 展平为序列: (B, num_patches, embed_dim)
        # num_patches = T' * H' * W'
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', D)
        x = self.norm(x)
        return x

    def get_output_shape(self, input_shape):
        """计算输出的序列长度。"""
        C, T, H, W = input_shape
        T_out = T // self.patch_size
        H_out = H // self.patch_size
        W_out = W // self.patch_size
        return T_out * H_out * W_out


# ============================================================================
# 第 3 步：时间步嵌入
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码——用于时间步嵌入。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ============================================================================
# 第 4 步：简化版视频扩散 Transformer
# ============================================================================

class SimpleVideoTransformerBlock(nn.Module):
    """简化版 Transformer 块——处理时空 patches。"""

    def __init__(self, embed_dim=768, num_heads=8, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Linear(embed_dim * ff_mult, embed_dim),
        )

    def forward(self, x):
        """自注意力 + 前馈网络。"""
        # 自注意力
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x

        return x


class SimpleVideoDiffusion(nn.Module):
    """简化版视频扩散模型——展示时空 patches + Transformer 的架构。"""

    def __init__(self, in_channels=4, embed_dim=256, num_heads=4,
                 num_layers=4, patch_size=2):
        super().__init__()
        self.patch_size = patch_size

        # 时空 patches 嵌入
        self.patch_embed = SpaceTimePatchEmbedding(
            in_channels, patch_size=patch_size, embed_dim=embed_dim,
        )

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Transformer 层
        self.transformer = nn.ModuleList([
            SimpleVideoTransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # 输出投影：还原为 patches
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, in_channels * patch_size ** 3),
        )

    def forward(self, noisy_video, timestep):
        """
        前向传播：预测噪声。
        Args:
            noisy_video: 带噪视频 (B, C, T, H, W)
            timestep: 时间步 (B,)
        Returns:
            predicted_noise: 预测的噪声 (B, C, T, H, W)
        """
        B, C, T, H, W = noisy_video.shape

        # 时空 patches 嵌入
        x = self.patch_embed(noisy_video)  # (B, num_patches, embed_dim)

        # 时间步嵌入
        t_emb = self.time_embed(timestep)  # (B, embed_dim)
        x = x + t_emb.unsqueeze(1)  # 广播加到每个 patch

        # Transformer 处理
        for block in self.transformer:
            x = block(x)

        # 输出投影
        x = self.output_proj(x)  # (B, num_patches, C * patch_size^3)

        # 还原为视频形状
        # patch_size^3 = ps^3
        ps = self.patch_size
        x = x.reshape(B, T // ps, H // ps, W // ps, C, ps, ps, ps)
        x = x.permute(0, 5, 1, 6, 2, 7, 3, 4)  # (B, C, T//ps, ps, H//ps, ps, W//ps, ps)
        x = x.reshape(B, C, T, H, W)

        return x


# ============================================================================
# 第 5 步：训练循环
# ============================================================================

def train_video_diffusion(model, dataloader, num_epochs=5, lr=1e-4):
    """训练视频扩散模型。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        model.train()

        for batch_video in dataloader:
            batch_video = batch_video.to(device)
            B, C, T, H, W = batch_video.shape

            # 随机采样时间步
            t = torch.randint(0, 1000, (B,), device=device)

            # 添加噪声
            noise = torch.randn_like(batch_video)
            alpha_bar = torch.linspace(0.999, 0.001, 1000, device=device)
            ab = alpha_bar[t].view(-1, 1, 1, 1, 1)
            noisy_video = ab * batch_video + (1 - ab).sqrt() * noise

            # 预测噪声
            pred_noise = model(noisy_video, t)
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
# 主程序：演示
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 创建模型
    model = SimpleVideoDiffusion(
        in_channels=4,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        patch_size=2,
    ).to(device)

    # 模拟输入：一段 8 帧的 64×64 视频
    batch_size = 2
    dummy_video = torch.randn(batch_size, 4, 8, 64, 64).to(device)
    dummy_t = torch.randint(0, 1000, (batch_size,)).to(device)

    # 前向传播
    with torch.no_grad():
        output = model(dummy_video, dummy_t)

    print(f"输入形状: {dummy_video.shape}")
    print(f"输出形状: {output.shape}")
    assert output.shape == dummy_video.shape, "输入输出形状不匹配！"

    # 时空 patches 信息
    patch_embed = SpaceTimePatchEmbedding(in_channels=4, patch_size=2, embed_dim=128)
    num_patches = patch_embed.get_output_shape((4, 8, 64, 64))
    print(f"\n时空 patches 信息:")
    print(f"  输入视频: (4, 8, 64, 64)")
    print(f"  Patch 大小: 2×2×2")
    print(f"  序列长度: {num_patches} patches")
    print(f"  每个 patch 包含: 2帧 × 16像素 × 16像素 = 512 个像素值")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")

    print("\n完成！")
