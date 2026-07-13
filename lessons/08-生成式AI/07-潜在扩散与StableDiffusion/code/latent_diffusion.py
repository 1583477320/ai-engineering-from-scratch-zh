# 潜在扩散模型简化实现
# 演示 Stable Diffusion 的三阶段架构：编码→扩散→解码

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 第 1 步：简化 VAE（编码器 + 解码器）
# ============================================================================

class SimpleVAEEncoder(nn.Module):
    """简化 VAE 编码器：将像素空间图像压缩到潜空间。"""

    def __init__(self, in_channels=3, latent_channels=4, base_channels=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),  # H/2
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),  # H/4
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, latent_channels, 3, padding=1),  # 保持 H/4
        )

    def forward(self, x):
        """编码：图像 → 潜向量。"""
        return self.layers(x)


class SimpleVAEDecoder(nn.Module):
    """简化 VAE 解码器：将潜空间向量还原为像素空间图像。"""

    def __init__(self, latent_channels=4, out_channels=3, base_channels=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),  # H*4
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, out_channels, 4, stride=2, padding=1),  # H*8
        )

    def forward(self, z):
        """解码：潜向量 → 图像。"""
        return self.layers(z)


# ============================================================================
# 第 2 步：带交叉注意力的简化 U-Net
# ============================================================================

class CrossAttention(nn.Module):
    """交叉注意力层——将文本条件注入图像特征。"""

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.query_dim = query_dim
        self.context_dim = context_dim or query_dim

        # Q、K、V 投影矩阵
        self.to_q = nn.Linear(query_dim, heads * dim_head)
        self.to_k = nn.Linear(context_dim, heads * dim_head)
        self.to_v = nn.Linear(context_dim, heads * dim_head)

        # 输出投影
        self.to_out = nn.Linear(heads * dim_head, query_dim)

    def forward(self, x, text_context):
        """
        交叉注意力前向传播。
        Args:
            x: 图像特征 (B, L, query_dim)
            text_context: 文本嵌入 (B, T, context_dim)
        Returns:
            注入文本条件后的图像特征 (B, L, query_dim)
        """
        h = self.heads

        # 投影为 Q、K、V
        q = self.to_q(x).reshape(x.size(0), -1, h, self.dim_head).transpose(1, 2)
        k = self.to_k(text_context).reshape(text_context.size(0), -1, h, self.dim_head).transpose(1, 2)
        v = self.to_v(text_context).reshape(text_context.size(0), -1, h, self.dim_head).transpose(1, 2)

        # 缩放点积注意力
        # shape: (B, heads, seq_len_q, seq_len_k)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 加权求和
        out = torch.matmul(attn_weights, v)  # (B, heads, seq_len_q, dim_head)
        out = out.transpose(1, 2).reshape(x.size(0), -1, h * self.dim_head)

        # 输出投影
        return self.to_out(out)


class SimpleUNetBlock(nn.Module):
    """简化 U-Net 块——包含残差卷积和交叉注意力。"""

    def __init__(self, channels, emb_channels, text_dim):
        super().__init__()
        # 残差卷积层
        self.res_conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, channels),
        )
        # 交叉注意力
        self.cross_attn = CrossAttention(channels, text_dim)
        # 输出卷积
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x, t_emb, text_ctx):
        """
        前向传播：
        1. 残差卷积
        2. 注入时间嵌入
        3. 交叉注意力注入文本条件
        4. 输出卷积
        """
        h = self.res_conv(x)
        # 注入时间嵌入
        t_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb
        # 转为 [B, L, C] 供交叉注意力使用
        B, C, H, W = h.shape
        h_flat = h.permute(0, 2, 3, 1).reshape(B, -1, C)
        # 交叉注意力
        h_flat = self.cross_attn(h_flat, text_ctx)
        # 转回 [B, C, H, W]
        h = h_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # 输出卷积
        return x + self.out_conv(h)


# ============================================================================
# 第 3 步：完整潜在扩散模型
# ============================================================================

class LatentDiffusionModel(nn.Module):
    """
    简化版潜在扩散模型——展示 Stable Diffusion 的三阶段架构：
    VAE 编码 → 潜空间扩散 → VAE 解码
    """

    def __init__(
        self,
        image_channels=3,
        latent_channels=4,
        latent_size=32,
        base_channels=64,
        text_dim=768,
        embed_dim=256,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.latent_channels = latent_channels

        # VAE
        self.vae_encoder = SimpleVAEEncoder(image_channels, latent_channels, base_channels)
        self.vae_decoder = SimpleVAEDecoder(latent_channels, image_channels, base_channels)

        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.SinusoidalPositionEmbeddings(embed_dim) if hasattr(nn, 'SinusoidalPositionEmbeddings') else nn.Identity(),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # 潜空间 U-Net
        self.unet_input = nn.Conv2d(latent_channels, base_channels, 3, padding=1)
        self.unet_block1 = SimpleUNetBlock(base_channels, embed_dim, text_dim)
        self.unet_block2 = SimpleUNetBlock(base_channels * 2, embed_dim, text_dim)
        self.unet_output = nn.Conv2d(base_channels, latent_channels, 1)

        # 文本嵌入投影
        self.text_projection = nn.Linear(text_dim, text_dim)

    def encode(self, image):
        """VAE 编码：将图像压缩到潜空间。"""
        return self.vae_encoder(image)

    def decode(self, latent):
        """VAE 解码：将潜向量还原为图像。"""
        return self.vae_decoder(latent)

    def forward(self, latent, timestep, text_embed):
        """
        潜空间扩散前向传播。
        Args:
            latent: 潜空间表示 (B, C, H, W)
            timestep: 时间步 (B,)
            text_embed: 文本嵌入 (B, T, D)
        """
        # 时间嵌入
        t_emb = self.time_embed(timestep)

        # U-Net 主干
        h = self.unet_input(latent)
        h = self.unet_block1(h, t_emb, self.text_projection(text_embed))
        h = self.unet_block2(h, t_emb, self.text_projection(text_embed))
        noise_pred = self.unet_output(h)

        return noise_pred


# ============================================================================
# 主程序：演示完整流程
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = LatentDiffusionModel(
        image_channels=3,
        latent_channels=4,
        latent_size=32,
        base_channels=32,
        text_dim=768,
        embed_dim=128,
    ).to(device)

    # 模拟输入
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 128, 128).to(device)  # 小图像用于测试
    dummy_text = torch.randn(batch_size, 77, 768).to(device)  # CLIP 文本嵌入 (77 词元)
    dummy_timestep = torch.randint(0, 1000, (batch_size,)).to(device)

    # 编码
    latent = model.encode(dummy_image)
    print(f"图像形状: {dummy_image.shape}")
    print(f"潜向量形状: {latent.shape}")

    # 扩散前向
    noise_pred = model(latent, dummy_timestep, dummy_text)
    print(f"预测噪声形状: {noise_pred.shape}")

    # 解码
    reconstructed = model.decode(latent)
    print(f"重建图像形状: {reconstructed.shape}")

    # 总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
