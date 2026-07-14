# 神经音频编解码器概念实现
# 演示 RVQ（残差向量量化）和音频 token 编解码

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# 第 1 步：RVQ（残差向量量化）
# ============================================================================

class ResidualVectorQuantization(nn.Module):
    """
    残差向量量化——EnCodec 的核心组件。
    使用多级码本（quantizers），每级量化上一级的残差。

    RVQ 比单级量化的优势：
    - 用多个小码本替代一个大码本（训练更容易）
    - 可扩展：只用前 K 级解码即可获得不同质量的音频
    - 信息分级：第一级编码大结构，后续级编码精细细节
    """

    def __init__(self, dim=256, num_quantizers=4, codebook_size=1024):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.dim = dim

        # 多级码本（每个量化器有自己的码本）
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in range(num_quantizers)
        ])

        # 码本初始化（使用正态分布）
        for cb in self.codebooks:
            nn.init.normal_(cb.weight, mean=0.0, std=1.0 / np.sqrt(dim))

    def forward(self, z):
        """
        多级量化：逐级量化残差。
        Args:
            z: 输入潜向量 (B, D, T)
        Returns:
            z_q: 量化后的潜向量 (B, D, T)
            all_indices: 每级量化的码本索引 (num_quantizers, B, T)
        """
        B, D, T = z.shape
        # 转置为 (B, T, D) 便于向量运算
        z = z.permute(0, 2, 1).contiguous()  # (B, T, D)
        residual = z  # 当前残差
        z_q = torch.zeros_like(z)  # 累加量化输出
        all_indices = []

        for level in range(self.num_quantizers):
            # 量化当前残差
            codebook = self.codebooks[level].weight  # (codebook_size, D)

            # 对每个向量找到最近的码本条
            # 码本: (1, codebook_size, D) 扩展为 (B, codebook_size, D)
            # z: (B, T, D) -> dist: (B, T, codebook_size)
            z_exp = residual.unsqueeze(2)  # (B, T, 1, D)
            cb_exp = codebook.unsqueeze(0).unsqueeze(0)  # (1, 1, codebook_size, D)
            dist = torch.cdist(z_exp.float(), cb_exp.float(), p=2.0).squeeze(2)  # (B, T, codebook_size)
            # 实际简化计算：用 squared 距离
            # dist = (z_exp - cb_exp).pow(2).sum(dim=-1)  # (B, T, codebook_size)

            indices = dist.argmin(dim=-1)  # (B, T)
            # 查找码本向量
            z_quantized = codebook[indices]  # (B, T, D)

            # 累加量化输出
            z_q = z_q + z_quantized
            # 更新残差
            residual = residual - z_quantized

            all_indices.append(indices)

        all_indices = torch.stack(all_indices, dim=0)  # (num_quantizers, B, T)
        z_q = z_q.permute(0, 2, 1)  # (B, D, T)

        return z_q, all_indices

    def decode(self, indices):
        """
        从码本索引解码。
        Args:
            indices: 码本索引 (num_quantizers, B, T)
        Returns:
            z_q: 解码后的潜向量 (B, D, T)
        """
        z_q = 0
        for level in range(self.num_quantizers):
            codebook = self.codebooks[level].weight
            z_q = z_q + codebook[indices[level]]  # (B, T, D)

        return z_q.permute(0, 2, 1)  # (B, D, T)

    def encode(self, z):
        """
        编码：只返回索引（用于存储或传输）。
        """
        B, D, T = z.shape
        z = z.permute(0, 2, 1).contiguous()
        residual = z
        all_indices = []

        for level in range(self.num_quantizers):
            codebook = self.codebooks[level].weight
            z_exp = residual.unsqueeze(2)
            cb_exp = codebook.unsqueeze(0).unsqueeze(0)
            dist = torch.cdist(z_exp.float(), cb_exp.float(), p=2.0).squeeze(2)
            indices = dist.argmin(dim=-1)
            z_quantized = codebook[indices]
            residual = residual - z_quantized
            all_indices.append(indices)

        return torch.stack(all_indices, dim=0)


# ============================================================================
# 第 2 步：简化版音频编解码器
# ============================================================================

class SimpleAudioCodec(nn.Module):
    """
    简化版神经音频编解码器。
    编码器：音频 → 潜向量 → 量化索引
    解码器：量化索引 → 潜向量 → 音频重建
    """

    def __init__(self, input_dim=80, hidden_dim=256, num_quantizers=4,
                 codebook_size=1024):
        super().__init__()
        # 编码器（1D 卷积层 + GRU）
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Tanh(),
        )

        # RVQ 量化器
        self.quantizer = ResidualVectorQuantization(
            dim=hidden_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, input_dim, 3, padding=1),
        )

    def forward(self, mel):
        """
        完整编解码流程（用于训练）。
        Args:
            mel: 梅尔谱图 (B, input_dim, T)
        Returns:
            mel_recon: 重建的梅尔谱图 (B, input_dim, T)
            indices: 码本索引 (num_quantizers, B, T')
        """
        z = self.encoder(mel)  # (B, hidden_dim, T)
        z_q, indices = self.quantizer(z)  # 量化
        mel_recon = self.decoder(z_q)  # (B, input_dim, T)
        return mel_recon, indices

    def encode_to_indices(self, mel):
        """编码为 token 索引——用于推理。"""
        z = self.encoder(mel)
        return self.quantizer.encode(z)

    def decode_from_indices(self, indices, output_length=None):
        """从 token 索引解码为梅尔谱图。"""
        z_q = self.quantizer.decode(indices)
        return self.decoder(z_q)


# ============================================================================
# 第 3 步：训练损失
# ============================================================================

def codec_loss(mel_recon, mel_target, z_q, z, indices, codebooks):
    """
    编解码器训练损失：重建损失 + 码本损失 + 承诺损失。
    """
    # 1. 重建损失（L1 + L2 混合）
    l1_loss = F.l1_loss(mel_recon, mel_target)
    l2_loss = F.mse_loss(mel_recon, mel_target)
    recon_loss = l1_loss + l2_loss

    # 2. 码本损失（仅更新码本向量）
    # 真实实现较复杂，简化版跳过

    # 3. 承诺损失（仅更新编码器输出）
    # 让编码器输出"承诺"去接近选择的码本向量
    # simplified: just use recon_loss

    return recon_loss


# ============================================================================
# 第 4 步：压缩率统计
# ============================================================================

def compute_compression_ratio(input_frames, num_quantizers, codebook_size):
    """
    计算 Codec 的压缩率。
    Args:
        input_frames: 输入音频帧数（如 1000 帧 @ 16kHz = 62.5ms）
        num_quantizers: 量化器数量
        codebook_size: 码本大小
    Returns:
        bits_per_second: 比特率 (bps)
        compression_ratio: 压缩比（相对于 16-bit PCM）
    """
    # PCM 比特率：44100 Hz × 16 bit = 705600 bps
    pcm_bitrate = 44100 * 16

    # EnCodec 在 24kHz 下的压缩
    # 每帧 320 个采样点，每秒 75 帧
    frames_per_second = 75
    bits_per_frame = num_quantizers * np.log2(codebook_size)  # log2(1024) ≈ 10
    codec_bitrate = frames_per_second * bits_per_frame

    compression_ratio = pcm_bitrate / codec_bitrate

    return codec_bitrate, compression_ratio


# ============================================================================
# 主程序：演示
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 创建编解码器
    codec = SimpleAudioCodec(
        input_dim=80,      # 梅尔谱图通道数
        hidden_dim=256,
        num_quantizers=4,  # 4 级 RVQ
        codebook_size=1024, # 每级码本 1024
    ).to(device)

    # 模拟输入：一段梅尔谱图
    batch_size = 2
    dummy_mel = torch.randn(batch_size, 80, 200).to(device)  # 200 帧

    # 前向传播
    mel_recon, indices = codec(dummy_mel)
    print(f"输入形状: {dummy_mel.shape}")
    print(f"重建形状: {mel_recon.shape}")
    print(f"码本索引形状: {indices.shape}")  # (num_quantizers, B, T)

    # 索引信息
    B = dummy_mel.size(0)
    T_indices = indices.size(2)
    total_tokens = B * T_indices
    storage_bits = indices.numel() * np.log2(1024)  # 每级码本 1024

    print(f"\n编解码统计:")
    print(f"  输入帧数 (T): {dummy_mel.size(2)}")
    print(f"  RVQ 量化器数量: {codec.quantizer.num_quantizers}")
    print(f"  输出索引数量: {T_indices} 帧 × {codec.quantizer.num_quantizers} 级 = {indices.numel()} 个 token")
    print(f"  每级码本大小: 1024 (≈10 比特)")

    # 压缩率
    bitrate, ratio = compute_compression_ratio(
        input_frames=200,
        num_quantizers=codec.quantizer.num_quantizers,
        codebook_size=1024,
    )
    print(f"\n压缩率:")
    print(f"  Codec 比特率: {bitrate:.0f} bps")
    print(f"  PCM 比特率 (44.1kHz×16bit): {44100 * 16} bps")
    print(f"  压缩比: {ratio:.0f}:1")

    print("\n完成！")
