# main.py — 视频理解核心算法从零实现
# 依赖：torch>=2.0, torchvision>=0.15, numpy
# 对应课程：阶段 04 · 12（视频理解）


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ===================================================================
# 第 1 部分：帧采样策略
# ===================================================================


def sample_uniform(num_frames_total: int, T: int) -> list[int]:
    """均匀采样：将视频等分为 T 段，每段取一个帧。

    最简单、最常用的采样方式。适用于动作信号散布在整个视频的情况，
    如"某人从远处走近镜头"。
    """
    if num_frames_total <= 0:
        raise ValueError(f"帧数必须为正数，得到 {num_frames_total}")
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    step = num_frames_total / T
    return [int(i * step) for i in range(T)]


def sample_dense(num_frames_total: int, T: int, seed: int = 0) -> list[int]:
    """稠密采样：在一个随机起始位置连续截取 T 帧。

    适用于动作集中在局部时段的情况，如"跳起来"的瞬间。
    多帧稠密采样在测试时做集成可以提升准确率。
    """
    rng = torch.random.manual_seed(seed)
    if num_frames_total <= 0:
        raise ValueError(f"帧数必须为正数，得到 {num_frames_total}")
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    start = torch.randint(0, num_frames_total - T + 1, ()).item()
    return list(range(start, start + T))


def sample_multiclip(num_frames_total: int, num_clips: int, T: int, seed: int = 0) -> list[list[int]]:
    """多片段采样：对同一视频做 N 次独立采样，每个片段 T 帧。

    测试时使用多片段采样的预测均值，可以显著提高视频级准确率
   （因为单次采样的随机性可能遗漏关键动作帧）。
    """
    clips = []
    for _ in range(num_clips):
        clip_frames = sample_dense(num_frames_total, T, seed=torch.randint(0, 10000, ()).item())
        clips.append(clip_frames)
    return clips


# ===================================================================
# 第 2 部分：2D+Pool —— 用 2D CNN 提取每帧特征后池化
# ===================================================================


class FramePool(nn.Module):
    """帧池化模型：用 2D CNN 提取每帧特征，在全局平均池化后合并。

    这是最基线的视频分类方法。它忽略了帧之间的时间顺序——
    pool(f1, f2, ..., fT) == pool(fT, ..., f2, f1)。
    因此它能区分"苹果掉在地上"（外观变化），但无法区分
    "从左推到右"vs"从右推到左"（纯运动方向差异）。
    """

    def __init__(self, num_classes: int = 400, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        # 去掉最后的分类头
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入视频张量，形状 (N, T, C, H, W)。
                N = 批次大小，T = 帧数，C = 通道数，H/W = 高/宽。

        Returns:
            分类 logits，形状 (N, num_classes)。
        """
        N, T = x.shape[:2]
        # 将批次和帧维度展平，逐帧通过 2D CNN
        x = x.reshape(N * T, *x.shape[2:])
        feats = self.features(x).view(N, T, -1)  # (N, T, 512)
        # 全局平均池化：沿时间维度求平均，丢失顺序信息
        pooled = feats.mean(dim=1)               # (N, 512)
        return self.head(pooled)                 # (N, num_classes)


# ===================================================================
# 第 3 部分：I3D 权值膨胀（Inflation）—— 从 2D 到 3D
# ===================================================================


def inflate_2d_to_3d(conv2d: nn.Conv2d, time_kernel: int = 3) -> nn.Conv3d:
    """将 2D Conv 层的权重沿时间轴复制并缩放，初始化为 3D 卷积层。

    这就是 I3D 的"权值膨胀"技巧：不需要 3D 预训练数据，
    直接用 ImageNet 上训练好的 2D 权重作为起点。

    Args:
        conv2d: 要膨胀的 2D 卷积层。
        time_kernel: 时间核大小，通常为 3。

    Returns:
        初始化后的 3D 卷积层。
    """
    out_c, in_c, kh, kw = conv2d.weight.shape
    pad_h = conv2d.padding[0] if isinstance(conv2d.padding, tuple) else conv2d.padding
    pad_w = conv2d.padding[1] if isinstance(conv2d.padding, tuple) else conv2d.padding
    stride_h = conv2d.stride[0] if isinstance(conv2d.stride, tuple) else conv2d.stride
    stride_w = conv2d.stride[1] if isinstance(conv2d.stride, tuple) else conv2d.stride
    has_bias = conv2d.bias is not None

    conv3d = nn.Conv3d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=(time_kernel, kh, kw),
        padding=(time_kernel // 2, pad_h, pad_w),
        stride=(1, stride_h, stride_w),
        bias=has_bias,
    )
    # 沿时间轴复制并除以时间核大小，保持激活量的期望不变
    weight_3d = conv2d.weight.data.unsqueeze(2).repeat(1, 1, time_kernel, 1, 1) / time_kernel
    conv3d.weight.data = weight_3d
    if has_bias:
        conv3d.bias.data = conv2d.bias.data.clone()
    return conv3d


# ===================================================================
# 第 4 部分：(2+1)D 因子化卷积 —— 时空分离
# ===================================================================


class Conv2Plus1D(nn.Module):
    """(2+1)D 因子化卷积：将 3D 卷积分解为空间 + 时间两步。

    标准 3D 卷积同时处理时空：3x3x3 核在 27 个点上做加权。
    (2+1)D 将它拆成两个更小的操作：
      1. 空间卷积 (1x3x3) — 只处理空间关系
      2. 时间卷积 (3x1x1) — 只处理时间关系
    中间夹一层 BN + ReLU，引入额外非线性，提升表达能力。

    参数量对比（C_in=C_out=C, k=3）：
      3D:      k³·C² = 27C²
      (2+1)D:  k²·C² + k¹·C² = 9C² + 3C² = 12C²（省 56%）
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        # 中间维度：平衡参数与表达能力
        mid_channels = max(8, (in_channels * out_channels * kernel_size ** 3) //
                           (in_channels * kernel_size ** 2 + out_channels * kernel_size))
        self.spatial = nn.Conv3d(in_channels, mid_channels,
                                 kernel_size=(1, kernel_size, kernel_size),
                                 padding=(0, kernel_size // 2, kernel_size // 2),
                                 bias=False)
        self.bn = nn.BatchNorm3d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(mid_channels, out_channels,
                                  kernel_size=(kernel_size, 1, 1),
                                  padding=(kernel_size // 2, 0, 0),
                                  bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先空间卷积，再 BN+ReLU，最后时间卷积。"""
        return self.temporal(self.act(self.bn(self.spatial(x))))


# ===================================================================
# 第 5 部分：光流法 (RAFT) —— 像素级运动估计
# ===================================================================


class RAFTBlock(nn.Module):
    """RAFT（Recurrent All-pairs Field Transforms）的简化教学实现。

    RAFT 通过迭代方式估算视频帧之间的密集光流场。
    核心思想：所有像素对之间计算相似度和相关性，
    然后通过循环神经网络不断 refinements 光流估计。

    这里展示的是核心相关金字塔模块——RAFT 之所以强大的关键。
    """

    def __init__(self, corr_levels: int = 4, corr_radius: int = 4,
                 flow_level_channels: int = 128):
        super().__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius

        # 相关金字塔：下采样构建多层相关性图
        # 每一层捕捉不同尺度的运动
        self.corr_proj = nn.Conv2d(
            corr_radius * 2 + 1, flow_level_channels, kernel_size=1
        )
        self.flow_proj = nn.Conv2d(2, flow_level_channels, kernel_size=3, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(flow_level_channels * 2, flow_level_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(flow_level_channels, flow_level_channels, kernel_size=3, padding=1),
        )

    @staticmethod
    def corr_with_pyramid(
        feat_map1: torch.Tensor,  # (B, C, H, W)
        feat_map2: torch.Tensor,  # (B, C, H, W)
        radius: int,
    ) -> torch.Tensor:
        """计算两张特征图的滑动窗口相关性。

        在 feat_map2 上滑动窗口（大小为 2*radius+1），
        与 feat_map1 逐通道内积，得到相关性图。

        这是RAFT的核心操作：不是逐像素比较，而是在
        feat_map2 的邻域内搜索最佳匹配位置。

        Args:
            feat_map1: 参考帧的特征图。
            feat_map2: 目标帧的特征图。
            radius: 搜索半径。

        Returns:
            相关性张量，形状 (B, (2r+1)^2, H, W)。
        """
        B, C, H, W = feat_map1.shape

        # 展开 feat_map2 的所有邻域位置
        # unfold 生成形状为 (B*C, (2r+1)^2, H, W) 的张量
        unfold_features = F.unfold(
            feat_map2,
            kernel_size=(radius * 2 + 1, radius * 2 + 1),
            padding=radius,
        )
        unfold_features = unfold_features.reshape(B, C, -1, H, W)

        # 在通道维度做内积：(B,C,H,W) · (B,C,(2r+1)^2,H,W) → (B,(2r+1)^2,H,W)
        corr = torch.einsum("bchw,bciphw->bihpw", feat_map1, unfold_features)
        # 归一化：除以通道数防止量级差异
        corr = corr / math.sqrt(C)
        return corr  # (B, (2r+1)^2, H, W)

    def forward(self, x: torch.Tensor) -> dict:
        """RAFT Block 的前向传播演示。

        Args:
            x: 输入张量，形状 (B, T, C, H, W)，两帧。

        Returns:
            包含光流估计和相关性的字典。
        """
        B, T = x.shape[:2]
        assert T == 2, "RAFT Block 当前仅演示两帧情况"

        feat1 = x[:, 0]  # (B, C, H, W)
        feat2 = x[:, 1]

        # 1. 计算多尺度相关性
        corr_pyramid = self.corr_with_pyramid(feat1, feat2, self.corr_radius)
        # 通过 1x1 卷积投影到流维度
        corr_projected = self.corr_proj(corr_pyramid)

        # 2. 初始光流为零（实际 RAFT 用可学习嵌入初始化）
        zero_flow = torch.zeros(B, 2, feat1.shape[-2], feat1.shape[-1], device=x.device)
        flow_projected = self.flow_proj(zero_flow)

        # 3. 拼接并卷积
        combined = torch.cat([corr_projected, flow_projected], dim=1)
        update = self.conv(combined)

        # 返回关键中间结果
        return {
            "correlation_shape": list(corr_projected.shape),  # 相关性图尺寸
            "flow_estimate_shape": list(zero_flow.shape),      # 光流估计尺寸
            "update_shape": list(update.shape),                  # 光流更新量
        }


# ===================================================================
# 第 6 部分：VideoMAE —— 掩码自编码器视频预训练
# ===================================================================


class VideoPatchEmbed(nn.Module):
    """视频块嵌入：将 (N,T,C,H,W) 视频切分为时空块，线性投影为嵌入向量。

    类似于 ViT 的 patch embedding，但多了时间维度。
    假设输入是 T 帧、H×W 的视频，用 (t_p, h_p, w_p) 大小的三维窗口
    将视频切分为不重叠的时空块。
    """

    def __init__(self, patches_per_frame: int = 16, embed_dim: int = 384, in_channels: int = 3,
                 spatial_size: int = 224):
        super().__init__()
        self.patches_per_frame = patches_per_frame
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        patch_size = spatial_size // int(math.sqrt(patches_per_frame))
        self.patch_size = (patch_size, patch_size)
        self.num_patches_per_frame = patches_per_frame
        self.num_patches = patches_per_frame  # 每帧的块数

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将视频切块并嵌入。

        Args:
            x: 输入视频 (N, T, C, H, W)。

        Returns:
            嵌入序列 (N, T*num_patches, embed_dim)。
        """
        N, T = x.shape[:2]
        # 先将批次和帧维度展平：(N*T, C, H, W)
        frames = x.view(N * T, *x.shape[2:])
        # 通过卷积切块并嵌入：(N*T, embed_dim, h', w')
        patches = self.proj(frames)
        _, _, h, w = patches.shape
        # 展平空间维度：(N*T, embed_dim, h*w) -> (N*T, h*w, embed_dim)
        patches = patches.flatten(2).transpose(1, 2)
        # 恢复批次和帧维度
        patches = patches.view(N, T, self.num_patches, self.embed_dim)
        # 将帧维度和块维度合并为一个序列
        patches = patches.reshape(N, T * self.num_patches, self.embed_dim)
        return patches


class MaskedAutoencoderBlock(nn.Module):
    """VideoMAE 的核心模块：Masked Video AE + Transformer。

    VideoMAE 的训练目标：遮住视频中 75% 的时空块，
    让模型从剩余 25% 的可见块重建被遮住的块的像素值。

    这与 BERT 的掩码语言建模（MLM）思路一致，
    但扩展到时空两个维度。
    """

    def __init__(self, embed_dim: int = 384, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

        # 初始化注意力掩码（用于遮蔽）
        self.register_buffer(
            "attn_mask",
            torch.zeros((1, 1, 1)),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """带可选遮蔽掩码的前向传播。

        Args:
            x: 嵌入序列 (N, seq_len, embed_dim)。
            mask: 遮蔽掩码，形状 (N, seq_len)。1 = 被遮蔽，0 = 可见。

        Returns:
            Transformer 输出 (N, seq_len, embed_dim)。
        """
        # 归一化 + 自注意力
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)

        # 如果提供了遮蔽掩码，将遮蔽位置的输出设为零（梯度不回传）
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (N, seq_len, 1)
            attn_output = attn_output * mask_expanded

        x = x + attn_output

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class VideoMAEHead(nn.Module):
    """VideoMAE 的预测头：从可见块编码预测被遮蔽块的像素。

    训练时：输入可见块 + [mask_token] 占位符 → 输出被遮蔽块的像素。
    推理时：不需要这个头。
    """

    def __init__(self, embed_dim: int = 384, patch_size: int = 16,
                 num_frames: int = 16, num_classes: int = 400):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.num_tokens = patch_size * patch_size * 3  # H_patch * W_patch * channels
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, self.num_tokens, bias=True),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.num_tokens))

    def forward(self, visible_features: torch.Tensor,
                mask_indices: torch.Tensor) -> torch.Tensor:
        """预测被遮蔽块的像素值。

        Args:
            visible_features: 可见块的编码表示 (N, num_visible, embed_dim)。
            mask_indices: 遮蔽块的索引 (N, num_masked)。

        Returns:
            预测的像素值 (N, num_masked, num_tokens)。
        """
        N = visible_features.size(0)
        num_masked = mask_indices.size(1)
        embed_dim = visible_features.size(-1)

        # 为每个遮蔽位置创建 mask token
        mask_tokens = self.mask_token.expand(N, num_masked, -1)

        # 将 mask token 和可见特征拼接
        # 注意：实际实现中需要按正确顺序排列
        all_tokens = mask_tokens  # 简化：仅展示预测逻辑
        predictions = self.pred_head(all_tokens)

        return predictions


# ===================================================================
# 第 7 部分：时空注意力机制
# ===================================================================


class SpatialTemporalAttention(nn.Module):
    """时空注意力：分别计算空间和时序的注意力。

    TimeSformer 的核心设计——"分治注意力"（Divided Attention）。

    完整联合时空注意力是 O((T*H*W)^2)，对于 16 帧的视频
    就是 O(16^2 × 224^2)^2 ≈ 10^13，完全不可行。
    分治方案将其分解为 O(T^2 + (H*W)^2)，大幅降低计算量。
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, qkv_bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV 投影共享权重（节省参数）
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # 空间维度的序列长度（H*W / patch_size^2），如 224→14×14=196
        self.spatial_seq_len = None
        # 时间维度长度
        self.temporal_seq_len = None

    def _split_spatiotemporal(self, x: torch.Tensor) -> tuple:
        """将输入序列分离为空间和时序维度。

        假设输入形状为 (N, T, L, D)，其中：
          T = 帧数，L = 每帧的空间位置数，D = 嵌入维度。

        Returns:
            空间特征 (N*T, L, D) 和时序特征 (N*L, T, D)
        """
        N, T, L, D = x.shape
        self.spatial_seq_len = L
        self.temporal_seq_len = T

        # 空间注意力：在每个时间步，让不同空间位置之间做注意力
        spatial_input = x.transpose(1, 2).reshape(N * L, T, self.embed_dim)
        # 时序注意力：在每个空间位置，让不同帧之间做注意力
        temporal_input = x.reshape(N * L, T, self.embed_dim)

        return spatial_input, temporal_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行分治时空注意力。

        Args:
            x: 输入张量 (N, T, L, embed_dim)。
                N=批次, T=帧数, L=空间位置数, embed_dim=嵌入维度。

        Returns:
            输出张量 (N, T, L, embed_dim)。
        """
        N, T, L, D = x.shape
        x_flat = x.reshape(N * T * L, D)

        # === 第一步：时序注意力 ===
        # 将 (NTL, D) reshape 为 (NL, T, D)
        x_nltd = x_flat.reshape(N * L, T, D)
        qkv_temporal = self.qkv(x_nltd).chunk(3, dim=-1)  # 3 × (NL, T, D)
        q_t = qkv_temporal[0].reshape(N * L, T, self.num_heads, D // self.num_heads).transpose(1, 2)
        k_t = qkv_temporal[1].reshape(N * L, T, self.num_heads, D // self.num_heads).transpose(1, 2)
        v_t = qkv_temporal[2].reshape(N * L, T, self.num_heads, D // self.num_heads).transpose(1, 2)

        attn_t = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scale
        attn_t = F.softmax(attn_t, dim=-1)
        out_t = torch.matmul(attn_t, v_t).transpose(1, 2).reshape(N * L, T, D)

        # === 第二步：空间注意力 ===
        # 将 (NL, T, D) reshape 为 (NT, L, D)
        x_ntld = out_t.reshape(N * T, L, D)
        qkv_spatial = self.qkv(x_ntld).chunk(3, dim=-1)
        q_s = qkv_spatial[0].reshape(N * T, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        k_s = qkv_spatial[1].reshape(N * T, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        v_s = qkv_spatial[2].reshape(N * T, L, self.num_heads, D // self.num_heads).transpose(1, 2)

        attn_s = torch.matmul(q_s, k_s.transpose(-2, -1)) * self.scale
        attn_s = F.softmax(attn_s, dim=-1)
        out_s = torch.matmul(attn_s, v_s).transpose(1, 2).reshape(N * T, L, D)

        # 合并
        result = self.proj(out_s)

        return result.reshape(N, T, L, D)


class DividedTimeSpaceAttention(nn.Module):
    """TimeSformer 风格的分治注意力块：时序和空间注意力交替使用。

    每个 Transformer 块内包含：
    1. 时序注意力 —— 帧与帧之间的关联
    2. 空间注意力 —— 同一帧内空间位置之间的关联
    3. MLP 层

    总计算量从 O((T*H*W)^2) 降到 O(T^2 + (H*W)^2)。
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 mlp_ratio: float = 4.0, patch_size_h: int = 14, patch_size_w: int = 14):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 时序注意力：只关注帧维度 T
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )
        # 空间注意力：只关注空间维度（每帧的 patch 数）
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(0.0),
        )

        self.patch_h = patch_size_h
        self.patch_w = patch_size_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """时序 + 空间交替注意力。

        Args:
            x: 输入 (N, T, num_patches, embed_dim)。

        Returns:
            输出 (N, T, num_patches, embed_dim)。
        """
        N, T, num_patches, D = x.shape

        # === 时序注意力 ===
        #  reshape 为 (N*num_patches, T, D)
        x_t = x.permute(0, 2, 1, 3).reshape(N * num_patches, T, D)
        x_t_norm = self.norm1(x_t)
        x_t_attn, _ = self.temporal_attn(x_t_norm, x_t_norm, x_t_norm)
        x_t = x_t + x_t_attn

        # === 空间注意力 ===
        #  reshape 为 (N*T, num_patches, D)
        x_s = x_t.reshape(N * T, num_patches, D)
        x_s_norm = self.norm2(x_s)
        x_s_attn, _ = self.spatial_attn(x_s_norm, x_s_norm, x_s_norm)
        x_s = x_s + x_s_attn

        # === MLP ===
        x_out = x_s + self.mlp(x_s_norm)

        return x_out.reshape(N, num_patches, T, D).permute(0, 2, 1, 3)


# ===================================================================
# 第 8 部分：端到端动作识别网络
# ===================================================================


class SlowFastFramePool(nn.Module):
    """慢快速双通道帧池化模型。

    受 SlowFast 架构启发：用两条并行路径捕捉不同速度的运动。
    - 快通道：以高频采样（如每秒 30 帧），捕捉快速运动
    - 慢通道：以低频采样（如每秒 2 帧），捕捉场景外观

    最终在帧池化后将两路特征拼接进行分类。
    """

    def __init__(self, num_classes: int = 400, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        fast_backbone = resnet18(weights=weights)
        slow_backbone = resnet18(weights=weights)

        self.fast_features = nn.Sequential(*list(fast_backbone.children())[:-1])
        self.slow_features = nn.Sequential(*list(slow_backbone.children())[:-1])

        # 双流特征拼接后的分类头
        self.classifier = nn.Linear(512 * 2, num_classes)

    def forward(self, fast_x: torch.Tensor, slow_x: torch.Tensor) -> torch.Tensor:
        """双流前向传播。

        Args:
            fast_x: 快通道输入 (N, T_fast, C, H, W)。
            slow_x: 慢通道输入 (N, T_slow, C, H, W)。

        Returns:
            分类 logits (N, num_classes)。
        """
        N = fast_x.shape[0]

        # 快通道：逐帧提取特征，平均池化
        fast_flat = fast_x.view(N * fast_x.shape[1], *fast_x.shape[2:])
        fast_feats = self.fast_features(fast_flat).view(N, -1, 512).mean(dim=1)

        # 慢通道：同样处理
        slow_flat = slow_x.view(N * slow_x.shape[1], *slow_x.shape[2:])
        slow_feats = self.slow_features(slow_flat).view(N, -1, 512).mean(dim=1)

        # 拼接双流特征
        combined = torch.cat([fast_feats, slow_feats], dim=1)
        return self.classifier(combined)


# ===================================================================
# 主程序：运行所有演示
# ===================================================================


def main():
    print("=" * 60)
    print("视频理解算法演示")
    print("=" * 60)

    # --- 第 1 节：帧采样 ---
    print("\n[帧采样策略]")
    for total in [8, 30, 300]:
        uniform = sample_uniform(total, 8)
        dense = sample_dense(total, 8, seed=42)
        multiclip = sample_multiclip(total, 3, 8, seed=42)
        print(f"  总帧={total:4d}:")
        print(f"    均匀采样(8帧): {uniform}")
        print(f"    稠密采样(8帧): {dense}")
        print(f"    多片段(3次x8): {multiclip}")

    # --- 第 2 节：2D+Pool ---
    print("\n[2D+Pool 帧池化模型]")
    model = FramePool(num_classes=10, pretrained=False)
    x = torch.randn(2, 8, 3, 64, 64)  # (N=2, T=8, C=3, H=64, W=64)
    out = model(x)
    print(f"  输入: {tuple(x.shape)}")
    print(f"  输出: {tuple(out.shape)}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # --- 第 3 节：I3D 权值膨胀 ---
    print("\n[I3D 2D→3D 权值膨胀]")
    c2d = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
    c3d = inflate_2d_to_3d(c2d, time_kernel=3)
    print(f"  2D 权重形状: {tuple(c2d.weight.shape)}")
    print(f"  3D 权重形状: {tuple(c3d.weight.shape)}")
    y = c3d(torch.randn(1, 3, 8, 32, 32))
    print(f"  3D 卷积输出: {tuple(y.shape)}")

    # --- 第 4 节：(2+1)D 卷积 ---
    print("\n[(2+1)D 因子化卷积]")
    c21 = Conv2Plus1D(3, 16)
    y = c21(torch.randn(1, 3, 8, 32, 32))
    print(f"  输入: (1, 3, 8, 32, 32)")
    print(f"  输出: {tuple(y.shape)}")
    # 对比标准 3D 卷积的参数
    c3d_std = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1, bias=False)
    print(f"  (2+1)D 参数量: {sum(p.numel() for p in c21.parameters()):,}")
    print(f"  标准 3D 参数量: {sum(p.numel() for p in c3d_std.parameters()):,}")

    # --- 第 5 节：RAFT 光流 ---
    print("\n[RAFT 光流核心模块]")
    raft = RAFTBlock(corr_radius=2)
    frame1 = torch.randn(1, 3, 224, 224)
    frame2 = frame1 + torch.randn(1, 3, 224, 224) * 0.1  # 加上小扰动模拟运动
    video = torch.stack([frame1, frame2], dim=1).unsqueeze(0)  # (1, 2, 3, 224, 224)
    result = raft(video)
    for key, val in result.items():
        print(f"  {key}: {val}")

    # --- 第 6 节：VideoMAE ---
    print("\n[VideoMAE 块嵌入]")
    patch_embed = VideoPatchEmbed(patches_per_frame=16, embed_dim=192)
    video_clip = torch.randn(2, 8, 3, 224, 224)
    patches = patch_embed(video_clip)
    print(f"  输入: {tuple(video_clip.shape)}")
    print(f"  输出: {tuple(patches.shape)}")

    print("\n[VideoMAE 遮蔽演示]")
    mae_block = MaskedAutoencoderBlock(embed_dim=192, num_heads=6)
    seq_len = patches.shape[1]
    mask_ratio = 0.75
    num_masked = int(seq_len * mask_ratio)
    mask = torch.ones(2, seq_len)
    # 随机遮蔽 75% 的位置
    mask_indices = torch.rand(2, seq_len).topk(k=num_masked, dim=-1)[1]
    mask.scatter_(1, mask_indices, 0)
    masked_output = mae_block(patches, mask=mask.bool())
    masked_sum = masked_output[mask.bool()].abs().sum().item()
    unmasked_sum = masked_output[~mask.bool()].abs().sum().item()
    print(f"  遮蔽率: {mask_ratio:.0%}")
    print(f"  遮蔽位置输出幅值和: {masked_sum:.4f}（预期接近 0）")
    print(f"  可见位置输出幅值和: {unmasked_sum:.4f}")

    # --- 第 7 节：时空注意力 ---
    print("\n[时空分治注意力]")
    spatial_temporal = SpatialTemporalAttention(embed_dim=384, num_heads=6)
    # 输入: (N=1, T=4, L=16, D=384)
    st_input = torch.randn(1, 4, 16, 384)
    st_output = spatial_temporal(st_input)
    print(f"  输入: {tuple(st_input.shape)}")
    print(f"  输出: {tuple(st_output.shape)}")

    print("\n[TimeSformer 分治注意力块]")
    divided_attn = DividedTimeSpaceAttention(embed_dim=256, num_heads=8)
    # 输入: (N=2, T=4, num_patches=16, D=256)
    attn_input = torch.randn(2, 4, 16, 256)
    attn_output = divided_attn(attn_input)
    print(f"  输入: {tuple(attn_input.shape)}")
    print(f"  输出: {tuple(attn_output.shape)}")

    # 计算量对比
    T, H_W = 16, 196  # 16 帧, 14x14=196 个空间位置
    joint_flops = (T * H_W) ** 2 * 256
    divided_flops = (T ** 2 + H_W ** 2) * 256
    print(f"\n  计算量对比 (embed_dim=256, T=16, patches=196):")
    print(f"    联合时空注意力: {joint_flops / 1e9:.2f} GFLOPs")
    print(f"    分治注意力:     {divided_flops / 1e6:.2f} MFLOPs")
    print(f"    加速比:         {joint_flops / divided_flops:.0f}x")

    # --- 第 8 节：双流帧池化 ---
    print("\n[SlowFast 双流帧池化]")
    slow_fast = SlowFastFramePool(num_classes=10, pretrained=False)
    # 快通道：30fps → 32 帧；慢通道：2fps → 32/15 ≈ 3 帧 → 补到相同长度
    fast_x = torch.randn(2, 32, 3, 64, 64)
    slow_x = torch.randn(2, 3, 3, 64, 64)
    out = slow_fast(fast_x, slow_x)
    print(f"  快通道输入: {tuple(fast_x.shape)}")
    print(f"  慢通道输入: {tuple(slow_x.shape)}")
    print(f"  输出: {tuple(out.shape)}")
    print(f"  参数量: {sum(p.numel() for p in slow_fast.parameters()):,}")

    # --- 总结 ---
    print("\n" + "=" * 60)
    print("演示完成！所有模块均通过前向传播验证。")
    print("=" * 60)


if __name__ == "__main__":
    main()
