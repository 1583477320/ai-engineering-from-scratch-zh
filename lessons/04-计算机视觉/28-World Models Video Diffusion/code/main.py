# main.py — 微型视频扩散 Transformer 结构演示
# 对应课程：阶段 04 · 28（World Models 视频扩散）
# 依赖：torch>=2.0
# 安装：pip install torch

"""
本文件演示视频扩散模型的核心组件：

1.  VideoPatch3D — 时空 3D 分块（patchify）
2.  DividedAttentionBlock — 分解注意力（时间注意力 + 空间注意力）
3.  TinyVideoDiT — 微型视频 Diffusion Transformer
4.  count_tokens — 计算给定视频分辨率下的词元数

这些组件是 Sora、Wan-Video、HunyuanVideo 等现代视频扩散模型的基础。
"""

import torch
import torch.nn as nn
import math


# ============================================================
# 第 1 步：时空 3D 分块（Spatio-Temporal 3D Patchify）
# ============================================================

class VideoPatch3D(nn.Module):
    """将视频切分成时空 patch，每个 patch 映射为嵌入向量。

    输入形状: (N, C, T, H, W)
    输出形状: (N, num_patches, dim)

    其中 num_patches = (T/p_t) * (H/p_h) * (W/p_w)
    p_t, p_h, p_w 分别是时间、高、宽方向的 patch 大小。

    这是 Sora 论文中提出的"spacetime latent patches"思想的实现。
    """

    def __init__(self, in_channels=4, dim=64, patch_t=2, patch_h=2, patch_w=2):
        """
        Args:
            in_channels: 输入视频的通道数（潜空间中通常为 4）
            dim: 每个 patch 映射到的嵌入维度
            patch_t: 时间维度上每个 patch 包含的帧数
            patch_h: 高度方向上每个 patch 包含的像素数
            patch_w: 宽度方向上每个 patch 包含的像素数
        """
        super().__init__()
        # 使用 3D 卷积实现分块 —— 卷积核大小等于步长，不重叠
        self.proj = nn.Conv3d(
            in_channels, dim,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w),
        )
        self.patch_t = patch_t
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, x):
        # x: (N, C, T, H, W)
        x = self.proj(x)                     # (N, dim, T/p_t, H/p_h, W/p_w)
        n, c, t, h, w = x.shape
        # 将空间和时间维度展平为序列维度
        tokens = x.reshape(n, c, t * h * w).transpose(1, 2)  # (N, T*H*W/patch^3, dim)
        return tokens, (t, h, w)             # 返回词元序列和网格尺寸（用于位置编码）


# ============================================================
# 第 2 步：3D 旋转位置编码（简化的加法形式）
# ============================================================

def rope_3d(tokens, grid, t_dim=16, h_dim=24, w_dim=24):
    """为时空词元添加 3D 位置编码（简化的加法版本）。

    真正的 RoPE 会成对旋转通道，这里使用简化的加法形式来演示位置信息注入。
    实际生产模型（如 Sora、Wan-Video）使用标准的旋转位置嵌入。

    Args:
        tokens: 词元序列 (N, T*H*W, D)
        grid: 网格尺寸 (T_tok, H_tok, W_tok)
        t_dim: 分配给时间维度的编码维度
        h_dim: 分配给高度维度的编码维度
        w_dim: 分配给宽度维度的编码维度

    Returns:
        添加了位置编码的词元序列
    """
    T_tok, H_tok, W_tok = grid
    n, seq, d = tokens.shape
    d_total = t_dim + h_dim + w_dim

    # 检查维度分配是否正确
    assert d_total <= d, f"t_dim({t_dim})+h_dim({h_dim})+w_dim({w_dim}) 不能超过 {d}"
    assert seq == T_tok * H_tok * W_tok, f"序列长度 {seq} 不等于 {T_tok}*{H_tok}*{W_tok}"

    # 构建每个词元的位置索引
    # 时间索引: 每个时间片重复 H_tok * W_tok 次
    t_idx = torch.arange(T_tok, device=tokens.device).repeat_interleave(H_tok * W_tok)
    # 高度索引: 每个高度值重复 W_tok 次，再整体重复 T_tok 次
    h_idx = torch.arange(H_tok, device=tokens.device).repeat_interleave(W_tok).repeat(T_tok)
    # 宽度索引: 每个宽度值连续排列，再整体重复 T_tok * H_tok 次
    w_idx = torch.arange(W_tok, device=tokens.device).repeat(T_tok * H_tok)

    # 计算不同轴上的频率（与 Transformer 位置编码相同）
    freqs_t = torch.exp(
        -math.log(10000.0) * torch.arange(t_dim // 2, device=tokens.device) / (t_dim // 2)
    )
    freqs_h = torch.exp(
        -math.log(10000.0) * torch.arange(h_dim // 2, device=tokens.device) / (h_dim // 2)
    )
    freqs_w = torch.exp(
        -math.log(10000.0) * torch.arange(w_dim // 2, device=tokens.device) / (w_dim // 2)
    )

    # 拼接正弦和余弦编码
    emb_t = torch.cat([
        torch.sin(t_idx[:, None] * freqs_t),
        torch.cos(t_idx[:, None] * freqs_t)
    ], dim=-1)
    emb_h = torch.cat([
        torch.sin(h_idx[:, None] * freqs_h),
        torch.cos(h_idx[:, None] * freqs_h)
    ], dim=-1)
    emb_w = torch.cat([
        torch.sin(w_idx[:, None] * freqs_w),
        torch.cos(w_idx[:, None] * freqs_w)
    ], dim=-1)

    # 将三个轴的编码拼接后加到词元上
    pos_embed = torch.cat([emb_t, emb_h, emb_w], dim=-1)  # (seq, d_total)

    # 如果模型的隐藏维度大于 d_total，用零填充剩余维度
    if d_total < d:
        padding = torch.zeros(seq, d - d_total, device=tokens.device)
        pos_embed = torch.cat([pos_embed, padding], dim=-1)

    return tokens + pos_embed.unsqueeze(0)  # (N, seq, d)


# ============================================================
# 第 3 步：分解注意力块（Divided Attention）
# ============================================================

class DividedAttentionBlock(nn.Module):
    """分解注意力 Transformer 块。

    每次注意力计算分两步：
    1. 时间注意力：同一空间位置跨时间帧做注意力
    2. 空间注意力：同一时间帧跨空间位置做注意力

    这比全连接注意力的计算量小得多：
        - 全连接: O((T*H*W)^2)
        - 分解后: O(H*W * T^2) + O(T * (H*W)^2)

    TimeSformer (Bertasius et al., 2021) 首次提出了这个模式，
    现在被 Sora、Wan-Video、HunyuanVideo 等所有现代视频 DiT 使用。
    """

    def __init__(self, dim=64, heads=2):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.space_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x, grid):
        """
        Args:
            x: 词元序列 (N, T*H*W, D)
            grid: (T_tok, H_tok, W_tok)

        Returns:
            处理后的词元序列 (N, T*H*W, D)
        """
        T_tok, H_tok, W_tok = grid
        n, seq, d = x.shape

        # === 时间注意力 ===
        # 将词元重新排列为 (N*H*W, T, D) — 每个空间位置独立做时间注意力
        # 原始形状: (N, T, H*W, D) -> 转置 -> (N, H*W, T, D) -> 合并前两维
        xt = x.view(n, T_tok, H_tok * W_tok, d)
        xt = xt.permute(0, 2, 1, 3).reshape(n * H_tok * W_tok, T_tok, d)

        # 时间维度上的自注意力
        a, _ = self.time_attn(
            self.ln1(xt), self.ln1(xt), self.ln1(xt),
            need_weights=False
        )
        # 残差连接后恢复原始形状
        xt = (xt + a).reshape(n, H_tok * W_tok, T_tok, d)
        xt = xt.permute(0, 2, 1, 3).reshape(n, seq, d)

        # === 空间注意力 ===
        # 将词元重新排列为 (N*T, H*W, D) — 每帧独立做空间注意力
        xs = xt.view(n, T_tok, H_tok * W_tok, d)
        xs = xs.reshape(n * T_tok, H_tok * W_tok, d)

        # 空间维度上的自注意力
        a, _ = self.space_attn(
            self.ln2(xs), self.ln2(xs), self.ln2(xs),
            need_weights=False
        )
        # 残差连接后恢复
        xs = (xs + a).reshape(n, T_tok, H_tok * W_tok, d).reshape(n, seq, d)

        # === 前馈网络（MLP）===
        xs = xs + self.mlp(self.ln3(xs))
        return xs


# ============================================================
# 第 4 步：组合完整的微型视频 DiT
# ============================================================

class TinyVideoDiT(nn.Module):
    """微型视频 Diffusion Transformer。

    这是一个教学演示模型，不是可用的视频生成器。
    它展示了视频 DiT 的完整结构：3D 分块 → 位置编码 → 分解注意力块 → 输出投影。

    真实视频 DiT（如 Sora、Wan-Video）在此基础上增加：
    - AdaLN 条件注入（时间步、文本嵌入）
    - Rectified Flow 训练目标（而非 DDPM）
    - 大规模并行训练
    """

    def __init__(self, in_channels=4, dim=64, depth=2, heads=2):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim

        # 时空分块
        self.patch = VideoPatch3D(
            in_channels=in_channels, dim=dim,
            patch_t=2, patch_h=2, patch_w=2
        )
        # 分解注意力层堆叠
        self.blocks = nn.ModuleList([
            DividedAttentionBlock(dim, heads) for _ in range(depth)
        ])
        # 输出投影：将每个词元投影回 patch 大小的像素空间
        # 输出维度: in_channels * patch_t * patch_h * patch_w
        self.out = nn.Linear(dim, in_channels * 2 * 2 * 2)

    def forward(self, x):
        """
        Args:
            x: 输入视频 (N, C, T, H, W)

        Returns:
            out: 每个词元对应的像素预测 (N, seq, C*8)
            grid: 词元网格尺寸 (T_tok, H_tok, W_tok)
        """
        # 1) 时空分块
        tokens, grid = self.patch(x)

        # 2) 通过所有分解注意力块
        for blk in self.blocks:
            tokens = blk(tokens, grid)

        # 3) 输出投影
        out = self.out(tokens)
        return out, grid


# ============================================================
# 第 5 步：词元计数工具
# ============================================================

def count_tokens(T, H, W, p_t=2, p_h=8, p_w=8):
    """计算给定分辨率视频经过时空分块后的总词元数。

    Args:
        T: 总帧数
        H: 视频高度（像素）
        W: 视频宽度（像素）
        p_t: 时间 patch 大小
        p_h: 高度 patch 大小
        p_w: 宽度 patch 大小

    Returns:
        总词元数
    """
    return (T // p_t) * (H // p_h) * (W // p_w)


# ============================================================
# 主程序：形状验证和计算量分析
# ============================================================

def main():
    print("=" * 60)
    print("视频 DiT 结构演示")
    print("=" * 60)

    # === 演示 1：词元计数和计算量分析 ===
    print("\n[演示 1] 5 秒 360p 视频的词元数和注意力计算量")
    print("-" * 50)
    # 5秒@30fps = 150 帧，360p分辨率 = 480x360
    fps = 30
    duration = 5
    T_total = fps * duration  # 150 帧
    H, W = 480, 360

    # 分块参数（与 Sora 论文一致：时间分块 2，空间分块 8x8）
    p_t, p_h, p_w = 2, 8, 8

    tokens = count_tokens(T_total, H, W, p_t, p_h, p_w)
    T_tok = T_total // p_t       # 时间方向词元数: 75
    H_tok = H // p_h             # 高度方向词元数: 60
    W_tok = W // p_w             # 宽度方向词元数: 45
    S_tok = H_tok * W_tok        # 每帧的空间词元数: 2700

    print(f"  输入视频: {duration}s @ {fps}fps, 分辨率 {H}x{W}")
    print(f"  分块参数: p_t={p_t}, p_h={p_h}, p_w={p_w}")
    print(f"  总词元数: {tokens:,}")
    print(f"    其中 T_tok={T_tok}, H_tok={H_tok}, W_tok={W_tok}")
    print(f"    每帧空间词元: {S_tok:,}")
    print()

    # 对比全连接注意力和分解注意力的计算量
    joint_pairs = tokens ** 2
    divided_time = S_tok * T_tok ** 2
    divided_space = T_tok * S_tok ** 2
    divided_total = divided_time + divided_space
    speedup = joint_pairs / divided_total

    print(f"  注意力计算量对比：")
    print(f"    全连接注意力对: {joint_pairs:,}")
    print(f"    分解-时间注意力: {divided_time:,}")
    print(f"    分解-空间注意力: {divided_space:,}")
    print(f"    分解注意力总计: {divided_total:,}")
    print(f"    🚀 速度提升: {speedup:.1f}x")
    print()

    # === 演示 2：模型形状验证 ===
    print("\n[演示 2] TinyVideoDiT 形状验证")
    print("-" * 50)

    torch.manual_seed(42)
    # 模拟一个 4 通道的潜空间视频（类似 VAE 编码后的表示）
    # 形状: (N=1, C=4, T=8, H=16, W=16)
    vid = torch.randn(1, 4, 8, 16, 16)

    model = TinyVideoDiT(in_channels=4, dim=64, depth=2, heads=2)
    out, grid = model(vid)

    T_tok, H_tok, W_tok = grid
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  输入形状:           {tuple(vid.shape)}")
    print(f"  词元网格 (T, H, W): ({T_tok}, {H_tok}, {W_tok})")
    print(f"  总词元数:            {T_tok * H_tok * W_tok}")
    print(f"  输出形状:            {tuple(out.shape)}")
    print(f"  模型参数量:          {n_params:,}")
    print()

    # 验证输出形状与预期一致
    expected_seq = (8 // 2) * (16 // 2) * (16 // 2)  # 4 * 8 * 8 = 256
    expected_out = (1, expected_seq, 4 * 2 * 2 * 2)   # (1, 256, 32)

    assert out.shape == expected_out, \
        f"输出形状不匹配: 期望 {expected_out}, 得到 {out.shape}"
    assert grid == (4, 8, 8), \
        f"网格尺寸不匹配: 期望 (4, 8, 8), 得到 {grid}"

    print("  ✓ 形状验证通过!")
    print()

    # === 演示 3：参数量对比 ===
    print("\n[演示 3] 不同参数设置下的模型容量")
    print("-" * 50)

    configs = [
        ("微型 (Tiny)", 64, 2, 2),
        ("小型 (Small)", 128, 4, 4),
        ("中型 (Medium)", 256, 6, 8),
    ]

    print(f"  {'配置':<16} {'隐藏维度':<10} {'层数':<8} {'注意力头':<10} {'参数量':<12}")
    print(f"  {'-'*56}")
    for name, dim, depth, heads in configs:
        m = TinyVideoDiT(in_channels=4, dim=dim, depth=depth, heads=heads)
        params = sum(p.numel() for p in m.parameters())
        print(f"  {name:<16} {dim:<10} {depth:<8} {heads:<10} {params:<12,}")

    print()
    print("笔记: 真实视频 DiT（如 Wan-Video 14B）有 140 亿参数,")
    print("      我们的微型模型仅用于理解结构原理。")
    print()

    # === 总结 ===
    print("=" * 60)
    print("关键结论")
    print("=" * 60)
    print("1. 视频 DiT 将视频视为时空词元序列")
    print("2. 3D 分块将 (C, T, H, W) 转换为 (T*H*W/p^3, dim)")
    print("3. 分解注意力将 O(N^2) 降为 O(HW*T^2 + T*(HW)^2)")
    print("4. 现代视频扩散模型（Sora, Wan-Video, HunyuanVideo）")
    print("   都基于此结构，增加了条件注入和大规模训练")
    print("=" * 60)


if __name__ == "__main__":
    main()
