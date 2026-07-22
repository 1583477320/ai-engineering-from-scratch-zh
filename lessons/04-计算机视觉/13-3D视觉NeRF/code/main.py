# main.py — NeRF 从零实现：SIREN 激活函数、位置编码、体渲染与 Instant-NGP 哈希网格
# 依赖：torch>=2.0, numpy
# 安装：pip install torch numpy
# 对应课程：阶段 04 · 13（3D 视觉 NeRF）

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 第 1 步：位置编码（Positional Encoding）
# ============================================================
# 原始 MLP 对低频信号敏感，难以表示高频细节（如纹理、锐利边缘）。
# 位置编码将每个坐标映射为多频正弦/余弦组合，让 MLP "看到" 高频结构。
# 与 Transformer 中的位置编码同构——都是 sin/cos 傅里叶特征拼接。


def positional_encoding(x: torch.Tensor, num_frequencies: int = 10) -> torch.Tensor:
    """
    对输入坐标执行傅里叶位置编码。
    gamma(p) = (sin(2^0 * pi * p), cos(2^0 * pi * p), sin(2^1 * pi * p), cos(2^1 * pi * p), ...)
    最高频率为 2^(L-1)，其中 L 为频率层数。

    Args:
        x: 输入坐标张量，形状 (..., D)，通常为 3D 空间坐标 (x, y, z)
        num_frequencies: 频率层数（每层含 sin + cos），默认 10

    Returns:
        编码后的特征向量，形状 (..., D * 2 * num_frequencies)
    """
    # 将坐标扩展维度以便广播：(N, 3) → (N, 3, 1)
    # 频率向量形状: (num_frequencies,)
    # 乘积广播后形状: (N, 3, num_frequencies)
    scaled = x.unsqueeze(-1) * torch.arange(num_frequencies, dtype=x.dtype, device=x.device).float() * math.pi

    # 对每个频率分别计算 sin 和 cos，沿维度 -2（频率维度前）拼接
    # 形状: (N, 3, num_frequencies) → (N, 3, 2 * num_frequencies)
    # 最终 reshape 为 (N, 3 * 2 * num_frequencies) = (N, 60)
    encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-2)

    return encoded.reshape(*x.shape[:-1], -1)


# 验证：将 (N, 3) 的 3D 坐标编码为 (N, 60)
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=== 第 1 步：位置编码 ===")

    coords = torch.randn(5, 3)       # 5 个 3D 点
    encoded = positional_encoding(coords, num_frequencies=10)
    print(f"输入形状: {tuple(coords.shape)} -> 编码后形状: {tuple(encoded.shape)}")
    print(f"（3 维坐标 × 10 层频率 × 2 (sin+cos) = 60 维特征）\n")

    # 展示第一个点的部分编码值
    print(f"第一个点的前 6 个编码值: {encoded[0, :6].tolist()}")
    print(f"（前 3 个是低频 sin，后 3 个是低频 cos）\n")


# ============================================================
# 第 2 步：SIREN 激活函数
# ============================================================
# 标准 ReLU 在 NeRF 上表现极差——它会将低频分量压缩为零，导致场景完全模糊。
# SIREN（Sinusoidal Representation Network，Cunliffe et al., 2020）使用 sin() 作为
# 内部激活函数：sin(f(x)) 的导数仍然是 sin/cos 形式，保持梯度流通，
# 天然适合拟合高频信号（如图像、3D 场景的辐射场）。
# 权重初始化也特殊：第一层使用均匀分布 U(-1/fan_in, 1/fan_in)，
# 后续层使用 U(-sqrt(6/fan_in/scale), sqrt(6/fan_in/scale))，其中 scale 是频率倍数。


class SineLayer(nn.Module):
    """
    SIREN 的正弦激活层。
    f(x) = sin(w * x + b)，其中 w 按照 SIREN 论文的特殊方式初始化。
    """

    def __init__(self, in_features: int, out_features: int, is_first: bool = False, omega: float = 30.0):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        self.is_first = is_first

        # SIREN 专用初始化：第一层不使用 omega 缩放
        if is_first:
            with torch.no_grad():
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
        else:
            with torch.no_grad():
                self.linear.weight.uniform_(
                    -math.sqrt(6 / in_features) / omega,
                    math.sqrt(6 / in_features) / omega,
                )

        self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(x))


class TinySirenNeRF(nn.Module):
    """
    使用 SIREN 激活函数的微型 NeRF 模型。

    网络结构：
      输入: 位置编码后的坐标 (3 * 2 * L_pos,) + 方向编码 (3 * 2 * L_dir,)
      Trunk: [SIREN × 4] → σ (标量，不透明度)
      Branch: [SIREN → Linear → Sigmoid] → RGB (3 通道颜色)

    与原始 NeRF 的区别：
    - 原始 NeRF 有 2 个深度为 8 的 MLP 分支（密度/法线共享 trunk，颜色单独）
    - 本模型为教学简化版，使用 SIREN 替代 ReLU
    """

    def __init__(
        self,
        num_pos_freqs: int = 10,
        num_dir_freqs: int = 4,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        omega: float = 30.0,
    ):
        super().__init__()
        self.num_pos_freqs = num_pos_freqs
        self.num_dir_freqs = num_dir_freqs

        pos_encoded_dim = 3 * 2 * num_pos_freqs   # 位置编码维度
        dir_encoded_dim = 3 * 2 * num_dir_freqs   # 方向编码维度

        # ---- 构建 Trunk: 输入 → 隐藏层 → 输出 σ ----
        layers = []
        for i in range(num_hidden_layers):
            is_first = (i == 0)
            in_dim = pos_encoded_dim if i == 0 else hidden_dim
            layers.append(SineLayer(in_dim, hidden_dim, is_first=is_first, omega=omega))

        self.trunk = nn.Sequential(*layers)
        self.sigma_layer = nn.Linear(hidden_dim, 1)

        # ---- 构建颜色分支: Trunk 输出 + 方向编码 → RGB ----
        self.color_net = nn.Sequential(
            SineLayer(hidden_dim + dir_encoded_dim, hidden_dim // 2, is_first=True, omega=omega),
            nn.ReLU(),                                  # 第二层回退到 ReLU：方向信息已足够丰富
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),                               # 颜色范围约束在 [0, 1]
        )

    def forward(self, positions: torch.Tensor, directions: torch.Tensor):
        """
        前向传播。

        Args:
            positions: 3D 坐标，形状 (N, 3)
            directions: 观察方向，形状 (N, 3)

        Returns:
            sigma: 不透明度（体积密度），形状 (N,)
            rgb:   颜色值，形状 (N, 3)
        """
        N = positions.shape[0]

        # 位置编码 + 方向编码
        pos_enc = positional_encoding(positions, self.num_pos_freqs)
        dir_enc = positional_encoding(directions, self.num_dir_freqs)

        # Trunk: 坐标 → 隐藏特征 → σ
        trunk_features = self.trunk(pos_enc)             # (N, hidden_dim)
        sigma = torch.relu(self.sigma_layer(trunk_features)).squeeze(-1)  # (N,)，确保非负

        # 颜色分支: 融合方向信息 → RGB
        color_input = torch.cat([trunk_features, dir_enc], dim=-1)  # (N, hidden + dir_enc)
        rgb = self.color_net(color_input)                 # (N, 3)

        return sigma, rgb


# ============================================================
# 第 3 步：体渲染（Volumetric Rendering）
# ============================================================
# NeRF 的核心数学。给定一条光线上的 N 个采样点及其 (σ, c)，
# 体渲染将这些点合成一个像素的颜色值。核心公式：
#
#   C(r) = Σ_i T_i · (1 - exp(-σ_i · δ_i)) · c_i
#
#   T_i = exp(-Σ_{j<i} σ_j · δ_j)    → 透射率：光线到达点 i 时剩余的光强
#   δ_i = t_{i+1} - t_i              → 相邻采样点间距
#   (1 - exp(-σ_i · δ_i))            → 采样点 i 处的不透明度 α
#
# 这与计算机图形学中的经典体积渲染方程完全一致，只是 σ 和 c 由神经网络预测。


def volumetric_render(
    sigma: torch.Tensor,
    rgb: torch.Tensor,
    ray_distances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    沿单条光线进行体渲染，将 N 个采样点合成为单个像素。

    Args:
        sigma: 体积密度，形状 (N,)，非负
        rgb:   颜色值，形状 (N, 3)，范围 [0, 1]
        ray_distances: 沿光线的采样距离，形状 (N,)，递增排列

    Returns:
        rendered_color: 渲染后的像素颜色，形状 (3,)
        depth: 加权平均深度，形状 ()
    """
    # 计算相邻采样点之间的距离 δ
    delta = torch.cat([ray_distances[1:] - ray_distances[:-1], torch.full_like(ray_distances[:1], 1e10)])

    # 计算不透明度 α = 1 - exp(-σ · δ)
    alpha = 1.0 - torch.exp(-sigma * delta)

    # 计算透射率 T_i = exp(-Σ_{j<i} σ_j · δ_j)
    # 使用累积连乘：T_i = Π_{j<i} (1 - α_j)，加 1e-10 防止除零
    transmittance = torch.cumprod(
        torch.cat([torch.ones(1, device=sigma.device), 1.0 - alpha + 1e-10]),
        dim=0,
    )[:-1]

    # 渲染权重: weights_i = α_i · T_i
    weights = alpha * transmittance

    # 像素颜色 = Σ weights_i · c_i
    rendered_color = (weights.unsqueeze(-1) * rgb).sum(dim=0)

    # 加权深度 = Σ weights_i · t_i
    depth = (weights * ray_distances).sum()

    return rendered_color, depth


# ============================================================
# 第 4 步：Instant-NGP 多层哈希编码
# ============================================================
# 原始 NeRF 的瓶颈：MLP 太慢。推理一张 800×600 的图片需要约 48 万次 MLP 前向传播。
# Instant-NGP（Müller et al., 2022）用多层哈希表替换了坐标的位置编码输入方式。
#
# 核心思想：
#   1. 将 3D 空间划分为多层分辨率的网格（类似多级纹理米普链）
#   2. 用哈希函数将网格单元映射到小型可学习 Embedding 表
#   3. 查询 3D 坐标时，读取该坐标在所有层级上的特征并拼接
#   4. 最后用一个小型 MLP（2 层）将拼接后的特征映射到 (σ, RGB)
#
# 效果：训练速度提升 100-1000 倍，但仍能保持与原始 NeRF 相近的质量。
# 局限：哈希冲突会导致伪影；3D Gaussian Splatting 进一步取代了 NeRF 的工程地位。


class TinyHashGrid(nn.Module):
    """
    多层哈希网格的简化教学实现。
    使用真正的 PyTorch 可微哈希 + 嵌入查找来演示原理。

    实际 Instant-NGP 实现（tiny-cuda-nn）使用 CUDA kernel，
    这里用 Python/Tensor 实现以保证可读性和可运行性。
    """

    def __init__(
        self,
        num_levels: int = 4,
        features_per_level: int = 4,
        base_resolution: int = 16,
        learnable_interp: bool = True,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.base_resolution = base_resolution
        self.learnable_interp = learnable_interp

        # 每层哈希表的容量：resolution * 4（随层级指数增长）
        self.hashes = []
        self.embeddings = []

        for level in range(num_levels):
            table_size = base_resolution * (2 ** level)
            # 哈希表大小限制：避免内存爆炸
            table_size = min(table_size, 65536)

            hash_offset = sum(min(base_resolution * (2 ** l), 65536) for l in range(level))

            # 为每一层创建独立的嵌入表
            emb = nn.Embedding(table_size, features_per_level)
            # Xavier 初始化
            nn.init.xavier_uniform_(emb.weight)
            self.embeddings.append(emb)

        self.register_buffer("_hash_offsets", torch.tensor(
            [sum(min(base_resolution * (2 ** l), 65536) for l in range(lvl))
             for lvl in range(num_levels)]
        ))
        self._hash_table_sizes = torch.tensor([
            min(base_resolution * (2 ** l), 65536)
            for l in range(num_levels)
        ])

        # 最终融合 MLP：将拼接的所有层级特征缩放到 SIREN 输入维度
        total_features = num_levels * features_per_level
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def _hash(self, value: torch.Tensor, table_size: int) -> torch.Tensor:
        """简单的 32 位多项式哈希（可微近似使用取模而非取整）。"""
        # 对浮点数取模：不是真正的整数哈希，但足以演示原理
        hashed = (value * 7351 + 17) % table_size
        return hashed

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        查询多层哈希网格。

        Args:
            positions: 归一化到 [0, 1] 的 3D 坐标，形状 (N, 3)

        Returns:
            融合后的特征向量，形状 (N, 32)
        """
        N = positions.shape[0]
        all_features = []

        for level_idx, emb in enumerate(self.embeddings):
            resolution = self.base_resolution * (2 ** level_idx)
            resolution = min(resolution, 65536 // 4)  # 安全上限

            # 将坐标缩放到 [0, resolution) 范围
            scaled_coords = positions * resolution

            # 获取每个坐标所在网格单元的索引
            ix = scaled_coords[:, 0].floor().long()
            iy = scaled_coords[:, 1].floor().long()
            iz = scaled_coords[:, 2].floor().long()

            # 对维度进行哈希以防止冲突
            ix_hashed = self._hash(ix.float(), resolution)
            iy_hashed = self._hash(iy.float(), resolution)
            iz_hashed = self._hash(iz.float(), resolution)

            # 组合三个维度的哈希值
            combined = ix_hashed * 73856093 ^ iy_hashed * 19349663 ^ iz_hashed * 83492791
            indices = combined % self._hash_table_sizes[level_idx]

            # 查找嵌入
            features = emb(indices)                      # (N, features_per_level)
            all_features.append(features)

        # 拼接所有层级的特征并融合
        concatenated = torch.cat(all_features, dim=-1)   # (N, num_levels * features_per_level)
        fused = self.fusion(concatenated)                # (N, 32)

        return fused


# ============================================================
# 主程序：完整流程演示
# ============================================================

def main():
    torch.manual_seed(0)
    device = torch.device("cpu")

    print("=" * 60)
    print("NeRF 从零实现教学演示")
    print("=" * 60)

    # --- 演示 1: 位置编码 ---
    print("\n[1] 位置编码")
    coords = torch.randn(5, 3)
    encoded = positional_encoding(coords, num_frequencies=10)
    print(f"  输入: {tuple(coords.shape)} 3D 坐标")
    print(f"  输出: {tuple(encoded.shape)} 编码向量")
    print(f"  （3 × 10 × 2 = 60 维：每个维度 10 层 sin/cos 频率）")

    # --- 演示 2: SIREN vs ReLU 对比 ---
    print("\n[2] SIREN 激活函数演示")
    siren_layer = SineLayer(3, 64, is_first=True)
    relu_layer = nn.Sequential(nn.Linear(3, 64), nn.ReLU())
    test_data = torch.randn(10, 3)
    siren_out = siren_layer(test_data)
    relu_out = relu_layer(test_data)[1]
    print(f"  SIREN 输出范围: [{siren_out.min():.4f}, {siren_out.max():.4f}]")
    print(f"  ReLU 输出范围: [{relu_out.min():.4f}, {relu_out.max():.4f}]")
    print(f"  SIREN 可以拟合振荡信号（高频），ReLU 只能拟合分段常数")

    # --- 演示 3: TinySirenNeRF 前向传播 ---
    print("\n[3] TinySirenNeRF 前向传播")
    nerf = TinySirenNeRF(
        num_pos_freqs=10,
        num_dir_freqs=4,
        hidden_dim=128,
        num_hidden_layers=4,
    ).to(device)
    num_params = sum(p.numel() for p in nerf.parameters())
    print(f"  模型参数量: {num_params:,}")

    # 随机生成 256 个采样点
    sample_positions = torch.randn(256, 3) * 2.0      # 在 [-2, 2]^3 范围内
    sample_directions = torch.randn(256, 3)             # 随机观察方向
    sample_directions = sample_directions / sample_directions.norm(dim=-1, keepdim=True)

    sigma, rgb = nerf(sample_positions, sample_directions)
    print(f"  输入坐标: {tuple(sample_positions.shape)}")
    print(f"  输出 sigma（密度）: {tuple(sigma.shape)}，范围 [{sigma.min():.4f}, {sigma.max():.4f}]")
    print(f"  输出 rgb（颜色）:   {tuple(rgb.shape)}，范围 [{rgb.min():.4f}, {rgb.max():.4f}]")

    # --- 演示 4: 体渲染 ---
    print("\n[4] 体渲染（单条光线）")
    num_samples = 64
    t_vals = torch.linspace(2.0, 6.0, num_samples)    # 距离相机 2~6 单位的 64 个采样点
    ray_sigma = torch.rand(num_samples) * 0.8          # 模拟体积密度
    ray_rgb = torch.rand(num_samples, 3)               # 模拟颜色
    rendered_color, depth = volumetric_render(ray_sigma, ray_rgb, t_vals)
    print(f"  采样点数: {num_samples}")
    print(f"  渲染颜色: [{rendered_color[0]:.4f}, {rendered_color[1]:.4f}, {rendered_color[2]:.4f}]")
    print(f"  加权深度: {depth:.4f}")
    print(f"  采样距离范围: [{t_vals[0]:.2f}, {t_vals[-1]:.2f}]")

    # --- 演示 5: 多层哈希网格 ---
    print("\n[5] 多层哈希网格编码（简化版 Instant-NGP）")
    hash_grid = TinyHashGrid(num_levels=4, features_per_level=4, base_resolution=16)
    hash_query = torch.rand(100, 3)                     # 100 个归一化的 3D 坐标
    hash_output = hash_grid(hash_query)
    print(f"  输入: {tuple(hash_query.shape)} 归一化 3D 坐标")
    print(f"  输出: {tuple(hash_output.shape)} 融合特征向量")
    print(f"  层级数: 4，每层特征: 4，融合后: 32 维")

    # --- 演示 6: 变换不变性验证 ---
    print("\n[6] 位置编码的变换性质验证")
    x1 = torch.randn(10, 3)
    x2 = x1 + 1.0
    enc1 = positional_encoding(x1)
    enc2 = positional_encoding(x2)
    # 平移不改变编码的相对差异
    diff = (enc1 - enc2).abs().mean()
    print(f"  随机两个点的编码差异均值: {diff:.6f}")
    print(f"  （编码对绝对位置敏感——这正是 NeRF 需要的）")

    # --- 演示 7: 体渲染的透射率可视化 ---
    print("\n[7] 体渲染透明度效应验证")
    # 高密度假设：射线穿过浓雾，近处就挡住了远处
    fog_sigma = torch.ones(num_samples) * 2.0
    fog_rgb = torch.linspace(0.0, 1.0, num_samples).view(-1, 1).repeat(1, 3)
    fog_color, fog_depth = volumetric_render(fog_sigma, fog_rgb, t_vals)
    # 低密度假设：射线穿过稀薄气体，大部分穿透
    mist_sigma = torch.ones(num_samples) * 0.05
    mist_color, mist_depth = volumetric_render(mist_sigma, fog_rgb, t_vals)
    print(f"  浓雾场景 (σ=2.0): 渲染颜色={fog_color.mean():.3f}, 深度={fog_depth:.3f}")
    print(f"  稀薄场景 (σ=0.05): 渲染颜色={mist_color.mean():.3f}, 深度={mist_depth:.3f}")
    print(f"  （高密度使前景像素更不透明，低密度允许背景穿透）")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
