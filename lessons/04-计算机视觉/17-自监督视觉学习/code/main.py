# main.py — 自监督视觉学习：从零实现对比学习与掩码建模
# 依赖：torch>=2.0, torchvision>=0.15, numpy
# 安装：pip install torch torchvision numpy
# 对应课程：阶段 04 · 17（自监督视觉学习）

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 第 1 步：NT-Xent (InfoNCE) 损失 — 对比学习的核心
# ============================================================================

def nt_xent_loss(z1, z2, temperature=0.1):
    """
    计算 NT-Xent（InfoNCE）损失。

    给定一批图像的两个增强视图，将同一图像的两个视图作为正样本对，
    批次中其他所有样本的视图作为负样本。

    Args:
        z1: (N, D) 第一个增强视图的嵌入向量（未归一化）
        z2: (N, D) 第二个增强视图的嵌入向量（未归一化）
        temperature: 温度参数 tau，控制分布的尖锐程度

    Returns:
        loss: 标量损失值
    """
    batch_size = z1.shape[0]

    # 先 L2 归一化，这样点积等于余弦相似度
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # 拼接两个视图的嵌入，得到 (2N, D)
    embeddings = torch.cat([z1, z2], dim=0)

    # 计算所有样本对的余弦相似度矩阵 (2N, 2N)
    similarity_matrix = embeddings @ embeddings.T / temperature

    # 构建掩码：移除对角线元素（自身与自己比较没有意义）
    # mask[i,j]=True 表示该位置不参与损失计算
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

    # 目标标签：对于第 i 个样本的第一个视图，目标是它的第二个视图（索引 N+i）
    # 对于第 i 个样本的第二个视图，目标是它第一个视图（索引 i）
    labels = torch.cat(
        [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)],
        dim=0,
    ).to(z1.device)

    # CrossEntropyLoss 内部会执行 log-softmax + NLL
    return F.cross_entropy(similarity_matrix, labels)


# ============================================================================
# 第 2 步：MoCo 风格的动量队列 — 用历史特征充当额外负样本
# ============================================================================

class MoCoQueue(nn.Module):
    """
    动量对比学习队列。

    SimCLR 需要很大的批次来提供足够的负样本（512+），而 MoCo
    通过维护一个动态队列，用之前批次提取的特征作为负样本来绕过这个限制。
    同时使用动量编码器（EMA）保证特征的一致性。
    """

    def __init__(self, embedding_dim=128, queue_size=4096, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size

        # 键编码器（动量编码器）的初始权重，与查询编码器相同
        self.register_buffer("queue", torch.randn(embedding_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)

        # 动量系数，决定新旧信息的融合速度
        self.momentum = 0.999

    @torch.no_grad()
    def update_queue(self, new_features):
        """
        用新特征更新队列：新到前面，旧的往后推，溢出部分丢弃。

        Args:
            new_features: (K, D) 当前批次键编码器的输出
        """
        K = new_features.shape[0]
        if K >= self.queue_size:
            # 如果批次比队列还大，直接覆盖
            self.queue[:, :K] = new_features.T
        else:
            # 滚动更新
            move_amount = self.queue_size - K
            self.queue[:, move_amount:] = self.queue[:, :move_amount]
            self.queue[:, :K] = new_features.T

    @torch.no_grad()
    def update_momentum_encoder(self, encoder_q, encoder_k):
        """
        用指数移动平均（EMA）更新动量编码器。

        Args:
            encoder_q: 查询编码器（学生），正在做梯度更新
            encoder_k: 动量编码器（教师），权重缓慢跟随学生变化
        """
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = self.momentum * param_k.data + (
                1.0 - self.momentum
            ) * param_q.data.detach()


def demo_moco_queue():
    """演示 MoCo 队列的使用过程。"""
    print("[MoCo 队列演示]")
    queue = MoCoQueue(embedding_dim=32, queue_size=16, temperature=0.07)

    # 模拟三个批次的更新
    for batch_idx in range(3):
        batch_size = 4
        query = F.normalize(torch.randn(batch_size, 32), dim=-1)
        key = F.normalize(torch.randn(batch_size, 32), dim=-1)

        queue.update_queue(key)

        # 计算对比损失（简化版：只对当前批次内和队列中的 key 配对）
        sim_matrix = (query @ queue.queue / queue.temperature).T
        logits = sim_matrix  # (batch_size, queue_size)

        # 每个 query 的目标是它与自己的那个 key（队列的最后一批）
        labels = torch.full((batch_size,), fill_value=batch_size - 1, dtype=torch.long)

        loss = F.cross_entropy(logits, labels)
        print(f"  批次 {batch_idx + 1}（队列大小 = {min((batch_idx + 1) * batch_size, 16)}）: " f"loss = {loss.item():.4f}")

    print()


# ============================================================================
# 第 3 步：MAE 风格随机掩码
# ============================================================================

def random_mask_indices(num_patches, mask_ratio=0.75, seed=0):
    """
    MAE 风格的随机掩码生成器。

    选择一定比例的 token 作为可见部分，其余标记为掩码。
    返回排序后的索引，确保结果可复现。

    Args:
        num_patches: 总片段数（如 ViT-S/16 在 224x224 图像上为 196）
        mask_ratio: 掩码比例（推荐 0.75）
        seed: 随机种子，保证可复现性

    Returns:
        visible: 可见片段的排序后索引，形状 (n_keep,)
        masked: 掩码片段的排序后索引，形状 (n_masked,)
    """
    g = torch.Generator().manual_seed(seed)
    n_keep = int(num_patches * (1 - mask_ratio))

    # 随机打乱，取前 n_keep 个作为可见片段
    perm = torch.randperm(num_patches, generator=g)
    visible = perm[:n_keep].sort().values
    masked = perm[n_keep:].sort().values
    return visible, masked


class MaskedAutoencoderViT(nn.Module):
    """
    最简版 MAE 架构，用于演示 Encoder-Decoder 设计。

    核心设计：
    - 大编码器只处理 25% 的可见片段
    - 小解码器接收所有片段（含 mask token）
    - 只在掩码位置上计算重建损失
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 encoder_dim=384, decoder_dim=512, encoder_layers=6,
                 decoder_layers=6, num_patches=196, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio

        # Patch Embedding：将图像切块并投影到 d_model 维
        self.patch_embed = nn.Conv2d(
            in_chans, encoder_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, encoder_dim))

        # 编码器：Transformer 编码块
        encoder_block = nn.TransformerEncoderLayer(
            d_model=encoder_dim, nhead=6, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_block, num_layers=encoder_layers)

        # 解码器：处理所有 token（含掩码位置）
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        decoder_block = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=8, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_block, num_layers=decoder_layers)
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # 重建头：预测原始像素值
        self.head = nn.Linear(decoder_dim, in_chans * patch_size * patch_size)

        # 初始化权重
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward_encoder(self, x, visible_mask):
        """
        编码可见片段。

        Args:
            x: 输入图像 (B, C, H, W)
            visible_mask: 可见片段索引 (B, n_visible)

        Returns:
            encoded: 编码后的特征 (B, n_visible+1, encoder_dim)
                     +1 包含 CLS token
        """
        B = x.shape[0]

        # Patch 嵌入
        patches = self.patch_embed(x)  # (B, D, H//P, W//P)
        patches = patches.flatten(2)   # (B, D, N)
        patches = patches.transpose(1, 2)  # (B, N, D)

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_with_cls = torch.cat([cls_tokens, patches], dim=1)
        x_with_cls = x_with_cls + self.pos_embed

        # 根据可见掩码选择片段
        n_total = x_with_cls.shape[1]  # N + 1 (含 CLS)
        cls_pos = torch.tensor(0, device=x.device)

        # 对 batch 中每个样本分别选择可见片段
        encoded_list = []
        for b in range(B):
            # cls_token 始终保留，加上可见的 patches
            visible_with_cls = torch.cat([
                x_with_cls[b:b + 1, cls_pos:cls_pos + 1],
                x_with_cls[b:b + 1, visible_mask[b] + 1],
            ], dim=1)
            encoded_list.append(visible_with_cls)

        x_enc = torch.cat(encoded_list, dim=0)

        # 通过编码器
        x_enc = self.encoder(x_enc)
        return x_enc

    def forward_decoder(self, x_enc, visible_mask):
        """
        解码器接收编码后的可见片段，补充 mask token 后完成重建。

        Args:
            x_enc: 编码器输出 (B, n_visible+1, encoder_dim)
            visible_mask: 可见片段索引 (B, n_visible)

        Returns:
            predictions: 重建的像素值 (B, n_masked, P*P*C)
            mask_indices: 掩码位置 (B, n_masked)
        """
        B = x_enc.shape[0]
        n_visible = x_enc.shape[1] - 1  # 减去 CLS token
        n_masked = self.num_patches - n_visible
        D = x_enc.shape[-1]

        # 将编码特征映射到解码器维度
        x_dec = self.decoder_embed(x_enc)

        # 在每个掩码位置插入 mask token
        mask_tokens = self.mask_token.expand(B, n_masked + 1, -1)
        # mask_tokens[0] 占据 CLS 位置占位，实际会被替换
        x_dec_ = torch.cat([x_dec[:, :1, :], mask_tokens[:, 1:, :]], dim=-2)

        # 为每个样本构造完整的 token 序列（CLS + visible + masked）
        decoder_inputs = []
        for b in range(B):
            visible_tokens = x_dec[b:b + 1, :n_visible + 1, :]
            masked_tokens = mask_tokens[b:b + 1, 1:n_masked + 1, :]
            full_seq = torch.cat([visible_tokens, masked_tokens], dim=1)
            decoder_inputs.append(full_seq)

        x_dec = torch.cat(decoder_inputs, dim=0)

        # 通过解码器
        x_dec = self.decoder(x_dec)
        x_dec = self.decoder_norm(x_dec)

        # 只取出掩码位置的预测
        predictions = []
        for b in range(B):
            pred = self.head(x_dec[b:b + 1, n_visible + 1:])
            predictions.append(pred)

        predictions = torch.cat(predictions, dim=0)
        return predictions

    def forward(self, x, visible_mask):
        """
        完整的前向传播：编码可见片段 → 解码重建 → 计算掩码位置 MSE 损失。

        Args:
            x: 输入图像 (B, C, H, W)
            visible_mask: 可见片段索引 (B, n_visible)

        Returns:
            loss: 仅计算掩码位置的重建 MSE 损失
        """
        latent = self.forward_encoder(x, visible_mask)
        pred = self.forward_decoder(latent, visible_mask)

        # Ground truth：提取掩码位置的原始 patch 像素
        patches = self.patch_embed(x).flatten(2)  # (B, D, N)
        patches = patches.transpose(1, 2)  # (B, N, D)

        target_list = []
        for b in range(patches.shape[0]):
            mask_idx = visible_mask[b] + 1  # +1 因为 pos_embed 包含 CLS
            target_list.append(patches[b:b + 1, mask_idx])
        target = torch.cat(target_list, dim=1)

        # 计算重建损失（仅在掩码位置）
        loss = F.mse_loss(pred, target)
        return loss


# ============================================================================
# 第 4 步：BYOL 风格不匹配对比 — 无负样本的学习方式
# ============================================================================

class SimpleMLP(nn.Module):
    """最简单的三层投影头，用于 BYOL/DINO 的教学演示。"""

    def __init__(self, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


def demo_dino_head_centre_sharpen():
    """
    演示 DINO 训练头的中心化和锐化机制。

    这是防止表示崩溃（representation collapse）的关键技巧：
    - 中心化（Centering）：减去教师输出的移动平均，防止某一维度过度激活
    - 锐化（Sharpening）：用低温度 softmax，让教师输出更确定
    """
    print("[DINO 教师头的中心化与锐化演示]")

    # 模拟 64 个样本、128 维输出的特征
    feat_dim = 128
    num_samples = 64

    student_head = SimpleMLP(feat_dim, 256, feat_dim)
    teacher_head = SimpleMLP(feat_dim, 256, feat_dim)

    # 用相同的权重初始化
    with torch.no_grad():
        teacher_head.load_state_dict(student_head.state_dict())

    center = torch.zeros(feat_dim)
    momentum = 0.996
    student_temp = 0.1
    teacher_temp = 0.04

    student_feat = torch.randn(num_samples, feat_dim)
    teacher_feat = torch.randn(num_samples, feat_dim)

    print(f"  学生温度 = {student_temp}，教师温度 = {teacher_temp}")
    print(f"  学生输出熵（未中心化）: ", end="")

    # 学生输出（标准 softmax）
    student_out = F.softmax(student_head(student_feat) / student_temp, dim=-1)
    entropy_student = -(student_out * torch.log(student_out + 1e-8)).sum(dim=-1).mean()
    print(f"{entropy_student.item():.4f}")
    print(f"  教师输出熵（未中心化）: ", end="")

    # 教师输出（未中心化）
    teacher_out_raw = F.softmax(teacher_head(teacher_feat) / teacher_temp, dim=-1)
    entropy_teacher_raw = -(teacher_out_raw * torch.log(teacher_out_raw + 1e-8)).sum(dim=-1).mean()
    print(f"{entropy_teacher_raw.item():.4f}")
    print(f"    （高熵意味着输出接近均匀分布——模型没有置信度）")

    # 应用中心化（从教师输出减去移动平均）
    teacher_output_centered = teacher_head(teacher_feat) - center
    teacher_out_sharp = F.softmax(teacher_output_centered / teacher_temp, dim=-1)

    # 更新中心
    with torch.no_grad():
        current_mean = teacher_head(teacher_feat).mean(dim=0)
        center = momentum * center + (1 - momentum) * current_mean

    # 计算中心化和锐化后的熵
    entropy_teacher_sharp = -(teacher_out_sharp * torch.log(teacher_out_sharp + 1e-8)).sum(dim=-1).mean()
    print(f"\n  教师输出熵（中心化 + 锐化后）: {entropy_teacher_sharp.item():.4f}")
    print(f"    （熵降低意味着输出更集中，模型对某些维度有了置信度）")

    print(f"\n  结论：没有中心化和锐化的教师输出趋向均匀分布，")
    print(f"  学生学不到有意义的东西——这就是表示崩溃。")
    print()


# ============================================================================
# 主程序：运行所有演示
# ============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. NT-Xent 损失验证
    # ------------------------------------------------------------------
    print("=" * 60)
    print("1. NT-Xent（InfoNCE）损失验证")
    print("=" * 60)

    batch_size = 16
    embed_dim = 128

    # 场景 A：同一图像的两个视图 → 应该得到低损失
    z = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
    z_view1 = z
    z_view2 = z.clone()
    loss_identical = nt_xent_loss(z_view1, z_view2, temperature=0.1).item()
    print(f"  相同视图对 (N={batch_size}) 的 InfoNCE 损失: {loss_identical:.4f}")
    print(f"    （期望值 ≈ 0，因为正样本完全对齐）")

    # 场景 B：随机配对 → 应该接近 log(2N-1)
    z_random = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
    loss_random = nt_xent_loss(z_view1, z_random, temperature=0.1).item()
    expected_random = math.log(2 * batch_size - 1)
    print(f"  随机视图对的 InfoNCE 损失: {loss_random:.4f}")
    print(f"    （期望值 ≈ log(2N-1) = log({2 * batch_size - 1}) = {expected_random:.4f}）")

    # 场景 C：部分相关 → 介于两者之间
    z_partial = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
    # 让一半的 z_partial 与 z_view1 相同
    z_partial[::2] = z.view1 if False else z[::2]
    z_partial[1::2] = F.normalize(torch.randn(batch_size // 2, embed_dim), dim=-1)
    loss_partial = nt_xent_loss(z, z_partial, temperature=0.1).item()
    print(f"  部分相关视图对的 InfoNCE 损失: {loss_partial:.4f}")
    print(f"    （介于 0 和 {expected_random:.4f} 之间——符合预期）")

    # 温度敏感性
    print(f"\n  温度敏感性测试（N={batch_size}，完全对齐的视图对）:")
    for tau in [0.05, 0.1, 0.2, 0.5]:
        l = nt_xent_loss(
            F.normalize(torch.randn(32, embed_dim), dim=-1),
            F.normalize(torch.randn(32, embed_dim), dim=-1),
            temperature=tau,
        ).item()
        print(f"    tau = {tau:>5}: loss = {l:.4f}")

    # ------------------------------------------------------------------
    # 2. MoCo 队列演示
    # ------------------------------------------------------------------
    demo_moco_queue()

    # ------------------------------------------------------------------
    # 3. MAE 掩码生成演示
    # ------------------------------------------------------------------
    print("=" * 60)
    print("3. MAE 随机掩码生成")
    print("=" * 60)

    num_patches_options = [196, 576, 256]
    for np_ in num_patches_options:
        visible, masked = random_mask_indices(np_, mask_ratio=0.75)
        print(f"  ViT patch 数 = {np_:>3} ({np_**0.5:.0f}x{np_**0.5:.0f}) -> " f"可见 {len(visible):>3}，掩码 {len(masked):>3} ({len(masked)/np_*100:.0f}%)")

    # ------------------------------------------------------------------
    # 4. DINO 教师头中心化演示
    # ------------------------------------------------------------------
    demo_dino_head_centre_sharpen()

    # ------------------------------------------------------------------
    # 5. 数据增强策略对比
    # ------------------------------------------------------------------
    print("=" * 60)
    print("5. 自监督数据增强的重要性")
    print("=" * 60)
    print()
    print("  对比学习中，两个增强视图必须是：")
    print("  - 外观不同（让模型学会不变性）")
    print("  - 语义相同（保持标签一致）")
    print()
    print("  关键增强操作及它们教给模型的不变性：")
    print()
    print("  | 增强操作              | 教的不变性               | 强度建议 |")
    print("  |----------------------|------------------------|---------|")
    print("  | 随机水平翻转          | 物体方向无关             | p=0.5   |")
    print("  | 颜色抖动（亮度/对比度）| 光照条件无关             | 0.4     |")
    print("  | 随机裁剪缩放          | 尺度和位置无关           | 0.2~1.0 |")
    print("  | 随机灰度              | 颜色无关                 | p=0.2   |")
    print("  | Gaussian Blur        | 分辨率/清晰度无关        | p=0.5   |")
    print("  | Solarization         | 高级光照鲁棒性           | p=0.2   |")
    print()
    print("  ⚠ 注意：增强太弱 → 模型学到 trivial 特征（比如只记住了背景）")
    print("  ⚠ 太强 → 语义被破坏（翻转之后不再是同一个物体）")
    print()

    # ------------------------------------------------------------------
    # 6. 方法选型速查表
    # ------------------------------------------------------------------
    print("=" * 60)
    print("6. 自监督方法选型参考")
    print("=" * 60)
    print()
    print("  | 方法       | 核心思想                | 需要的负样本 | 典型 GPU 时 | 适用场景           |")
    print("  |-----------|-----------------------|-----------|----------|-----------------|")
    print("  | SimCLR    | 同图两视图拉到一起，异图推开 | 批次本身    | ~1000     | 分类，中小数据集       |")
    print("  | MoCo      | 用队列缓存历史特征当负样本   | 队列中缓存  | ~1000     | 分类，GPU 显存有限     |")
    print("  | BYOL      | 学生预测教师输出，无需负样本  | 无需       | ~1000     | 分类，不需要大批次     |")
    print("  | DINO      | 多裁剪教师-学生蒸馏       | 无需       | ~1000     | 通用特征，检测/分割     |")
    print("  | MAE       | 掩码重建可见像素         | 无需       | ~1000     | ViT，下游密集任务      |")
    print("  | DINOv2    | DINO 的大规模工业版本    | 无需       | ~160000   | 最强通用特征（2026）   |")
    print()
    print("  2026 年生产默认推荐：DINOv2（特征质量最强）或 MAE（高效预训练）")
    print()

    print("=" * 60)
    print("全部演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
