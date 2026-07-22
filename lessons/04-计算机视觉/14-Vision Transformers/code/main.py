# === code/main.py — Vision Transformer 从零实现（教学版） ===
# 涵盖：ViT、Swin 核心思想、DETR 匈牙利匹配逻辑、注意力可视化
# 依赖：torch>=2.0, torchvision
# 安装：pip install torch torchvision


import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 1. Patch Embedding —— 将图像切分为片段并投影为词元嵌入
# ============================================================================


class PatchEmbedding(nn.Module):
    """将图像切分为片段并投影为词元嵌入。

    用单个 2D 卷积完成两件事：切分图像 + 投影到低维嵌入空间。
    例如 224x224 图像 + 16x16 patch_size -> 14x14 = 196 个词元。
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, dim: int = 192):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> conv -> (B, dim, H/p, W/p) -> flatten -> (B, dim, num_patches) -> transpose -> (B, num_patches, dim)
        x = self.proj(x)             # (B, dim, H/p, W/p)
        x = x.flatten(2)             # (B, dim, num_patches)
        x = x.transpose(1, 2)        # (B, num_patches, dim)
        return x


# ============================================================================
# 2. Transformer Block —— Pre-LN 标准编码块
# ============================================================================


class TransformerBlock(nn.Module):
    """标准的 Pre-LN Transformer 编码块。

    结构：x = x + MSA(LN(x))；x = x + FFN(LN(x))
    预归一化比后归一化在深层网络中训练更稳定。
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, need_attn: bool = False):
        # 多头自注意力——查询、键、值都是同一个输入（自注意力）
        if need_attn:
            attn_out, attn_weights = self.attn(
                self.norm1(x), self.norm1(x), self.norm1(x), need_weights=True
            )
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        else:
            attn_out, _ = self.attn(
                self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False
            )
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x


# ============================================================================
# 3. Swin Block 核心——窗口注意力 + 移位操作（教学简化版）
# ============================================================================


class ShiftWindowAttention(nn.Module):
    """教学简化的移位窗口注意力核心逻辑。

    Swin Transformer 不在全图上做自注意力，而是在局部窗口内计算。
    交替块将窗口网格偏移半个窗口大小，实现跨窗口信息流动。
    """
    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 3):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        # 窗口内的可学习双偏置 (2*Ws-1)(2*Ws-1) * nH
        self.mat = nn.Parameter(torch.randn((2 * window_size - 1), (2 * window_size - 1), num_heads))

    def _get_rel_bias(self) -> torch.Tensor:
        """获取相对位置偏置，用于修正窗口内各位置的注意力分数。"""
        return self.mat.unsqueeze(0)  # (1, 2*Ws-1, 2*Ws-1, nH)

    @staticmethod
    def _shifted_window_mask(w: int, shift_size: int) -> torch.Tensor:
        """构建移位窗口的掩码——确保每个位置只能看到同窗口内的其他位置。

        当窗口向某个方向偏移时，原来的相邻窗口边界发生变化，
        需要重新标记哪些位置在同一窗口内。
        """
        img_size = w * 2  # 假设原始大小为窗口大小的两倍
        # 创建相对坐标矩阵
        coord_h = torch.arange(w * 2)
        coord_w = torch.arange(w * 2)
        coord_grid = torch.stack(torch.meshgrid([coord_h, coord_w], indexing='ij'))  # (2, 2W)
        # 应用窗口偏移
        shifts = torch.tensor([[shift_size % w, shift_size % w]])
        windows_h = ((coord_grid[0:1] + shifts[0]) // w).flatten()
        windows_w = ((coord_grid[1:2] + shifts[1]) // w).flatten()
        # 同一窗口的位置对掩码设为 0，不同窗口设为 -inf
        window_diff = (windows_h[:, None] - windows_w[None, :]).abs()
        mask = window_diff != 0
        return (~mask).float()  # True = 可以在同一窗口内交互

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """前向传播（简化版，不实际执行完整的移位操作）。

        真实 Swin 实现会先对特征图做循环移位，再在窗口内做注意力，
        最后还原移位。教学版仅展示掩码构建逻辑。
        """
        B, L, D = x.shape
        W = self.window_size
        N = W * W  # 每个窗口内的词元数

        print(f"ShiftWindowAttention 教学演示")
        print(f"  批次大小: {B}, 总词元数: {L}, 嵌入维度: {D}")
        print(f"  窗口大小: {W}x{W} = {N} 词元/窗口")
        print(f"  头数: {self.num_heads}, 缩放因子: {self.scale:.4f}")

        # 计算需要的窗口数量
        num_windows_per_dim = max(1, math.ceil(math.sqrt(L)) // W)
        total_windows = num_windows_per_dim ** 2
        print(f"  估算窗口数: ~{total_windows} 个 (2D)")

        if attn_mask is not None:
            print(f"  移位窗口掩码已构建: ({attn_mask.shape})")

        # 对比：全图注意力 vs 窗口注意力的计算量
        global_attn_cost = L * L * D
        local_attn_cost = total_windows * N * N * D
        print(f"\n  计算复杂度对比:")
        print(f"    全局注意力: O({L}² × {D}) ≈ {global_attn_cost:,} 次乘法")
        print(f"    窗口注意力: O({total_windows} × {N}² × {D}) ≈ {local_attn_cost:,} 次乘法")
        print(f"    加速比: {global_attn_cost / max(local_attn_cost, 1):.2f}x")

        return x


class SwinBlock(nn.Module):
    """Swin Transformer 块的简化教学实现。

    核心创新：
    1. 窗口内自注意力（而非全局注意力）
    2. 交替块的移位窗口操作
    3. 层次化分辨率降低（通过 Patch Merging）
    """
    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 3, shift_size: int = 0):
        super().__init__()
        self.shift_size = shift_size
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ShiftWindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        W = self.window_size

        # 构建移位窗口掩码
        if self.shift_size > 0:
            attn_mask = ShiftWindowAttention._shifted_window_mask(W, self.shift_size)
        else:
            attn_mask = None

        print(f"\n--- Swin Block (shift={self.shift_size}) ---")
        self.attn(x, attn_mask)

        # 标准化 + 残差连接（简化展示）
        h = self.norm1(x)
        h = self.attn(h, attn_mask)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# 4. DETR 核心——匈牙利算法匹配（教学简化版）
# ============================================================================


class HungarianMatcher(nn.Module):
    """DETR 的匈牙利匹配器（教学简化版）。

    真实实现使用 scipy.optimize.linear_sum_assignment 求解最小代价匹配问题。
    这里用一个简化的二维距离矩阵来展示匹配逻辑。
    """
    def __init__(self, cost_class_weight: float = 1.0, cost_bbox_weight: float = 5.0,
                 cost_giou_weight: float = 2.0):
        super().__init__()
        self.cost_class = cost_class_weight
        self.cost_bbox = cost_bbox_weight
        self.cost_giou = cost_giou_weight

    @staticmethod
    def _sigmoid(bbox_pred: torch.Tensor) -> torch.Tensor:
        """将边界框预测转换为概率形式。"""
        x_center, y_center, width, height = bbox_pred.unbind(-1)
        pred_prob = torch.sigmoid(torch.stack([x_center, y_center, width, height], dim=-1))
        return pred_prob

    def compute_costs(self, pred_boxes: torch.Tensor, pred_logits: torch.Tensor,
                      target_boxes: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """计算预测与标注之间的成本矩阵。

        Args:
            pred_boxes: (M, 4) 预测边界框 [cx, cy, w, h]
            pred_logits: (M, num_classes) 预测类别 logits
            target_boxes: (N, 4) 真实边界框
            target_labels: (N,) 真实标签

        Returns:
            costs: (M, N) 成本矩阵——越小表示匹配越好
        """
        num_preds = pred_boxes.size(0)
        num_targets = target_boxes.size(0)

        if num_targets == 0 or num_preds == 0:
            return torch.zeros(num_preds, num_targets)

        # 类别成本：负对数概率（预测的类别越不像目标，成本越高）
        probs = pred_logits.softmax(-1)
        class_costs = -probs[:, target_labels]  # (M, N) 取目标类别的概率取负

        # 回归成本：L1 距离
        bbox_costs = torch.cdist(pred_boxes, target_boxes, p=1)  # (M, N)

        # 总成本 = 加权组合
        costs = (self.cost_class * class_costs +
                 self.cost_bbox * bbox_costs)

        return costs

    def match(self, pred_boxes: torch.Tensor, pred_logits: torch.Tensor,
              target_boxes: torch.Tensor, target_labels: torch.Tensor):
        """执行匈牙利匹配，返回预测到目标的映射。

        Returns:
            indices: List[Tensor] 每个批次的匹配索引 (M, 2) -> (pred_idx, target_idx)
        """
        costs = self.compute_costs(pred_boxes, pred_logits, target_boxes, target_labels)

        # 简化：贪心匹配（替代真正的 scipy 匈牙利算法）
        # 真实实现: row_ind, col_ind = linear_sum_assignment(costs.cpu())
        matches = []
        for b in range(costs.size(0)):
            batch_costs = costs[b]
            matched_targets = set()
            match_indices = []
            # 逐行选择最小代价且未匹配的列
            for m in range(batch_costs.size(0)):
                if batch_costs[m].nan_to_num(float('inf')).argmin().item() not in matched_targets:
                    best_col = batch_costs[m].nan_to_num(float('inf')).argmin().item()
                    if best_col < batch_costs.size(1):
                        match_indices.append([m, best_col])
                        matched_targets.add(best_col)
            if match_indices:
                matches.append(torch.tensor(match_indices, dtype=torch.long))

        return matches


def demo_hungarian_matching():
    """演示匈牙利匹配器的完整流程。"""
    print("=" * 60)
    print("DETR 匈牙利匹配演示")
    print("=" * 60)

    # 模拟数据：5 个预测框，4 个真实标注
    num_preds = 5
    num_targets = 4
    num_classes = 3

    torch.manual_seed(42)
    pred_boxes = torch.rand(num_preds, 4) * 100  # (5, 4)
    pred_logits = torch.randn(num_preds, num_classes)
    target_boxes = torch.rand(num_targets, 4) * 100  # (4, 4)
    target_labels = torch.randint(0, num_classes, (num_targets,))

    print(f"\n预测框数量: {num_preds}")
    print(f"真实标注数量: {num_targets}")
    print(f"类别数量: {num_classes}")
    print(f"真实标签: {target_labels.tolist()}")

    matcher = HungarianMatcher()
    costs = matcher.compute_costs(pred_boxes, pred_logits, target_boxes, target_labels)

    print(f"\n成本矩阵 ({num_preds} × {num_targets}):")
    print(f"{'':>10}", end="")
    for t in range(num_targets):
        print(f" T{t:>2}", end="")
    print()
    for m in range(num_preds):
        print(f" P{m:>2}", end=" ")
        for n in range(num_targets):
            print(f" {costs[m, n]:>8.3f}", end="")
        print()

    matches = matcher.match(pred_boxes, pred_logits, target_boxes, target_labels)
    print(f"\n匹配结果 (预测索引 -> 目标索引):")
    for match in matches:
        for pred_idx, target_idx in match:
            print(f"  预测 P{pred_idx.item()} <-> 标注 T{target_idx.item()} "
                  f"(代价: {costs[pred_idx.item(), target_idx.item()]:.4f})")

    unmatched_preds = num_preds - len(matches)
    print(f"\n未匹配的预测框: {unmatched_preds} 个 → 会被分类为背景（no object）")
    print("=" * 60)


# ============================================================================
# 5. 完整 ViT —— 组合所有组件
# ============================================================================


class ViT(nn.Module):
    """完整的 Vision Transformer。

    流水线：Patch Embedding -> 拼接 CLS Token -> 加位置编码 -> Transformer Block x N -> CLS 输出 -> 分类头
    """
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 10,
        dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, dim)
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B = x.size(0)

        patches = self.patch_embed(x)           # (B, num_patches, dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        tokens = torch.cat([cls_tokens, patches], dim=1)  # (B, num_patches+1, dim)
        tokens = tokens + self.pos_embed        # (B, num_patches+1, dim)

        if return_attn:
            attn_layers = []
            for block in self.blocks:
                tokens, attn_w = block(tokens, need_attn=True)
                attn_layers.append(attn_w)
            cls_output = self.norm(tokens[:, 0])
            return self.head(cls_output), attn_layers
        else:
            for block in self.blocks:
                tokens = block(tokens)
            cls_output = self.norm(tokens[:, 0])
            return self.head(cls_output)


# ============================================================================
# 6. 辅助工具 —— 形状追踪与可视化
# ============================================================================


def debug_shapes(model: nn.Module, input_shape: tuple = (2, 3, 64, 64)):
    """逐层打印张量形状的调试工具。"""
    print("=" * 50)
    print(f"ViT 形状追踪 | 输入: {input_shape}")
    print("=" * 50)

    model.eval()
    x = torch.randn(*input_shape)

    with torch.no_grad():
        patches = model.patch_embed(x)
        print(f"Patch embedding:      {x.shape} -> {patches.shape}")

        cls_tokens = model.cls_token.expand(input_shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        print(f"Concat CLS token:     {patches.shape} -> {tokens.shape}")

        tokens = tokens + model.pos_embed
        print(f"Add position embed:   {tokens.shape} -> {tokens.shape}")

        for i, block in enumerate(model.blocks):
            tokens = block(tokens)
            if i < 2 or i == len(model.blocks) - 1:
                print(f"Block {i+1:>2d}:      {tokens.shape}")

        cls_output = model.norm(tokens[:, 0])
        print(f"CLS output (norm):    {tokens.shape} -> {cls_output.shape}")

        logits = model.head(cls_output)
        print(f"Classification head:  {cls_output.shape} -> {logits.shape}")
    print("=" * 50)


@torch.no_grad()
def visualize_attention(model: nn.Module, x: torch.Tensor, layer_idx: int = 0):
    """可视化指定 Transformer 块的注意力权重。"""
    model.eval()
    B = x.size(0)

    patches = model.patch_embed(x)
    cls_tokens = model.cls_token.expand(B, -1, -1)
    tokens = torch.cat([cls_tokens, patches], dim=1)
    tokens = tokens + model.pos_embed

    attn_block = model.blocks[layer_idx]
    _, attn_weights = attn_block.attn(
        attn_block.norm1(tokens),
        attn_block.norm1(tokens),
        attn_block.norm1(tokens),
        need_weights=True,
    )

    chars = " .:-=+*#%@"
    patch_len = int(math.sqrt(attn_weights.shape[2] - 1))
    display_size = min(patch_len, 10)

    for head in range(min(attn_weights.shape[1], 3)):
        print(f"\n第 {layer_idx} 块, 注意力头 {head} 的热力图:")
        print(f"          {'CLS':>4} ", end="")
        for j in range(display_size):
            print(f"p{j+1:>4}", end="")
        print()

        attn_map = attn_weights[0, head, :, :].cpu().numpy()
        for i in range(min(display_size + 1, attn_map.shape[0])):
            row_label = "CLS" if i == 0 else f"p{i}"
            print(f"{row_label:>4} ", end="")
            for j in range(min(display_size + 1, attn_map.shape[1])):
                weight = attn_map[i, j]
                idx = min(int(weight * (len(chars) - 1)), len(chars) - 1)
                print(f" {chars[idx]} ", end="")
            print()

        print(f"  (注: ' ' 低注意力 -> '@' 高注意力)\n")

    model.train()


# ============================================================================
# 7. 训练循环演示
# ============================================================================


def train_vit_on_synthetic_data(model, num_epochs: int = 5, lr: float = 1e-3):
    """在合成数据上演示 ViT 的训练流程。"""
    print("\n" + "=" * 50)
    print("在合成数据上训练 ViT")
    print("=" * 50)

    # 生成合成数据（3 类随机图像模式）
    num_samples = 600
    image_size = 64
    X = torch.randn(num_samples, 3, image_size, image_size)
    y = torch.randint(0, 3, (num_samples,))

    # 每类注入不同的通道模式，让模型可以学到一些信号
    for i in range(num_samples):
        if y[i] == 0:
            X[i, 0] = X[i, 0] + 0.5  # 第 1 类：增强红色通道
        elif y[i] == 1:
            X[i, 1] = X[i, 1] + 0.5  # 第 2 类：增强绿色通道
        else:
            X[i, 2] = X[i, 2] + 0.5  # 第 3 类：增强蓝色通道

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(-1) == batch_y).sum().item()
            total += batch_x.size(0)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_correct += (logits.argmax(-1) == batch_y).sum().item()
                val_total += batch_x.size(0)

        train_acc = correct / total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Loss: {total_loss/total:.4f} | "
              f"Train Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss/val_total:.4f} | "
              f"Val Acc: {val_acc:.2%}")

    print("=" * 50)


# ============================================================================
# 主程序入口
# ============================================================================


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # --- Step 1: 创建模型并检查形状 ---
    print("# 1. 创建 ViT 并追踪形状\n")
    model = ViT(
        image_size=64, patch_size=16,
        num_classes=10, dim=192, depth=6, num_heads=3,
    ).to(device)
    debug_shapes(model, input_shape=(2, 3, 64, 64))

    # --- Step 2: 单样本推理 ---
    print("\n# 2. 单样本推理测试\n")
    x = torch.randn(1, 3, 64, 64).to(device)
    logits = model(x)
    probs = logits.softmax(-1)
    print(f"输入形状: {x.shape}")
    print(f"logits 形状: {logits.shape}")
    print(f"预测类别: {probs.argmax(-1).item()}")
    print(f"最大置信度: {probs.max().item():.4f}")
    print(f"概率总和: {probs.sum().item():.6f} (应为 1.0)")

    # --- Step 3: 参数量统计 ---
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n# 3. 参数量统计: {total_params:,}")

    # --- Step 4: 注意力可视化 ---
    print("\n# 4. 注意力可视化 (第 0 块)\n")
    x_batch = torch.randn(1, 3, 64, 64).to(device)
    visualize_attention(model, x_batch, layer_idx=0)

    # --- Step 5: Swin Transformer 核心逻辑演示 ---
    print("\n# 5. Swin Transformer 移位窗口注意力演示\n")
    swin_block = SwinBlock(dim=64, window_size=7, num_heads=3, shift_size=3)
    batch_size = 2
    seq_len = 49  # 7x7 窗口
    swin_input = torch.randn(batch_size, seq_len, 64)
    _ = swin_block(swin_input)

    # 第二块带移位
    print("\n--- Swin Block (shift=3, 交叉窗口) ---")
    swin_shift = SwinBlock(dim=64, window_size=7, num_heads=3, shift_size=3)
    _ = swin_shift(swin_input)

    # --- Step 6: DETR 匈牙利匹配演示 ---
    print("\n# 6. DETR 匈牙利匹配演示")
    demo_hungarian_matching()

    # --- Step 7: 在合成数据上训练 ---
    print("\n# 7. 合成数据训练演示\n")
    small_model = ViT(
        image_size=64, patch_size=16,
        num_classes=3, dim=64, depth=3, num_heads=2,
    ).to(device)
    train_vit_on_synthetic_data(small_model, num_epochs=5, lr=5e-4)

    # --- 总结 ---
    print("\n" + "=" * 50)
    print("全部演示完成！")
    print("=" * 50)
    print("""
ViT 核心要点回顾:
1. Patch Embedding: 一个 Conv 同时完成切分和投影
2. CLS Token: 可学习向量聚合全局信息
3. Positional Embedding: 可学习向量补偿无位置感知
4. Pre-LN: 先归一化再注意力, 训练更稳定
5. Swin: 窗口注意力 + 移位窗口 = 线性复杂度 + 跨窗口信息交换
6. DETR: 集合预测 + 匈牙利匹配 = 端到端检测, 无需 NMS
""")
