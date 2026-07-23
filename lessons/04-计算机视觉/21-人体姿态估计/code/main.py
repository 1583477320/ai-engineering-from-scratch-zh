# main.py — 从零实现人体姿态估计核心流程
# 依赖：torch>=2.0, numpy>=1.24
# 安装：pip install torch numpy
# 对应课程：阶段 04 · 21（人体姿态估计）

"""
人体姿态估计从零实现：
1. 高斯热力图生成（训练目标）
2. 小型关键点检测网络（类 Hourglass 架构）
3. 子像素精度提取
4. 合成数据训练流水线
5. 评估与可视化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 第 1 步：高斯热力图生成
# ============================================================

def gaussian_heatmap(size, cx, cy, sigma=2.0):
    """生成单个关键点的高斯热力图。

    热力图是姿态估计的训练目标——每个关键点对应一张 H×W 的图，
    高斯峰值位于关键点的真实坐标处。网络学习预测这些热力图，
    推理时取 argmax 得到坐标。

    Args:
        size: 热力图的空间尺寸（正方形）
        cx: 关键点 x 坐标（列）
        cy: 关键点 y 坐标（行）
        sigma: 高斯核标准差，控制热力图的"模糊程度"

    Returns:
        shape (size, size) 的 float32 热力图
    """
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


# ============================================================
# 第 2 步：小型关键点检测网络（类 Hourglass 架构）
# ============================================================

class ResBlock(nn.Module):
    """残差卷积块——Hourglass 网络的基础组件。"""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class TinyKeypointNet(nn.Module):
    """简化版姿态估计网络。

    架构思路借鉴 Hourglass Network：
    - 编码路径：两层下采样，逐步扩大感受野
    - 中间瓶颈：残差块处理最深层特征
    - 解码路径：转置卷积上采样恢复空间分辨率

    输入 (N, 3, H, W)，输出 (N, K, H, W) 的热力图。
    K = 关键点数量（COCO 标准为 17）。
    """

    def __init__(self, num_keypoints=17, base_channels=64):
        super().__init__()
        # 编码路径：逐步下采样
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResBlock(base_channels),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResBlock(base_channels * 2),
        )
        # 瓶颈层：处理最低分辨率的特征
        self.bottleneck = ResBlock(base_channels * 2)
        # 解码路径：转置卷积上采样
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResBlock(base_channels),
        )
        self.dec2 = nn.ConvTranspose2d(base_channels, num_keypoints, 4, stride=2, padding=1)

    def forward(self, x):
        h1 = self.enc1(x)       # (N, base, H/2, W/2)
        h2 = self.enc2(h1)      # (N, base*2, H/4, W/4)
        h3 = self.bottleneck(h2) # (N, base*2, H/4, W/4)
        u1 = self.dec1(h3)      # (N, base, H/2, W/2)
        u2 = self.dec2(u1)      # (N, K, H, W)
        return u2


# ============================================================
# 第 3 步：热力图到坐标提取（含子像素精度）
# ============================================================

def heatmap_to_coords(heatmaps):
    """从热力图中提取关键点坐标（整数 argmax）。

    Args:
        heatmaps: (N, K, H, W) 张量

    Returns:
        coords: (N, K, 2) 浮点坐标，格式为 [x, y]
        conf: (N, K) 每个关键点的置信度（热力图峰值）
    """
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    conf, idx = flat.max(dim=-1)
    xs = (idx % W).float()
    ys = (idx // W).float()
    coords = torch.stack([xs, ys], dim=-1)
    return coords, conf


def subpixel_refine(heatmaps, coords):
    """子像素精度优化。

    整数 argmax 有最高 0.5 像素的量化误差。对于需要精确坐标的
    场景（体育分析、医学标志点），子像素精度是必需的。

    方法：在 argmax 位置的邻域上做一阶差分估计。
    dx = 0.25 * (hm[y, x+1] - hm[y, x-1])
    这个系数是高斯热力图的近似最优值。

    Args:
        heatmaps: (N, K, H, W) 热力图
        coords: (N, K, 2) 整数坐标

    Returns:
        refined: (N, K, 2) 子像素精度坐标
    """
    N, K, H, W = heatmaps.shape
    refined = coords.clone()
    for n in range(N):
        for k in range(K):
            x, y = int(coords[n, k, 0]), int(coords[n, k, 1])
            # 边界检查：边缘像素无法做差分，保持原值
            if 0 < x < W - 1 and 0 < y < H - 1:
                hm = heatmaps[n, k]
                dx = 0.25 * (hm[y, x + 1] - hm[y, x - 1])
                dy = 0.25 * (hm[y + 1, x] - hm[y - 1, x])
                refined[n, k, 0] = x + dx
                refined[n, k, 1] = y + dy
    return refined


# ============================================================
# 第 4 步：合成数据生成
# ============================================================

def make_synthetic_sample(image_size=64, num_keypoints=5, rng=None):
    """生成一个合成关键点样本。

    关键点形成一个简化的骨架结构——有固定的相对关系，
    而不是随机散布。这更接近真实场景：关键点之间有结构性依赖。

    关键点示意（5 点简化骨架）：
        0（头部中心）
           / \
        1（左肩）  2（右肩）
          |        |
        3（左手）  4（右手）

    每次生成时整体做随机平移和缩放，增加多样性。

    Args:
        image_size: 图像尺寸
        num_keypoints: 关键点数量
        rng: 随机数生成器

    Returns:
        img: (3, H, W) 图像
        hms: (K, H, W) 热力图目标
        kps: (K, 2) 关键点坐标 [x, y]
    """
    rng = rng or np.random.default_rng()
    img = np.ones((3, image_size, image_size), dtype=np.float32)

    # 骨架在标准化坐标系中的相对位置（归一化到 [-1, 1]）
    skeleton = np.array([
        [0.0, -0.6],   # 0: 头部
        [-0.5, -0.1],  # 1: 左肩
        [0.5, -0.1],   # 2: 右肩
        [-0.6, 0.6],   # 3: 左手
        [0.6, 0.6],    # 4: 右手
    ])

    # 随机缩放（骨架大小变化）和平移（骨架位置变化）
    scale = rng.uniform(0.4, 0.6)
    center_x = rng.uniform(image_size * 0.35, image_size * 0.65)
    center_y = rng.uniform(image_size * 0.35, image_size * 0.65)

    kps = np.zeros((num_keypoints, 2))
    for i, (dx, dy) in enumerate(skeleton):
        cx = int(center_x + dx * image_size * scale)
        cy = int(center_y + dy * image_size * scale)
        cx = np.clip(cx, 5, image_size - 5)
        cy = np.clip(cy, 5, image_size - 5)
        kps[i] = [cx, cy]
        # 每个关键点用不同灰度值的色块表示，让网络可以区分它们
        gray_val = 0.2 + 0.15 * i  # 从深到浅：0.2, 0.35, 0.5, 0.65, 0.8
        radius = 3 if i == 0 else 2
        x0, y0 = max(int(cx) - radius, 0), max(int(cy) - radius, 0)
        x1, y1 = min(int(cx) + radius + 1, image_size), min(int(cy) + radius + 1, image_size)
        img[:, y0:y1, x0:x1] = gray_val

    # 生成热力图目标
    hms = np.stack([gaussian_heatmap(image_size, int(kps[i, 0]), int(kps[i, 1]))
                    for i in range(num_keypoints)])
    return img, hms, kps.astype(np.float32)


# ============================================================
# 第 5 步：训练与评估
# ============================================================

def train_and_evaluate():
    """完整的训练和评估流程。"""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    NUM_KEYPOINTS = 5
    IMAGE_SIZE = 64
    BATCH_SIZE = 32
    NUM_STEPS = 500
    LEARNING_RATE = 5e-3

    # 初始化模型和优化器
    model = TinyKeypointNet(num_keypoints=NUM_KEYPOINTS, base_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"关键点数量: {NUM_KEYPOINTS}")
    print(f"训练步数: {NUM_STEPS}")
    print("-" * 50)

    # === 训练阶段 ===
    model.train()
    for step in range(NUM_STEPS):
        # 生成一个批次的合成数据
        batch = [make_synthetic_sample(IMAGE_SIZE, NUM_KEYPOINTS, rng)
                 for _ in range(BATCH_SIZE)]
        imgs = torch.from_numpy(np.stack([b[0] for b in batch]))
        hms = torch.from_numpy(np.stack([b[1] for b in batch]))

        # 前向传播：网络输出的热力图可能和目标尺寸不一致
        pred = model(imgs)
        pred = F.interpolate(pred, size=hms.shape[-2:],
                             mode="bilinear", align_corners=False)

        # 损失函数：逐像素 MSE（热力图回归的标准选择）
        loss = F.mse_loss(pred, hms)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  步骤 {step:3d}  |  MSE 损失 {loss.item():.6f}")

    print("-" * 50)
    print("训练完成\n")

    # === 评估阶段 ===
    model.eval()
    with torch.no_grad():
        eval_batch = [make_synthetic_sample(IMAGE_SIZE, NUM_KEYPOINTS, rng)
                      for _ in range(16)]
        imgs = torch.from_numpy(np.stack([b[0] for b in eval_batch]))
        gt_coords = torch.from_numpy(np.stack([b[2] for b in eval_batch]))

        pred = model(imgs)
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE),
                             mode="bilinear", align_corners=False)

        # 整数 argmax 提取
        coords_int, conf_int = heatmap_to_coords(pred)
        # 子像素精度优化
        coords_sub = subpixel_refine(pred, coords_int)

        # 计算平均 L2 误差（像素）
        l2_int = (coords_int - gt_coords).norm(dim=-1).mean().item()
        l2_sub = (coords_sub - gt_coords).norm(dim=-1).mean().item()

        # PCK（Percentage of Correct Keypoints）：距离 < threshold 的比例
        # threshold = 像素，这里用3像素作为阈值
        threshold = 5.0
        error_per_kp = (coords_sub - gt_coords).norm(dim=-1)  # (batch, K)
        pck = (error_per_kp < threshold).float().mean().item() * 100

        print("评估结果:")
        print(f"  整数 argmax  平均 L2 误差: {l2_int:.3f} 像素")
        print(f"  子像素优化  平均 L2 误差: {l2_sub:.3f} 像素")
        if l2_int > 0:
            print(f"  子像素优化提升: {(1 - l2_sub / l2_int) * 100:.1f}%")
        print(f"  PCK@{threshold:.0f}px: {pck:.1f}% （距离 < {threshold:.0f} 像素的关键点比例）")

    # === 可视化一个样本的预测结果 ===
    print("\n预测 vs 真实（第 1 个样本，5 个关键点）:")
    kp_names = ["头部", "左肩", "右肩", "左手", "右手"]
    for k in range(NUM_KEYPOINTS):
        gx, gy = gt_coords[0, k]
        px, py = coords_sub[0, k]
        dist = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)
        status = "OK " if dist < threshold else "MISS"
        print(f"  {kp_names[k]}: 真实=({gx:.0f},{gy:.0f})  "
              f"子像素=({px:.1f},{py:.1f})  距离={dist:.1f}px  [{status}]")

    print(f"\n  总体平均 L2 误差: {l2_sub:.2f} 像素")
    print(f"  PCK@{threshold:.0f}px: {pck:.1f}%")


if __name__ == "__main__":
    train_and_evaluate()
