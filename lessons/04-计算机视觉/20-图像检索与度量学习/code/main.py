# main.py — 图像检索与度量学习完整实现
# 依赖：torch>=2.0, torchvision>=0.15, numpy>=1.24, faiss-cpu>=1.7
# 安装：pip install torch torchvision numpy faiss-cpu
# 对应课程：第 04 阶段 · 20（图像检索与度量学习）
#
# 本文件实现完整的度量学习流水线：
#   Triplet Loss 训练 -> 嵌入提取 -> FAISS 索引构建 -> 检索评估

import argparse
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# ====================================================================
# 常量定义
# ====================================================================

TRAIN_EMBEDDING_DIM = 128
VAL_EMBEDDING_DIM = 128
TRAIN_MARGIN = 0.5
SEARCH_K_VALUES = [1, 5, 10, 20]

# Fashion-MNIST 标签（用于中文展示）
FASHION_MNIST_NAMES = [
    "T恤/上衣", "裤子", "套头衫", "连衣裙", "外套",
    "凉鞋", "衬衫", "运动鞋", "包", "短靴",
]


# ====================================================================
# 第 1 步：Triplet Loss 从零实现 (NumPy)
# ====================================================================

def euclidean_distance_squared(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """计算两个向量之间的平方 L2 距离。

    Args:
        x: 形状 (d,) 或 (batch, d)
        y: 形状 (d,) 或 (batch, d)

    Returns:
        平方 L2 距离，形状 () 或 (batch,)
    """
    diff = x - y
    return np.sum(diff ** 2, axis=-1)


def triplet_loss_forward(
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    margin: float = TRAIN_MARGIN
) -> Tuple[float, np.ndarray, np.ndarray]:
    """计算 Triplet Loss 前向传播。

    Args:
        anchors: (B, D) 锚点嵌入
        positives: (B, D) 正样本嵌入
        negatives: (B, D) 负样本嵌入
        margin: 边际值

    Returns:
        loss: 标量损失
        dist_ap: 每对三元组的 (锚点, 正样本) 距离
        dist_an: 每对三元组的 (锚点, 负样本) 距离
    """
    dist_ap = np.sqrt(euclidean_distance_squared(anchors, positives))
    dist_an = np.sqrt(euclidean_distance_squared(anchors, negatives))

    losses = np.maximum(0, dist_ap - dist_an + margin)
    loss = float(np.mean(losses))

    return loss, dist_ap, dist_an


def triplet_loss_gradient(
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    margin: float = TRAIN_MARGIN
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 Triplet Loss 对各输入的梯度。

    梯度逻辑：当 loss > 0 时，
    - 把正样本拉近 (anchor - positive)
    - 把负样本推远 (anchor - negative) 的反方向

    Returns:
        grad_anchors, grad_positives, grad_negatives
    """
    batch_size = anchors.shape[0]
    dist_ap = np.sqrt(euclidean_distance_squared(anchors, positives))
    dist_an = np.sqrt(euclidean_distance_squared(anchors, negatives))

    safe_dist_ap = np.where(dist_ap < 1e-8, 1e-8, dist_ap)
    safe_dist_an = np.where(dist_an < 1e-8, 1e-8, dist_an)

    # 找哪些 triplet 触发了损失
    raw_losses = dist_ap - dist_an + margin
    active_mask = (raw_losses > 0).astype(float)  # (B,)

    grad_anchors = np.zeros_like(anchors)
    grad_positives = np.zeros_like(positives)
    grad_negatives = np.zeros_like(negatives)

    for i in range(batch_size):
        if active_mask[i] > 0:
            diff_ap = anchors[i] - positives[i]
            diff_an = anchors[i] - negatives[i]

            grad_anchors[i] += diff_ap / safe_dist_ap[i]
            grad_anchors[i] -= diff_an / safe_dist_an[i]

            grad_positives[i] -= diff_ap / safe_dist_ap[i]
            grad_negatives[i] += diff_an / safe_dist_an[i]

    # 按 batch 平均
    eps = 1e-8
    grad_anchors /= batch_size + eps
    grad_positives /= batch_size + eps
    grad_negatives /= batch_size + eps

    return grad_anchors, grad_positives, grad_negatives


def demo_triplet_loss_numpy():
    """演示 NumPy 版 Triplet Loss 的数值行为。"""
    print("=" * 60)
    print("第 1 部分：NumPy 版 Triplet Loss 演示")
    print("=" * 60)

    np.random.seed(42)
    dim = 128
    batch_size = 32

    # 随机初始化嵌入
    anchors = np.random.randn(batch_size, dim).astype(np.float32)
    positives = np.random.randn(batch_size, dim).astype(np.float32)
    negatives = np.random.randn(batch_size, dim).astype(np.float32)

    loss, d_ap, d_an = triplet_loss_forward(anchors, positives, negatives)
    ga, gp, gn = triplet_loss_gradient(anchors, positives, negatives)

    print(f"  Batch size: {batch_size}, Embedding dim: {dim}")
    print(f"  初始 Loss: {loss:.4f}")
    print(f"  平均 d(ap): {d_ap.mean():.3f}")
    print(f"  平均 d(an): {d_an.mean():.3f}")
    print(f"  触发损失的 triplet 比例: {(d_ap - d_an + TRAIN_MARGIN > 0).mean():.1%}")

    print(f"\n  梯度幅度:")
    print(f"    |grad_anchor|: {np.linalg.norm(ga):.3f}")
    print(f"    |grad_positive|: {np.linalg.norm(gp):.3f}")
    print(f"    |grad_negative|: {np.linalg.norm(gn):.3f}")


# ====================================================================
# 第 2 步：AP@K 评估指标
# ====================================================================

def precision_at_k(relevant_set: set, retrieved_k: set, k: int) -> float:
    """计算 @K 精确率。"""
    if k == 0:
        return 0.0
    hits = len(relevant_set & retrieved_k)
    return hits / k


def average_precision_at_k(
    query_relevant: set, ranked_list: List[int], k: int
) -> float:
    """计算单个查询的 AP@K。"""
    ranked_k = ranked_list[:k]
    accumulated_hits = 0
    precision_sum = 0.0

    for i, item_id in enumerate(ranked_k, start=1):
        if item_id in query_relevant:
            accumulated_hits += 1
            precision_sum += accumulated_hits / i

    num_relevant_found = min(len(query_relevant & set(ranked_k)), k)
    if num_relevant_found == 0:
        return 0.0

    return precision_sum / num_relevant_found


def mean_average_precision_at_k(
    query_results: List[Tuple[set, List[int]]], k: int
) -> float:
    """计算多个查询的平均 mAP@K。"""
    aps = []
    for relevant_set, ranked_list in query_results:
        ap = average_precision_at_k(relevant_set, ranked_list, k)
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def demo_ap_k():
    """演示 AP@K 的计算过程。"""
    print("\n" + "=" * 60)
    print("第 2 部分：AP@K 评估指标演示")
    print("=" * 60)

    # 假设相关文档为 [1, 3, 5, 7, 9]
    relevant = {1, 3, 5, 7, 9}
    ranked = [3, 8, 1, 4, 5, 0, 7, 6, 2, 9]

    print(f"  相关文档集合: {sorted(relevant)}")
    print(f"  检索排名列表: {ranked}")
    print()
    print(f"  {'位置':>4} {'ID':>4} {'是否相关':>8} {'Precision@i':>12}")
    print(f"  {'-'*4} {'-'*4} {'-'*8} {'-'*12}")

    acc_hits = 0
    for i, item in enumerate(ranked, start=1):
        is_rel = item in relevant
        if is_rel:
            acc_hits += 1
        p_i = acc_hits / i
        marker = "[!]" if is_rel else "   "
        print(f"  {marker}{i:>4} {item:>4} {'是' if is_rel else '否':>8} {p_i:>12.3f}")

    k = 5
    ap_at_5 = average_precision_at_k(relevant, ranked, k)
    ap_at_10 = average_precision_at_k(relevant, ranked, 10)

    print(f"\n  AP@{k}: {ap_at_5:.3f}")
    print(f"  AP@10: {ap_at_10:.3f}")
    print(f"  Precision@5: {precision_at_k(relevant, set(ranked[:5]), 5):.3f}")


# ====================================================================
# 第 3 步：嵌入网络
# ====================================================================

class EmbeddingNet(nn.Module):
    """轻量级图像嵌入网络。

    将图像映射到 TRAIN_EMBEDDING_DIM 维向量空间。
    输出自动 L2 归一化，使嵌入落在单位超球面上。
    """

    def __init__(self, input_channels: int = 3, embedding_dim: int = TRAIN_EMBEDDING_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 8x8 -> 1x1
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        features = self.features(x)
        embedding = self.projection(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


# ====================================================================
# 第 4 步：数据增强与 Triplet Dataset
# ====================================================================

def get_transforms():
    """获取训练和验证的数据增强策略。

    训练时用随机裁剪和翻转增加数据多样性，
    验证时只做标准化确保公平比较。
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(24, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2861], std=[0.3530]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2861], std=[0.3530]),
    ])

    return train_transform, val_transform


class TripletFashionDataset(Dataset):
    """从 Fashion-MNIST 构建 Triplet 训练集。

    对于每个图片作为锚点：
    1. 随机选取同类图片作为正样本
    2. 随机选取异类图片作为负样本
    """

    def __init__(self, images, labels, transform=None, seed=42):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.rng = np.random.default_rng(seed)
        self._same_cache = {}
        self._diff_cache = {}
        self._build_caches()

    def _build_caches(self):
        """预构建同类和异类索引，加速采样。"""
        for label in range(10):
            indices = np.where(self.labels == label)[0]
            self._same_cache[label] = indices
        for label in range(10):
            other = np.concatenate([
                idx for lbl, idx in self._same_cache.items() if lbl != label
            ])
            self._diff_cache[label] = other

    def __len__(self):
        return len(self.images)

    def _sample_triplet(self, anchor_idx):
        """采样 (正样本, 负样本) 索引。"""
        anchor_label = self.labels[anchor_idx]
        same = self._same_cache[anchor_label]
        same_others = same[same != anchor_idx]
        pos_idx = self.rng.choice(same_others) if len(same_others) > 0 else same[0]
        neg_idx = self.rng.choice(self._diff_cache[anchor_label])
        return int(pos_idx), int(neg_idx)

    def __getitem__(self, idx):
        anchor = self.images[idx]
        if self.transform:
            anchor = self.transform(anchor)

        pos_idx, neg_idx = self._sample_triplet(idx)

        pos = self.images[pos_idx]
        neg = self.images[neg_idx]

        if self.transform:
            pos = self.transform(pos)
            neg = self.transform(neg)

        return anchor, pos, neg


# ====================================================================
# 第 5 步：训练流程
# ====================================================================

def train_epoch(model, loader, margin, device):
    """执行一个训练轮次。"""
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

    model.train()
    total_loss = 0.0
    n_batches = 0

    for anchor_imgs, pos_imgs, neg_imgs in loader:
        a = anchor_imgs.to(device)
        p = pos_imgs.to(device)
        n = neg_imgs.to(device)

        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)

        loss = criterion(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        scheduler.step()

    return total_loss / max(n_batches, 1)


def train_model(model, train_loader, epochs, margin, device, dataset_name):
    """完整训练循环。"""
    class_name = "Fashion-MNIST" if dataset_name == "fashion-mnist" else "CIFAR-10"
    print(f"\n在 {class_name} 上训练嵌入模型")
    print(f"  设备: {device}")
    print(f"  嵌入维度: {TRAIN_EMBEDDING_DIM}")
    print(f"  Margin: {margin}")
    print(f"  Epochs: {epochs}\n")

    losses = []
    for epoch in range(1, epochs + 1):
        avg_loss = train_epoch(model, train_loader, margin, device)
        losses.append(avg_loss)

        if epoch % 2 == 0 or epoch == 1:
            print(f"  轮次 [{epoch}/{epochs}]  Loss: {avg_loss:.4f}")

    return losses


# ====================================================================
# 第 6 步：嵌入提取
# ====================================================================

def extract_all_embeddings(model, images, labels, batch_size=256, device="cpu"):
    """从图像数组中提取全部嵌入向量。

    Args:
        model: 训练好的嵌入网络
        images: numpy 数组 (N, H, W, C)，值域 [0, 1]
        labels: 标签数组 (N,)
        batch_size: 批次大小
        device: 计算设备

    Returns:
        embeddings: (N, D) 嵌入矩阵
        unique_labels: 去重后的标签
    """
    model.eval()
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(images).permute(0, 3, 1, 2).float()
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    all_embs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            emb = model(batch)
            all_embs.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embs).astype(np.float32)

    # 构建标签到同类 ID 的映射
    label_to_ids = {}
    for i, lbl in enumerate(labels):
        lbl = int(lbl)
        if lbl not in label_to_ids:
            label_to_ids[lbl] = []
        label_to_ids[lbl].append(i)

    unique_labels = sorted(label_to_ids.keys())

    return embeddings, unique_labels, label_to_ids


# ====================================================================
# 第 7 步：FAISS 检索（不依赖 FAISS 时的回退实现）
# ====================================================================

class FAISSError(Exception):
    """FAISS 不可用时的异常。"""
    pass


def build_faiss_index(embeddings, index_type="Flat"):
    """构建 FAISS 向量索引。

    如果 faiss-cpu 未安装，使用纯 NumPy 回退实现。
    """
    try:
        import faiss
        return _build_faiss_index_with_lib(embeddings, index_type)
    except ImportError:
        print("  提示：faiss-cpu 未安装，使用 NumPy 回退实现（仅适用于小数据集）")
        return _build_numpy_index(embeddings)


def _build_faiss_index_with_lib(embeddings, index_type="Flat"):
    """使用 faiss Python 库构建索引。"""
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]

    if index_type == "Flat":
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
    elif index_type == "IVF":
        n_clusters = min(64, max(4, embeddings.shape[0] // 100))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_L2)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(8, n_clusters)
    elif index_type == "HNSW":
        index = faiss.IndexHNSWFlat(dim, M=16)
        index.hnsw.efConstruction = 64
        index.add(embeddings)
        index.hnsw.efSearch = 64
    else:
        raise ValueError(f"不支持的索引类型: {index_type}")

    return index


def _build_numpy_index(embeddings):
    """纯 NumPy 近似——直接返回嵌入矩阵本身。"""
    return embeddings


def search_nearest(embeddings_db, query_embedding, k=10, index=None):
    """在数据库中搜索最近的 K 个邻居。

    如果提供了 FAISS 索引则使用它，否则回退到 NumPy 暴力搜索。
    """
    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
    db_size = embeddings_db.shape[0]

    if index is not None and hasattr(index, 'search'):
        # 使用 FAISS
        distances, indices = index.search(query_embedding, min(k, db_size))
        return distances[0], indices[0]
    else:
        # NumPy 暴力搜索
        distances = np.sqrt(np.sum((embeddings_db - query_embedding) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:k]
        return distances[nearest_indices], nearest_indices


def evaluate_with_retrieval(
    embeddings, unique_labels, label_to_ids, query_indices, k_values=None
):
    """评估检索性能。"""
    if k_values is None:
        k_values = [1, 5, 10]

    results = {}
    for k in k_values:
        recall_scores = []
        ap_scores = []

        for qi in query_indices:
            q_label = int(unique_labels[qi])
            true_relevant = set(label_to_ids[q_label]) - {int(qi)}

            if len(true_relevant) == 0:
                recall_scores.append(0.0)
                continue

            # 暴力搜索最近邻
            dists, ids = search_nearest(
                embeddings, embeddings[qi], k=max(k, len(true_relevant))
            )

            top_k_ids = set(ids[:k])
            retrieved = list(ids[:k])

            # Recall@K
            hits = len(true_relevant & top_k_ids)
            recall_scores.append(hits / len(true_relevant))

            # AP@K
            ap = average_precision_at_k(true_relevant, retrieved, k)
            ap_scores.append(ap)

        results[f"Recall@{k}"] = np.mean(recall_scores)
        results[f"mAP@{k}"] = np.mean(ap_scores)

    return results


# ====================================================================
# 主程序
# ====================================================================

def run_all(dataset_name="fashion-mnist", epochs=10):
    """运行完整的度量学习演示。

    1. 加载数据
    2. 训练嵌入模型
    3. 提取所有图像的嵌入
    4. 构建 FAISS 索引
    5. 评估检索性能
    """
    print("=" * 60)
    print(f"图像检索与度量学习：端到端演示 ({dataset_name})")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}\n")

    # ---------------------------
    # 1. 加载数据
    # ---------------------------
    print("Step 1: 加载数据...")

    if dataset_name == "fashion-mnist":
        from torchvision import datasets as tv_datasets
        from torchvision import transforms as tv_transforms

        # 训练集：带增强
        train_tf = tv_transforms.Compose([
            tv_transforms.RandomCrop(24, padding=4),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize((0.2861,), (0.3530,)),
        ])
        train_ds = tv_datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=train_tf
        )

        # 验证集：无增强
        val_tf = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize((0.2861,), (0.3530,)),
        ])
        val_ds = tv_datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=val_tf
        )

        # 使用裁剪后的大小
        img_h, img_w = 24, 24

    elif dataset_name == "cifar-10":
        train_tf = tv_transforms.Compose([
            tv_transforms.RandomCrop(24, padding=4),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616)),
        ])
        train_ds = tv_datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_tf
        )

        val_tf = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616)),
        ])
        val_ds = tv_datasets.CIFAR10(
            root="./data", train=False, download=True, transform=val_tf
        )

        img_h, img_w = 24, 24

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 限制数据规模以加速演示
    max_samples = 5000
    train_sampler = torch.utils.data.SubsetRandomSampler(
        np.random.choice(len(train_ds), min(max_samples, len(train_ds)), replace=False)
    )
    val_sampler = torch.utils.data.SubsetRandomSampler(
        np.random.choice(len(val_ds), min(1000, len(val_ds)), replace=False)
    )

    train_loader = DataLoader(train_ds, batch_size=128, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=256, sampler=val_sampler)

    # 取出验证集的原始图像（不带增强）用于检索测试
    val_images_raw, val_labels = [], []
    for img, lbl in val_ds:
        val_images_raw.append(img.numpy().transpose(1, 2, 0))
        val_labels.append(int(lbl))
    val_images_raw = np.array(val_images_raw)
    val_labels = np.array(val_labels)

    print(f"  训练集: {len(train_ds)} 样本 (采样 {max_samples})")
    print(f"  验证集: {len(val_ds)} 样本 (采样 {min(1000, len(val_ds))})")

    # ---------------------------
    # 2. 训练嵌入模型
    # ---------------------------
    print("\nStep 2: 训练嵌入模型...")
    model = EmbeddingNet(
        input_channels=1 if dataset_name == "fashion-mnist" else 3,
        embedding_dim=TRAIN_EMBEDDING_DIM,
    ).to(device)

    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 构建 triplet 数据加载器
    triplet_train_ds = TripletFashionDataset(
        images=np.array([img.numpy().transpose(1, 2, 0) for img, _ in train_ds]),
        labels=np.array([lbl for _, lbl in train_ds]),
        transform=tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize((0.2861,) if dataset_name == "fashion-mnist"
                                     else (0.4914, 0.4822, 0.4465),
                                    (0.3530,) if dataset_name == "fashion-mnist"
                                                else (0.2470, 0.2435, 0.2616)),
        ]),
    )
    triplet_loader = DataLoader(triplet_train_ds, batch_size=64, shuffle=True,
                                num_workers=0)

    loss_history = train_model(
        model, triplet_loader, epochs, TRAIN_MARGIN, device, dataset_name
    )

    # ---------------------------
    # 3. 提取嵌入向量
    # ---------------------------
    print("\nStep 3: 提取嵌入向量...")
    embeddings, unique_labels, label_to_ids = extract_all_embeddings(
        model, val_images_raw, val_labels, batch_size=256, device=device
    )
    print(f"  嵌入矩阵形状: {embeddings.shape}")
    print(f"  类别数: {len(unique_labels)}")

    # 验证 L2 归一化
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  嵌入范数范围: [{norms.min():.4f}, {norms.max():.4f}]")

    # ---------------------------
    # 4. 构建 FAISS 索引
    # ---------------------------
    print("\nStep 4: 构建 FAISS 索引...")

    index = build_faiss_index(embeddings, index_type="Flat")
    n_probe = 1  # Flat 不需要 probe
    print(f"  索引类型: Flat（精确暴力搜索）")
    if hasattr(index, 'ntotal'):
        print(f"  索引中的向量数: {index.ntotal}")

    # ---------------------------
    # 5. 检索与评估
    # ---------------------------
    print("\nStep 5: 执行检索并评估...")

    # 使用每个类别的前 50 个样本作为查询
    query_indices = []
    for label in unique_labels:
        candidates = label_to_ids[label]
        selected = candidates[:50]
        query_indices.extend(selected)

    query_indices = np.array(query_indices[:200])  # 最多 200 个查询
    print(f"  查询数量: {len(query_indices)}")

    retrieval_results = evaluate_with_retrieval(
        embeddings, unique_labels, label_to_ids, query_indices, k_values=SEARCH_K_VALUES
    )

    print("\n  检索结果:")
    print(f"  {'指标':>12} {'值':>10}")
    print(f"  {'-'*12} {'-'*10}")
    for metric_name, value in retrieval_results.items():
        print(f"  {metric_name:>12} {value:>10.4f}")

    # ---------------------------
    # 6. 展示样例检索结果
    # ---------------------------
    print("\nStep 6: 展示样例检索结果...")
    demo_retrieval(model, val_images_raw, val_labels, embeddings, unique_labels,
                   label_to_ids, index, device)


def demo_retrieval(model, images, labels, embeddings, unique_labels,
                   label_to_ids, index, device):
    """展示具体的检索样例。"""
    # 选第一个类别的前 5 张图作为查询
    first_label = int(unique_labels[0])
    query_candidates = label_to_ids[first_label][:5]

    print(f"\n  查询类别: {FASHION_MNIST_NAMES[first_label]} (Label={first_label})")
    print(f"\n  {'查询图':>6} {'检索 Top-5':>40} {'匹配度':>10}")
    print(f"  {'-'*6} {'-'*40} {'-'*10}")

    for qi in query_candidates:
        q_emb = embeddings[qi:qi+1]
        dists, ids = search_nearest(embeddings, q_emb[0], k=5, index=index)

        top5_names = ", ".join(
            FASHION_MNIST_NAMES[int(unique_labels[id])]
            if int(unique_labels[id]) < len(FASHION_MNIST_NAMES)
            else f"?{id}"
            for id in ids
        )
        dist_str = " ".join(f"{d:.2f}" for d in dists)

        print(f"  图 {qi:>3}: {top5_names}")
        print(f"        距离: {dist_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图像检索与度量学习")
    parser.add_argument("--dataset", default="fashion-mnist",
                        choices=["fashion-mnist", "cifar-10"],
                        help="使用的数据集")
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮次（默认 10）")
    args = parser.parse_args()

    # 先运行概念演示
    demo_triplet_loss_numpy()
    demo_ap_k()

    # 再运行端到端训练
    run_all(dataset_name=args.dataset, epochs=args.epochs)
