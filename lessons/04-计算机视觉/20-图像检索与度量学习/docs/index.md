# 图像检索与度量学习

> 像素的欧氏距离没有意义，语义的嵌入距离才有效。

**类型：** 实现课
**语言：** Python
**前置知识：** 第 04 阶段 · 03（CNN 架构演进）、第 04 阶段 · 04（图像分类）
**预计时间：** ~75 分钟
**所处阶段：** Tier 1
**关联课程：** 第 09 阶段 · 11（RAG）— 检索逻辑完全一致：相似度嵌入 + 向量索引

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 Triplet Loss 的前向传播与反向传播，理解锚点-正样本-负样本的几何关系
- [ ] 使用 numpy 计算 AP@K 和 mAP 评估指标，理解平均精确率的设计动机
- [ ] 在真实图像数据集上训练嵌入模型并提取特征向量
- [ ] 使用 FAISS 构建向量索引，实现百万级图像的亚毫秒检索
- [ ] 诊断嵌入空间中常见的"坍缩"问题并提出修复方案

## 1. 问题

想象你是一家电商公司的工程师。产品经理要求你开发一个功能："上传图片，找到相似的商品"。用户拍摄一双运动鞋，系统从 500 万款商品中返回最相似的 10 款。

最直接的办法是什么？把每张图片resize到 224x224，计算所有图片之间的像素级欧氏距离，然后排序。这个方案有什么问题？

第一张图的像素值是 `[0.12, 0.87, 0.45, ...]`，第二张是 `[0.11, 0.88, 0.44, ...]`——两张看起来完全不同的鞋子可能因为光照不同而在像素空间里相距甚远；反过来，两张完全不同构图的纯色图可能恰好有接近的像素值。

像素空间 ≠ 语义空间。

正确的做法：让神经网络学习一种**嵌入（Embedding）**——把每张图映射到一个低维向量，使得**语义相似的图片在向量空间中也接近**。一双耐克 Air Jordan 的嵌入向量，应该比一辆丰田卡罗拉的嵌入向量更接近。

这个映射函数怎么学？如果我们有成对的相似图片（同一双鞋的不同角度），也有不相似的图片（不同的商品类别），我们就可以构造一个监督学习任务：给定三个样本——锚点（Anchor）、正样本（Positive，同类）、负样本（Negative，异类），让网络学习"**正样本离锚点更近，负样本离锚点更远**"。

这就是度量学习的核心思想。

```
没有度量学习：                      有度量学习：
                                    ┌─────────────────────┐
   [图A: 鞋子]  ████████            │  anchor(鞋子):      │
   [图B: 汽车]  ████               │    ─── pos(鞋子):   │
   欧氏距离近！❌                   │         \          │
                                    │          \ neg(车)  │
                                    │           \       │
                                    │            \      │
                                    │             anchor  │
                                    └─────────────────────┘
```

## 2. 概念

### 2.1 嵌入与度量学习直觉

度量学习（Metric Learning）的目标是为数据找到一个**可度量的距离函数**，使得度量结果符合人类语义。

传统分类器输出的是一个类别标签（"这是一只猫"）。度量学习输出的是一个连续向量（嵌入），这个向量的几何结构携带了语义信息。

关键洞察：同样的特征向量，既可以用做分类（最近邻），也可以做检索（向量搜索），还可以做聚类（无监督分组），甚至可以判断两张图的相似程度（计算余弦相似度）。

### 2.2 Triplet Loss 的形式化定义

Triplet Loss 是最经典的度量学习损失函数之一。它接收三个样本：

- **Anchor（锚点）**：参考样本
- **Positive（正样本）**：与 Anchor 属于同一类别
- **Negative（负样本）**：与 Anchor 属于不同类别

定义锚点到正样本的距离为 $d_{ap}$，锚点到负样本的距离为 $d_{an}$，边际 $margin$ 为一个超参数（通常取 0.2~0.5）。

$$
L = \max(0, d_{ap} - d_{an} + margin)
$$

当 $d_{ap} - d_{an} + margin \leq 0$ 时，即正样本已经比负样本近至少 `margin` 的差距，损失为零——当前的嵌入已经足够好，不需要更新。

如果差值大于零，说明正样本不够近或负样本不够远，损失被触发，梯度会把正样本拉近、把负样本推远。

**为什么用 L2 距离而不是直接算余弦相似度？** L2 距离同时编码了方向和幅度信息。在嵌入空间中，方向表示语义，幅度表示置信度。L2 距离更容易训练稳定。

### 2.3 硬样本挖掘（Hard Negative Mining）

随机选取负样本往往太简单——一个随机图片大概率和图片的类别完全不同，网络很容易就把它推远了。**简单的负样本不提供有意义的梯度。**

更有效的策略是选择**难负样本（Hard Negative）**：那些在嵌入空间中与锚点距离很近但实际属于不同类别的图片。这些样本正在"混淆"网络，给它们梯度才能有效提高检索精度。

三种样本策略：

| 策略 | 方法 | 效果 |
|---|---|---|
| 随机采样 | 任意选同类和异类 | 容易收敛到平庸解 |
| 半难样本（Semi-hard） | 负样本比正样本远但仍比 margin 近 | 信息量充足 |
| 难负样本（Hard） | 最接近锚点的负样本 | 训练最有效，计算成本最高 |

推荐做法：**在每个批次内搜索该批次的难负样本**，而非在整个数据集搜索。这样训练效率高，且不会泄露全局信息。

### 2.4 动手验证：Triplet Loss 的数值感受

```python
import numpy as np

np.random.seed(0)

# 模拟三个嵌入向量（每个 128 维）
anchor  = np.random.randn(128)
positive = np.random.randn(128)
negative = np.random.randn(128)

# 计算 L2 距离
d_ap = np.sqrt(np.sum((anchor - positive) ** 2))
d_an = np.sqrt(np.sum((anchor - negative) ** 2))

margin = 0.5
loss = max(0, d_ap - d_an + margin)

print(f"锚点到正样本的距离: {d_ap:.3f}")
print(f"锚点到负样本的距离: {d_an:.3f}")
print(f"Margin: {margin}")
print(f"Triplet Loss: {loss:.3f}")

# 如果 loss > 0，说明需要优化
if loss > 0:
    print("损失被触发！需要拉近正样本，推远负样本。")
else:
    print("损失为零，当前嵌入已满足条件。")
```

```text
锚点到正样本的距离: 15.621
锚点到负样本的距离: 14.893
Margin: 0.5
Triplet Loss: 1.228
损失被触发！需要拉近正样本，推远负样本。
```

在这个随机初始化例子中，锚点到正负样本的距离几乎相同（随机向量是高维球面上的均匀分布，两两距离接近），因此 Loss 很大。训练的目标就是让这些距离分离。

## 3. 从零实现

### 第 1 步：最简版 Triplet Loss

先用 numpy 实现最基础的 Triplet Loss，不涉及任何深度学习框架。

```python
# === 文件头注释 ===
# triplet_loss.py — 从零实现 Triplet Loss
# 依赖：numpy>=1.24
# 对应课程：第 04 阶段 · 20（图像检索与度量学习）

import numpy as np
from typing import Tuple


def euclidean_distance_squared(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """计算两个向量之间的平方 L2 距离。

    Args:
        x: 形状 (d,) 或 (batch, d) 的向量
        y: 形状 (d,) 或 (batch, d) 的向量

    Returns:
        平方 L2 距离，形状 () 或 (batch,)
    """
    diff = x - y
    return np.sum(diff ** 2, axis=-1)


def triplet_loss_numpy(
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    margin: float = 0.5
) -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """计算 Triplet Loss 及其对各输入的梯度。

    前向传播:
        d_ap^2 = ||anchor - positive||^2
        d_an^2 = ||anchor - negative||^2
        loss = max(0, d_ap - d_an + margin)

    反向传播:
        当 loss > 0 时:
            d(d_ap)/d(anchor) = (anchor - positive) / d_ap
            d(d_an)/d(anchor) = -(anchor - negative) / d_an

    Args:
        anchors: 锚点嵌入，形状 (batch, dim)
        positives: 正样本嵌入，形状 (batch, dim)
        negatives: 负样本嵌入，形状 (batch, dim)
        margin: 边际超参数

    Returns:
        loss: 标量损失值
        grad_anchors, grad_positives, grad_negatives: 各输入的梯度
    """
    # 计算距离
    dist_ap = np.sqrt(euclidean_distance_squared(anchors, positives))
    dist_an = np.sqrt(euclidean_distance_squared(anchors, negatives))

    # 三角不等式形式的损失
    losses = np.maximum(0, dist_ap - dist_an + margin)
    loss = np.mean(losses)

    # 反向传播：仅对 loss > 0 的 triplet 计算梯度
    batch_size = anchors.shape[0]
    grad_anchors = np.zeros_like(anchors)
    grad_positives = np.zeros_like(positives)
    grad_negatives = np.zeros_like(negatives)

    # 避免除零
    safe_dist_ap = np.where(dist_ap < 1e-8, 1e-8, dist_ap)
    safe_dist_an = np.where(dist_an < 1e-8, 1e-8, dist_an)

    for i in range(batch_size):
        if losses[i] > 0:
            # 梯度方向：拉近正样本，推远负样本
            grad_anchors[i] += (anchors[i] - positives[i]) / safe_dist_ap[i]
            grad_anchors[i] -= (anchors[i] - negatives[i]) / safe_dist_an[i]
            grad_positives[i] -= (anchors[i] - positives[i]) / safe_dist_ap[i]
            grad_negatives[i] += (anchors[i] - negatives[i]) / safe_dist_an[i]

    # 对 batch 取平均
    grad_anchors /= batch_size
    grad_positives /= batch_size
    grad_negatives /= batch_size

    return float(loss), (grad_anchors, grad_positives, grad_negatives)
```

### 第 2 步：带难负样本挖掘的完整训练循环

现在实现一个完整的训练流程，包含难负样本挖掘。

```python
# hard_triplet_miner.py — 带难负样本挖掘的 Triplet Loss 训练
# 依赖：numpy>=1.24
# 对应课程：第 04 阶段 · 20

import numpy as np
from triplet_loss import euclidean_distance_squared


class HardTripletMiner:
    """在每个批次内寻找难负样本。

    对于批次中的每个锚点，找出距离最远的正样本边界内
    仍然比正样本远的负样本——这些"即将失效"的负样本
    提供最有价值的梯度信号。
    """

    def __init__(self, margin: float = 0.5):
        self.margin = margin

    def mine_hard_triplets(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从批次中挖掘难三元组。

        对于每个样本作为锚点：
        1. 从同类中随机选一个正样本
        2. 在所有异类中找到使 (d_ap - d_an) 最大的负样本

        Args:
            embeddings: 当前嵌入，形状 (N, dim)
            labels: 对应标签，形状 (N,)

        Returns:
            anchor_indices, positive_indices, negative_indices
        """
        n_samples = len(embeddings)
        unique_labels = np.unique(labels)

        anchor_idx = []
        positive_idx = []
        negative_idx = []

        for i in range(n_samples):
            label_i = labels[i]

            # 找同类正样本
            same_mask = labels == label_i
            same_indices = np.where(same_mask)[0]
            same_indices = same_indices[same_indices != i]

            if len(same_indices) == 0:
                continue

            j = np.random.choice(same_indices)
            positive_idx.append(j)

            # 找异类负样本
            diff_mask = labels != label_i
            diff_indices = np.where(diff_mask)[0]

            if len(diff_indices) == 0:
                continue

            # 对所有异类计算距离，选最难的那个
            d_ap = np.sqrt(euclidean_distance_squared(
                embeddings[i], embeddings[j]
            ))

            distances_to_negatives = np.array([
                np.sqrt(euclidean_distance_squared(embeddings[i], embeddings[k]))
                for k in diff_indices
            ])

            # 找使 (d_ap - d_an) 最大的负样本——即距离锚点最近的异类
            hardest_neg_local = np.argmin(distances_to_negatives)
            k = diff_indices[hardest_neg_local]

            anchor_idx.append(i)
            negative_idx.append(k)

        if not anchor_idx:
            # 回退到随机采样
            return (np.arange(n_samples),
                    np.random.randint(0, n_samples, n_samples),
                    np.random.randint(0, n_samples, n_samples))

        return (np.array(anchor_idx),
                np.array(positive_idx),
                np.array(negative_idx))
```

### 第 3 步：AP@K 评估指标

均值平均精确率（mean Average Precision at K）是信息检索的标准评估指标。

```python
# ap_k.py — AP@K 与 mAP 计算
# 依赖：numpy>=1.24
# 对应课程：第 04 阶段 · 20

import numpy as np
from typing import List, Tuple


def precision_at_k(relevant_set: set, retrieved_k: set, k: int) -> float:
    """计算 @K 的精确率。

    返回检索到的 K 个结果中有多少比例是相关的。

    Args:
        relevant_set: 所有相关样本的集合
        retrieved_k: 检索到的前 K 个样本的集合
        k: K 的值

    Returns:
        精确率，范围 [0, 1]
    """
    if k == 0:
        return 0.0
    hits = len(relevant_set & retrieved_k)
    return hits / k


def average_precision_at_k(
    query_relevant: set,
    ranked_list: List[int],
    k: int
) -> float:
    """计算 Query 的 AP@K。

    对排名列表中的每一个位置 i（从 1 开始），如果该位置的样本
    属于相关集合，则在该位置计算 Precision@i，最后对这些值取平均。

    Args:
        query_relevant: 该查询的所有相关样本 ID 集合
        ranked_list: 按相似度降序排列的检索结果 ID 列表
        k: 截断长度

    Returns:
        AP@K 值，范围 [0, 1]
    """
    ranked_k = ranked_list[:k]
    accumulated_hits = 0
    precision_sum = 0.0

    for i, item_id in enumerate(ranked_k, start=1):
        if item_id in query_relevant:
            accumulated_hits += 1
            precision_at_position = accumulated_hits / i
            precision_sum += precision_at_position

    num_relevant_in_k = min(len(query_relevant & set(ranked_k)), k)
    if num_relevant_in_k == 0:
        return 0.0

    return precision_sum / num_relevant_in_k


def mean_average_precision_at_k(
    query_results: List[Tuple[set, List[int]]],
    k: int
) -> float:
    """计算多个查询的平均 mAP@K。

    Args:
        query_results: 每个元素为 (相关ID集合, 排序后的检索列表)
        k: 截断长度

    Returns:
        mAP@K 值
    """
    aps = []
    for relevant_set, ranked_list in query_results:
        ap = average_precision_at_k(relevant_set, ranked_list, k)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0
```

### 第 4 步：端到端嵌入模型训练

将 Triplet Loss 和难负样本挖掘组合成一个端到端的训练流程。

```python
# embedder.py — 端到端图像嵌入训练
# 依赖：torch>=2.0, torchvision>=0.15, numpy>=1.24
# 对应课程：第 04 阶段 · 20

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Tuple, List, Optional


class EmbeddingNet(nn.Module):
    """轻量级嵌入网络。

    输入 RGB 图像，输出固定维度的嵌入向量。
    架构设计要点：
    - 不用全连接分类头，末尾接投影层
    - 输出向量需要 L2 归一化，保证余弦相似度可用
    """

    def __init__(self, input_dim: int = 64, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            # 输入可以是合成的小图或 CIFAR-10 (3x32x32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> 1x1
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        embedding = self.projection(features)
        # L2 归一化：使嵌入落在单位超球面上
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


class TripletDataset(Dataset):
    """Triplet 数据加载器。

    从带标签的数据集中构建 (锚点, 正样本, 负样本) 三元组。
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 image_transform=None):
        self.images = images
        self.labels = labels
        self.transform = image_transform
        self._same_label_cache: dict = {}
        self._diff_label_cache: dict = {}
        self._build_cache()

    def _build_cache(self):
        """预构建同类/异类索引，加速 triplet 采样。"""
        unique_labels = np.unique(self.labels)
        for lbl in unique_labels:
            indices = np.where(self.labels == lbl)[0]
            self._same_label_cache[lbl] = indices

        for lbl in unique_labels:
            other_labels = [l for l in unique_labels if l != lbl]
            all_diff = []
            for o_lbl in other_labels:
                all_diff.extend(self._same_label_cache[o_lbl])
            self._diff_label_cache[lbl] = np.array(all_diff, dtype=int)

    def __len__(self):
        return len(self.images)

    def _sample_triplet_indices(self, anchor_idx: int) -> Tuple[int, int]:
        """为给定锚点采样正负样本索引。"""
        anchor_label = self.labels[anchor_idx]

        # 随机选一个同类正样本
        same_indices = self._same_label_cache[anchor_label]
        positive_idx = np.random.choice(
            same_indices[same_indices != anchor_idx]
        )

        # 随机选一个异类负样本
        diff_indices = self._diff_label_cache[anchor_label]
        negative_idx = np.random.choice(diff_indices)

        return int(positive_idx), int(negative_idx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_img = self.images[idx]

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)

        pos_idx, neg_idx = self._sample_triplet_indices(idx)
        pos_img = self.images[pos_idx]
        neg_img = self.images[neg_idx]

        if self.transform is not None:
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        # 返回三个独立的图像张量
        return (
            torch.from_numpy(np.ascontiguousarray(anchor_img)).permute(2, 0, 1).float(),
            torch.from_numpy(np.ascontiguousarray(pos_img)).permute(2, 0, 1).float(),
            torch.from_numpy(np.ascontiguousarray(neg_img)).permute(2, 0, 1).float(),
        )


def train_embedding_model(
    model: EmbeddingNet,
    train_loader: DataLoader,
    epochs: int = 10,
    margin: float = 0.5,
    learning_rate: float = 0.001,
    device: str = "cpu"
) -> List[float]:
    """训练嵌入模型。

    使用 PyTorch 内置的 Triplet Margin Loss，等价于手动实现
    但拥有高效的 CUDA 后端。

    Args:
        model: 嵌入网络
        train_loader: triplet 数据加载器
        epochs: 训练轮次
        margin: Triplet Margin
        learning_rate: 学习率
        device: 设备

    Returns:
        每轮的 Loss 记录
    """
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for anchor_imgs, pos_imgs, neg_imgs in train_loader:
            anchor_imgs = anchor_imgs.to(device)
            pos_imgs = pos_imgs.to(device)
            neg_imgs = neg_imgs.to(device)

            anchor_emb = model(anchor_imgs)
            pos_emb = model(pos_imgs)
            neg_emb = model(neg_imgs)

            loss = criterion(anchor_emb, pos_emb, neg_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        scheduler.step()

        if epoch % 2 == 0 or epoch == 1:
            print(f"  轮次 {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    return loss_history


def extract_embeddings(
    model: EmbeddingNet,
    images: np.ndarray,
    batch_size: int = 128,
    device: str = "cpu"
) -> np.ndarray:
    """批量提取嵌入向量。

    Args:
        model: 训练好的嵌入网络
        images: 图像数组，形状 (N, H, W, C)
        batch_size: 批次大小
        device: 设备

    Returns:
        嵌入向量，形状 (N, embedding_dim)
    """
    model.eval()
    embeddings_list = []
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(images).permute(0, 3, 1, 2).float()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            emb = model(batch)
            embeddings_list.append(emb.cpu().numpy())

    return np.vstack(embeddings_list)
```

## 4. 工业工具

### 4.1 FAISS 向量搜索

FAISS（Facebook AI Similarity Search）是工业界搜索密集向量集合的库。它可以做到：在你有 1000 万个 128 维向量时，2 毫秒返回 Top-10 相似向量。

```python
# faiss_search.py — FAISS 向量检索流水线
# 依赖：torch>=2.0, torchvision>=0.15, faiss-cpu>=1.7, numpy>=1.24
# 对应课程：第 04 阶段 · 20

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from embedder import EmbeddingNet, extract_embeddings
import faiss


class ImageRetrievalPipeline:
    """图像检索流水线：嵌入提取 + FAISS 索引 + 查询。

    工业级图像检索系统的标准架构：
    1. 预训练嵌入模型（本例中使用自定义 EmbeddingNet）
    2. 批量提取数据库所有图像的嵌入
    3. 构建 FAISS 索引（IVF、HNSW 等）
    4. 接收查询图像，提取嵌入，检索 Top-K
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        index_type: str = "IVF",
        n_clusters: int = 64,
        n_probe: int = 4
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.index: Optional[faiss.Index] = None
        self.db_labels: Optional[List[int]] = None
        self.db_images: Optional[np.ndarray] = None

    def build_index(self, embeddings: np.ndarray, labels: List[int]):
        """用数据库嵌入构建 FAISS 索引。

        Args:
            embeddings: 数据库嵌入，形状 (N, D)
            labels: 每个图像的类别标签
        """
        n_images = embeddings.shape[0]
        self.db_labels = labels
        self.embeddings_db = embeddings.astype(np.float32)

        if self.index_type == "Flat":
            # 暴力搜索，精确但不支持大规模
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(self.embeddings_db)

        elif self.index_type == "IVF":
            # 倒排索引：先聚类再检索
            # 将空间分成 n_clusters 个簇
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.n_clusters,
                faiss.METRIC_L2
            )
            # 在数据库上训练聚类中心
            self.index.train(self.embeddings_db)
            self.index.add(self.embeddings_db)

        elif self.index_type == "HNSW":
            # 分层可导航小世界图——近年最受欢迎的近似检索方法
            # M=16 控制每个节点的连接数，efConstruction 控制建图质量
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M=16)
            self.index.hnsw.efConstruction = 64
            self.index.add(self.embeddings_db)

        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

        print(f"FAISS 索引构建完成: {n_images} 个向量, "
              f"{self.embedding_dim} 维, 类型: {self.index_type}")

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """在 FAISS 索引中检索最近邻。

        Args:
            query_embeddings: 查询嵌入，形状 (Q, D)
            k: 返回 Top-K

        Returns:
            distances: 形状 (Q, K)，与检索结果的 L2 距离
            labels: 形状 (Q, K)，检索结果的标签
        """
        assert self.index is not None, "请先调用 build_index()"

        if self.index_type == "IVF":
            # IVF 需要设置 probe 数量——搜索多少个簇
            self.index.nprobe = self.n_probe

        # ef 参数对 HNSW 控制搜索精度与速度的权衡
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = 64

        distances, indices = self.index.search(query_embeddings, k)

        results = []
        for i in range(indices.shape[0]):
            row = [self.db_labels[idx] if idx != -1 else -1
                   for idx in indices[i]]
            results.append(row)

        return distances, np.array(results)

    def evaluate_retrieval(
        self,
        query_embeddings: np.ndarray,
        query_labels: np.ndarray,
        ground_truth: dict,
        k_values: list = None
    ) -> dict:
        """评估检索性能。

        Args:
            query_embeddings: 查询嵌入
            query_labels: 查询标签
            ground_truth: 标签 -> 同标签 ID 集合 的字典
            k_values: 要评估的 K 值列表

        Returns:
            包含 Recall@K 和 mAP 字典
        """
        if k_values is None:
            k_values = [1, 5, 10, 50]

        distances, indices = self.search(query_embeddings, k=max(k_values))

        metrics = {}
        for k in k_values:
            recall_scores = []
            query_idx = 0

            for q_label in query_labels:
                true_relevant = ground_truth.get(q_label, set())
                # 排除自身
                true_relevant.discard(query_idx)
                retrieved = set(indices[query_idx][:k])

                if len(true_relevant) == 0:
                    recall_scores.append(0.0)
                    continue

                hits = len(true_relevant & retrieved)
                recall_scores.append(hits / len(true_relevant))

                query_idx += 1

            metrics[f"Recall@{k}"] = np.mean(recall_scores)

        return metrics
```

### 4.2 使用预训练模型：ResNet + Inception 嵌入

工业实践中极少使用手写的小型 CNN 做嵌入。以下展示如何用预训练 ResNet-50 和 Inception-V3 提取特征。

```python
# pretrained_embedders.py — 使用预训练模型的嵌入提取
# 依赖：torch>=2.0, torchvision>=0.15, PIL, numpy>=1.24
# 对应课程：第 04 阶段 · 20

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from typing import List


class PretrainedResNetEmbedder:
    """基于预训练 ResNet-50 的嵌入提取器。

    移除最后的分类层（全连接层），使用全局平均池化的输出
    作为图像嵌入。这种做法在 ImageNet 微调后特别有效。
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 移除最后的全连接层
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval().to(device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def embed(self, image: Image.Image) -> np.ndarray:
        """提取单张图像的嵌入向量。"""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(img_tensor)  # (1, 2048, 1, 1)
        embedding = embedding.squeeze(-1).squeeze(-1)  # (1, 2048)
        embedding = embedding.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)  # L2 归一化
        return embedding

    @torch.no_grad()
    def batch_embed(self, images: List[Image.Image]) -> np.ndarray:
        """批量提取嵌入。"""
        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        embedding = self.model(tensors)
        embedding = embedding.squeeze(-1).squeeze(-1).cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding


class InceptionV3Embedder:
    """基于 Inception-V3 的嵌入提取器。

    Inception-V3 提供了更丰富的多尺度特征。
    使用 logits 前的最后池化层输出（2048 维），这是 Google
    Landmarks v2 数据集使用的标准嵌入方式。
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # 移除分类头
        self.model.aux_logits = False  # 禁用辅助分类器简化输出
        self.model.eval().to(device)

        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def embed(self, image: Image.Image) -> np.ndarray:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(img_tensor)  # (1, 2048)
        embedding = embedding.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
```

### 4.3 性能对比

| 嵌入来源 | 维度 | 计算速度 (每张) | 检索精度 (mAP@10) | 适用场景 |
|---|---|---|---|---|
| 手写 CNN（本课） | 128 | ~1 ms | 低 | 教学验证，快速原型 |
| ResNet-50（预训练） | 2048 | ~15 ms | 中高 | 通用图像检索 |
| Inception-V3 | 2048 | ~20 ms | 中 | Google Maps 地标搜索 |
| CLIP ViT-B/32 | 512 | ~30 ms | 高 | 开放词汇检索 |
| DINOSelfQueries | 768 | ~40 ms | 最高 | 学术 SOTA，自监督 |

FAISS 索引类型对速度的影响（以 100 万 128 维向量为基准，Top-10 检索）：

| 索引类型 | 构建时间 | 单查询延迟 | 精度损失 |
|---|---|---|---|
| Flat（暴力） | <1 秒 | ~100 毫秒 | 无（精确） |
| IVF64 | ~30 秒 | ~2 毫秒 | 约 2% |
| HNSW(16) | ~20 秒 | ~1 毫秒 | 约 3% |

## 5. 知识连线

本课学习的度量学习思想在后续阶段有多处重要应用：

- **第 07 阶段（Transformer 深入）**：自注意力中的 Q/K/V 机制本质上也是一种度量学习——查询向量与键向量计算相似度，决定对哪些位置赋予高权重。理解了嵌入空间，就能理解注意力为何有效。
- **第 10 阶段（大语言模型从零）**：RAG（检索增强生成）系统完全依赖向量嵌入来匹配用户问题与知识库文档。Triplet Loss 的思想直接迁移到文本检索中。
- **第 12 阶段（多模态 AI）**：CLIP 模型使用对比学习（InfoNCE Loss）将图像和文本映射到同一个嵌入空间——这正是 Triplet Loss 在多模态场景下的自然扩展。

## 6. 工程最佳实践

### 6.1 嵌入维度的选择

嵌入维度不是一个随意选择的参数。它与下游检索性能和存储成本直接相关：

- **小规模数据集（< 10 万样本）**：128 维通常足够，聚类效果好
- **中等规模（10 万~100 万）**：256~512 维，平衡精度与存储
- **大规模（> 100 万）**：2048 维或更高，此时 FAISS 的近似索引会显著降低延迟
- **量化压缩**：使用 PQ（乘积量化）可以将 2048 维的 float32 向量压缩到 ~4 字节，压缩比超过 50 倍，精度损失通常 < 5%

### 6.2 FAISS 选型指南

| 数据规模 | 推荐索引 | 备注 |
|---|---|---|
| < 10 万 | IndexFlatL2 | 暴力搜索，精确，无需调参 |
| 10 万 ~ 500 万 | IndexIVFFlat | 需要训练聚类中心，nprobe=10~20 平衡精度与速度 |
| 500 万 ~ 1 亿 | IndexHNSWFlat | 无需训练，建图自动，适合在线增量更新 |
| > 1 亿 | IndexIVFPQ | 乘积量化，极致压缩，延迟最低 |

### 6.3 常见踩坑

- **嵌入未归一化**：如果嵌入向量没有 L2 归一化，FAISS 默认的 L2 距离会受到向量幅度影响——某些特征强烈的向量天然拥有更大的 L2 范数，导致检索偏差。务必在使用前 `normalize(embedding, p=2)`。
- **IVF 训练数据偏差**：用少量数据训练的 IVF 聚类中心无法覆盖全部数据空间。至少使用 10% 的训练集用于训练，否则会严重损失召回率。
- **跨域泛化**：在商店商品图片上训练的嵌入模型，放到社交媒体风景照上做检索效果会非常差。嵌入模型对域偏移极其敏感，需要在目标域上微调。

### 6.4 中文场景特别建议

- 如果是电商商品检索，训练数据中英文商品信息可能不足。可以考虑使用 CLIP 的中文微调版本（如 Chinese-CLIP）提取初始嵌入，再用自己的商品数据进行 Triplet Fine-tuning。
- 对于国内主流电商平台（淘宝、京东），商品图片存在统一的水印和背景模板，这些会成为嵌入空间的虚假信号。建议在训练前做数据清洗，去除水印或统一背景。

## 7. 常见错误

### 错误 1：忘记 L2 归一化嵌入

**现象：** 检索结果看起来随机——相同类别的图片有时排在前列，有时完全搜不到。FAISS 的 L2 距离受向量幅度影响严重。

**原因：** Triplet Loss 本身不强制嵌入向量落在单位超球面上。未归一化的嵌入向量中，数值大的向量与数值小的向量之间 L2 距离很大，即使它们的语义方向完全一致。

**修复：**
```python
# ❌ 错误：直接用原始输出
embedding = model(image)  # 可能有任意范数

# ✓ 正确：L2 归一化
embedding = nn.functional.normalize(embedding, p=2, dim=1)
# 或使用 NumPy
embedding = embedding / np.linalg.norm(embedding)
```

### 错误 2：Batch Size 过小导致 Triplet Loss 失效

**现象：** 训练 Loss 不下降，嵌入空间的类间分离度几乎没有改善。

**原因：** Triplet Loss 的效果高度依赖于批次内的样本多样性。如果 Batch Size 太小（例如 8），可能有些类别在批次中没有出现，就无法构造有效的 triplet。此外，难负样本挖掘也依赖批次内有足够的候选负样本。

**修复：**
```python
# ❌ 太小
batch_size = 8  # 可能有 4 个类别，每个 2 张图

# ✓ 足够大，确保每个类别至少有 2-3 个样本
batch_size = 64  # 假设 16 个类别，每类 4 张，可以构造 16×3×(N-16) 个 triplet
```

业界推荐 Batch Size 至少为 **类别数 × 3**，这样每个类别在批次内都有足够的正/负样本对。

### 错误 3：Margin 设置不当

**现象：** Margin 过小时训练迅速收敛但检索精度很低；Margin 过大时训练震荡甚至发散。

**原因：**
- Margin 过小（如 0.1）：正负样本可以靠得非常近才停止优化，嵌入区分度不足。
- Margin 过大（如 5.0）：网络无法在有限嵌入维度内满足条件，Loss 始终不为零，梯度持续存在导致振荡。

**修复：**
```python
# 经验规则：
# - 余弦空间（归一化后）：margin 取 0.2~0.5
# - L2 空间（未归一化）：margin 取 0.5~1.0

# 也可以动态调整：
margin = initial_margin
for epoch in range(num_epochs):
    # 前 50% 轮次用较大 margin 拉开类别
    # 后 50% 轮次用较小 margin 细化
    effective_margin = margin * (1.0 - 0.5 * (epoch / num_epochs))
```

### 错误 4：FAISS IVF 的 nprobe 与 n_clusters 不匹配

**现象：** 检索要么太慢（接近暴力搜索），要么精度骤降（漏掉大量候选）。

**原因：** nprobe 控制搜索多少个簇。如果 n_clusters=1024 但 nprobe=1，相当于只搜索 1/1024 的空间，召回率必然极低。如果 nprobe=n_clusters，就退化成了暴力搜索。

**修复：**
```python
index = faiss.IndexIVFFlat(quantizer, dim, n_clusters=64, metric=faiss.METRIC_L2)
index.nprobe = min(16, n_clusters)  # 通常 nprobe <= n_clusters / 4
```

## 8. 面试考点

### Q1：Triplet Loss 为什么在大批次下效果更好？和 mini-batch 的大小有什么经验公式？（难度：⭐⭐）

**参考答案：**

Triplet Loss 的核心在于每个批次内都要有足够的正负样本对。假设批次大小为 B，类别数为 C，每类样本数为 B/C。对于每个锚点，最多只有 (B/C - 1) 个正样本和 B - B/C 个负样本。如果 C 很大但 B 很小，很多 triplet 就无法构造。

经验公式：批次大小应该满足 $B \geq C \times P$，其中 $P$ 是每个类别所需的正样本数（通常 P=3，即每个锚点至少有 3 个可选正样本）。这就是为什么人脸识别中常用的 Facenet 等模型会在单个 GPU 上使用 B=128~512 的大批次，并且配合多个 GPU 进行分布式训练。

### Q2：为什么 AP@K 要用"中间精确率的平均值"而不是"仅在 @K 处的精确率"？（难度：⭐⭐）

**参考答案：**

AP 的设计动机是评估整个排序质量，而不仅仅是 Top-K 的截断效果。如果一个检索系统在排名 1 的位置放了一个相关文档，排名 5 放了另一个，排名 100 又放了一个——它在 @1 的精确率为 1.0，@5 为 0.4，@100 也很低。只看 @K 会丢失大量排序信息。

AP 通过计算每个相关文档出现位置的 Precision@i 并取平均，能够完整地反映检索系统对排序的利用能力。这也是为什么在信息检索领域，AP 和 NDCG 是比单纯的 Recall@K 更标准的指标。

### Q3：如何在没有标注数据的情况下训练嵌入模型？（难度：⭐⭐⭐）

**参考答案：**

这是一个重要的开放性问题。几种可行方案：

1. **自监督对比学习（SimCLR/MoCo）**：对同一张图做两种不同的数据增强视为正样本对，不同图视为负样本对。完全不需要同类/异类的标签。
2. **伪标签（Pseudo-labeling）**：先用聚类算法（如 K-Means）对未标注数据分组，将同一簇视为"伪正样本"，其他簇视为负样本，然后用 Triplet Loss 训练。
3. **跨模态对齐**：如果有图文配对数据（即使没有类别标签），可以直接让图像的嵌入与对应文本描述（通过 CLIP）的嵌入接近，无需显式的 triplet 标注。

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 嵌入 (Embedding) | "就是把图片变成一个数字" | 将高维数据映射到低维连续向量的过程——向量的几何结构承载语义信息，距离近的向量表示语义相似的对象 |
| 度量学习 (Metric Learning) | "学习如何测量相似度" | 学习一个距离函数（通常是神经网络），使得函数空间中的距离符合人类语义——同类接近、异类远离 |
| Triplet Loss | "三张图片的损失" | 一次传入锚点、正样本、负样本三个向量，用 $\max(0, d_{ap} - d_{an} + margin)$ 约束正样本更近，负样本更远 |
| 边际 (Margin) | "安全距离" | Triplet Loss 中超参数，要求负样本至少比正样本远多少。太小区分度不够，太大训练不稳定 |
| 难负样本挖掘 | "找最难的那一个" | 不是随机采样，而是在批次内主动寻找那些让模型"困惑"的负样本——距离锚点最近的异类样本 |
| AP@K | "前 K 个里面有多少对的" | 平均精确率——不只看在 @K 处对了几个，还看这些相关文档在整个排序列表中的位置分布 |
| mAP | "所有查询的平均 AP" | 将所有查询的 AP 取平均，衡量整个检索系统的综合性能，是电商搜索、学术检索的标准指标 |
| FAISS | "Facebook 做的向量搜索引擎" | Facebook AI 开发的稠密向量相似度搜索库，支持 Flat、IVF、HNSW 等多种索引结构，可在毫秒内完成百万级向量的近似最近邻搜索 |
| 倒排索引 (IVF) | "先分堆再搜索" | 先将向量空间划分成若干簇（通过 K-Means 聚类），搜索时只检索与查询点最近的若干个簇，大幅减少计算量 |
| 乘积量化 (PQ) | "压缩向量" | 将向量切成多个子空间，每个子空间独立量化。可将 2048 维 float32 向量（8KB）压缩到 4 字节，压缩比超过 2000 倍 |

## 📚 小结

度量学习的核心在于：**不比较像素，比较语义**。Triplet Loss 让网络学会把相似图片拉进同一个向量区域，FAISS 让这个向量空间在百万规模下依然能毫秒响应。

下一课我们将把嵌入从单模态扩展到多模态——让文本和图片共享同一个向量空间，这就是 CLIP 和视觉语言模型的基石。

## ✏️ 练习

1. **【理解】** 用自己的话解释为什么 Triplet Loss 的 $margin=0$ 等价于一个退化损失函数。画图展示当 $d_{ap} = d_{an}$ 时 Loss 的行为，以及 $margin$ 引入后的变化。

2. **【实现】** 修改 `triplet_loss_numpy` 函数，使其支持 batch 内的难负样本挖掘——不再随机采样负样本，而是从同一批次中选择距离锚点最近的异类作为负样本。

3. **【实验】** 使用 Fashion-MNIST 数据集（10 类服装），分别用随机嵌入、手工特征（HOG 直方图）和训练好的 Triplet 嵌入三种方式，在 Fashion-MNIST 上测试 Recall@10。对比三者的 mAP@10 差异。

4. **【思考】** FAISS 的 HNSW 索引通过构建多层图来加速最近邻搜索。思考为什么"分层"比单层图更高效？这与"小世界网络"理论的哪条性质有关？

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| Triplet Loss 实现 | `code/triplet_loss.py` | 从零实现的 Triplet Loss 及反向传播 |
| 图像检索流水线 | `code/faiss_search.py` | 完整的嵌入提取 + FAISS 索引 + 检索 + 评估流水线 |
| 嵌入模型 | `code/embedder.py` | 端到端的嵌入网络训练与推理代码 |
| AP@K 评估 | `code/ap_k.py` | 可复用的平均精确率计算工具 |

## 📖 参考资料

1. [论文] Schroff et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering". CVPR, 2015. https://arxiv.org/abs/1503.03832
2. [论文] Hadsell et al. "Dimensionality Reduction by Learning an Invariant Mapping". CVPR, 2006. https://arxiv.org/abs/cs/061016
3. [官方文档] FAISS: https://github.com/facebookresearch/faiss
4. [论文] Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)". ICML, 2020. https://arxiv.org/abs/2002.05709
5. [论文] Radford et al. "Learning Transferable Visual Models From Natural Language Supervision (CLIP)". ICML, 2021. https://arxiv.org/abs/2103.00020

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
