# main.py — 异常检测算法从零实现与对比
# 依赖：numpy>=1.24
# 安装：pip install numpy
# 对应课程：阶段 02 · 16（异常检测）

import numpy as np


# =============================================================================
# 1. 统计方法：Z-score 与 IQR
# =============================================================================

def zscore_detect(X, threshold=3.0):
    """基于 Z-score 的异常检测。

    假设数据服从正态分布，计算每个样本偏离均值的标准差倍数。
    超过阈值的样本被标记为异常。

    Args:
        X: 输入数据，形状 (n_samples, n_features)
        threshold: Z-score 阈值，默认 3.0（99.7% 的数据在范围内）

    Returns:
        labels: 异常标签，形状 (n_samples,)，True 表示异常
        scores: 异常分数，形状 (n_samples,)，取各特征 Z-score 的最大值
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # 防止除零：标准差为 0 的特征视为无信息
    std[std == 0] = 1.0
    z = np.abs((X - mean) / std)
    # 取所有特征中 Z-score 的最大值作为该样本的异常分数
    scores = z.max(axis=1)
    labels = scores > threshold
    return labels, scores


def iqr_detect(X, factor=1.5):
    """基于四分位距（IQR）的异常检测。

    不假设正态分布，使用四分位数定义"正常"范围。
    超出 Q1 - factor*IQR 或 Q3 + factor*IQR 的样本被标记为异常。

    Args:
        X: 输入数据，形状 (n_samples, n_features)
        factor: IQR 倍数，默认 1.5（常规异常），3.0（极端异常）

    Returns:
        labels: 异常标签，形状 (n_samples,)，True 表示异常
        scores: 异常分数，形状 (n_samples,)，表示偏离边界的程度
    """
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    # 计算每个样本偏离边界的程度
    below = (X - lower) / iqr
    above = (X - upper) / iqr
    scores = np.maximum(-np.minimum(below, 0), np.maximum(above, 0)).max(axis=1)
    labels = ((X < lower) | (X > upper)).any(axis=1)
    return labels, scores


# =============================================================================
# 2. 孤立森林（Isolation Forest）
# =============================================================================

def _c_factor(n):
    """计算孤立树的平均路径长度修正因子。

    用于将路径长度归一化到 [0, 1] 区间，使得不同样本量下的分数可参考。
    公式来自 Liu et al. (2008) 的论文推导。
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    # 欧拉-马歇罗尼常数
    euler_mascheroni = 0.5772156649
    h = np.log(n - 1) + euler_mascheroni
    return 2.0 * h - (2.0 * (n - 1.0) / n)


class IsolationTree:
    """单棵孤立树。

    核心思想：随机选择特征和切分值，递归划分数据。
    异常点因为"少且不同"，往往在较浅的深度就被孤立。
    """

    def __init__(self, max_depth=10, rng=None):
        self.max_depth = max_depth
        self.rng = rng if rng is not None else np.random.RandomState()
        self.is_leaf = False
        self.size = 0          # 叶子节点包含的样本数
        self.feature = None    # 当前节点的切分特征索引
        self.threshold = None  # 当前节点的切分值
        self.left = None
        self.right = None

    def fit(self, X, depth=0):
        """递归构建孤立树。"""
        n, p = X.shape

        # 终止条件：达到最大深度或样本数不足
        if depth >= self.max_depth or n <= 1:
            self.is_leaf = True
            self.size = n
            return self

        # 随机选择一个特征
        self.feature = self.rng.randint(p)
        x_col = X[:, self.feature]
        x_min = x_col.min()
        x_max = x_col.max()

        # 如果该特征所有值相同，无法切分
        if x_min == x_max:
            self.is_leaf = True
            self.size = n
            return self

        # 在 [min, max] 之间随机选择切分值
        self.is_leaf = False
        self.threshold = self.rng.uniform(x_min, x_max)

        left_mask = x_col < self.threshold
        right_mask = ~left_mask

        self.left = IsolationTree(self.max_depth, self.rng)
        self.left.fit(X[left_mask], depth + 1)

        self.right = IsolationTree(self.max_depth, self.rng)
        self.right.fit(X[right_mask], depth + 1)

        return self

    def path_length(self, x, depth=0):
        """计算单个样本在树中的路径长度。"""
        if self.is_leaf:
            # 叶子节点：用修正因子估计未完全展开时的平均路径
            return depth + _c_factor(self.size)

        if x[self.feature] < self.threshold:
            return self.left.path_length(x, depth + 1)
        else:
            return self.right.path_length(x, depth + 1)


class IsolationForest:
    """孤立森林：多棵孤立树的集成。

    通过随机采样和随机切分构建多棵树，
    综合所有树的平均路径长度来计算异常分数。
    """

    def __init__(self, n_estimators=100, max_samples=256, seed=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed
        self.trees = []
        self.n_train = 0

    def fit(self, X):
        """训练孤立森林。"""
        self.n_train = X.shape[0]
        rng = np.random.RandomState(self.seed)
        self.trees = []

        # 每棵树的样本量：论文推荐 256（足够检测异常，又不会太大）
        sample_size = min(self.max_samples, X.shape[0])
        # 最大深度限制为 log2(sample_size)，模拟完全二叉树
        max_depth = int(np.ceil(np.log2(sample_size)))

        for _ in range(self.n_estimators):
            # 无放回随机采样
            idx = rng.choice(X.shape[0], size=sample_size, replace=False)
            tree_rng = np.random.RandomState(rng.randint(0, 2**31))
            tree = IsolationTree(max_depth=max_depth, rng=tree_rng)
            tree.fit(X[idx])
            self.trees.append(tree)

        return self

    def anomaly_score(self, X):
        """计算异常分数。

        分数越接近 1 表示越异常，越接近 0 表示越正常。
        公式：s(x, n) = 2^(-E(h(x)) / c(n))
        """
        n = X.shape[0]
        avg_path = np.zeros(n)

        for tree in self.trees:
            for i in range(n):
                avg_path[i] += tree.path_length(X[i])

        avg_path /= self.n_estimators
        sample_size = min(self.max_samples, self.n_train)
        c = _c_factor(sample_size)
        scores = 2.0 ** (-avg_path / c) if c > 0 else np.zeros(n)

        return scores

    def predict(self, X, threshold=0.5):
        """预测异常标签。"""
        scores = self.anomaly_score(X)
        return scores > threshold, scores


# =============================================================================
# 3. 局部异常因子（LOF）
# =============================================================================

def lof_detect(X, k=20, threshold=1.5):
    """基于局部异常因子（LOF）的异常检测。

    核心思想：比较一个点的局部密度与其邻居的局部密度。
    如果该点的密度明显低于邻居，则可能是异常。

    Args:
        X: 输入数据，形状 (n_samples, n_features)
        k: 邻居数量
        threshold: LOF 阈值，> threshold 视为异常

    Returns:
        labels: 异常标签
        scores: LOF 分数
    """
    n = X.shape[0]
    # 计算所有样本对之间的欧氏距离
    dist_matrix = np.sqrt(((X[:, np.newaxis] - X[np.newaxis, :]) ** 2).sum(axis=2))

    # 找到每个样本的 k 近邻（排除自身）
    # 对距离排序，取第 k+1 个（因为包含自身距离 0）
    k_distances = np.partition(dist_matrix, k + 1, axis=1)[:, k + 1]

    # 获取 k 近邻索引
    neighbor_idx = np.argsort(dist_matrix, axis=1)[:, 1:k + 1]

    # 计算局部可达密度（LRD）
    lrd = np.zeros(n)
    for i in range(n):
        # 可达距离 = max(k_distance[邻居], 实际距离)
        reach_dists = np.maximum(k_distances[neighbor_idx[i]], dist_matrix[i, neighbor_idx[i]])
        # LRD = 1 / 平均可达距离
        lrd[i] = 1.0 / (reach_dists.mean() + 1e-10)

    # 计算 LOF 分数
    lof_scores = np.zeros(n)
    for i in range(n):
        # LOF = 邻居 LRD 的平均值 / 自身 LRD
        neighbor_lrd_avg = lrd[neighbor_idx[i]].mean()
        lof_scores[i] = neighbor_lrd_avg / (lrd[i] + 1e-10)

    labels = lof_scores > threshold
    return labels, lof_scores


# =============================================================================
# 4. 自编码器异常检测
# =============================================================================

class SimpleAutoencoder:
    """简单的全连接自编码器，用于异常检测。

    训练目标：仅用正常数据训练，使重构误差最小。
    推理时，异常数据因为与训练分布不同，重构误差会显著增大。
    """

    def __init__(self, input_dim, hidden_dim=8, learning_rate=0.01, seed=42):
        rng = np.random.RandomState(seed)
        # Xavier 初始化
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + input_dim))
        self.W1 = rng.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, input_dim) * scale2
        self.b2 = np.zeros(input_dim)
        self.lr = learning_rate

    def encode(self, X):
        """编码器：将输入压缩到低维表示。"""
        return np.maximum(0, X @ self.W1 + self.b1)  # ReLU 激活

    def decode(self, H):
        """解码器：从低维表示重构输入。"""
        return H @ self.W2 + self.b2

    def forward(self, X):
        """前向传播。"""
        H = self.encode(X)
        X_hat = self.decode(H)
        return X_hat, H

    def train(self, X, epochs=200):
        """训练自编码器。"""
        for epoch in range(epochs):
            # 前向传播
            X_hat, H = self.forward(X)
            loss = np.mean((X - X_hat) ** 2)

            # 反向传播（手动计算梯度）
            d_out = 2.0 * (X_hat - X) / X.shape[0]
            dW2 = H.T @ d_out
            db2 = d_out.sum(axis=0)
            dH = d_out @ self.W2.T
            # ReLU 梯度
            dH[H <= 0] = 0
            dW1 = X.T @ dH
            db1 = dH.sum(axis=0)

            # 参数更新
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch + 1:>4d}, Loss: {loss:.6f}")

    def reconstruction_error(self, X):
        """计算重构误差，作为异常分数。"""
        X_hat, _ = self.forward(X)
        return np.mean((X - X_hat) ** 2, axis=1)


# =============================================================================
# 5. 评估工具
# =============================================================================

def precision_recall(y_true, y_pred):
    """计算精确率、召回率和 F1 分数。"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def precision_at_k(y_true, scores, k):
    """计算 Precision@k：分数最高的 k 个样本中，真正异常的比例。"""
    top_k_idx = np.argsort(scores)[-k:]
    return np.mean(y_true[top_k_idx] == 1)


# =============================================================================
# 6. 数据生成
# =============================================================================

def make_anomaly_data(n_normal=500, n_anomaly=25, n_features=2, seed=42):
    """生成单簇异常检测数据集。"""
    rng = np.random.RandomState(seed)

    center = rng.uniform(-2, 2, n_features)
    cov = np.eye(n_features) * 0.5
    X_normal = rng.multivariate_normal(center, cov, n_normal)

    # 异常点：在远离中心的区域均匀采样
    X_anomaly = rng.uniform(
        X_normal.min(axis=0) - 3,
        X_normal.max(axis=0) + 3,
        (n_anomaly * 3, n_features),
    )
    distances = np.linalg.norm(X_anomaly - center, axis=1)
    far_enough = distances > 3.0
    X_anomaly = X_anomaly[far_enough][:n_anomaly]

    if len(X_anomaly) < n_anomaly:
        extra = rng.uniform(
            center - 6, center + 6,
            (n_anomaly - len(X_anomaly), n_features),
        )
        X_anomaly = np.vstack([X_anomaly, extra]) if len(X_anomaly) > 0 else extra

    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normal + [1] * len(X_anomaly))

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


def make_multimodal_data(n_per_cluster=200, n_anomaly=20, seed=42):
    """生成多簇数据集（用于展示 Z-score 的局限性）。"""
    rng = np.random.RandomState(seed)

    c1 = rng.multivariate_normal([0, 0], [[0.3, 0], [0, 0.3]], n_per_cluster)
    c2 = rng.multivariate_normal([5, 5], [[0.5, 0.1], [0.1, 0.5]], n_per_cluster)
    c3 = rng.multivariate_normal([-3, 4], [[0.4, -0.1], [-0.1, 0.4]], n_per_cluster)

    anomalies = rng.uniform(-6, 8, (n_anomaly, 2))

    X = np.vstack([c1, c2, c3, anomalies])
    y = np.array([0] * (3 * n_per_cluster) + [1] * n_anomaly)

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


# =============================================================================
# 7. 演示
# =============================================================================

def print_separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_zscore():
    print_separator("Z-SCORE 异常检测")

    X, y_true = make_anomaly_data(n_normal=500, n_anomaly=25, seed=42)
    print(f"数据集：{X.shape[0]} 个样本，{(y_true == 1).sum()} 个异常\n")

    print(f"{'阈值':>10} {'精确率':>10} {'召回率':>10} {'F1':>10} {'标记数':>10}")
    print("-" * 52)

    for threshold in [2.0, 2.5, 3.0, 3.5, 4.0]:
        y_pred, _ = zscore_detect(X, threshold=threshold)
        prec, rec, f1 = precision_recall(y_true, y_pred)
        flagged = y_pred.sum()
        print(f"{threshold:>10.1f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {flagged:>10}")


def demo_iqr():
    print_separator("IQR 异常检测")

    X, y_true = make_anomaly_data(n_normal=500, n_anomaly=25, seed=42)
    print(f"数据集：{X.shape[0]} 个样本，{(y_true == 1).sum()} 个异常\n")

    print(f"{'因子':>10} {'精确率':>10} {'召回率':>10} {'F1':>10} {'标记数':>10}")
    print("-" * 52)

    for factor in [1.0, 1.5, 2.0, 2.5, 3.0]:
        y_pred, _ = iqr_detect(X, factor=factor)
        prec, rec, f1 = precision_recall(y_true, y_pred)
        flagged = y_pred.sum()
        print(f"{factor:>10.1f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {flagged:>10}")


def demo_isolation_forest():
    print_separator("孤立森林（从零实现）")

    X, y_true = make_anomaly_data(n_normal=500, n_anomaly=25, seed=42)
    print(f"数据集：{X.shape[0]} 个样本，{(y_true == 1).sum()} 个异常\n")

    iso = IsolationForest(n_estimators=100, max_samples=256, seed=42)
    iso.fit(X)
    scores = iso.anomaly_score(X)

    print("异常分数统计：")
    print(f"  正常点：均值={scores[y_true == 0].mean():.4f}, 标准差={scores[y_true == 0].std():.4f}")
    print(f"  异常点：均值={scores[y_true == 1].mean():.4f}, 标准差={scores[y_true == 1].std():.4f}\n")

    print(f"{'阈值':>10} {'精确率':>10} {'召回率':>10} {'F1':>10} {'标记数':>10}")
    print("-" * 52)

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        y_pred = scores > threshold
        prec, rec, f1 = precision_recall(y_true, y_pred)
        flagged = y_pred.sum()
        print(f"{threshold:>10.2f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {flagged:>10}")

    n_anomalies = y_true.sum()
    pak = precision_at_k(y_true, scores, n_anomalies)
    print(f"\nPrecision@{n_anomalies}: {pak:.4f}")


def demo_lof():
    print_separator("局部异常因子（LOF）")

    X, y_true = make_anomaly_data(n_normal=500, n_anomaly=25, seed=42)
    print(f"数据集：{X.shape[0]} 个样本，{(y_true == 1).sum()} 个异常\n")

    labels, scores = lof_detect(X, k=20, threshold=1.5)
    prec, rec, f1 = precision_recall(y_true, labels)
    n_anomalies = int(y_true.sum())
    pak = precision_at_k(y_true, scores, n_anomalies)

    print(f"LOF 检测结果（k=20, 阈值=1.5）：")
    print(f"  精确率: {prec:.4f}, 召回率: {rec:.4f}, F1: {f1:.4f}")
    print(f"  Precision@{n_anomalies}: {pak:.4f}")


def demo_autoencoder():
    print_separator("自编码器异常检测")

    X, y_true = make_anomaly_data(n_normal=500, n_anomaly=25, n_features=5, seed=42)
    print(f"数据集：{X.shape[0]} 个样本，{X.shape[1]} 维特征，{(y_true == 1).sum()} 个异常\n")

    # 仅用正常数据训练自编码器
    X_normal = X[y_true == 0]
    ae = SimpleAutoencoder(input_dim=5, hidden_dim=8, learning_rate=0.01, seed=42)
    print("训练过程（仅使用正常数据）：")
    ae.train(X_normal, epochs=200)

    # 计算重构误差
    errors = ae.reconstruction_error(X)
    print(f"\n重构误差统计：")
    print(f"  正常点：均值={errors[y_true == 0].mean():.4f}, 标准差={errors[y_true == 0].std():.4f}")
    print(f"  异常点：均值={errors[y_true == 1].mean():.4f}, 标准差={errors[y_true == 1].std():.4f}")

    n_anomalies = int(y_true.sum())
    pak = precision_at_k(y_true, errors, n_anomalies)
    print(f"\nPrecision@{n_anomalies}: {pak:.4f}")


def demo_comparison():
    print_separator("方法对比")

    X, y_true = make_anomaly_data(n_normal=500, n_anomaly=25, seed=42)
    n_anomalies = int(y_true.sum())
    print(f"数据集：{X.shape[0]} 个样本，{n_anomalies} 个异常\n")

    _, z_scores = zscore_detect(X, threshold=3.0)
    _, iqr_scores = iqr_detect(X, factor=1.5)

    iso = IsolationForest(n_estimators=100, max_samples=256, seed=42)
    iso.fit(X)
    iso_scores = iso.anomaly_score(X)

    _, lof_scores = lof_detect(X, k=20, threshold=1.5)

    print(f"Precision@{n_anomalies}（按异常分数排序的 top-k）：")
    print(f"  Z-score:          {precision_at_k(y_true, z_scores, n_anomalies):.4f}")
    print(f"  IQR:              {precision_at_k(y_true, iqr_scores, n_anomalies):.4f}")
    print(f"  孤立森林:         {precision_at_k(y_true, iso_scores, n_anomalies):.4f}")
    print(f"  LOF:              {precision_at_k(y_true, lof_scores, n_anomalies):.4f}")

    print()
    z_pred, _ = zscore_detect(X, threshold=3.0)
    iqr_pred, _ = iqr_detect(X, factor=1.5)
    iso_pred = iso_scores > 0.6
    lof_pred = lof_scores > 1.5

    print(f"{'方法':<20} {'精确率':>10} {'召回率':>10} {'F1':>10}")
    print("-" * 52)

    for name, pred in [
        ("Z-score (t=3.0)", z_pred),
        ("IQR (f=1.5)", iqr_pred),
        ("孤立森林 (t=0.6)", iso_pred),
        ("LOF (t=1.5)", lof_pred),
    ]:
        prec, rec, f1 = precision_recall(y_true, pred)
        print(f"{name:<20} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")


def demo_multimodal():
    print_separator("多簇数据（简单方法的局限）")

    X, y_true = make_multimodal_data(n_per_cluster=200, n_anomaly=20, seed=42)
    n_anomalies = int(y_true.sum())
    print(f"数据集：{X.shape[0]} 个样本，{n_anomalies} 个异常，3 个簇\n")

    z_pred, z_scores = zscore_detect(X, threshold=3.0)
    iqr_pred, iqr_scores = iqr_detect(X, factor=1.5)

    iso = IsolationForest(n_estimators=100, max_samples=256, seed=42)
    iso.fit(X)
    iso_scores = iso.anomaly_score(X)
    iso_pred = iso_scores > 0.6

    print(f"{'方法':<20} {'精确率':>10} {'召回率':>10} {'F1':>10} {'P@k':>10}")
    print("-" * 62)

    for name, pred, scores in [
        ("Z-score (t=3.0)", z_pred, z_scores),
        ("IQR (f=1.5)", iqr_pred, iqr_scores),
        ("孤立森林 (t=0.6)", iso_pred, iso_scores),
    ]:
        prec, rec, f1 = precision_recall(y_true, pred)
        pak = precision_at_k(y_true, scores, n_anomalies)
        print(f"{name:<20} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {pak:>10.4f}")

    print()
    print("Z-score 在多簇数据上表现差（簇间点在每个特征上看起来正常，")
    print("但在联合空间中实际是异常）。孤立森林天然支持多簇场景。")


if __name__ == "__main__":
    demo_zscore()
    demo_iqr()
    demo_isolation_forest()
    demo_lof()
    demo_autoencoder()
    demo_comparison()
    demo_multimodal()
