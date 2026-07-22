# main.py — 从零实现 KNN 与距离度量
# 依赖：无（纯 Python 标准库）
# 对应课程：阶段 02 · 06（KNN 与距离）

import math
import random
import time


# ============================================================
# 第 1 步：距离函数
# ============================================================

def l2_distance(a, b):
    """欧几里得距离（L2）——直线距离，最常用的默认度量。"""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def l1_distance(a, b):
    """曼哈顿距离（L1）——各维度绝对差之和，对异常值更鲁棒。"""
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


def cosine_distance(a, b):
    """余弦距离——衡量向量方向的差异，忽略大小。适用于文本和嵌入向量。"""
    dot_val = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai ** 2 for ai in a))
    norm_b = math.sqrt(sum(bi ** 2 for bi in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot_val / (norm_a * norm_b)


def minkowski_distance(a, b, p=2):
    """闵可夫斯基距离——L1 和 L2 的推广。

    p=1 → 曼哈顿距离
    p=2 → 欧几里得距离
    p→∞ → 切比雪夫距离（最大分量差）
    """
    if p == float("inf"):
        return max(abs(ai - bi) for ai, bi in zip(a, b))
    return sum(abs(ai - bi) ** p for ai, bi in zip(a, b)) ** (1 / p)


def hamming_distance(a, b):
    """汉明距离——两个等长向量在多少位置上不同。适用于二值/类别特征。"""
    return sum(1 for ai, bi in zip(a, b) if ai != bi)


# ============================================================
# 第 2 步：特征标准化
# ============================================================

def standardize(X):
    """Z -score 标准化：减去均值、除以标准差。

    KNN 对特征尺度极其敏感。一个取值范围 0-1000 的特征会主导
    一个取值范围 0-1 的特征。标准化让每个特征"平等说话"。
    """
    n = len(X)
    d = len(X[0])
    means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    stds = [
        max(1e-10, (sum((X[i][j] - means[j]) ** 2 for i in range(n)) / n) ** 0.5)
        for j in range(d)
    ]
    X_scaled = [
        [(X[i][j] - means[j]) / stds[j] for j in range(d)] for i in range(n)
    ]
    return X_scaled, means, stds


def apply_standardize(X, means, stds):
    """使用训练集的均值和标准差标准化新数据（测试集场景）。"""
    return [[(x[j] - means[j]) / stds[j] for j in range(len(x))] for x in X]


# ============================================================
# 第 3 步：KNN 分类器与回归器
# ============================================================

class KNN:
    """K 近邻算法——支持分类和回归，可选距离加权。

    核心思想：不学习任何参数，仅在预测时查找最近的 K 个邻居，
    让它们投票（分类）或取平均（回归）。
    """

    def __init__(self, k=5, distance_fn=l2_distance, weighted=False,
                 task="classification"):
        self.k = k
        self.distance_fn = distance_fn
        self.weighted = weighted
        self.task = task  # "classification" 或 "regression"
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """训练阶段：仅存储数据（惰性学习的核心体现）。"""
        self.X_train = list(X)
        self.y_train = list(y)

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def predict_with_neighbors(self, x):
        """预测并返回邻居信息（用于可解释性分析）。"""
        distances = []
        for i in range(len(self.X_train)):
            d = self.distance_fn(x, self.X_train[i])
            distances.append((d, i, self.y_train[i]))
        distances.sort(key=lambda t: t[0])
        neighbors = distances[: self.k]
        prediction = self._predict_one(x)
        return prediction, neighbors

    def _predict_one(self, x):
        distances = []
        for i in range(len(self.X_train)):
            d = self.distance_fn(x, self.X_train[i])
            distances.append((d, self.y_train[i]))
        distances.sort(key=lambda pair: pair[0])
        neighbors = distances[: self.k]

        if self.task == "classification":
            return self._classify(neighbors)
        return self._regress(neighbors)

    def _classify(self, neighbors):
        """分类预测：多数投票或加权投票。"""
        if self.weighted:
            votes = {}
            for dist, label in neighbors:
                w = 1.0 / (dist + 1e-10)  # 防止距离为 0
                votes[label] = votes.get(label, 0) + w
            return max(votes, key=votes.get)
        else:
            votes = {}
            for _, label in neighbors:
                votes[label] = votes.get(label, 0) + 1
            return max(votes, key=votes.get)

    def _regress(self, neighbors):
        """回归预测：平均值或加权平均值。"""
        if self.weighted:
            w_sum = 0.0
            val_sum = 0.0
            for dist, val in neighbors:
                w = 1.0 / (dist + 1e-10)
                val_sum += w * val
                w_sum += w
            return val_sum / w_sum if w_sum > 0 else 0.0
        return sum(val for _, val in neighbors) / len(neighbors)


# ============================================================
# 第 4 步：KD 树
# ============================================================

class KDNode:
    """KD 树的节点。"""
    def __init__(self, point, index, axis, left=None, right=None):
        self.point = point      # 该节点存储的数据点
        self.index = index      # 原始数据集中的索引
        self.axis = axis        # 当前分割维度
        self.left = left        # 左子树
        self.right = right      # 右子树


class KDTree:
    """KD 树——通过递归沿坐标轴划分空间来加速最近邻搜索。

    低维空间（d < 20）中查询复杂度约 O(log n)，
    高维空间（d > 20）中退化为接近 O(n)。
    """

    def __init__(self, X):
        self.dim = len(X[0])
        indexed = [(X[i], i) for i in range(len(X))]
        self.root = self._build(indexed, depth=0)

    def _build(self, points, depth):
        """递归构建 KD 树：在当前维度上取中位数分割。"""
        if not points:
            return None
        axis = depth % self.dim
        points.sort(key=lambda p: p[0][axis])
        mid = len(points) // 2
        return KDNode(
            point=points[mid][0],
            index=points[mid][1],
            axis=axis,
            left=self._build(points[:mid], depth + 1),
            right=self._build(points[mid + 1 :], depth + 1),
        )

    def query(self, point, k=1):
        """查找 k 个最近邻。返回 [(距离, 索引, 点坐标), ...]。"""
        best = []
        self._search(self.root, point, k, best)
        best.sort(key=lambda x: x[0])
        return best

    def _search(self, node, point, k, best):
        """递归搜索 + 回溯剪枝。"""
        if node is None:
            return

        dist = l2_distance(point, node.point)

        # 维护当前最优的 k 个邻居
        if len(best) < k:
            best.append((dist, node.index, node.point))
            best.sort(key=lambda x: x[0])
        elif dist < best[-1][0]:
            best[-1] = (dist, node.index, node.point)
            best.sort(key=lambda x: x[0])

        # 决定先搜索哪一侧
        axis = node.axis
        diff = point[axis] - node.point[axis]
        if diff <= 0:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        self._search(first, point, k, best)

        # 剪枝：只有当另一侧可能包含更近的点时才搜索
        if len(best) < k or abs(diff) < best[-1][0]:
            self._search(second, point, k, best)


# ============================================================
# 第 5 步：评估函数
# ============================================================

def accuracy(y_true, y_pred):
    """分类准确率。"""
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def mse(y_true, y_pred):
    """均方误差（回归评估）。"""
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


# ============================================================
# 第 6 步：数据生成
# ============================================================

def generate_classification_data(n_samples=200, n_classes=3, seed=42):
    """生成多分类数据集：每个类别是一个高斯分布簇。"""
    random.seed(seed)
    X, y = [], []
    centers = [[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
    for _ in range(n_samples):
        c = random.randint(0, n_classes - 1)
        x1 = centers[c][0] + random.gauss(0, 0.5)
        x2 = centers[c][1] + random.gauss(0, 0.5)
        X.append([x1, x2])
        y.append(c)
    return X, y


def generate_regression_data(n_samples=200, seed=42):
    """生成回归数据集：y = sin(x) + 噪声。"""
    random.seed(seed)
    X, y = [], []
    for _ in range(n_samples):
        x = random.uniform(-3, 3)
        target = math.sin(x) + random.gauss(0, 0.15)
        X.append([x])
        y.append(target)
    return X, y


def generate_high_dim_data(n_samples=500, n_dims=2, seed=42):
    """生成高维数据集：标签仅依赖前两个维度，其余为噪声。

    用于演示维度灾难——随着维度增加，噪声维度淹没信号。
    """
    random.seed(seed)
    X, y = [], []
    for _ in range(n_samples):
        point = [random.uniform(0, 1) for _ in range(n_dims)]
        label = 1 if sum(point[:2]) > 1.0 else 0
        X.append(point)
        y.append(label)
    return X, y


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """随机划分训练集和测试集。"""
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return (
        [X[i] for i in train_idx],
        [y[i] for i in train_idx],
        [X[i] for i in test_idx],
        [y[i] for i in test_idx],
    )


# ============================================================
# 演示函数
# ============================================================

def demo_basic_knn():
    print("=" * 65)
    print("演示 1：KNN 分类基础")
    print("=" * 65)
    print()

    X, y = generate_classification_data(200, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    print(f"  数据集：{len(X)} 个样本，2 个特征，3 个类别")
    print(f"  训练集：{len(X_train)}  测试集：{len(X_test)}")
    print()

    k_values = [1, 3, 5, 7, 11, 15, 25, 50]
    print(f"  {'K':>6s}  {'训练准确率':>12s}  {'测试准确率':>12s}")
    print(f"  {'-' * 6}  {'-' * 12}  {'-' * 12}")

    for k in k_values:
        knn = KNN(k=k, task="classification")
        knn.fit(X_train, y_train)
        train_acc = accuracy(y_train, knn.predict(X_train))
        test_acc = accuracy(y_test, knn.predict(X_test))
        print(f"  {k:>6d}  {train_acc:>12.4f}  {test_acc:>12.4f}")

    print()
    print("  K=1：训练准确率 100%（记住了所有数据），测试准确率较低（过拟合）。")
    print("  增大 K 使决策边界更平滑，测试准确率先升后降。")
    print()


def demo_distance_metrics():
    print("=" * 65)
    print("演示 2：距离度量的影响")
    print("=" * 65)
    print()

    X, y = generate_classification_data(200, seed=42)
    X_scaled, means, stds = standardize(X)
    X_train, y_train, X_test, y_test = train_test_split(X_scaled, y)

    metrics = [
        ("L2 (欧几里得)", l2_distance),
        ("L1 (曼哈顿)", l1_distance),
        ("余弦距离", cosine_distance),
    ]

    k = 5
    print(f"  K = {k}，特征已标准化")
    print()
    print(f"  {'度量':<20s}  {'测试准确率':>12s}")
    print(f"  {'-' * 20}  {'-' * 12}")

    for name, dist_fn in metrics:
        knn = KNN(k=k, distance_fn=dist_fn, task="classification")
        knn.fit(X_train, y_train)
        test_acc = accuracy(y_test, knn.predict(X_test))
        print(f"  {name:<20s}  {test_acc:>12.4f}")

    print()

    # 展示不同度量下邻居的差异
    query = X_test[0]
    print(f"  查询点：[{query[0]:.3f}, {query[1]:.3f}]，真实标签：{y_test[0]}")
    print()

    for name, dist_fn in metrics:
        knn = KNN(k=k, distance_fn=dist_fn, task="classification")
        knn.fit(X_train, y_train)
        pred, neighbors = knn.predict_with_neighbors(query)
        print(f"  {name}：预测 = {pred}")
        for dist, idx, label in neighbors:
            print(f"    邻居 idx={idx}，标签={label}，距离={dist:.4f}")
        print()


def demo_weighted_knn():
    print("=" * 65)
    print("演示 3：加权 vs 非加权 KNN")
    print("=" * 65)
    print()

    X, y = generate_classification_data(200, seed=42)
    X_scaled, _, _ = standardize(X)
    X_train, y_train, X_test, y_test = train_test_split(X_scaled, y)

    k_values = [3, 7, 15, 25]
    print(f"  {'K':>6s}  {'非加权准确率':>14s}  {'加权准确率':>12s}  {'差异':>8s}")
    print(f"  {'-' * 6}  {'-' * 14}  {'-' * 12}  {'-' * 8}")

    for k in k_values:
        knn_uw = KNN(k=k, weighted=False, task="classification")
        knn_w = KNN(k=k, weighted=True, task="classification")
        knn_uw.fit(X_train, y_train)
        knn_w.fit(X_train, y_train)
        acc_uw = accuracy(y_test, knn_uw.predict(X_test))
        acc_w = accuracy(y_test, knn_w.predict(X_test))
        diff = acc_w - acc_uw
        print(f"  {k:>6d}  {acc_uw:>14.4f}  {acc_w:>12.4f}  {diff:>+8.4f}")

    print()
    print("  加权 KNN 对大 K 值不那么敏感——远距离邻居贡献极小。")
    print()


def demo_regression():
    print("=" * 65)
    print("演示 4：KNN 回归——拟合 sin(x)")
    print("=" * 65)
    print()

    X, y = generate_regression_data(200, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    k_values = [1, 3, 5, 10, 20, 50]
    print(f"  目标函数：y = sin(x) + 噪声")
    print(f"  训练集：{len(X_train)}  测试集：{len(X_test)}")
    print()
    print(f"  {'K':>6s}  {'非加权 MSE':>14s}  {'加权 MSE':>12s}")
    print(f"  {'-' * 6}  {'-' * 14}  {'-' * 12}")

    for k in k_values:
        knn_uw = KNN(k=k, task="regression", weighted=False)
        knn_w = KNN(k=k, task="regression", weighted=True)
        knn_uw.fit(X_train, y_train)
        knn_w.fit(X_train, y_train)
        mse_uw = mse(y_test, knn_uw.predict(X_test))
        mse_w = mse(y_test, knn_w.predict(X_test))
        print(f"  {k:>6d}  {mse_uw:>14.6f}  {mse_w:>12.6f}")

    print()
    print("  K=1 过拟合（跟随噪声），大 K 欠拟合（过度平滑）。")
    print("  加权 KNN 在保持局部结构的同时产生更平滑的预测。")
    print()

    # 展示部分预测结果
    knn = KNN(k=5, task="regression", weighted=True)
    knn.fit(X_train, y_train)
    print("  部分预测结果（K=5，加权）：")
    print(f"  {'x':>8s}  {'真实 y':>8s}  {'预测 y':>8s}  {'误差':>8s}")
    print(f"  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for i in range(min(10, len(X_test))):
        pred = knn.predict([X_test[i]])[0]
        err = abs(y_test[i] - pred)
        print(f"  {X_test[i][0]:>8.3f}  {y_test[i]:>8.3f}  {pred:>8.3f}  {err:>8.3f}")
    print()


def demo_curse_of_dimensionality():
    print("=" * 65)
    print("演示 5：维度灾难")
    print("=" * 65)
    print()

    dims = [2, 5, 10, 20, 50, 100]
    n_points = 200

    print("  第一部分：距离比值收敛")
    print(f"  {n_points} 个 [0, 1]^d 均匀随机点")
    print()
    print(f"  {'维度':>10s}  {'最大/最小距离':>16s}  {'平均距离':>10s}  {'距离标准差':>12s}")
    print(f"  {'-' * 10}  {'-' * 16}  {'-' * 10}  {'-' * 12}")

    for d in dims:
        random.seed(42)
        points = [[random.uniform(0, 1) for _ in range(d)] for _ in range(n_points)]

        distances = []
        sample_size = min(500, n_points * (n_points - 1) // 2)
        for _ in range(sample_size):
            i = random.randint(0, n_points - 1)
            j = random.randint(0, n_points - 1)
            if i != j:
                distances.append(l2_distance(points[i], points[j]))

        if distances:
            max_d = max(distances)
            min_d = min(d_val for d_val in distances if d_val > 0)
            mean_d = sum(distances) / len(distances)
            std_d = (sum((d_val - mean_d) ** 2 for d_val in distances) / len(distances)) ** 0.5
            ratio = max_d / min_d if min_d > 0 else float("inf")
            print(f"  {d:>10d}  {ratio:>16.4f}  {mean_d:>10.4f}  {std_d:>12.4f}")

    print()
    print("  随着维度增长，最大/最小距离比值趋近于 1。")
    print("  所有点几乎等距——'最近邻'失去意义。")
    print()

    print("  第二部分：KNN 准确率随维度退化")
    print(f"  二分类：标签 = 1 当且仅当 x[0] + x[1] > 1，其余维度为纯噪声")
    print()
    print(f"  {'维度':>10s}  {'K=5 准确率':>12s}  {'K=15 准确率':>12s}")
    print(f"  {'-' * 10}  {'-' * 12}  {'-' * 12}")

    for d in [2, 5, 10, 20, 50]:
        X, y = generate_high_dim_data(400, n_dims=d, seed=42)
        X_scaled, _, _ = standardize(X)
        X_train, y_train, X_test, y_test = train_test_split(X_scaled, y)

        knn5 = KNN(k=5, task="classification")
        knn15 = KNN(k=15, task="classification")
        knn5.fit(X_train, y_train)
        knn15.fit(X_train, y_train)
        acc5 = accuracy(y_test, knn5.predict(X_test))
        acc15 = accuracy(y_test, knn15.predict(X_test))
        print(f"  {d:>10d}  {acc5:>12.4f}  {acc15:>12.4f}")

    print()
    print("  准确率随噪声维度增加而下降——信号被噪声淹没。")
    print()


def demo_kdtree():
    print("=" * 65)
    print("演示 6：KD 树加速最近邻搜索")
    print("=" * 65)
    print()

    random.seed(42)
    sizes = [100, 500, 1000, 5000]

    print(f"  2 维数据，查找 5 个最近邻")
    print()
    print(f"  {'数据量':>10s}  {'暴力搜索':>14s}  {'KD 树':>14s}  {'加速比':>10s}")
    print(f"  {'-' * 10}  {'-' * 14}  {'-' * 14}  {'-' * 10}")

    for n in sizes:
        X = [[random.uniform(0, 10) for _ in range(2)] for _ in range(n)]
        k = 5
        n_queries = 100
        queries = [[random.uniform(0, 10) for _ in range(2)] for _ in range(n_queries)]

        # 暴力搜索
        start = time.time()
        for q in queries:
            dists = [(l2_distance(q, X[i]), i) for i in range(n)]
            dists.sort()
            _ = dists[:k]
        brute_time = time.time() - start

        # KD 树搜索
        tree = KDTree(X)
        start = time.time()
        for q in queries:
            _ = tree.query(q, k=k)
        kd_time = time.time() - start

        speedup = brute_time / kd_time if kd_time > 0 else float("inf")
        print(f"  {n:>10d}  {brute_time:>14.4f}s  {kd_time:>14.4f}s  {speedup:>10.1f}x")

    print()

    # 验证 KD 树结果与暴力搜索一致
    X = [[random.uniform(0, 10) for _ in range(2)] for _ in range(100)]
    tree = KDTree(X)
    query = [5.0, 5.0]

    brute = [(l2_distance(query, X[i]), i) for i in range(len(X))]
    brute.sort()
    brute_top5 = [(d, idx) for d, idx in brute[:5]]
    kd_top5 = [(d, idx) for d, idx, _ in tree.query(query, k=5)]

    print("  正确性验证（100 个点，k=5）：")
    print(f"    暴力搜索：{[(round(d, 4), idx) for d, idx in brute_top5]}")
    print(f"    KD 树：    {[(round(d, 4), idx) for d, idx in kd_top5]}")
    match = set(idx for _, idx in brute_top5) == set(idx for _, idx in kd_top5)
    print(f"    结果一致：{match}")
    print()


def demo_scaling_importance():
    print("=" * 65)
    print("演示 7：特征标准化的重要性")
    print("=" * 65)
    print()

    random.seed(42)
    X, y = [], []
    for _ in range(200):
        age = random.gauss(40, 15)
        salary = random.gauss(50000, 20000)
        label = 1 if age > 45 and salary < 40000 else 0
        X.append([age, salary])
        y.append(label)

    X_train, y_train, X_test, y_test = train_test_split(X, y)

    # 未标准化
    knn_raw = KNN(k=5, task="classification")
    knn_raw.fit(X_train, y_train)
    acc_raw = accuracy(y_test, knn_raw.predict(X_test))

    # 标准化后
    X_train_s, means, stds = standardize(X_train)
    X_test_s = apply_standardize(X_test, means, stds)
    knn_scaled = KNN(k=5, task="classification")
    knn_scaled.fit(X_train_s, y_train)
    acc_scaled = accuracy(y_test, knn_scaled.predict(X_test_s))

    print(f"  特征：年龄（范围约 10-70），薪资（范围约 10k-90k）")
    print()
    print(f"  未标准化准确率：{acc_raw:.4f}")
    print(f"  标准化后准确率：{acc_scaled:.4f}")
    print()

    # 展示距离差异
    query = X_test[0]
    query_s = X_test_s[0]
    dists_raw = sorted([l2_distance(query, X_train[i]) for i in range(5)])
    dists_scaled = sorted([l2_distance(query_s, X_train_s[i]) for i in range(5)])

    print(f"  第一个测试点的 5 个最近距离：")
    print(f"    未标准化：{[round(d, 1) for d in dists_raw]}")
    print(f"    标准化后：{[round(d, 4) for d in dists_scaled]}")
    print()
    print("  未标准化时薪资主导距离（数万 vs 数十岁），年龄几乎被忽略。")
    print("  标准化后两个特征对距离的贡献相等。")
    print()


def demo_minkowski_family():
    print("=" * 65)
    print("演示 8：闵可夫斯基距离族")
    print("=" * 65)
    print()

    a = [1.0, 2.0, 3.0]
    b = [4.0, 0.0, 6.0]

    p_values = [1, 1.5, 2, 3, 5, 10, float("inf")]
    print(f"  a = {a}")
    print(f"  b = {b}")
    print()
    print(f"  {'p':>8s}  {'距离':>12s}  {'名称':>15s}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 15}")

    for p in p_values:
        d = minkowski_distance(a, b, p)
        if p == 1:
            name = "曼哈顿 (L1)"
        elif p == 2:
            name = "欧几里得 (L2)"
        elif p == float("inf"):
            name = "切比雪夫 (L∞)"
        else:
            name = f"Lp (p={p})"
        p_str = "∞" if p == float("inf") else str(p)
        print(f"  {p_str:>8s}  {d:>12.4f}  {name:>15s}")

    print()
    print("  随着 p 增大，距离由最大分量差主导。")
    print("  恒有：L∞ ≤ L2 ≤ L1。")
    print()


def demo_k_selection():
    print("=" * 65)
    print("演示 9：交叉验证选择 K")
    print("=" * 65)
    print()

    X, y = generate_classification_data(300, seed=42)

    n = len(X)
    random.seed(42)
    indices = list(range(n))
    random.shuffle(indices)

    n_folds = 5
    fold_size = n // n_folds
    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]

    print(f"  {n_folds} 折交叉验证，{n} 个样本")
    print()
    print(f"  {'K':>6s}  {'平均准确率':>12s}  {'标准差':>10s}  {'可视化':>20s}")
    print(f"  {'-' * 6}  {'-' * 12}  {'-' * 10}  {'-' * 20}")

    best_k = 1
    best_mean = 0.0

    for k in k_values:
        fold_accs = []

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size
            val_idx = indices[val_start:val_end]
            train_idx = indices[:val_start] + indices[val_end:]

            X_tr = [X[i] for i in train_idx]
            y_tr = [y[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]

            knn = KNN(k=k, task="classification")
            knn.fit(X_tr, y_tr)
            acc_val = accuracy(y_val, knn.predict(X_val))
            fold_accs.append(acc_val)

        mean_acc = sum(fold_accs) / len(fold_accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in fold_accs) / len(fold_accs)) ** 0.5

        bar_len = int(mean_acc * 20)
        bar = "#" * bar_len

        if mean_acc > best_mean:
            best_mean = mean_acc
            best_k = k

        print(f"  {k:>6d}  {mean_acc:>12.4f}  {std_acc:>10.4f}  {bar}")

    print()
    print(f"  最优 K = {best_k}，平均准确率 = {best_mean:.4f}")
    print()


def demo_lazy_vs_eager():
    print("=" * 65)
    print("演示 10：惰性学习 vs 急迫学习")
    print("=" * 65)
    print()

    random.seed(42)
    sizes = [100, 500, 1000, 5000]

    print(f"  {'N':>6s}  {'KNN 训练时间':>14s}  {'KNN 预测时间':>14s}  {'总计':>10s}")
    print(f"  {'-' * 6}  {'-' * 14}  {'-' * 14}  {'-' * 10}")

    for n in sizes:
        X = [[random.gauss(0, 1) for _ in range(5)] for _ in range(n)]
        y = [random.choice([0, 1]) for _ in range(n)]

        n_test = min(50, n // 5)
        X_test_local = [[random.gauss(0, 1) for _ in range(5)] for _ in range(n_test)]

        knn = KNN(k=5, task="classification")

        start = time.time()
        knn.fit(X, y)
        train_time = time.time() - start

        start = time.time()
        knn.predict(X_test_local)
        pred_time = time.time() - start

        total = train_time + pred_time
        print(f"  {n:>6d}  {train_time:>14.6f}s  {pred_time:>14.6f}s  {total:>10.6f}s")

    print()
    print("  KNN 训练为 O(1)：仅存储数据。")
    print("  KNN 预测为 O(n·d) 每次查询：计算所有距离。")
    print("  急迫学习（如神经网络）的模式恰好相反。")
    print()


def print_summary():
    print()
    print("=" * 65)
    print("总结")
    print("=" * 65)
    print()
    print("  1. KNN 是惰性学习：零训练开销，所有计算在预测时进行。")
    print("  2. K 控制偏差-方差权衡：小 K 过拟合，大 K 欠拟合。")
    print("  3. 距离度量选择至关重要。L2 是默认值，余弦距离用于文本。")
    print("  4. 必须标准化特征。未标准化的特征会扭曲距离。")
    print("  5. 加权 KNN 通过降低远距离邻居的权重来减少对 K 的敏感度。")
    print("  6. 维度灾难：KNN 在超过约 20-50 维后性能急剧退化。")
    print("  7. KD 树在低维空间加速搜索，球树适用于中等维度。")
    print("  8. KNN 是向量数据库和 RAG 检索背后的同一算法。")
    print()


if __name__ == "__main__":
    demo_basic_knn()
    demo_distance_metrics()
    demo_weighted_knn()
    demo_regression()
    demo_minkowski_family()
    demo_curse_of_dimensionality()
    demo_scaling_importance()
    demo_kdtree()
    demo_lazy_vs_eager()
    demo_k_selection()
    print_summary()
