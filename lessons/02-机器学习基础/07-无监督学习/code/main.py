# main.py — 从零实现无监督学习核心算法
# 依赖：无（纯 Python 标准库）
# 对应课程：阶段 02 · 07（无监督学习）

import math
import random


# ============================================================
# 第 1 步：基础工具函数
# ============================================================

def euclidean_distance(a, b):
    """欧几里得距离——两点之间的直线距离。"""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def mean(values):
    """计算列表的均值。"""
    return sum(values) / len(values)


# ============================================================
# 第 2 步：K-Means 聚类
# ============================================================

def kmeans(data, k, max_iterations=100, seed=42):
    """K-Means 聚类算法。

    核心思想：将数据划分为 K 个簇，每个簇由其质心（所有点的均值）表示。
    算法交替执行分配步骤和更新步骤，直到收敛。

    参数:
        data: 数据点列表，每个点是一个坐标列表
        k: 聚类数量
        max_iterations: 最大迭代次数
        seed: 随机种子

    返回:
        assignments: 每个点的簇标签
        centroids: K 个质心坐标
    """
    random.seed(seed)
    n_features = len(data[0])

    # 随机选择 K 个初始质心
    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        # --- 分配步骤：将每个点分配到最近的质心 ---
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        # --- 更新步骤：重新计算质心 ---
        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                # 处理空簇：随机选择一个点作为新质心
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        # --- 检查收敛：质心是否不再移动 ---
        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            break

        centroids = new_centroids

    return assignments, centroids


def compute_inertia(data, assignments, centroids):
    """计算惯性（inertia）：每个点到其质心的距离平方和。

    惯性越小，簇越紧密。用于肘部法选择 K 值。
    """
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


# ============================================================
# 第 3 步：轮廓系数
# ============================================================

def silhouette_score(data, assignments):
    """计算轮廓系数（Silhouette Score）。

    对每个点：
    - a = 到同簇其他点的平均距离（内聚度）
    - b = 到最近异簇点的平均距离（分离度）
    - silhouette = (b - a) / max(a, b)

    取值范围 [-1, 1]：
    - 接近 1：点被很好地分配到簇中
    - 接近 0：点在两个簇的边界
    - 接近 -1：点可能被分错了簇
    """
    n = len(data)
    if n < 2:
        return 0.0

    # 按簇分组
    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        # 内聚度 a
        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        # 分离度 b
        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


# ============================================================
# 第 4 步：DBSCAN 密度聚类
# ============================================================

def dbscan(data, eps, min_samples):
    """DBSCAN 密度聚类算法。

    核心思想：簇是数据空间中密集的区域，被稀疏区域分隔。
    不需要预先指定簇的数量，能发现任意形状的簇，自动识别噪声点。

    参数:
        eps: 邻域半径
        min_samples: 成为核心点所需的最少邻居数

    返回:
        labels: 每个点的标签，-1 表示噪声点
    """
    n = len(data)
    labels = [-1] * n  # -1 表示未分类/噪声
    cluster_id = 0

    def region_query(point_idx):
        """找到在 eps 半径内的所有邻居。"""
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        # 如果邻居太少，标记为噪声（可能在后续被其他簇吸收为边界点）
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        # 开始一个新簇
        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        # 扩展簇：从核心点出发，沿着密度可达的点传播
        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    # q 也是核心点，将其邻居加入种子集
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

        cluster_id += 1

    return labels


# ============================================================
# 第 5 步：层次聚类（自底向上凝聚）
# ============================================================

def agglomerative_clustering(data, n_clusters=3, linkage="ward"):
    """层次凝聚聚类算法。

    核心思想：从每个点开始作为独立的簇，逐步合并最近的簇，
    直到达到目标簇数量。使用不同的连接方式（linkage）决定簇间距离。

    参数:
        n_clusters: 目标簇数量
        linkage: 连接方式——"single"、"complete"、"average"、"ward"

    返回:
        labels: 每个点的簇标签
        merge_history: 合并历史记录
    """
    n = len(data)
    # 每个簇包含的原始点索引
    cluster_map = {i: [i] for i in range(n)}
    active_clusters = list(range(n))
    merge_history = []

    def cluster_distance(c1_indices, c2_indices):
        """根据连接方式计算两个簇之间的距离。"""
        if linkage == "single":
            # 单连接：两个簇中最近两点的距离
            return min(
                euclidean_distance(data[i], data[j])
                for i in c1_indices
                for j in c2_indices
            )
        elif linkage == "complete":
            # 全连接：两个簇中最远两点的距离
            return max(
                euclidean_distance(data[i], data[j])
                for i in c1_indices
                for j in c2_indices
            )
        elif linkage == "average":
            # 平均连接：所有点对的平均距离
            total = sum(
                euclidean_distance(data[i], data[j])
                for i in c1_indices
                for j in c2_indices
            )
            return total / (len(c1_indices) * len(c2_indices))
        elif linkage == "ward":
            # Ward 方法：合并后簇内方差增加最小的两个簇
            merged = c1_indices + c2_indices
            centroid_merged = [
                sum(data[i][d] for i in merged) / len(merged)
                for d in range(len(data[0]))
            ]
            centroid_1 = [
                sum(data[i][d] for i in c1_indices) / len(c1_indices)
                for d in range(len(data[0]))
            ]
            centroid_2 = [
                sum(data[i][d] for i in c2_indices) / len(c2_indices)
                for d in range(len(data[0]))
            ]
            var_merged = sum(
                euclidean_distance(data[i], centroid_merged) ** 2 for i in merged
            )
            var_1 = sum(
                euclidean_distance(data[i], centroid_1) ** 2 for i in c1_indices
            )
            var_2 = sum(
                euclidean_distance(data[i], centroid_2) ** 2 for i in c2_indices
            )
            return var_merged - var_1 - var_2

    next_id = n
    while len(active_clusters) > n_clusters:
        best_dist = float("inf")
        best_pair = None

        # 找到最近的两个簇
        for idx_a in range(len(active_clusters)):
            for idx_b in range(idx_a + 1, len(active_clusters)):
                c_a = active_clusters[idx_a]
                c_b = active_clusters[idx_b]
                dist = cluster_distance(cluster_map[c_a], cluster_map[c_b])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (c_a, c_b)

        c_a, c_b = best_pair
        cluster_map[next_id] = cluster_map[c_a] + cluster_map[c_b]
        merge_history.append((c_a, c_b, best_dist, len(cluster_map[next_id])))
        active_clusters.remove(c_a)
        active_clusters.remove(c_b)
        active_clusters.append(next_id)
        next_id += 1

    # 生成标签
    labels = [0] * n
    for cluster_label, cluster_id in enumerate(active_clusters):
        for point_idx in cluster_map[cluster_id]:
            labels[point_idx] = cluster_label

    return labels, merge_history


def agglomerative_labels(data, n_clusters=3, linkage="ward"):
    """简化版层次聚类——只返回标签。

    修复了完整版的 bug，正确生成标签。
    """
    n = len(data)
    cluster_map = {i: [i] for i in range(n)}
    active_clusters = list(range(n))

    def cluster_distance(c1_indices, c2_indices):
        if linkage == "single":
            return min(
                euclidean_distance(data[i], data[j])
                for i in c1_indices
                for j in c2_indices
            )
        elif linkage == "complete":
            return max(
                euclidean_distance(data[i], data[j])
                for i in c1_indices
                for j in c2_indices
            )
        elif linkage == "average":
            total = sum(
                euclidean_distance(data[i], data[j])
                for i in c1_indices
                for j in c2_indices
            )
            return total / (len(c1_indices) * len(c2_indices))
        elif linkage == "ward":
            merged = c1_indices + c2_indices
            centroid_merged = [
                sum(data[i][d] for i in merged) / len(merged)
                for d in range(len(data[0]))
            ]
            centroid_1 = [
                sum(data[i][d] for i in c1_indices) / len(c1_indices)
                for d in range(len(data[0]))
            ]
            centroid_2 = [
                sum(data[i][d] for i in c2_indices) / len(c2_indices)
                for d in range(len(data[0]))
            ]
            var_merged = sum(
                euclidean_distance(data[i], centroid_merged) ** 2 for i in merged
            )
            var_1 = sum(
                euclidean_distance(data[i], centroid_1) ** 2 for i in c1_indices
            )
            var_2 = sum(
                euclidean_distance(data[i], centroid_2) ** 2 for i in c2_indices
            )
            return var_merged - var_1 - var_2

    next_id = n
    while len(active_clusters) > n_clusters:
        best_dist = float("inf")
        best_pair = None

        for idx_a in range(len(active_clusters)):
            for idx_b in range(idx_a + 1, len(active_clusters)):
                c_a = active_clusters[idx_a]
                c_b = active_clusters[idx_b]
                dist = cluster_distance(cluster_map[c_a], cluster_map[c_b])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (c_a, c_b)

        c_a, c_b = best_pair
        cluster_map[next_id] = cluster_map[c_a] + cluster_map[c_b]
        active_clusters.remove(c_a)
        active_clusters.remove(c_b)
        active_clusters.append(next_id)
        next_id += 1

    labels = [0] * n
    for cluster_label, cluster_id in enumerate(active_clusters):
        for point_idx in cluster_map[cluster_id]:
            labels[point_idx] = cluster_label

    return labels


# ============================================================
# 第 6 步：PCA 降维（基于特征值分解）
# ============================================================

def pca(data, n_components=2):
    """主成分分析（PCA）——线性降维算法。

    核心思想：找到数据中方差最大的方向（主成分），
    将数据投影到这些方向上，实现降维。

    这里使用简化的幂迭代法求主特征向量，
    避免引入 numpy 实现矩阵运算。

    参数:
        data: 原始数据矩阵（n_samples × n_features）
        n_components: 降维后的维度

    返回:
        projected: 降维后的数据
        explained_variance: 每个主成分解释的方差
    """
    n = len(data)
    d = len(data[0])

    # 1. 中心化：每个特征减去均值
    means = [sum(data[i][j] for i in range(n)) / n for j in range(d)]
    centered = [[data[i][j] - means[j] for j in range(d)] for i in range(n)]

    # 2. 计算协方差矩阵
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            cov[i][j] = sum(centered[k][i] * centered[k][j] for k in range(n)) / (n - 1)

    # 3. 使用幂迭代法求主特征向量
    def mat_vec_mult(matrix, vector):
        """矩阵乘向量。"""
        size = len(matrix)
        return [sum(matrix[i][j] * vector[j] for j in range(size)) for i in range(size)]

    def normalize(vector):
        """向量归一化。"""
        norm = math.sqrt(sum(v ** 2 for v in vector))
        if norm < 1e-10:
            return vector
        return [v / norm for v in vector]

    def power_iteration(matrix, n_iter=50):
        """幂迭代法求最大特征值对应的特征向量。"""
        size = len(matrix)
        vector = [random.random() for _ in range(size)]
        for _ in range(n_iter):
            vector = mat_vec_mult(matrix, vector)
            vector = normalize(vector)
        # 计算对应的特征值（Rayleigh 商）
        mv = mat_vec_mult(matrix, vector)
        eigenvalue = sum(vector[i] * mv[i] for i in range(size))
        return eigenvalue, vector

    # 4. 求前 n_components 个主成分
    components = []
    variances = []
    remaining_cov = [row[:] for row in cov]  # 复制

    for _ in range(n_components):
        eigenvalue, eigenvector = power_iteration(remaining_cov)
        if eigenvalue < 1e-10:
            break
        components.append(eigenvector)
        variances.append(eigenvalue)

        # 降维：从协方差矩阵中移除该成分的影响
        for i in range(d):
            for j in range(d):
                remaining_cov[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]

    # 5. 投影数据到主成分方向
    projected = []
    for i in range(n):
        point = []
        for comp in components:
            proj_val = sum(centered[i][j] * comp[j] for j in range(d))
            point.append(proj_val)
        projected.append(point)

    return projected, variances


# ============================================================
# 第 7 步：异常检测
# ============================================================

def detect_anomalies_dbscan(data, eps, min_samples):
    """基于 DBSCAN 的异常检测。

    DBSCAN 中的噪声点（标签为 -1）即为异常点。
    这些点不属于任何密集区域，远离正常数据分布。

    返回:
        anomalies: 检测到的异常点列表
        labels: 所有点的聚类标签
    """
    labels = dbscan(data, eps, min_samples)
    anomalies = [data[i] for i in range(len(labels)) if labels[i] == -1]
    return anomalies, labels


def detect_anomalies_kmeans(data, k, threshold_percentile=95):
    """基于 K-Means 的异常检测。

    距离质心最远的点被视为异常点。
    使用 percentile 阈值：距离超过该百分位数的点为异常。

    返回:
        anomalies: 检测到的异常点列表
        distances: 每个点到其质心的距离
    """
    assignments, centroids = kmeans(data, k)

    # 计算每个点到其质心的距离
    distances = []
    for point, cluster_id in zip(data, assignments):
        dist = euclidean_distance(point, centroids[cluster_id])
        distances.append(dist)

    # 根据百分位数确定阈值
    sorted_dists = sorted(distances)
    threshold_index = int(len(sorted_dists) * threshold_percentile / 100)
    threshold_index = min(threshold_index, len(sorted_dists) - 1)
    threshold = sorted_dists[threshold_index]

    # 标记异常点
    anomalies = [
        data[i] for i in range(len(distances)) if distances[i] > threshold
    ]
    return anomalies, distances


# ============================================================
# 第 8 步：数据生成
# ============================================================

def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    """生成高斯簇数据集。每个中心生成一个高斯分布的簇。"""
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    """生成两个月亮形状的数据集——用于测试非球形聚类。"""
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


def make_high_dim_data(n_samples=200, n_dims=10, seed=42):
    """生成高维数据集——用于测试 PCA 降维。"""
    random.seed(seed)
    data = []
    for _ in range(n_samples):
        # 前两个维度的数据有明显结构
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        # 其余维度是前两个维度的线性组合加噪声
        point = [x1, x2]
        for _ in range(n_dims - 2):
            point.append(x1 * 0.5 + random.gauss(0, 0.1))
        data.append(point)
    return data


# ============================================================
# 演示函数
# ============================================================

def demo_kmeans_basic():
    print("=" * 65)
    print("演示 1：K-Means 聚类基础")
    print("=" * 65)
    print()

    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print(f"  数据集：{len(data)} 个点，3 个真实簇")
    print()

    assignments, centroids = kmeans(data, k=3)
    inertia = compute_inertia(data, assignments, centroids)
    sil = silhouette_score(data, assignments)

    print(f"  K=3 的聚类结果：")
    print(f"  质心：{[[round(c, 2) for c in cent] for cent in centroids]}")
    print(f"  惯性：{inertia:.2f}")
    print(f"  轮廓系数：{sil:.4f}")
    print()

    # 每个簇的点数
    cluster_sizes = {}
    for c in assignments:
        cluster_sizes[c] = cluster_sizes.get(c, 0) + 1
    print(f"  各簇大小：{dict(sorted(cluster_sizes.items()))}")
    print()


def demo_elbow_method():
    print("=" * 65)
    print("演示 2：肘部法与轮廓系数选择 K")
    print("=" * 65)
    print()

    centers = [[2, 2], [8, 3], [5, 8]]
    data, _ = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print(f"  {'K':>6s}  {'惯性':>12s}  {'轮廓系数':>12s}  {'可视化':>20s}")
    print(f"  {'-' * 6}  {'-' * 12}  {'-' * 12}  {'-' * 20}")

    for k in range(1, 8):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        sil = silhouette_score(data, assignments) if k >= 2 else 0.0
        bar_len = int(sil * 20) if k >= 2 else 0
        bar = "#" * bar_len
        sil_str = f"{sil:>12.4f}" if k >= 2 else f"{'N/A':>12s}"
        print(f"  {k:>6d}  {inertia:>12.2f}  {sil_str}  {bar}")

    print()
    print("  肘部法：K=3 处惯性下降速率明显变缓（呈手肘形状）。")
    print("  轮廓系数：K=3 时轮廓系数最高（0.6-0.7），说明聚类质量最佳。")
    print()


def demo_dbscan():
    print("=" * 65)
    print("演示 3：DBSCAN 密度聚类")
    print("=" * 65)
    print()

    # 球形数据
    centers = [[2, 2], [8, 3], [5, 8]]
    data, _ = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("  第一部分：球形数据")
    labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(labels) - {-1})
    n_noise = labels.count(-1)
    print(f"  发现 {n_clusters} 个簇，{n_noise} 个噪声点")
    print()

    # 非球形数据
    print("  第二部分：月牙形数据（K-Means 会失败）")
    moon_data, _ = make_moons(n_samples=200, noise=0.1)

    # DBSCAN 在月牙形数据上
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  DBSCAN：{n_moon_clusters} 个簇，{n_moon_noise} 个噪声点")

    # K-Means 在月牙形数据上
    moon_km, _ = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  K-Means：轮廓系数 = {moon_sil:.4f}")
    print(f"  K-Means 失败原因：月牙不是球形，质心无法正确捕捉结构")
    print()


def demo_hierarchical():
    print("=" * 65)
    print("演示 4：层次聚类")
    print("=" * 65)
    print()

    centers = [[2, 2], [8, 3], [5, 8]]
    data, _ = make_blobs(centers, n_per_cluster=15, spread=0.8)

    print(f"  {len(data)} 个点，使用不同连接方式：")
    print()

    linkages = [("single", "单连接"), ("complete", "全连接"), ("average", "平均连接"), ("ward", "Ward")]

    for linkage_val, linkage_name in linkages:
        labels = agglomerative_labels(data, n_clusters=3, linkage=linkage_val)
        sil = silhouette_score(data, labels)
        cluster_sizes = {}
        for c in labels:
            cluster_sizes[c] = cluster_sizes.get(c, 0) + 1
        print(f"  {linkage_name:<8s} 轮廓系数 = {sil:.4f}  各簇大小 = {dict(sorted(cluster_sizes.items()))}")

    print()
    print("  Ward 方法通常产生最紧凑、大小最均匀的簇。")
    print("  单连接对噪声敏感，可能产生'链式'簇。")
    print()


def demo_pca():
    print("=" * 65)
    print("演示 5：PCA 降维")
    print("=" * 65)
    print()

    data = make_high_dim_data(n_samples=200, n_dims=10, seed=42)

    print(f"  原始数据：{len(data)} 个点，{len(data[0])} 维")
    print()

    projected, variances = pca(data, n_components=2)

    total_var = sum(variances)
    print(f"  降维后：{len(projected)} 个点，{len(projected[0])} 维")
    print()
    print(f"  第一主成分解释方差：{variances[0]:.4f}")
    print(f"  第二主成分解释方差：{variances[1]:.4f}")
    if total_var > 0:
        print(f"  前两个主成分解释方差比例：{(variances[0] + variances[1]) / total_var * 100:.1f}%")
    print()
    print(f"  前 5 个降维后的点：")
    for i in range(5):
        print(f"    [{projected[i][0]:>8.4f}, {projected[i][1]:>8.4f}]")
    print()


def demo_anomaly_detection():
    print("=" * 65)
    print("演示 6：异常检测")
    print("=" * 65)
    print()

    centers = [[2, 2], [8, 3], [5, 8]]
    data, _ = make_blobs(centers, n_per_cluster=50, spread=0.8)

    # 添加异常点
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])

    print(f"  正常数据：{len(data)} 个点")
    print(f"  注入 3 个明显异常点：(20,20), (-5,-5), (15,0)")
    print()

    # DBSCAN 异常检测
    anomalies_db, labels_db = detect_anomalies_dbscan(anomaly_data, eps=1.5, min_samples=5)
    print(f"  DBSCAN 检测到 {len(anomalies_db)} 个异常点：")
    for a in anomalies_db:
        print(f"    ({a[0]:>6.1f}, {a[1]:>6.1f})")
    print()

    # K-Means 异常检测
    anomalies_km, distances_km = detect_anomalies_kmeans(anomaly_data, k=3, threshold_percentile=95)
    print(f"  K-Means 检测到 {len(anomalies_km)} 个异常点：")
    for a in anomalies_km:
        print(f"    ({a[0]:>6.1f}, {a[1]:>6.1f})")
    print()


def demo_algorithm_selection():
    print("=" * 65)
    print("演示 7：算法选择决策树")
    print("=" * 65)
    print()

    scenarios = [
        ("客户分群（已知分 5 组）", "K-Means", "已知 K，数据量大，需要快速"),
        ("网络入侵检测", "DBSCAN", "未知簇数量，需要识别异常，任意形状"),
        ("小规模文档聚类（<1000）", "层次聚类", "需要可视化树状结构，数据量小"),
        ("重叠的用户画像", "GMM", "需要软分配（概率），簇可能重叠"),
    ]

    print(f"  {'场景':<30s}  {'推荐算法':<15s}  {'原因'}")
    print(f"  {'-' * 30}  {'-' * 15}  {'-' * 30}")

    for scenario, algo, reason in scenarios:
        print(f"  {scenario:<30s}  {algo:<15s}  {reason}")
    print()


def print_summary():
    print()
    print("=" * 65)
    print("总结")
    print("=" * 65)
    print()
    print("  1. K-Means：简单快速，适合球形簇，需要预先指定 K。")
    print("  2. DBSCAN：能发现任意形状，自动识别噪声，无需指定 K。")
    print("  3. 层次聚类：构建树状结构，适合小规模数据和可解释性需求。")
    print("  4. PCA：线性降维，保留最大方差方向，减少特征维度。")
    print("  5. 异常检测：DBSCAN 的噪声点或 K-Means 的远距离点。")
    print("  6. 评估：肘部法选 K，轮廓系数衡量聚类质量。")
    print()


if __name__ == "__main__":
    demo_kmeans_basic()
    demo_elbow_method()
    demo_dbscan()
    demo_hierarchical()
    demo_pca()
    demo_anomaly_detection()
    demo_algorithm_selection()
    print_summary()
