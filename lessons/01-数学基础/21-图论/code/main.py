# main.py — 图论基础算法从零实现
# 依赖：numpy>=1.24
# 对应课程：阶段 01 · 21（图论）

import numpy as np
from collections import deque


# === 第 1 步：图的类与基本表示 ===

class Graph:
    """无向/有向图的邻接表表示。

    内部使用字典的字典存储边，同时支持邻接矩阵和度矩阵的构造。
    """

    def __init__(self, n_nodes, directed=False):
        """
        Args:
            n_nodes: 节点数量，节点编号从 0 到 n_nodes-1
            directed: 是否为有向图
        """
        self.n = n_nodes
        self.directed = directed
        # 邻接表：adj[u][v] = weight，表示边 u->v 的权重
        self.adj = {i: {} for i in range(n_nodes)}

    def add_edge(self, u, v, weight=1.0):
        """添加一条边。无向图会同时添加反向边。"""
        self.adj[u][v] = weight
        if not self.directed:
            self.adj[v][u] = weight

    def neighbors(self, node):
        """返回节点的邻居列表。"""
        return list(self.adj[node].keys())

    def degree(self, node):
        """返回节点的度（邻居数量）。"""
        return len(self.adj[node])

    def weighted_degree(self, node):
        """返回节点的加权度（所有邻接边权重之和）。"""
        return sum(self.adj[node].values())

    def adjacency_matrix(self):
        """构造邻接矩阵 A（稠密矩阵，用于谱分析）。"""
        A = np.zeros((self.n, self.n))
        for u in range(self.n):
            for v, w in self.adj[u].items():
                A[u][v] = w
        return A

    def degree_matrix(self):
        """构造度矩阵 D（对角矩阵，D[i][i] = 节点 i 的加权度）。"""
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            D[i][i] = self.weighted_degree(i)
        return D

    def laplacian(self):
        """计算拉普拉斯矩阵 L = D - A。"""
        return self.degree_matrix() - self.adjacency_matrix()

    def adjacency_list(self):
        """返回邻接表表示（节省稀疏图的内存）。"""
        return {u: list(neighbors.keys()) for u, neighbors in self.adj.items()}

    def __repr__(self):
        edges = []
        seen = set()
        for u in range(self.n):
            for v, w in self.adj[u].items():
                key = (min(u, v), max(u, v)) if not self.directed else (u, v)
                if key not in seen:
                    seen.add(key)
                    if w == 1.0:
                        edges.append(f"{u}-{v}")
                    else:
                        edges.append(f"{u}-{v}({w})")
        return f"Graph(n={self.n}, directed={self.directed}, edges=[{', '.join(edges)}])"


# === 第 2 步：BFS 与 DFS 遍历 ===

def bfs(graph, start):
    """广度优先搜索。

    使用队列（FIFO），逐层遍历图。可以计算无权图的最短路径。

    Args:
        graph: Graph 实例
        start: 起始节点

    Returns:
        order: 访问顺序列表
        distances: 字典，记录每个节点到起点的距离（边数）
    """
    visited = set()
    order = []
    distances = {}
    # 队列中存储 (节点, 距离) 元组
    queue = deque([(start, 0)])
    visited.add(start)

    while queue:
        node, dist = queue.popleft()
        order.append(node)
        distances[node] = dist
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return order, distances


def dfs(graph, start):
    """深度优先搜索。

    使用栈（LIFO），沿着一条路径尽可能深入再回溯。
    可用于检测连通分量、环检测、拓扑排序。

    Args:
        graph: Graph 实例
        start: 起始节点

    Returns:
        order: 访问顺序列表
    """
    visited = set()
    order = []
    stack = [start]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        # reversed 保证按节点编号降序入栈，从而升序出栈（便于验证）
        for neighbor in reversed(graph.neighbors(node)):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


# === 第 3 步：连通分量与拉普拉斯特征值 ===

def connected_components(graph):
    """找出图中所有连通分量。

    通过对每个未访问节点运行 BFS，收集其可达的所有节点。
    """
    visited = set()
    components = []
    for node in range(graph.n):
        if node not in visited:
            order, _ = bfs(graph, node)
            visited.update(order)
            components.append(order)
    return components


def laplacian_eigenvalues(graph):
    """计算拉普拉斯矩阵的特征值（升序排列）。

    使用 eigvalsh 因为 L 总是对称矩阵（对无向图）。
    特征值 0 的个数 == 连通分量个数。
    """
    L = graph.laplacian()
    eigenvalues = np.linalg.eigvalsh(L)
    return eigenvalues


def fiedler_vector(graph):
    """计算 Fiedler 向量（拉普拉斯矩阵第二小特征值对应的特征向量）。

    Fiedler 向量用于谱聚类：符号分割将图分为两组。
    """
    L = graph.laplacian()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # 跳过第一个特征值（0），取第二个
    return eigenvectors[:, 1]


# === 第 4 步：谱聚类 ===

def spectral_clustering(graph, k=2):
    """谱聚类算法。

    步骤：计算拉普拉斯矩阵 → 取前 k 个特征向量 → K-means 聚类。
    对于 k=2，直接用 Fiedler 向量的符号分割即可。
    """
    if graph.n < 2:
        raise ValueError("谱聚类需要至少 2 个节点")
    if not (2 <= k <= graph.n):
        raise ValueError(f"k 必须满足 2 <= k <= {graph.n}，当前 k={k}")

    L = graph.laplacian()
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # k=2 时用符号分割，不需要 K-means
    if k == 2:
        fiedler = eigenvectors[:, 1]
        labels = np.zeros(graph.n, dtype=int)
        labels[fiedler < 0] = 1
        return labels

    # k>2 时取前 k 个非平凡特征向量，运行 K-means
    features = eigenvectors[:, 1:k + 1]
    # 归一化节点特征向量，使 K-means 基于角度聚类
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    features = features / norms

    # K-means 聚类（固定随机种子保证结果可复现）
    rng = np.random.RandomState(42)
    centroids = features[rng.choice(graph.n, k, replace=False)]

    for _ in range(100):
        # 分配步骤：每个节点归入最近的质心
        dists = np.zeros((graph.n, k))
        for c in range(k):
            dists[:, c] = np.linalg.norm(features - centroids[c], axis=1)
        labels = np.argmin(dists, axis=1)

        # 更新步骤：重新计算质心
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centroids[c] = features[mask].mean(axis=0)

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


# === 第 5 步：Dijkstra 最短路径 ===

def dijkstra(graph, start):
    """Dijkstra 算法——加权图的最短路径（非负权重）。

    使用优先队列（最小堆），每次选择当前距离最小的节点进行松弛。

    Args:
        graph: Graph 实例
        start: 起始节点

    Returns:
        distances: 每个节点到起点的最短距离
        predecessors: 最短路径上的前驱节点（用于回溯路径）
    """
    import heapq

    INF = float('inf')
    distances = {i: INF for i in range(graph.n)}
    predecessors = {i: None for i in range(graph.n)}
    distances[start] = 0

    # 优先队列：(距离, 节点)
    pq = [(0, start)]
    visited = set()

    while pq:
        dist, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        for v in graph.neighbors(u):
            weight = graph.adj[u][v]
            if weight < 0:
                raise ValueError("Dijkstra 算法不支持负权重")
            new_dist = dist + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = u
                heapq.heappush(pq, (new_dist, v))

    return distances, predecessors


# === 第 6 步：图神经网络消息传递 ===

def message_passing(graph, features, weight_matrix):
    """一轮 GNN 消息传递。

    每个节点聚合邻居的特征（取均值），然后乘以权重矩阵。
    这就是 GCN 单层的核心操作。

    公式：H' = A_norm @ H @ W
    其中 A_norm 是行归一化的邻接矩阵。

    Args:
        graph: Graph 实例
        features: 节点特征矩阵，形状 (n_nodes, in_features)
        weight_matrix: 权重矩阵，形状 (in_features, out_features)

    Returns:
        output: 新特征矩阵，形状 (n_nodes, out_features)
    """
    A = graph.adjacency_matrix()
    # 行归一化：每个节点的邻居特征取均值
    # 孤立节点（度为 0）保持特征不变，避免除零
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_norm = A / row_sums

    # 聚合邻居特征 → 线性变换
    aggregated = A_norm @ features
    output = aggregated @ weight_matrix
    return output


# === 第 7 步：PageRank ===

def pagerank(graph, damping=0.85, max_iter=100, tol=1e-6):
    """PageRank 算法。

    模拟随机冲浪者：以概率 damping 跟随链接，以概率 (1-damping) 跳转到随机节点。
    处理悬挂节点（无出边的节点）：将其分数均匀分配给所有节点。

    Args:
        graph: Graph 实例
        damping: 阻尼系数，通常取 0.85
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        scores: PageRank 分数数组，形状 (n_nodes,)
    """
    n = graph.n
    scores = np.ones(n) / n

    for iteration in range(max_iter):
        # 基础分：均匀跳转的概率
        new_scores = np.ones(n) * (1 - damping) / n
        # 悬挂节点分数需要均匀分配
        dangling_sum = 0.0

        for u in range(n):
            out_deg = graph.degree(u)
            if out_deg > 0:
                for v in graph.neighbors(u):
                    new_scores[v] += damping * scores[u] / out_deg
            else:
                dangling_sum += scores[u]

        # 悬挂节点分数均匀分配
        new_scores += damping * dangling_sum / n

        # 检查收敛
        diff = np.abs(new_scores - scores).sum()
        scores = new_scores
        if diff < tol:
            break

    return scores


# === 第 8 步：最小生成树（Kruskal） ===

class UnionFind:
    """并查集数据结构，用于 Kruskal 算法。"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """路径压缩。"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """按秩合并。"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # 已在同一分量，合并会成环
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def kruskal_mst(graph):
    """Kruskal 算法求最小生成树。

    1. 按权重排序所有边
    2. 贪心地选择不会形成环的最小边（用并查集检查）
    3. 直到选了 n-1 条边

    Returns:
        mst_edges: 最小生成树的边列表 [(u, v, weight), ...]
        total_weight: 总权重
    """
    # 收集所有边（无向图只取一次）
    edges = []
    seen = set()
    for u in range(graph.n):
        for v, w in graph.adj[u].items():
            key = (min(u, v), max(u, v)) if not graph.directed else (u, v)
            if key not in seen:
                seen.add(key)
                edges.append((w, u, v))

    # 按权重排序
    edges.sort()

    uf = UnionFind(graph.n)
    mst_edges = []
    total_weight = 0.0

    for weight, u, v in edges:
        if uf.union(u, v):  # 使用并查集检查是否成环
            mst_edges.append((u, v, weight))
            total_weight += weight
            if len(mst_edges) == graph.n - 1:
                break

    return mst_edges, total_weight


# === 测试代码 ===

if __name__ == "__main__":
    print("=" * 60)
    print("示例 1：小型社交网络——BFS 与 DFS 遍历")
    print("=" * 60)

    g = Graph(6)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 5)

    print(f"\n图结构: {g}")
    print(f"\n邻接矩阵:\n{g.adjacency_matrix().astype(int)}")

    for node in range(g.n):
        print(f"  节点 {node}: 度={g.degree(node)}, 邻居={g.neighbors(node)}")

    bfs_order, bfs_dist = bfs(g, 0)
    print(f"\nBFS 从节点 0 开始:")
    print(f"  访问顺序: {bfs_order}")
    print(f"  距离:     {bfs_dist}")

    dfs_order = dfs(g, 0)
    print(f"\nDFS 从节点 0 开始:")
    print(f"  访问顺序: {dfs_order}")

    print("\n" + "=" * 60)
    print("示例 2：拉普拉斯矩阵与连通分量")
    print("=" * 60)

    # 构造一个包含 3 个连通分量的图
    g2 = Graph(7)
    g2.add_edge(0, 1)
    g2.add_edge(1, 2)
    g2.add_edge(0, 2)   # 分量 1: {0,1,2}
    g2.add_edge(3, 4)   # 分量 2: {3,4}
    g2.add_edge(5, 6)   # 分量 3: {5,6}

    print(f"\n图结构: {g2}")
    print(f"连通分量: {connected_components(g2)}")

    L = g2.laplacian()
    eigenvalues = laplacian_eigenvalues(g2)
    print(f"\n拉普拉斯矩阵:\n{L.astype(int)}")
    print(f"\n特征值: {np.round(eigenvalues, 4)}")

    n_zeros = np.sum(np.abs(eigenvalues) < 1e-8)
    print(f"零特征值个数: {n_zeros}")
    print(f"连通分量数: {len(connected_components(g2))}")
    print(f"匹配: {n_zeros == len(connected_components(g2))}")

    print("\n" + "=" * 60)
    print("示例 3：GNN 消息传递")
    print("=" * 60)

    # 5 个节点的小型图
    g3 = Graph(5)
    g3.add_edge(0, 1)
    g3.add_edge(0, 2)
    g3.add_edge(1, 2)
    g3.add_edge(2, 3)
    g3.add_edge(3, 4)

    rng = np.random.RandomState(42)
    # 5 个节点，每个节点 3 维特征
    features = rng.randn(5, 3)
    # 权重矩阵: 3 -> 2
    W1 = rng.randn(3, 2) * 0.5

    print(f"\n图结构: {g3}")
    print(f"\n初始节点特征 (5 个节点, 每个 3 维):")
    for i in range(5):
        print(f"  节点 {i}: {np.round(features[i], 4)}")

    # 第 1 轮消息传递：每个节点获得 1 跳邻居信息
    output1 = message_passing(g3, features, W1)
    print(f"\n第 1 轮消息传递后 (输出维度 2):")
    for i in range(5):
        print(f"  节点 {i}: {np.round(output1[i], 4)}")

    # 第 2 轮：每个节点获得 2 跳邻居信息
    W2 = rng.randn(2, 2) * 0.5
    output2 = message_passing(g3, output1, W2)
    print(f"\n第 2 轮消息传递后 (2 跳邻居信息):")
    for i in range(5):
        print(f"  节点 {i}: {np.round(output2[i], 4)}")

    print("\n" + "=" * 60)
    print("示例 4：谱聚类与 PageRank")
    print("=" * 60)

    # 构造两个团（clique），中间用一条边连接
    g4 = Graph(10)
    for i in range(5):
        for j in range(i + 1, 5):
            g4.add_edge(i, j)
    for i in range(5, 10):
        for j in range(i + 1, 10):
            g4.add_edge(i, j)
    g4.add_edge(2, 7)  # 桥接边

    print(f"\n图结构: 两个团 (0-4 和 5-9)，通过边 2-7 连接")

    labels = spectral_clustering(g4, k=2)
    print(f"\n谱聚类标签: {labels}")
    print(f"聚类 0: {np.where(labels == 0)[0]}")
    print(f"聚类 1: {np.where(labels == 1)[0]}")

    # Fiedler 值
    L4 = g4.laplacian()
    eigenvalues4 = laplacian_eigenvalues(g4)
    print(f"\n拉普拉斯特征值: {np.round(eigenvalues4, 4)}")
    print(f"Fiedler 值（连通性度量）: {eigenvalues4[1]:.4f}")

    # PageRank
    scores = pagerank(g4)
    print(f"\nPageRank 分数:")
    for i in range(g4.n):
        print(f"  节点 {i}: {scores[i]:.4f}")

    bridge_nodes = [2, 7]
    non_bridge = [n for n in range(g4.n) if n not in bridge_nodes]
    print(f"\n桥接节点 {bridge_nodes} 平均 PageRank: {np.mean(scores[bridge_nodes]):.4f}")
    print(f"非桥接节点平均 PageRank: {np.mean(scores[non_bridge]):.4f}")
    print("桥接节点 PageRank 更高——它们是连接社区的关键节点。")

    print("\n" + "=" * 60)
    print("示例 5：Dijkstra 最短路径")
    print("=" * 60)

    g5 = Graph(6, directed=True)
    g5.add_edge(0, 1, 7.0)
    g5.add_edge(0, 2, 9.0)
    g5.add_edge(0, 5, 14.0)
    g5.add_edge(1, 2, 10.0)
    g5.add_edge(1, 3, 15.0)
    g5.add_edge(2, 3, 11.0)
    g5.add_edge(2, 5, 2.0)
    g5.add_edge(3, 4, 6.0)
    g5.add_edge(4, 5, 9.0)

    distances, predecessors = dijkstra(g5, 0)
    print(f"\n从节点 0 到各点的最短距离:")
    for node in range(g5.n):
        print(f"  节点 {node}: {distances[node]:.1f}")

    print("\n" + "=" * 60)
    print("示例 6：最小生成树 (Kruskal)")
    print("=" * 60)

    g6 = Graph(4)
    g6.add_edge(0, 1, 1.0)
    g6.add_edge(0, 2, 4.0)
    g6.add_edge(0, 3, 3.0)
    g6.add_edge(1, 3, 2.0)
    g6.add_edge(2, 3, 5.0)

    mst_edges, total = kruskal_mst(g6)
    print(f"\n最小生成树边:")
    for u, v, w in mst_edges:
        print(f"  {u} - {v} (权重 {w})")
    print(f"总权重: {total}")
