# main.py — 最近质心分类器从零实现
# 依赖：numpy>=1.24
# 安装：pip install numpy
# 对应课程：第 02 阶段 · 01（什么是机器学习）

import numpy as np


class NearestCentroid:
    """最近质心分类器——最简单的机器学习算法之一。

    核心思想：对每个类计算训练样本的均值向量（质心），
    预测时将样本分配到距离最近的质心所属的类。
    """

    def __init__(self):
        self.classes = None
        self.centroids = None

    def fit(self, X, y):
        """训练：计算每个类的质心（均值向量）。

        Args:
            X: 训练数据，形状 (n_samples, n_features)
            y: 标签，形状 (n_samples,)
        """
        self.classes = np.unique(y)
        # 对每个类，计算该类所有样本在每个特征上的均值
        self.centroids = np.array([
            X[y == c].mean(axis=0) for c in self.classes
        ])

    def predict(self, X):
        """预测：将每个样本分配到最近的质心。

        Args:
            X: 测试数据，形状 (n_samples, n_features)

        Returns:
            预测标签，形状 (n_samples,)
        """
        # 计算每个样本到每个质心的欧氏距离
        distances = np.array([
            np.sqrt(((X - c) ** 2).sum(axis=1))
            for c in self.centroids
        ])
        # 选择距离最小的类
        return self.classes[distances.argmin(axis=0)]

    def score(self, X, y):
        """计算准确率。

        Args:
            X: 测试数据
            y: 真实标签

        Returns:
            准确率（0-1之间）
        """
        return np.mean(self.predict(X) == y)


def generate_classification_data(n_per_class=100, n_features=2, separation=2.0, seed=42):
    """生成二分类数据集。

    Args:
        n_per_class: 每类样本数
        n_features: 特征维度
        separation: 两类中心之间的距离，越大越容易分开
        seed: 随机种子

    Returns:
        X: 数据，形状 (2*n_per_class, n_features)
        y: 标签，形状 (2*n_per_class,)
    """
    rng = np.random.RandomState(seed)
    # 两类中心分别位于 separation/2 和 -separation/2
    center_0 = np.ones(n_features) * (separation / 2)
    center_1 = np.ones(n_features) * (-separation / 2)
    X_class0 = rng.randn(n_per_class, n_features) + center_0
    X_class1 = rng.randn(n_per_class, n_features) + center_1
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


def train_test_split(X, y, test_fraction=0.3, seed=42):
    """将数据随机划分为训练集和测试集。

    Args:
        X: 数据
        y: 标签
        test_fraction: 测试集比例
        seed: 随机种子

    Returns:
        X_train, X_test, y_train, y_test
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * (1 - test_fraction))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


def random_baseline(y_train, y_test, seed=42):
    """随机猜测基线：按训练集类别比例随机预测。

    Args:
        y_train: 训练集标签
        y_test: 测试集标签
        seed: 随机种子

    Returns:
        随机猜测准确率
    """
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y_train, return_counts=True)
    probs = counts / counts.sum()
    preds = rng.choice(classes, size=len(y_test), p=probs)
    return np.mean(preds == y_test)


def majority_baseline(y_train, y_test):
    """多数类基线：总是预测训练集中最常见的类。

    Args:
        y_train: 训练集标签
        y_test: 测试集标签

    Returns:
        多数类预测准确率
    """
    values, counts = np.unique(y_train, return_counts=True)
    majority_class = values[np.argmax(counts)]
    preds = np.full(len(y_test), majority_class)
    return np.mean(preds == y_test)


def demo_nearest_centroid():
    """演示最近质心分类器的基本用法。"""
    print("=" * 60)
    print("最近质心分类器 —— 从零实现")
    print("=" * 60)
    print()

    # 生成数据：两类，每类 150 个样本
    X, y = generate_classification_data(n_per_class=150, separation=2.0)
    X_train, X_test, y_train, y_train_full = train_test_split(X, y)

    print(f"数据集: {len(y)} 个样本, {X.shape[1]} 个特征, 2 个类")
    print(f"训练集: {len(y_train)} 个样本, 测试集: {len(y_train_full)} 个样本")
    print()

    # 训练模型
    clf = NearestCentroid()
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_train_full)

    print(f"各类质心:")
    for i, c in enumerate(clf.classes):
        print(f"  类 {c}: [{clf.centroids[i][0]:.3f}, {clf.centroids[i][1]:.3f}]")
    print()

    # 与基线对比
    print(f"{'方法':<25} {'训练准确率':>10} {'测试准确率':>10}")
    print("-" * 50)
    print(f"{'最近质心分类器':<25} {train_acc:>10.3f} {test_acc:>10.3f}")

    rand_acc = random_baseline(y_train, y_train_full)
    print(f"{'随机基线':<25} {'--':>10} {rand_acc:>10.3f}")

    maj_acc = majority_baseline(y_train, y_train_full)
    print(f"{'多数类基线':<25} {'--':>10} {maj_acc:>10.3f}")

    print()
    improvement_over_random = (test_acc - rand_acc) / rand_acc * 100
    print(f"最近质心分类器比随机基线好 {improvement_over_random:.1f}%")


def demo_varying_difficulty():
    """演示不同类别分离度对准确率的影响。"""
    print()
    print("=" * 60)
    print("类别分离度对准确率的影响")
    print("=" * 60)
    print()

    separations = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    print(f"{'分离度':>12} {'训练准确率':>10} {'测试准确率':>10} {'随机基线':>10}")
    print("-" * 50)

    for sep in separations:
        X, y = generate_classification_data(n_per_class=150, separation=sep)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        clf = NearestCentroid()
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        rand_acc = random_baseline(y_train, y_test)

        print(f"{sep:>12.1f} {train_acc:>10.3f} {test_acc:>10.3f} {rand_acc:>10.3f}")

    print()
    print("分离度小：类重叠严重，准确率下降。")
    print("分离度大：类相距远，即使简单模型也能表现优异。")


def demo_higher_dimensions():
    """演示特征维度对分类性能的影响。"""
    print()
    print("=" * 60)
    print("最近质心分类器在高维空间中的表现")
    print("=" * 60)
    print()

    dimensions = [2, 5, 10, 20, 50]

    print(f"{'特征数':>10} {'测试准确率':>10}")
    print("-" * 25)

    for d in dimensions:
        X, y = generate_classification_data(
            n_per_class=200, n_features=d, separation=2.0
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        clf = NearestCentroid()
        clf.fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        print(f"{d:>10d} {test_acc:>10.3f}")

    print()
    print("对于高斯数据，固定分离度下，维度增加有助于区分。")
    print("质心在高维空间中更容易分开。")
    print("实际数据不同——当大量特征是噪声时，会出现维度灾难。")


def demo_multiclass():
    """演示多分类场景（3 个类）。"""
    print()
    print("=" * 60)
    print("多分类最近质心（3 个类）")
    print("=" * 60)
    print()

    rng = np.random.RandomState(42)
    n_per_class = 100
    # 三个类的中心构成等边三角形
    centers = np.array([[2, 0], [-1, 1.7], [-1, -1.7]])
    X_parts = [rng.randn(n_per_class, 2) * 0.8 + c for c in centers]
    X = np.vstack(X_parts)
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    shuffle_idx = rng.permutation(len(y))
    X, y = X[shuffle_idx], y[shuffle_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = NearestCentroid()
    clf.fit(X_train, y_train)

    print(f"三分类问题: {len(y)} 个样本")
    print(f"各类质心:")
    for i, c in enumerate(clf.classes):
        print(f"  类 {c}: [{clf.centroids[i][0]:.3f}, {clf.centroids[i][1]:.3f}]")
    print()
    print(f"测试准确率: {clf.score(X_test, y_test):.3f}")
    print(f"随机基线 (1/3): {random_baseline(y_train, y_test):.3f}")


def demo_overfitting_signal():
    """演示过拟合信号：训练准确率远高于验证准确率。"""
    print()
    print("=" * 60)
    print("过拟合信号演示")
    print("=" * 60)
    print()

    # 生成两类高度重叠的数据
    X, y = generate_classification_data(n_per_class=50, separation=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = NearestCentroid()
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"数据: 50 个样本/类, 分离度 0.5 (高度重叠)")
    print(f"训练准确率: {train_acc:.3f}")
    print(f"测试准确率: {test_acc:.3f}")
    print(f"差距: {train_acc - test_acc:.3f}")
    print()
    print("当训练准确率明显高于测试准确率时，")
    print("需要警惕过拟合风险（更复杂模型中更明显）。")


def demo_bias_variance_intuition():
    """用不同复杂度的模型演示偏差-方差权衡的直觉。"""
    print()
    print("=" * 60)
    print("偏差-方差权衡直觉")
    print("=" * 60)
    print()

    rng = np.random.RandomState(42)
    n_samples = 100
    x = np.linspace(0, 10, n_samples)
    # 真实关系：正弦曲线 + 噪声
    y_true = np.sin(x) + rng.randn(n_samples) * 0.3

    print(f"{'模型复杂度':<20} {'训练误差':>10} {'测试误差':>10}")
    print("-" * 45)

    # 简单模型：常数预测（高偏差）
    train_pred_simple = np.full(n_samples, y_true.mean())
    train_err_simple = np.mean((train_pred_simple - y_true) ** 2)

    # 中等模型：线性预测
    coeffs = np.polyfit(x, y_true, 1)
    train_pred_medium = np.polyval(coeffs, x)
    train_err_medium = np.mean((train_pred_medium - y_true) ** 2)

    # 复杂模型：高次多项式（高方差）
    coeffs_high = np.polyfit(x, y_true, 15)
    train_pred_complex = np.polyval(coeffs_high, x)
    train_err_complex = np.mean((train_pred_complex - y_true) ** 2)

    print(f"{'高偏差（常数）':<20} {train_err_simple:>10.3f} {'--':>10}")
    print(f"{'中等（线性）':<20} {train_err_medium:>10.3f} {'--':>10}")
    print(f"{'高方差（15次）':<20} {train_err_complex:>10.3f} {'--':>10}")
    print()
    print("高偏差模型训练误差也高（欠拟合）。")
    print("高方差模型训练误差极低但泛化差（过拟合）。")


if __name__ == "__main__":
    # 运行所有演示
    demo_nearest_centroid()
    demo_varying_difficulty()
    demo_higher_dimensions()
    demo_multiclass()
    demo_overfitting_signal()
    demo_bias_variance_intuition()
    print()
    print("所有演示完成。")
