# main.py — 集成学习从零实现
# 依赖：numpy>=1.24, scikit-learn>=1.3
# 安装：pip install numpy scikit-learn
# 对应课程：第 02 阶段 · 11（集成学习）

import numpy as np
from collections import Counter


# ============================================================
# 工具函数
# ============================================================

def make_classification_data(n_samples=300, n_features=5, noise=0.1, seed=42):
    """生成二分类数据集，标签为 +1 / -1。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # 非线性边界：包含平方项，线性模型难以拟合
    boundary = 0.5 * X[:, 0] + 0.3 * X[:, 1] ** 2 - 0.2 * X[:, 2]
    y = np.where(boundary + rng.normal(0, noise, n_samples) > 0, 1, -1)
    return X, y


def make_regression_data(n_samples=300, n_features=5, noise=0.3, seed=42):
    """生成回归数据集。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = 2.0 * X[:, 0] + np.sin(3 * X[:, 1]) - 0.5 * X[:, 2] ** 2
    y += rng.normal(0, noise, n_samples)
    return X, y


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """将数据集随机划分为训练集和测试集。"""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    split = int(len(y) * (1 - test_ratio))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


# ============================================================
# 第 1 步：决策树桩（Decision Stump）—— AdaBoost 的基学习器
# ============================================================

class DecisionStump:
    """单层决策树：仅在一个特征的一个阈值上做一次分裂。

    这是最弱的分类器——比随机猜好一点，但远不够单独使用。
    AdaBoost 通过组合大量这样的弱分类器来构建强分类器。
    """

    def __init__(self):
        self.feature_idx = None  # 用于分裂的特征索引
        self.threshold = None    # 分裂阈值
        self.polarity = 1        # 分类方向（1 或 -1）
        self.alpha = None        # 该分类器在集成中的权重

    def fit(self, X, y, weights):
        """在加权数据上训练决策树桩。

        遍历所有特征 × 所有阈值 × 两个方向，选择加权错误率最低的分裂。
        """
        n_samples, n_features = X.shape
        best_error = float("inf")

        for f in range(n_features):
            thresholds = np.unique(X[:, f])
            for thresh in thresholds:
                for polarity in [1, -1]:
                    # polarity=1: 特征值 < 阈值 → 预测 -1
                    # polarity=-1: 特征值 >= 阈值 → 预测 -1
                    pred = np.ones(n_samples)
                    pred[polarity * X[:, f] < polarity * thresh] = -1
                    error = np.sum(weights[pred != y])
                    if error < best_error:
                        best_error = error
                        self.feature_idx = f
                        self.threshold = thresh
                        self.polarity = polarity

    def predict(self, X):
        """预测：根据学习到的分裂规则输出 +1 或 -1。"""
        n = X.shape[0]
        pred = np.ones(n)
        idx = self.polarity * X[:, self.feature_idx] < self.polarity * self.threshold
        pred[idx] = -1
        return pred


# ============================================================
# 第 2 步：AdaBoost 从零实现
# ============================================================

class AdaBoostScratch:
    """AdaBoost（自适应提升）的从零实现。

    核心思想：串行训练多个弱分类器，每个新分类器聚焦于前一个分错的样本。
    被分错的样本权重增加，分对的样本权重降低。
    最终预测是所有弱分类器的加权投票。
    """

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []   # 弱分类器列表
        self.alphas = []   # 每个弱分类器的权重

    def fit(self, X, y):
        """训练 AdaBoost。

        算法步骤：
        1. 初始化所有样本权重为 1/N
        2. 对每一轮 t：
           a. 在加权数据上训练弱分类器
           b. 计算加权错误率 err_t
           c. 计算分类器权重 alpha_t = 0.5 * ln((1-err_t)/err_t)
           d. 更新样本权重：分错的乘以 exp(alpha_t)，分对的乘以 exp(-alpha_t)
           e. 归一化权重
        """
        n = X.shape[0]
        weights = np.full(n, 1.0 / n)

        for t in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            pred = stump.predict(X)

            # 加权错误率
            err = np.sum(weights[pred != y])
            err = np.clip(err, 1e-10, 1 - 1e-10)  # 防止 log(0)

            # 分类器权重：错误率越低，权重越大
            alpha = 0.5 * np.log((1 - err) / err)

            # 更新样本权重：分错的样本权重增加
            weights *= np.exp(-alpha * y * pred)
            weights /= weights.sum()

            stump.alpha = alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        """加权投票预测。"""
        total = sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))
        return np.sign(total)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# ============================================================
# 第 3 步：简单回归树（梯度提升的基学习器）
# ============================================================

class TreeNode:
    """回归树的节点。"""

    def __init__(self, value=None):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = value  # 叶节点的预测值


class SimpleRegressionTree:
    """用于梯度提升的简单回归树。

    使用方差减少作为分裂准则，支持最大深度限制。
    """

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)

    def _build(self, X, y, depth):
        """递归构建回归树。"""
        n_samples, n_features = X.shape

        # 停止条件：达到最大深度或样本数不足
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return TreeNode(value=np.mean(y))

        best_gain = -float("inf")
        best_feature = None
        best_threshold = None

        # 当前节点的方差（乘以样本数，避免重复计算）
        current_var = np.var(y) * n_samples

        for f in range(n_features):
            thresholds = np.unique(X[:, f])
            # 特征值过多时，用分位数采样加速
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, f], np.linspace(0, 100, 20))

            for thresh in thresholds:
                left_mask = X[:, f] <= thresh
                right_mask = ~left_mask

                if left_mask.sum() < 1 or right_mask.sum() < 1:
                    continue

                # 方差减少 = 父节点方差 - 子节点方差之和
                left_var = np.var(y[left_mask]) * left_mask.sum()
                right_var = np.var(y[right_mask]) * right_mask.sum()
                gain = current_var - left_var - right_var

                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = thresh

        # 无法找到有效分裂
        if best_feature is None or best_gain <= 0:
            return TreeNode(value=np.mean(y))

        left_mask = X[:, best_feature] <= best_threshold
        node = TreeNode()
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)


# ============================================================
# 第 4 步：梯度提升从零实现
# ============================================================

class GradientBoostingScratch:
    """梯度提升回归的从零实现。

    核心思想：串行训练多个回归树，每棵树拟合当前集成的残差（负梯度）。
    最终预测 = 初始值 + lr * 树1 + lr * 树2 + ...

    对于平方损失，伪残差就是普通残差：r_i = y_i - F(x_i)
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        """训练梯度提升模型。

        算法步骤：
        1. 初始化 F_0(x) = mean(y)
        2. 对每一轮 t：
           a. 计算残差 r_i = y_i - F_{t-1}(x_i)
           b. 训练回归树 h_t 拟合残差
           c. 更新 F_t(x) = F_{t-1}(x) + lr * h_t(x)
        """
        self.initial_pred = np.mean(y)
        current_pred = np.full(len(y), self.initial_pred)

        for _ in range(self.n_estimators):
            # 计算残差（平方损失下的伪残差）
            residuals = y - current_pred
            # 训练回归树拟合残差
            tree = SimpleRegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            # 更新预测
            update = tree.predict(X)
            current_pred += self.lr * update
            self.trees.append(tree)

    def predict(self, X):
        """预测：初始值 + 所有树的加权贡献。"""
        pred = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred

    def mse(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)


# ============================================================
# 第 5 步：Bagging 分类器
# ============================================================

class BaggingClassifier:
    """Bagging（Bootstrap Aggregating）分类器的从零实现。

    核心思想：对训练数据做有放回抽样，生成多个自助样本，
    在每个样本上训练一个模型，最终通过多数投票组合预测。

    每个自助样本平均包含 63.2% 的原始样本，剩余 36.8% 可作为袋外验证集。
    """

    def __init__(self, n_estimators=20, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """训练 Bagging 集成。"""
        rng = np.random.RandomState(42)
        n = len(y)

        for _ in range(self.n_estimators):
            # 有放回抽样：生成自助样本
            idx = rng.choice(n, size=n, replace=True)
            tree = SimpleRegressionTree(max_depth=self.max_depth)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        """多数投票：取所有树预测的符号平均值，再取符号。"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.sign(np.mean(predictions, axis=0))

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# ============================================================
# 第 6 步：Stacking 集成
# ============================================================

class StackingClassifier:
    """Stacking（堆叠）集成的从零实现。

    核心思想：
    1. 用 K 折交叉验证生成基学习器的预测作为元特征
    2. 用元特征训练一个元学习器（这里用简单的梯度下降逻辑回归）
    3. 最终用全量数据重新训练基学习器，用元学习器组合预测

    关键：基学习器的元特征必须通过交叉验证生成，避免数据泄露。
    """

    def __init__(self, base_models, meta_lr=0.1, n_folds=5):
        self.base_models = base_models  # 基学习器构造函数列表
        self.meta_lr = meta_lr
        self.n_folds = n_folds
        self.meta_weights = None
        self.meta_bias = None
        self.fitted_models = []

    def fit(self, X, y):
        """训练 Stacking 集成。"""
        n = len(y)
        meta_features = np.zeros((n, len(self.base_models)))

        # 第 1 阶段：K 折交叉验证生成元特征
        fold_size = n // self.n_folds
        indices = np.arange(n)

        for fold in range(self.n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            for m_idx, model_fn in enumerate(self.base_models):
                model = model_fn()
                model.fit(X[train_idx], y[train_idx])
                meta_features[val_idx, m_idx] = model.predict(X[val_idx])

        # 第 2 阶段：训练元学习器（梯度下降优化）
        self.meta_weights = np.zeros(len(self.base_models))
        self.meta_bias = 0.0

        for _ in range(200):
            logits = meta_features @ self.meta_weights + self.meta_bias
            preds = np.tanh(logits)  # 用 tanh 作为激活函数
            errors = y - preds
            grad_w = -2 * meta_features.T @ errors / n
            grad_b = -2 * np.sum(errors) / n
            self.meta_weights -= self.meta_lr * grad_w
            self.meta_bias -= self.meta_lr * grad_b

        # 第 3 阶段：用全量数据重新训练基学习器
        self.fitted_models = []
        for model_fn in self.base_models:
            model = model_fn()
            model.fit(X, y)
            self.fitted_models.append(model)

    def predict(self, X):
        """用元学习器组合基学习器的预测。"""
        meta_features = np.column_stack([m.predict(X) for m in self.fitted_models])
        logits = meta_features @ self.meta_weights + self.meta_bias
        return np.sign(logits)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# ============================================================
# 演示函数
# ============================================================

def demo_adaboost():
    print("=" * 60)
    print("演示 1：AdaBoost — 串行聚焦错误样本")
    print("=" * 60)

    X, y = make_classification_data(n_samples=400, n_features=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 展示随着弱分类器数量增加，准确率如何提升
    for n_est in [1, 5, 10, 25, 50]:
        model = AdaBoostScratch(n_estimators=n_est)
        model.fit(X_train, y_train)
        train_acc = model.accuracy(X_train, y_train)
        test_acc = model.accuracy(X_test, y_test)
        print(f"  弱分类器数={n_est:>3d}  训练准确率={train_acc:.3f}  测试准确率={test_acc:.3f}")

    print()

    # 与单个决策树桩对比
    stump = DecisionStump()
    stump.fit(X_train, y_train, np.full(len(y_train), 1.0 / len(y_train)))
    stump_acc = np.mean(stump.predict(X_test) == y_test)
    model_50 = AdaBoostScratch(n_estimators=50)
    model_50.fit(X_train, y_train)
    ada_acc = model_50.accuracy(X_test, y_test)
    print(f"  单个决策树桩准确率: {stump_acc:.3f}")
    print(f"  AdaBoost (50个)准确率: {ada_acc:.3f}")
    print(f"  提升幅度: {ada_acc - stump_acc:+.3f}")
    print()


def demo_gradient_boosting():
    print("=" * 60)
    print("演示 2：梯度提升 — 串行拟合残差")
    print("=" * 60)

    X, y = make_regression_data(n_samples=400, n_features=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    for n_est in [1, 10, 50, 100, 200]:
        model = GradientBoostingScratch(n_estimators=n_est, learning_rate=0.1)
        model.fit(X_train, y_train)
        train_mse = model.mse(X_train, y_train)
        test_mse = model.mse(X_test, y_test)
        print(f"  树数量={n_est:>3d}  训练MSE={train_mse:.4f}  测试MSE={test_mse:.4f}")

    print()

    # 与单棵回归树对比
    single_tree = SimpleRegressionTree(max_depth=3)
    single_tree.fit(X_train, y_train)
    tree_mse = np.mean((single_tree.predict(X_test) - y_test) ** 2)
    model_final = GradientBoostingScratch(n_estimators=100, learning_rate=0.1)
    model_final.fit(X_train, y_train)
    gbm_mse = model_final.mse(X_test, y_test)
    print(f"  单棵回归树 MSE: {tree_mse:.4f}")
    print(f"  梯度提升 (100棵) MSE: {gbm_mse:.4f}")
    print()


def demo_learning_rate_effect():
    print("=" * 60)
    print("演示 3：学习率与树数量的权衡")
    print("=" * 60)

    X, y = make_regression_data(n_samples=400)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    configs = [
        (0.5, 20),   # 大学习率，少树
        (0.1, 100),  # 中等学习率
        (0.05, 200), # 小学习率，多树
        (0.01, 500), # 很小学习率，很多树
    ]

    for lr, n_est in configs:
        model = GradientBoostingScratch(n_estimators=n_est, learning_rate=lr)
        model.fit(X_train, y_train)
        test_mse = model.mse(X_test, y_test)
        print(f"  学习率={lr:.2f}, 树数量={n_est:>3d}  测试MSE={test_mse:.4f}")

    print()
    print("规律：学习率越小，需要越多树，但通常泛化越好。")
    print()


def demo_bagging():
    print("=" * 60)
    print("演示 4：Bagging — 并行降低方差")
    print("=" * 60)

    X, y = make_classification_data(n_samples=400)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 单棵深度为 5 的树（容易过拟合）
    single_tree = SimpleRegressionTree(max_depth=5)
    single_tree.fit(X_train, y_train)
    single_acc = np.mean(np.sign(single_tree.predict(X_test)) == y_test)

    # Bagging 集成 20 棵树
    bagging = BaggingClassifier(n_estimators=20, max_depth=5)
    bagging.fit(X_train, y_train)
    bag_acc = bagging.accuracy(X_test, y_test)

    print(f"  单棵决策树 (深度=5) 准确率: {single_acc:.3f}")
    print(f"  Bagging (20棵树) 准确率: {bag_acc:.3f}")
    print(f"  方差降低带来的提升: {bag_acc - single_acc:+.3f}")
    print()


def demo_stacking():
    print("=" * 60)
    print("演示 5：Stacking — 元学习器组合异质模型")
    print("=" * 60)

    X, y = make_classification_data(n_samples=400)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 定义不同深度的回归树作为基学习器
    class TreeWrapper:
        def __init__(self, max_depth):
            self.max_depth = max_depth
            self.tree = None

        def fit(self, X, y):
            self.tree = SimpleRegressionTree(max_depth=self.max_depth)
            self.tree.fit(X, y)

        def predict(self, X):
            return np.sign(self.tree.predict(X))

    base_models = [
        lambda: TreeWrapper(3),
        lambda: TreeWrapper(5),
        lambda: TreeWrapper(7),
    ]

    # 训练 Stacking
    stack = StackingClassifier(base_models=base_models, meta_lr=0.05)
    stack.fit(X_train, y_train)

    # 展示每个基学习器的独立表现
    for depth, model_fn in zip([3, 5, 7], base_models):
        m = model_fn()
        m.fit(X_train, y_train)
        acc = np.mean(m.predict(X_test) == y_test)
        print(f"  决策树 (深度={depth}) 准确率: {acc:.3f}")

    stack_acc = stack.accuracy(X_test, y_test)
    print(f"  Stacking 集成准确率: {stack_acc:.3f}")
    print(f"  元学习器权重: {np.round(stack.meta_weights, 3)}")
    print()


def demo_full_comparison():
    print("=" * 60)
    print("演示 6：全方法对比")
    print("=" * 60)

    X, y = make_classification_data(n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 单棵决策树
    single = SimpleRegressionTree(max_depth=5)
    single.fit(X_train, y_train)
    print(f"  单棵决策树 (深度=5):  {np.mean(np.sign(single.predict(X_test)) == y_test):.3f}")

    # Bagging
    bag = BaggingClassifier(n_estimators=20, max_depth=5)
    bag.fit(X_train, y_train)
    print(f"  Bagging (20棵树):     {bag.accuracy(X_test, y_test):.3f}")

    # AdaBoost
    ada = AdaBoostScratch(n_estimators=50)
    ada.fit(X_train, y_train)
    print(f"  AdaBoost (50个树桩):  {ada.accuracy(X_test, y_test):.3f}")

    print()
    print("Bagging 降低方差（比单棵树更稳定）。")
    print("Boosting 降低偏差（从弱分类器学到复杂边界）。")
    print()


def demo_sklearn_comparison():
    print("=" * 60)
    print("演示 7：与 scikit-learn 对比验证")
    print("=" * 60)

    try:
        from sklearn.ensemble import (
            AdaBoostClassifier,
            GradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("  scikit-learn 未安装，跳过对比。")
        print()
        return

    X, y = make_classification_data(n_samples=500)
    # sklearn 的 AdaBoost 需要 0/1 标签
    y_01 = (y + 1) // 2
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    _, _, y_train_01, y_test_01 = train_test_split(X, y_01)

    # 我们的 AdaBoost
    ada_ours = AdaBoostScratch(n_estimators=50)
    ada_ours.fit(X_train, y_train)
    print(f"  我们的 AdaBoost:      {ada_ours.accuracy(X_test, y_test):.3f}")

    # sklearn 的 AdaBoost
    ada_sk = AdaBoostClassifier(n_estimators=50, random_state=42, algorithm="SAMME")
    ada_sk.fit(X_train, y_train_01)
    print(f"  sklearn AdaBoost:     {accuracy_score(y_test_01, ada_sk.predict(X_test)):.3f}")

    # sklearn 随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train_01)
    print(f"  sklearn 随机森林:     {accuracy_score(y_test_01, rf.predict(X_test)):.3f}")

    # sklearn 梯度提升
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train_01)
    print(f"  sklearn 梯度提升:     {accuracy_score(y_test_01, gb.predict(X_test)):.3f}")

    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    demo_adaboost()
    demo_gradient_boosting()
    demo_learning_rate_effect()
    demo_bagging()
    demo_stacking()
    demo_full_comparison()
    demo_sklearn_comparison()
    print("所有集成学习演示完成。")
