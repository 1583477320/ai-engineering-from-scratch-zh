# main.py — 从零实现决策树与随机森林
# 依赖：无（仅使用 Python 标准库）
# 对应课程：阶段 02 · 04（决策树）
# 本代码为教学实现，不引入任何第三方库

import math
import random


# =============================================================================
# 第 1 步：不纯度度量
# =============================================================================

def gini_impurity(labels):
    """计算基尼不纯度。

    基尼不纯度衡量从节点中随机抽取两个样本，它们标签不一致的概率。
    公式：Gini = 1 - sum(p_k^2)
    其中 p_k 是第 k 类样本所占的比例。
    纯节点 = 0，二分类 50/50 混合 = 0.5。
    """
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def entropy(labels):
    """计算信息熵。

    信息熵衡量节点的混乱程度，来自信息论。
    公式：H = -sum(p_k * log2(p_k))
    纯节点 = 0，二分类 50/50 混合 = 1.0 bit。
    """
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum(
        (c / n) * math.log2(c / n) for c in counts.values() if c > 0
    )


def information_gain(parent_labels, left_labels, right_labels, criterion="gini"):
    """计算信息增益。

    信息增益 = 父节点不纯度 - 加权平均的子节点不纯度。
    使用基尼不纯度或信息熵作为度量标准。
    """
    measure = gini_impurity if criterion == "gini" else entropy
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = measure(parent_labels)
    child_impurity = (
        (n_left / n) * measure(left_labels)
        + (n_right / n) * measure(right_labels)
    )
    return parent_impurity - child_impurity


def variance_reduction(parent_values, left_values, right_values):
    """计算方差缩减（用于回归树）。

    方差缩减是信息增益在回归问题中的对应版本。
    选择使目标变量方差缩减最大的切分方式。
    """
    if len(left_values) == 0 or len(right_values) == 0:
        return 0.0
    n = len(parent_values)
    parent_var = _variance(parent_values)
    child_var = (
        (len(left_values) / n) * _variance(left_values)
        + (len(right_values) / n) * _variance(right_values)
    )
    return parent_var - child_var


def _variance(values):
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def _mean(values):
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def majority_vote(labels):
    """多数投票：返回出现次数最多的标签。"""
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


# =============================================================================
# 第 2 步：决策树
# =============================================================================

class DecisionTree:
    """决策树分类器/回归树的从零实现。

    支持基尼不纯度和信息熵两种切分标准，
    通过预剪枝参数控制树的生长。

    Args:
        max_depth: 树的最大深度（None 表示不限制）
        min_samples_split: 节点分裂所需的最小样本数
        min_samples_leaf: 叶节点所需的最小样本数
        criterion: 切分标准，"gini" 或 "entropy"
        max_features: 每次分裂时考虑的特征数
            - None: 考虑所有特征
            - "sqrt": 考虑 sqrt(n_features) 个特征
            - int: 考虑指定数量的特征
        task: "classification" 或 "regression"
    """

    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini",
                 max_features=None, task="classification"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.task = task
        self.tree = None
        self.feature_importances_ = None
        self.n_features = 0
        self.n_samples = 0

    def fit(self, X, y):
        """训练决策树。

        Args:
            X: 样本特征列表，每个样本是一个特征值列表
            y: 标签列表
        """
        self.n_features = len(X[0])
        self.feature_importances_ = [0.0] * self.n_features
        self.n_samples = len(X)
        self.tree = self._build(X, y, depth=0)
        # 归一化特征重要性
        total = sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = [
                fi / total for fi in self.feature_importances_
            ]

    def predict(self, X):
        """对样本列表进行预测。"""
        return [self._predict_one(x, self.tree) for x in X]

    def _build(self, X, y, depth):
        """递归构建决策树。

        在每个节点：
        1. 检查是否满足停止条件（纯度达标、深度过大、样本不足）
        2. 寻找最优切分（特征和阈值）
        3. 递归构建左右子树
        """
        # 所有样本标签相同，无需继续分裂
        if len(set(y)) == 1:
            return self._make_leaf(y)

        # 预剪枝：达到最大深度
        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(y)

        # 预剪枝：样本数不足以分裂
        if len(y) < self.min_samples_split:
            return self._make_leaf(y)

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        # 找不到有效的切分
        if best_feature is None or best_gain <= 0:
            return self._make_leaf(y)

        # 执行切分
        left_X, left_y, right_X, right_y = self._split_data(
            X, y, best_feature, best_threshold
        )

        # 预剪枝：分裂后叶节点样本不足
        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            return self._make_leaf(y)

        # 更新特征重要性（加权不纯度减少量）
        weight = len(y) / self.n_samples
        self.feature_importances_[best_feature] += weight * best_gain

        # 递归构建子树
        left_child = self._build(left_X, left_y, depth + 1)
        right_child = self._build(right_X, right_y, depth + 1)

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_child,
            "right": right_child,
        }

    def _make_leaf(self, y):
        """创建叶节点。分类任务返回多数类，回归任务返回均值。"""
        if self.task == "classification":
            return {"leaf": True, "value": majority_vote(y)}
        else:
            return {"leaf": True, "value": _mean(y)}

    def _best_split(self, X, y):
        """寻找最优切分点。

        遍历所有特征（或随机子集），对每个特征尝试所有可能的阈值，
        返回信息增益最大的（特征，阈值，增益）三元组。
        """
        best_feature = None
        best_threshold = None
        best_gain = -1.0

        # 根据 max_features 决定使用哪些特征
        if self.max_features is None:
            feature_indices = list(range(self.n_features))
        elif self.max_features == "sqrt":
            k = max(1, int(math.sqrt(self.n_features)))
            feature_indices = random.sample(range(self.n_features), k)
        elif isinstance(self.max_features, int):
            k = min(self.max_features, self.n_features)
            feature_indices = random.sample(range(self.n_features), k)
        else:
            feature_indices = list(range(self.n_features))

        for feature_idx in feature_indices:
            # 获取该特征的所有唯一取值并排序
            values = sorted(set(X[i][feature_idx] for i in range(len(X))))
            if len(values) <= 1:
                continue

            # 尝试相邻值的中点作为阈值
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2.0
                left_y = [y[j] for j in range(len(X)) if X[j][feature_idx] <= threshold]
                right_y = [y[j] for j in range(len(X)) if X[j][feature_idx] > threshold]

                # 检查叶节点最小样本数约束
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue

                if self.task == "classification":
                    gain = information_gain(y, left_y, right_y, self.criterion)
                else:
                    gain = variance_reduction(y, left_y, right_y)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _split_data(self, X, y, feature, threshold):
        """根据特征和阈值将数据集切分为左右两个子集。"""
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        return left_X, left_y, right_X, right_y

    def _predict_one(self, x, node):
        """对单个样本进行预测，从根节点走到叶节点。"""
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def print_tree(self, node=None, indent=""):
        """打印树的结构，便于理解和调试。"""
        if node is None:
            node = self.tree
        if node["leaf"]:
            print(f"{indent}预测: {node['value']}")
            return
        print(f"{indent}特征 {node['feature']} <= {node['threshold']:.4f}?")
        print(f"{indent}  是:")
        self.print_tree(node["left"], indent + "    ")
        print(f"{indent}  否:")
        self.print_tree(node["right"], indent + "    ")


# =============================================================================
# 第 3 步：随机森林
# =============================================================================

class RandomForest:
    """随机森林分类器/回归器。

    通过 Bagging（自举采样）和特征随机化构建多棵决策树，
    然后聚合它们的预测结果：
    - 分类任务：多数投票
    - 回归任务：取平均

    Args:
        n_trees: 森林中树的数量
        max_depth: 每棵树的最大深度
        min_samples_split: 节点分裂所需的最小样本数
        max_features: 每次分裂时考虑的特征数（默认 "sqrt"）
        criterion: 切分标准
        task: "classification" 或 "regression"
    """

    def __init__(self, n_trees=100, max_depth=None,
                 min_samples_split=2, max_features="sqrt",
                 criterion="gini", task="classification"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.task = task
        self.trees = []

    def fit(self, X, y):
        """训练随机森林。

        对每棵树：
        1. 从训练集做有放回自举采样（bootstrap sample）
        2. 在采样后的数据上训练一棵决策树
        由于每棵树看到的数据和特征都不同，森林中的树是多样化的。
        """
        self.trees = []
        n = len(X)
        for _ in range(self.n_trees):
            # 自举采样：有放回地随机抽取 n 个样本
            indices = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                criterion=self.criterion,
                task=self.task,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        """对样本列表进行预测，聚合所有树的输出。"""
        all_preds = [tree.predict(X) for tree in self.trees]
        predictions = []
        for i in range(len(X)):
            if self.task == "classification":
                # 多数投票
                votes = {}
                for preds in all_preds:
                    v = preds[i]
                    votes[v] = votes.get(v, 0) + 1
                predictions.append(max(votes, key=votes.get))
            else:
                # 取平均
                predictions.append(
                    sum(preds[i] for preds in all_preds) / len(all_preds)
                )
        return predictions

    def feature_importances(self):
        """聚合所有树的特征重要性。"""
        n_features = self.trees[0].n_features
        importances = [0.0] * n_features
        for tree in self.trees:
            for j in range(n_features):
                importances[j] += tree.feature_importances_[j]
        total = sum(importances)
        if total > 0:
            importances = [imp / total for imp in importances]
        return importances


# =============================================================================
# 辅助函数
# =============================================================================

def accuracy(y_true, y_pred):
    """计算分类准确率。"""
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def mse(y_true, y_pred):
    """计算均方误差（用于回归任务）。"""
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def generate_classification_data(n_samples=200, seed=42):
    """生成三分类数据集。

    生成二维特征、三标签的分类数据，用于演示决策树的决策边界。
    """
    random.seed(seed)
    X = []
    y = []
    for _ in range(n_samples):
        x1 = random.uniform(-3, 3)
        x2 = random.uniform(-3, 3)
        noise = random.gauss(0, 0.3)
        if x1 ** 2 + x2 ** 2 + noise < 3:
            label = 0
        elif x1 + x2 + noise > 1:
            label = 1
        else:
            label = 2
        X.append([x1, x2])
        y.append(label)
    return X, y


def generate_regression_data(n_samples=200, seed=42):
    """生成回归数据集。目标：y = sin(x) * x + 噪声。"""
    random.seed(seed)
    X = []
    y = []
    for _ in range(n_samples):
        x = random.uniform(-3, 3)
        target = math.sin(x) * x + random.gauss(0, 0.2)
        X.append([x])
        y.append(target)
    return X, y


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """将数据集随机切分为训练集和测试集。"""
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, y_train, X_test, y_test


# =============================================================================
# 演示函数
# =============================================================================

def demo_split_criteria():
    """演示不同分布下的基尼不纯度和信息熵。"""
    print("=" * 65)
    print("切分标准：基尼不纯度 vs 信息熵")
    print("=" * 65)
    print()

    test_cases = [
        ("纯节点 [A,A,A,A]", ["A", "A", "A", "A"]),
        ("均衡 [A,A,B,B]", ["A", "A", "B", "B"]),
        ("不均衡 [A,A,A,B]", ["A", "A", "A", "B"]),
        ("三类 [A,A,B,C]", ["A", "A", "B", "C"]),
        ("均匀四类", ["A", "B", "C", "D"]),
    ]

    print(f"  {'分布':<30s} {'基尼':>8s} {'信息熵':>8s}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8}")
    for name, labels in test_cases:
        g = gini_impurity(labels)
        e = entropy(labels)
        print(f"  {name:<30s} {g:>8.4f} {e:>8.4f}")

    print()
    print("  两种度量标准结论一致：纯节点=0，均衡分布=最大值。")
    print("  信息熵在多类情况下增长略快于基尼不纯度。")
    print()


def demo_information_gain():
    """演示信息增益如何评估切分质量。"""
    print("=" * 65)
    print("信息增益：选择最优切分")
    print("=" * 65)
    print()

    parent = ["猫", "猫", "猫", "猫", "狗", "狗", "狗",
              "鸟", "鸟", "鸟"]

    splits = [
        ("特征A: [猫,猫,猫,狗] | [猫,狗,狗,鸟,鸟,鸟]",
         ["猫", "猫", "猫", "狗"],
         ["猫", "狗", "狗", "鸟", "鸟", "鸟"]),
        ("特征B: [猫,猫,猫,猫] | [狗,狗,狗,鸟,鸟,鸟]",
         ["猫", "猫", "猫", "猫"],
         ["狗", "狗", "狗", "鸟", "鸟", "鸟"]),
        ("特征C: [猫,猫,狗,鸟] | [猫,猫,狗,狗,鸟,鸟]",
         ["猫", "猫", "狗", "鸟"],
         ["猫", "猫", "狗", "狗", "鸟", "鸟"]),
    ]

    print(f"  父节点: {parent}")
    print(f"  父节点基尼: {gini_impurity(parent):.4f}")
    print(f"  父节点信息熵: {entropy(parent):.4f}")
    print()

    print(f"  {'切分':<55s} {'IG(基尼)':>10s} {'IG(信息熵)':>12s}")
    print(f"  {'-' * 55} {'-' * 10} {'-' * 12}")

    for name, left, right in splits:
        ig_gini = information_gain(parent, left, right, "gini")
        ig_ent = information_gain(parent, left, right, "entropy")
        print(f"  {name:<55s} {ig_gini:>10.4f} {ig_ent:>12.4f}")

    print()
    print("  特征B完美分离了猫——信息增益最大。")
    print()


def demo_decision_tree():
    """演示不同深度决策树的训练效果与过拟合现象。"""
    print("=" * 65)
    print("决策树：分类")
    print("=" * 65)
    print()

    X, y = generate_classification_data(200, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    print(f"  数据集: {len(X)} 个样本, 2 个特征, 3 个类别")
    print(f"  训练集: {len(X_train)}  测试集: {len(X_test)}")
    print()

    depths = [1, 2, 3, 5, 10, None]
    print(f"  {'最大深度':>10s}  {'训练准确率':>10s}  {'测试准确率':>10s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for d in depths:
        tree = DecisionTree(max_depth=d, criterion="gini")
        tree.fit(X_train, y_train)
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        train_acc = accuracy(y_train, train_pred)
        test_acc = accuracy(y_test, test_pred)
        d_str = str(d) if d is not None else "无限制"
        print(f"  {d_str:>10s}  {train_acc:>10.4f}  {test_acc:>10.4f}")

    print()
    print("  浅树欠拟合，深树过拟合。最佳深度在中间某个位置。")
    print()

    # 打印深度为 3 的树结构
    tree = DecisionTree(max_depth=3, criterion="gini")
    tree.fit(X_train, y_train)
    print("  树结构（最大深度=3）:")
    tree.print_tree()
    print()


def demo_random_forest():
    """演示随机森林的集成效果。"""
    print("=" * 65)
    print("随机森林：集成的力量")
    print("=" * 65)
    print()

    random.seed(42)
    X, y = generate_classification_data(300, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    print(f"  数据集: {len(X)} 个样本, 2 个特征, 3 个类别")
    print(f"  训练集: {len(X_train)}  测试集: {len(X_test)}")
    print()

    tree_counts = [1, 3, 5, 10, 25, 50, 100]
    print(f"  {'树的数量':>8s}  {'训练准确率':>10s}  {'测试准确率':>10s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}")

    for n in tree_counts:
        rf = RandomForest(n_trees=n, max_depth=5, criterion="gini")
        rf.fit(X_train, y_train)
        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)
        train_acc = accuracy(y_train, train_pred)
        test_acc = accuracy(y_test, test_pred)
        print(f"  {n:>8d}  {train_acc:>10.4f}  {test_acc:>10.4f}")

    print()
    print("  更多树 = 更好的泛化能力（但存在收益递减）。")
    print("  测试准确率会趋于平稳但不会下降（森林的抗过拟合性）。")
    print()


def demo_feature_importance():
    """演示随机森林如何识别重要特征。"""
    print("=" * 65)
    print("特征重要性")
    print("=" * 65)
    print()

    random.seed(42)
    n = 200
    X = []
    y = []
    for _ in range(n):
        important1 = random.uniform(-2, 2)
        important2 = random.uniform(-2, 2)
        noise1 = random.gauss(0, 1)
        noise2 = random.gauss(0, 1)
        # 标签仅由前两个特征决定
        label = 1 if important1 + important2 > 0 else 0
        X.append([important1, important2, noise1, noise2])
        y.append(label)

    feature_names = ["重要特征_1", "重要特征_2", "噪声_1", "噪声_2"]

    rf = RandomForest(n_trees=50, max_depth=5)
    rf.fit(X, y)
    importances = rf.feature_importances()

    print(f"  目标: 如果 特征0 + 特征1 > 0 则为 1，否则为 0")
    print(f"  特征 2 和 3 是纯噪声。")
    print()

    print(f"  {'特征':<15s}  {'重要性':>12s}")
    print(f"  {'-' * 15}  {'-' * 12}")
    for name, imp in sorted(zip(feature_names, importances),
                            key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        print(f"  {name:<15s}  {imp:>12.4f}  {bar}")

    print()
    print("  森林正确地识别出了哪些特征是重要的。")
    print()


def demo_regression_tree():
    """演示回归树的分段常数逼近效果。"""
    print("=" * 65)
    print("回归树：分段常数逼近")
    print("=" * 65)
    print()

    X, y = generate_regression_data(200, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    depths = [1, 2, 3, 5, 10]
    print(f"  目标: y = sin(x) * x + 噪声")
    print(f"  训练集: {len(X_train)}  测试集: {len(X_test)}")
    print()

    print(f"  {'最大深度':>10s}  {'训练MSE':>10s}  {'测试MSE':>10s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for d in depths:
        tree = DecisionTree(max_depth=d, task="regression")
        tree.fit(X_train, y_train)
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        train_mse = mse(y_train, train_pred)
        test_mse = mse(y_test, test_pred)
        print(f"  {d:>10d}  {train_mse:>10.4f}  {test_mse:>10.4f}")

    print()

    # 随机森林回归效果
    rf = RandomForest(n_trees=50, max_depth=5, task="regression")
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mse = mse(y_test, rf_pred)
    print(f"  随机森林（50 棵树, 深度=5）测试 MSE: {rf_mse:.4f}")
    print()
    print("  森林对多个分段取平均，得到更平滑的输出。")
    print()


def demo_gini_vs_entropy():
    """对比基尼不纯度和信息熵作为切分标准的实际差异。"""
    print("=" * 65)
    print("基尼 vs 信息熵：它们会产生不同的结果吗？")
    print("=" * 65)
    print()

    random.seed(42)
    X, y = generate_classification_data(200, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    for depth in [3, 5, 10]:
        tree_gini = DecisionTree(max_depth=depth, criterion="gini")
        tree_entropy = DecisionTree(max_depth=depth, criterion="entropy")
        tree_gini.fit(X_train, y_train)
        tree_entropy.fit(X_train, y_train)

        acc_gini = accuracy(y_test, tree_gini.predict(X_test))
        acc_entropy = accuracy(y_test, tree_entropy.predict(X_test))

        print(f"  深度={depth:<4d}  基尼准确率: {acc_gini:.4f}  "
              f"信息熵准确率: {acc_entropy:.4f}  "
              f"差值: {abs(acc_gini - acc_entropy):.4f}")

    print()
    print("  实际中，基尼和信息熵生成的树几乎相同。")
    print("  基尼略快（无需计算对数）。")
    print()


def demo_single_tree_vs_forest():
    """对比单棵决策树与随机森林的稳定性。"""
    print("=" * 65)
    print("单棵树 vs 随机森林：稳定性对比")
    print("=" * 65)
    print()

    X, y = generate_classification_data(200, seed=42)

    print("  在略有不同的数据子集上训练 5 棵单树:")
    single_accs = []
    for trial in range(5):
        random.seed(trial * 10)
        indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
        X_sub = [X[i] for i in indices]
        y_sub = [y[i] for i in indices]
        X_tr, y_tr, X_te, y_te = train_test_split(X_sub, y_sub, seed=trial)
        tree = DecisionTree(max_depth=5)
        tree.fit(X_tr, y_tr)
        acc = accuracy(y_te, tree.predict(X_te))
        single_accs.append(acc)
        print(f"    试验 {trial + 1}: 准确率 = {acc:.4f}")

    print()
    print("  在相同数据子集上训练 5 个随机森林:")
    forest_accs = []
    for trial in range(5):
        random.seed(trial * 10)
        indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
        X_sub = [X[i] for i in indices]
        y_sub = [y[i] for i in indices]
        X_tr, y_tr, X_te, y_te = train_test_split(X_sub, y_sub, seed=trial)
        rf = RandomForest(n_trees=30, max_depth=5)
        rf.fit(X_tr, y_tr)
        acc = accuracy(y_te, rf.predict(X_te))
        forest_accs.append(acc)
        print(f"    试验 {trial + 1}: 准确率 = {acc:.4f}")

    single_std = (sum((a - sum(single_accs) / 5) ** 2 for a in single_accs) / 5) ** 0.5
    forest_std = (sum((a - sum(forest_accs) / 5) ** 2 for a in forest_accs) / 5) ** 0.5

    print()
    print(f"  单棵树:     均值 = {sum(single_accs)/5:.4f}, "
          f"标准差 = {single_std:.4f}")
    print(f"  随机森林:   均值 = {sum(forest_accs)/5:.4f}, "
          f"标准差 = {forest_std:.4f}")
    print()
    print("  森林在不同数据扰动下更稳定（方差更低）。")
    print()


def print_summary():
    """打印课程知识总结。"""
    print()
    print("=" * 65)
    print("小结")
    print("=" * 65)
    print()
    print("  1. 决策树通过最大化信息增益来切分数据。")
    print("  2. 基尼不纯度和信息熵生成的切分几乎相同。")
    print("  3. 单棵树不稳定——数据的小变化会导致完全不同的树。")
    print("  4. 随机森林通过平均多棵树来获得稳定、强力的预测。")
    print("  5. 自举采样 + 特征随机化使树之间互不相关。")
    print("  6. 特征重要性通过不纯度减少量自然得出。")
    print("  7. 在表格数据上，树模型通常优于神经网络。")
    print()


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    demo_split_criteria()
    demo_information_gain()
    demo_decision_tree()
    demo_gini_vs_entropy()
    demo_random_forest()
    demo_feature_importance()
    demo_regression_tree()
    demo_single_tree_vs_forest()
    print_summary()
