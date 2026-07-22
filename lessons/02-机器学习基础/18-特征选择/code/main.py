# 特征选择 — 从零实现过滤法、包裹法与嵌入法
# 依赖：numpy>=1.24
# 对应课程：阶段 02 · 18（特征选择）

import numpy as np


# === 数据生成 ===

def make_feature_selection_data(n_samples=500, seed=42):
    """生成包含信息特征、相关特征和噪声特征的模拟数据集。

    真实标签仅由前 3 个信息特征决定，其余 12 个特征均为冗余或噪声。
    """
    rng = np.random.RandomState(seed)

    # 3 个独立的信息特征
    x1 = rng.randn(n_samples)
    x2 = rng.randn(n_samples)
    x3 = rng.randn(n_samples)

    # 2 个与信息特征高度相关的冗余特征
    x4 = x1 + 0.1 * rng.randn(n_samples)
    x5 = x2 + 0.1 * rng.randn(n_samples)

    informative = np.column_stack([x1, x2, x3, x4, x5])

    # 5 个由信息特征线性组合生成的相关特征
    correlated = np.column_stack([
        x1 * 0.9 + 0.1 * rng.randn(n_samples),
        x2 * 0.8 + 0.2 * rng.randn(n_samples),
        x3 * 0.7 + 0.3 * rng.randn(n_samples),
        x1 * 0.5 + x2 * 0.5 + 0.1 * rng.randn(n_samples),
        x2 * 0.6 + x3 * 0.4 + 0.1 * rng.randn(n_samples),
    ])

    # 10 个纯噪声特征
    noise = rng.randn(n_samples, 10) * 0.5

    X = np.hstack([informative, correlated, noise])
    # 真实决策边界仅依赖 x1、x2、x3
    y = (2 * x1 - 1.5 * x2 + x3 + 0.5 * rng.randn(n_samples) > 0).astype(int)

    feature_names = (
        [f"info_{i}" for i in range(5)]
        + [f"corr_{i}" for i in range(5)]
        + [f"noise_{i}" for i in range(10)]
    )

    return X, y, feature_names


# === 第 1 步：过滤法 — 方差阈值 ===

def variance_threshold(X, threshold=0.01):
    """移除方差低于阈值的特征。

    方差极低的特征几乎为常数，对模型没有区分能力。
    这是特征选择的第一步——零成本、零风险。
    """
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return mask, variances


# === 第 2 步：过滤法 — 互信息 ===

def discretize(x, n_bins=10):
    """将连续特征离散化为 n_bins 个区间。"""
    min_val, max_val = x.min(), x.max()
    if max_val == min_val:
        return np.zeros_like(x, dtype=int)
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned = np.digitize(x, bin_edges[1:-1])
    return binned


def mutual_information(X, y, n_bins=10):
    """计算每个特征与标签之间的互信息（Mutual Information）。

    互信息衡量的是：知道特征 X 之后，对标签 Y 的不确定性减少了多少。
    与皮尔逊相关系数不同，互信息可以捕捉任意非线性关系。
    """
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)

    y_vals, y_counts = np.unique(y, return_counts=True)
    p_y = y_counts / n_samples

    for f in range(n_features):
        x_binned = discretize(X[:, f], n_bins)
        x_vals, x_counts = np.unique(x_binned, return_counts=True)
        p_x = dict(zip(x_vals, x_counts / n_samples))

        mi = 0.0
        for xv in x_vals:
            for yi, yv in enumerate(y_vals):
                joint_mask = (x_binned == xv) & (y == yv)
                p_xy = np.sum(joint_mask) / n_samples
                if p_xy > 0:
                    mi += p_xy * np.log(p_xy / (p_x[xv] * p_y[yi]))
        mi_scores[f] = mi

    return mi_scores


# === 第 3 步：包裹法 — 递归特征消除（RFE） ===

def simple_logistic_importance(X, y, lr=0.1, epochs=100):
    """训练一个简单逻辑回归，返回权重向量。

    权重的绝对值大小反映了特征对决策边界的影响程度。
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y
        w -= lr * (X.T @ error) / n_samples
        b -= lr * np.mean(error)

    return w, b


def rfe(X, y, n_features_to_select=5, lr=0.1, epochs=100):
    """递归特征消除（Recursive Feature Elimination）。

    反复训练模型，每次移除权重绝对值最小的特征，
    直到剩余特征数达到目标值。

    为什么不能一次性移除所有低重要性特征？
    因为特征重要性是相对的——移除一个相关特征后，
    其搭档的重要性可能会上升。迭代移除让模型在每步重新评估。
    """
    n_total = X.shape[1]
    remaining = list(range(n_total))
    rankings = np.ones(n_total, dtype=int)
    rank = n_total

    while len(remaining) > n_features_to_select:
        X_subset = X[:, remaining]
        w, _ = simple_logistic_importance(X_subset, y, lr, epochs)
        importances = np.abs(w)

        least_idx = np.argmin(importances)
        original_idx = remaining[least_idx]
        rankings[original_idx] = rank
        rank -= 1
        remaining.pop(least_idx)

    for idx in remaining:
        rankings[idx] = 1

    selected_mask = rankings == 1
    return selected_mask, rankings


# === 第 4 步：嵌入法 — L1 正则化（Lasso） ===

def soft_threshold(w, alpha):
    """软阈值函数——L1 正则化的核心操作。

    将每个权重向零收缩 alpha，小于 alpha 的权重直接变为 0。
    这是 L1 能产生稀疏解的根本原因。
    """
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)


def l1_feature_selection(X, y, alpha=0.1, lr=0.01, epochs=500):
    """通过 L1 正则化进行特征选择。

    L1 惩罚项的几何形状是菱形（高维是菱形多面体），
    最优解容易落在菱形的顶点上——此时某些权重恰好为 0。
    这等价于自动完成了特征选择。
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y

        gradient_w = (X.T @ error) / n_samples
        gradient_b = np.mean(error)

        w -= lr * gradient_w
        # 关键步骤：软阈值让不重要的特征权重归零
        w = soft_threshold(w, lr * alpha)
        b -= lr * gradient_b

    selected_mask = np.abs(w) > 1e-6
    return selected_mask, w


# === 第 5 步：嵌入法 — 树模型特征重要性 ===

def gini_impurity(y):
    """计算基尼不纯度——衡量节点中类别的混杂程度。"""
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def best_split(X, y, feature_idx):
    """对单个特征寻找最佳分裂阈值。"""
    values = np.unique(X[:, feature_idx])
    if len(values) <= 1:
        return None, -1.0
    best_threshold, best_gain = None, -1.0
    parent_gini = gini_impurity(y)
    n = len(y)
    # 最多采样 20 个候选阈值，避免过慢
    step = max(1, (len(values) - 1) // min(20, len(values) - 1))
    for i in range(0, len(values) - 1, step):
        threshold = (values[i] + values[i + 1]) / 2.0
        left_mask = X[:, feature_idx] <= threshold
        n_left, n_right = np.sum(left_mask), n - np.sum(left_mask)
        if n_left == 0 or n_right == 0:
            continue
        gain = parent_gini - (n_left / n) * gini_impurity(y[left_mask]) - (n_right / n) * gini_impurity(y[~left_mask])
        if gain > best_gain:
            best_gain, best_threshold = gain, threshold
    return best_threshold, best_gain


def _build_tree_importance(X, y, feature_subset, max_depth, depth=0):
    """递归构建决策树，并累加每个特征的重要性。"""
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    if depth >= max_depth or len(np.unique(y)) <= 1 or len(y) < 4:
        return importances

    best_feature = None
    best_threshold = None
    best_gain = -1.0

    for f in feature_subset:
        threshold, gain = best_split(X, y, f)
        if gain > best_gain:
            best_gain = gain
            best_feature = f
            best_threshold = threshold

    if best_feature is None or best_gain <= 0:
        return importances

    # 特征重要性 = 不纯度减少量 × 样本数
    importances[best_feature] += best_gain * len(y)

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    importances += _build_tree_importance(X[left_mask], y[left_mask], feature_subset, max_depth, depth + 1)
    importances += _build_tree_importance(X[right_mask], y[right_mask], feature_subset, max_depth, depth + 1)

    return importances


def tree_importance(X, y, n_trees=50, max_depth=5, seed=42):
    """基于随机森林思想计算特征重要性。

    通过多棵树的平均来降低单棵树的随机性，
    每棵树随机采样样本和特征子集。
    """
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    importances = np.zeros(n_features)

    for _ in range(n_trees):
        sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
        n_subset = max(1, int(np.sqrt(n_features)))
        feature_subset = rng.choice(n_features, size=n_subset, replace=False)

        X_boot = X[sample_idx]
        y_boot = y[sample_idx]

        tree_imp = _build_tree_importance(X_boot, y_boot, feature_subset, max_depth)
        importances += tree_imp

    total = importances.sum()
    if total > 0:
        importances /= total

    return importances


# === 评估与输出 ===

def evaluate_accuracy(X, y, selected_mask, lr=0.1, epochs=200):
    """在选定特征上训练逻辑回归并返回测试准确率。"""
    X_selected = X[:, selected_mask]
    n = len(y)
    split = int(0.8 * n)

    X_train, X_test = X_selected[:split], X_selected[split:]
    y_train, y_test = y[:split], y[split:]

    w, b = simple_logistic_importance(X_train, y_train, lr, epochs)
    z = X_test @ w + b
    preds = (1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))) >= 0.5).astype(int)
    return np.mean(preds == y_test)


def feature_group(name):
    """判断特征属于哪个分组。"""
    if "noise" in name:
        return "NOISE"
    if "corr" in name:
        return "CORR"
    return "INFO"


def print_feature_scores(names, scores, label, top_k=None):
    """按分数降序打印特征排名。"""
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    print(f"\n  {label}:")
    for i, (idx, s) in enumerate(ranked[:top_k or len(ranked)]):
        print(f"    {i+1:>2}. {names[idx]:<12} {s:>8.4f} [{feature_group(names[idx])}]")


# === 主程序 ===

if __name__ == "__main__":
    print("=" * 60)
    print("特征选择方法对比实验")
    print("=" * 60)

    X, y, feature_names = make_feature_selection_data(500, seed=42)
    print(f"\n数据集：{X.shape[0]} 个样本，{X.shape[1]} 个特征")
    print(f"特征分组：5 个信息特征、5 个相关特征、10 个噪声特征")
    print(f"标签分布：y=1 有 {np.sum(y)} 个，y=0 有 {np.sum(y==0)} 个")

    # 划分训练集和测试集
    n = len(y)
    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 标准化：使用训练集的均值和标准差
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_scaled_train = (X_train - mean) / std
    X_scaled_test = (X_test - mean) / std
    X_scaled = np.vstack([X_scaled_train, X_scaled_test])

    # --- 方法 1：方差阈值 ---
    print("\n" + "-" * 60)
    print("1. 方差阈值（Variance Threshold）")
    print("-" * 60)
    var_mask, variances = variance_threshold(X_train, threshold=0.01)
    print(f"  阈值：0.01，保留 {np.sum(var_mask)} / {len(var_mask)} 个特征")
    print_feature_scores(feature_names, variances, "方差值", top_k=10)

    # --- 方法 2：互信息 ---
    print("\n" + "-" * 60)
    print("2. 互信息（Mutual Information）")
    print("-" * 60)
    mi_scores = mutual_information(X_train, y_train, n_bins=10)
    print_feature_scores(feature_names, mi_scores, "互信息分数（前 10）", top_k=10)
    mi_selected = np.zeros(len(feature_names), dtype=bool)
    mi_selected[np.argsort(mi_scores)[-5:]] = True

    # --- 方法 3：递归特征消除 ---
    print("\n" + "-" * 60)
    print("3. 递归特征消除（RFE）")
    print("-" * 60)
    rfe_mask, rfe_rankings = rfe(X_scaled_train, y_train, n_features_to_select=5, lr=0.1, epochs=200)
    print(f"  选中特征：{[feature_names[i] for i in range(len(feature_names)) if rfe_mask[i]]}")
    for idx, rank in sorted(enumerate(rfe_rankings), key=lambda x: x[1]):
        print(f"    排名 {rank:>2}：{feature_names[idx]:<12} [{feature_group(feature_names[idx])}]")

    # --- 方法 4：L1 正则化 ---
    print("\n" + "-" * 60)
    print("4. L1 正则化（Lasso）")
    print("-" * 60)
    l1_mask, l1_weights = l1_feature_selection(X_scaled_train, y_train, alpha=0.05, lr=0.01, epochs=1000)
    print(f"  非零权重数量：{np.sum(l1_mask)}")
    print(f"  选中特征：{[feature_names[i] for i in range(len(feature_names)) if l1_mask[i]]}")
    print_feature_scores(feature_names, np.abs(l1_weights), "|权重|（前 10）", top_k=10)

    # --- 方法 5：树模型重要性 ---
    print("\n" + "-" * 60)
    print("5. 树模型特征重要性")
    print("-" * 60)
    tree_imp = tree_importance(X_train, y_train, n_trees=100, max_depth=6, seed=42)
    print_feature_scores(feature_names, tree_imp, "重要性（前 10）", top_k=10)
    tree_selected = np.zeros(len(feature_names), dtype=bool)
    tree_selected[np.argsort(tree_imp)[-5:]] = True

    # --- 方法一致性对比 ---
    print("\n" + "=" * 60)
    print("方法一致性对比")
    print("=" * 60)
    all_masks = {"MI": mi_selected, "RFE": rfe_mask, "L1": l1_mask, "Tree": tree_selected}
    header = f"  {'Feature':<12}" + "".join(f" {n:>6}" for n in all_masks) + f" {'Total':>6}"
    print(f"\n{header}")
    print(f"  {'-'*12}" + " ------" * (len(all_masks) + 1))
    for i, fname in enumerate(feature_names):
        row = f"  {fname:<12}"
        count = sum(1 for m in all_masks.values() if m[i])
        for mask in all_masks.values():
            row += f" {'YES':>6}" if mask[i] else f" {'---':>6}"
        print(f"{row} {count:>6}")

    # --- 准确率对比 ---
    print("\n" + "=" * 60)
    print("准确率对比")
    print("=" * 60)

    all_features_mask = np.ones(len(feature_names), dtype=bool)
    info_only_mask = np.array([i < 5 for i in range(len(feature_names))])

    experiments = [
        ("全部 20 个特征", all_features_mask),
        ("仅信息特征（5）", info_only_mask),
        ("互信息 Top-5", mi_selected),
        ("RFE Top-5", rfe_mask),
        ("L1 选择", l1_mask),
        ("树重要性 Top-5", tree_selected),
    ]

    print(f"\n  {'方法':<20} {'特征数':>10} {'准确率':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")

    for name, mask in experiments:
        if np.sum(mask) == 0:
            print(f"  {name:<20} {int(np.sum(mask)):>10} {'N/A':>10}")
            continue
        acc = evaluate_accuracy(X_scaled, y, mask, lr=0.1, epochs=300)
        print(f"  {name:<20} {int(np.sum(mask)):>10} {acc:>10.4f}")

    print("\n完成。")
