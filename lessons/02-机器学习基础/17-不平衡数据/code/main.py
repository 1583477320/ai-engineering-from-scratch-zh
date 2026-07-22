# main.py — 不平衡数据处理：从零实现与对比
# 依赖：numpy>=1.24, scikit-learn>=1.3, imbalanced-learn>=0.11
# 安装：pip install numpy scikit-learn imbalanced-learn
# 对应课程：阶段 02 · 17（不平衡数据）

import numpy as np
from collections import Counter


# =============================================================================
# 第 1 步：生成不平衡数据集
# =============================================================================

def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    """生成一个二分类不平衡数据集。

    多数类中心在 (0, 0)，少数类中心在 (2.5, 2.5)，两类有一定重叠。
    """
    rng = np.random.RandomState(seed)
    # 多数类：标准差较大，覆盖范围广
    X_majority = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    # 少数类：标准差较小，聚集在右上角
    X_minority = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_majority, X_minority])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    # 打乱顺序
    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


# =============================================================================
# 第 2 步：SMOTE —— 合成少数类过采样
# =============================================================================

def euclidean_distance(a, b):
    """计算两个向量之间的欧氏距离。"""
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    """找到样本 idx 在 X 中的 k 个最近邻（排除自身）。"""
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    """SMOTE：合成少数类过采样技术。

    对每个少数类样本，随机选一个 k 近邻，在两点连线上的随机位置生成新样本。
    公式：new = x + rand(0, 1) * (neighbor - x)

    Args:
        X_minority: 少数类样本矩阵，形状 (n_minority, n_features)
        k: 近邻数量
        n_synthetic: 需要生成的合成样本数量
        seed: 随机种子

    Returns:
        合成样本矩阵，形状 (n_synthetic, n_features)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)  # 防止 k 超过样本数
    synthetic = []

    for _ in range(n_synthetic):
        # 随机选一个少数类样本
        idx = rng.randint(0, n_samples)
        # 找到它的 k 个近邻
        neighbors = find_k_neighbors(X_minority, idx, k)
        # 随机选一个近邻
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        # 在两点之间随机插值
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)


# =============================================================================
# 第 3 步：随机过采样与欠采样
# =============================================================================

def random_oversample(X, y, seed=42):
    """随机过采样：复制少数类样本，使各类样本数一致。"""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    """随机欠采样：丢弃多数类样本，使各类样本数一致。"""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


# =============================================================================
# 第 4 步：带类别权重的逻辑回归
# =============================================================================

def sigmoid(z):
    """Sigmoid 激活函数，使用 clip 防止溢出。"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def compute_class_weights(y):
    """计算每个样本的权重，使少数类样本获得更高的权重。

    公式：weight_i = n_samples / (n_classes * count_of_class_i)

    这是 scikit-learn 中 class_weight='balanced' 的标准实现。
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    """带样本权重的逻辑回归。

    损失函数：weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])

    Args:
        X: 特征矩阵，形状 (n_samples, n_features)
        y: 标签向量，形状 (n_samples,)
        weights: 每个样本的权重，形状 (n_samples,)
        lr: 学习率
        epochs: 训练轮次

    Returns:
        w: 权重向量
        b: 偏置
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        # 关键：误差乘以样本权重，少数类误差被放大
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


# =============================================================================
# 第 5 步：阈值优化
# =============================================================================

def find_optimal_threshold(y_true, y_probs, metric="f1"):
    """在验证集上搜索最优分类阈值。

    默认阈值为 0.5，但在不平衡数据上通常不是最优。
    通过遍历阈值，找到使目标指标最大的阈值。

    Args:
        y_true: 真实标签
        y_probs: 模型预测的正类概率
        metric: 优化目标，可选 "f1"、"recall"、"precision"

    Returns:
        best_threshold: 最优阈值
        best_score: 对应的最优分数
    """
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


# =============================================================================
# 第 6 步：评估指标
# =============================================================================

def confusion_matrix_values(y_true, y_pred):
    """计算混淆矩阵的四个值：TP、TN、FP、FN。"""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    """计算完整的评估指标：准确率、精确率、召回率、F1、MCC。"""
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # MCC（马修斯相关系数）：-1 到 +1，对不平衡数据更鲁棒
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }


def compute_auprc(y_true, y_probs):
    """计算 AUPRC（精确率-召回率曲线下面积）。

    使用梯形法则近似计算。AUPRC 比 AUC-ROC 更适合不平衡数据。
    """
    # 按概率降序排列
    sorted_indices = np.argsort(-y_probs)
    y_sorted = y_true[sorted_indices]

    n_pos = np.sum(y_true == 1)
    if n_pos == 0 or n_pos == len(y_true):
        return 0.0

    tp_count = 0
    precisions = []
    recalls = []

    for i, label in enumerate(y_sorted):
        if label == 1:
            tp_count += 1
        precision = tp_count / (i + 1)
        recall = tp_count / n_pos
        precisions.append(precision)
        recalls.append(recall)

    # 梯形法则
    auprc = 0.0
    for i in range(1, len(recalls)):
        auprc += (recalls[i] - recalls[i - 1]) * precisions[i]
    return auprc


# =============================================================================
# 第 7 步：代价敏感学习
# =============================================================================

def cost_sensitive_predict(y_probs, cost_fp=1, cost_fn=10):
    """基于代价矩阵的最优预测。

    当假阴性（漏检）的代价远高于假阳性（误报）时，
    最优阈值 = cost_fp / (cost_fp + cost_fn)。

    推导：最小化期望代价 = P(y=1) * (1-threshold) * cost_fn + P(y=0) * threshold * cost_fp
    对 threshold 求导并令为 0，得到上述公式。

    Args:
        y_probs: 预测概率
        cost_fp: 假阳性的代价
        cost_fn: 假阴性的代价

    Returns:
        预测标签
    """
    threshold = cost_fp / (cost_fp + cost_fn)
    return (y_probs >= threshold).astype(int), threshold


# =============================================================================
# 主程序：对比所有策略
# =============================================================================

def main():
    print("=" * 60)
    print("不平衡数据处理策略对比")
    print("=" * 60)

    # 生成数据
    X, y = make_imbalanced_data(n_majority=950, n_minority=50, seed=42)
    print(f"\n数据集：共 {len(y)} 个样本")
    print(f"  类别分布：{dict(Counter(y.astype(int)))}")
    print(f"  不平衡比例：{(y == 0).sum() / (y == 1).sum():.0f}:1")

    # 划分训练集和测试集（保持类别比例）
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n训练集：{dict(Counter(y_train.astype(int)))}")
    print(f"测试集：{dict(Counter(y_test.astype(int)))}")

    results = {}

    # --- 策略 1：基线（不做任何处理）---
    w_base, b_base = logistic_regression_weighted(
        X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
    )
    probs_base = sigmoid(X_test @ w_base + b_base)
    preds_base = (probs_base >= 0.5).astype(int)
    results["基线（无处理）"] = compute_metrics(y_test, preds_base)

    # --- 策略 2：随机过采样 ---
    X_over, y_over = random_oversample(X_train, y_train, seed=42)
    w_over, b_over = logistic_regression_weighted(
        X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
    )
    preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)
    results["随机过采样"] = compute_metrics(y_test, preds_over)

    # --- 策略 3：SMOTE ---
    minority_mask = y_train == 1
    X_minority = X_train[minority_mask]
    n_minority = int(minority_mask.sum())
    n_needed = len(y_train) - 2 * n_minority  # 需要生成的合成样本数
    if n_needed > 0:
        synthetic = smote(X_minority, k=5, n_synthetic=n_needed, seed=42)
        X_smote = np.vstack([X_train, synthetic])
        y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
    else:
        X_smote, y_smote = X_train, y_train
    w_sm, b_sm = logistic_regression_weighted(
        X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
    )
    preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)
    results["SMOTE"] = compute_metrics(y_test, preds_smote)

    # --- 策略 4：类别权重 ---
    sample_weights = compute_class_weights(y_train)
    w_cw, b_cw = logistic_regression_weighted(
        X_train, y_train, sample_weights, lr=0.1, epochs=300
    )
    probs_cw = sigmoid(X_test @ w_cw + b_cw)
    preds_cw = (probs_cw >= 0.5).astype(int)
    results["类别权重"] = compute_metrics(y_test, preds_cw)

    # --- 策略 5：类别权重 + 阈值优化 ---
    # 在训练集上做简单的 hold-out 验证来选择阈值
    val_split = int(0.8 * len(y_train))
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_sub, y_sub = X_train[:val_split], y_train[:val_split]
    sub_weights = compute_class_weights(y_sub)
    w_val, b_val = logistic_regression_weighted(
        X_sub, y_sub, sub_weights, lr=0.1, epochs=300
    )
    probs_val = sigmoid(X_val @ w_val + b_val)
    best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
    preds_thresh = (probs_cw >= best_thresh).astype(int)
    results["类别权重+阈值优化"] = compute_metrics(y_test, preds_thresh)

    # --- 策略 6：代价敏感学习 ---
    preds_cost, cost_thresh = cost_sensitive_predict(probs_cw, cost_fp=1, cost_fn=10)
    results["代价敏感（1:10）"] = compute_metrics(y_test, preds_cost)

    # --- 打印结果 ---
    print("\n" + "-" * 60)
    print(f"{'策略':<20} {'准确率':>8} {'精确率':>8} {'召回率':>8} {'F1':>8} {'MCC':>8}")
    print("-" * 60)
    for name, metrics in results.items():
        print(
            f"{name:<20} "
            f"{metrics['accuracy']:>8.3f} "
            f"{metrics['precision']:>8.3f} "
            f"{metrics['recall']:>8.3f} "
            f"{metrics['f1']:>8.3f} "
            f"{metrics['mcc']:>8.3f}"
        )

    # --- AUPRC 对比 ---
    print("\n" + "-" * 60)
    print("AUPRC 对比（精确率-召回率曲线下面积）：")
    print("-" * 60)
    auprc_base = compute_auprc(y_test, probs_base)
    auprc_cw = compute_auprc(y_test, probs_cw)
    print(f"  基线（无处理）：  AUPRC = {auprc_base:.3f}")
    print(f"  类别权重：      AUPRC = {auprc_cw:.3f}")

    # --- 阈值优化详情 ---
    print(f"\n阈值优化：最优阈值 = {best_thresh:.2f}（默认 0.50）")
    print(f"  代价敏感阈值 = {cost_thresh:.2f}（假阴性代价是假阳性的 10 倍）")

    print("\n" + "=" * 60)
    print("结论：在不平衡数据上，准确率是虚假的繁荣。")
    print("  类别权重 + 阈值优化通常是最具性价比的组合。")
    print("=" * 60)


if __name__ == "__main__":
    main()
