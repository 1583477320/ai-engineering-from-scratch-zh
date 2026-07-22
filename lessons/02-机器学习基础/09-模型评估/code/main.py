# 模型评估——从零实现分类与回归评估全流程
# 依赖：numpy>=1.24, matplotlib>=3.7, scipy>=1.10
# 对应课程：阶段 02 · 09（模型评估）

import math
import random
from collections import Counter
from typing import List, Tuple, Optional


# ============================================================
# 第 1 步：数据集划分
# ============================================================

def train_val_test_split(X: List, y: List,
                         train_ratio: float = 0.6,
                         val_ratio: float = 0.2,
                         seed: int = 42) -> Tuple:
    """将数据集按比例划分为训练集、验证集和测试集。

    注意：使用固定的 seed 保证可复现，实际工程中应在划分前 shuffle。
    """
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def kfold_split(n: int, k: int = 5, seed: int = 42) -> List:
    """生成 K 折交叉验证的索引对。

    每一折中，验证集约占总数据的 1/k，训练集占 (k-1)/k。
    """
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []
    for i in range(k):
        # 最后一个 fold 接收剩余所有样本（不丢弃）
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))
    return folds


def stratified_kfold_split(y: List, k: int = 5, seed: int = 42) -> List:
    """分层 K 折交叉验证——每折保持类别比例与全集一致。

    对分类任务至关重要。普通 K 折在小数据集上可能使某折缺少少数类。
    """
    random.seed(seed)

    # 按类别收集索引
    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    # 每个类别内随机打乱
    for label in class_indices:
        random.shuffle(class_indices[label])

    folds = [{"train": [], "val": []} for _ in range(k)]
    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(indices)
            val_part = indices[start:end]
            train_part = indices[:start] + indices[end:]
            folds[i]["val"].extend(val_part)
            folds[i]["train"].extend(train_part)

    return [(f["train"], f["val"]) for f in folds]


# ============================================================
# 第 2 步：混淆矩阵与分类指标
# ============================================================

def confusion_matrix(y_true: List, y_pred: List) -> Tuple:
    """计算二分类混淆矩阵的四个值：TP、TN、FP、FN。

    混淆矩阵是分类评估的基石，所有高阶指标（精确率、召回率、F1）
    都由这四个数字推导而来。
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy(y_true: List, y_pred: List) -> float:
    """准确率。

    最直觉的指标，但在类别不平衡时会严重误导。
    95% 负样本的数据上，永远猜"负"的模型准确率 95%——但毫无用处。
    """
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true: List, y_pred: List) -> float:
    """精确率——模型预测为正的那些，有多少确实为正。

    关注"预测质量"。误诊（FP）代价高时优先看精确率。
    公式：TP / (TP + FP)
    """
    tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true: List, y_pred: List) -> float:
    """召回率——所有真实为正的样本，被模型找出了多少。

    关注"覆盖度"。漏检（FN）代价高时优先看召回率。
    公式：TP / (TP + FN)
    """
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true: List, y_pred: List) -> float:
    """F1 分数——精确率和召回率的调和平均。

    调和平均的特点是：两个值中有一个较低时，F1 也会被拉低。
    这迫使你同时关注两者，不能靠牺牲一个来提升另一个。
    公式：2 × P × R / (P + R)
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ============================================================
# 第 3 步：ROC 曲线与 AUC
# ============================================================

def roc_curve(y_true: List, y_scores: List) -> Tuple:
    """计算 ROC 曲线上的点（FPR, TPR）。

    ROC 曲线描绘了当分类阈值从高到低变化时：
    - 横轴 FPR：负样本被误判为正的比例（越低越好）
    - 纵轴 TPR：正样本被正确找出的比例（越高越好）
    """
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list = []
    fpr_list = []

    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives

    for threshold in thresholds:
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)

        tpr = tp / total_positives if total_positives > 0 else 0.0
        fpr = fp / total_negatives if total_negatives > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list, thresholds


def auc_roc(y_true: List, y_scores: List) -> float:
    """计算 ROC 曲线下面积（AUC）。

    AUC 的直觉含义：随机抽一个正样本和一个负样本，
    模型给正样本的打分高于负样本的概率。
    - AUC = 1.0：完美排序
    - AUC = 0.5：相当于随机猜
    - AUC < 0.5：模型反向了，翻转预测即可
    """
    fpr_list, tpr_list, _ = roc_curve(y_true, y_scores)

    # 按 FPR 排序后梯形积分
    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    area = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i - 1]
        height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        area += width * height
    return area


# ============================================================
# 第 4 步：回归评估指标
# ============================================================

def mse(y_true: List, y_pred: List) -> float:
    """均方误差。对大误差惩罚更重（平方项放大）。"""
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def rmse(y_true: List, y_pred: List) -> float:
    """均方根误差。与目标同量纲，业务上更易解释。"""
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true: List, y_pred: List) -> float:
    """平均绝对误差。对异常值比 MSE 更鲁棒。"""
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def r_squared(y_true: List, y_pred: List) -> float:
    """决定系数 R²。回答：模型比"永远猜均值"好多少？

    R² = 1.0  完美拟合
    R² = 0.0  和猜均值一样差
    R² < 0.0  比猜均值更差（说明模型出大问题了）
    """
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ============================================================
# 第 5 步：统计检验——配对 t 检验与 McNemar 检验
# ============================================================

def paired_ttest(scores_a: List, scores_b: List) -> Tuple:
    """配对 t 检验——两模型在同一折上的成绩是否有显著差异。

    常用于交叉验证后比较两个模型。
    当 |t| > t_critical（df=k-1, alpha=0.05）时拒绝零假设。
    注意：5 折交叉验证的自由度仅为 4，检验效力（power）有限，
    这是工业界更倾向使用 McNemar 或重复交叉验证的原因。
    """
    n = len(scores_a)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / n
    # 样本标准差（除以 n-1 得无偏估计）
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)) if n > 1 else 1.0
    t_stat = mean_diff / (std_diff / math.sqrt(n)) if std_diff > 0 else 0.0
    return t_stat, mean_diff, std_diff


def mcnemar_test(y_true: List, pred_a: List, pred_b: List) -> Tuple:
    """McNemar 检验——比较两个分类器在同一测试集上的表现。

    核心思想：不看双方都对的样本，只关注"双方分歧"的部分。
    - b_count：A 错 B 对的样本数
    - c_count：A 对 B 错的样本数

    卡方统计量：χ² = (|b - c| - 1)² / (b + c)   （连续性校正）
    适用于在单次测试集上做比较，是配对 t 检验之外的另一种选择。
    """
    b_count = 0  # A 错 B 对
    c_count = 0  # A 对 B 错
    for yt, pa, pb in zip(y_true, pred_a, pred_b):
        if pa != yt and pb == yt:
            b_count += 1
        elif pa == yt and pb != yt:
            c_count += 1

    if b_count + c_count == 0:
        return 0.0, b_count, c_count  # 无分歧，无法检验

    # 连续性校正的卡方统计量
    chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    return chi2, b_count, c_count


# ============================================================
# 第 6 步：学习曲线
# ============================================================

def learning_curve(X: List, y: List, model_fn, metric_fn,
                  train_sizes: Optional[List] = None,
                  seed: int = 42) -> Tuple:
    """绘制学习曲线——训练集大小 Vs 训练/验证分数。

    学习曲线用于诊断偏差-方差权衡：
    - 两线都低且接近 → 欠拟合（高偏差）
    - 训练高、验证低、间隙大 → 过拟合（高方差）
    - 两线接近且都高 → 理想状态
    """
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    # 固定 20% 作为验证集，不随训练集变化
    val_size = int(n * 0.2)
    val_idx = indices[:val_size]
    pool_idx = indices[val_size:]

    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    if train_sizes is None:
        train_sizes = [int(len(pool_idx) * r) for r in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]

    train_scores = []
    val_scores = []

    for size in train_sizes:
        subset = pool_idx[:size]
        X_train = [X[i] for i in subset]
        y_train = [y[i] for i in subset]

        model = model_fn()
        model.fit(X_train, y_train)

        train_pred = [model.predict(x) for x in X_train]
        val_pred = [model.predict(x) for x in X_val]

        train_scores.append(metric_fn(y_train, train_pred))
        val_scores.append(metric_fn(y_val, val_pred))

    return train_sizes, train_scores, val_scores


# ============================================================
# 第 7 步：交叉验证封装
# ============================================================

def cross_validate(X: List, y: List, model_fn, k: int = 5,
                   metric_fn=None, stratified: bool = False) -> List:
    """对任意模型和评估指标执行 K 折或分层 K 折交叉验证。

    返回每折的评估分数，最终取均值和标准差作为模型评估。
    """
    n = len(X)
    if stratified:
        folds = stratified_kfold_split(y, k)
    else:
        folds = kfold_split(n, k)

    scores = []
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        predictions = [model.predict(x) for x in X_val]

        if metric_fn:
            score = metric_fn(y_val, predictions)
        else:
            score = sum(1 for yt, yp in zip(y_val, predictions) if yt == yp) / len(y_val)
        scores.append(score)
    return scores


# ============================================================
# 简单模型（用于演示）
# ============================================================

class SimpleLogistic:
    """逻辑回归（随机梯度下降版本）——仅用于演示评估流程。"""

    def __init__(self, lr: float = 0.1, epochs: int = 100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z: float) -> float:
        z = max(-500, min(500, z))  # 防止溢出
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X: List, y: List) -> None:
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                pred = self._sigmoid(z)
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j]
                self.bias += self.lr * error

    def predict_proba(self, x) -> float:
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self._sigmoid(z)

    def predict(self, x) -> int:
        return 1 if self.predict_proba(x) >= 0.5 else 0


class SimpleLinearRegression:
    """线性回归（梯度下降版本）——仅用于演示。"""

    def __init__(self, lr: float = 0.01, epochs: int = 200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X: List, y: List) -> None:
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(X)
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j] / n
                self.bias += self.lr * error / n

    def predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias


# ============================================================
# 数据生成器
# ============================================================

def make_classification_data(n: int = 300, seed: int = 42):
    """生成线性可分的二分类数据。"""
    random.seed(seed)
    X, y = [], []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        label = 1 if (x1 + x2 + random.gauss(0, 0.5)) > 0 else 0
        X.append([x1, x2])
        y.append(label)
    return X, y


def make_regression_data(n: int = 200, seed: int = 42):
    """生成线性回归数据（带噪声）。"""
    random.seed(seed)
    X, y = [], []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 5)
        target = 3 * x1 + 2 * x2 + random.gauss(0, 2)
        X.append([x1, x2])
        y.append(target)
    return X, y


def make_imbalanced_data(n: int = 300, minority_ratio: float = 0.05, seed: int = 42):
    """生成严重不平衡数据集（模拟欺诈检测/罕见病筛查）。

    正样本比例仅 5%，直观展示为何准确率在此场景毫无意义。
    """
    random.seed(seed)
    X, y = [], []
    for _ in range(n):
        if random.random() < minority_ratio:
            x1 = random.gauss(3, 0.5)
            x2 = random.gauss(3, 0.5)
            label = 1
        else:
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            label = 0
        X.append([x1, x2])
        y.append(label)
    return X, y


# ============================================================
# ASCII 可视化：学习曲线
# ============================================================

def ascii_plot_learning_curve(sizes, train_scores, val_scores, width=50, height=10):
    """用 ASCII 字符画学习曲线。"""
    all_scores = train_scores + val_scores
    min_s, max_s = min(all_scores), max(all_scores)
    if max_s == min_s:
        max_s = min_s + 1e-6

    # 将分数映射到行号
    def to_row(s):
        return int((s - min_s) / (max_s - min_s) * (height - 1))

    # 将大小映射到列号
    def to_col(idx):
        return int(idx / (len(sizes) - 1) * (width - 1)) if len(sizes) > 1 else 0

    grid = [[" " for _ in range(width)] for _ in range(height)]

    # 训练集曲线
    for i, s in enumerate(train_scores):
        r = height - 1 - to_row(s)
        c = to_col(i)
        if 0 <= r < height:
            grid[r][c] = "T"

    # 验证集曲线
    for i, s in enumerate(val_scores):
        r = height - 1 - to_row(s)
        c = to_col(i)
        if 0 <= r < height:
            grid[r][c] = "V"

    print(f"\n    学习曲线 (纵轴: 准确率 [{min_s:.3f}, {max_s:.3f}])")
    print(f"    {'':>8} T=训练  V=验证")
    print(f"    {'':>8} " + "-" * width)
    for row in grid:
        print(f"    {'':>8} |{''.join(row)}|")
    print(f"    {'':>8} " + "-" * width)
    size_labels = " ".join(f"{s:>6}" for s in sizes)
    print(f"    {'':8} 训练集大小 →\n")


# ============================================================
# 主流程演示
# ============================================================

def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    # ---- 1. 训练/验证/测试集划分 ----
    X_clf, y_clf = make_classification_data(300)

    print_section("数据集划分")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X_clf, y_clf
    )
    print(f"  训练集: {len(X_train)} 样本 (正样本 {sum(y_train)} 个)")
    print(f"  验证集: {len(X_val)} 样本 (正样本 {sum(y_val)} 个)")
    print(f"  测试集: {len(X_test)} 样本 (正样本 {sum(y_test)} 个)")

    # ---- 2. 分类指标 ----
    print_section("分类指标：混淆矩阵 / Precision / Recall / F1")
    model = SimpleLogistic(lr=0.1, epochs=200)
    model.fit(X_train, y_train)
    y_pred = [model.predict(x) for x in X_test]
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)

    print(f"  混淆矩阵:")
    print(f"             预测正    预测负")
    print(f"    真实正  {tp:>5}    {fn:>5}  (召回={recall(y_test, y_pred):.3f})")
    print(f"    真实负  {fp:>5}    {tn:>5}")
    print(f"            精确率={precision(y_test, y_pred):.3f}")

    print(f"\n  准确率:  {accuracy(y_test, y_pred):.4f}")
    print(f"  精确率:  {precision(y_test, y_pred):.4f}")
    print(f"  召回率:  {recall(y_test, y_pred):.4f}")
    print(f"  F1 分数: {f1_score(y_test, y_pred):.4f}")

    # ---- 3. ROC / AUC ----
    print_section("ROC 曲线与 AUC")
    y_scores = [model.predict_proba(x) for x in X_test]
    auc = auc_roc(y_test, y_scores)
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  (AUC=1.0 完美排序; AUC=0.5 相当于随机)")

    # ---- 4. K 折交叉验证 ----
    print_section("5 折交叉验证")
    cv_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
    )
    mean_cv = sum(cv_scores) / len(cv_scores)
    std_cv = math.sqrt(sum((s - mean_cv) ** 2 for s in cv_scores) / len(cv_scores))
    print(f"  各折准确率: {[round(s, 4) for s in cv_scores]}")
    print(f"  均值: {mean_cv:.4f} ± {std_cv:.4f}")

    # ---- 5. 分层 K 折 vs 普通 K 折 ----
    print_section("分层 K 折 vs 普通 K 折（不平衡数据上差异显著）")
    X_imb, y_imb = make_imbalanced_data(300, minority_ratio=0.05)
    print(f"  数据集: 正样本 {sum(y_imb)} / {len(y_imb)} "
          f"({sum(y_imb) / len(y_imb) * 100:.1f}%)")

    normal_scores = cross_validate(
        X_imb, y_imb,
        model_fn=lambda: SimpleLogistic(lr=0.5, epochs=500),
        k=5, metric_fn=accuracy, stratified=False,
    )
    strat_scores = cross_validate(
        X_imb, y_imb,
        model_fn=lambda: SimpleLogistic(lr=0.5, epochs=500),
        k=5, metric_fn=accuracy, stratified=True,
    )
    print(f"  普通 K 折准确率: {[round(s, 4) for s in normal_scores]}")
    print(f"  分层 K 折准确率: {[round(s, 4) for s in strat_scores]}")

    # ---- 6. 准确率的陷阱 ----
    print_section("准确率的陷阱：全猜负样本也会 95%")
    always_negative = [0] * len(y_imb)
    print(f"  永远猜'负'的模型:")
    print(f"    准确率:  {accuracy(y_imb, always_negative):.4f}")
    print(f"    精确率:  {precision(y_imb, always_negative):.4f}")
    print(f"    召回率:  {recall(y_imb, always_negative):.4f}")
    print(f"    F1 分数: {f1_score(y_imb, always_negative):.4f}")

    # ---- 7. 回归指标 ----
    print_section("回归指标：MSE / RMSE / MAE / R²")
    X_reg, y_reg = make_regression_data(200)
    X_tr_r, y_tr_r, X_v_r, y_v_r, X_te_r, y_te_r = train_val_test_split(X_reg, y_reg)
    reg_model = SimpleLinearRegression(lr=0.01, epochs=500)
    reg_model.fit(X_tr_r, y_tr_r)
    y_pred_r = [reg_model.predict(x) for x in X_te_r]

    print(f"  MSE:       {mse(y_te_r, y_pred_r):.4f}")
    print(f"  RMSE:      {rmse(y_te_r, y_pred_r):.4f}")
    print(f"  MAE:       {mae(y_te_r, y_pred_r):.4f}")
    print(f"  R-squared: {r_squared(y_te_r, y_pred_r):.4f}")

    # 与"永远猜均值"基线比较
    mean_baseline = [sum(y_tr_r) / len(y_tr_r)] * len(y_te_r)
    print(f"\n  均值基线:")
    print(f"    MSE:       {mse(y_te_r, mean_baseline):.4f}")
    print(f"    R-squared: {r_squared(y_te_r, mean_baseline):.4f}")

    # ---- 8. 统计检验 ----
    print_section("模型比较：配对 t 检验 vs McNemar 检验")
    # 两个不同超参的模型
    scores_a = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=100),
        k=5, metric_fn=accuracy,
    )
    scores_b = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.01, epochs=500),
        k=5, metric_fn=accuracy,
    )
    t_stat, mean_diff, std_diff = paired_ttest(scores_a, scores_b)
    print(f"  模型 A (lr=0.1)  均值: {sum(scores_a) / len(scores_a):.4f}")
    print(f"  模型 B (lr=0.01) 均值: {sum(scores_b) / len(scores_b):.4f}")
    print(f"  配对 t 统计量: {t_stat:.4f}")
    print(f"  (|t| > 2.78 在 df=4, p<0.05 下显著)")
    print(f"  结论: {'差异显著' if abs(t_stat) > 2.78 else '差异不显著，选更简单模型'}")

    # McNemar：在测试集上的比较
    model_a = SimpleLogistic(lr=0.1, epochs=100)
    model_a.fit(X_train, y_train)
    pred_a = [model_a.predict(x) for x in X_test]

    model_b = SimpleLogistic(lr=0.01, epochs=500)
    model_b.fit(X_train, y_train)
    pred_b = [model_b.predict(x) for x in X_test]

    chi2, b_cnt, c_cnt = mcnemar_test(y_test, pred_a, pred_b)
    print(f"\n  McNemar 检验 (在测试集上):")
    print(f"    A 错 B 对的样本数: {b_cnt}")
    print(f"    A 对 B 错的样本数: {c_cnt}")
    print(f"    χ² 统计量: {chi2:.4f}")
    print(f"    (|χ²| > 3.84 在 df=1, p<0.05 下显著)")
    print(f"    结论: {'差异显著' if chi2 > 3.84 else '差异不显著，选更简单模型'}")

    # ---- 9. 学习曲线 ----
    print_section("学习曲线（可视化偏差-方差权衡）")
    sizes, train_sc, val_sc = learning_curve(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        metric_fn=accuracy,
    )
    print(f"  {'训练大小':>8}  {'训练分':>8}  {'验证分':>8}")
    for s, tr, va in zip(sizes, train_sc, val_sc):
        print(f"  {s:>8}  {tr:>8.4f}  {va:>8.4f}")
    ascii_plot_learning_curve(sizes, train_sc, val_sc)

    print("评估演示完成。" + "  ")


if __name__ == "__main__":
    main()
