# logistic_regression.py — 从零实现逻辑回归
# 依赖：numpy, scikit-learn（可选，用于对比）
# 安装：pip install numpy scikit-learn
# 对应课程：阶段 02 · 03（逻辑回归）

import math
import random

# === 第 1 步：Sigmoid 函数与数据生成 ===

def sigmoid(z):
    """Sigmoid 函数：将任意实数映射到 (0, 1) 区间。

    通过裁剪 z 的范围防止 exp 溢出。
    """
    # 防止 exp(-z) 溢出：当 z < -500 时直接返回 0，当 z > 500 时直接返回 1
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


def generate_binary_data(n_samples=200, seed=42):
    """生成二分类数据集：两个高斯分布的簇。"""
    random.seed(seed)
    X = []
    y = []

    # 类别 0：中心在 (2, 2)
    for _ in range(n_samples // 2):
        X.append([random.gauss(2, 1), random.gauss(2, 1)])
        y.append(0)

    # 类别 1：中心在 (5, 5)
    for _ in range(n_samples // 2):
        X.append([random.gauss(5, 1), random.gauss(5, 1)])
        y.append(1)

    # 打乱顺序
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)


# === 第 2 步：逻辑回归模型 ===

class LogisticRegression:
    """从零实现的逻辑回归分类器。

    使用二元交叉熵损失 + 梯度下降优化。
    """

    def __init__(self, n_features, learning_rate=0.01):
        # 权重初始化为 0，偏置初始化为 0
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_proba(self, x):
        """预测样本属于类别 1 的概率。"""
        # 先计算线性组合 z = w·x + b，再通过 sigmoid 映射到 (0, 1)
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        """根据概率和阈值预测类别。"""
        return 1 if self.predict_proba(x) >= threshold else 0

    def compute_loss(self, X, y):
        """计算二元交叉熵损失。

        L = -(1/n) * Σ [y·log(p) + (1-y)·log(1-p)]

        裁剪 p 防止 log(0)。
        """
        n = len(y)
        total = 0.0
        for i in range(n):
            p = self.predict_proba(X[i])
            # 裁剪概率值，防止 log(0)
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self, X, y, epochs=1000, print_every=200):
        """训练模型：批量梯度下降。

        梯度公式：
            dw = (1/n) * Σ (p - y) * x
            db = (1/n) * Σ (p - y)
        """
        n = len(y)
        n_features = len(X[0])

        for epoch in range(epochs):
            # 初始化梯度
            dw = [0.0] * n_features
            db = 0.0

            # 计算所有样本的梯度累加
            for i in range(n):
                p = self.predict_proba(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error

            # 更新权重和偏置
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)

            # 记录损失
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)

            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | "
                      f"w: [{self.weights[0]:.3f}, {self.weights[1]:.3f}] | "
                      f"b: {self.bias:.3f}")
        return self

    def accuracy(self, X, y):
        """计算分类准确率。"""
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


# === 第 3 步：分类评估指标 ===

class ClassificationMetrics:
    """从零实现的分类评估指标：精确率、召回率、F1 分数、混淆矩阵。"""

    def __init__(self, y_true, y_pred):
        # 计算混淆矩阵的四个值
        self.tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        self.tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        self.fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        self.fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def precision(self):
        """精确率：预测为正类的样本中，真正为正类的比例。"""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0

    def recall(self):
        """召回率：真实为正类的样本中，被正确预测的比例。"""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0

    def f1(self):
        """F1 分数：精确率和召回率的调和平均。"""
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def print_confusion_matrix(self):
        """打印混淆矩阵。"""
        print(f"\n  混淆矩阵:")
        print(f"                  预测")
        print(f"                  正类  负类")
        print(f"  真实 正类     {self.tp:4d}  {self.fn:4d}")
        print(f"  真实 负类     {self.fp:4d}  {self.tn:4d}")

    def print_report(self):
        """打印完整分类报告。"""
        self.print_confusion_matrix()
        print(f"\n  准确率:   {self.accuracy():.4f}")
        print(f"  精确率:   {self.precision():.4f}")
        print(f"  召回率:   {self.recall():.4f}")
        print(f"  F1 分数:  {self.f1():.4f}")


# === 第 4 步：多分类 Softmax 回归 ===

class SoftmaxRegression:
    """从零实现的 Softmax 回归（多分类逻辑回归）。

    使用类别交叉熵损失 + 梯度下降优化。
    """

    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        # 每个类别有独立的权重向量和偏置
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.biases = [0.0] * n_classes

    def softmax(self, scores):
        """Softmax 函数：将分数向量转换为概率分布。

        减去最大值防止 exp 溢出（数值稳定性）。
        """
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    def predict_proba(self, x):
        """预测每个类别的概率。"""
        scores = [
            sum(self.weights[k][j] * x[j] for j in range(self.n_features)) + self.biases[k]
            for k in range(self.n_classes)
        ]
        return self.softmax(scores)

    def predict(self, x):
        """预测类别：取概率最高的类别。"""
        probs = self.predict_proba(x)
        return probs.index(max(probs))

    def fit(self, X, y, epochs=1000, print_every=200):
        """训练 Softmax 回归模型。"""
        n = len(y)

        for epoch in range(epochs):
            grad_w = [[0.0] * self.n_features for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            total_loss = 0.0

            for i in range(n):
                probs = self.predict_proba(X[i])
                # 对每个类别计算梯度
                for k in range(self.n_classes):
                    target = 1.0 if y[i] == k else 0.0
                    error = probs[k] - target
                    for j in range(self.n_features):
                        grad_w[k][j] += error * X[i][j]
                    grad_b[k] += error
                # 累加损失
                true_prob = max(probs[y[i]], 1e-15)
                total_loss -= math.log(true_prob)

            # 更新参数
            for k in range(self.n_classes):
                for j in range(self.n_features):
                    self.weights[k][j] -= self.lr * (grad_w[k][j] / n)
                self.biases[k] -= self.lr * (grad_b[k] / n)

            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {total_loss / n:.4f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


# === 第 5 步：可视化逻辑回归的 sigmoid 函数 ===

def visualize_sigmoid():
    """用 ASCII 可视化 sigmoid 函数的形状。"""
    print("\n=== Sigmoid 函数可视化 ===")
    print("sigmoid(z) = 1 / (1 + e^(-z))\n")

    # 横轴范围 -6 到 6
    width = 50
    for z in range(-6, 7):
        sig = sigmoid(z)
        # 将 (0, 1) 映射到 0~width 的字符位置
        pos = int(sig * width)
        bar = " " * pos + "*" + " " * (width - pos)
        print(f"  z = {z:3d} | {bar} | {sig:.4f}")


# === 主程序 ===

if __name__ == "__main__":
    # --- 生成数据 ---
    X, y = generate_binary_data(n_samples=200)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"生成 {len(X)} 个样本（2 个类别，2 个特征）")
    print(f"类别 0 中心: (2, 2)，类别 1 中心: (5, 5)")

    # --- 训练逻辑回归 ---
    print("\n=== 训练逻辑回归 ===")
    model = LogisticRegression(n_features=2, learning_rate=0.1)
    model.fit(X_train, y_train, epochs=1000, print_every=200)

    print(f"\n训练集准确率: {model.accuracy(X_train, y_train):.4f}")
    print(f"测试集准确率:  {model.accuracy(X_test, y_test):.4f}")
    print(f"权重: [{model.weights[0]:.4f}, {model.weights[1]:.4f}]")
    print(f"偏置: {model.bias:.4f}")

    # --- 评估指标 ---
    y_pred_test = [model.predict(x) for x in X_test]
    print("\n=== 测试集分类报告 ===")
    metrics = ClassificationMetrics(y_test, y_pred_test)
    metrics.print_report()

    # --- 决策边界 ---
    print("\n=== 决策边界 ===")
    w1, w2 = model.weights
    b = model.bias
    print(f"决策边界: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0")
    if abs(w2) > 1e-10:
        print(f"解出 x2:   x2 = {-w1/w2:.4f}*x1 + {-b/w2:.4f}")

    # --- Sigmoid 可视化 ---
    visualize_sigmoid()

    # --- 阈值调优 ---
    print("\n=== 阈值调优 ===")
    print("默认阈值 0.5。调整阈值可以在精确率和召回率之间权衡。\n")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"{'阈值':>8} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1':>10}")
    print("-" * 50)

    for t in thresholds:
        y_pred_t = [1 if model.predict_proba(x) >= t else 0 for x in X_test]
        m = ClassificationMetrics(y_test, y_pred_t)
        print(f"{t:>8.1f} {m.accuracy():>10.4f} {m.precision():>10.4f} "
              f"{m.recall():>10.4f} {m.f1():>10.4f}")

    # --- Softmax 多分类 ---
    random.seed(42)
    X_3class = []
    y_3class = []
    centers = [(1, 1), (5, 1), (3, 5)]

    for label, (cx, cy) in enumerate(centers):
        for _ in range(50):
            X_3class.append([random.gauss(cx, 0.8), random.gauss(cy, 0.8)])
            y_3class.append(label)

    combined = list(zip(X_3class, y_3class))
    random.shuffle(combined)
    X_3class, y_3class = zip(*combined)
    X_3class, y_3class = list(X_3class), list(y_3class)

    split_3 = int(0.8 * len(X_3class))
    X_train_3 = X_3class[:split_3]
    y_train_3 = y_3class[:split_3]
    X_test_3 = X_3class[split_3:]
    y_test_3 = y_3class[split_3:]

    print("\n=== Softmax 回归（3 个类别）===")
    softmax_model = SoftmaxRegression(n_features=2, n_classes=3, learning_rate=0.1)
    softmax_model.fit(X_train_3, y_train_3, epochs=1000, print_every=200)
    print(f"\n训练集准确率: {softmax_model.accuracy(X_train_3, y_train_3):.4f}")
    print(f"测试集准确率:  {softmax_model.accuracy(X_test_3, y_test_3):.4f}")

    print("\n样本预测:")
    for i in range(5):
        probs = softmax_model.predict_proba(X_test_3[i])
        pred = softmax_model.predict(X_test_3[i])
        prob_str = ", ".join(f"{p:.3f}" for p in probs)
        print(f"  真实: {y_test_3[i]}, 预测: {pred}, 概率: [{prob_str}]")

    # --- 与 scikit-learn 对比 ---
    print("\n=== 与 scikit-learn 对比 ===")
    try:
        from sklearn.linear_model import LogisticRegression as SklearnLR
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix, classification_report)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        np.random.seed(42)
        X_0 = np.random.randn(100, 2) + [2, 2]
        X_1 = np.random.randn(100, 2) + [5, 5]
        X_sk = np.vstack([X_0, X_1])
        y_sk = np.array([0] * 100 + [1] * 100)

        X_tr, X_te, y_tr, y_te = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        lr = SklearnLR()
        lr.fit(X_tr_sc, y_tr)
        y_pred_sk = lr.predict(X_te_sc)

        print(f"准确率:   {accuracy_score(y_te, y_pred_sk):.4f}")
        print(f"精确率:   {precision_score(y_te, y_pred_sk):.4f}")
        print(f"召回率:   {recall_score(y_te, y_pred_sk):.4f}")
        print(f"F1 分数:  {f1_score(y_te, y_pred_sk):.4f}")
        print(f"\n混淆矩阵:\n{confusion_matrix(y_te, y_pred_sk)}")
        print(f"\n分类报告:\n{classification_report(y_te, y_pred_sk)}")

    except ImportError:
        print("scikit-learn 未安装。安装命令: pip install scikit-learn")
        print("从零实现的所有功能无需任何外部依赖即可运行。")
