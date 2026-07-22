# main.py — 从零实现朴素贝叶斯分类器
# 依赖：numpy>=1.24, scikit-learn>=1.3, scipy>=1.10
# 安装：pip install numpy scikit-learn scipy
# 对应课程：阶段 02 · 14（朴素贝叶斯）

import numpy as np
from collections import Counter


# ============================================================
# 第 1 步：多项式朴素贝叶斯（MultinomialNB）
# ============================================================

class MultinomialNB:
    """多项式朴素贝叶斯 — 适用于词频计数特征（如文本分类）。

    核心思想：对每个类别，统计每个特征出现的频率，
    使用拉普拉斯平滑避免零概率问题。
    """

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: 拉普拉斯平滑系数。
                   alpha=1 为标准拉普拉斯平滑，
                   alpha<1 为 Lidstone 平滑。
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None   # 类别先验概率的对数
        self.feature_log_prob_ = None  # 特征条件概率的对数

    def fit(self, X, y):
        """训练模型：统计每个类别下每个特征的出现频率。

        Args:
            X: 形状 (n_samples, n_features)，非负计数值
            y: 形状 (n_samples,)，类别标签

        Returns:
            self
        """
        if np.any(X < 0):
            raise ValueError("MultinomialNB 要求特征值非负（词频不能为负）")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            # 取出属于类别 c 的所有样本
            X_c = X[y == c]

            # 类别先验：该类样本数 / 总样本数
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])

            # 特征计数 + 拉普拉斯平滑
            counts = X_c.sum(axis=0) + self.alpha
            total_count = counts.sum()

            # 条件概率的对数：P(特征_i | 类别_c)
            self.feature_log_prob_[i] = np.log(counts / total_count)

        return self

    def predict_log_proba(self, X):
        """计算每个样本属于每个类别的对数概率。

        log P(c | x) ∝ log P(c) + Σ x_i * log P(特征_i | c)
        上式可转化为矩阵乘法。
        """
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_proba(self, X):
        """计算概率值（用于查看，不用于预测决策）。"""
        log_proba = self.predict_log_proba(X)
        # 数值稳定性：减去最大值后再 exp
        log_proba -= log_proba.max(axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """预测类别：取对数概率最大的类别。"""
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def score(self, X, y):
        """计算准确率。"""
        return np.mean(self.predict(X) == y)


# ============================================================
# 第 2 步：高斯朴素贝叶斯（GaussianNB）
# ============================================================

class GaussianNB:
    """高斯朴素贝叶斯 — 假设连续特征在每个类别下服从正态分布。

    对每个类别的每个特征，估计均值和方差，
    然后使用高斯概率密度函数计算条件概率。
    """

    def __init__(self, var_smoothing=1e-9):
        """
        Args:
            var_smoothing: 加到方差上的微小值，防止除以零。
        """
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.means_ = None   # 每个类别每个特征的均值
        self.vars_ = None    # 每个类别每个特征的方差
        self.priors_ = None  # 类别先验概率

    def fit(self, X, y):
        """训练模型：对每个类别，计算每个特征的均值和方差。"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            # 方差加上平滑项，防止零方差导致除零错误
            self.vars_[i] = X_c.var(axis=0) + self.var_smoothing
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self

    def _log_likelihood(self, X):
        """计算高斯概率密度的对数似然。

        log P(x_i | c) = -0.5 * log(2πσ²) - (x - μ)² / (2σ²)
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        log_proba = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            diff = X - self.means_[i]
            # 高斯概率密度的对数
            log_prob_features = (
                -0.5 * np.log(2 * np.pi * self.vars_[i])
                - 0.5 * (diff ** 2) / self.vars_[i]
            )
            # 对所有特征求和 + 类别先验
            log_proba[:, i] = log_prob_features.sum(axis=1) + np.log(self.priors_[i])

        return log_proba

    def predict(self, X):
        """预测类别。"""
        log_proba = self._log_likelihood(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def predict_proba(self, X):
        """计算概率值。"""
        log_proba = self._log_likelihood(X)
        log_proba -= log_proba.max(axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def score(self, X, y):
        """计算准确率。"""
        return np.mean(self.predict(X) == y)


# ============================================================
# 第 3 步：伯努利朴素贝叶斯（BernoulliNB）
# ============================================================

class BernoulliNB:
    """伯努利朴素贝叶斯 — 适用于二元特征（出现/不出现）。

    与多项式 NB 不同，伯努利 NB 显式建模特征的不出现。
    对于"词是否出现在文档中"这类二元特征，效果往往更好。
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None  # P(特征出现 | 类别)

    def fit(self, X, y):
        """训练：将特征二值化后统计出现频率。"""
        # 二值化：非零即 1
        X_binary = (X > 0).astype(float)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X_binary[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X_binary.shape[0])

            # 特征出现的文档数 + 平滑
            feature_count = X_c.sum(axis=0) + self.alpha
            total_docs = X_c.shape[0] + 2 * self.alpha

            self.feature_log_prob_[i] = np.log(feature_count / total_docs)

        return self

    def predict_log_proba(self, X):
        """计算对数概率。

        需要同时考虑特征出现和不出现的贡献：
        log P(c | x) ∝ log P(c) + Σ [x_i * log P(f_i|c) + (1-x_i) * log(1-P(f_i|c))]
        """
        X_binary = (X > 0).astype(float)
        n_classes = len(self.classes_)
        n_samples = X_binary.shape[0]

        log_proba = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # 出现的概率对数
            log_prob_present = X_binary * self.feature_log_prob_[i]
            # 不出现的概率对数
            log_prob_absent = (1 - X_binary) * np.log(1 - np.exp(self.feature_log_prob_[i]))
            log_proba[:, i] = log_prob_present.sum(axis=1) + log_prob_absent.sum(axis=1) + self.class_log_prior_[i]

        return log_proba

    def predict(self, X):
        """预测类别。"""
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def score(self, X, y):
        """计算准确率。"""
        return np.mean(self.predict(X) == y)


# ============================================================
# 辅助函数
# ============================================================

def make_text_data(n_samples=1000, n_features=200, seed=42):
    """生成模拟文本分类数据。

    模拟两类文章：科技类和体育类。
    - 特征 0-39：科技类高频词（如"算法"、"芯片"）
    - 特征 40-79：两类都有的中等词频
    - 特征 80-119：体育类高频词（如"进球"、"冠军"）
    - 其余：噪声词
    """
    rng = np.random.RandomState(seed)

    # 科技类词频权重
    tech_weights = np.zeros(n_features)
    tech_weights[:40] = rng.uniform(3, 10, 40)      # 科技高频词
    tech_weights[40:80] = rng.uniform(0.5, 2, 40)   # 共有中等词
    tech_weights[80:] = rng.uniform(0.1, 1, 120)     # 噪声词

    # 体育类词频权重
    sports_weights = np.zeros(n_features)
    sports_weights[:40] = rng.uniform(0.1, 1, 40)     # 噪声词
    sports_weights[40:80] = rng.uniform(0.5, 2, 40)   # 共有中等词
    sports_weights[80:120] = rng.uniform(3, 10, 40)   # 体育高频词
    sports_weights[120:] = rng.uniform(0.1, 1, 80)    # 噪声词

    n_tech = n_samples // 2
    n_sports = n_samples - n_tech

    # 用泊松分布模拟词频计数
    X_tech = rng.poisson(tech_weights, (n_tech, n_features)).astype(float)
    X_sports = rng.poisson(sports_weights, (n_sports, n_features)).astype(float)

    X = np.vstack([X_tech, X_sports])
    y = np.array([0] * n_tech + [1] * n_sports)

    shuffle_idx = rng.permutation(n_samples)
    return X[shuffle_idx], y[shuffle_idx]


def make_continuous_data(n_samples=300, seed=42):
    """生成模拟连续特征数据（类似鸢尾花数据集）。"""
    rng = np.random.RandomState(seed)
    n_per_class = n_samples // 3

    # 三类，每类有不同的均值和方差，模拟三个鸢尾花品种
    class_0 = rng.multivariate_normal(
        [5.0, 3.4, 1.4, 0.2],
        np.diag([0.12, 0.14, 0.03, 0.01]),
        n_per_class,
    )
    class_1 = rng.multivariate_normal(
        [5.9, 2.8, 4.3, 1.3],
        np.diag([0.27, 0.10, 0.22, 0.04]),
        n_per_class,
    )
    class_2 = rng.multivariate_normal(
        [6.6, 3.0, 5.6, 2.0],
        np.diag([0.40, 0.10, 0.30, 0.08]),
        n_per_class,
    )

    X = np.vstack([class_0, class_1, class_2])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """将数据集随机划分为训练集和测试集。"""
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def print_separator(title):
    """打印分隔线标题。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ============================================================
# 演示 1：多项式 NB 处理文本分类
# ============================================================

def demo_multinomial():
    print_separator("演示 1：多项式朴素贝叶斯 — 文本分类")

    X, y = make_text_data(n_samples=1200, n_features=200, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.25, seed=42)

    print(f"训练样本数：{X_train.shape[0]}")
    print(f"测试样本数：{X_test.shape[0]}")
    print(f"特征数（词表大小）：{X_train.shape[1]}")
    print(f"类别：科技 (0) vs 体育 (1)")
    print()

    # 训练模型
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)

    train_acc = mnb.score(X_train, y_train)
    test_acc = mnb.score(X_test, y_test)
    print(f"从零实现的 MultinomialNB：")
    print(f"  训练准确率：{train_acc:.4f}")
    print(f"  测试准确率：{test_acc:.4f}")

    # 展示前 5 个样本的预测概率
    proba = mnb.predict_proba(X_test[:5])
    print(f"\n前 5 个测试样本的预测概率：")
    for i in range(5):
        pred = "科技" if proba[i, 0] > proba[i, 1] else "体育"
        print(f"  样本 {i}: P(科技)={proba[i, 0]:.4f}, P(体育)={proba[i, 1]:.4f} -> 预测：{pred}")

    # 对比不同平滑系数的效果
    print(f"\n不同平滑系数 (alpha) 的对比：")
    for alpha in [0.01, 0.1, 1.0, 5.0, 10.0]:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"  alpha={alpha:5.2f} -> 测试准确率：{acc:.4f}")


# ============================================================
# 演示 2：高斯 NB 处理连续特征
# ============================================================

def demo_gaussian():
    print_separator("演示 2：高斯朴素贝叶斯 — 连续特征分类")

    X, y = make_continuous_data(n_samples=450, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.25, seed=42)

    print(f"训练样本数：{X_train.shape[0]}")
    print(f"测试样本数：{X_test.shape[0]}")
    print(f"特征数：{X_train.shape[1]}")
    print(f"类别：0, 1, 2（模拟鸢尾花三品种）")
    print()

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    train_acc = gnb.score(X_train, y_train)
    test_acc = gnb.score(X_test, y_test)
    print(f"从零实现的 GaussianNB：")
    print(f"  训练准确率：{train_acc:.4f}")
    print(f"  测试准确率：{test_acc:.4f}")

    # 展示学到的参数
    print(f"\n学到的参数：")
    for i, c in enumerate(gnb.classes_):
        print(f"  类别 {c}：")
        print(f"    均值：{gnb.means_[i].round(3)}")
        print(f"    方差：{gnb.vars_[i].round(4)}")
        print(f"    先验：{gnb.priors_[i]:.3f}")


# ============================================================
# 演示 3：伯努利 NB 与多项式 NB 对比
# ============================================================

def demo_bernoulli():
    print_separator("演示 3：伯努利朴素贝叶斯 — 二元特征")

    X, y = make_text_data(n_samples=1000, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=42)

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)
    mnb_acc = mnb.score(X_test, y_test)

    bnb = BernoulliNB(alpha=1.0)
    bnb.fit(X_train, y_train)
    bnb_acc = bnb.score(X_test, y_test)

    print(f"样本数：{X.shape[0]}，特征数：{X.shape[1]}")
    print(f"MultinomialNB（基于词频）准确率：{mnb_acc:.4f}")
    print(f"BernoulliNB（基于出现/不出现）准确率：{bnb_acc:.4f}")
    print(f"胜出：{'BernoulliNB' if bnb_acc > mnb_acc else 'MultinomialNB'}")
    print()
    print("说明：在短文本场景下，BernoulliNB 往往更有优势，")
    print("      因为词频信息噪声较大，出现/不出现更稳定。")


# ============================================================
# 演示 4：与 sklearn 的对比验证
# ============================================================

def demo_sklearn_comparison():
    print_separator("演示 4：与 scikit-learn 对比验证")

    try:
        from sklearn.naive_bayes import GaussianNB as SklearnGNB
        from sklearn.naive_bayes import MultinomialNB as SklearnMNB
    except ImportError:
        print("未安装 scikit-learn，跳过此演示。")
        return

    # 多项式 NB 对比
    X, y = make_text_data(n_samples=1000, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=42)

    our_mnb = MultinomialNB(alpha=1.0)
    our_mnb.fit(X_train, y_train)
    our_mnb_acc = our_mnb.score(X_test, y_test)

    sklearn_mnb = SklearnMNB(alpha=1.0)
    sklearn_mnb.fit(X_train, y_train)
    sklearn_mnb_acc = sklearn_mnb.score(X_test, y_test)

    print("MultinomialNB 对比：")
    print(f"  我们的实现：   准确率 = {our_mnb_acc:.4f}")
    print(f"  scikit-learn： 准确率 = {sklearn_mnb_acc:.4f}")
    print(f"  差异：{abs(our_mnb_acc - sklearn_mnb_acc):.4f}")

    # 高斯 NB 对比
    X, y = make_continuous_data(n_samples=450, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=42)

    our_gnb = GaussianNB()
    our_gnb.fit(X_train, y_train)
    our_gnb_acc = our_gnb.score(X_test, y_test)

    sklearn_gnb = SklearnGNB()
    sklearn_gnb.fit(X_train, y_train)
    sklearn_gnb_acc = sklearn_gnb.score(X_test, y_test)

    print(f"\nGaussianNB 对比：")
    print(f"  我们的实现：   准确率 = {our_gnb_acc:.4f}")
    print(f"  scikit-learn： 准确率 = {sklearn_gnb_acc:.4f}")
    print(f"  差异：{abs(our_gnb_acc - sklearn_gnb_acc):.4f}")


# ============================================================
# 演示 5：训练集大小对准确率的影响
# ============================================================

def demo_training_size():
    print_separator("演示 5：训练集大小对准确率的影响")

    X_full, y_full = make_text_data(n_samples=2000, n_features=200, seed=42)
    X_test_full = X_full[1600:]
    y_test_full = y_full[1600:]

    print(f"{'训练集大小':>12}  {'准确率':>10}")
    print(f"{'-' * 24}")

    for n_train in [20, 50, 100, 200, 500, 1000, 1600]:
        X_train = X_full[:n_train]
        y_train = y_full[:n_train]

        mnb = MultinomialNB(alpha=1.0)
        mnb.fit(X_train, y_train)
        acc = mnb.score(X_test_full, y_test_full)
        print(f"{n_train:>12}  {acc:>10.4f}")

    print()
    print("观察：即使只有 20 个训练样本，NB 也能达到不错的准确率。")
    print("      这是 NB 的核心优势——在数据稀缺时表现稳健。")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    demo_multinomial()
    demo_gaussian()
    demo_bernoulli()
    demo_sklearn_comparison()
    demo_training_size()
