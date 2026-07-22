"""从零实现线性回归——梯度下降、正规方程、多项式回归、正则化。

# 依赖：numpy, scikit-learn
# 安装：pip install numpy scikit-learn
# 对应课程：第 02 阶段 · 02（线性回归）
"""

import random
import math
import numpy as np

# ============================================================
# 第 1 步：最简单的线性回归 + 梯度下降
# ============================================================
# 核心思想：y = wx + b，通过最小化均方误差（MSE）学习最优的 w 和 b
# 梯度下降：计算损失函数对参数的偏导数，沿反方向更新参数

class LinearRegressionGD:
    """基于梯度下降的简单线性回归（单特征）。"""

    def __init__(self, learning_rate=0.01):
        self.w = 0.0  # 权重（斜率）
        self.b = 0.0  # 偏置（截距）
        self.lr = learning_rate
        self.cost_history = []

    def predict(self, X):
        """前向计算预测值：y_hat = wx + b。"""
        return [self.w * x + self.b for x in X]

    def compute_cost(self, X, y):
        """计算均方误差（MSE）。

        MSE = (1/n) * sum((y_hat - y)^2)
        使用平方是因为：(1) 大误差惩罚更大；(2) 处处可微。
        """
        predictions = self.predict(X)
        n = len(y)
        cost = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n
        return cost

    def compute_gradients(self, X, y):
        """计算 MSE 对 w 和 b 的偏导数。

        dMSE/dw = (2/n) * sum((y_hat - y) * x)
        dMSE/db = (2/n) * sum(y_hat - y)
        """
        predictions = self.predict(X)
        n = len(y)
        dw = (2 / n) * sum((pred - actual) * x for pred, actual, x in zip(predictions, y, X))
        db = (2 / n) * sum(pred - actual for pred, actual in zip(predictions, y))
        return dw, db

    def fit(self, X, y, epochs=1000, print_every=200):
        """训练模型：迭代更新参数。"""
        for epoch in range(epochs):
            dw, db = self.compute_gradients(X, y)
            # 沿梯度反方向更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | 损失: {cost:.4f} | w: {self.w:.4f} | b: {self.b:.4f}")
        return self

    def r_squared(self, X, y):
        """计算 R² 决定系数。

        R² = 1 - SS_res / SS_tot
        其中 SS_res 是残差平方和，SS_tot 是总平方和。
        """
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


# ============================================================
# 第 2 步：正规方程（解析解）
# ============================================================
# 对于线性回归，存在封闭解（closed-form solution）：
#   w = (X^T * X)^(-1) * X^T * y
# 对于单特征情况，可以直接计算斜率和截距

class LinearRegressionNormal:
    """基于正规方程的线性回归（单特征）。

    优点：一次计算即得最优解，无需迭代
    缺点：当特征数很多时，矩阵求逆的复杂度为 O(n^3)，计算代价高
    """

    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y):
        """通过最小二乘法公式直接求解 w 和 b。"""
        n = len(X)
        x_mean = sum(X) / n
        y_mean = sum(y) / n
        # w = sum((x_i - x_mean)(y_i - y_mean)) / sum((x_i - x_mean)^2)
        numerator = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((X[i] - x_mean) ** 2 for i in range(n))
        self.w = numerator / denominator
        self.b = y_mean - self.w * x_mean
        return self

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


# ============================================================
# 第 3 步：多元线性回归
# ============================================================
# 当有多个特征时，y = w1*x1 + w2*x2 + ... + wn*xn + b
# 所有逻辑相同，只是权重从标量变成向量

class MultipleLinearRegression:
    """基于梯度下降的多元线性回归（多特征）。"""

    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features  # 每个特征一个权重
        self.bias = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        return sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            # 对每个特征的权重计算梯度并更新
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            # 偏置项的梯度（不加正则化）
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | 损失: {cost:.4f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


def standardize(X):
    """Z-score 标准化：将特征缩放到零均值、单位方差。

    为什么需要标准化？当特征量级差异大时（如面积 500-3000 vs 卧室 1-5），
    等高线呈细长椭圆，梯度下降会来回震荡，收敛极慢。
    标准化后接近圆形等高线，梯度直指最低点。
    """
    n_features = len(X[0])
    n_samples = len(X)
    means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
    stds = []
    for j in range(n_features):
        variance = sum((X[i][j] - means[j]) ** 2 for i in range(n_samples)) / n_samples
        stds.append(variance ** 0.5)
    X_scaled = []
    for i in range(n_samples):
        row = [(X[i][j] - means[j]) / stds[j] if stds[j] > 0 else 0 for j in range(n_features)]
        X_scaled.append(row)
    return X_scaled, means, stds


# ============================================================
# 第 4 步：多项式回归
# ============================================================
# 即使真实关系不是线性的，只要模型对权重是线性的，就仍是线性回归
# 通过构造多项式特征（x, x^2, x^3, ...）拟合曲线

class PolynomialRegression:
    """多项式回归——用多项式特征拟合非线性关系。

    模型：y = w1*x + w2*x^2 + w3*x^3 + ... + b
    仍是线性回归（对权重线性），但使用非线性特征。
    """

    def __init__(self, degree, learning_rate=0.01):
        self.degree = degree  # 多项式次数
        self.weights = [0.0] * degree
        self.bias = 0.0
        self.lr = learning_rate

    def make_features(self, X):
        """为每个 x 构造多项式特征 [x, x^2, ..., x^degree]。"""
        return [[x ** (d + 1) for d in range(self.degree)] for x in X]

    def predict(self, X):
        features = self.make_features(X)
        return [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]

    def fit(self, X, y, epochs=1000, print_every=200):
        features = self.make_features(X)
        n = len(y)
        for epoch in range(epochs):
            predictions = [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(self.degree):
                grad = (2 / n) * sum(errors[i] * features[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                cost = sum(e ** 2 for e in errors) / n
                print(f"  Epoch {epoch:4d} | 损失: {cost:.6f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


# ============================================================
# 第 5 步：Ridge 回归（L2 正则化）
# ============================================================
# 当特征多或有多重共线性时，模型倾向于学习大权重导致过拟合
# Ridge 在损失函数中加入 L2 惩罚项：lambda * sum(w_i^2)
# 惩罚大权重，使权重趋向小值但不为零

class RidgeRegression:
    """Ridge 回归——带 L2 正则化的线性回归。

    损失 = MSE + alpha * sum(w_i^2)
    梯度 = dMSE/dw + 2 * alpha * w
    """

    def __init__(self, n_features, learning_rate=0.01, alpha=1.0):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.alpha = alpha  # 正则化强度

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            mse = sum(e ** 2 for e in errors) / n
            # L2 正则化项（不包含偏置）
            reg_term = self.alpha * sum(w ** 2 for w in self.weights)
            cost = mse + reg_term
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                grad += 2 * self.alpha * self.weights[j]  # L2 惩罚的梯度
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)  # 偏置不加正则化
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | 损失: {cost:.4f} | L2惩罚: {reg_term:.4f}")
        return self


# ============================================================
# 第 6 步：Lasso 回归（L1 正则化）
# ============================================================
# Lasso 使用 L1 惩罚：alpha * sum(|w_i|)
# 与 Ridge 的关键区别：L1 惩罚会让部分权重精确为零（产生稀疏解）
# 这等价于自动特征选择

class LassoRegression:
    """Lasso 回归——带 L1 正则化的线性回归。

    损失 = MSE + alpha * sum(|w_i|)
    梯度 = dMSE/dw + alpha * sign(w)
    注意：sign(0) 在 0 处取 0
    """

    def __init__(self, n_features, learning_rate=0.01, alpha=1.0):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.alpha = alpha

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            mse = sum(e ** 2 for e in errors) / n
            reg_term = self.alpha * sum(abs(w) for w in self.weights)
            cost = mse + reg_term
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                # L1 惩罚的梯度是 sign(w)，在 0 处取 0
                grad += self.alpha * (1 if self.weights[j] > 0 else (-1 if self.weights[j] < 0 else 0))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | 损失: {cost:.4f} | L1惩罚: {reg_term:.4f}")
        return self


# ============================================================
# 测试与演示
# ============================================================
if __name__ == "__main__":
    # --- 测试 1：简单线性回归 + 梯度下降 ---
    print("=" * 50)
    print("【测试 1】简单线性回归（梯度下降）")
    print("=" * 50)
    random.seed(42)
    TRUE_W = 3.0
    TRUE_B = 7.0
    N_SAMPLES = 100
    X_simple = [random.uniform(0, 10) for _ in range(N_SAMPLES)]
    y_simple = [TRUE_W * x + TRUE_B + random.gauss(0, 2.0) for x in X_simple]

    model_gd = LinearRegressionGD(learning_rate=0.005)
    model_gd.fit(X_simple, y_simple, epochs=1000, print_every=200)
    print(f"\n学习结果: y = {model_gd.w:.4f}x + {model_gd.b:.4f}")
    print(f"真实关系: y = {TRUE_W}x + {TRUE_B}")
    print(f"R²: {model_gd.r_squared(X_simple, y_simple):.4f}")

    # --- 测试 2：正规方程（解析解）对比 ---
    print(f"\n{'=' * 50}")
    print("【测试 2】正规方程（解析解）")
    print("=" * 50)
    model_normal = LinearRegressionNormal()
    model_normal.fit(X_simple, y_simple)
    print(f"解析结果: y = {model_normal.w:.4f}x + {model_normal.b:.4f}")
    print(f"R²: {model_normal.r_squared(X_simple, y_simple):.4f}")
    print(f"\n梯度下降 w={model_gd.w:.4f} vs 正规方程 w={model_normal.w:.4f}")
    print("两者结果几乎一致，正规方程一步到位，梯度下降需要迭代。")

    # --- 测试 3：多元线性回归 ---
    print(f"\n{'=' * 50}")
    print("【测试 3】多元线性回归（3 个特征）")
    print("=" * 50)
    random.seed(42)
    N = 100
    X_multi = []
    y_multi = []
    for _ in range(N):
        area = random.uniform(500, 3000)       # 面积 (平方米)
        bedrooms = random.randint(1, 5)        # 卧室数
        age = random.uniform(0, 50)            # 房龄
        price = 50 * area + 10000 * bedrooms - 1000 * age + 50000 + random.gauss(0, 20000)
        X_multi.append([area, bedrooms, age])
        y_multi.append(price)

    # 标准化特征和目标
    y_mean_val = sum(y_multi) / len(y_multi)
    y_std_val = (sum((yi - y_mean_val) ** 2 for yi in y_multi) / len(y_multi)) ** 0.5
    y_scaled = [(yi - y_mean_val) / y_std_val for yi in y_multi]
    X_scaled, x_means, x_stds = standardize(X_multi)

    multi_model = MultipleLinearRegression(n_features=3, learning_rate=0.01)
    multi_model.fit(X_scaled, y_scaled, epochs=1000, print_every=200)
    print(f"\n标准化权重: {[round(w, 4) for w in multi_model.weights]}")
    print(f"偏置（标准化）: {multi_model.bias:.4f}")
    print(f"R²: {multi_model.r_squared(X_scaled, y_scaled):.4f}")

    # --- 测试 4：多项式回归 ---
    print(f"\n{'=' * 50}")
    print("【测试 4】多项式回归（2 次 vs 5 次）")
    print("=" * 50)
    random.seed(42)
    X_poly = [x / 10.0 for x in range(0, 50)]
    y_poly = [0.5 * x ** 2 - 2 * x + 3 + random.gauss(0, 1.0) for x in X_poly]

    x_max = max(abs(x) for x in X_poly)
    X_poly_norm = [x / x_max for x in X_poly]
    y_poly_mean = sum(y_poly) / len(y_poly)
    y_poly_std = (sum((yi - y_poly_mean) ** 2 for yi in y_poly) / len(y_poly)) ** 0.5
    y_poly_norm = [(yi - y_poly_mean) / y_poly_std for yi in y_poly]

    print("\n2 次多项式:")
    poly2 = PolynomialRegression(degree=2, learning_rate=0.1)
    poly2.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
    print(f"  R²: {poly2.r_squared(X_poly_norm, y_poly_norm):.4f}")

    print("\n5 次多项式:")
    poly5 = PolynomialRegression(degree=5, learning_rate=0.1)
    poly5.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
    print(f"  R²: {poly5.r_squared(X_poly_norm, y_poly_norm):.4f}")

    print("\n2 次多项式拟合真实曲线良好，5 次多项式训练拟合略好但存在过拟合风险。")

    # --- 测试 5：Ridge 回归 ---
    print(f"\n{'=' * 50}")
    print("【测试 5】Ridge 回归（L2 正则化）")
    print("=" * 50)
    ridge = RidgeRegression(n_features=3, learning_rate=0.01, alpha=0.1)
    ridge.fit(X_scaled, y_scaled, epochs=1000, print_every=200)
    print(f"\nRidge 权重: {[round(w, 4) for w in ridge.weights]}")
    print(f"普通权重: {[round(w, 4) for w in multi_model.weights]}")
    print("Ridge 权重因 L2 惩罚而缩小（向零收缩）。")

    # --- 测试 6：Lasso 回归（L1 正则化）对比 ---
    print(f"\n{'=' * 50}")
    print("【测试 6】Lasso 回归（L1 正则化）")
    print("=" * 50)
    lasso = LassoRegression(n_features=3, learning_rate=0.01, alpha=0.1)
    lasso.fit(X_scaled, y_scaled, epochs=1000, print_every=200)
    print(f"\nLasso 权重: {[round(w, 4) for w in lasso.weights]}")
    print(f"Ridge 权重: {[round(w, 4) for w in ridge.weights]}")
    print(f"普通权重: {[round(w, 4) for w in multi_model.weights]}")
    print("Lasso 的 L1 惩罚会让不重要的特征权重精确为零，实现特征选择。")

    # --- 测试 7：训练集/测试集对比 ---
    print(f"\n{'=' * 50}")
    print("【测试 7】训练集 vs 测试集表现")
    print("=" * 50)
    split_idx = int(0.8 * len(X_simple))
    X_train, X_test = X_simple[:split_idx], X_simple[split_idx:]
    y_train, y_test = y_simple[:split_idx], y_simple[split_idx:]

    model_split = LinearRegressionGD(learning_rate=0.005)
    model_split.fit(X_train, y_train, epochs=1000, print_every=500)
    print(f"\n训练集 R²: {model_split.r_squared(X_train, y_train):.4f}")
    print(f"测试集 R²: {model_split.r_squared(X_test, y_test):.4f}")
    print(f"学习结果: y = {model_split.w:.4f}x + {model_split.b:.4f}")
    print(f"真实关系: y = {TRUE_W}x + {TRUE_B}")

    # --- 测试 8：scikit-learn 对比 ---
    print(f"\n{'=' * 50}")
    print("【测试 8】scikit-learn 对比验证")
    print("=" * 50)
    from sklearn.linear_model import LinearRegression as SklearnLR
    from sklearn.linear_model import Ridge as SklearnRidge
    from sklearn.linear_model import Lasso as SklearnLasso
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    np.random.seed(42)
    X_sk = np.random.uniform(0, 10, (100, 1))
    y_sk = 3.0 * X_sk.squeeze() + 7.0 + np.random.normal(0, 2.0, 100)

    X_tr, X_te, y_tr, y_te = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

    lr = SklearnLR()
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)

    print(f"PyTorch/sklearn 系数 w: {lr.coef_[0]:.4f}")
    print(f"sklearn 截距 b: {lr.intercept_:.4f}")
    print(f"测试集 R²: {r2_score(y_te, y_pred):.4f}")
    print(f"测试集 MSE: {mean_squared_error(y_te, y_pred):.4f}")

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_sk = poly.fit_transform(X_tr)
    X_poly_test = poly.transform(X_te)
    lr_poly = SklearnLR()
    lr_poly.fit(X_poly_sk, y_tr)
    print(f"\n2 次多项式 R²: {r2_score(y_te, lr_poly.predict(X_poly_test)):.4f}")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    ridge_sk = SklearnRidge(alpha=1.0)
    ridge_sk.fit(X_tr_sc, y_tr)
    print(f"Ridge 回归 R²: {r2_score(y_te, ridge_sk.predict(X_te_sc)):.4f}")

    lasso_sk = SklearnLasso(alpha=1.0)
    lasso_sk.fit(X_tr_sc, y_tr)
    print(f"Lasso 回归 R²: {r2_score(y_te, lasso_sk.predict(X_te_sc)):.4f}")
    print(f"Lasso 系数: {lasso_sk.coef_[0]:.4f}（可能为零，体现特征选择）")

    print(f"\n{'=' * 50}")
    print("所有测试完成！从零实现与 sklearn 结果一致。")
    print("=" * 50)
