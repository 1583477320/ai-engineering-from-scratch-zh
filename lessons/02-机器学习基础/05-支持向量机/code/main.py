# main.py — 支持向量机（SVM）从零实现
# 依赖：numpy>=1.24
# 安装：pip install numpy
# 对应课程：第 02 阶段 · 05（支持向量机）

import math
import random


# === 向量工具函数 ===

def dot(a, b):
    """计算两个向量的点积。"""
    return sum(ai * bi for ai, bi in zip(a, b))


def vec_sub(a, b):
    """向量减法。"""
    return [ai - bi for ai, bi in zip(a, b)]


def vec_norm(a):
    """计算向量的 L2 范数（长度）。"""
    return math.sqrt(dot(a, a))


# === 核函数 ===

def linear_kernel(x, z):
    """线性核：原始空间中的点积。"""
    return dot(x, z)


def polynomial_kernel(x, z, degree=3, c=1.0):
    """多项式核：捕获特征之间的交互关系。

    K(x, z) = (x · z + c)^degree
    """
    return (dot(x, z) + c) ** degree


def rbf_kernel(x, z, gamma=0.5):
    """RBF 核（高斯核）：基于局部相似度的非线性核。

    K(x, z) = exp(-γ * ||x - z||²)
    距离近的点核值接近 1，距离远的点核值接近 0。
    """
    diff = vec_sub(x, z)
    return math.exp(-gamma * dot(diff, diff))


# === 铰链损失 ===

def hinge_loss(X, y, w, b):
    """计算铰链损失——SVM 的核心损失函数。

    L = (1/n) * Σ max(0, 1 - y_i * (w·x_i + b))

    当样本被正确分类且位于间隔之外时，损失为零。
    当样本位于间隔内或分类错误时，损失线性增长。
    """
    n = len(X)
    total = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total += max(0.0, 1.0 - margin)
    return total / n


def svm_objective(X, y, w, b, lambda_param):
    """SVM 的完整目标函数：L2 正则化 + 铰链损失。

    J = (λ/2) * ||w||² + (1/n) * Σ max(0, 1 - y_i * (w·x_i + b))
    """
    reg = 0.5 * lambda_param * dot(w, w)
    loss = hinge_loss(X, y, w, b)
    return reg + loss


# === 线性 SVM（原始形式，梯度下降训练） ===

class LinearSVM:
    """线性支持向量机——通过梯度下降最小化正则化铰链损失。

    原始形式（primal formulation）直接优化 w 和 b，
    每轮迭代复杂度 O(n*d)，适合大规模稀疏数据。
    """

    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def fit(self, X, y):
        """训练线性 SVM。

        对每个样本：
        - 若 y_i * (w·x_i + b) >= 1：样本在间隔外，仅做权重衰减（L2 正则化）
        - 否则：样本在间隔内或分类错误，更新 w 和 b 以减小损失

        Args:
            X: 训练数据，形状 (n_samples, n_features)
            y: 标签（取值为 +1 或 -1）
        """
        n_features = len(X[0])
        n_samples = len(X)
        self.w = [0.0] * n_features
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.n_epochs):
            indices = list(range(n_samples))
            random.shuffle(indices)

            for i in indices:
                margin = y[i] * (dot(self.w, X[i]) + self.b)

                if margin >= 1:
                    # 样本在间隔外：只应用权重衰减
                    self.w = [
                        wj - self.lr * self.lambda_param * wj
                        for wj in self.w
                    ]
                else:
                    # 样本在间隔内或分类错误：更新 w 和 b
                    self.w = [
                        wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                        for j, wj in enumerate(self.w)
                    ]
                    self.b -= self.lr * (-y[i])

            if epoch % 100 == 0 or epoch == self.n_epochs - 1:
                loss = svm_objective(X, y, self.w, self.b, self.lambda_param)
                self.loss_history.append((epoch, loss))

    def predict(self, X):
        """预测样本类别（+1 或 -1）。"""
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]

    def decision_function(self, X):
        """计算决策函数值 w·x + b（有符号距离）。"""
        return [dot(self.w, x) + self.b for x in X]

    def margin_width(self):
        """计算间隔宽度：2 / ||w||。

        ||w|| 越小，间隔越宽，模型越"自信"。
        """
        w_norm = vec_norm(self.w)
        if w_norm == 0:
            return 0.0
        return 2.0 / w_norm

    def find_support_vectors(self, X, y, tol=0.1):
        """识别支持向量——间隔边界上的样本。

        支持向量满足 y_i * (w·x_i + b) ≈ 1。
        只有这些样本决定了决策边界。

        Args:
            tol：容差，判断 margin 是否接近 1

        Returns:
            支持向量在 X 中的索引列表
        """
        svs = []
        for i in range(len(X)):
            margin = y[i] * (dot(self.w, X[i]) + self.b)
            if abs(margin - 1.0) < tol:
                svs.append(i)
        return svs


# === SMO 算法（简化版） ===

class SVMWithKernel:
    """使用核技巧的 SVM——通过简化版 SMO 算法求解对偶问题。

    对偶形式（dual formulation）只涉及样本间的点积，
    因此可以用核函数替换点积，处理非线性边界。
    """

    def __init__(self, kernel_fn=linear_kernel, C=1.0, gamma=0.5,
                 n_epochs=100, tol=1e-3):
        self.kernel_fn = kernel_fn
        self.C = C          # 正则化参数：越大越不容忍分类错误
        self.gamma = gamma  # RBF 核的宽度参数
        self.n_epochs = n_epochs
        self.tol = tol
        self.alpha = None   # 拉格朗日乘子
        self.b = 0.0
        self.X_train = None
        self.y_train = None

    def _kernel(self, x, z):
        """调用核函数，传入 gamma 参数（如果适用）。"""
        if self.kernel_fn == rbf_kernel:
            return rbf_kernel(x, z, gamma=self.gamma)
        return self.kernel_fn(x, z)

    def fit(self, X, y):
        """使用简化版 SMO 算法训练 SVM。

        SMO（序列最小优化）：每次选择两个拉格朗日乘子 α_i 和 α_j，
        固定其他乘子，求解一个二次规划子问题，逐步收敛到全局最优。

        Args:
            X: 训练数据
            y: 标签（+1 或 -1）
        """
        n = len(X)
        self.alpha = [0.0] * n
        self.b = 0.0
        self.X_train = X
        self.y_train = y

        # 预计算核矩阵（避免重复计算）
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                val = self._kernel(X[i], X[j])
                K[i][j] = val
                K[j][i] = val

        for epoch in range(self.n_epochs):
            num_changed = 0

            for i in range(n):
                # 计算预测值 f(x_i)
                f_i = sum(
                    self.alpha[j] * y[j] * K[i][j] for j in range(n)
                ) + self.b
                # 误差 E_i = f(x_i) - y_i
                E_i = f_i - y[i]

                # 检查是否违反 KKT 条件
                if ((y[i] * E_i < -self.tol and self.alpha[i] < self.C) or
                        (y[i] * E_i > self.tol and self.alpha[i] > 0)):
                    # 随机选择 j ≠ i
                    j = random.choice([k for k in range(n) if k != i])
                    f_j = sum(
                        self.alpha[k] * y[k] * K[j][k] for k in range(n)
                    ) + self.b
                    E_j = f_j - y[j]

                    # 保存旧的 α 值
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    # 计算上下界 L 和 H（裁剪范围）
                    if y[i] != y[j]:
                        L = max(0.0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0.0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    # η = 2 * K(x_i, x_j) - K(x_i, x_i) - K(x_j, x_j)
                    eta = 2 * K[i][j] - K[i][i] - K[j][j]
                    if eta >= 0:
                        continue

                    # 更新 α_j
                    self.alpha[j] = alpha_j_old - y[j] * (E_i - E_j) / eta
                    # 裁剪 α_j 到 [L, H] 区间
                    self.alpha[j] = max(L, min(H, self.alpha[j]))

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # 更新 α_i（保持等式约束 Σ α_i * y_i = 0）
                    self.alpha[i] = alpha_i_old + y[i] * y[j] * (
                        alpha_j_old - self.alpha[j]
                    )

                    # 更新偏置 b
                    b1 = (self.b - E_i
                          - y[i] * (self.alpha[i] - alpha_i_old) * K[i][i]
                          - y[j] * (self.alpha[j] - alpha_j_old) * K[i][j])
                    b2 = (self.b - E_j
                          - y[i] * (self.alpha[i] - alpha_i_old) * K[i][j]
                          - y[j] * (self.alpha[j] - alpha_j_old) * K[j][j])

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_changed += 1

            # 若无 α 更新，提前收敛
            if num_changed == 0:
                break

    def predict(self, X):
        """使用训练好的 SVM 预测新样本。"""
        predictions = []
        for x in X:
            result = sum(
                self.alpha[j] * self.y_train[j] * self._kernel(x, self.X_train[j])
                for j in range(len(self.X_train))
            ) + self.b
            predictions.append(1 if result >= 0 else -1)
        return predictions

    def find_support_vectors(self):
        """返回支持向量的索引（α_i > 0 的样本）。"""
        return [i for i in range(len(self.alpha)) if self.alpha[i] > 1e-5]


# === 辅助函数 ===

def accuracy(y_true, y_pred):
    """计算准确率。"""
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def generate_linear_data(n_samples=100, margin=1.0, seed=42):
    """生成线性可分数据集。"""
    random.seed(seed)
    X = []
    y = []
    for _ in range(n_samples):
        x1 = random.uniform(-3, 3)
        x2 = random.uniform(-3, 3)
        val = x1 + x2
        if val > margin / 2:
            X.append([x1, x2])
            y.append(1)
        elif val < -margin / 2:
            X.append([x1, x2])
            y.append(-1)
    return X, y


def generate_noisy_data(n_samples=200, noise=0.5, seed=42):
    """生成带噪声的不可分数据集。"""
    random.seed(seed)
    X = []
    y = []
    for _ in range(n_samples):
        x1 = random.uniform(-3, 3)
        x2 = random.uniform(-3, 3)
        val = x1 - 0.5 * x2 + random.gauss(0, noise)
        label = 1 if val > 0 else -1
        X.append([x1, x2])
        y.append(label)
    return X, y


def generate_circular_data(n_samples=200, seed=42):
    """生成环形（非线性可分）数据集。"""
    random.seed(seed)
    X = []
    y = []
    for _ in range(n_samples):
        r = random.uniform(0, 3)
        angle = random.uniform(0, 2 * math.pi)
        x1 = r * math.cos(angle) + random.gauss(0, 0.1)
        x2 = r * math.sin(angle) + random.gauss(0, 0.1)
        label = 1 if r > 1.5 else -1
        X.append([x1, x2])
        y.append(label)
    return X, y


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """将数据随机划分为训练集和测试集。"""
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return (
        [X[i] for i in train_idx],
        [y[i] for i in train_idx],
        [X[i] for i in test_idx],
        [y[i] for i in test_idx],
    )


def compute_kernel_matrix(X, kernel_fn, **kwargs):
    """计算核矩阵（Gram 矩阵）。

    K[i][j] = K(x_i, x_j)——所有样本对之间的核函数值。
    """
    n = len(X)
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = kernel_fn(X[i], X[j], **kwargs)
            K[i][j] = val
            K[j][i] = val
    return K


# === 演示函数 ===

def demo_hinge_loss():
    """演示铰链损失与逻辑损失的对比。"""
    print("=" * 65)
    print("铰链损失：SVM 的损失函数")
    print("=" * 65)
    print()

    margins = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    print(f"  {'y * f(x)':>10s}  {'铰链损失':>10s}  {'逻辑损失':>10s}  {'可视化':>20s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 20}")

    for m in margins:
        h_loss = max(0.0, 1.0 - m)
        l_loss = math.log(1 + math.exp(-m))
        bar_len = int(h_loss * 5)
        bar = "#" * bar_len
        print(f"  {m:>10.1f}  {h_loss:>10.3f}  {l_loss:>10.3f}  {bar}")

    print()
    print("  当 y*f(x) >= 1 时，铰链损失为零（间隔外）。")
    print("  逻辑损失永远不会精确为零——它使用所有样本。")
    print()


def demo_linear_svm():
    """演示线性 SVM 在可分数据上的训练。"""
    print("=" * 65)
    print("线性 SVM：最大间隔分类器")
    print("=" * 65)
    print()

    X, y = generate_linear_data(200, margin=1.0, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    print(f"  数据集：{len(X)} 个样本，线性可分")
    print(f"  训练集：{len(X_train)}  测试集：{len(X_test)}")
    print()

    svm = LinearSVM(lr=0.001, lambda_param=0.01, n_epochs=500)
    svm.fit(X_train, y_train)

    train_acc = accuracy(y_train, svm.predict(X_train))
    test_acc = accuracy(y_test, svm.predict(X_test))

    print(f"  权重 w: [{svm.w[0]:.4f}, {svm.w[1]:.4f}]")
    print(f"  偏置 b: {svm.b:.4f}")
    print(f"  间隔宽度: {svm.margin_width():.4f}")
    print(f"  训练集准确率: {train_acc:.4f}")
    print(f"  测试集准确率: {test_acc:.4f}")

    svs = svm.find_support_vectors(X_train, y_train, tol=0.3)
    print(f"  支持向量：{len(svs)} / {len(X_train)} 个训练样本")
    print()


def demo_c_parameter():
    """演示 C 参数（正则化强度）的影响。"""
    print("=" * 65)
    print("C 参数：正则化与拟合的权衡")
    print("=" * 65)
    print()

    X, y = generate_noisy_data(300, noise=0.8, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    print(f"  数据集：{len(X)} 个含噪声样本")
    print(f"  训练集：{len(X_train)}  测试集：{len(X_test)}")
    print()

    # C 与 lambda 的关系：lambda = 1 / (C * n)
    c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    print(f"  {'C':>8s}  {'λ':>10s}  {'训练准确率':>10s}  {'测试准确率':>10s}  {'间隔':>8s}  {'SV 数量':>8s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 8}")

    for c in c_values:
        lam = 1.0 / (c * len(X_train))
        svm = LinearSVM(lr=0.001, lambda_param=lam, n_epochs=500)
        svm.fit(X_train, y_train)

        train_acc = accuracy(y_train, svm.predict(X_train))
        test_acc = accuracy(y_test, svm.predict(X_test))
        margin = svm.margin_width()
        n_sv = len(svm.find_support_vectors(X_train, y_train, tol=0.3))

        print(f"  {c:>8.3f}  {lam:>10.6f}  {train_acc:>10.4f}  {test_acc:>10.4f}  "
              f"{margin:>8.4f}  {n_sv:>8d}")

    print()
    print("  小 C（大 λ）：间隔宽，容许更多错误，倾向于欠拟合。")
    print("  大 C（小 λ）：间隔窄，不容忍错误，倾向于过拟合。")
    print()


def demo_kernel_functions():
    """演示三种核函数在不同点上的相似度计算。"""
    print("=" * 65)
    print("核函数：在不同空间度量相似度")
    print("=" * 65)
    print()

    x = [1.0, 0.0]
    points = [
        ("同方向", [2.0, 0.0]),
        ("垂直", [0.0, 1.0]),
        ("相近", [1.1, 0.1]),
        ("远处同向", [5.0, 0.0]),
        ("反方向", [-1.0, 0.0]),
    ]

    print(f"  参考点: {x}")
    print()
    print(f"  {'点':<16s}  {'线性核':>8s}  {'多项式(d=2)':>12s}  {'多项式(d=3)':>12s}  {'RBF(γ=0.5)':>12s}")
    print(f"  {'-' * 16}  {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 12}")

    for name, z in points:
        k_lin = linear_kernel(x, z)
        k_p2 = polynomial_kernel(x, z, degree=2)
        k_p3 = polynomial_kernel(x, z, degree=3)
        k_rbf = rbf_kernel(x, z, gamma=0.5)
        print(f"  {name:<16s}  {k_lin:>8.3f}  {k_p2:>12.3f}  {k_p3:>12.3f}  {k_rbf:>12.4f}")

    print()
    print("  线性核：原始点积，度量投影。")
    print("  多项式核：捕获 d 阶以内的特征交互。")
    print("  RBF 核：基于距离的局部相似度，近处高远处低。")
    print()


def demo_kernel_trick():
    """演示核技巧在环形数据上的效果。"""
    print("=" * 65)
    print("核技巧：在环形数据上对比线性核与 RBF 核")
    print("=" * 65)
    print()

    X, y = generate_circular_data(150, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    print(f"  数据集：{len(X)} 个样本，环形边界（非线性可分）")
    print(f"  训练集：{len(X_train)}  测试集：{len(X_test)}")
    print()

    # 线性 SVM
    svm_linear = LinearSVM(lr=0.001, lambda_param=0.01, n_epochs=500)
    svm_linear.fit(X_train, y_train)
    linear_acc = accuracy(y_test, svm_linear.predict(X_test))

    print(f"  线性 SVM 测试准确率: {linear_acc:.4f}")
    print()

    # RBF 核 SVM（简化版 SMO）
    svm_rbf = SVMWithKernel(kernel_fn=rbf_kernel, C=10.0, gamma=1.0, n_epochs=50)
    svm_rbf.fit(X_train, y_train)
    rbf_acc = accuracy(y_test, svm_rbf.predict(X_test))

    svs = svm_rbf.find_support_vectors()
    print(f"  RBF 核 SVM 测试准确率: {rbf_acc:.4f}")
    print(f"  支持向量数量: {len(svs)} / {len(X_train)}")
    print()
    print("  线性核无法处理环形边界，RBF 核映射到高维空间后线性可分。")
    print("  核技巧的关键：不需要显式计算高维映射，只需在原空间计算核函数。")
    print()


def demo_soft_margin():
    """演示软间隔的松弛变量概念。"""
    print("=" * 65)
    print("软间隔：通过松弛变量处理噪声")
    print("=" * 65)
    print()

    # 一个简单的噪声数据集
    X = [[0, 0], [1, 1], [2, 2], [3, 3], [0.5, 2], [2, 0.5]]
    y = [1, 1, 1, 1, -1, -1]

    svm = LinearSVM(lr=0.001, lambda_param=0.01, n_epochs=1000)
    svm.fit(X, y)

    print(f"  训练数据：6 个样本")
    print(f"  权重 w: [{svm.w[0]:.4f}, {svm.w[1]:.4f}]")
    print(f"  偏置 b: {svm.b:.4f}")
    print()

    print(f"  样本          标签   决策值    间隔值      状态")
    print(f"  {'-' * 16}  {'-' * 4}  {'-' * 8}  {'-' * 10}  {'-' * 12}")
    for i in range(len(X)):
        d_val = svm.decision_function([X[i]])[0]
        margin = y[i] * d_val
        if margin >= 1.0:
            status = "间隔外"
        elif margin > 0:
            status = "间隔内"
        else:
            status = "分类错误"
        print(f"  {str(X[i]):<16s}  {y[i]:>4d}  {d_val:>8.4f}  {margin:>10.4f}  {status}")

    print()


def print_summary():
    """打印本课核心知识总结。"""
    print()
    print("=" * 65)
    print("总结")
    print("=" * 65)
    print()
    print("  1. SVM 寻找两个类别之间的最大间隔超平面。")
    print("  2. 只有支持向量（间隔边界上的点）决定决策边界。")
    print("  3. 铰链损失产生稀疏模型——间隔外的样本贡献为零。")
    print("  4. C 参数控制间隔宽度与分类错误的权衡。")
    print("  5. 软间隔通过松弛变量处理噪声和不可分数据。")
    print("  6. 核技巧通过核函数隐式映射到高维空间，无需显式计算。")
    print("  7. RBF 核将数据映射到无限维空间，可学习任意平滑边界。")
    print("  8. SMO 算法通过迭代优化拉格朗日乘子高效求解对偶问题。")
    print("  9. 线性 SVM 每轮迭代 O(n*d)，适合大规模稀疏数据。")
    print("  10. SVM 在小数据集和高维度稀疏数据上仍然有优势。")
    print()


if __name__ == "__main__":
    demo_hinge_loss()
    demo_linear_svm()
    demo_c_parameter()
    demo_kernel_functions()
    demo_kernel_trick()
    demo_soft_margin()
    print_summary()
