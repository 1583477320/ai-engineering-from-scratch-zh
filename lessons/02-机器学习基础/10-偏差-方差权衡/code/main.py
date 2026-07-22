# main.py — 偏差-方差权衡从零实现
# 依赖：numpy>=1.24
# 安装：pip install numpy
# 对应课程：第 02 阶段 · 10（偏差-方差权衡）

import numpy as np
import warnings

warnings.filterwarnings("ignore")


def true_function(x):
    """真实函数：sin(1.5x) + 0.5x，用于生成模拟数据。"""
    return np.sin(1.5 * x) + 0.5 * x


def generate_data(n_samples=30, noise_std=0.5, x_range=(-3, 3), seed=None):
    """生成带有噪声的一维回归数据。

    Args:
        n_samples: 样本数量
        noise_std: 噪声标准差
        x_range: x 的取值范围
        seed: 随机种子

    Returns:
        x: 输入，形状 (n_samples,)
        y: 输出，形状 (n_samples,)
    """
    rng = np.random.RandomState(seed)
    x = rng.uniform(x_range[0], x_range[1], n_samples)
    y = true_function(x) + rng.normal(0, noise_std, n_samples)
    return x, y


def fit_polynomial(x_train, y_train, degree, lam=0.0):
    """拟合多项式回归（可选 L2 正则化）。

    Args:
        x_train: 训练输入
        y_train: 训练输出
        degree: 多项式次数
        lam: L2 正则化强度（Ridge），0 表示无正则化

    Returns:
        w: 多项式系数向量
    """
    # 构建 Vandermonde 矩阵：每一列是 x 的 0 到 degree 次幂
    X = np.column_stack([x_train ** d for d in range(degree + 1)])
    if lam > 0:
        # Ridge 回归：在损失函数中添加 λ||w||²，但不惩罚偏置项（第 0 列）
        penalty = lam * np.eye(X.shape[1])
        penalty[0, 0] = 0
        w = np.linalg.solve(X.T @ X + penalty, X.T @ y_train)
    else:
        # 无正则化时直接用最小二乘
        w = np.linalg.lstsq(X, y_train, rcond=None)[0]
    return w


def predict_polynomial(x, w):
    """用多项式系数进行预测。

    Args:
        x: 输入
        w: 多项式系数

    Returns:
        预测值
    """
    degree = len(w) - 1
    X = np.column_stack([x ** d for d in range(degree + 1)])
    return X @ w


def bias_variance_decomposition(
    degrees,
    n_bootstrap=200,
    n_train=30,
    noise_std=0.5,
    n_test=100,
    lam=0.0,
):
    """通过 Bootstrap 采样计算偏差-方差分解。

    核心思想：从同一分布中重复采样多个训练集，对每个训练集拟合模型，
    然后在固定测试点上计算预测值的均值和方差，从而分别估计偏差²和方差。

    Args:
        degrees: 要评估的多项式次数列表
        n_bootstrap: Bootstrap 采样轮数
        n_train: 每轮训练样本数
        noise_std: 噪声标准差
        n_test: 测试点数量
        lam: L2 正则化强度

    Returns:
        results: dict，键为 degree，值为 bias_sq、variance、total_error、noise
    """
    rng = np.random.RandomState(42)
    # 在 [-2.5, 2.5] 均匀取测试点
    x_test = np.linspace(-2.5, 2.5, n_test)
    y_true = true_function(x_test)

    results = {}

    for degree in degrees:
        # 存储每轮 bootstrap 在测试点上的预测值
        predictions = np.zeros((n_bootstrap, n_test))

        for b in range(n_bootstrap):
            x_train, y_train = generate_data(
                n_samples=n_train, noise_std=noise_std, seed=rng.randint(0, 100000)
            )
            w = fit_polynomial(x_train, y_train, degree, lam=lam)
            predictions[b] = predict_polynomial(x_test, w)

        # 所有 bootstrap 预测的均值 → 模型的"平均预测"
        mean_pred = predictions.mean(axis=0)
        # 偏差²：(平均预测 - 真实值)² 的均值
        bias_sq = np.mean((mean_pred - y_true) ** 2)
        # 方差：每个测试点上预测值的方差的均值
        variance = np.mean(predictions.var(axis=0))
        # 总误差：均方误差
        total_error = np.mean(np.mean((predictions - y_true) ** 2, axis=1))

        results[degree] = {
            "bias_sq": bias_sq,
            "variance": variance,
            "total_error": total_error,
            "noise": noise_std ** 2,
        }

    return results


def print_decomposition(results):
    """打印偏差-方差分解表。"""
    print(f"{'次数':>6}  {'偏差²':>10}  {'方差':>10}  {'噪声':>10}  {'总误差':>10}  {'B²+V+N':>10}")
    print("-" * 68)
    for degree, r in sorted(results.items()):
        bvn = r["bias_sq"] + r["variance"] + r["noise"]
        print(
            f"{degree:>6d}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  "
            f"{r['noise']:>10.4f}  {r['total_error']:>10.4f}  {bvn:>10.4f}"
        )


def find_optimal(results):
    """找到使总误差最小的多项式次数。"""
    return min(results, key=lambda d: results[d]["total_error"])


def demo_basic_decomposition():
    """演示 1：基础偏差-方差分解。"""
    print("=" * 70)
    print("演示 1：偏差-方差分解")
    print("真实函数: sin(1.5x) + 0.5x")
    print("噪声标准差: 0.5, 训练样本: 30, Bootstrap 轮数: 200")
    print("=" * 70)
    print()

    degrees = [1, 2, 3, 5, 7, 10, 15]
    results = bias_variance_decomposition(degrees)
    print_decomposition(results)

    best = find_optimal(results)
    print(f"\n最优次数: {best}")
    print(f"  偏差²:   {results[best]['bias_sq']:.4f}")
    print(f"  方差:    {results[best]['variance']:.4f}")
    print(f"  总误差:  {results[best]['total_error']:.4f}")


def demo_complexity_tradeoff():
    """演示 2：不同复杂度下的偏差-方差权衡。"""
    print()
    print("=" * 70)
    print("演示 2：模型复杂度与偏差-方差权衡")
    print("扫描多项式次数 1 到 15")
    print("=" * 70)
    print()

    degrees = list(range(1, 16))
    results = bias_variance_decomposition(degrees)

    print(f"{'次数':>6}  {'偏差²':>10}  {'方差':>10}  {'总误差':>10}  {'主导因素':>12}")
    print("-" * 60)
    for degree in degrees:
        r = results[degree]
        dominant = "偏差" if r["bias_sq"] > r["variance"] else "方差"
        print(
            f"{degree:>6d}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  "
            f"{r['total_error']:>10.4f}  {dominant:>12}"
        )

    # 找到偏差和方差交叉点（主导因素切换的位置）
    crossover = None
    for d in degrees[:-1]:
        if results[d]["bias_sq"] > results[d]["variance"]:
            if results[d + 1]["bias_sq"] <= results[d + 1]["variance"]:
                crossover = d + 1
                break

    if crossover:
        print(f"\n偏差-方差交叉点在第 {crossover} 次")
        print("低于此值：偏差主导（欠拟合）")
        print("高于此值：方差主导（过拟合）")


def demo_regularization_effect():
    """演示 3：L2 正则化对偏差-方差的影响（正则化路径）。"""
    print()
    print("=" * 70)
    print("演示 3：正则化路径（L2 / Ridge）")
    print("固定多项式次数=10，扫描正则化强度 λ")
    print("=" * 70)
    print()

    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    print(f"{'λ':>10}  {'偏差²':>10}  {'方差':>10}  {'总误差':>10}")
    print("-" * 50)

    for lam in lambdas:
        results = bias_variance_decomposition([10], lam=lam)
        r = results[10]
        print(f"{lam:>10.3f}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  {r['total_error']:>10.4f}")

    print()
    print("λ 增大时：")
    print("  - 方差减小（模型被约束，不易过拟合）")
    print("  - 偏差增大（模型被强制简化）")
    print("  - 最优 λ 在这两个效应之间取得平衡")


def demo_data_size_effect():
    """演示 4：训练数据量对偏差-方差的影响。"""
    print()
    print("=" * 70)
    print("演示 4：训练数据量的影响")
    print("固定多项式次数=5，改变训练集大小")
    print("=" * 70)
    print()

    sizes = [10, 20, 50, 100, 200, 500]

    print(f"{'样本数':>8}  {'偏差²':>10}  {'方差':>10}  {'总误差':>10}")
    print("-" * 50)

    for n in sizes:
        results = bias_variance_decomposition([5], n_train=n)
        r = results[5]
        print(f"{n:>8d}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  {r['total_error']:>10.4f}")

    print()
    print("更多数据会降低方差，但不影响偏差。")
    print("如果问题是高偏差，加数据无济于事。")


def demo_diagnosis():
    """演示 5：通过训练/测试误差模式诊断过拟合、欠拟合。"""
    print()
    print("=" * 70)
    print("演示 5：欠拟合 vs 过拟合 诊断")
    print("=" * 0)
    print()

    rng = np.random.RandomState(42)
    x_train, y_train = generate_data(n_samples=30, seed=42)
    x_test, y_test = generate_data(n_samples=100, seed=99)

    cases = [
        (1, "线性（1 次）"),
        (4, "多项式（4 次）"),
        (15, "多项式（15 次）"),
    ]

    for degree, name in cases:
        w = fit_polynomial(x_train, y_train, degree)
        train_pred = predict_polynomial(x_train, w)
        test_pred = predict_polynomial(x_test, w)

        train_mse = np.mean((train_pred - y_train) ** 2)
        test_mse = np.mean((test_pred - y_test) ** 2)
        gap = test_mse - train_mse

        # 诊断逻辑
        if train_mse > 0.5 and test_mse > 0.5 and gap < train_mse * 0.5:
            diagnosis = "高偏差（欠拟合）"
        elif gap > train_mse * 2:
            diagnosis = "高方差（过拟合）"
        else:
            diagnosis = "拟合合理"

        print(f"{name}:")
        print(f"  训练 MSE: {train_mse:.4f}")
        print(f"  测试 MSE:  {test_mse:.4f}")
        print(f"  差距:      {gap:.4f}")
        print(f"  诊断:      {diagnosis}")
        print()


def demo_learning_curves():
    """演示 6：学习曲线。"""
    print()
    print("=" * 70)
    print("演示 6：学习曲线")
    print("训练集大小增长时的训练/测试误差变化")
    print("=" * 70)
    print()

    rng = np.random.RandomState(42)
    x_test = np.linspace(-2.5, 2.5, 200)
    y_test = true_function(x_test)

    sizes = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300]

    for degree, label in [(1, "1 次（高偏差）"), (5, "5 次（平衡）"), (12, "12 次（高方差）")]:
        print(f"  {label}:")
        print(f"  {'样本数':>8}  {'训练 MSE':>10}  {'测试 MSE':>10}  {'差距':>10}")
        print(f"  {'-' * 52}")

        for n in sizes:
            train_errors = []
            test_errors = []
            for seed in range(50):
                x_train, y_train = generate_data(n_samples=n, seed=rng.randint(0, 100000))
                try:
                    w = fit_polynomial(x_train, y_train, degree)
                    train_pred = predict_polynomial(x_train, w)
                    test_pred = predict_polynomial(x_test, w)
                    train_mse = np.mean((train_pred - y_train) ** 2)
                    test_mse = np.mean((test_pred - y_test) ** 2)
                    train_errors.append(train_mse)
                    test_errors.append(test_mse)
                except (np.linalg.LinAlgError, ValueError):
                    continue

            if train_errors:
                avg_train = np.mean(train_errors)
                avg_test = np.mean(test_errors)
                gap = avg_test - avg_train
                print(f"  {n:>8d}  {avg_train:>10.4f}  {avg_test:>10.4f}  {gap:>10.4f}")

        print()

    print("高偏差（1 次）：两条曲线都收敛到高误差，差距小。")
    print("高方差（12 次）：训练误差保持低，测试误差保持高。")
    print("更多数据能降低方差，但无法修复偏差。")


def demo_regularization_sweep():
    """演示 7：正则化扫描（调节 λ）。"""
    print()
    print("=" * 70)
    print("演示 7：正则化扫描（α vs 偏差/方差）")
    print("固定多项式次数=15，扫描 α 从 0.001 到 100")
    print("=" * 70)
    print()

    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    print(f"  {'α':>10}  {'偏差²':>10}  {'方差':>10}  {'总误差':>10}  {'主导因素':>12}")
    print(f"  {'-' * 64}")

    best_alpha = None
    best_total = float("inf")

    for alpha in alphas:
        results = bias_variance_decomposition([15], lam=alpha, n_bootstrap=200)
        r = results[15]
        dominant = "偏差" if r["bias_sq"] > r["variance"] else "方差"
        print(
            f"  {alpha:>10.3f}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  "
            f"{r['total_error']:>10.4f}  {dominant:>12}"
        )
        if r["total_error"] < best_total:
            best_total = r["total_error"]
            best_alpha = alpha

    print()
    print(f"最优 α: {best_alpha}")
    print(f"  最优 α 下总误差: {best_total:.4f}")
    print()
    print("小 α：方差主导（模型几乎无约束，追逐噪声）")
    print("大 α：偏差主导（模型过度约束，丢失信号）")
    print("最优 α 位于两者底部——U 形曲线的最低点。")


def demo_double_descent():
    """演示 8：双重下降现象（过参数化区域的测试误差再次下降）。"""
    print()
    print("=" * 70)
    print("演示 8：双重下降现象（Double Descent）")
    print("当过参数化程度超过插值阈值时，测试误差再次下降")
    print("=" * 70)
    print()

    rng = np.random.RandomState(42)
    n_train = 20
    x_train, y_train = generate_data(n_samples=n_train, seed=42)
    x_test = np.linspace(-2.5, 2.5, 200)
    y_test = true_function(x_test)

    # 从 1 次到 40 次多项式（40 > 20，过参数化）
    degrees = list(range(1, 41))

    print(f"{'次数':>6}  {'训练 MSE':>10}  {'测试 MSE':>10}  {'区域':>15}")
    print("-" * 50)

    test_errors = []
    for degree in degrees:
        try:
            w = fit_polynomial(x_train, y_train, degree)
            train_pred = predict_polynomial(x_train, w)
            test_pred = predict_polynomial(x_test, w)
            train_mse = np.mean((train_pred - y_train) ** 2)
            test_mse = np.mean((test_pred - y_test) ** 2)
            test_errors.append(test_mse)

            if degree < n_train - 2:
                region = "欠参数化"
            elif abs(degree - n_train) <= 2:
                region = "★ 插值阈值"
            else:
                region = "过参数化（双重下降）"

            if degree <= 5 or degree >= n_train - 3:
                print(f"{degree:>6d}  {train_mse:>10.4f}  {test_mse:>10.4f}  {region:>15}")
            elif degree == 6:
                print(f"  ... {'':>10}  {'':>10}  {'':>15}")
        except (np.linalg.LinAlgError, ValueError):
            test_errors.append(np.nan)
            print(f"{degree:>6d}  {'奇异':>10}  {'奇异':>10}  {'不稳定':>15}")

    print()
    print("关键观察：")
    print("  - 经典区域（次数 < 样本数）：U 形曲线，测试误差先降后升")
    print("  - 插值阈值（次数 ≈ 样本数）：测试误差出现尖锐峰值（方差爆炸）")
    print("  - 过参数化区域（次数 > 样本数）：测试误差再次下降")
    print("  - 这正是为什么过参数化神经网络仍然能够泛化")


if __name__ == "__main__":
    # 运行所有演示
    demo_basic_decomposition()
    demo_complexity_tradeoff()
    demo_regularization_effect()
    demo_data_size_effect()
    demo_diagnosis()
    demo_learning_curves()
    demo_regularization_sweep()
    demo_double_descent()
    print()
    print("所有偏差-方差权衡演示完成。")
