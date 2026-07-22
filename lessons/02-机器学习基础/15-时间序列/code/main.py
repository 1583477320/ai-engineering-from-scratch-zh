# 时间序列分析 — 核心算法的从零实现与工业工具演示
# 依赖：numpy>=1.24, scipy>=1.10, matplotlib>=3.7
# 可选依赖（工业工具演示）：scikit-learn>=1.3, statsmodels>=0.14, prophet>=1.1
# 安装：pip install numpy scipy matplotlib scikit-learn statsmodels prophet
# 对应课程：第 02 阶段 · 15（时间序列）

import numpy as np


# ============================================================================
# 第一部分：数据生成
# ============================================================================

def make_synthetic_trend_series(n=500, seed=42):
    """生成带趋势和季节性的合成时间序列。

    模拟场景：某电商平台每日订单量（趋势上涨 + 月度波动 + 噪声）。
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=float)
    trend = 0.05 * t                              # 缓慢上涨趋势
    seasonality = 10 * np.sin(2 * np.pi * t / 30)  # 30 天为周期的季节性
    noise = rng.normal(0, 2, n)                   # 随机扰动
    series = 50 + trend + seasonality + noise
    return series


def make_seasonal_daily_series(n=365, period=7, seed=42):
    """生成带周季节性的合成日频序列。

    模拟场景：外卖平台每日订单量（周末高峰 + 月度波动）。
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=float)
    trend = 0.02 * t
    weekly = 5 * np.sin(2 * np.pi * t / period)     # 周季节性
    monthly = 3 * np.sin(2 * np.pi * t / 30)        # 月度波动
    noise = rng.normal(0, 1.5, n)
    series = 100 + trend + weekly + monthly + noise
    return series


# ============================================================================
# 第二部分：时间与平稳性
# ============================================================================

def difference(series, order=1):
    """差分操作：用差值替换原始值。

    差分是使非平稳序列变平稳的核心操作。
    order=1 做一阶差分，order=2 做二阶差分。

    Args:
        series: 原始序列
        order: 差分阶数

    Returns:
        差分后的序列（长度减少 order）
    """
    result = series.copy()
    for _ in range(order):
        result = result[1:] - result[:-1]
    return result


def check_stationarity(series, window=50):
    """检查序列的平稳性。

    通过滚动统计量和前后半段对比做启发式判断。
    真正的统计检验（如 ADF 检验）需要渐近分布表，这里不实现。

    Args:
        series: 输入序列
        window: 滚动窗口大小

    Returns:
        rolling_mean: 滚动均值序列
        rolling_std: 滚动标准差序列
        is_stationary: 是否平稳（启发式判断）
    """
    n = len(series)
    rolling_mean = np.zeros(n)
    rolling_std = np.zeros(n)

    for i in range(n):
        start = max(0, i - window + 1)
        segment = series[start:i + 1]
        rolling_mean[i] = segment.mean()
        rolling_std[i] = segment.std() if len(segment) > 1 else 0.0

    # 前后半段均值差异
    first_half_mean = series[:n // 2].mean()
    second_half_mean = series[n // 2:].mean()
    first_half_var = series[:n // 2].var()
    second_half_var = series[n // 2:].var()

    mean_shift = abs(first_half_mean - second_half_mean)
    var_ratio = (
        max(first_half_var, second_half_var)
        / max(min(first_half_var, second_half_var), 1e-10)
    )

    # 启发式判断：均值偏移小于半个标准差，且方差比小于 2 倍
    is_stationary = mean_shift < 0.5 * series.std() and var_ratio < 2.0

    return rolling_mean, rolling_std, is_stationary


def autocorrelation(series, max_lag=20):
    """计算自相关函数（ACF）。

    ACF(k) = Cov(series[t], series[t-k]) / Var(series)

    Args:
        series: 输入序列
        max_lag: 最大滞后阶数

    Returns:
        长度为 max_lag+1 的自相关数组
    """
    n = len(series)
    mean = series.mean()
    var = series.var()
    acf = np.zeros(max_lag + 1)

    for k in range(max_lag + 1):
        if k >= n:
            break
        cov = np.mean((series[:n - k] - mean) * (series[k:] - mean))
        acf[k] = cov / var if var > 0 else 0.0

    return acf


# ============================================================================
# 第三部分：特征构造与建模
# ============================================================================

def make_lag_features(series, n_lags):
    """构造滞后特征矩阵。

    将一维时间序列转换为监督学习的特征矩阵：
    - 特征：y[t-1], y[t-2], ..., y[t-k]
    - 标签：y[t]

    这是把时间序列问题转化为标准 ML 问题的关键桥梁。

    Args:
        series: 一维序列
        n_lags: 滞后阶数

    Returns:
        X: 特征矩阵，形状 (samples, n_lags)
        y: 标签向量，形状 (samples,)
    """
    n = len(series)
    X = np.full((n, n_lags), np.nan)

    for lag in range(1, n_lags + 1):
        X[lag:, lag - 1] = series[:-lag]

    # 去掉含 NaN 的行（前 n_lags 行没有完整的滞后信息）
    valid_mask = ~np.isnan(X).any(axis=1)
    return X[valid_mask], series[valid_mask]


def make_rolling_features(series, window):
    """构造滚动统计量特征。

    计算给定窗口内的滚动均值和标准差，帮助模型捕捉
    近期趋势和波动性。

    Args:
        series: 输入序列
        window: 滚动窗口大小

    Returns:
        rolling_mean: 滚动均值序列
        rolling_std: 滚动标准差序列
    """
    n = len(series)
    rolling_mean = np.zeros(n)
    rolling_std = np.zeros(n)

    for i in range(n):
        start = max(0, i - window + 1)
        segment = series[start:i + 1]
        rolling_mean[i] = segment.mean()
        rolling_std[i] = segment.std() if len(segment) > 1 else 0.0

    return rolling_mean, rolling_std


def make_calendar_features(n, freq="daily"):
    """构造日历特征。

    生成星期几（0-6）和月份（1-12）的独热编码，
    帮助模型捕捉日历效应。

    Args:
        n: 序列长度
        freq: 序列频率（"daily" 或 "hourly"）

    Returns:
        dow_features: 星期几的 sin/cos 编码，形状 (n, 2)
        month_features: 月份的 sin/cos 编码，形状 (n, 2)
    """
    dow = np.arange(n) % 7  # 假设第 0 天是星期一
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    month = (np.arange(n) % 365) / 30.4  # 近似月份（每月 30.4 天）
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    dow_features = np.column_stack([dow_sin, dow_cos])
    month_features = np.column_stack([month_sin, month_cos])

    return dow_features, month_features


class SimpleAR:
    """简单自回归模型（AR 模型）。

    概念上等同于对滞后特征做线性回归。
    AR(p) 公式：y[t] = bias + w1*y[t-1] + w2*y[t-2] + ... + wp*y[t-p]
    """

    def __init__(self, n_lags=5):
        self.n_lags = n_lags
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """用最小二乘法训练模型。"""
        # 添加偏置列（全 1）
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        # 正规方程求解
        theta = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
        self.bias = theta[0]
        self.weights = theta[1:]
        return self

    def predict(self, X):
        """用训练好的模型做预测。"""
        return X @ self.weights + self.bias

    def fit_series(self, series):
        """从原始序列直接训练。"""
        X, y = make_lag_features(series, self.n_lags)
        return self.fit(X, y)

    def forecast(self, last_values, n_steps):
        """递归多步预测。

        每一步的预测结果会被用作下一步的输入。这是最常用的
        多步预测策略，但缺点是误差会逐步累积。

        Args:
            last_values: 已知的历史值（至少 n_lags 个）
            n_steps: 预测步数

        Returns:
            长度为 n_steps 的预测数组
        """
        if len(last_values) < self.n_lags:
            raise ValueError(
                f"需要至少 {self.n_lags} 个历史点，当前只有 {len(last_values)} 个"
            )
        history = list(last_values[-self.n_lags:])
        predictions = []

        for _ in range(n_steps):
            features = np.array(history[-self.n_lags:]).reshape(1, -1)
            pred = self.predict(features)[0]
            predictions.append(pred)
            history.append(pred)

        return np.array(predictions)


# ============================================================================
# 第四部分：评估方法
# ============================================================================

def walk_forward_split(n_samples, n_splits=5, min_train=50):
    """生成前向滚动的训练/测试折（Walk-Forward Validation）。

    时间序列评估的核心方法。每一折的训练数据严格在
    测试数据之前，确保不会出现"未来信息泄漏"。

    Args:
        n_samples: 样本总数
        n_splits: 折数
        min_train: 训练集最小样本数

    Yields:
        train_slice: 训练集切片
        test_slice: 测试集切片
    """
    if n_samples <= min_train:
        return
    step = max(1, (n_samples - min_train) // n_splits)
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, n_samples)
        if train_end >= n_samples:
            break
        yield slice(0, train_end), slice(train_end, test_end)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    """平均绝对百分比误差。"""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    """对称 MAPE（Symmetric MAPE）。

    解决了 MAPE 在真实值接近 0 时发散的问题。
    公式：sMAPE = (2 * |pred - true|) / (|pred| + |true|) * 100
    """
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator > 0
    if mask.sum() == 0:
        return 0.0
    values = 2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return np.mean(values) * 100


def mase(y_true, y_pred, y_train, seasonal_period=1):
    """平均绝对标度误差（MASE）。

    MASE = MAE / MAE_naive
    其中 MAE_naive 是用朴素策略（季节性滞后）的预测误差。
    MASE < 1 表示模型优于朴素基线。

    Args:
        y_true: 真实值
        y_pred: 预测值
        y_train: 训练集（用于计算朴素基线）
        seasonal_period: 季节性周期（1 表示无季节性）
    """
    model_mae = mae(y_true, y_pred)
    if len(y_train) <= seasonal_period:
        return float("inf")
    # 朴素预测：用 seasonal_period 步前的值作为预测
    naive_errors = np.abs(
        y_train[seasonal_period:] - y_train[:-seasonal_period]
    )
    naive_mae = np.mean(naive_errors)
    if naive_mae == 0:
        return float("inf")
    return model_mae / naive_mae


def seasonal_naive_forecast(train_series, n_steps, period=7):
    """季节性朴素预测基线。

    用 period 步前的值作为预测。这是时间序列预测中
    最强的简单基线之一。
    """
    predictions = []
    for i in range(n_steps):
        idx = len(train_series) - period + (i % period)
        predictions.append(train_series[idx])
    return np.array(predictions)


# ============================================================================
# 第五部分：指数平滑（从零实现）
# ============================================================================

class SimpleExponentialSmoothing:
    """简单指数平滑（SES）。

    适用于无趋势、无季节性的序列。
    公式：s[t] = alpha * y[t] + (1 - alpha) * s[t-1]
    预测：y_hat[t+h] = s[t]
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.level = None

    def fit(self, series):
        """拟合指数平滑模型。"""
        n = len(series)
        self.level = np.zeros(n + 1)
        self.level[0] = series[0]
        for t in range(n):
            self.level[t + 1] = (
                self.alpha * series[t] + (1 - self.alpha) * self.level[t]
            )
        return self

    def forecast(self, n_steps):
        """预测未来 n_steps 步。

        简单指数平滑的所有步预测值相同（水平线）。
        """
        if self.level is None:
            raise ValueError("模型尚未训练，请先调用 fit()")
        last_level = self.level[-1]
        return np.full(n_steps, last_level)


class HoltWintersSmoothing:
    """Holt-Winters 双指数平滑（带趋势）。

    适用于有趋势但无季节性的序列。
    维护两个状态：水平（level）和趋势（trend）。
    """

    def __init__(self, alpha=0.3, beta=0.1):
        self.alpha = alpha  # 水平平滑参数
        self.beta = beta    # 趋势平滑参数
        self.level = None
        self.trend = None

    def fit(self, series):
        """拟合 Holt 线性趋势模型。"""
        n = len(series)
        self.level = np.zeros(n + 1)
        self.trend = np.zeros(n + 1)
        self.level[0] = series[0]
        self.trend[0] = series[1] - series[0] if n > 1 else 0

        for t in range(n):
            # 更新水平
            self.level[t + 1] = (
                self.alpha * series[t] + (1 - self.alpha) * (self.level[t] + self.trend[t])
            )
            # 更新趋势
            self.trend[t + 1] = (
                self.beta * (self.level[t + 1] - self.level[t])
                + (1 - self.beta) * self.trend[t]
            )
        return self

    def forecast(self, n_steps):
        """预测未来 n_steps 步。

        预测 = 最后水平 + h * 最后趋势（线性外推）。
        """
        if self.level is None:
            raise ValueError("模型尚未训练，请先调用 fit()")
        last_level = self.level[-1]
        last_trend = self.trend[-1]
        return np.array([last_level + h * last_trend for h in range(1, n_steps + 1)])


# ============================================================================
# 第六部分：时间序列分解
# ============================================================================

def decompose_series(series, period):
    """将时间序列分解为趋势、季节性和残差三个分量。

    使用加法模型：Y[t] = Trend[t] + Seasonal[t] + Residual[t]

    Args:
        series: 原始序列
        period: 季节性周期

    Returns:
        trend: 趋势分量（用中心移动平均估计）
        seasonal: 季节性分量
        residual: 残差分量
    """
    n = len(series)

    # 1. 趋势：用周期长度的中心移动平均
    trend = np.zeros(n)
    half = period // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        trend[i] = series[start:end].mean()

    # 2. 季节性：对每个周期位置取平均
    detrended = series - trend
    seasonal_means = np.zeros(period)
    for p in range(period):
        indices = list(range(p, n, period))
        if indices:
            seasonal_means[p] = np.mean(detrended[indices])
    # 中心化季节性分量（均值为 0）
    seasonal_means -= seasonal_means.mean()
    seasonal = np.array([seasonal_means[i % period] for i in range(n)])

    # 3. 残差
    residual = series - trend - seasonal

    return trend, seasonal, residual


# ============================================================================
# 第七部分：演示函数
# ============================================================================

def print_separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_stationarity():
    """演示平稳性检查与差分。"""
    print_separator("平稳性检查与差分")

    series = make_synthetic_trend_series(n=300, seed=42)
    _, _, is_stat = check_stationarity(series)
    print(f"原始序列（含趋势 + 季节性）：")
    print(f"  均值: {series.mean():.2f}, 标准差: {series.std():.2f}")
    print(f"  是否平稳: {is_stat}")

    diff1 = difference(series, order=1)
    _, _, is_stat1 = check_stationarity(diff1)
    print(f"\n一阶差分后：")
    print(f"  均值: {diff1.mean():.4f}, 标准差: {diff1.std():.2f}")
    print(f"  是否平稳: {is_stat1}")

    diff2 = difference(series, order=2)
    _, _, is_stat2 = check_stationarity(diff2)
    print(f"\n二阶差分后：")
    print(f"  均值: {diff2.mean():.4f}, 标准差: {diff2.std():.2f}")
    print(f"  是否平稳: {is_stat2}")


def demo_autocorrelation():
    """演示自相关分析。"""
    print_separator("自相关分析（ACF）")

    series = make_seasonal_daily_series(n=365, period=7, seed=42)
    diff_series = difference(series, order=1)
    acf = autocorrelation(diff_series, max_lag=30)

    print("差分后序列的 ACF（前 15 阶）：")
    print(f"{'滞后':>6} {'ACF':>8} {'显著性':>8}")
    print(f"{'-' * 24}")
    threshold = 1.96 / np.sqrt(len(diff_series))
    for k in range(15):
        sig = "***" if abs(acf[k]) > threshold else ""
        bar = "#" * int(abs(acf[k]) * 30)
        print(f"{k:>6} {acf[k]:>8.4f} {sig:>4} {bar}")

    print(f"\n显著性阈值（95%）：+/-{threshold:.4f}")
    print(f"滞后 7 和 14 处应有峰值（周季节性）")


def demo_decomposition():
    """演示时间序列分解。"""
    print_separator("时间序列分解")

    series = make_seasonal_daily_series(n=365, period=7, seed=42)
    trend, seasonal, residual = decompose_series(series, period=7)

    print(f"原始序列 — 均值: {series.mean():.2f}, 标准差: {series.std():.2f}")
    print(f"趋势分量 — 均值: {trend.mean():.2f}, 标准差: {trend.std():.2f}")
    print(f"季节分量 — 均值: {seasonal.mean():.4f}, 标准差: {seasonal.std():.2f}")
    print(f"残差分量 — 均值: {residual.mean():.4f}, 标准差: {residual.std():.2f}")

    print(f"\n前 7 天的分解结果：")
    print(f"{'日期':>6} {'原始':>8} {'趋势':>8} {'季节':>8} {'残差':>8}")
    print("-" * 42)
    for i in range(7):
        print(f"{i:>6} {series[i]:>8.2f} {trend[i]:>8.2f} {seasonal[i]:>8.2f} {residual[i]:>8.2f}")


def demo_lag_features():
    """演示滞后特征构造与 AR 模型训练。"""
    print_separator("滞后特征与 AR 模型")

    series = make_synthetic_trend_series(n=400, seed=42)
    n_lags = 10

    X, y = make_lag_features(series, n_lags)
    print(f"序列长度: {len(series)}")
    print(f"特征矩阵: {X.shape}（样本数 × 滞后阶数）")
    print(f"标签向量: {y.shape}")

    print(f"\n前 3 个样本：")
    for i in range(3):
        lags_str = ", ".join(f"{v:.1f}" for v in X[i, :5])
        print(f"  滞后: [{lags_str}, ...] → 标签: {y[i]:.1f}")

    ar = SimpleAR(n_lags=n_lags)
    ar.fit(X, y)

    print(f"\nAR({n_lags}) 模型权重：")
    for i, w in enumerate(ar.weights):
        print(f"  滞后 {i+1}: {w:+.4f}")
    print(f"  偏置:  {ar.bias:+.4f}")


def demo_walk_forward():
    """演示前向滚动验证。"""
    print_separator("前向滚动验证（Walk-Forward Validation）")

    series = make_synthetic_trend_series(n=400, seed=42)
    n_lags = 10
    X, y = make_lag_features(series, n_lags)

    n_splits = 5
    fold_scores = []

    print(f"前向滚动验证（{n_splits} 折）：")
    print(f"{'折数':>6} {'训练集':>10} {'测试集':>10} {'MSE':>10} {'MAE':>10}")
    print(f"{'-' * 50}")

    for fold, (train_sl, test_sl) in enumerate(
        walk_forward_split(len(X), n_splits=n_splits, min_train=100)
    ):
        X_train, y_train = X[train_sl], y[train_sl]
        X_test, y_test = X[test_sl], y[test_sl]

        ar = SimpleAR(n_lags=n_lags)
        ar.fit(X_train, y_train)
        y_pred = ar.predict(X_test)

        fold_mse = mse(y_test, y_pred)
        fold_mae = mae(y_test, y_pred)
        fold_scores.append(fold_mse)

        print(
            f"{fold+1:>6} {X_train.shape[0]:>10} {X_test.shape[0]:>10} "
            f"{fold_mse:>10.4f} {fold_mae:>10.4f}"
        )

    print(f"\n平均 MSE: {np.mean(fold_scores):.4f}")
    print(f"MSE 标准差: {np.std(fold_scores):.4f}")


def demo_random_vs_walk_forward():
    """对比随机划分与前向滚动验证的差异。"""
    print_separator("随机划分 vs 前向滚动验证")

    series = make_synthetic_trend_series(n=500, seed=42)
    n_lags = 10
    X, y = make_lag_features(series, n_lags)

    # 随机 80/20 划分
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    ar_random = SimpleAR(n_lags=n_lags)
    ar_random.fit(X[train_idx], y[train_idx])
    random_mse = mse(y[test_idx], ar_random.predict(X[test_idx]))

    # 前向滚动验证
    wf_scores = []
    for train_sl, test_sl in walk_forward_split(len(X), n_splits=5, min_train=100):
        ar_wf = SimpleAR(n_lags=n_lags)
        ar_wf.fit(X[train_sl], y[train_sl])
        y_pred = ar_wf.predict(X[test_sl])
        wf_scores.append(mse(y[test_sl], y_pred))

    wf_mse = np.mean(wf_scores)

    print(f"随机 80/20 划分 MSE:  {random_mse:.4f}")
    print(f"前向滚动验证平均 MSE:  {wf_mse:.4f}")
    print(f"比值（随机/前向）:     {random_mse / wf_mse:.4f}")
    print()
    if random_mse < wf_mse:
        print("随机划分 MSE 更低 —— 这是未来信息泄漏导致的乐观偏差。")
        print("前向滚动验证的分数才是对生产环境性能的诚实估计。")
    else:
        print("前向滚动验证给出相似或更低的 MSE —— 序列可能足够平稳，")
        print("未来信息泄漏在此处不是主要因素。")


def demo_exponential_smoothing():
    """演示指数平滑模型。"""
    print_separator("指数平滑模型")

    series = make_synthetic_trend_series(n=300, seed=42)
    train = series[:250]
    test = series[250:270]

    # 简单指数平滑
    ses = SimpleExponentialSmoothing(alpha=0.3)
    ses.fit(train)
    ses_pred = ses.forecast(len(test))

    # Holt 双指数平滑
    holt = HoltWintersSmoothing(alpha=0.3, beta=0.1)
    holt.fit(train)
    holt_pred = holt.forecast(len(test))

    # 季节性朴素基线
    naive_pred = seasonal_naive_forecast(train, len(test), period=30)

    print(f"训练集: {len(train)} 个点，测试集: {len(test)} 个点")
    print()
    print(f"{'模型':<25} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
    print(f"{'-' * 57}")
    print(
        f"{'简单指数平滑 (SES)':<25} {mae(test, ses_pred):>10.4f} "
        f"{rmse(test, ses_pred):>10.4f} {mape(test, ses_pred):>9.2f}%"
    )
    print(
        f"{'Holt 双指数平滑':<25} {mae(test, holt_pred):>10.4f} "
        f"{rmse(test, holt_pred):>10.4f} {mape(test, holt_pred):>9.2f}%"
    )
    print(
        f"{'季节性朴素基线':<25} {mae(test, naive_pred):>10.4f} "
        f"{rmse(test, naive_pred):>10.4f} {mape(test, naive_pred):>9.2f}%"
    )


def demo_evaluation_metrics():
    """演示 MASE 和 sMAPE 评估指标。"""
    print_separator("预测评估指标（MASE / sMAPE）")

    series = make_seasonal_daily_series(n=365, period=7, seed=42)
    train = series[:300]
    test = series[300:]

    # AR 模型预测
    n_lags = 14
    ar = SimpleAR(n_lags=n_lags)
    X_train, y_train = make_lag_features(train, n_lags)
    ar.fit(X_train, y_train)
    ar_pred = ar.forecast(train, len(test))

    # 季节性朴素基线
    naive_pred = seasonal_naive_forecast(train, len(test), period=7)

    ar_mase = mase(test, ar_pred, train, seasonal_period=7)
    naive_mase = mase(test, naive_pred, train, seasonal_period=7)
    ar_smape = smape(test, ar_pred)
    naive_smape = smape(test, naive_pred)

    print(f"测试集长度: {len(test)}")
    print()
    print(f"{'指标':<15} {'AR 模型':>12} {'朴素基线':>12}")
    print(f"{'-' * 41}")
    print(f"{'MAE':<15} {mae(test, ar_pred):>12.4f} {mae(test, naive_pred):>12.4f}")
    print(f"{'RMSE':<15} {rmse(test, ar_pred):>12.4f} {rmse(test, naive_pred):>12.4f}")
    print(f"{'MAPE (%)':<15} {mape(test, ar_pred):>12.2f} {mape(test, naive_pred):>12.2f}")
    print(f"{'sMAPE (%)':<15} {ar_smape:>12.2f} {naive_smape:>12.2f}")
    print(f"{'MASE':<15} {ar_mase:>12.4f} {naive_mase:>12.4f}")
    print()
    print(f"MASE < 1 表示模型优于朴素基线。")
    print(f"AR 模型 MASE = {ar_mase:.4f}，{'优于' if ar_mase < 1 else '不优于'}朴素基线。")


def demo_forecasting():
    """演示多步预测。"""
    print_separator("多步预测")

    series = make_synthetic_trend_series(n=300, seed=42)
    train_series = series[:250]
    true_future = series[250:270]

    n_lags = 10
    ar = SimpleAR(n_lags=n_lags)
    X, y = make_lag_features(train_series, n_lags)
    ar.fit(X, y)

    forecast = ar.forecast(train_series, n_steps=20)

    print(f"训练集: {len(train_series)} 个点，预测未来 {len(true_future)} 步")
    print()
    print(f"{'步数':>6} {'真实值':>10} {'预测值':>10} {'误差':>10}")
    print(f"{'-' * 40}")

    for i in range(len(true_future)):
        error = true_future[i] - forecast[i]
        print(f"{i+1:>6} {true_future[i]:>10.2f} {forecast[i]:>10.2f} {error:>+10.2f}")

    print(f"\n预测 MSE:  {mse(true_future, forecast):.4f}")
    print(f"预测 MAE:  {mae(true_future, forecast):.4f}")
    print(f"预测 MAPE: {mape(true_future, forecast):.2f}%")


def demo_industrial_tools():
    """演示工业工具的使用（scikit-learn + statsmodels）。

    如果未安装可选依赖，会跳过并提示安装方法。
    """
    print_separator("工业工具演示")

    # --- scikit-learn 演示 ---
    try:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_absolute_error

        series = make_synthetic_trend_series(n=400, seed=42)
        n_lags = 10
        X, y = make_lag_features(series, n_lags)

        # 使用 sklearn 的 TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        ridge_scores = []
        gb_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            ridge_scores.append(mean_absolute_error(y_test, ridge.predict(X_test)))

            gb = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, random_state=42
            )
            gb.fit(X_train, y_train)
            gb_scores.append(mean_absolute_error(y_test, gb.predict(X_test)))

        print("scikit-learn 前向滚动验证结果：")
        print(f"  Ridge 回归平均 MAE: {np.mean(ridge_scores):.4f}")
        print(f"  GBDT 平均 MAE:      {np.mean(gb_scores):.4f}")

    except ImportError:
        print("scikit-learn 未安装，跳过。安装命令：pip install scikit-learn")

    # --- statsmodels ARIMA 演示 ---
    try:
        from statsmodels.tsa.arima.model import ARIMA

        series = make_synthetic_trend_series(n=300, seed=42)
        train = series[:250]
        test = series[250:]

        model = ARIMA(train, order=(5, 1, 2))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(test))

        print(f"\nstatsmodels ARIMA(5,1,2) 预测结果：")
        print(f"  预测步数: {len(test)}")
        print(f"  MAE: {mae(test, forecast):.4f}")
        print(f"  RMSE: {rmse(test, forecast):.4f}")

    except ImportError:
        print("\nstatsmodels 未安装，跳过。安装命令：pip install statsmodels")

    # --- Prophet 演示 ---
    try:
        from prophet import Prophet
        import pandas as pd

        series = make_synthetic_trend_series(n=300, seed=42)
        train = series[:250]
        test = series[250:]

        # Prophet 需要特定格式的 DataFrame
        df_train = pd.DataFrame({
            "ds": pd.date_range("2023-01-01", periods=len(train), freq="D"),
            "y": train
        })

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
        )
        model.fit(df_train)

        future = model.make_future_dataframe(periods=len(test), freq="D")
        forecast = model.predict(future)
        prophet_pred = forecast["yhat"].values[-len(test):]

        print(f"\nProphet 预测结果：")
        print(f"  预测步数: {len(test)}")
        print(f"  MAE: {mae(test, prophet_pred):.4f}")
        print(f"  RMSE: {rmse(test, prophet_pred):.4f}")

    except ImportError:
        print("\nProphet 未安装，跳过。安装命令：pip install prophet")
    except Exception as e:
        print(f"\nProphet 运行出错：{e}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    demo_stationarity()
    demo_autocorrelation()
    demo_decomposition()
    demo_lag_features()
    demo_walk_forward()
    demo_random_vs_walk_forward()
    demo_exponential_smoothing()
    demo_evaluation_metrics()
    demo_forecasting()
    demo_industrial_tools()
