# main.py — ML 流水线从零实现与工业实践
# 依赖：numpy>=1.24, scikit-learn>=1.3（可选）、joblib>=1.3（可选）
# 安装：pip install numpy scikit-learn joblib
# 对应课程：阶段 02 · 13（ML 流水线）

import numpy as np
from typing import List, Dict, Tuple, Any, Optional


# ============================================================
# 第 1 步：自定义变换器（从零实现）
# ============================================================

class MedianImputer:
    """中位数填充器——处理缺失值的最简实现。

    思想：在 fit 阶段只记录训练集的中位数，transform 阶段用这些
    中位数填充缺失值。测试集绝不参与中位数的计算。
    """

    def __init__(self):
        self.medians: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MedianImputer":
        # nanmedian 忽略 NaN 计算中位数，axis=0 按列计算
        self.medians = np.nanmedian(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X.copy()
        for col in range(X.shape[1]):
            mask = np.isnan(X_out[:, col])
            X_out[mask, col] = self.medians[col]
        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SimpleScaler:
    """标准化缩放器——零均值单位方差。

    注意：除以标准差前先处理 std=0 的列，否则会产生除以零错误。
    """

    def __init__(self):
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SimpleScaler":
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        # 常数列标准差为零，避免除以零
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.means) / self.stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SimpleOneHotEncoder:
    """独热编码器——将类别列转换为一组二进制列。

    handle_unknown="ignore" 的含义：遇到训练时未见过的类别时，
    返回全零向量，而不是报错。这对生产环境至关重要——新城市、
    新商品、新用户都可能出现训练集里从未有过的类别。
    """

    def __init__(self, handle_unknown: str = "ignore"):
        self.categories: List[List[str]] = []
        self.handle_unknown = handle_unknown

    def fit(self, X: np.ndarray) -> "SimpleOneHotEncoder":
        self.categories = []
        for col in range(X.shape[1]):
            self.categories.append(sorted(set(X[:, col])))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        encoded_cols = []
        for col in range(X.shape[1]):
            cats = self.categories[col]
            for cat in cats:
                encoded_cols.append((X[:, col] == cat).astype(float))
        return np.column_stack(encoded_cols) if encoded_cols else np.zeros((X.shape[0], 0))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ============================================================
# 第 2 步：流水线从零实现
# ============================================================

class PipelineFromScratch:
    """最简流水线——将多个变换器和一个模型串联成一个对象。

    核心保证：fit 时只在训练数据上学习参数，predict 时使用
    已学习的参数处理新数据。整个对象可以序列化、部署、复现。
    """

    def __init__(self, steps: List[Tuple[str, Any]]):
        # steps: [(name, transformer), ..., (name, model)]
        # 最后一步必须是模型（有 fit/predict），前面都是变换器（有 fit_transform/transform）
        self.steps = steps

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PipelineFromScratch":
        X_current = X.copy()
        # 除了最后一步（模型），其余全部 fit_transform
        for name, step in self.steps[:-1]:
            X_current = step.fit_transform(X_current)
        # 最后一步是模型，用处理后的数据训练
        _, model = self.steps[-1]
        model.fit(X_current, y)
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """对变换器部分执行 fit_transform（不含最后的模型）。"""
        X_current = X.copy()
        for name, step in self.steps:
            X_current = step.fit_transform(X_current)
        return X_current

    def transform(self, X: np.ndarray) -> np.ndarray:
        """对变换器部分执行 transform（不含最后的模型）。"""
        X_current = X.copy()
        for name, step in self.steps:
            X_current = step.transform(X_current)
        return X_current

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_current = X.copy()
        # 变换器只 transform，不重新学习参数
        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)
        _, model = self.steps[-1]
        return model.predict(X_current)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        return float(np.mean(pred == y))


class LogisticRegressionSimple:
    """逻辑回归的极简实现（梯度下降），仅用于教学。"""

    def __init__(self, lr: float = 0.01, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionSimple":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iter):
            z = X @ self.weights + self.bias
            pred = self._sigmoid(z)
            dw = (1 / n_samples) * X.T @ (pred - y)
            db = (1 / n_samples) * np.sum(pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return (self._sigmoid(z) >= 0.5).astype(int)


# ============================================================
# 第 3 步：数据生成与切分
# ============================================================

def make_mixed_data(n_samples: int = 500, seed: int = 42) -> Dict:
    """生成一个包含数值列、类别列、缺失值的模拟数据集。

    模拟场景：一个电商用户是否会购买会员的预测任务。
    特征：年龄、收入、信用评分（数值），城市、会员计划（类别）。
    目标：是否购买高级会员（0/1）。
    """
    rng = np.random.RandomState(seed)

    # 数值列
    age = rng.normal(35, 12, n_samples).clip(18, 80)
    income = rng.lognormal(10.5, 0.8, n_samples)
    score = rng.uniform(300, 850, n_samples)

    # 类别列
    cities = np.array(["beijing", "shanghai", "guangzhou", "shenzhen", "hangzhou"])
    city = rng.choice(cities, n_samples)
    plans = np.array(["free", "basic", "premium"])
    plan = rng.choice(plans, n_samples, p=[0.5, 0.3, 0.2])

    # 随机制造缺失值（模拟真实数据）
    age_with_missing = age.copy()
    age_with_missing[rng.random(n_samples) < 0.05] = np.nan

    income_with_missing = income.copy()
    income_with_missing[rng.random(n_samples) < 0.03] = np.nan

    # 目标变量（基于特征的非线性组合 + 噪声）
    boundary = (
        0.01 * (age - 35)
        + 0.00001 * (income - 40000)
        + 0.002 * (score - 600)
        + 0.5 * (plan == "premium").astype(float)
        - 0.3 * (plan == "free").astype(float)
        + rng.normal(0, 0.5, n_samples)
    )
    target = (boundary > 0).astype(int)

    return {
        "age": age_with_missing,
        "income": income_with_missing,
        "score": score,
        "city": city,
        "plan": plan,
        "target": target,
    }


def train_test_split_dict(data: Dict, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(data["target"])
    idx = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = idx[:split], idx[split:]
    train = {k: v[train_idx] for k, v in data.items()}
    test = {k: v[test_idx] for k, v in data.items()}
    return train, test


# ============================================================
# 第 4 步：完整流水线（支持混合数据类型）
# ============================================================

class FullPipeline:
    """完整流水线——处理数值列和类别列，防止数据泄露。

    设计原则：所有预处理器在 fit 阶段只在训练数据上学习参数，
    predict 阶段复用这些参数处理新数据。整个对象可序列化部署。
    """

    def __init__(self, model: Any, numeric_cols: List[str], categorical_cols: List[str]):
        self.model = model
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.num_pipeline = PipelineFromScratch([
            ("impute", MedianImputer()),
            ("scale", SimpleScaler()),
        ])
        self.cat_encoder = SimpleOneHotEncoder(handle_unknown="ignore")

    def _prepare_X(self, data: Dict) -> np.ndarray:
        """将字典格式的数据合并为特征矩阵。"""
        X_num = np.column_stack([data[c] for c in self.numeric_cols])
        X_cat = np.column_stack([data[c] for c in self.categorical_cols])
        X_num_processed = self.num_pipeline.transform(X_num)
        X_cat_processed = self.cat_encoder.transform(X_cat)
        return np.hstack([X_num_processed, X_cat_processed])

    def fit(self, data: Dict) -> "FullPipeline":
        X_num = np.column_stack([data[c] for c in self.numeric_cols])
        X_cat = np.column_stack([data[c] for c in self.categorical_cols])

        X_num_processed = self.num_pipeline.fit_transform(X_num)
        X_cat_processed = self.cat_encoder.fit_transform(X_cat)

        X = np.hstack([X_num_processed, X_cat_processed])
        self.model.fit(X, data["target"])
        return self

    def predict(self, data: Dict) -> np.ndarray:
        X = self._prepare_X(data)
        return self.model.predict(X)

    def score(self, data: Dict) -> float:
        pred = self.predict(data)
        return float(np.mean(pred == data["target"]))


# ============================================================
# 第 5 步：交叉验证（防泄露）
# ============================================================

def cross_validate(pipeline_factory, data: Dict, n_folds: int = 5, seed: int = 42) -> List[float]:
    """K 折交叉验证——确保每个折的预处理器只在训练折上拟合。

    这是 Pipeline 防泄露能力的关键体现：即使做交叉验证，
    每个折内的 imputer、scaler、encoder 都只能看到训练折数据。
    """
    rng = np.random.RandomState(seed)
    n = len(data["target"])
    idx = rng.permutation(n)
    fold_size = n // n_folds
    scores = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = idx[val_start:val_end]
        train_idx = np.concatenate([idx[:val_start], idx[val_end:]])

        train_data = {k: v[train_idx] for k, v in data.items()}
        val_data = {k: v[val_idx] for k, v in data.items()}

        pipe = pipeline_factory()
        pipe.fit(train_data)
        scores.append(pipe.score(val_data))

    return scores


# ============================================================
# 演示函数
# ============================================================

def demo_data_leakage():
    """演示数据泄露：scaler 在全量数据上拟合会导致准确率高估。"""
    print("=" * 60)
    print("演示 1：数据泄露的影响")
    print("=" * 60)

    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    # --- 泄露做法：scaler 在全量数据上拟合 ---
    scaler_leaky = SimpleScaler()
    X_scaled_leaky = scaler_leaky.fit_transform(X)
    X_train_leaky = X_scaled_leaky[:160]
    X_test_leaky = X_scaled_leaky[160:]
    y_train, y_test = y[:160], y[160:]

    model_leaky = LogisticRegressionSimple(lr=0.1, n_iter=500)
    model_leaky.fit(X_train_leaky, y_train)
    acc_leaky = float(np.mean(model_leaky.predict(X_test_leaky) == y_test))

    # --- 正确做法：scaler 只在训练集上拟合 ---
    X_train = X[:160]
    X_test = X[160:]
    scaler_clean = SimpleScaler()
    X_train_clean = scaler_clean.fit_transform(X_train)
    X_test_clean = scaler_clean.transform(X_test)

    model_clean = LogisticRegressionSimple(lr=0.1, n_iter=500)
    model_clean.fit(X_train_clean, y_train)
    acc_clean = float(np.mean(model_clean.predict(X_test_clean) == y_test))

    print(f"  泄露做法（scaler 看全量数据）: {acc_leaky:.3f}")
    print(f"  正确做法（scaler 只看训练集）: {acc_clean:.3f}")
    print(f"  差异: {acc_leaky - acc_clean:+.3f}")
    print()
    print("  在本例中差异可能不大，但在涉及目标编码或特征选择的")
    print("  真实场景中，泄露可将准确率高估 10-30%。")
    print()


def demo_pipeline_basic():
    """演示基本流水线的 fit/predict 流程。"""
    print("=" * 60)
    print("演示 2：从零实现的流水线")
    print("=" * 60)

    rng = np.random.RandomState(42)
    X = rng.randn(300, 5)
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] > 0).astype(int)

    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    pipe = PipelineFromScratch([
        ("scaler", SimpleScaler()),
        ("model", LogisticRegressionSimple(lr=0.1, n_iter=500)),
    ])

    pipe.fit(X_train, y_train)
    train_acc = pipe.score(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)

    print(f"  流水线（scaler + 逻辑回归）:")
    print(f"  训练集准确率: {train_acc:.3f}")
    print(f"  测试集准确率: {test_acc:.3f}")
    print()


def demo_full_pipeline():
    """演示完整流水线处理混合数据类型。"""
    print("=" * 60)
    print("演示 3：混合数据类型的完整流水线")
    print("=" * 60)

    data = make_mixed_data(n_samples=500)
    train, test = train_test_split_dict(data)

    pipe = FullPipeline(
        model=LogisticRegressionSimple(lr=0.05, n_iter=1000),
        numeric_cols=["age", "income", "score"],
        categorical_cols=["city", "plan"],
    )

    pipe.fit(train)
    train_acc = pipe.score(train)
    test_acc = pipe.score(test)

    print(f"  完整流水线（填充 + 缩放 + 编码 + 逻辑回归）:")
    print(f"  训练集准确率: {train_acc:.3f}")
    print(f"  测试集准确率: {test_acc:.3f}")
    print()


def demo_cross_validation():
    """演示交叉验证防泄露。"""
    print("=" * 60)
    print("演示 4：交叉验证防泄露")
    print("=" * 60)

    data = make_mixed_data(n_samples=500)

    def make_pipeline():
        return FullPipeline(
            model=LogisticRegressionSimple(lr=0.05, n_iter=1000),
            numeric_cols=["age", "income", "score"],
            categorical_cols=["city", "plan"],
        )

    scores = cross_validate(make_pipeline, data, n_folds=5)

    print(f"  5 折交叉验证准确率: {[f'{s:.3f}' for s in scores]}")
    print(f"  均值: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    print()
    print("  每一折的预处理器只在训练折上拟合，验证折绝不参与。")
    print()


def demo_handle_unknown():
    """演示未知类别的处理——生产环境必备。"""
    print("=" * 60)
    print("演示 5：未知类别的优雅处理")
    print("=" * 60)

    # 训练时见过的城市
    train_cats = np.array([["beijing"], ["shanghai"], ["guangzhou"]])
    encoder = SimpleOneHotEncoder(handle_unknown="ignore")
    encoder.fit(train_cats)

    print(f"  训练时见过的类别: {encoder.categories[0]}")
    print(f"  'beijing' 编码结果: {encoder.transform(np.array([['beijing']]))[0]}")

    # 生产环境出现新城市
    unknown = np.array([["chengdu"]])
    unknown_encoded = encoder.transform(unknown)
    print(f"  'chengdu'（未知）编码结果: {unknown_encoded[0]}")
    print(f"  全零向量表示——模型不会崩溃，只是没有先验信息。")
    print()


def demo_serialization():
    """演示模型持久化——joblib 序列化完整流水线。"""
    print("=" * 60)
    print("演示 6：流水线持久化（joblib）")
    print("=" * 60)

    try:
        import joblib
        import tempfile
        import os
    except ImportError:
        print("  joblib 未安装，跳过持久化演示。")
        print("  安装: pip install joblib")
        print()
        return

    data = make_mixed_data(n_samples=500)
    train, test = train_test_split_dict(data)

    # 训练并保存
    pipe = FullPipeline(
        model=LogisticRegressionSimple(lr=0.05, n_iter=1000),
        numeric_cols=["age", "income", "score"],
        categorical_cols=["city", "plan"],
    )
    pipe.fit(train)

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        model_path = f.name

    joblib.dump(pipe, model_path)
    print(f"  流水线已保存到: {model_path}")

    # 模拟"另一个进程"加载并预测
    loaded_pipe = joblib.load(model_path)
    original_acc = pipe.score(test)
    loaded_acc = loaded_pipe.score(test)

    print(f"  原始流水线测试准确率: {original_acc:.3f}")
    print(f"  加载后流水线测试准确率: {loaded_acc:.3f}")
    print(f"  结果一致: {abs(original_acc - loaded_acc) < 1e-10}")

    # 清理
    os.unlink(model_path)
    print()


def demo_reproducibility():
    """演示可复现性——相同种子得到相同结果。"""
    print("=" * 60)
    print("演示 7：可复现性验证")
    print("=" * 60)

    data = make_mixed_data(n_samples=500, seed=42)

    def make_pipeline():
        return FullPipeline(
            model=LogisticRegressionSimple(lr=0.05, n_iter=1000),
            numeric_cols=["age", "income", "score"],
            categorical_cols=["city", "plan"],
        )

    # 相同种子，相同数据
    run1 = cross_validate(make_pipeline, data, n_folds=5, seed=42)
    run2 = cross_validate(make_pipeline, data, n_folds=5, seed=42)
    run3 = cross_validate(make_pipeline, data, n_folds=5, seed=99)

    print(f"  第 1 次（seed=42）: {[f'{s:.4f}' for s in run1]}")
    print(f"  第 2 次（seed=42）: {[f'{s:.4f}' for s in run2]}")
    print(f"  第 3 次（seed=99）: {[f'{s:.4f}' for s in run3]}")
    print(f"  第 1 次 == 第 2 次: {all(abs(a - b) < 1e-10 for a, b in zip(run1, run2))}")
    print(f"  第 1 次 == 第 3 次: {all(abs(a - b) < 1e-10 for a, b in zip(run1, run3))}")
    print()
    print("  相同种子、相同数据、相同结果——这就是可复现性。")
    print()


def demo_sklearn_pipeline():
    """演示 sklearn 工业级流水线——ColumnTransformer + Pipeline。"""
    print("=" * 60)
    print("演示 8：sklearn 工业级流水线")
    print("=" * 60)

    try:
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.compose import ColumnTransformer as SkColumnTransformer
        from sklearn.preprocessing import StandardScaler as SkScaler
        from sklearn.preprocessing import OneHotEncoder as SkOHE
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        import pandas as pd
    except ImportError:
        print("  scikit-learn 未安装，跳过此演示。")
        print("  安装: pip install scikit-learn")
        print()
        return

    data = make_mixed_data(n_samples=500)
    df = pd.DataFrame({
        "age": data["age"],
        "income": data["income"],
        "score": data["score"],
        "city": data["city"],
        "plan": data["plan"],
    })
    y = data["target"]

    # 数值列流水线：中位数填充 + 标准化
    numeric_pipe = SkPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", SkScaler()),
    ])

    # 类别列流水线：众数填充 + 独热编码（ignore 未知类别）
    cat_pipe = SkPipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", SkOHE(handle_unknown="ignore", sparse_output=False)),
    ])

    # ColumnTransformer：不同类型列走不同流水线
    preprocessor = SkColumnTransformer([
        ("num", numeric_pipe, ["age", "income", "score"]),
        ("cat", cat_pipe, ["city", "plan"]),
    ])

    # 完整流水线：预处理 + 模型
    full_pipe = SkPipeline([
        ("preprocess", preprocessor),
        ("model", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ])

    # 5 折交叉验证（整个流水线在每个折上独立拟合）
    scores = cross_val_score(full_pipe, df, y, cv=5, scoring="accuracy")

    print(f"  sklearn GBM + ColumnTransformer:")
    print(f"  5 折交叉验证: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  每折准确率: {[f'{s:.3f}' for s in scores]}")
    print()

    # 持久化
    full_pipe.fit(df, y)
    print(f"  流水线步骤: {[name for name, _ in full_pipe.steps]}")
    print(f"  预处理器子流水线: {[name for name, _, _ in preprocessor.transformers]}")
    print()


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    demo_data_leakage()
    demo_pipeline_basic()
    demo_full_pipeline()
    demo_cross_validation()
    demo_handle_unknown()
    demo_serialization()
    demo_reproducibility()
    demo_sklearn_pipeline()
    print("所有流水线演示完成。")
