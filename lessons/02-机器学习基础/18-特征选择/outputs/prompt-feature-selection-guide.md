---
name: feature-selection-guide
description: 根据数据集特征和工程约束，选择最合适的特征选择方法
version: 1.0.0
phase: 2
lesson: 18
tags: [feature-selection, filter, wrapper, embedded, lasso, rfe, mutual-information]
---

# 特征选择方法决策指南

根据当前数据集特征和工程约束，选择最合适的特征选择方法。

## 第 1 步：基础清理（必做，零成本）

在应用任何方法之前，先移除显然无用的特征：

- **常数特征**：方差 = 0。直接移除。
- **近常数特征**：方差 < 0.01（或自定义阈值）。直接移除。
- **重复特征**：完全相同的列。保留一个，移除其余。
- **ID 列**：每行唯一值，不包含可泛化信息。直接移除。

这一步通常能在几秒钟内移除 10% ~ 30% 的噪声特征。

## 第 2 步：根据场景选择方法

### 快速决策树

```
你的数据集有多少特征？
│
├── < 50 个
│   └── 使用互信息评分，保留 Top-K
│
├── 50 ~ 500 个
│   ├── 线性模型？→ L1 正则化（Lasso）
│   └── 树模型？→ 树模型特征重要性
│
└── > 500 个
    └── 链式策略：
        方差阈值 → 互信息筛选 Top-50% → RFE 精挑细选
```

### 按需求选择

| 你的需求 | 推荐方法 | 原因 |
|---|---|---|
| 快速筛选 | 互信息 | 不依赖模型，秒级完成 |
| 需要稀疏可解释的解 | L1 正则化 | 精确给出"选/不选"的二元结论 |
| 需要捕捉非线性关系 | 互信息 或 树重要性 | 不假设线性关系 |
| 需要考虑特征交互 | RFE 或 树重要性 | 过滤法只看单个特征 |
| 最终验证 | 置换重要性 | 模型无关，可靠但慢 |

### 方法速查表

| 方法 | 何时使用 | 何时避免 |
|---|---|---|
| 方差阈值 | 所有场景的第一步 | 永远不要跳过 |
| 互信息 | 快速排序、非线性关系 | 需要捕捉特征交互时 |
| RFE | 精确选择、特征数 < 500 | 模型训练极慢、特征数 > 1000 |
| L1 / Lasso | 线性模型、需要稀疏解 | 非线性问题、特征高度相关 |
| 树重要性 | 非线性关系、特征交互 | 数据中有高基数类别特征 |

## 第 3 步：验证选择结果

- 在**选定特征**上训练模型，与**使用全部特征**的模型对比
- 使用交叉验证，而非单次训练/测试划分
- 如果性能下降超过 1% ~ 2%，说明可能移除了有用特征
- 如果性能提升，说明成功移除了噪声

## 第 4 步：避免常见陷阱

### 相关特征
- L1 会"任意"保留相关特征组中的一个，将其他置零
- 如果在意这个行为，先用相关系数矩阵做人工决策
- 树模型重要性在相关特征间分散重要性——不一定全是坏事

### 数据泄露
- 特征选择器只能在训练集上拟合
- 相同选择器应用到测试集
- 在交叉验证中，特征选择必须发生在每个 fold 内部

### 过拟合选择
- RFE 迭代次数过多会过拟合训练数据
- 使用 RFECV（带交叉验证的 RFE）自动确定最优特征数

## 第 5 步：生产部署检查清单

- [ ] 方差阈值作为第一步已应用
- [ ] 特征选择器仅在训练集上拟合
- [ ] 被选中的特征已记录（名称、方法、分数）
- [ ] 使用选定特征 vs 全部特征的性能已对比
- [ ] 使用交叉验证评估，而非单次划分
- [ ] 特征选择已集成到训练流水线中（非手动操作）
- [ ] 已部署特征漂移监控（选中的特征可能随时间失效）

---

## 使用示例

### Python 代码片段：链式特征选择

```python
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# 第 1 步：方差阈值
selector_var = VarianceThreshold(threshold=0.01)
X_clean = selector_var.fit_transform(X_train)

# 第 2 步：互信息筛选 Top-50%
mi_scores = mutual_info_classif(X_clean, y_train)
top_half = np.argsort(mi_scores)[-X_clean.shape[1] // 2:]
X_mi = X_clean[:, top_half]

# 第 3 步：L1 精挑细选
selector_l1 = SelectFromModel(Lasso(alpha=0.01))
X_final = selector_l1.fit_transform(X_mi, y_train)

print(f"原始特征数：{X_train.shape[1]}")
print(f"最终特征数：{X_final.shape[1]}")
```
