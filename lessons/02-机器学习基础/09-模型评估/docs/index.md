# 模型评估——你的模型到底行不行，不是拍脑袋决定的

> 没有评估的机器学习只是调参游戏。测试集是你唯一的陪审团，用过一次就作废。

**类型：** 实现课
**语言：** Python
**前置知识：** 第 02 阶段 · 01-08（全部基础知识）
**预计时间：** ~90 分钟
**所处阶段：** Tier 1
**关联课程：** 第 02 阶段 · 08（过拟合与正则化）— 评估是判断过拟合的唯一手段

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现混淆矩阵、精确率、召回率、F1 分数、ROC/AUC，解释每个指标的适用场景
- [ ] 解释为什么准确率在类别不平衡时是误导性的，并选择合适的替代指标
- [ ] 实现 K 折与分层 K 折交叉验证，理解如何选择 K 值
- [ ] 通过配对 t 检验和 McNemar 检验判断两个模型是否有显著差异
- [ ] 绘制学习曲线并据此诊断模型是否过拟合或欠拟合

---

## 1. 问题

你训练了一个癌症筛查模型，准确率 99%。听起来不错？如果你的测试数据里只有 1% 是患者，一个永远说"你没事"的模型准确率就是 99%——它一个患者都没找出来，但数字很漂亮。

更常见的是：你训练了三个模型，准确率分别是 92.1%、92.4%、92.5%。哪个最好？你选最高的交差上线，结果到了生产环境发现其实没区别——或者更糟，是更差的那个。因为这些数字可能是噪声，根本不是真实的差异。

模型评估要回答的从来不只是"准不准"，而是：

1. **用什么指标衡量**——准确率可能是在骗你
2. **这个分数能信吗**——一次随机划分可能恰好"友好"
3. **模型 A 真的比模型 B 好吗**——还是只是运气
4. **还能改进吗**——问题在偏差还是方差

搞错了任何一个，你的论文就不可复现，你的线上模型就会让你加班。

---

## 2. 概念

### 2.1 混淆矩阵——所有分类指标的源头

二分类问题的每一次预测有四种可能：

```
              预测正类          预测负类
真实正类  │  真正例 (TP)   │   假反例 (FN)   │
真实负类  │  假正例 (FP)   │   真反例 (TN)   │
```

这四个数字是分类评估的"原子"。所有高阶指标都从这里推导：

| 指标 | 公式 | 直觉含义 |
|---|---|---|
| **准确率** | (TP+TN) / 总数 | 总体猜对的比例——可能骗人 |
| **精确率** | TP / (TP+FP) | 模型说"是"的那些，多少是真的 |
| **召回率** | TP / (TP+FN) | 实际为"是"的那些，被找出了多少 |
| **F1 分数** | 2·P·R / (P+R) | 精确率和召回率的"绑匪均值" |

### 2.2 精确率 vs 召回率：不可能同时最大化

```
阈值降低 → 召回率上升（抓出更多正例），精确率下降（更多误报）
阈值升高 → 精确率上升（选出的更准），   召回率下降（漏掉更多）
```

```
精确率 ──────────────── 高精确率：宁可漏掉不可错杀
                                    ↑
                            垃圾邮件分类
                            法律风险预测
                                    │
                         ← 两者平衡 →
                                    │
                        疾病诊断
                        欺诈检测
                                    ↑
召回率 ──────────────── 高召回率：宁可错杀不可漏掉
```

### 2.3 ROC 曲线与 AUC

**ROC 曲线**描绘了阈值变化时，模型的真阳率（TPR）和假阳率（FPR）之间的权衡：

```
    TPR  ^
    1.0  │          ********
         │        **
    0.8  │      **
         │    **
    0.6  │   *
         │  *
    0.4  │ *
         │*
    0.2  │
         │*
    0.0  └──────────────────→ FPR
         0   0.2  0.4  0.6  0.8  1.0

  *****  优秀模型的 ROC（贴近左上角）
  -----  对角线 = 随机猜测（AUC = 0.5）
```

**AUC** 是 ROC 曲线下的面积，取值 0 到 1。它的直觉："随机抽一个正样本和一个负样本，模型给正样本的打分高于负样本的概率"。

### 2.4 交叉验证：让你的评估稳定可靠

单次划分存在随机性。**交叉验证**通过多次划分取平均来降低方差：

```
普通 5 折交叉验证:

折 1: [验证] [训练] [训练] [训练] [训练]   得分 s1
折 2: [训练] [验证] [训练] [训练] [训练]   得分 s2
折 3: [训练] [训练] [验证] [训练] [训练]   得分 s3
折 4: [训练] [训练] [训练] [验证] [训练]   得分 s4
折 5: [训练] [训练] [训练] [训练] [验证]   得分 s5

最终得分 = 均值(s1..s5)   稳定性 = 标准差(s1..s5)
```

**分层 K 折**：每折保持类别比例与全集一致。对不平衡数据——如欺诈检测中正样本只有 1%——普通 K 折可能使某折完全没有正样本，分层保证了代表性。

### 2.5 统计检验：差异是真实的，还是噪声？

两个模型在同一数据集上分别得到 87.3% 和 87.6% 的准确率。差别是真的改进还是随机波动？

**配对 t 检验**（交叉验证场景）：

$$t = \frac{\bar{d}}{s_d / \sqrt{K}}$$

其中 $d_i$ 是第 $i$ 折两模型的得分差，$K$ 是折数。当 $|t| > t_{\alpha/2, K-1}$ 时拒绝"两模型无差异"的零假设。

**McNemar 检验**（单次测试集场景）：只关注两个模型预测结果不同（分歧）的部分。零假设：两个模型的错误率相同。

$$\chi^2 = \frac{(|b-c| - 1)^2}{b+c}$$

其中 $b$ 是"A 错 B 对"的样本数，$c$ 是"A 对 B 错"的样本数。

### 2.6 学习曲线：诊断偏差与方差

```
情况 A：高偏差（欠拟合）
准确率 ^
  0.80 │--T, V--T, V--T, V--
       │              两线接近且都低
  0.60 │
       └──────────────────→ 训练集大小
       收集更多数据无法改善

情况 B：高方差（过拟合）
准确率 ^
  1.00 │T
       │ T         训练分远高于验证分
  0.90 │  T           间隙大
       │   T
  0.80 │    V---V---V   验证分趋于平坦
       └──────────────────→ 训练集大小
       更多数据可能改善

情况 C：理想状态
准确率 ^
  0.95 │    T
       │  T   T        两线接近且都高
       │ V     V
  0.85 │   V    V
       └──────────────────→ 训练集大小
```

---

## 3. 从零实现

### 第 1 步：数据划分与交叉验证

```python
import random
import math
from collections import Counter
from typing import List, Tuple, Optional


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    """将数据按比例划分为训练集、验证集、测试集。"""
    random.seed(seed)  # 保证可复现
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # 根据索引提取子集
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, y_train, X_val, y_val, X_test, y_test


def kfold_split(n, k=5, seed=42):
    """生成 K 折索引。最后一个 fold 接收所有剩余样本，不丢弃。"""
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n  # 最后一折兜底
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))
    return folds


def stratified_kfold_split(y, k=5, seed=42):
    """分层 K 折——每个类别独立切分，保持比例一致。"""
    random.seed(seed)
    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    for label in class_indices:
        random.shuffle(class_indices[label])

    folds = [{"train": [], "val": []} for _ in range(k)]
    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(indices)
            folds[i]["val"].extend(indices[start:end])
            folds[i]["train"].extend(indices[:start] + indices[end:])
    return [(f["train"], f["val"]) for f in folds]
```

**关键设计**：`stratified_kfold_split` 先按类别分组，再在每个类别内独立切分。这保证了即使某个类别只有 3 个样本，每个折也至少有 1 个。

### 第 2 步：混淆矩阵与基础指标

```python
def confusion_matrix(y_true, y_pred):
    """计算四格：TP、TN、FP、FN。"""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    """准确率——直觉但可能骗人。"""
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true, y_pred):
    """精确率——所有被预测为正的，多少是真正为正。"""
    tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    """召回率——所有真实为正的，被找出了多少。"""
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    """F1 分数：P 和 R 的调和平均。任一低，F1 即低。"""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
```

### 第 3 步：ROC 曲线与 AUC

```python
def roc_curve(y_true, y_scores):
    """计算 ROC 曲线上的点。遍历所有可能的阈值。"""
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list, fpr_list = [], []

    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos

    for threshold in thresholds:
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)

        tpr_list.append(tp / total_pos if total_pos > 0 else 0.0)
        fpr_list.append(fp / total_neg if total_neg > 0 else 0.0)

    return fpr_list, tpr_list, thresholds


def auc_roc(y_true, y_scores):
    """梯形法计算 ROC 曲线下面积。"""
    fpr_list, tpr_list, _ = roc_curve(y_true, y_scores)

    # 按 FPR 排序保证积分正确
    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    area = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i - 1]
        height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        area += width * height
    return area
```

**算法要点**：ROC 需要对每个可能的阈值重新计算预测结果。这在样本量大时效率低，实际工程中会先按分数排序再扫描一次（$O(n \log n)$）。

### 第 4 步：回归指标

```python
def mse(y_true, y_pred):
    """均方误差。对大误差惩罚更重。"""
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true, y_pred):
    """均方根误差。和原始数据同量纲。"""
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """平均绝对误差。对异常值更鲁棒。"""
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)


def r_squared(y_true, y_pred):
    """决定系数 R²。1=完美，0=和猜均值一样，负=比猜均值还差。"""
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
```

### 第 5 步：统计检验

```python
def paired_ttest(scores_a, scores_b):
    """配对 t 检验。返回 (|t|, mean_diff, std_diff)。

    当 |t| > t_critical（df=K-1, alpha=0.05）时，"两模型有差异"
    结论在统计上显著。5 折交叉验证对应 t_critical ≈ 2.78。
    """
    n = len(scores_a)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / n
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)) if n > 1 else 1.0
    t_stat = mean_diff / (std_diff / math.sqrt(n)) if std_diff > 0 else 0.0
    return t_stat, mean_diff, std_diff


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar 检验——基于双方分歧构建统计量。

    b: A 错 B 对的样本数
    c: A 对 B 错的样本数
    卡方 = (|b-c|-1)^2 / (b+c)，df=1, p<0.05 临界值 3.84
    """
    b = sum(1 for yt, pa, pb in zip(y_true, pred_a, pred_b) if pa != yt and pb == yt)
    c = sum(1 for yt, pa, pb in zip(y_true, pred_a, pred_b) if pa == yt and pb != yt)
    if b + c == 0:
        return 0.0, b, c
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # 连续性校正
    return chi2, b, c
```

### 第 6 步：完整演示

```python
if __name__ == "__main__":
    # 1. 生成不平衡数据演示准确率的陷阱
    X_imb, y_imb = make_imbalanced_data(300, minority_ratio=0.05)
    print(f"全猜负的准确率: {accuracy(y_imb, [0]*len(y_imb)):.2%}")  # 95%!

    # 2. 交叉验证
    cv_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5, metric_fn=accuracy, stratified=True,
    )
    print(f"5 折准确率: {[round(s, 4) for s in cv_scores]}")
    print(f"均值±标准差: {sum(cv_scores)/len(cv_scores):.4f} ± "
          f"{(sum((s-sum(cv_scores)/len(cv_scores))**2 for s in cv_scores)/len(cv_scores))**0.5:.4f}")

    # 3. 统计检验
    t_stat, _, _ = paired_ttest(scores_a, scores_b)
    print(f"t 统计量 = {t_stat:.4f}, {'显著' if abs(t_stat) > 2.78 else '不显著'}")
```

> 完整可运行代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 scikit-learn 核心评估接口

```python
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# 一次性获得所有分类指标
print(classification_report(y_true, y_pred))
```

输出示例：

```text
              precision    recall  f1-score   support

     class 0       0.89      0.93      0.91        41
     class 1       0.88      0.82      0.85        28

    accuracy                           0.89        69
   macro avg       0.89      0.88      0.88        69
weighted avg       0.89      0.89      0.89        69
```

### 4.2 交叉验证的高级用法

```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

# 同时返回多个指标（ sklearn ≥ 0.24 ）
scoring = ["accuracy", "f1_macro", "roc_auc"]
cv_results = cross_validate(
    RandomForestClassifier(),
    X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=scoring,
    return_train_score=True,  # 同时返回训练集分数——有助于诊断过拟合
)

for metric in scoring:
    train_mean = cv_results[f"train_{metric}"].mean()
    val_mean = cv_results[f"test_{metric}"].mean()
    val_std = cv_results[f"test_{metric}"].std()
    print(f"{metric}: 训练={train_mean:.4f}  验证={val_mean:.4f} ± {val_std:.4f}")
```

### 4.3 不平衡数据的专用工具

```python
from sklearn.utils import compute_class_weight
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn

# 计算类别权重——让少数类错分的惩罚更重
weights = compute_class_weight("balanced", classes=[0, 1], y=y_train)
class_weight = {0: weights[0], 1: weights[1]}
print(f"类别权重: {class_weight}")  # {0: 0.52, 1: 9.5} —— 正样本分错代价是负的 18 倍

# 或者使用 SMOTE 过采样少数类
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 4.4 性能对比

| 工具 | 适用场景 | 优势 |
|---|---|---|
| `scikit-learn.metrics` | 快速单次评估 | `report()` 一行输出完整分类报告 |
| `sklearn.cross_validate` | 交叉验证 | 多指标、并行、train/test 同时返回 |
| `imbalanced-learn` | 不平衡数据 | SMOTE、代价敏感权重、不平衡 F1 |
| `scipy.stats` | 统计检验 | `ttest_rel()`（配对 t）、`chi2_contingency()`（McNemar） |
| `mlflow / wandb` | 实验记录 | 跟踪每次实验的指标、参数、代码版本 |

---

## 5. 知识连线

本课学习的评估方法，是后续所有机器学习和深度学习课程的必备基础：

- **阶段 02（机器学习基础）**：本阶段的后续项目（如 K aggle 竞赛）都需要严格的评估框架来比较多种模型，交叉验证将成为日常工具
- **阶段 03（深度学习核心）**：训练神经网络时，学习曲线是判断是否过拟合的黄金工具；验证集指标决定何时提前停止（Early Stopping）
- **阶段 11（大语言模型工程）**：LLM 评估使用 BLEU、ROUGE 等序列指标，以及困惑度（Perplexity）——这些指标的构建逻辑（"和基线相比提升多少"）与本课一脉相承

---

## 6. 工程最佳实践

### 6.1 评估流程检查清单

| 步骤 | 操作 | 为什么 |
|---|---|---|
| 1. 划定测试集 | 在任何分析之前划分并锁住 | 防止无意识的数据泄露 |
| 2. 选基线 | 多数类分类器 / 均值回归 | 不能击败基线的模型是废料 |
| 3. 交叉验证 | K=5 或 10，分类任务用分层 | 单次划分方差大，容易乐观 |
| 4. 符合业务 | 选与业务目标对齐的指标 | 准确率不等于业务结果 |
| 5. 统计检验 | 差异 ≥ 1% 时做 t 检验或 McNemar | 噪声区间内选简单模型 |

### 6.2 中文场景特别建议

- 文本分类时，中文应以**句子**或**篇章**为单位切分，而非按字；否则 K 折会产生长度偏差
- OCR 后处理模型的评估，建议同时报告字级别准确率与句子级别准确率——前者对过程优化有用，后者对业务有用
- 中文命名实体识别（NER）常用 F1 而非准确率，因为"非实体" token 超过 90%，准确率毫无信息量
- 搜索排序场景使用 **NDCG** 或 **MAP**，不要沿用分类指标

### 6.3 踩坑经验

- **先做预处理再做划分**——导致数据泄露。标准化、缺失值填充的均值必须只在训练集上计算，然后应用到验证/测试集
- **在不平衡数据上用准确率调参**——模型把全部分类都预测成多数类，准确率很好看但召回率为零
- **用了 100 次测试集**——每次调参都看测试集，测试集变成了测试+验证。应该用验证集调参，测试集只最终用一次
- **K=2 或 K=3 做交叉验证**——折数太少，方差大，评估不稳定。至少 K=5
- **学习曲线没观察训练分数**——只看验证分数无法诊断是高偏差还是高方差，两者对应的补救措施完全相反

---

## 7. 常见错误

### 错误 1：准确率当万能指标

**现象：** 违约预测模型准确率 98%，上线后一笔违约都没拦住。

**原因：** 数据中违约样本仅占 2%——模型全都判"正常"准确率就是 98%。它学到的唯一东西是"猜多数类"。

**修复：**

```python
# ❌ 只看准确率
print(f"准确率: {accuracy_score(y_test, y_pred)}")  # 0.98

# ✅ 看完整分类报告 + 混淆矩阵
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# 召回率 = 0 → 违约样本全部漏掉
```

### 错误 2：数据泄露——先归一化再划分

**现象：** 模型在验证集上表现极好，但上线后效果远低于预期。

**原因：** 在整个数据集上计算了均值和标准差用于归一化——测试集的信息已经"泄漏"到训练数据中了。

**修复：**

```python
# ❌ 数据泄露：测试集参与拟合了 scaler
scaler = StandardScaler().fit(X)  # 包含了测试集
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled)

# ✅ 正确：只在训练集拟合，转换全部数据集
X_train, X_test = train_test_split(X)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # 用训练集的统计量转换
```

### 错误 3：不等模型稳定就选最高的

**现象：** 模型 A 比模型 B 高 0.3% 的准确率，选 A 上线后发现并不更好。

**原因：** 在单次划分上的 0.3% 差异可能完全是随机波动。

**修复：**

```python
# ✅ 用交叉验证 + 统计检验
scores_a = cross_val_score(model_a, X, y, cv=5)
scores_b = cross_val_score(model_b, X, y, cv=5)
t_stat, p_val = scipy.stats.ttest_rel(scores_a, scores_b)
if p_val < 0.05:
    print("差异显著，选胜率高的模型")
else:
    print("差异不显著，选更简单的模型（奥卡姆剃刀）")
```

### 错误 4：在时间序列上做随机 K 折

**现象：** 用过去 30 天预测股价，交叉验证效果很好，实际交易亏损。

**原因：** 随机 K 折意味着模型在训练时可能看到了未来的数据（时间点 t+10 被纳入训练集，t+5 在验证集）——这是最隐蔽的数据泄露之一。

**修复：**

```python
# ✅ 时间序列必须用前向链（前 k 折训练，第 k+1 折验证）
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    assert max(train_idx) < min(val_idx)  # 保证训练集在验证集之前
```

---

## 8. 面试考点

### Q1：精确率和召回率有什么区别？什么场景下优先哪个？（难度：⭐⭐）

**参考答案：**
精确率回答"模型说'是'的里面有多少是真的"，召回率回答"所有真的里面被找出了多少"。垃圾邮件分类优先精确率（宁可不拦别误拦），疾病筛查优先召回率（宁可误诊别漏诊）。F1 是两者的调和平均，常用于类别不平衡的二分类评估。

### Q2：AUC-ROC = 0.5 意味着什么？AUC = 0.8 的模型一定有 80% 准确率吗？（难度：⭐⭐⭐）

**参考答案：**
AUC=0.5 意味着模型的排序能力等于随机——它把正样本排在负样本上面的概率是 50%。AUC=0.8 不是说准确率 80%；准确率取决于阈值选择。AUC 越高，存在某个阈值使得 TPR 高同时 FPR 低的可能性越大。高 AUC 但低准确率的情况通常发生在阈值选得不合适时。

### Q3：为什么 5 折交叉验证的 t 检验自由度是 4 而不是 N-1？（难度：⭐⭐⭐）

**参考答案：**
5 折交叉验证每折产生一个得分，两个模型各 5 个分数组成 5 对差值。配对 t 检验的自由度是 $K-1$（$K$ 是配对数），所以是 $5-1=4$，而不是样本量减 1。这就是为什么 5 折交叉验证做检验效力很低——临界值 2.78 对应需要均值差相对标准差足够大。工业界通常使用重复交叉验证来提高检验效力。

### Q4：实现一个函数，在不使用 sklearn 的情况下计算混淆矩阵和 F1。（难度：⭐⭐⭐）

**参考答案：**

```python
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def f1_score(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
```

### Q5：模型在测试集上准确率 95% 但生产环境表现很差。列出至少三个可能原因。（难度：⭐⭐⭐⭐）

**参考答案：**
1. **数据泄露**——训练时使用了测试集信息（特征工程包含目标关联、测试数据混入训练集）
2. **分布偏移**——生产环境数据与测试集分布不同（时间变化、用户群体变化）
3. **评估指标不当**——数据严重不平衡，准确率被多数类拉高
4. **测试集被多次使用**——根据测试集反复调参，实质上测试集变成验证集

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 混淆矩阵 | "统计对了几条" | 二分类的四格表（TP/TN/FP/FN），所有分类指标的源头 |
| 精确率 | "预测为正的那些准不准" | 模型的"宣称"中有多少是真相——关注预测质量，不关注遗漏 |
| 召回率 | "正样本被找出了多少" | 真相中有多少被找到——关注不遗漏，不关注误报 |
| F1 分数 | "精确和召回的平均" | 两者的调和平均——任一低则 F1 低，不能靠牺牲一个提升另一个 |
| ROC 曲线 | "模型好坏的图" | 阈值变化下 TPR 与 FPR 的权衡曲线，对角线对应随机 |
| AUC | "ROC 的值" | ROC 曲线下面积，等于"正样本得分高于负样本的概率" |
| K 折交叉验证 | "把数据折几下算平均" | K 次训练验证取均值，降低评估方差；K=5 或 10 最常用 |
| 分层抽样 | "让每折比例一样" | 保持类别比例与全集一致，分类任务的标准做法 |
| 配对 t 检验 | "看分数差显不显著" | 检验两模型在同一折上的差异是否超越随机噪声 |
| McNemar 检验 | "数两个模型谁错得多" | 仅看双方分歧样本的卡方检验，适用于单次测试集的比较 |
| 数据泄露 | "测试集混进训练了" | 训练过程间接使用了测试集信息——最常见的工程错误 |

---

## 📚 小结

混淆矩阵是分类评估的原子单位，精确率、召回率和 F1 是不同业务场景下的透镜，AUC 提供排序能力的无阈值衡量。交叉验证让单次评估变得稳定可靠，统计检验让模型比较从"看数字大小"升级到"看差异是否真实"。

下一课我们将进入深度学习核心领域——模型的评估逻辑完全延续本课，但面对神经网络时，偏差-方差诊断和学习曲线的使用将更加频繁。

---

## ✏️ 练习

1. 【理解】用自己的话解释"精确率和召回率不能同时提升"的含义。举一个实际场景说明为什么这不一定是个坏事。

2. 【实现】修改 `auc_roc` 函数，使其在输入包含完全相同的分数时仍能正确处理（即当 ROC 曲线出现"竖直/水平"段时）。添加一个测试用例验证。

3. 【实验】使用 scikit-learn 的 `fetch_20newsgroups`（子集="sci.med" 和 "sci.space"），训练一个朴素贝叶斯分类器。分别用 5 折、10 折、分层 5 折交叉验证评估准确率，记录各方案的均值和标准差，并解释差异原因。

4. 【思考】如果你的模型在测试集上 F1 是 0.8，而生产环境中只有 0.5。列出至少 4 个可能的原因，并说明你会如何验证每一个。

5. 【挑战】用 McNemar 检验的逻辑，推导当两个模型在测试集上完全一致（b=c=0）时，为什么不能得出"两模型等效"的结论——缺少了什么证据？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 模型评估完整实现 | `code/main.py` | 从混淆矩阵、ROC/AUC 到统计检验和全演示脚本 |
| 评估策略提示词 | `outputs/prompt-evaluation-strategy.md` | 在任何新数据集上启动模型评估的策略模板 |

---

## 📖 参考资料

1. [论文] Goodfellow, Bengio, Courville. "Deep Learning". MIT Press, 2016. 第 5 章：Machine Learning Basics. https://www.deeplearningbook.org
2. [官方文档] scikit-learn. "Model selection and evaluation". https://scikit-learn.org/stable/model_evaluation.html
3. [官方文档] scikit-learn. "Cross-validation". https://scikit-learn.org/stable/modules/cross_validation.html
4. [论文] Dietterich, T. G. "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms". Neural Computation, 1998. https://doi.org/10.1162/089976698300017197
5. [书籍] 周志华. 《机器学习》. 清华大学出版社, 2016. 第 2 章：模型评估与选择.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、知识连线、工程最佳实践、常见错误、面试考点等均为原创内容。
