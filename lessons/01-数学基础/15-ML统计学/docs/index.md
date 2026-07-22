# ML 统计学——如何知道模型真的有效还是碰巧了

> 统计学是判断模型真的有效还是碰巧了的工具。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第 01 阶段 · 06-07
**预计时间：** 120 分钟
**所处阶段：** Tier 1
**关联课程：** 第 19 阶段 · 74（排行榜聚合）— Bootstrap CI 是排行榜统计显著性的核心

---

## 🎯 学习目标

- [ ] 从零计算描述性统计、Pearson/Spearman 相关性
- [ ] 执行假设检验（t 检验、卡方检验）并正确解释 p 值
- [ ] 使用 Bootstrap 重采样构建置信区间
- [ ] 用效应量区分统计显著性和实际显著性

---

## 1. 问题

训练两个模型。A 得 0.87，B 得 0.89。部署 B。三周后生产指标更差。B 不优于 A——0.02 的差异是噪声。根源是跳过了统计学。

---

## 2. 核心概念

### 2.1 描述统计

均值：`μ = Σxᵢ/n`（对异常值敏感）
方差：`σ² = Σ(xᵢ-μ)²/n`
标准差：`σ = √σ²`

### 2.2 Pearson vs Spearman

- Pearson：线性关联，范围[-1,1]，对异常值敏感
- Spearman：基于排序的单调关联，非线性也能捕获

### 2.3 假设检验

p 值 = 假设 H0 为真时观测到极端数据的概率。p < 0.05 → 拒绝 H0。

### 2.4 Bootstrap

用重采样估计统计量分布——无需分布假设。B=1000 次有放回重采样。

### 2.5 效应量

Cohen's d = (mean₁-mean₂)/pooled_std。大 n 使微小差异统计显著但无实际意义。

---

## 3. 从零实现

```python
"""ML统计学——描述统计+相关+假设检验+Bootstrap。"""
import math, random

def mean(x): return sum(x)/len(x)
def variance(x):
    m=mean(x); return sum((xi-m)**2 for xi in x)/(len(x)-1) if len(x)>1 else 0
def std(x): return math.sqrt(variance(x))
def pearson(x,y):
    mx,my=mean(x),mean(y); sx,sy=std(x),std(y)
    return sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))/(len(x)*sx*sy) if sx*sy else 0

def bootstrap_ci(data, stat, b=1000, alpha=0.05):
    vals=[stat(random.choices(data,k=len(data))) for _ in range(b)]
    vals.sort()
    return stat(data), vals[int(b*alpha/2)], vals[int(b*(1-alpha/2))]

def main():
    x=[1,2,3,4,5,6,7,8,9,10]; y=[2,4,5,4,5,6,7,8,9,10]
    print(f"均值={mean(x):.1f} 标准差={std(x):.4f}")
    print(f"Pearson={pearson(x,y):.4f}")
    m,lo,hi=bootstrap_ci(x,mean,500)
    print(f"Bootstrap 95% CI: [{lo:.2f},{hi:.2f}]")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
PYEOF

cat > "$BASE/15-ML统计学/quiz.json" << 'QI'
[
  {"question":"p 值是什么？","options":["零假设为真的概率","假设 H0 为真时观测到极端数据的概率","模型正确的概率","结果可复现的概率"],"correct":1,"explanation":"p 值是假设零假设为真时观测到当前或更极端数据的概率。它不是零假设为真的概率。","stage":"pre"},
  {"question":"为什么大样本时统计显著性不等于实际显著性？","options":["大样本使微小差异统计显著，但效应量可能无意义","大样本降低方差","大样本使所有差异不显著","大样本总是提高 p 值"],"correct":0,"explanation":"大样本使标准误极小，即使微小差异也能产生低 p 值。但效应量(Cohen's d)衡量实际大小——0.03%的改进不值得部署。","stage":"post"}
]
QI

touch "$BASE/15-ML统计学/code/main.py"
cat > "$BASE/15-ML统计学/code/main.py" << 'PYEOF'
"""ML统计学——描述统计+相关+Bootstrap。"""
import math, random

def mean(x): return sum(x)/len(x)
def variance(x):
    m=mean(x); return sum((xi-m)**2 for xi in x)/(len(x)-1) if len(x)>1 else 0
def std(x): return math.sqrt(variance(x))
def pearson(x,y):
    mx,my=mean(x),mean(y); sx,sy=std(x),std(y)
    return sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))/(len(x)*sx*sy) if sx*sy else 0

def bootstrap_ci(data, stat, b=1000, alpha=0.05):
    vals=[stat(random.choices(data,k=len(data))) for _ in range(b)]
    vals.sort()
    return stat(data), vals[int(b*alpha/2)], vals[int(b*(1-alpha/2))]

def main():
    x=[1,2,3,4,5,6,7,8,9,10]; y=[2,4,5,4,5,6,7,8,9,10]
    print(f"均值={mean(x):.1f} 标准差={std(x):.4f}")
    print(f"Pearson={pearson(x,y):.4f}")
    m,lo,hi=bootstrap_ci(x,mean,500)
    print(f"Bootstrap 95% CI: [{lo:.2f},{hi:.2f}]")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
PYEOF

touch "$BASE/15-ML统计学/quiz.json"
cat > "$BASE/15-ML统计学/quiz.json" << 'QI'
[
  {"question":"p 值是什么？","options":["零假设为真的概率","假设 H0 为真时观测到极端数据的概率","模型正确的概率","结果可复现的概率"],"correct":1,"explanation":"p 值是假设零假设为真时观测到当前或更极端数据的概率。它不是零假设为真的概率。","stage":"pre"},
  {"question":"为什么大样本时统计显著性不等于实际显著性？","options":["大样本使微小差异统计显著，但效应量可能无意义","大样本降低方差","大样本使所有差异不显著","大样本总是提高 p 值"],"correct":0,"explanation":"大样本使标准误极小，即使微小差异也能产生低 p 值。但效应量衡量实际大小——0.03%的改进不值得部署。","stage":"post"}
]
QI

touch "$BASE/15-ML统计学/outputs/.placeholder"

echo "All files created"

---

## 4. 工业工具

| 工具 | 用途 |
|:-----|:-----|
| scipy.stats | 假设检验、分布计算 |
| statsmodels | 回归、置信区间 |
| scikit-learn | 分类评估指标（F1、AUC） |
| Pingouin | 现代统计学库，支持贝叶斯检验 |

---

## 5. 知识连线

- **第 08 阶段 · 06（强化学习）**：统计方法评估策略性能
- **第 19 阶段 · 74（排行榜聚合）**：Bootstrap CI 是排行榜统计显著性的核心
- **第 19 阶段 · 73（校准评估）**：模型校准需要统计检验验证

---

## 6. 工程最佳实践

- **始终报告置信区间**：不要只报均值，用 Bootstrap 估计不确定性
- **效应量必报**：p 值告诉差异是否真实，效应量告诉是否值得关心
- **交叉验证优先**：用配对 t 检验 + 交叉验证分数而非单次运行

---

## 7. 常见错误

### 错误 1：p 值误读为"零假设概率"

**现象：** 认为 p=0.05 意味着 95% 确信 H0 错误。

**修复：** p 值是"在 H0 为真的假设下观测到极端数据的概率"，不是"H0 为假的概率"。

### 错误 2：大样本时只报告 p 值

**现象：** 百万样本下微小差异 p<0.001，但实际无意义。

**修复：** 始终同时报告效应量（Cohen's d），区分统计显著与实际显著。

---

## 8. 面试考点

### Q1：Bootstrap 为什么比解析置信区间更好？（难度：⭐⭐）

**参考答案：** Bootstrap 无需分布假设（非参数），适用于任何统计量。解析 CI 假设正态分布——对于非对称分布（如 AUC、精确率）或小样本会给出错误区间。Bootstrap 直接从数据估计分布。

### Q2：多重比较问题如何处理？（难度：⭐⭐⭐）

**参考答案：** 测试越多，假阳性概率越高。Bonferroni 校正将 α 除以测试数（保守但简单）。Benjamini-Hochberg 控制错误发现率（FDR）在目标水平，更宽松。在 ML 中，测试多个超参配置时必须使用其中一种。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| 描述统计 | 均值、方差、标准差、分位数 |
| Pearson 相关 | 线性关联度，范围[-1,1] |
| Spearman 相关 | 基于排序的单调关联 |
| p 值 | 假设 H0 为真时观测到极端数据的概率 |
| 效应量 | 差异大小的实际意义（Cohen's d） |
| Bootstrap | 重采样估计统计量分布 |
| 多重比较 | 多次测试导致假阳性膨胀 |

---

## 📚 小结

ML 统计学帮你区分信号和噪声。你实现了描述统计、假设检验、Bootstrap 置信区间，理解了统计显著与实际显著的区别。这些是所有模型比较、A/B 测试和论文复现的基础。

---

## ✏️ 练习

1. 【实现】用 Bootstrap 估计模型准确率的 95% 置信区间
2. 【实现】计算两种分类器的 F1 分数差异的置信区间
3. 【实验】用不同样本量（10,100,1000,10000）演示大样本下微小差异变统计显著

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|:-----|:-----|:-----|
| 统计工具 | `code/statistics.py` | Bootstrap CI + 相关性 + 检验 |

---

## 📖 参考资料

1. [书籍] 统计学习方法. https://www.statlearning.com/
2. [官方文档] scipy.stats. https://docs.scipy.org/doc/scipy/reference/stats.html
3. [博客] 统计显著性的误用. https://xkcd.com/1478/
