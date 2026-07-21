# 数值稳定性——浮点数是个漏水的抽象

> 浮点数是个漏水的抽象。它会在训练中咬你，而你毫无察觉。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第 01 阶段 · 01-04
**预计时间：** 120 分钟
**所处阶段：** Tier 1
**关联课程：** 第 03 阶段 · 05（反向传播）— 梯度检查是调试反向传播的金标准

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现数值稳定的 softmax 和 log-sum-exp（最大值减法技巧）
- [ ] 识别浮点计算中的溢出、下溢和灾难性抵消
- [ ] 用有限差分验证解析梯度
- [ ] 解释为什么训练优先使用 bfloat16 而非 float16

---

## 1. 问题

模型训练 3 小时后损失变 NaN。日志显示第 9000 步 logits 正常，第 9001 步变为 inf，第 9002 步所有梯度为 NaN，训练死亡。

或者：模型完成训练但准确率比论文低 2%。架构、超参、数据全部匹配。问题是你用了 float16 而论文用 float32——32 位累积舍入误差悄悄吞掉了你的准确率。

数值稳定性不是理论问题——它是训练成功与静默失败之间的差距。

---

## 2. 核心概念

### 2.1 IEEE 754 浮点格式

| 格式 | 位数 | 指数 | 尾数 | 约精确位数 | 范围 |
|:-----|:-----|:-----|:-----|:---------|:-----|
| float64 | 64 | 11 | 52 | ~15-16 | ±1.8e308 |
| float32 | 32 | 8 | 23 | ~7-8 | ±3.4e38 |
| float16 | 16 | 5 | 10 | ~3-4 | ±65,504 |
| bfloat16 | 16 | 8 | 7 | ~2-3 | ±3.4e38 |

### 2.2 为什么 0.1 + 0.2 ≠ 0.3

0.1 在二进制中是无限循环小数，float32 截断后存储为 ~0.100000001490116。修复：永远不要用 `==` 比较浮点数，用 `abs(a-b) < epsilon`。

### 2.3 灾难性抵消

减去两个几乎相等的浮点数导致有效数字抵消，相对误差可达 19%。在 ML 中发生在计算方差（`E[x²] - E[x]²`）时。

### 2.4 Log-Sum-Exp 技巧

```python
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

减去最大值后溢出消除——这是 softmax 数值稳定的基础。

### 2.5 混合精度训练

```
1. float32 主权重副本
2. 前向传播 fp16（快速）
3. 损失在 fp32（防溢出）
4. 反向传播 fp16
5. 损失缩放防止梯度下溢
6. 更新 fp32 主权重
```

---

## 3. 从零实现

```python
"""数值稳定性——稳定softmax+log-sum-exp+梯度检查。"""
import math

def softmax_stable(logits):
    mx = max(logits); exps = [math.exp(z-mx) for z in logits]
    s = sum(exps); return [e/s for e in exps]

def logsumexp_stable(values):
    c = max(values); return c + math.log(sum(math.exp(v-c) for v in values))

def cross_entropy_stable(true_class, logits):
    mx = max(logits); shifted = [z-mx for z in logits]
    lse = math.log(sum(math.exp(s) for s in shifted))
    return -(shifted[true_class] - lse)

def numerical_gradient(f, x, h=1e-5):
    return [(f([xi+(h if j==i else 0) for j,xi in enumerate(x)])-
             f([xi-(h if j==i else 0) for j,xi in enumerate(x)]))/(2*h) for i,x in enumerate(x)]

def check_gradient(analytical, numerical, tol=1e-5):
    ok = True
    for i,(a,n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a),abs(n),1e-8); err = abs(a-n)/denom
        if err > tol: ok = False
        print(f"  param{i}: a={a:.6f} n={n:.6f} err={err:.2e} {'OK' if err<tol else 'FAIL'}")
    return ok

def main():
    print("=== 稳定 softmax ===")
    print(f"安全: {softmax_stable([2.0,1.0,0.1])}")
    print(f"危险: {softmax_stable([100.0,101.0,102.0])}")

    print("\n=== log-sum-exp ===")
    print(f"稳定: {logsumexp_stable([500.0,501.0,502.0]):.6f}")

    print("\n=== 交叉熵 ===")
    print(f"稳定: {cross_entropy_stable(1,[2.0,5.0,1.0]):.6f}")

    print("\n=== 梯度检查 ===")
    f=lambda p: p[0]**2+3*p[0]*p[1]+p[1]**3
    g=lambda p: [2*p[0]+3*p[1], 3*p[0]+3*p[1]**2]
    ok = check_gradient(g([2.0,1.0]), numerical_gradient(f,[2.0,1.0]))
    print(f"梯度检查: {'通过' if ok else '失败'}")
    return 0
if __name__ == "__main__": import sys; sys.exit(main())
PYEOF

cat > "$BASE/13-数值稳定性/quiz.json" << 'QI'
[
  {"question":"Log-Sum-Exp 技巧为什么有效？","options":["减去最大值后 exp 最大值为 1，溢出消除","它使用更高精度计算","它跳过大值","它使用整数运算"],"correct":0,"explanation":"减去 max(x) 后，最大指数为 exp(0)=1，不可能溢出；至少一个项为 1，和至少为 1，不可能下溢到 -inf。","stage":"pre"},
  {"question":"为什么训练优先使用 bfloat16 而非 float16？","options":["bfloat16 更精确","bfloat16 有与 float32 相同的指数范围，训练中范围比精度更重要","bfloat16 速度更快","两者没有区别"],"correct":1,"explanation":"float16 最大值仅 65,504，训练中经常溢出。bfloat16 有 8 位指数（范围 ±3.4e38），梯度不需要损失缩放。","stage":"post"}
]
QI

# === Section 14: 范数与距离 ===
cat > "$BASE/14-范数与距离/docs/index.md" << 'MDEOF'
# 范数与距离——定义"相似"的度量

> 你的距离函数定义了什么是"相似"。选错了，下游一切都会崩。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第 01 阶段 · 01-02
**预计时间：** 90 分钟
**所处阶段：** Tier 1
**关联课程：** 第 19 阶段 · 65（混合检索）— BM25 与余弦相似度的融合

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 L1、L2、余弦、马氏距离、Jaccard 和编辑距离
- [ ] 为给定 ML 任务选择合适距离度量并解释替代方案为何失败
- [ ] 将 L1/L2 范数与 LASSO/Ridge 正则化的几何约束区域联系起来
- [ ] 演示同一数据集在不同度量下产生不同最近邻

---

## 1. 问题

你有两条向量。可能是词嵌入、用户画像或像素数组。你需要知道：它们有多近？

答案完全取决于你选哪个距离函数。两个数据点在一个度量下可能是最近邻，在另一个度量下可能很远。KNN、推荐系统、向量数据库、聚类算法、损失函数——全依赖这个选择。选错了，模型就优化错了目标。

---

## 2. 核心概念

### 2.1 六种距离

| 距离 | 公式 | 适用 |
|:-----|:-----|:-----|
| L1 | Σ\|xᵢ\| | 稀疏高维、异常值鲁棒 |
| L2 | √Σxᵢ² | 空间数据、图像像素 |
| 余弦 | 1-cos(θ) | NLP、嵌入、推荐 |
| 马氏 | √((x-y)^T S⁻¹ (x-y)) | 离群检测、相关特征 |
| Jaccard | 1-\|A∩B\|/\|A∪B\| | 集合重叠、标签 |
| 编辑距离 | 最少插入/删除/替换 | 拼写检查、DNA 序列 |

### 2.2 正则化与范数

```
L1 正则化 (Lasso): loss + λ‖w‖₁  → 稀疏权重（特征选择）
L2 正则化 (Ridge): loss + λ‖w‖₂² → 小权重（平滑解）
```

---

## 3. 从零实现

```python
"""距离度量——L1+L2+余弦+马氏+Jaccard+编辑距离。"""
import math, numpy as np

def l1_dist(a,b): return sum(abs(x-y) for x,y in zip(a,b))
def l2_dist(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def cosine_sim(a,b):
    dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x**2 for x in a)); nb=math.sqrt(sum(x**2 for x in b))
    return dot/(na*nb) if na*nb else 0
def cosine_dist(a,b): return 1-cosine_sim(a,b)

def jaccard_sim(a,b):
    sa,sb=set(a),set(b); return len(sa&sb)/max(len(sa|sb),1)

def edit_distance(a,b):
    n,m=len(a),len(b)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j]=dp[i-1][j-1] if a[i-1]==b[j-1] else 1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    return dp[n][m]

def main():
    a,b=(1,2,3),(4,0,6)
    print(f"L1={l1_dist(a,b)} L2={l2_dist(a,b):.4f}")
    print(f"余弦距离={cosine_dist(a,b):.4f}")
    print(f"Jaccard({{'cat','dog'}},{{'cat','bird','fish'}})={jaccard_sim(['cat','dog'],['cat','bird','fish']):.3f}")
    print(f"编辑距离('kitten','sitting')={edit_distance('kitten','sitting')}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
PYEOF

cat > "$BASE/14-范数与距离/quiz.json" << 'QI'
[
  {"question":"文本相似性最常用哪个距离？","options":["L2","余弦相似度","编辑距离","Jaccard"],"correct":1,"explanation":"余弦相似度忽略向量长度（文档长度），只关注方向（词分布），是 NLP 和嵌入的标准度量。","stage":"pre"},
  {"question":"L1 正则化为什么产生稀疏权重？","options":["L1 惩罚大值","L1 约束区域是菱形，顶点在轴上，损失等高线倾向与之相切","L1 使用整数运算","L1 只惩罚正权重"],"correct":1,"explanation":"L1 约束区域是菱形，损失函数的等高线最容易与菱形顶点相切——某些权重为零。","stage":"post"}
]
QI

# === Section 15: ML统计学 ===
cat > "$BASE/15-ML统计学/docs/index.md" << 'MDEOF'
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

完成本课后，你能够：

- [ ] 从零计算描述性统计、Pearson/Spearman 相关性和协方差矩阵
- [ ] 执行假设检验（t 检验、卡方检验）并正确解释 p 值和置信区间
- [ ] 使用 Bootstrap 重采样构建任意指标的置信区间
- [ ] 用效应量区分统计显著性和实际显著性

---

## 1. 问题

你训练了两个模型。A 得 0.87，B 得 0.89。你部署了 B。三周后生产指标更差。B 实际上并不优于 A——0.02 的差异是噪声。你的测试集太小、方差太高。

这在 ML 中反复发生：Kaggle 排名震荡、论文无法复现、A/B 测试基于几百个样本宣布赢家。根源总是同一个：跳过了统计学。

---

## 2. 核心概念

### 2.1 描述性统计

```
均值: μ = (1/n) × Σxᵢ（对异常值敏感）
中位数: 排序后中间值（对异常值鲁棒）
方差: σ² = (1/n) × Σ(xᵢ-μ)²
标准差: σ = √σ²（与数据同单位）
```

### 2.2 相关性

**Pearson**：线性关联，范围 [-1,1]。对异常值敏感。
**Spearman**：基于排序的单调关联。不受线性假设限制。

金律：相关不等于因果。

### 2.3 假设检验

```
零假设 H0: 无效果
p值: 假设 H0 为真时，观测到极端数据的概率
p < 0.05 → 拒绝 H0（统计显著）
p ≥ 0.05 → 未能拒绝 H0
```

**置信区间**：参数的合理值范围。"95% 置信"意味着重复实验 95% 的区间包含真值。

### 2.4 Bootstrap

用重采样估计统计量的抽样分布——无需分布假设。

### 2.5 效应量 vs 统计显著性

```
模型 A: 0.9234  模型 B: 0.9237  n=1,000,000
p 值 = 0.001  → 统计显著 ✓
效应量 d=0.003 → 实际无意义 ✗
```

始终同时报告 p 值和效应量。

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
def spearman(x,y):
    rank_x=[sorted(x).index(xi)+1 for xi in x]
    rank_y=[sorted(y).index(yi)+1 for yi in y]
    return pearson(rank_x,rank_y)

def bootstrap_ci(data, statistic, b=1000, alpha=0.05):
    vals=[statistic(random.choices(data,k=len(data))) for _ in range(b)]
    vals.sort(); lo=vals[int(b*alpha/2)]; hi=vals[int(b*(1-alpha/2))]
    return statistic(data),lo,hi

def main():
    x=[1,2,3,4,5,6,7,8,9,10]; y=[2,4,5,4,5,6,7,8,9,10]
    print(f"均值={mean(x):.1f} 标准差={std(x):.4f}")
    print(f"Pearson={pearson(x,y):.4f} Spearman={spearman(x,y):.4f}")
    m,lo,hi=bootstrap_ci(x,mean,500)
    print(f"Bootstrap 95% CI: [{lo:.2f}, {hi:.2f}] (均值={m:.2f})")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
PYEOF

cat > "$BASE/15-ML统计学/quiz.json" << 'QI'
[
  {"question":"p 值是什么？","options":["零假设为真的概率","假设 H0 为真时观测到极端数据的概率","模型正确的概率","结果可复现的概率"],"correct":1,"explanation":"p 值是假设零假设为真时，观测到当前或更极端数据的概率。它不是零假设为真的概率。","stage":"pre"},
  {"question":"为什么大样本时统计显著性不等于实际显著性？","options":["大样本总是提高 p 值","大样本使微小差异统计显著，但实际效果量可能无意义","大样本降低方差","大样本使所有差异不显著"],"correct":1,"explanation":"大样本使标准误极小，即使微小差异也能产生极低 p 值。但效应量（Cohen's d）衡量实际大小——0.03% 的改进不值得部署。","stage":"post"}
]
QI

# Output files for all sections
for d in "11-奇异值分解" "12-张量运算" "13-数值稳定性" "14-范数与距离" "15-ML统计学"; do
  dir=$(ls -d "$BASE/$d"*/ 2>/dev/null | head -1)
  [ -z "$dir" ] && continue
  touch "$dir/outputs/.placeholder"
done

echo "All files created"
