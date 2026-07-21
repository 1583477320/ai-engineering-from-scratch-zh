# 概率与分布——AI 表达不确定性的语言

> 概率是 AI 用来表达不确定性的语言。

**类型：** 概念课
**编程语言：** Python
**前置知识：** 第 01 阶段 · 01-04
**预计时间：** 75 分钟
**所处阶段：** Tier 1
**关联课程：** 第 01 阶段 · 07（贝叶斯定理）— 本节的概率基础是贝叶斯推理的前导

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现伯努利、分类、泊松、均匀和正态分布的 PMF/PDF
- [ ] 计算期望值和方差，用中心极限定理解释为什么高斯分布无处不在
- [ ] 构建 softmax 和 log-softmax 函数（含数值稳定性技巧）
- [ ] 计算交叉熵损失并连接到负对数似然

---

## 1. 问题

分类器输出 `[0.03, 0.91, 0.06]`。语言模型从 50,000 个候选词中选下一个词。扩散模型通过采样生成图像。这些都是概率在行动。

模型的每个预测都是概率分布。每个损失函数衡量预测分布与真实分布的距离。每个训练步骤调整参数使一个分布更像另一个。

---

## 2. 核心概念

### 2.1 关键分布

| 分布 | 形式 | AI 应用 |
|:-----|:-----|:--------|
| 伯努利 | P(1)=p, P(0)=1-p | 二分类 |
| 分类 | P(i)=pᵢ, Σpᵢ=1 | 多分类（softmax 输出） |
| 正态 | f(x)=exp(-(x-μ)²/2σ²)/√(2πσ²) | 权重初始化、梯度噪声 |
| 均匀 | f(x)=1/(b-a) | 随机初始化 |

### 2.2 中心极限定理

多个独立随机变量的均值趋近正态分布——无论原始分布是什么。

```
掷 1 骰子：均匀分布（平坦）
2 个骰子均值：三角形（有峰）
30 个骰子均值：几乎完美的钟形曲线
```

### 2.3 对数概率

```
P(句子) = P(w₁) × P(w₂) × ... × P(wₙ)  → 溢出到 0
log P(句子) = log P(w₁) + log P(w₂) + ...   → 有限数
```

乘法→加法，避免下溢。

### 2.4 Softmax 与交叉熵

```python
softmax(z_i) = exp(z_i) / Σexp(z_j)
```

数值稳定技巧：先减去最大 logit，再 exp。交叉熵 = 负对数概率。

---

## 3. 从零实现

```python
"""概率与分布——PMF/PDF+期望+softmax+交叉熵。"""
import math, random

def factorial(n):
    r = 1
    for i in range(2, n + 1): r *= i
    return r

def bernoulli_pmf(k, p): return p if k == 1 else 1 - p
def poisson_pmf(k, lam): return (lam ** k) * math.exp(-lam) / factorial(k)
def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def softmax(logits):
    mx = max(logits)
    shifted = [z - mx for z in logits]
    exps = [math.exp(z) for z in shifted]
    s = sum(exps)
    return [e / s for e in exps]

def log_softmax(logits):
    mx = max(logits)
    lse = mx + math.log(sum(math.exp(z - mx) for z in logits))
    return [z - lse for z in logits]

def cross_entropy(logits, target_idx):
    return -log_softmax(logits)[target_idx]

def expected_value(vals, probs):
    return sum(v * p for v, p in zip(vals, probs))

def variance(vals, probs):
    mu = expected_value(vals, probs)
    return sum(p * (v - mu) ** 2 for v, p in zip(vals, probs))


def main():
    print("=== 期望与方差 ===")
    die_vals = [1, 2, 3, 4, 5, 6]
    die_probs = [1/6] * 6
    print(f"骰子: E[X]={expected_value(die_vals, die_probs):.4f} Var={variance(die_vals, die_probs):.4f}")

    print("\n=== Softmax ===")
    logits = [2.0, 1.0, 0.1]
    probs = softmax(logits)
    print(f"logits: {logits}")
    print(f"softmax: {[f'{p:.4f}' for p in probs]}, sum={sum(probs):.4f}")

    print("\n=== 交叉熵 ===")
    ce = cross_entropy([2.0, 1.0, 0.1], 0)
    print(f"CE(logits=[2,1,0.1], target=0) = {ce:.4f}")

    print("\n=== 正态分布采样 (Box-Muller) ===")
    samples = []
    for _ in range(10000):
        u1, u2 = random.random(), random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(z)
    mu = sum(samples) / len(samples)
    var = sum((s - mu) ** 2 for s in samples) / len(samples)
    print(f"采样均值: {mu:.4f} (应接近 0), 方差: {var:.4f} (应接近 1)")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 用途 |
|:-----|:-----|
| PyTorch `F.cross_entropy` | 带数值稳定的交叉熵 |
| PyTorch `F.softmax` | 含温度参数的 softmax |
| SciPy `stats.norm` | 正态分布计算 |

---

## 5. 知识连线

- **第 01 阶段 · 07（贝叶斯定理）**：概率条件更新是贝叶斯推理的基础
- **第 03 阶段 · 06（损失函数）**：交叉熵是分类任务的标准损失
- **第 10 阶段（LLM 从零）**：困惑度 = exp(平均交叉熵)

---

## 6. 工程最佳实践

- **对数概率是唯一正确的方式**：乘法溢出是 LLM 中的常见 bug
- **softmax + 交叉熵组合**：用 `F.cross_entropy` 而非先 softmax 再 log
- **中文场景建议**：交叉熵损失在多语言评估中需要调整词表大小

---

## 7. 常见错误

### 错误 1：softmax 未减去最大值

**现象：** `math.exp(1000)` 溢出为 Inf。

**修复：** `exp(z - max(z))` — 相对值不变但避免溢出。

### 错误 2：对数概率忘记取负

**现象：** 困惑度 < 1（不可能值）。

**修复：** 交叉熵 = -log_softmax[target_index]。

---

## 8. 面试考点

### Q1：为什么 LLM 使用对数概率而非原始概率？（难度：⭐⭐）

**参考答案：** 语言模型计算 50,000+ 词元的概率分布。多个概率相乘快速下溢到 0（~30 项后浮点无法表示）。对数将乘法转为加法——`log(P₁×P₂×P₃) = log(P₁) + log(P₂) + log(P₃)`——避免下溢。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| 概率质量函数 (PMF) | 离散随机变量的精确概率 |
| 概率密度函数 (PDF) | 连续变量的密度——需积分得概率 |
| 期望值 | 概率加权的平均结果 |
| 中心极限定理 | 多个独立样本均值趋近正态分布 |
| Softmax | 将原始分数转为有效概率分布 |

---

## 📚 小结

概率是 AI 表达不确定性的语言。你实现了常见分布、softmax、交叉熵，并理解了对数概率防止下溢的原因。下一课学习贝叶斯定理。

---

## ✏️ 练习

1. 【实现】实现逆变换采样从指数分布生成样本
2. 【实验】计算 5 类分类器的交叉熵，验证 PyTorch 结果
3. 【理解】为什么正态分布在 AI 中无处不在？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|:-----|:-----|:-----|
| 概率工具 | `code/probability.py` | 分布、采样、softmax、交叉熵 |

---

## 📖 参考资料

1. [视频] 3Blue1Brown 中心极限定理. https://www.youtube.com/watch?v=zeJD6dqJ5lo
2. [论文] Stanford CS229 概率复习. https://cs229.stanford.edu/section/cs229-prob.pdf
