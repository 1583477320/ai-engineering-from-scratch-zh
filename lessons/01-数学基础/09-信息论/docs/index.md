# 信息论——衡量惊讶程度的数学

> 信息论衡量惊讶。损失函数建立在它之上。

**类型：** 概念课
**编程语言：** Python
**前置知识：** 第 01 阶段 · 06（概率与分布）
**预计时间：** 60 分钟
**所处阶段：** Tier 1
**关联课程：** 第 08 阶段 · 06（强化学习）— KL 散度是策略优化的核心

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零计算熵、交叉熵和 KL 散度，解释它们的关系
- [ ] 推导最小化交叉熵损失等价于最大化对数似然
- [ ] 计算特征与目标之间的互信息来排序特征重要性
- [ ] 解释困惑度作为语言模型选择的有效词汇量

---

## 1. 问题

你在每个分类模型中调用 `CrossEntropyLoss()`。每个语言模型论文提到"困惑度"。VAE、蒸馏、RLHF 中读到 KL 散度。这些不是不相关的概念——它们是同一个思想的不同帽子。

Claude Shannon 在 1948 年发明信息论解决通信问题。训练神经网络也是一个通信问题：模型试图通过噪声权重信道传输正确的标签。

---

## 2. 核心概念

### 2.1 信息量（惊讶度）

```
I(x) = -log(p(x))
```

确定性事件信息量为 0。罕见事件信息量大。

### 2.2 熵（平均惊讶度）

```
H(P) = -Σ p(x) × log(p(x))
```

公平硬币：1 bit（最大熵）。偏置硬币（99% 正面）：0.08 bits。

### 2.3 交叉熵

```
H(P, Q) = -Σ p(x) × log(q(x))
```

用分布 Q 编码来自 P 的事件的平均代价。P 是标签，Q 是模型预测。

**分类简化：** P 是 one-hot → `H(P,Q) = -log(q(正确类别))`

### 2.4 KL 散度

```
D_KL(P||Q) = H(P,Q) - H(P)
```

交叉熵减去熵。训练中 H(P) 常数，所以最小化交叉熵 = 最小化 KL 散度。

**KL 不对称：** D_KL(P||Q) ≠ D_KL(Q||P)

### 2.5 困惑度

```
Perplexity = 2^H(P,Q)  (bits)
           = e^H(P,Q)   (nats)
```

模型困惑度 50 = 平均在 50 个候选词中犹豫。

---

## 3. 从零实现

```python
"""信息论——熵+交叉熵+KL散度+困惑度。"""
import math

def entropy(probs, base=2):
    return sum(-p * math.log(p) / math.log(base) for p in probs if p > 0)

def cross_entropy(p, q, base=2):
    return sum(-pi * math.log(qi) / math.log(base) for pi, qi in zip(p, q) if pi > 0)

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

def softmax(logits):
    mx = max(logits); exps = [math.exp(z - mx) for z in logits]
    s = sum(exps); return [e / s for e in exps]

def cross_entropy_loss(true_class, logits):
    return -math.log(softmax(logits)[true_class])


def main():
    fair_coin = [0.5, 0.5]; biased_coin = [0.99, 0.01]
    print(f"公平硬币熵: {entropy(fair_coin):.4f} bits")
    print(f"偏置硬币熵: {entropy(biased_coin):.4f} bits")

    true_dist = [0.7, 0.2, 0.1]; good = [0.6, 0.25, 0.15]; bad = [0.1, 0.1, 0.8]
    print(f"CE (好模型): {cross_entropy(true_dist, good):.4f} bits")
    print(f"CE (坏模型): {cross_entropy(true_dist, bad):.4f} bits")
    print(f"KL (好): {kl_divergence(true_dist, good):.4f} bits")
    print(f"KL (坏): {kl_divergence(true_dist, bad):.4f} bits")

    logits = [2.0, 1.0, 0.1]; probs = softmax(logits)
    loss = cross_entropy_loss(0, logits)
    print(f"\nlogits={logits} softmax={[f'{p:.4f}' for p in probs]}")
    print(f"交叉熵损失: {loss:.4f} nats, 困惑度: {math.exp(loss):.2f}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())

---

## 4. 工业工具

| 工具 | 用途 |
|:-----|:-----|
| PyTorch `nn.CrossEntropyLoss` | 带数值稳定的交叉熵 |
| PyTorch `nn.KLDivLoss` | KL 散度损失（VAE） |
| NumPy | 快速熵和互信息计算 |

---

## 5. 知识连线

- **第 08 阶段 · 06（强化学习）**：KL 散度在策略优化中约束策略偏移
- **第 10 阶段（LLM 从零）**：困惑度是语言模型的核心评估指标

---

## 6. 工程最佳实践

- **对数概率是唯一正确方式**：乘法下溢是 LLM 中的常见 bug
- **交叉熵 + softmax 组合**：用 `nn.CrossEntropyLoss` 而非先 softmax 再 log
- **标签平滑**：ε=0.1 防止模型过度自信，改善校准

---

## 7. 常见错误

- **softmax 未减去最大值**：`exp(1000)` 溢出为 Inf。修复：`exp(z - max(z))`。
- **对数概率忘记取负**：困惑度 < 1（不可能值）。修复：交叉熵 = -log_softmax[target]。

---

## 8. 面试考点

### Q1：为什么交叉熵是分类的标准损失？（难度：⭐⭐）

**参考答案：** 三个视角：(1) 信息论——最小化使用模型分布编码真实标签的浪费比特；(2) 最大似然——最小化交叉熵 = 最大化训练数据的似然；(3) 梯度——对 logits 的梯度就是 `(预测 - 真实)`，干净稳定。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| 熵 | 平均惊讶度——分布不可压缩的极限 |
| 交叉熵 | 用 Q 编码 P 事件的平均代价 |
| KL 散度 | 交叉熵减去熵——衡量分布距离 |
| 困惑度 | 交叉熵的指数——模型犹豫的有效选择数 |
| 对数概率 | 乘法转加法，避免下溢 |

---

## 📚 小结

信息论提供了理解损失函数的统一语言。你实现了熵、交叉熵、KL 散度，并理解了为什么交叉熵是分类的标准损失。下一课学习降维。

---

## ✏️ 练习

1. 【实验】计算英文字母表的熵（均匀 vs 实际频率）
2. 【实现】验证 KL 散度的不对称性
3. 【实现】构建序列困惑度函数

---

## 📖 参考资料

1. [论文] Shannon. "A Mathematical Theory of Communication". 1948.
2. [博客] Chris Olah 信息论可视化. https://colah.github.io/posts/2015-09-Visual-Information/
