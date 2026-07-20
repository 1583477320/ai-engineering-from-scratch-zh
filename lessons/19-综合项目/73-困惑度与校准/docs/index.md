# 综合项目73——困惑度与校准（Perplexity and Calibration）

> 模型说 90% 置信度但只对了 60%，它没有校准好。校准是可信评估的一半。另一半是困惑度——告诉你模型是否认为留出文本合理。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第70-71节
**预计时间：** 90分钟

---

## 学习目标

- 计算留出语料上的困惑度（从模型提供的负对数概率）
- 计算预期校准误差（ECE）
- 计算 Brier 分数
- 构建可靠性图数据

---

## 1. 问题

困惑度=1 表示模型对每个实际词元赋予概率 1。困惑度=词表大小表示模型均匀分布，什么都没学到。强基线模型在 WikiText-103 上约 8-12，差的在 50+。

ECE 将预测按置信度分组，测量每组中置信度与准确率的平均差距。Brier 分数测量每预测的平方误差——惩罚分散的预测。

---

## 2. 核心概念

### 2.1 困惑度

```python
perplexity = exp(total_neg_log_prob / total_tokens)
```

适配器提供每词元负对数概率。框架聚合。

### 2.2 ECE（预期校准误差）

将 N 个预测按置信度分成 M 个桶。每桶计算平均置信度和平均准确率。差距 = |平均置信度 - 平均准确率|。ECE = 按桶大小加权的差距和。

### 2.3 Brier 分数

```python
brier = mean((p_i - y_i)^2)  # p=置信度, y=正确性(0或1)
```

分解为可靠性、分辨率和不确定性。

### 2.4 可靠性图数据

返回三个数组：每桶平均置信度、每桶平均准确率、每桶计数。用于绘制置信度-准确率曲线。

---

## 3. 从零实现

```python
"""困惑度与校准——ECE + Brier + 可靠性图。"""
import math


def perplexity(neg_log_probs, token_counts):
    total_nll = sum(neg_log_probs)
    total_tokens = sum(token_counts)
    if total_tokens == 0: return float("nan")
    return math.exp(total_nll / total_tokens)


def expected_calibration_error(confidences, correct, bins=10):
    if not confidences: return 0.0, 0
    bin_edges = [i / bins for i in range(bins + 1)]
    ece, populated = 0.0, 0
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [(p, y) for p, y in zip(confidences, correct) if lo <= p < hi or (i == bins - 1 and p == hi)]
        if not in_bin: continue
        avg_conf = sum(p for p, _ in in_bin) / len(in_bin)
        avg_acc = sum(y for _, y in in_bin) / len(in_bin)
        ece += len(in_bin) / len(confidences) * abs(avg_conf - avg_acc)
        populated += 1
    return ece, populated


def brier_score(confidences, correct):
    if not confidences: return 0.0
    return sum((p - y) ** 2 for p, y in zip(confidences, correct)) / len(confidences)


def reliability_diagram(confidences, correct, bins=10):
    bin_edges = [i / bins for i in range(bins + 1)]
    mean_conf, mean_acc, counts = [], [], []
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [(p, y) for p, y in zip(confidences, correct) if lo <= p < hi or (i == bins - 1 and p == hi)]
        if in_bin:
            mean_conf.append(sum(p for p, _ in in_bin) / len(in_bin))
            mean_acc.append(sum(y for _, y in in_bin) / len(in_bin))
            counts.append(len(in_bin))
        else:
            mean_conf.append(0); mean_acc.append(0); counts.append(0)
    return mean_conf, mean_acc, counts


def main():
    print("=== 困惑度与校准 ===\n")

    nlls = [0.1, 0.5, 0.3, 0.8, 0.2]
    counts = [5, 5, 5, 5, 5]
    ppl = perplexity(nlls, counts)
    print(f"困惑度: {ppl:.3f}")

    confs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9]
    corr = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    ece, populated = expected_calibration_error(confs, corr, 5)
    brier = brier_score(confs, corr)
    print(f"ECE: {ece:.3f} (填充桶: {populated})")
    print(f"Brier: {brier:.3f}")

    mc, ma, co = reliability_diagram(confs, corr, 5)
    print(f"可靠性图: 桶置信度={[f'{c:.2f}' for c in mc]} 准确率={[f'{a:.2f}' for a in ma]}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 指标 | 工具 | 特点 |
|:----|:-----|:-----|
| 困惑度 | HuggingFace `evaluate` | 标准化 |
| ECE | `scalabel.calibration` | 可靠性图 |
| Brier | 手动 | 简单直接 |

---

## 5. 工程最佳实践

- 困惑度和校准是全模型报告指标，不是逐任务指标
- ECE 受桶数和样本量影响——小数据集报告填充桶数
- **中文场景建议**：校准对中文多语言评估尤其重要——不同语言的置信度分布可能不同

---

## 6. 常见错误

- **负对数概率忘记取反**：返回 log p 而非 -log p 导致困惑度 < 1（不可能）
- **ECE 用 10 个桶但只有 10 个预测**：几乎每个桶只有 1 个样本，ECE 值无意义
- **混淆校准和准确率**：高准确率模型可能校准很差

---

## 7. 面试考点

**Q1：ECE 和 Brier 分数的互补关系是什么？**（难度：⭐⭐⭐）

**参考答案：** ECE 衡量按桶平均的校准差距——对过度自信和不足自信互相抵消的模型会给出低 ECE。Brier 分数测量每个预测的平方误差，惩罚分散的预测——即使平均校准好但局部偏差大，Brier 也会高。两者结合才能全面评估校准质量。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 困惑度 | exp(平均负对数概率)——越低越好 |
| ECE | 按桶平均的校准差距 |
| Brier | 每预测平方误差的均值 |
| 可靠性图 | 置信度 vs 准确率的桶图 |

---

## 📚 小结

困惑度衡量模型对留出文本的拟合度，校准衡量置信度的可靠性。你实现了 ECE、Brier 和可靠性图数据。下一节构建排行榜聚合。

---

## ✏️ 练习

1. 【实现】在完美校准数据（50%置信度=50%准确率）上验证 ECE=0
2. 【实验】对比过校准模型和欠校准模型的 Brier 分数

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 困惑度与校准 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Guo et al. "On Calibration of Modern Neural Networks". ICML 2017.
2. [论文] Brier. "Verification of Forecasts Expressed in Terms of Probability". 1950.
