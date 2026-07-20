# 综合项目71——经典指标（Classical Metrics）

> BLEU、ROUGE-L、F1、精确匹配、准确率——五个指标至今覆盖了大多数已发表的 LLM 评估数字。从零实现，理解每个数字的含义。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第70节
**预计时间：** 90分钟

---

## 学习目标

- 从零实现精确匹配、F1 和准确率
- 实现 BLEU-4：修正 n-gram 精度、几何平均、简短惩罚
- 实现 ROUGE-L：最长公共子序列 + F-beta 组合
- 按 `metric_name` 字段分发，运行器保持指标无关

---

## 1. 问题

你会读到报告 BLEU 28.3 的论文和报告 BLEU 0.283 的论文。会发现两个库的 ROUGE-L 分数差 10 分。最快的方法是自己实现——然后指向分词器决定的那行和使用的平滑方法那行。

标准库加 numpy 足够。BLEU 是计数+钳位。ROUGE-L 是动态规划。F1 是词元集合交集。最难的部分是选择分词器并承诺使用。

---

## 2. 核心概念

### 2.1 分词器契约

```python
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
def tokenize(text): return TOKEN_RE.findall(text.lower())
```

所有指标使用完全相同的分词器。运行器不能选择。换分词器 = 运行不同的基准测试。

### 2.2 精确匹配

```python
def exact_match(pred, targets):
    return float(any(pred.strip() == t.strip() for t in targets))
```

返回 1.0 或 0.0。数据集上的聚合是均值。

### 2.3 Token 级 F1

```
precision = 交集 / 预测词元数
recall = 交集 / 参考词元数
F1 = 2 × P × R / (P + R)
```

多目标时取最佳 F1。

### 2.4 BLEU-4

```
BLEU-4 = BP × exp(mean(log p1, log p2, log p3, log p4))
```

- 修正 n-gram 精度：候选计数被参考中最大计数钳位
- 简短惩罚：`BP = 1 if c>=r else exp(1 - r/c)`
- 平滑：分子分母各加 1 防止 log(0)

### 2.5 ROUGE-L

```python
def lcs_length(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            dp[i+1][j+1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i+1][j], dp[i][j+1])
    return dp[n][m]
```

F1 = `2 × P × R / (P + R)`，其中 P = lcs/候选长度，R = lcs/参考长度。

---

## 3. 从零实现

```python
"""经典指标——BLEU-4 + ROUGE-L + F1 + 精确匹配。"""
import re, math


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
def tokenize(text): return TOKEN_RE.findall(text.lower())


def exact_match(pred, targets):
    return float(any(pred.strip() == t.strip() for t in targets))


def accuracy(pred, targets):
    return float(pred.strip().lower() == targets[0].strip().lower()) if targets else 0.0


def f1_score(pred, target):
    p_tokens = tokenize(pred)
    t_tokens = tokenize(target)
    if not p_tokens and not t_tokens: return 1.0
    if not p_tokens or not t_tokens: return 0.0
    common = Counter(p_tokens) & Counter(t_tokens)
    inter = sum(common.values())
    prec = inter / len(p_tokens)
    rec = inter / len(t_tokens)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

from collections import Counter


def n_grams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens) - n + 1))]


def bleu4(pred, ref, smooth=True):
    weights = [0.25] * 4
    p_scores = []
    for n in range(1, 5):
        pred_ng = n_grams(tokenize(pred), n)
        ref_ng = n_grams(tokenize(ref), n)
        if not pred_ng:
            p_scores.append(1e-10 if smooth else 0)
            continue
        ref_counts = Counter(ref_ng)
        clipped = sum(min(pred_ng.count(ng), ref_counts.get(ng, 0)) for ng in set(pred_ng))
        p = clipped / len(pred_ng)
        if p == 0 and smooth: p = 1e-10
        p_scores.append(p)
    pred_len = len(tokenize(pred))
    ref_len = len(tokenize(ref))
    bp = min(1.0, math.exp(1 - ref_len / max(pred_len, 1))) if pred_len < ref_len else 1.0
    return bp * math.exp(sum(w * math.log(max(p, 1e-10)) for w, p in zip(weights, p_scores)))


def lcs_length(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            dp[i+1][j+1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i+1][j], dp[i][j+1])
    return dp[n][m]


def rouge_l(pred, target):
    p_tokens = tokenize(pred)
    t_tokens = tokenize(target)
    if not p_tokens or not t_tokens: return 0.0
    lcs = lcs_length(p_tokens, t_tokens)
    prec = lcs / len(p_tokens)
    rec = lcs / len(t_tokens)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def score(metric_name, pred, targets):
    if metric_name == "exact_match": return exact_match(pred, targets)
    if metric_name == "f1": return max(f1_score(pred, t) for t in targets)
    if metric_name == "bleu_4": return max(bleu4(pred, t) for t in targets)
    if metric_name == "rouge_l": return max(rouge_l(pred, t) for t in targets)
    if metric_name == "accuracy": return accuracy(pred, targets)
    raise ValueError(f"未知指标: {metric_name}")


def main():
    examples = [
        ("exact_match", "42", ["42", "四十二"]),
        ("exact_match", "41", ["42"]),
        ("f1", "cat on mat", ["the cat sat on the mat"]),
        ("bleu_4", "the cat sat on the mat", ["a cat is on the mat"]),
        ("rouge_l", "the cat sat on the mat", ["the cat is on the mat"]),
        ("accuracy", "B", ["B"]),
    ]
    for metric, pred, targets in examples:
        s = score(metric, pred, targets)
        print(f"  {metric}: pred='{pred[:30]}' targets={targets[:1]} → {s:.4f}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 指标 | 工具 | 特点 |
|:----|:-----|:-----|
| BLEU | `sacrebleu` | 标准化实现 |
| ROUGE | `rouge-score` | Google 官方 |
| F1 | 手动 | 简单直接 |
| BERTScore | `bertscore` | 基于嵌入 |

---

## 5. 工程最佳实践

- 分词器是契约，不是旋钮——所有指标共享同一个
- BLEU 和 ROUGE 对不同长度的输出偏差不同——混合使用
- **中文场景建议**：`\w+` 正则对中文按字符分割，适合大多数场景

---

## 6. 常见错误

- **BLEU 论文中 28.3 vs 0.283**：百分比 vs 比例。始终确认尺度
- **ROUGE-L 未用 F1**：只用召回率或精确率的 ROUGE-L 不可比
- **BLEU 零平滑**：缺少 4-gram 时 log(0) 导致整个分数为 0

---

## 7. 面试考点

**Q1：为什么 ROUGE-L 比 ROUGE-1 更适合摘要评估？**（难度：⭐⭐）

**参考答案：** ROUGE-1 只检查词元重叠（无序），ROUGE-L 基于最长公共子序列保留了词序。摘要的核心不只是包含关键词——还需要合理的顺序。ROUGE-L 惩罚顺序混乱的生成。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 精确匹配 | 预测与参考完全一致 |
| BLEU-4 | 1-4-gram 精度的几何平均 + 简短惩罚 |
| ROUGE-L | 基于 LCS 的 F1 |
| 修正 n-gram 精度 | 候选计数被参考最大计数钳位 |

---

## 📚 小结

五个经典指标构成了 LLM 评估的基线。你从零实现了它们，理解了每个数字背后的数学。下一节构建代码执行指标。

---

## ✏️ 练习

1. 【实现】添加 BLEU 的 tokenized 版本 vs 逐字符版本，比较差异
2. 【实验】在中英文混合文本上测试 tokenizer `\w+` 的行为

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 经典指标 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Papineni et al. "BLEU". ACL 2002.
2. [论文] Lin. "ROUGE". ACL 2004.
