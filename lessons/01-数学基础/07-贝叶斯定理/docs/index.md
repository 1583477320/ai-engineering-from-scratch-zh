# 贝叶斯定理——概率是预期，贝叶斯是学习

> 概率是关于你预期什么。贝叶斯定理是关于你学到了什么。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第 01 阶段 · 06（概率与分布）
**预计时间：** 75 分钟
**所处阶段：** Tier 1
**关联课程：** 第 08 阶段 · 06（强化学习）— 贝叶斯推理是 Thompson 采样的基础

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 应用贝叶斯定理从先验和似然计算后验概率
- [ ] 从零构建带拉普拉斯平滑和对数空间计算的朴素贝叶斯文本分类器
- [ ] 比较 MLE 和 MAP 估计，解释 MAP 对应 L2 正则化
- [ ] 实现 Beta-Binomial 共轭先验的序列贝叶斯更新

---

## 1. 问题

医学检测准确率 99%。你检测阳性。实际患病概率是多少？

大多数人说 99%。真实答案取决于疾病多罕见——1/10,000 的患病率下，阳性结果只给出约 1% 的患病概率。其余 99% 的阳性结果是健康人的假警报。

这不是陷阱问题，这是贝叶斯定理。每个垃圾邮件过滤器、医疗诊断、量化不确定性的机器学习模型都使用这个推理。

---

## 2. 核心概念

### 2.1 贝叶斯定理

```
P(A|B) = P(B|A) × P(A) / P(B)

后验 = 似然 × 先验 / 证据
```

| 部分 | 名称 | 含义 |
|:-----|:-----|:-----|
| P(A\|B) | 后验 | 看到证据 B 后对 A 的更新信念 |
| P(B\|A) | 似然 | 如果 A 为真，证据 B 的概率 |
| P(A) | 先验 | 看到证据前对 A 的信念 |
| P(B) | 证据 | 所有可能性下看到 B 的总概率 |

### 2.2 医学检测例子

```
患病率 P(病) = 0.0001
检测灵敏度 P(阳性|病) = 0.99
假阳性率 P(阳性|健康) = 0.01

P(阳性) = 0.99 × 0.0001 + 0.01 × 0.9999 = 0.010098
P(病|阳性) = 0.99 × 0.0001 / 0.010098 = 0.0098 ≈ 1%
```

先验占主导——罕见疾病的准确检测仍产生大量假阳性。

### 2.3 朴素贝叶斯分类器

假设所有特征在给定类别条件下独立：

```
P(class|features) ∝ P(class) × ∏ P(feature_i|class)
```

"朴素"的部分就是独立性假设——在文本中词出现不独立，但分类器只需排序类别，不需要精确概率。

用对数空间避免下溢：乘法→加法。

### 2.4 MAP vs MLE

| 估计 | 优化目标 | ML 等价 |
|:-----|:---------|:--------|
| MLE | P(data\|params) | 无正则化训练 |
| MAP | P(data\|params) × P(params) | L2 正则化 |

高斯先验 = L2 正则化。拉普拉斯先验 = L1 正则化。

---

## 3. 从零实现

```python
"""朴素贝叶斯分类器——从零实现。"""
import math
from collections import defaultdict


class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab = set()

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            for word in doc.lower().split():
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def predict(self, document):
        words = document.lower().split()
        total = sum(self.class_counts.values())
        vocab_size = len(self.vocab)
        best_cls, best_score = None, float("-inf")
        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total)
            for word in words:
                count = self.word_counts[cls].get(word, 0)
                total_words = self.class_word_totals[cls]
                score += math.log((count + self.smoothing) / (total_words + self.smoothing * vocab_size))
            if score > best_score:
                best_score, best_cls = score, cls
        return best_cls


def main():
    print("=== 朴素贝叶斯分类器 ===")
    train_docs = [
        "win free money now", "free lottery ticket", "claim your prize free",
        "meeting tomorrow at noon", "project update attached", "can we schedule a call",
        "team standup notes", "please review the pull request",
    ]
    labels = ["spam", "spam", "spam", "ham", "ham", "ham", "ham", "ham"]

    clf = NaiveBayes()
    clf.train(train_docs, labels)

    for msg in ["free money waiting", "meeting on friday", "you won a prize"]:
        print(f"  '{msg}' → {clf.predict(msg)}")

    print(f"\n词汇表大小: {len(clf.vocab)}")
    print("顶层垃圾邮件词:", sorted(clf.word_counts["spam"].items(), key=lambda x: -x[1])[:3])
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 用途 |
|:-----|:-----|
| Scikit-learn `MultinomialNB` | 生产级朴素贝叶斯 |
| PyMC | 贝叶斯建模 |
| PyMC3/NumPyro | 概率编程 |

---

## 5. 知识连线

- **第 08 阶段 · 06（强化学习）**：Thompson 采样使用贝叶斯推理选择动作
- **第 19 阶段 · 73（校准）**：模型置信度校准是贝叶斯思想的直接应用

---

## 6. 工程最佳实践

- **拉普拉斯平滑防止零概率**：`+1` 到每个计数确保未见词不会杀死整个乘积
- **对数空间计算**：避免概率相乘的下溢
- **MAP 等价 L2 正则化**：每个正则化项都是一个贝叶斯陈述

---

## 7. 常见错误

### 错误 1：基础率谬误

**现象：** 检测 99% 准确认为阳性后患病概率 99%。

**原因：** 忽略了先验（疾病罕见性）。

**修复：** 始终应用贝叶斯定理——后验取决于先验。

### 错误 2：朴素独立性假设失效

**现象：** 朴素贝叶斯在某些分类任务上表现差。

**原因：** 词与词不独立（"New" 和 "York" 相关）。

**修复：** 分类器只需排序类别，不需要精确概率——朴素假设在实践中仍然有效。

---

## 8. 面试考点

### Q1：为什么朴素贝叶斯在独立性假设不成立时仍然有效？（难度：⭐⭐）

**参考答案：** 分类器只需要正确排序类别的后验概率，不需要精确值。即使独立性假设错误导致概率偏差，只要排序一致，分类决策就正确。实践中朴素贝叶斯在文本分类上出人意料地强大。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| 先验 | 看到证据前对假设的信念 |
| 似然 | 如果假设为真，证据的概率 |
| 后验 | 看到证据后更新的信念 |
| 朴素贝叶斯 | 假设特征条件独立的贝叶斯分类器 |
| 拉普拉斯平滑 | 加 1 防止零概率 |
| MAP | 最大后验估计 = MLE + 先验 = 正则化 |

---

## 📚 小结

贝叶斯定理是关于学习的数学。你实现了朴素贝叶斯分类器，理解了 MAP 对应 L2 正则化，学会了用 Beta-Binomial 进行序列更新。这些概念连接了概率与优化。

---

## ✏️ 练习

1. 【实验】患者两次阳性检测后，计算后验概率
2. 【实现】调整朴素贝叶斯的平滑参数，观察效果
3. 【理解】用 Beta-Binomial 模拟 A/B 测试

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|:-----|:-----|:-----|
| 朴素贝叶斯 | `code/naive_bayes.py` | 完整分类器+平滑+对数空间 |

---

## 📖 参考资料

1. [视频] 3Blue1Brown 贝叶斯定理. https://www.youtube.com/watch?v=HZGCoVF3YvM
2. [书籍] Think Bayes（免费）. https://greenteapress.com/wp/think-bayes/
