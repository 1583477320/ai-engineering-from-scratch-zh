# N-gram 语言模型——Transformer 之前的文本生成

> 如果一个词是惊喜的，模型就是差的。困惑度把惊喜变成了一个数字。平滑保证它不会变成无穷大。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05 · 01、阶段 02 · 14 | **预计时间：** ~45 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

- [ ] 从零实现 Bigram 语言模型——Laplace 平滑 vs Kneser-Ney 平滑，比较困惑度
- [ ] 理解 Kneser-Ney 的核心洞察——用"接续概率"替代原始频率，"Francisco"虽然高频但几乎只接在"San"后面

---

## 1. 问题

在 Transformer 之前，RNN 之前，词嵌入之前——语言模型预测下一个词的方法就是数数。数"the cat"后面跟"sat"出现了 47 次，跟"jumped"出现了 12 次，跟"refrigerator"出现了 0 次。归一化，得到概率分布。

这就是 N-gram 语言模型。它在 1980 到 2015 年间运行了每一个语音识别器、每一个拼写检查器、每一个基于短语的机器翻译系统。2026 年，当你需要在设备端以微秒级别判断一个词序列是否合理时——它仍然在运行。

---

## 2. 概念

### Laplace 平滑——给每个没见过的事件加 1

P(w | prev) = (count(prev,w) + 1) / (count(prev) + |V|)

简单、直觉、在小词表上能用。但把所有未见事件的概率平均分配——"cat → refrigerator"和"cat → quantum"得到相同的概率。信息量最高的区分被抹平了。

### Kneser-Ney 平滑——统计的不是频率，是"出现在多少不同的上下文中"

"San Francisco"是高频 bigram。"Francisco"在语料中出现了很多次——Laplace 会给"Francisco"高 unigram 概率。但 Kneser-Ney 注意到"Francisco"几乎只出现在"San"之后（只有一个上下文）。它用**接续概率**（continuation probability）替代了原始 unigram 频率——"Francisco 出现在了多少个不同的前驱词后面？"答案是 1。因此给"Francisco"的低阶概率远低于 Laplace 的估计。

### 困惑度（Perplexity）

Perplexity = exp(-平均 log P)。越低 = 模型越好。Perplexity=10 意味着"每步相当于从 10 个等概率的选项中选一个"。

---

## 3. 从零实现

```python
def kneser_ney_prob(prev, w, discount=0.75):
    count = bigrams.get((prev, w), 0)
    # 第一项：折扣后的已见概率
    first = max(count - discount, 0) / context_count(prev)
    # 第二项：回退权重 × 接续概率
    lam = discount * unique_follow_count(prev) / context_count(prev)
    continuation = num_contexts(w) / total_unique_bigrams
    return first + lam * continuation
```

完整代码（含采样生成）见 `code/ngram_lm.py`。

---

## 4. 从 N-gram 到 LLM 的演化

| 时代 | 模型 | 核心原理 |
|---|---|---|
| 1980s | N-gram + Kneser-Ney | 数数 + 平滑——固定窗口，无法捕捉长距离 |
| 2010s | RNN/LSTM | 隐藏状态传递——能学习长距离但训练串行 |
| 2017+ | Transformer | 自注意力——O(1) 距离的全序列交互 |
| 2020+ | GPT/LLM | 堆叠 Transformer + 大规模预训练——N-gram 的计数精神被 100B+ 参数的统计所继承 |

---

## 🔑 关键术语

| 术语 | 实际含义 |
|---|---|
| 困惑度 (Perplexity) | 模型对测试数据的"平均惊讶程度"。越低越好。GPT-3 在 WikiText 上 ~20，Bigram 模型 ~200 |
| Kneser-Ney 平滑 | "一个词接续了多少种不同的上下文"——N-gram 时代最精妙的平滑方法 |
| 回退 (Backoff) | N-gram 没看到 → 退回 N-1 gram。Katz/Kneser-Ney 的不同在于退回时用什么概率 |

---

## 📚 小结 | ✏️ 练习

N-gram 语言模型用极简的计数+平滑支撑了 NLP 三十年的基础。Kneser-Ney 的核心洞察——用"出现在多少不同上下文中"替换"出现了多少次"——是现代语言模型评价体系的直系前身。练习：在 10 万字的英文小说上训练 bigram+trigram 模型，比较 Laplace/Kneser-Ney 的困惑度差异和采样质量。

---

> 本课程参考了 AI Engineering From Scratch 的课程体系，在此基础上进行了重构和原创内容的扩充。
