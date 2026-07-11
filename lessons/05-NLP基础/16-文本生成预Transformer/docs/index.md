# N-gram 语言模型——Transformer 之前的文本生成

> 如果一个词是惊喜的，模型就是差的。困惑度把惊喜变成数字。平滑保证它不会变成无穷大。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 05 · 01（文本预处理）、阶段 02 · 14（朴素贝叶斯）
**预计时间：** ~45 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 08（RNN 文本建模）— N-gram 的固定窗口是被 RNN 的隐藏状态取代的第一个局限

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零构建 Trigram 语言模型——理解 `<s>`/`</s>` 边界标记和原始概率估计
- [ ] 实现从 Laplace 到 Kneser-Ney 的六种平滑方法——理解为什么 Kneser-Ney 用"接续概率"替代原始频率
- [ ] 计算困惑度（Perplexity）——解释为什么困惑度 100 意味着"模型每步从 100 个等可能选项中瞎猜"
- [ ] 用训练好的 N-gram 模型采样生成文本——理解为什么局部合理但全局不连贯

---

## 1. 问题

在 Transformer 之前，RNN 之前，词嵌入之前——语言模型预测下一个词的方法就是数数。数"the cat"后面跟"sat"出现了 47 次，跟"jumped"出现了 12 次，跟"refrigerator"出现了 0 次。归一化，得到概率分布。

这就是 N-gram 语言模型。它在 1980 到 2015 年间运行了每一个语音识别器、每一个拼写检查器、每一个基于短语的机器翻译系统。2026 年，当你需要在设备端以微秒级别做下一个词预测时——它仍然在运行。

**有趣的问题不是怎么数，而是怎么处理没见过的 N-gram。** 原始计数模型给任何没见过的事件分配零概率——这在真实文本上是灾难性的，因为句子很长，几乎所有长句都包含了至少一个从未在训练中出现的 N-gram 序列。平滑（Smoothing）的半个世纪研究修复了这一点。Kneser-Ney 平滑是结果——现代深度学习继承的是它的实证传统。

---

## 2. 概念

### 2.1 N-gram 概率

$$P(w_i \mid w_{i-n+1}, \dots, w_{i-1}) = \frac{\text{count}(w_{i-n+1}, \dots, w_i)}{\text{count}(w_{i-n+1}, \dots, w_{i-1})}$$

固定 `n`（通常三元=3，四元=4）。从计数中计算。**这是最大似然估计（MLE）的最简形式。**

### 2.2 零计数问题

N-gram 模型在训练中未见的任何序列上给出零概率。2007 年 Brown 语料库上的一项研究发现，即使是 4-gram 模型，留出数据中也有 30% 的 4-gram 在训练中从未出现。**不作平滑，你无法在任何真实文本上评估。**

### 2.3 平滑方法——从简单到精妙

1. **Laplace（加一）。** 每个计数加 1。简单。在罕见事件上表现极差——把太多概率质量给了从未发生的事
2. **Good-Turing。** 根据"频率的频率"将概率质量从高频事件重新分配到未见事件
3. **插值（Interpolation）。** 将 N-gram、(N-1)-gram、……的估计用可调权重组合。`P_interp = λ₁·P_trigram + λ₂·P_bigram + λ₃·P_unigram`
4. **回退（Backoff）。** 如果 N-gram 计数为零，退回到 (N-1)-gram。Katz 回退将其归一化——仅在 N-gram 计数为零时才回退，否则使用折扣后的 MLE
5. **绝对折扣（Absolute Discounting）。** 从所有计数中减去一个固定折扣 `D`，将释放的概率质量重新分配给未见事件
6. **Kneser-Ney。** 绝对折扣 + 低阶模型的巧妙选择：用**接续概率**（一个词出现在多少种不同的上下文中）替代原始频率

### 2.4 Kneser-Ney 的深层洞察

"San Francisco"是一个高频 bigram。Unigram"Francisco"大多数时候出现"San"之后。天真的绝对折扣给"Francisco"很高的 unigram 概率（因为计数高）。Kneser-Ney 注意到"Francisco"只出现在一个上下文中，降低了它的接续概率。结果：以"Francisco"结尾的新 bigram 获得与之相称的低概率。

**接续概率 = 这个词出现在了多少种不同的前驱词后面。** `cat` → 出现在"the cat"、"a cat"、"my cat"、"black cat"……多种上下文 = 高接续概率。"Francisco" → 几乎只出现在"San"后面 = 低接续概率。

### 2.5 评估——困惑度（Perplexity）

$$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum \log P(w_i \mid \text{context}_i)\right)$$

越低越好。困惑度 = 100 意味着模型每步和从 100 个等可能选项中瞎猜一样困惑。对于 Brown 语料库，一个调好的 4-gram KN 模型的困惑度约为 140。Transformer LM 在同一测试集上约为 15-30。差距大约是 10 倍——**这个差距是为什么整个领域转向了神经方法。**

---

## 3. 从零实现

### 第 1 步：Trigram 计数

```python
from collections import Counter

def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])    # 前 n-1 个词
            word = padded[i + n - 1]             # 第 n 个词
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts

def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分词后的句子列表。输出是 N-gram 计数和上下文计数。`<s>` 和 `</s>` 是句子边界标记——它们让模型学到"哪些词倾向于是句子的开头/结尾"。

### 第 2 步：Laplace 平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1          # 加一
    denominator = contexts.get(ctx, 0) + vocab_size       # 加 |V|
    return numerator / denominator
```

每个计数加 1。平滑但不聪明——给未见事件分配了过多的概率质量，同时伤害了稀有但已知的事件。

### 第 3 步：Kneser-Ney（Bigram，插值版）

```python
from collections import defaultdict

def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)  # 记录 w 出现在哪些前驱词之后

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams
        for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom            # 折扣后的已见概率
        lambda_prev = discount * len(unique_follow[prev]) / denom # 回退权重
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

**三个活动的部件。** `continuation_prob` 捕获了"这个词出现在多少种不同的上下文中"（Kneser-Ney 的创新）。`lambda_prev` 是折扣释放的质量，用于加权回退。最终概率 = 折扣后的主项 + 加权的接续项。

### 第 4 步：用采样生成文本

```python
import random

def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。不同 seed 总是给出不同输出。对于类似束搜索的输出——每步选 argmax（贪心）并加一个小的随机性调节（temperature）。

### 第 5 步：困惑度

```python
import math

def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于 Brown 语料库，一个调好的 4-gram KN 模型困惑度约 140。Transformer LM 在同一测试集上为 15-30。**这个 10 倍差距是为什么整个领域转向了神经方法。** 但 N-gram 模型 + KN 平滑在其语境中仍然是坚实的 baseline——如果你的 Transformer 不能以显著优势击败它，有什么地方出错了。

完整代码见 `code/ngram_lm.py`。

---

## 4. 工业工具

- **KenLM。** 生产级 N-gram 库。在需要低延迟的语音和 MT 系统中用作重打分器——仍在 2026 年使用。C++ 后端，Python 绑定，毫秒级的查询延迟
- **`nltk.lm`。** 教学用。有所有标准平滑方法的纯净实现
- **设备端自动补全。** 键盘中的三元模型。今天仍然在——因为延迟 < 1ms 且模型大小 < 5MB
- **Baseline。** 永远在宣布你的神经 LM 好之前先算一个 N-gram LM 困惑度。如果你的 Transformer 不能以显著优势击败 KN——有什么地方出错了

---

## 5. 知识连线

N-gram 语言模型是语言建模的进化起点。它的每一个设计局限都被后续架构逐一修复：

- **固定窗口（n=3 或 4）→ RNN/LSTM 的隐藏状态。** "the cat …（50 个词后）… sat"——跨越了任何 N-gram 能看到的范围。阶段 05 · 08 的 LSTM 用隐藏状态传递信息穿越时间步
- **计数稀疏 → 神经概率估计。** 神经 LM 通过分布式表示（词嵌入）自然地泛化到相似上下文——"the dog"的模式从"the cat"自动迁移
- **全局不连贯 → Transformer 自注意力。** N-gram 采样生成的文本局部合理但全局不连贯——因为"窗口外的一切"对模型不存在。Transformer 的自注意力让每个位置看到整个序列

---

## 6. 常见错误

### 错误 1：训练和测试用了不同的 tokenizer

**现象：** 困惑度计算出来比预期高 30-50%——不是模型的问题，是数据预处理的问题。

**原因：** 训练时分词用的是 A 方案（标点分离、小写化），测试时用了 B 方案（保留标点、保留大写）。同一个 N-gram 在两个方案中是不同的 token 序列——计数完全不匹配。

**修复：** 将分词函数封装为一个可复用的模块。训练和测试调用同一个引用。困惑度数字只有在完全相同的分词方案下才是可比的。

### 错误 2：用困惑度跨不同词表大小比较模型

**现象：** 模型 A 困惑度 100，模型 B 困惑度 80——"B 比 A 好 20%"。

**原因：** 困惑度对词表大小敏感。词表 10K 的模型和词表 50K 的模型——后者天然的困惑度更高，因为分母中有更多选项。这不是"更差"，是任务更难。

**修复：** 困惑度只能在相同词表上比较。跨词表比较时——使用 bits-per-character (BPC) 或 bits-per-word 作为替代指标。

---

## 7. 面试考点

### Q1：为什么 Kneser-Ney 是 N-gram 平滑的终点？（难度：⭐⭐）

**参考答案：**
因为它解决了之前所有平滑方法的结构性缺陷——**低阶模型的概率应该基于"接续多样性"而非"原始频率"。** "Francisco"高频但只出现在"San"后面——传统平滑会给它高 unigram 概率，导致任何以"Francisco"结尾的新 bigram 被高估。Kneser-Ney 用"出现在多少种不同前驱词后面"重新定义了低阶概率——这不是对前五种方法的增量改进，是一个概念上不同的度量。

### Q2：困惑度 100 意味着什么？（难度：⭐）

**参考答案：**
意味着模型在测试集的每个位置上，相当于从 100 个等可能选项中随机猜测下一个词。这是"均匀分布"的困惑度——一个真正好的模型应该远低于这个数（GPT-3 在 WikiText 上的困惑度约为 20）。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| N-gram | "N 个词一组的序列" | 连续的 N 个词元。三元=3，四元=4 |
| 平滑 | "避免零概率" | 重新分配概率质量——让未见事件获得非零概率 |
| 困惑度 | "语言模型的分数" | exp(-平均 log P)。越低越好。100=每步瞎猜，20=相当好 |
| 回退 | "看不到了就退一步" | 三元没看到 → 退回到二元。Katz 回退将其形式化 |
| Kneser-Ney | "最好的平滑" | 绝对折扣 + 接续概率（低阶模型不看频率看"接续了几种不同上下文"） |
| 接续概率 | "KN 独有" | 以词 w 出现在多少种不同上下文中的比例作为 P(w)——不是以总出现次数 |

---

## 📚 小结

N-gram 语言模型用极简的计数+平滑支撑了 NLP 三十年的底层基础设施。Kneser-Ney 平滑是这一传统的顶峰——用"出现在多少种不同上下文中"替换了"出现了多少次"。困惑度 10 倍的差距（140 → 15）是推动整个领域从计数转向神经方法的量化动力。

但 N-gram LM + KN 平滑仍然是 neural LM 最诚实的 baseline——如果你的 Transformer 不能以显著优势击败它，有什么地方出错了。

---

## ✏️ 练习

1. 【理解】在 1000 句莎士比亚语料库上训练三元 LM。生成 20 句话。它们将局部合理但全局不连贯——这是 N-gram 模型的经典演示。

2. 【实现】在 Shakespeare 留出拆分上为你 KN 模型计算困惑度。与 Laplace 对比。KN 的困惑度应该低 30-50%。

3. 【实验】构建一个三元拼写校正器：给定一个拼写错误的词和它的上下文，生成候选修正并按上下文概率排序。在 Birkbeck 拼写语料库（公开）上评估。

4. 【思考】你的 Transformer 在测试集上困惑度 18。同测试集的 4-gram KN 困惑度 140。差距很大——但 KN 的推理延迟 < 0.1ms。在什么场景下你仍然会选择 KN？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| N-gram LM + KN 从零实现 | `code/ngram_lm.py` | Laplace + Kneser-Ney + 困惑度 + 采样生成 |

---

## 📖 参考资料

1. [教材] Jurafsky and Martin. 《Speech and Language Processing》第 3 章（2026 草稿）. https://web.stanford.edu/~jurafsky/slp3/3.pdf — N-gram LM 和平滑的权威处理
2. [论文] Chen and Goodman. "An Empirical Study of Smoothing Techniques for Language Modeling". 1998. https://dash.harvard.edu/handle/1/25104739 — 确立了 Kneser-Ney 为最佳 N-gram 平滑方法的论文
3. [论文] Kneser and Ney. "Improved Backing-off for M-gram Language Modeling". ICASSP, 1995. https://ieeexplore.ieee.org/document/479394 — 原始 KN 论文
4. [工具] KenLM. https://kheafield.com/code/kenlm/ — 快速生产 N-gram LM，2026 年仍在延迟敏感应用中使用

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
