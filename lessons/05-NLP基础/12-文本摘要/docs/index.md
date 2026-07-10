# 文本摘要——抽取式与生成式

> 抽取式告诉你文章说了什么。生成式告诉你作者想表达什么。不同的任务，不同的陷阱。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 05 · 02（BoW + TF-IDF）、阶段 05 · 11（机器翻译）
**预计时间：** ~75 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 11（机器翻译）— 同样是 Seq2Seq 变长输出，但评估指标从 BLEU 换成了 ROUGE

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 TextRank——理解句子相似度图 + PageRank 如何无监督地选出关键句
- [ ] 实现 ROUGE-N 评估——理解为什么摘要指标用召回率而非精确率
- [ ] 区分抽取式和生成式摘要的适用场景——事实准确性 vs 压缩灵活性的权衡

---

## 1. 问题

一篇 2000 字的新闻出现在你的信息流里。你需要 120 字来概括它。

你可以做两件事：从文章里挑出最重要的三句话（抽取式），或者用自己的话把内容重写一遍（生成式）。两者都叫"摘要"——但它们是完全不同的问题。

**抽取式摘要是排序问题。** 给每句话打分，返回 top-k。输出永远是原文原句——语法永远正确。风险是分布在文章各处、没有任何一句独立完整涵盖的关键信息会被遗漏。

**生成式摘要是生成问题。** Transformer 在输入条件上生成新文本。输出流利、可以任意压缩——但可能"幻觉"出原文中不存在的事实。风险是自信的编造。

本课两种都做——各自带着各自的失败模式。

---

## 2. 概念

### 2.1 TextRank——句子投票选摘要

```
文章 → 分句 → 句间相似度矩阵 → PageRank 迭代 → 排名 → 选 top-k → 恢复原文顺序
```

TextRank 的核心洞察：**如果一个句子和很多重要句子相似，它自己也很重要。** 这是 PageRank 在句图上的再现——"被高分句子指向的句子也是高分的"。

- 节点 = 句子。边 = 句子间的词汇重叠度（归一化后）
- 阻尼系数（0.85）= 随机跳到任意句子的概率——防止孤立的句子得分为 0
- 收敛或达到最大迭代次数 → 按分数取 top-k
- 输出按原文顺序排列——不是按分数排列

**完全无监督。** 不需要任何标注数据。这也是它的优点和天花板——它不知道什么内容是"用户关心的"，只知道什么内容是"在文章中被反复讨论的"。

### 2.2 ROUGE——为什么用召回率？

ROUGE = **R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation。翻译：评估摘要的**召回率**导向指标。

```
ROUGE-N = (摘要和参考共有的 n-gram 数) / (参考中的 n-gram 总数)
```

分母是**参考的总数**，而非摘要的总数。这意味着 ROUGE 问的是：**"参考中提到的关键内容，你的摘要覆盖了多少？"** 漏掉重要内容比多写了几个词更严重——这是摘要评估选择召回率作为导向的根本原因。

ROUGE-1（unigram）= 关键词覆盖。ROUGE-2（bigram）= 语序和搭配覆盖。ROUGE-L（LCS）= 句子结构相似度。三个一起报告——单个指标都无法全面反映质量。

### 2.3 抽取式 vs 生成式——两张表

| | 抽取式 | 生成式 |
|---|---|---|
| **本质** | 排序 + 选择 | Seq2Seq 生成 |
| **输出** | 原文句子（原封不动） | 新文本（可能含新词/新结构） |
| **语法** | 100% 正确 | 可能不通顺 |
| **事实准确性** | 100%（不编造） | 可能幻觉 |
| **压缩能力** | 低（句子级选取） | 高（可任意压缩） |
| **代表方法** | TextRank, LexRank, BertSumExt | BART, T5, Pegasus |
| **适用场景** | 法律/医疗/金融——零容错 | 对话/标题/自由格式 |

---

## 3. 从零实现

### 第 1 步：句子相似度

```python
def similarity(s1, s2):
    """词汇重叠度——分母用 log 抑制长句优势。"""
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    return intersection / denom if denom else 0.0
```

如果用词数作为分母，长句天然更"不相似"——因为词太多，交集无论如何都是小数。log 压制了这个偏差。

### 第 2 步：TextRank 核心

```python
def textrank(text, top_k=3, damping=0.85):
    sentences = sentence_split(text)
    n = len(sentences)
    sim = [[similarity(sentences[i], sentences[j]) if i != j else 0.0
            for j in range(n)] for i in range(n)]

    scores = [1.0] * n  # 所有句子初始等权
    for _ in range(50):  # PageRank 迭代
        new_scores = [1 - damping] * n  # 随机跳转分量
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if converged(scores, new_scores):
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    return [sentences[i] for i in sorted(ranked)]  # 恢复原文顺序
```

`damping=0.85` = 每一步有 15% 的概率随机跳转到任意句子——这是 PageRank 在"断连图"（某些句子和所有其他句子都不相似时）上的安全保护。

### 第 3 步：ROUGE-N

```python
def rouge_n(hypothesis, reference, n=1):
    hyp_ngrams = Counter(ngrams(hypothesis.split(), n))
    ref_ngrams = Counter(ngrams(reference.split(), n))
    overlap = sum((hyp_ngrams & ref_ngrams).values())
    return overlap / sum(ref_ngrams.values())  # 分母 = 参考总数 → 召回率
```

完整代码见 `code/summarize.py`。

---

## 4. 工业工具

### 4.1 HuggingFace——生成式摘要两行代码

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(article, max_length=130, min_length=30)
```

### 4.2 中文生成式摘要模型

| 模型 | 特点 |
|---|---|
| `fnlp/bart-base-chinese` | 中文 BART——生成式摘要的默认起点 |
| `csebuetnlp/mT5_multilingual_XLSum` | mT5，多语言包括中文 |
| CPT（Chinese Pre-trained Transformer） | 中文专用预训练，摘要+对话+分类 |

### 4.3 生产环境的 ROUGE

```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
scores = scorer.score(reference, hypothesis)
```

ROUGE-L 在 30-40 之间 = "可用的摘要"。> 50 = "优秀的摘要"。差值 < 2 ROUGE-L = 噪声。中文评估固定 jieba 版本——分词器的选择可以导致 5-10 个 ROUGE 点的差异。

---

## 5. 知识连线

文本摘要 = Seq2Seq（阶段 05 · 09）+ 注意力（阶段 05 · 10）的另一个应用。和机器翻译一样是变长输入→变长输出，但从 BLEU 换成了 ROUGE——因为摘要的目标是"覆盖关键信息"而非"精确匹配参考"。

---

## 6. 常见错误

### 错误 1：中文 TextRank 不先分词

直接用字级 n-gram 构建句间相似度 → "机器学习"和"机器学习算法"被判断为高度相似（6 个字重叠 4 个）→ TextRank 选出高度冗余的句子。

**修复：** 先用 jieba 分词，在词级构建相似度。"机器学习"是两个词→一个匹配。"机器学习算法"是三个词→同样只有"机器学习"这一部分匹配。

### 错误 2：ROUGE 报告时未固定分词器

中文 ROUGE 在不同分词器下差异可达 5-10 点。论文 A 用 jieba 默认词典 + ROUGE-1=45，论文 B 用 HanLP + ROUGE-1=40——数字不可比较。

**修复：** 报告 sacrebleu 风格的"评估签名"——记录分词器及其版本、词典配置。所有对比实验用完全相同的分词流水线。

---

## 7. 面试考点

### Q1：什么场景选抽取式，什么场景选生成式？（难度：⭐⭐）

**参考答案：**
事实准确性有刚性要求 → 抽取式（法律文书、医疗记录、金融研报——一个编造的数字不能被容忍）。压缩灵活性和语言流畅度优先 → 生成式（对话摘要、新闻标题、个性化推送——用户期望自然的语言）。两者不互斥——生产系统常常用抽取式做第一轮筛选，生成式做最终压缩。

### Q2：TextRank 的一个已知局限——它偏向长句吗？（难度：⭐⭐）

**参考答案：**
偏向。长句天然含有更多词汇，与更多句子有更高的词汇重叠度，PageRank 分数偏向于高的那一端。缓解手段：（1）在相似度分母中用 log 压制长句优势。（2）在选 top-k 后做冗余惩罚——如果两句高度相似，只保留得分最高的那句。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| TextRank | "句子版 PageRank" | 在句间相似度图上运行 PageRank——无监督选出最'中心'的句子 |
| ROUGE | "摘要的 BLEU" | 召回率导向的 n-gram 覆盖评估。ROUGE-1/2/L 分别衡量关键词/语序/结构 |
| 抽取式 vs 生成式 | "两种摘要" | 一个从原文选句子（零容错），一个生成新文本（可能的幻觉）。不是谁更好——是不同任务需要不同保证 |
| 事实一致性 | "有没有胡说" | 生成式摘要生成的内容在原文中是否有对应。NLI（自然语言推理）是检测幻觉的标准工具 |

---

## 📚 小结

抽取式摘要用 TextRank 无监督地选出最有代表性的句子——零标注、零幻觉。生成式摘要用 BART/Pegasus 灵活地生成新文本——高压缩但需要事实一致性检查。ROUGE 的三个变体（1/2/L）衡量的是覆盖度而非精确度——因为漏掉关键信息比多写几个词更严重。

---

## ✏️ 练习

1. 【理解】用 TextRank 对一篇 1000 字中文新闻做 top-3 摘要。对比 top-3 和文章的核心内容——哪一点被遗漏了？为什么 TextRank 没选它？

2. 【实现】在 TextRank 中加入**冗余惩罚**——当选中的两句相似度 > 0.8 时，移除得分低的那句，用下一候选替代。

3. 【实验】用 HuggingFace 的 `fnlp/bart-base-chinese` 对同一新闻做生成式摘要。对比 TextRank 和 BART 的 ROUGE-L——但更重要的是，对比两者的**事实准确性**——BART 有没有在摘要中编造了原文不存在的信息？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| TextRank + ROUGE 从零实现 | `code/summarize.py` | 分句→相似度矩阵→PageRank→ROUGE-N，含中文分词版 |

---

## 📖 参考资料

1. [论文] Mihalcea and Tarau. "TextRank: Bringing Order into Text". EMNLP, 2004. https://aclanthology.org/W04-3252/ — TextRank 论文
2. [论文] Lin. "ROUGE: A Package for Automatic Evaluation of Summaries". ACL Workshop, 2004. https://aclanthology.org/W04-1013/ — ROUGE 论文
3. [官方文档] HuggingFace. "Summarization". https://huggingface.co/docs/transformers/tasks/summarization — 生成式摘要的标准调用

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
