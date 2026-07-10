# 情感分析——Naive Bayes 与 Logistic Regression

> NLP 的"Hello World"任务。每一个看起来简单的案例背后都藏着一个难的——这就是为什么情感分析是经典 NLP 最好的实验场。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 05 · 02（BoW + TF-IDF）、阶段 02 · 14（朴素贝叶斯）
**预计时间：** ~75 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 06（命名实体识别）— 情感分析中的方面级情感需要先识别实体

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现多项式朴素贝叶斯分类器，理解拉普拉斯平滑为什么是必须的
- [ ] 实现否定范围标记（中英文双版本），解释它为什么用三行代码就能显著提升情感分类准确率
- [ ] 在不平衡数据上选择正确的评估指标——宏平均 F1 而非准确率
- [ ] 判断一个情感分析任务该用 TF-IDF + 逻辑回归还是 Transformer

---

## 1. 问题

"这个电影一点都不好看。"
"这个电影太好看了。"

正面还是负面？

情感分析听起来太简单了。评论者说喜欢还是不喜欢，打标签就行。它之所以成为 NLP 的经典任务，恰恰是因为**每一个看起来简单的案例，背后都藏着一个难的。**

否定翻转了语义。"一点都不好看"的每个词拆开看都不负面——"一点"、"都"、"不"、"好看"——但四个词组合在一起是彻底的差评。词袋模型看到的只是 `{一点, 都, 不, 好看}`，它不知道"不"在修饰"好看"。

反讽把语义彻底反转了。"这演技真是绝了"——在大拇指和笑脸之间，到底是夸奖还是嘲讽？人类有时都分不清，模型更分不清。

领域词汇改变了一切。"这个手机太烫了"是差评，"这碗汤太烫了"可能是好评（说明刚出锅）。

还有 emoji、语气词、程度副词、双重否定……情感分析是经典 NLP 的完整实验场。如果你理解了每一个朴素 baseline 为什么有特定的失败模式，你就理解了为什么每一个更复杂的模型被发明出来。

---

## 2. 概念

### 2.1 经典情感分析的两步公式

```
文本 → [表示] → 特征向量 → [分类] → 正面/负面
```

**第一步——表示。** 把文本变成固定长度的特征向量。BoW、TF-IDF、n-gram——阶段 05 · 02 的全部内容都在这里发挥作用。

**第二步——分类。** 在有标注的样本上拟合一个线性模型。本课聚焦两个：朴素贝叶斯和逻辑回归。

### 2.2 朴素贝叶斯——最笨但能用的模型

朴素贝叶斯做了一个"明显是错的"的假设：**给定类别后，每个特征相互独立。** 也就是说，它假设"好"和"看"在正面评论中出现的概率是独立的——尽管它们在"好看"这个词中打包出现。

这个假设在文本上严重不成立。但奇怪的是，结果好得惊人。原因是：

**在稀疏的文本特征和中等数据量下，分类器关心的是"每个词更偏向哪一边"的方向性信息，而不是精确的联合概率。** "好"在正面评论中出现 50 次，在负面中出现 5 次——这个 10:1 的偏向性信号已经足够做正确的分类决策。朴素贝叶斯不需要知道"好"和"看"经常一起出现——它只需要知道"好"单独出现时已经强烈指向正面。

训练过程——从标签文档中统计：

```
P("好" | 正面) = (正面评论中"好"出现的次数 + alpha) /
                 (正面评论总词数 + alpha × 词表大小)

P(正面 | 文档) ∝ P(正面) × P(词1|正面) × P(词2|正面) × ...
```

**alpha（拉普拉斯平滑）**不是可选的——它是必须的。一个词在训练集中从未在负面评论中出现过，不代表它永远不会在负面评论中出现。不做平滑 → `P(word|负面) = 0` → `log(0) = -∞` → 整个文档的得分直接爆炸。

### 2.3 逻辑回归——修复独立性假设

逻辑回归给每个特征学一个权重——包括负权重。`not good` 作为 bigram 特征获得一个负的权重，模型就学到了"当 `not_good` 出现时，推向负面"。

朴素贝叶斯做不到这一点——因为它的特征之间是独立的，没有机制让"not"修改"good"的权重。

```
文档 → BoW/TF-IDF 向量 → w·x + b → sigmoid → P(正面)
```

L2 正则化在文本特征上尤其重要——特征是稀疏且高维的，没有 L2 的话模型直接把训练集背下来了。

### 2.4 否定处理——三行代码，显著提升

在 bigram 不可用的场景（如超小词表），一个简单粗暴但效果显著的方法：**否定标记（Negation Scoping）。**

"not good at all" → `['not', 'NOT_good', 'NOT_at', 'NOT_all']`

现在 `good` 和 `NOT_good` 在 BoW 中是完全不同的两个特征。分类器可以给 `good` 分配正面权重、给 `NOT_good` 分配负面权重——不需要理解句法，只需要两个独立的特征。

**中文版：** "一点都不好看" → `['不', 'NOT_好看', 'NOT_一点']`。中文的否定词更灵活——"不"、"没"、"毫无"、"绝不"都可以触发否定标记。但这套机制对中文的提升不如英文显著——因为中文的否定可以嵌入在复杂句式中（"好看不到哪去"——否定词"不"在句子中间，前方没有否定词直接修饰"好看"），简单的线性否定范围假设经常捉不住。

---

## 3. 从零实现

### 第 1 步：构建玩具数据集

故意放小——真实项目中你会用数千到数万条标注数据。数学完全一样。

```python
positive = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

negative = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

注意：这两组数据中，`movie` 同时出现在正面和负面中（`loved this movie` vs `waste of a movie`）。这是正确的——"movie"这个词本身不承载情感信号，它应该被两个类都学到。

### 第 2 步：否定标记

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";", "but"}

def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

`but` 是一个聪明但可选的终止符——"not good but funny" 的否定范围到 "good" 截止，"funny" 不应该被标记为否定。加不加 `but` 取决于你在精确率和召回率之间的选择。

### 第 3 步：多项式朴素贝叶斯

```python
import math
from collections import Counter

def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(docs) for docs in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        # 先验概率：P(class)
        class_priors[cls] = len(docs) / total_docs
        # 统词频
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        # 条件概率 + 拉普拉斯平滑
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }

    return class_priors, class_word_probs

def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])   # log P(class)
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])  # Σ log P(word|class)
        scores[cls] = s
    return max(scores, key=scores.get)
```

**两个关键设计决策：**

1. **为什么取 log？** 50 个词的概率连乘——每个 < 0.01 → 乘积下溢出为 0.0。log 把连乘变成加法，数值稳定，且单调性不变（得分最高的类不变）。

2. **alpha 的选择：** `alpha=1.0` 是教学默认（经典拉普拉斯）。`alpha=0.01` 是生产常用——越小的 alpha 意味着越依赖实际数据，越大的 alpha 意味着越信任"平权先验"。在小数据集上 alpha 大一点更安全；在百万级数据集上 alpha 再小都没问题。

### 第 4 步：评估——不是准确率

```python
def evaluate(y_true, y_pred, pos_label="+", neg_label="-"):
    tp = sum(1 for t, p in zip(y_true, y_pred)
             if t == pos_label and p == pos_label)
    fp = sum(1 for t, p in zip(y_true, y_pred)
             if t == neg_label and p == pos_label)
    fn = sum(1 for t, p in zip(y_true, y_pred)
             if t == pos_label and p == neg_label)
    tn = sum(1 for t, p in zip(y_true, y_pred)
             if t == neg_label and p == neg_label)

    # ... 计算精确率、召回率、F1（正负类各一组）
    macro_f1 = (f1_pos + f1_neg) / 2
    return {..., "macro_f1": macro_f1}
```

**宏平均 F1 是你处理不平衡数据时的默认指标。** 假设 80% 正面 + 20% 负面——一个"永远预测正面"的分类器有 80% 准确率但 0% 的负面召回率。宏平均 F1 把两个类等权对待——80% 正面 × 0% 负面 → 宏平均 F1 = 40%，真实反映模型的无效。

完整代码见 `code/sentiment.py`。

---

## 4. 工业工具

### 4.1 scikit-learn——六行代码，流程标准

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigram + bigram → 'not good' 成为独立特征
        min_df=2,              # 过滤噪音词
        sublinear_tf=True,     # 1+log(tf) 抑制高频词
        stop_words=None,       # 情感分析不去停用词！
    )),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

**这三个参数就是 75% baseline 和 85% baseline 之间的差距：** `ngram_range=(1, 2)` 捕获了 `not good`、`very bad` 这类否定和程度搭配。`stop_words=None` 保留了所有否定词。`sublinear_tf=True` 防止一个词反复出现时过度支配向量。

### 4.2 什么时候该用 Transformer

| 场景 | 选择 | 原因 |
|---|---|---|
| 讽刺检测 | Transformer | 经典模型在这里完全失败——字面意思和真实意图相反 |
| 长文档中情感转折 | Transformer | "前半段很无聊但结局惊艳"——需要理解文档结构 |
| 基于方面的情感分析 | Transformer | "相机很好但电池很差"——需要把情感归因到具体方面 |
| 低资源语言 | 多语言 BERT | 零样本基线，不需要你标注几百小时 |
| 其他所有场景 | TF-IDF + 逻辑回归 | 够快、够准、可解释——2026 年的生产 baseline |

### 4.3 中文情感分析特别建议

- **中文情感的微妙度更高。** "还行"、"一般般"、"也就那样"——这些表达在字面上没有情感词，但传达了明确的中性或偏负面的态度。这类"隐性情感"是中文 NLP 的独特挑战——`TfidfVectorizer` 只能捕捉显性情感词（"好"、"烂"），碰不到这类表达
- **领域适应比模型选择更重要。** 在手机评论上训练的情感模型直接用在酒店评论上——准确率可能从 85% 掉到 65%。"发热"在手机评论中是负面，在暖宝宝评论中是正面。**先标注 200 条目标领域的数据，fine-tune 你的 baseline，这个投资的回报远超更换模型架构**
- **中文情感数据集推荐：** ChnSentiCorp（酒店/笔记本/书籍评论）、weibo_senti_100k（微博情感 10 万条）、DianPing（大众点评评论）

---

## 5. 知识连线

情感分析是本阶段前四课的第一个"整合点"：

- **阶段 05 · 01（文本预处理）** 的分词器 → 情感分析的输入
- **阶段 05 · 02（BoW + TF-IDF）** 的 TfidfVectorizer → 情感分析的特征
- **阶段 05 · 03（Word2Vec）** 的词嵌入 → 为朴素贝叶斯和逻辑回归提供替代特征来源（TF-IDF 加权嵌入在中等数据量上常优于纯 TF-IDF）
- **阶段 07（Transformer 深入）** — 当你遇到讽刺、方面级情感、或跨语言需求时，回来找 Transformer

---

## 6. 工程最佳实践

### 6.1 Baseline 的再生产陷阱

情感分析模型的复训是常规操作。但重新评估 baseline 不是。

论文中报告的准确率数字使用了特定的数据划分、特定的预处理、特定的 tokenizer。如果你比较你的新模型和论文中的 baseline 数字——你很可能在比较苹果和橘子。**始终在你自己的流水线上重新生成 baseline 数字，而不是引用论文中的数字。**

### 6.2 中文特别建议

- **否定标记对中文的提升不如英文稳定**——中文的否定结构更多样。在简单否定（"不 + adj"）上否定标记提升显著，但在复杂句式（"也没好看到哪去"）上它常常捉不到真正的否定范围。作为基线改进值得尝试，但别期待和英文一样稳定的 ~5% F1 提升
- **训练数据和测试数据必须来自同源**——用微博数据训练，去测大众点评评论 = 灾难。社交媒体和电商评论的用词分布、情感表达习惯完全不同。如果必须跨域，至少做一次手动抽查——随机读 50 条目标域数据，确认你的特征词覆盖度
- **中文评论中 emoji 和颜文字是强情感信号**——(●'◡'●)、😂、😡、(╯°□°）╯︵ ┻━┻。这些信号在标准分词流水线中可能被当作标点丢弃。如果你做社交媒体中文情感分析，需要额外的 emoji→情感映射表或保留它们作为独立 token

### 6.3 踩坑经验

- **`C` 参数（L2 正则化的倒数）在文本上通常设 1.0 不够小**——文本特征 1-2 万维，C=0.1 或 C=0.01 常常更好。用 `LogisticRegressionCV` 自动搜索
- **报告 Micro-F1 而非 Macro-F1 在不平衡数据上会给你虚假的安全感**——Micro-F1 由多数类主导，看起来很高但掩盖了少数类的完全失败。情感分析中负面类通常是少数——你不想一个标榜 90% F1 的系统把所有负面评价都标成了正面
- **"not bad" 不等于"good"——否定标记之后，`NOT_bad` 是一个独立特征，但它没有学会"NOT_bad 接近 good"这个语义关系。** 如果有足够的数据，bigram + 逻辑回归能学到 `not bad` → 偏正面。否定标记是穷人的替代方案——在极小的标注数据集上效果最好

---

## 7. 常见错误

### 错误 1：情感分析中去停用词

**现象：** "不好看"和"好看"被模型判定为相同情感。

**原因：** 停用词表包含了"不"、"没"、"别"。去掉之后，"不好看"变成了"好看"——评论的情感极性被彻底翻转。同样的，`not`、`no`、`never` 在几乎所有英文停用词表中——去掉它们等于删除了情感分析最重要的信号。

**修复：**
```python
# ❌ 使用默认停用词
vectorizer = TfidfVectorizer(stop_words="english")

# ✓ 情感分析——不去任何停用词
vectorizer = TfidfVectorizer(stop_words=None)
```

### 错误 2：不平衡数据上报准确率为主指标

**现象：** 模型有 88% 准确率，但负面评论一个都识别不出来。

**原因：** 数据是 88% 正面 + 12% 负面。"全部预测正面"就获得 88% 准确率——但这个模型在生产环境毫无价值（用户抱怨"为什么差评的商品也被推荐给我"）。

**修复：** 报告宏平均 F1，同时展示混淆矩阵——后者会让你一眼看到负面那行全空。

### 错误 3：用训练集评估

**现象：** F1 99%，一上线就崩。

**原因：** 机器学习的第一条规则：不在你训练过的数据上评估。小数据集上更容易犯这个错——"反正就几十条，全用来训练吧"。朴素贝叶斯背下了训练集中的每个特征组合，但没见过的新表达完全无法泛化。

**修复：** 手写一个简单的留出法（hold-out）——随机抽 20% 作为测试集。对于 < 100 条的数据，用留一法交叉验证（leave-one-out）。

---

## 8. 面试考点

### Q1：朴素贝叶斯的"朴素"假设明显是错的——为什么它在文本分类上仍然好用？（难度：⭐⭐）

**参考答案：**
文本分类中，模型主要依赖的信号是每个词"更偏向哪一类"的方向性信息——"好"在正面中出现 10 倍于负面，这个 10:1 的偏向性已经足够做正确的决策。独立性假设错的是词间的联合概率（`P(好且看|正面)` ≠ `P(好|正面)×P(看|正面)`），但"方向性"信息对独立性不敏感——`P(好|正面)/P(好|负面)` 的比值不受特征依赖的影响。所以朴素贝叶斯虽然概率估计不准，但分类边界出奇地正确。

### Q2：否定标记和 bigram 都能处理否定——选哪个？（难度：⭐⭐）

**参考答案：**
Bigram 是更好的方案——`not good` 作为一个整体特征，分类器自动学到它的权重。否定标记的优势在于它不需要更多的标注数据——在极小数据集（< 200 条）上，`not good` 可能只出现一次但 `NOT_good` 因为 `good` 的高频而获得了足够的统计信号。**简单说：标注数据充足 → bigram。标注数据极其有限 → 否定标记作为替代。**

### Q3：情感分析模型上线后准确率持续下降，但代码没改——排查什么？（难度：⭐⭐⭐）

**参考答案：**
按可能性从高到低排查：**（1）领域漂移（Domain Drift）——** 用户的评论内容随时间变化（新品发布、社会热点引发新的表达方式）。`TOKEN_good` 的统计分布仍在原地，但用户开始用"真的绝绝子"来表达好评。**（2）分词器/预处理版本升级——** NLTK 或 jieba 的小版本变化导致同样的文本被切成不同的 token，和你训练时的分布对不上。**（3）标签泄漏修复——** 训练时某个强特征（如产品型号）在当前数据中出现的模式变了。排查方法：每月固定抽取 100 条最新预测结果，人工比对预测和实际情感，统计错误类型分布。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 情感极性 (Polarity) | "正面还是负面" | 二元标签。有时扩展为三元（正面/中性/负面）或细粒度（1-5 星） |
| 基于方面的情感分析 | "评价具体某个东西" | 将情感归属到文本中提到的具体实体或属性上。'相机很好但电池很差' → 相机=正面，电池=负面 |
| 否定标记 (Negation Scoping) | "给后面的词加 NOT_" | 否定词之后的词加前缀，直到标点，使它们成为独立特征。三行代码，在极小数据集上效果显著 |
| 拉普拉斯平滑 | "每个词都加 1" | 防止 P(word\|class)=0 → log 爆炸。alpha 越小越信任数据，越大越保守 |
| L2 正则化 | "给权重加个惩罚" | loss += λ·Σw²。在稀疏高维文本特征上防止背训练集——不用则模型直接过拟合 |
| 宏平均 F1 | "每个类算完再平均" | 等权平均每个类的 F1。不平衡数据的第一指标——少数类不会被多数类淹没 |

---

## 📚 小结

情感分析是 NLP 的"Hello World"——用最简单的工具（TF-IDF + 逻辑回归）就能做到生产可用的水平，但每一层简化都有精确的失败模式。你实现了一个从分词、否定标记、到朴素贝叶斯分类的完整流水线，理解了为什么宏平均 F1 而非准确率是不平衡数据的正确指标。

当否定、讽刺、方面级情感、跨语言这四道坎各自迈不过去时，回来找 Transformer。其他时候，TF-IDF + 逻辑回归框架已经足够回答"用户喜欢还是不喜欢"——快、准、可解释。

---

## ✏️ 练习

1. 【理解】"这个东西不怎么样"在去掉停用词后变成"东西/怎么样"——解释为什么这个简单的变换可能导致情感极性被完全错误地分类。写 80 字以内的说明。

2. 【实现】在朴素贝叶斯实现中加入 **TF-IDF 加权**——统计词频时用 TF-IDF 值而非原始计数。在小测试集上比较加权前后 F1 的变化，解释变化的原因。

3. 【实验】用 scikit-learn 的 `LogisticRegression` + `TfidfVectorizer` 在中文酒店评论数据（ChnSentiCorp）上训练。分三组实验：默认停用词、不去停用词、不去停用词 + bigram。报告三组的宏平均 F1 差异。

4. 【思考】"这部电影前半段很无聊但结局惊艳"——用词袋模型分析这句话的情感会遇到什么问题？如果设计一个最简单的补救方案（不需要 Transformer），你会怎么做？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 情感分析完整流水线 | `code/sentiment.py` | 朴素贝叶斯 + 否定标记（中英文）+ 评估指标，从零实现 |

---

## 📖 参考资料

1. [论文] Pang and Lee. "Opinion Mining and Sentiment Analysis". Foundations and Trends in IR, 2008. https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html — 情感分析的基础综述，前四章覆盖了所有经典方法
2. [论文] Wang and Manning. "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification". ACL, 2012. https://aclanthology.org/P12-2018/ — 证明 bigram + 朴素贝叶斯在短文本上难以超越
3. [官方文档] scikit-learn. "Working with text data". https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html — 情感分析流水线的官方教程

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、中文否定处理实现、中文情感分析特别建议、常见错误、面试考点等均为原创内容。
