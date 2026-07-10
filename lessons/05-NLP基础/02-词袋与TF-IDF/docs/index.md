# 词袋模型与 TF-IDF——文本向量化

> 先数数，再思考。在词的存在本身就是信号的任务上，TF-IDF 在 2026 年仍然能打败嵌入模型。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 05 · 01（文本预处理）、阶段 02 · 02（线性回归从零）
**预计时间：** ~75 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 03（Word2Vec 词嵌入）— 当 TF-IDF 不够用时的下一步

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现词袋模型（BoW）和 TF-IDF，理解每个步骤的数学含义和设计动机
- [ ] 解释 IDF 平滑公式中每一个组件的来源——为什么加 1、为什么加 log
- [ ] 使用 scikit-learn 的 `CountVectorizer` 和 `TfidfVectorizer` 完成生产级文本向量化
- [ ] 判断一个任务应该用 TF-IDF 还是嵌入模型——基于数据量、延迟要求、可解释性需求
- [ ] 实现 TF-IDF 加权词嵌入混合方案，理解它在什么场景下优于纯 TF-IDF 和纯嵌入

---

## 1. 问题

上一课我们把文本变成了词元。这一步解决的是"模型只认数字"。

现在你有了词元序列。但词元序列仍然是变长的——第一篇文档 50 个词，第二篇 300 个词，第三篇 15 个词。而机器学习的分类器（逻辑回归、SVM、全连接层）要求的输入是一个**固定长度的向量**。

你需要把变长的词元序列，压成一个固定长度的向量。怎么做？

最朴素的想法——也是 NLP 历史上第一个被大规模验证有效的想法——是：**数数**。统计每个词出现的次数。把所有词的计数拼成一个向量。这就是词袋模型（Bag of Words，BoW）。

它听起来太简单了。但就是这么一个"数数"的操作，支撑了垃圾邮件过滤、主题分类、日志异常检测、第一波情感分析系统，以及 NLP 学术界前十年的几乎所有基准测试。到了 2026 年，当你在一个定义清晰的小规模分类任务上跑 baseline 时，TF-IDF + 逻辑回归仍然是第一选择——它快、可解释、而且在"词是否存在就足够判断"的任务上，和一个 4 亿参数的嵌入模型没有显著差距。

本节从零开始构建这一切。

---

## 2. 概念

### 2.1 从文档到向量——三条路

```
文档: "猫坐在垫子上。狗也坐在垫子上。"
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
 词袋 (BoW)     TF-IDF         词嵌入
 "猫":1,       "猫":0.42,      "猫": [0.12, -0.34, ...]
 "狗":1,       "狗":0.55,      "狗": [0.21, -0.18, ...]
 "坐":2,       "坐":0.00,      "坐": [-0.05, 0.41, ...]
 "垫子":2,      "垫子":0.00,    "垫子":[0.33, -0.09, ...]
 "上":2         "上":0.00       "上": [0.01, 0.22, ...]
    │               │               │
    ▼               ▼               ▼
 向量长度 = 词表大小   向量长度 = 词表大小   向量长度 = 嵌入维度
 完全可解释          完全可解释           不可解释
```

**词袋模型（BoW）**抛弃了词序。对于每个文档，数一数词表中每个词出现了几次。向量的第 `i` 个位置就是词 `i` 的计数。

**TF-IDF** 在 BoW 的基础上重新赋权。一个词如果出现在所有文档中（如"的"、"是"、"了"），它不提供任何区分能力——降权。一个词如果只在一两篇文档中出现，它就是那几篇文档的"指纹"——升权。

**词嵌入**（下一课的主题）将每个词映射到一个低维稠密向量。它捕获的是语义相似性——"猫"和"狗"在嵌入空间中很接近，尽管在 BoW 中它们是完全不同的两个维度。

### 2.2 TF-IDF 公式拆解

$$
\text{TF-IDF}(w, d) = \underbrace{\frac{\text{count}(w, d)}{|d|}}_{\text{TF——词频}} \times \underbrace{\log\frac{N + 1}{\text{df}(w) + 1} + 1}_{\text{IDF——逆文档频率}}
$$

逐项解释：

| 符号 | 含义 | 为什么这样设计 |
|---|---|---|
| $\text{count}(w, d)$ | 词 $w$ 在文档 $d$ 中出现的次数 | 出现越多次越重要——最朴素的假设 |
| $\vert d\vert$ | 文档 $d$ 的总词数 | 消除文档长度偏差。1000 词的文档天然比 50 词的文档拥有更高的原始计数 |
| $N$ | 文档总数 | 语料库的规模 |
| $\text{df}(w)$ | 包含词 $w$ 的文档数 | 出现在越少的文档中，越有可能是一个有区分力的词 |
| $\log$ | 对数运算 | 抑制极端值。如果没有 log，一个只出现在 1 篇文档中的词 IDF = N，出现在 N 篇中的词 IDF = 1——前者是后者的 N 倍（N 可能是 10 万） |
| $+1$（分子分母） | 平滑项 | 避免 $\log(x/0)$。当某个词在测试集中首次出现（$\text{df}=0$）时，分母至少为 1 |
| $+1$（末尾） | 偏移项 | 确保无处不在的词 IDF = 1 而非 0——被降权但不被抹杀 |

### 2.3 中文场景的特殊性

TF-IDF 的原理与语言无关——任何语言都可以数词频、算 IDF。但中文的实践有两个要注意的点：

**第一，分词质量决定一切。** 如果 jieba 把"机器学习"切成了"机器"和"学习"，那么 TF-IDF 分别得到"机器"和"学习"的统计，而丢失了"机器学习"这个完整概念。高频词"学习"的 IDF 会被稀释——因为它出现在大量关于"学习"的文档中，而不只是 ML 相关的文档。

**第二，中文词表天然更大。** 英文一个单词就是一个词元。中文需要用分词器切分——不同的切分策略产生不同的词表。加上中文的词形变化（如"的"、"了"、"着"连接的大量词组），同样规模的中文语料库，中文词表通常比英文词表大 2-5 倍。这意味着 BoW 向量更长、更稀疏。

---

## 3. 从零实现

### 第 1 步：构建词表

把整个语料库中出现的所有不重复词收集起来，给每个词分配唯一索引。

```python
def build_vocab(docs):
    """从分词后的文档列表构建词表。
    
    返回 {词语: 索引} 映射。插入顺序决定索引——
    第一个被扫描到的词索引为 0。
    """
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入是分词后的文档列表。什么分词器都可以——上一节的 `tokenize_en()`、jieba、或者 `word_tokenize`。关键在于：**当你把 BoW 模型部署到生产环境时，必须使用和训练时完全相同的分词器**。否则词表对不上。

### 第 2 步：词袋模型

```python
def bag_of_words(docs, vocab):
    """将每个文档转为词频向量。"""
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

验证：

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

每一行是一个文档，每一列是词表中的一个词。`matrix[0][0] = 1` 表示"第一个词表中出现的词在文档 0 中出现了 1 次"。文档 1 中 `cat` 出现了 2 次，所以对应列为 2。`ran` 在文档 0 中从未出现，那一列就是 0。

**注意：** 此时矩阵是**稠密 Python 列表**。当词表有 10 万个词时，每个文档向量有 10 万个元素（绝大多数是 0）。生产环境中请使用 scipy 稀疏矩阵——可以节省 98% 以上的内存。

### 第 3 步：词频与文档频率

```python
import math

def term_frequency(doc_bow, doc_length):
    """TF = 词出现次数 / 文档总词数"""
    if doc_length == 0:
        return [0.0] * len(doc_bow)
    return [c / doc_length for c in doc_bow]

def document_frequency(bow_matrix):
    """DF = 包含该词的文档数"""
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df

def inverse_document_frequency(df, n_docs):
    """IDF = log((N+1)/(df+1)) + 1"""
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

IDF 公式中的两个平滑技巧值得单独说明：

| 平滑 | 位置 | 作用 |
|---|---|---|
| `(df+1)` 而非 `df` | 分母 | 测试集中出现了训练集没见过的词 → df=0 → 避免 $\log(N/0)$ 爆炸 |
| 末尾 `+1` | 整个表达式 | 无处不在的词（df=N）IDF 至少为 1 而非 0。**被降权但不被抹杀** |

这是一个关键的设计哲学：一个出现在 100% 文档中的词，你无法确定它没有信息量——只是信息量很低。设为 0 等于你断定它毫无价值。`+1` 表达的是"我不知道它的价值，所以给它最低的非零权重"。

### 第 4 步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([t * i for t, i in zip(tf, idf)])
    return out
```

用三篇文档验证 IDF 的效果：

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

| 词 | DF | IDF | 解读 |
|---|---|---|---|
| `the` | 3 | ~1.0 | 无处不在 → 被降权到最低 |
| `sat` | 2 | ~1.3 | 中等 → 有一定的区分度 |
| `cat` | 2 | ~1.3 | 中等 |
| `dog` | 1 | ~1.7 | 稀有 → 只出现在一篇文档中 |
| `ran` | 1 | ~1.7 | 稀有 → 权重最高 |

`dog` 和 `ran` 获得了最高的 IDF 权重——它们是最好的"文档指纹"。

### 第 5 步：L2 归一化

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0.0 for x in row])
    return out
```

不做归一化的话，一篇 1000 词的文档天然拥有比一篇 50 词文档更大的向量模长——在计算"和哪个文档最相似"时，长文档会不公平地排到前面。L2 归一化将所有文档向量映射到单位超球面上，此时**余弦相似度退化为点积**：

$$\text{cosine}(a, b) = \frac{a \cdot b}{||a|| \times ||b||} = \frac{a \cdot b}{1 \times 1} = a \cdot b$$

完整代码见 `code/vectorize.py`。

---

## 4. 工业工具

### 4.1 scikit-learn——三行代码完成所有操作

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = [
    "猫坐在垫子上",
    "狗坐在垫子上",
    "猫跑过房间",
]

# 词袋模型
bow = CountVectorizer()
X_bow = bow.fit_transform(docs)
print(bow.get_feature_names_out())  # ['猫' '坐' '垫子' '上' '狗' '跑' '房间']

# TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(docs)
print(X_tfidf.toarray().round(3))
```

两个关键点：

1. **返回的是稀疏矩阵**（`scipy.sparse.csr_matrix`）。对于 10 万篇文档 × 5 万词表的矩阵，稠密版本需要 ~4GB 内存，稀疏版本可能只需要 50MB。**在分类器要求之前，始终保持稀疏**。
2. **`CountVectorizer` 内置了分词**。默认使用 `r"(?u)\b\w+\b"` 正则——这对中文完全无效（中文没有 `\b` 词边界）。中文需要自定义 tokenizer。

### 4.2 中文场景：jieba + scikit-learn

```python
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

def zh_tokenizer(text):
    return list(jieba.cut(text))

vectorizer = TfidfVectorizer(
    tokenizer=zh_tokenizer,  # 中文分词器
    ngram_range=(1, 2),      # unigram + bigram
    min_df=3,                # 至少在 3 篇文档中出现
    max_df=0.95,             # 不出现在 95% 以上的文档中
    sublinear_tf=True,       # 使用 1+log(tf) 替代原始 tf
)
X = vectorizer.fit_transform(zh_corpus)
```

### 4.3 决定 TF-IDF 效果的五个参数

| 参数 | 作用 | 建议 |
|---|---|---|
| `ngram_range=(1, 2)` | 同时使用词和双词组合 | 分类任务几乎总是建议开启，能捕获局部词序信息 |
| `min_df=2` | 丢弃出现次数过少的词 | 噪音过滤——但设太高会丢掉稀疏类别中的关键特征 |
| `max_df=0.95` | 丢弃出现频率过高的词 | 近似于自动去停用词——但情感分析禁用 |
| `stop_words` | 停用词列表 | 情感分析**必须设为 None**——否定词是核心信号 |
| `sublinear_tf=True` | 用 `1+log(tf)` 替代 `tf` | 当同一个词在一篇文档中反复出现时，抑制其对向量的过度支配 |

### 4.4 TF-IDF 在 2026 年仍然赢的场景

| 场景 | 原因 |
|---|---|
| 垃圾邮件检测、主题打标、日志异常标记 | 词的存在本身就是信号，不需要语义理解 |
| 低数据量（几百条标注） | TF-IDF + 逻辑回归没有任何预训练成本 |
| 延迟敏感（< 5ms） | 稀疏向量 × 线性模型 = 微秒级推理。嵌入模型通过 Transformer 编码一篇文档需要 10-100ms |
| 必须解释预测结果 | 直接读取分类器的权重系数——正权重最高的词就是判定为该类的原因。BERT 嵌入做不到 |

### 4.5 TF-IDF 失败的场景

**语义盲区。** 看这两句中文：

- "这部电影一点都不好看。"
- "这部电影太好看了。"

它们的 TF-IDF 重叠是：`{这部电影, 好看}`。一个词袋分类器必须"记住" `不` 靠近 `好看` 时标签翻转。它可以在足够多的数据上学会这个模式——但从不会像理解句法的模型那样优雅地学会。

另一个失败场景：**推理时遇到训练没见过的词**。TF-IDF 模型本质上是查表——不在词表中的词直接消失。这在封闭领域的文本分类中可以接受，但在开放领域的对话和搜索中无法接受。子词嵌入（阶段 05 · 04）解决了这个问题——TF-IDF 解决不了。

### 4.6 混合方案：TF-IDF 加权词嵌入

2026 年中等规模分类任务的务实选择：用 TF-IDF 权重对词嵌入做加权平均。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    """用 TF-IDF 权重对词嵌入加权求和，再做均值池化。"""
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

这种方案拿到了两边的长处——词嵌入的语义能力 + TF-IDF 对稀有词的强调。在标注样本约 5000-50000 条的分类任务上，混合方案常常同时优于纯 TF-IDF 和纯均值池化嵌入。

---

## 5. 知识连线

BoW 和 TF-IDF 的"数数"思想是后续课程的重要参照系：

- **阶段 05 · 03（Word2Vec 词嵌入）**：BoW 认为"猫"和"狗"是两个完全不同的维度——它们在向量空间中正交（余弦相似度 = 0）。词嵌入修复了这一点——"猫"和"狗"的嵌入向量夹角很小。但你要先理解 BoW 的局限，才能理解词嵌入解决了什么问题
- **阶段 05 · 04（GloVe 与 FastText）**：FastText 用子词（subword）解决了 TF-IDF 的 OOV 问题——即使训练时没见过这个词，子词拆解也能给出一个合理的向量
- **阶段 07（Transformer 深入）**：自注意力机制可以视为一种"可学习的 TF-IDF"——每个词不再用全局的 IDF 来加权，而是根据上下文动态决定它应该关注哪些词

---

## 6. 工程最佳实践

### 6.1 稀疏矩阵的存活时间

```python
# ✓ 正确：保持稀疏直到分类器需要
X_sparse = tfidf_vectorizer.fit_transform(docs)  # scipy.sparse.csr_matrix
model = LogisticRegression().fit(X_sparse, y)     # LogisticRegression 原生支持稀疏

# ❌ 错误：过早转为稠密矩阵
X_dense = X_sparse.toarray()  # 10万×5万 = 4GB，直接 OOM
```

**经验法则：** 词表 > 5000 时，BoW/TF-IDF 矩阵必须保持稀疏。scikit-learn 的 `LogisticRegression`、`SGDClassifier`、`LinearSVC` 都原生支持稀疏矩阵输入。

### 6.2 中文特别建议

- **分词一致性是第一铁律**——训练和推理使用完全相同的 jieba 版本、词典、配置。一个词在训练时被切成"机器学习"，在推理时被切成"机器/学习"→ 模型发现"机器学习"这个特征从不出现 → 准确率下降但你看不出原因
- **在 `TfidfVectorizer` 之前先做繁简转换**——`opencc` 将繁体统一为简体。否则"学习"和"學習"在 BoW 中是完全不同的两个维度
- **中文停用词列表要谨慎设计**——流行的中文停用词表（如哈工大停用词表、百度停用词表）包含了数百个词。但"不"、"没"、"别"等否定词在很多表中被标记为停用词——如果做情感分析，去掉它们等于去掉了最重要的极性信号
- **对于中文大语料（> 10 万篇），考虑用 `max_features` 限制词表大小**——`TfidfVectorizer(max_features=20000)` 只保留 IDF 最高的 2 万个词。这既能控制内存，又能自动过滤掉低频噪音词

### 6.3 踩坑经验

- **`min_df` 设为 3 可能删掉关键类别词**——如果你有 12 个类别，其中一个类别只有 100 条样本，且其中 30 条用了一个独特的词（如产品型号），`min_df=5` 会直接把这个词删除。**先跑 `get_feature_names_out()` 看类别高频词是否被误删**
- **`sublinear_tf=True` 不是免费的**——它改善了高频词的过度优势，但也减弱了"一个词出现越多越重要"的信号。在短文本分类中（如标题、搜索查询），一个词通常只出现 1-2 次——sublinear_tf 对短文本几乎没有影响
- **`token_pattern` 默认正则只匹配 2 个字符以上的词**——`CountVectorizer` 的默认参数 `token_pattern=r"(?u)\b\w\w+\b"` 会丢弃单个字符。中文自定义 tokenizer 时会绕过这个问题（因为 jieba 已经做了分词），但要注意这个默认值在英文+混合场景中的影响
- **L2 归一化之后不能直接用向量做可解释性分析**——归一化改变了原始的 TF-IDF 值。如果你需要展示"哪些词对这个文档最重要"，使用归一化之前的 TF-IDF 值。用归一化之后的向量做相似度计算

---

## 7. 常见错误

### 错误 1：对中文直接使用 `CountVectorizer` 的默认参数

**现象：** 输入中文文本，`CountVectorizer` 输出的"词"全是单个汉字。

**原因：** 默认的 `token_pattern` 使用正则 `r"(?u)\b\w+\b"`，`\b` 在英文中表示词边界，但在中文字符之间不存在——所以它把每个汉字当成一个独立的"词"。

**修复：**
```python
# ❌ 默认参数——中文被切为单字
vectorizer = CountVectorizer()  # "机器学习" → ["机", "器", "学", "习"]

# ✓ 传入 jieba 分词器
import jieba
vectorizer = CountVectorizer(tokenizer=lambda text: list(jieba.cut(text)))
```

### 错误 2：情感分析任务中去掉否定词

**现象：** 情感分类模型始终区分不出"好"和"不好"。

**原因：** `stop_words` 列表包含了"不"、"没"、"无"等否定词。去掉之后，"不好"变成了"好"——两个完全不同极性的文本在 BoW 中完全重合。

**修复：**
```python
# ❌ 情感分析——停用词去掉了"不"
vectorizer = TfidfVectorizer(stop_words=chinese_stopwords)

# ✓ 情感分析——保留所有否定词，或者根本不用停用词
vectorizer = TfidfVectorizer(stop_words=None)
```

### 错误 3：过早将稀疏矩阵转为稠密

**现象：** 处理 10 万篇文档时，`.toarray()` 直接触发 MemoryError。

**原因：** TF-IDF 矩阵的形状是 `(n_docs, vocab_size)`。10 万文档 × 5 万词表 × 8 字节（float64）= 40GB。稀疏存储只保留非零项，通常只占稠密矩阵的 2-5%。

**修复：**
```python
# ❌ 直接转为稠密——OOM
X_dense = tfidf_vectorizer.fit_transform(docs).toarray()

# ✓ 保持稀疏，或者只在小样本上转稠密
X_sparse = tfidf_vectorizer.fit_transform(docs)
model = LogisticRegression().fit(X_sparse, y)  # 原生支持稀疏输入
```

---

## 8. 面试考点

### Q1：IDF 为什么取 log 而不是直接用 N/df？（难度：⭐⭐）

**参考答案：**
直接 N/df 会产生极端值。假设 N=100000，df=1 时 N/df=100000，df=N 时 N/df=1——前者的权重是后者的 10 万倍。log 是一个压缩函数，log(100000)≈11.5，log(1)=0，两端差距从 10 万倍缩小到约 11 倍。这保证了 IDF 权重在一个合理的量级范围内——稀有词被强调，但不会完全主导整个 TF-IDF 向量。

### Q2：中文 TF-IDF 和英文 TF-IDF 在工程上最大的区别是什么？（难度：⭐⭐）

**参考答案：**
最大的区别在分词这一层。英文可以依赖空格 + 标点做初步分词，`CountVectorizer` 的默认 `token_pattern` 对英文基本可用。中文必须先通过分词器（jieba/pkuseg/HanLP）将连续的汉字序列切分为词——`TfidfVectorizer` 必须通过 `tokenizer` 参数传入中文分词函数。分词的质量直接决定了 TF-IDF 的质量，而中文分词本身就是一个带歧义的问题（"研究生命" vs "研究生/命"）。

### Q3：什么场景下 TF-IDF + 逻辑回归可能优于 BERT？（难度：⭐⭐⭐）

**参考答案：**
三点条件同时满足时：**（1）任务只依赖关键词存在与否**（垃圾邮件中的"免费"、"中奖"），不需要深层语义理解；**（2）标注样本量中等**（几百到几千条——BERT 在这个量级上的微调效果可能不如预期稳定）；**（3）有延迟或可解释性约束**（TF-IDF 推理是微秒级、BERT 是 10-100ms；TF-IDF 权重可以直接读到哪些词驱动了分类，BERT 需要额外的解释工具如 SHAP/LIME）。垃圾邮件检测、日志分类、客服工单分类——这些场景中，TF-IDF baseline 往往比 BERT 更难被超越。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 词袋 (BoW) | "按词数数" | 将文档变为词频向量的方法。词表大小 = 向量维度，每个维度存储对应词的计数。完全不关心词序 |
| 词频 (TF) | "一个词出现的比例" | 词出现次数 / 文档总词数。归一化消除文档长度偏差 |
| 文档频率 (DF) | "有多少文档用了这个词" | 包含该词的文档数量。DF 高 = 这个词太普遍了，缺乏区分能力 |
| 逆文档频率 (IDF) | "稀有词的放大器" | log(N/df) + 平滑项。降权万金油词、升权指纹词。加 log 是为了把两端倍数从万倍压缩到十几倍 |
| 稀疏向量 | "大部分是 0 的向量" | 一篇 200 词的文档，词表有 5 万个词——98% 的位置是 0。稀疏存储只保留非零项，节省 95% 以上内存 |
| L2 归一化 | "让所有文档一样长" | 将向量缩放到模长为 1。消除长文档在相似度计算中的天然优势。归一化后余弦相似度 = 点积 |
| 语义盲区 | "模型看不懂否定" | BoW/TF-IDF 不知道'不'在修饰'好看'。它只知道两个文档都有'好看'。这是选 TF-IDF 时必须承担的风险 |

---

## 📚 小结

词袋模型和 TF-IDF 是 NLP 中最简单也最耐用的文本向量化方法——用数数和加权替代了语义理解，却在足够多的场景中证明了"词的存在本身就是最强的信号"。你从零建立了词表构建、BoW 映射、TF/DF/IDF 拆解、L2 归一化的完整流水线，理解了 IDF 平滑公式中每一个数字的设计动机。

当数据量小、延迟敏感、或需要解释预测结果时，先用 TF-IDF + 逻辑回归跑一个 baseline——它在 2026 年仍然是最可靠的参照系。下一课你将学习词嵌入（Word2Vec），开始在向量空间中编码语义相似性，补上 TF-IDF 最大的盲区。

---

## ✏️ 练习

1. 【理解】用自己的话解释：为什么 IDF 公式中的 `+1`（末尾的偏移项）不是可有可无的数学技巧，而是一个工程决策？写 100 字以内的说明。

2. 【实现】扩展 `bag_of_words` 函数，支持 `ngram_range` 参数。当 `n=2` 时，`["the", "cat", "sat"]` 应同时生成 unigram 计数和 bigram 计数（`["the cat", "cat sat"]`）。

3. 【实验】用 `TfidfVectorizer` 对 20 条中文新闻标题做向量化。分别用 jieba 默认分词、jieba + 自定义词典、逐字切分三种方案。比较三种方案得到的 IDF 排名前 10 的词有何不同，解释差异原因。

4. 【实验】用 scikit-learn 的 `fetch_20newsgroups` 数据集比较三个方案：TF-IDF + 逻辑回归、纯平均词嵌入 + 逻辑回归、TF-IDF 加权词嵌入 + 逻辑回归。哪个方案在样本数降到每类 50 条时仍然最稳？

5. 【思考】如果你的 TF-IDF 模型在测试集上的表现远差于训练集，且你已经确认没有训练/推理预处理不一致的问题，你会排查哪三个方向？按照可能性从高到低排列，并给出每个方向的具体排查步骤。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| BoW + TF-IDF 从零实现 | `code/vectorize.py` | 完整流水线：词表构建 → BoW → TF → DF → IDF → TF-IDF → L2 归一化 → 余弦相似度 |
| 可复用提示词 | `outputs/prompt-vectorization-picker.md` | 根据任务推荐向量化方案，支持中英文场景分析 |

---

## 📖 参考资料

1. [论文] Salton, G. & Buckley, C. "Term-weighting approaches in automatic text retrieval". Information Processing & Management, 1988. https://www.sciencedirect.com/science/article/pii/0306457388900210 — 这篇论文奠定了 TF-IDF 在一代检索系统中的默认地位
2. [官方文档] scikit-learn. "Feature extraction from text". https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction — 所有参数的权威参考
3. [官方文档] scikit-learn. "TfidfVectorizer". https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html — 每个参数的默认值和设计动机
4. [GitHub] jieba. "结巴中文分词". https://github.com/fxsjy/jieba — 中文 TF-IDF 的前置依赖

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、中文分词与 TF-IDF 结合实践、工程最佳实践、常见错误、面试考点等均为原创内容。
