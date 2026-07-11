# 主题建模——LDA 与 BERTopic

> LDA：文档是主题的混合，主题是词的分布。BERTopic：文档在嵌入空间聚类，每个簇是一个主题。相同的目标，不同的分解方式。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 05 · 02（BoW + TF-IDF）、阶段 05 · 03（Word2Vec）
**预计时间：** ~45 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 22（嵌入模型深入）— BERTopic 的嵌入步骤可以替换为阶段 22 中的任何稠密模型

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 使用 scikit-learn 拟合 LDA——理解为什么用 `CountVectorizer` 而非 `TfidfVectorizer`（LDA 需要原始计数）
- [ ] 使用 BERTopic 在语义空间中聚类文档——理解 UMAP → HDBSCAN → c-TF-IDF 的流水线
- [ ] 根据文档长度、计算资源和是否需要混合成员，在 LDA 和 BERTopic 之间做出选择
- [ ] 计算 c_v 主题一致性并理解它的局限性——为什么人类判断仍然是最后一道防线

---

## 1. 问题

你有 1 万条客服工单、5 万篇新闻、20 万条微博。你需要知道这些内容在谈什么——但你没有标注类别，甚至不知道有几个类别。

主题建模在无监督条件下回答这个问题。喂给它语料，回道少量连贯主题 + 每篇文档的主题分布。

两套算法主导这个领域。**LDA（2003）**：每个文档是潜主题的混合，每个主题是词的分布。推断是贝叶斯的。2026 年仍在需要混合成员主题分配和可解释词级概率分布的生产场景中运行。**BERTopic（2020）**：用 BERT 编码 → UMAP 降维 → HDBSCAN 聚类 → 基于类的 TF-IDF 提取主题词。在短文本、社交媒体和语义相似比词重叠更重要的一切场景中胜出。一篇文档一个主题——对长内容是一个局限。

本课建立对两者的直觉，并命名在给定语料库上选择哪一个。

---

## 2. 概念

### 2.1 LDA 的生成故事

每个主题是词的分布。每个文档是主题的混合。要生成文档中的一个词——从文档的主题混合中采样一个主题，再从该主题的词分布中采样一个词。

**推断逆转这个过程**——给定观察到的词，推断每篇文档的主题分布和每个主题的词分布。Collapsed Gibbs 采样（阶段 05 · 15 的 code 目录中有完整实现）或变分贝叶斯做这个数学。

LDA 的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行和为 1（文档的主题混合）
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行和为 1（主题的词分布）

### 2.2 BERTopic 流水线

```
文档 → [1. Sentence-BERT 编码] → 384d 向量
     → [2. UMAP 降维] → ~5d（BERT 嵌入对聚类来说维度太高）
     → [3. HDBSCAN 聚类] → 变大小簇 + "噪声"标签(-1)
     → [4. 类 TF-IDF] → 每个簇的 top 词
```

输出是每篇文档一个主题（加上 -1 异常标签）。可选的是通过 HDBSCAN 的概率向量得到软成员。

---

## 3. 从零实现

### 第 1 步：LDA——scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",    # LDA 需要去停用词——否则噪声主题淹没信号
        min_df=2,                # 过滤仅出现1次的噪音词
        max_df=0.9,              # 过滤接近全语料的通用词
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",  # 在线学习——比 batch 更快、适合大数据
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names

def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：用的是 `CountVectorizer`（不是 `TfidfVectorizer`）——因为 LDA 的生成过程基于词计数，TF-IDF 权重会破坏计数语义。`min_df` 和 `max_df` 过滤掉稀有词和无处不在的词。

### 第 2 步：BERTopic——生产版

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,   # HDBSCAN 最小簇大小。> 10,000 文档时增加到 50-100
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))

# 过滤掉异常簇 (-1)
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` 的过滤去掉了 BERTopic 的异常桶——HDBSCAN 无法聚类的文档。`min_topic_size` 控制 HDBSCAN 的最小簇大小；BERTopic 库默认是 10。本例明确设为 15。

### 第 3 步：评估——c_v 一致性、多样性、人工检查

- **主题一致性（c_v）。** 在滑动窗口上下文中将 top 词对的 NPMI（归一化逐点互信息）组合为分数。越高越好。用 `gensim.models.CoherenceModel` + `coherence="c_v"`
- **主题多样性。** 所有主题的 top 词中不重复词的比例。越高越好——主题不应该重叠
- **定性检查。** 阅读每个主题的 top 词。它们命名的是一件真实存在的事吗？人工判断仍然是最后一道防线

完整代码见 `code/lda_demo.py`（Collapsed Gibbs 采样从零实现）。

---

## 4. 何时选哪个

| 场景 | 选择 |
|---|---|
| 短文本（微博、评论、标题） | BERTopic |
| 长文档，主题混合 | LDA |
| 无 GPU / 计算受限 | LDA 或 NMF |
| 需要文档级多主题分布 | LDA |
| LLM 集成用于主题标注 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 最大语义一致性 | BERTopic |

**最大的实际考量是文档长度。** BERT 嵌入会截断；LDA 的计数在任何长度上都工作。对于超过嵌入模型上下文的文档——要么分块再加总，要么直接用 LDA。

---

## 5. 工业工具——2026 技术栈

- **BERTopic。** 短文本和语义优先场景的默认选择
- **`gensim.models.LdaModel`。** 经典 LDA 的生产实现——成熟、久经考验
- **`sklearn.decomposition.LatentDirichletAllocation`。** 实验用 LDA——简单
- **NMF。** 非负矩阵分解。LDA 的快速替代——在短文本上与 LDA 质量相当
- **Top2Vec。** 设计类似于 BERTopic。社区较小但在部分基准上表现好
- **FASTopic。** 较新，在超大规模语料上比 BERTopic 更快
- **LLM 标注。** 跑任意聚类，然后 prompt 模型为每个簇命名

### 中文主题建模特别建议

- **中文 LDA 必须先分词。** 直接用逐字 LDA → 主题词全是单字（"机"/"器"/"学"/"习"各自独立），完全不可解读。用 jieba 分词 + 去单字词 + n-gram 合并（"机器"+"学习" → "机器学习"）后再送入 `CountVectorizer`
- **BERTopic 中文嵌入选 `paraphrase-multilingual-MiniLM-L12-v2` 或 `bge-m3`** ——前者轻量且对中文足够好，后者在中文语义上更强
- **主题数经验公式：** `n_topics ≈ max(5, sqrt(n_docs))`，上限在 200（语料 < 4 万篇时）。超过 200 大概率是过度切分

---

## 6. 常见错误

### 错误 1：LDA 用 TfidfVectorizer 而非 CountVectorizer

**现象：** 主题词分布看起来"扁平"——每个词的概率几乎相同，没有一个词明显突出。

**原因：** LDA 的生成过程假设每个位置是一个从主题词分布中采样的词——而采样基于的是**计数频率**，不是 TF-IDF 权重。TF-IDF 已经把高频词压制了——LDA 被剥夺了识别"一个词在该主题中反复出现"这一最强信号的机会。

**修复：** 永远用 `CountVectorizer`（原始计数）做 LDA。

### 错误 2：中文 BERTopic 不经分词直接嵌入

**现象：** 每个中文文档被当成一整句话嵌入——主题聚类质量远低于英文。

**原因：** 英文嵌入模型在空格分隔的词序列上训练——中文的连续字符串对嵌入模型来说是"噪音流"。嵌入质量下降导致 UMAP 的流形结构和 HDBSCAN 的密度估计全部走偏。

**修复：** 先 jieba 分词——用空格拼接——再送入 BERTopic 的嵌入模型。或者直接用 `bge-m3`——它在中文 token 级别预训练了大量中文语料，逐字输入也能维持合理的嵌入质量。

---

## 7. 面试考点

### Q1：LDA 的 α 和 β 超参数控制了什么？（难度：⭐⭐）

**参考答案：**
α 控制文档-主题分布的稀疏度——小 α → 每篇文档只有 1-2 个主题。β 控制主题-词分布的稀疏度——小 β → 每个主题只有少数核心词。对大多数语料库，α=0.1, β=0.01 是安全的默认值。小数据集上调大 α（让文档可以混合更多主题），主题混乱时调小 β（强迫每个主题更聚焦）。

### Q2：什么时候 BERTopic 不如 LDA？（难度：⭐⭐）

**参考答案：**
两个条件同时成立时：**（1）文档长度超过嵌入模型的上下文窗口**——BERT 在 512 token 处截断（或 8192，视模型而定），剩余内容信息完全丢失；**（2）需要混合成员**——法律文档可能同时涉及合同、侵权和知识产权。BERTopic 把整篇文档塞进一个簇，丢失了多主题混合的信息。LDA 天然支持混合成员，且计数机制不受文档长度限制。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 主题 | "语料库谈论的一件事" | 词的概率分布（LDA）或相似文档的簇（BERTopic） |
| 混合成员 | "一篇文档可以有不止一个主题" | LDA 给每篇文档分配所有主题的分布。BERTopic 给一篇文档一个主题 |
| UMAP | "降维" | 保持局部结构的流形学习；BERTopic 使用它 |
| HDBSCAN | "密度聚类" | 发现变大小簇；对异常文档产生"噪声"标签 (-1) |
| c_v 一致性 | "主题的质量分" | 主题-top 词在滑动窗口内的平均逐点互信息。越高越好 |
| 困惑度 | "模型的惊讶" | 越低 = 模型越好。对 LDA——衡量未见文档的对数似然 |

---

## 📚 小结

LDA 和 BERTopic 代表了主题建模的两条路——贝叶斯概率建模（LDA）和语义嵌入聚类（BERTopic）。选择取决于文档长度（长→LDA）、是否需混合成员（是→LDA）、短文本/社交媒体（是→BERTopic）、以及计算预算（受限→LDA/NMF）。

中文主题建模的致命陷阱是跳过分词——逐字 LDA/BERTopic 输出的全是单字，不可解读。永远先 jieba 分词，再送入任一模型。

---

## ✏️ 练习

1. 【理解】在 20 Newsgroups 上拟合 5 主题的 LDA。打印每主题前 10 词。手工标注每个主题。算法找到了真实的类别了吗？

2. 【实现】在同一份 20 Newsgroups 子集上拟合 BERTopic。比较找到的主题数、top 词和定性一致性。哪个更干净地浮现了真实的类别？

3. 【实验】在你的语料上同时计算 LDA 和 BERTopic 的 c_v 一致性。各跑 5、10、20、50 个主题的配置。绘制一致性 vs 主题数的曲线。报告哪种方法在主题数变化时更稳定。

4. 【思考】你有一批 5 万字的中国法律文书——每篇 8000-15000 字。LDA 还是 BERTopic？给出两个具体理由。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| LDA Collapsed Gibbs 从零实现 | `code/lda_demo.py` | 完整采样推导 + 文档-主题 + 主题-词 三个输出 |

---

## 📖 参考资料

1. [论文] Blei, Ng, Jordan. "Latent Dirichlet Allocation". JMLR, 2003. https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf — LDA 论文
2. [论文] Grootendorst. "BERTopic: Neural topic modeling with a class-based TF-IDF procedure". 2022. https://arxiv.org/abs/2203.05794 — BERTopic 论文
3. [论文] Röder, Both, Hinneburg. "Exploring the Space of Topic Coherence Measures". WSDM, 2015. https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf — 引入 c_v 和其他指标的论文
4. [官方文档] BERTopic. https://maartengr.github.io/BERTopic/ — 生产参考，有优秀的示例

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文主题建模建议（jieba 分词前置、主题数公式）、工程最佳实践、常见错误、面试考点等均为原创内容。
