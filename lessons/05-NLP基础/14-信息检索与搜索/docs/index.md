# 信息检索与搜索

> BM25 精确但脆弱。稠密检索覆盖面广但漏关键词。混合是 2026 年的默认选择。其余都是调参。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 05 · 02（BoW + TF-IDF）、阶段 05 · 04（GloVe、FastText、子词）
**预计时间：** ~75 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 13（问答系统）— 本课的四层架构是每个 RAG 系统检索侧的底座

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 BM25 类——理解 k1（词频饱和）和 b（长度归一化）如何控制评分行为
- [ ] 用 Sentence-Transformers 构建稠密检索索引——理解双编码器和 L2 归一化
- [ ] 实现 RRF 融合——解释为什么 k=60 和对原始分数尺度的无视是 RRF 的核心设计
- [ ] 搭建混合检索 + Cross-encoder 重排的四层流水线——BM25 → 稠密 → RRF → 重排

---

## 1. 问题

用户输入"如果有人撒谎来骗取钱财会怎样"——期望找到的法律条文是"印度刑法第 420 条：欺诈罪"。关键词搜索完全找不到——没有共享词汇。语义搜索如果嵌入模型不是法律领域训练的，同样会漏。**真实的搜索必须两样都处理。**

IR 是每个 RAG 系统、每个搜索框、每个文档站点模糊查询的底层流水线。2026 年在生产中工作的架构不是单一方法——它是互补方法的链条，每一环捕获前一环的失败。

本课从零构建每一个环节，并明确每个环节捕获的失败。

---

## 2. 概念——四层混合检索

```
查询 → [1. BM25 稀疏] ──→ [3. RRF 融合] → [4. Cross-encoder 重排] → top-5 结果
      [2. 稠密检索  ] ──→
```

| 层 | 方法 | 延迟 | 擅长 | 盲区 |
|---|---|---|---|---|
| 1 | BM25 稀疏 | < 10ms | 精确关键词、代码/编号/实体名 | 同义词、释义 |
| 2 | 稠密检索 | 50-200ms | 语义相似、跨语言、paraphrase | 单字符差异、精确字符串 |
| 3 | RRF 融合 | < 1ms | 合并两路排名——不关心分数尺度 | — |
| 4 | Cross-encoder 重排 | 每对 5-10ms | top-30 精排——远比双编码器准 | 慢——只跑 top-30 |

三路检索（BM25 + 稠密 + 学到的稀疏如 SPLADE）在 2026 基准上优于两路，但需要 SPLADE 索引的基础设施。对大多数团队，两路 + Cross-encoder 重排是最佳位置。

---

## 3. 从零实现

### 第 1 步：BM25 从零

```python
import math, re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]")  # 英文 + 中文逐字

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus: raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1, self.b = k1, b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl, freq = len(doc), Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0: continue
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * num / den
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得了解的参数。**`k1=1.5`** 控制词频饱和——值越高，词重复出现时权重增长越多。**`b=0.75`** 控制长度归一化——0 完全忽略文档长度，1 完全归一化。默认值是 Robertson 在原始论文中的建议，很少需要调整。

### 第 2 步：稠密检索——双编码器

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def build_dense_index(corpus, model_id="all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings

def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()  # L2 归一化后点积 = 余弦
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2 归一化嵌入使点积等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维，快，对大多数英文检索足够强。多语言用 `paraphrase-multilingual-MiniLM-L12-v2`。中文首选 `bge-m3`。最高精度用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第 3 步：RRF——倒数排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

**为什么 RRF 的核心设计是"不看分数只看排名"？** BM25 的分数尺度在 0-50+，余弦相似度在 0-1。两条分数的绝对值不可比较。RRF 只依赖排名——"两路都排前三的文档"自动获得最高融合分。`k=60` 来自原始 RRF 论文——k 越高，排名差异的贡献越平缓；k 越低，顶部排名越主导。60 是发表的默认值，很少需要调整。

### 第 4 步：混合搜索 + 重排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_search(query, bm25, encoder, dense_emb, corpus,
                  top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_emb, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合。BM25 发现词汇匹配。稠密检索发现语义关联。RRF 合并两个排名——无需分数校准。Cross-encoder 用查询-文档对一起打分的方式重排 top-30——捕获双编码器漏掉的细粒度相关性。保留 top-5。

### 第 5 步：评估指标

| 指标 | 含义 |
|---|---|
| Recall@k | 正确答案存在的查询中，正确文档在前 k 名中的比例 |
| MRR（平均倒数排名） | 第一个相关文档排名的倒数的平均值 |
| nDCG@k | 考虑相关性多级（非二元），不仅是相关/不相关 |

对于 RAG，**检索器的 Recall@k 是最重要的数字。** 正确的段落不在检索集合中，生成器不可能回答正确。

调试技巧：对失败的查询，对比稀疏和稠密两个排序列。如果其中一路找到了正确文档而另一路没有——你有词汇失配（修复：补上缺失的那一路）或语义歧义（修复：更好的嵌入或重排器）。

完整代码见 `code/ir_demo.py`。

---

## 4. 工业工具——2026 技术栈

| 规模 | 技术栈 |
|---|---|
| 1k-10 万文档 | 内存 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。不需要独立数据库 |
| 10 万-1000 万文档 | FAISS 或 pgvector（稠密）+ Elasticsearch/OpenSearch（BM25）。并行运行 |
| 1000 万+文档 | Qdrant/Weaviate/Vespa/Milvus（混合支持）。Cross-encoder 重排 top-30 |
| 最高质量前沿 | 三路（BM25 + 稠密 + SPLADE）+ ColBERT 后交互重排 |

无论选择什么，为评估留出预算。**在基准测试端到端 RAG 准确率之前先基准测试检索召回率。** 生成器无法修复检索器漏掉的东西。

### 4.1 2026 年生产 RAG 的教训

- **80% 的 RAG 失败可以追溯到摄入和分块，不是模型。** 团队花几个星期换 LLM 和调 Prompt，而检索器悄悄地在每三个查询中返回一次错误的上下文。先修分块
- **分块策略比块大小更重要。** 固定大小分割破坏表格、代码和嵌套标题。句子感知是默认选择；对技术文档和产品手册，语义或 LLM 分块效果更好
- **父文档模式。** 检索小的"子"块以提高精确率。当多个来自同一父段落的子块出现时，换入父块保留上下文。这持续提升回答质量，无需重新训练
- **k_rerank=3 通常最优。** 每多一个块都增加 token 成本和生成延迟，却不提升回答质量。如果你 k=8 仍优于 k=3——重排器表现不佳
- **HyDE / 查询扩展。** 从查询生成假设答案，嵌入它，用它检索。弥合短问题和长文档之间的措辞差距。零训练成本提升精确率
- **上下文预算 < 8K token。** 持续命中这个限制说明重排器阈值太松
- **版本控制一切。** Prompt、分块规则、嵌入模型、重排器。任何漂移都会静默破坏回答质量。CI 门禁在忠实度、上下文精确率和未回答率上阻断回归
- **三路检索（BM25 + 稠密 + SPLADE）在 2026 基准上优于两路**，尤其是在混合专有名词和语义的查询上。当基础设施支持 SPLADE 索引时部署

正确的检索设计根据 2026 年行业测量减少 70-90% 的幻觉。大多数 RAG 性能提升来自更好的检索，而非模型微调。

### 4.2 中文检索特别建议

- **中文 BM25 的 tokenizer 需要适配。** 默认英文分词器（`r"[a-z0-9]+"`）对中文完全无效——所有中文字符被丢弃。中文 BM25 需要逐字切分或 jieba 分词——推荐在索引侧用 jieba 分词，查询侧同样——保证训练/推理一致性
- **中文稠密检索首选 BGE-M3。** 支持稠密+稀疏+多向量三合一。纯中文用 `bge-large-zh-v1.5`。中英混合用 BGE-M3 或 `jina-embeddings-v3`
- **父文档模式对中文技术文档尤其有效。** 中文技术文档（如本项目）天然有 `#`/`##` 标题层级——子块 = 单个标题下的段落，父块 = 整个章节。当同一章节的多个子块被检索到 → 换入父块保留完整上下文

---

## 5. 知识连线

本课的四层检索架构是后续所有 RAG 流水线的检索底座：

- **阶段 05 · 13（问答系统）→** 开放域 QA 的第一步就是本课的检索层——先用 BM25+稠密找到相关段落，再用 reader 抽取或生成答案
- **阶段 05 · 22（嵌入模型深入）→** 本课的稠密检索用的是通用嵌入——阶段 22 将深入 Matryoshka 嵌入、领域微调和维度预算
- **阶段 05 · 23（RAG 分块策略）→** "80% 的 RAG 失败追溯到分块"——本课的父文档模式和句子感知分割是阶段 23 的核心主题

---

## 6. 常见错误

### 错误 1：稠密检索 + BM25 不看重叠度就直接上线

**现象：** 混合检索在某些查询上不如单路 BM25——融合反而降低了排名质量。

**原因：** 稠密检索和 BM25 返回了完全不同的文档集合。如果稠密路在 50% 的查询上返回了噪音结果，RRF 会将这些噪音和 BM25 的好结果混合——导致好结果被排出 top-k。

**修复：** 在上线前，对 50 条测试查询跑 BM25-only、dense-only、hybrid 三组。如果 hybrid 的 Recall@10 不显著优于两者中的最优者——先修差的那一路再上线。

### 错误 2：Query 侧和 Doc 侧的 tokenizer 不一致

**现象：** 中文查询"机器学习"命中率为 0——但"机器"和"学习"在文档中确实存在。

**原因：** 索引侧用了 jieba 分词 → "机器学习"是一个 token。查询侧用了逐字切分 → "机"、"器"、"学"、"习"四个 token。同一个概念在两个不同的表示空间中——BM25 的 IDF 匹配完全失败。

**修复：** 索引侧和查询侧使用完全相同的 tokenize 函数。封装成一个函数，两边调用同一个引用。

---

## 7. 面试考点

### Q1：RRF 为什么只看排名不看分数？（难度：⭐⭐）

**参考答案：**
BM25 分数（0-50+）和余弦相似度（0-1）的绝对值不可比较。试图用 min-max 归一化或 z-score 标准化来对齐——在采样分布外的新查询上失效。RRF 绕开整个分数尺度问题——只问"两路都排前几名的共识文档是谁"。一个在两路都排第 2 的文档 > 一路排第 1、一路排第 20 的文档——因为后者很可能是一路的噪音。

### Q2：什么时候混合检索比单路检索更差？（难度：⭐⭐⭐）

**参考答案：**
当其中一路是系统性噪音源时。例如：你的稠密嵌入模型是在通用英文上训练的，而你的语料库是中文法律文档——那么稠密路对每个查询返回的都是随机噪音。RRF 平均过程中，这些噪音会周期性把好结果从 top-k 中排挤出去。**调试方法：** 取 20 条失败查询，单独看每条查询的 BM25 排序列和稠密排序列——如果一致性的模式是"稠密路永远在拖后腿"→ 关闭稠密路，替换为更好的领域嵌入，再开 hybrid。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| BM25 | "关键词搜索" | Okapi BM25——用词频、IDF 和长度归一化给文档打分。30 年后仍是稀疏检索的默认 |
| 稠密检索 | "向量搜索" | 将查询和文档编码为向量，找最近邻居 |
| 双编码器 | "嵌入模型" | 查询和文档独立编码。推理快 |
| Cross-encoder | "重排模型" | 查询+文档一起编码。慢但准 |
| RRF | "排名融合" | 两个排名合并——只看排名不看分数。k=60 是发表的默认值 |
| Recall@k | "检索指标" | 正确答案存在的查询中，正确文档在前 k 中的比例。RAG 最重要的数字 |

---

## 📚 小结

四层混合检索——BM25 捕捉关键词 → 稠密检索捕捉语义 → RRF 融合两路排名 → Cross-encoder 精排 top-30——是 2026 年每个生产 RAG 系统的检索底座。80% 的 RAG 失败可以追溯到检索和分块，而非生成模型。先基准测试检索召回率，再基准测试端到端生成质量。

---

## ✏️ 练习

1. 【实现】在 500 篇文档的语料库上实现 `hybrid_search`。测试 20 条查询。对比 BM25-only、dense-only 和 hybrid 的 Recall@5。

2. 【实现】加入 MRR 计算。对每条已知正确答案的查询，找出正确文档在三种排名中的位置。报告各路的 MRR。

3. 【实验】用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域上微调稠密编码器。从 500 对查询-文档中构建训练集。对比微调前后的 Recall@10。

4. 【思考】你的 RAG 系统在"产品型号"查询上召回率为 0——但在"如何使用"查询上召回率很高。排查哪一层是瓶颈？给出一个不超过三行代码的修复方案。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| BM25 + RRF + 混合检索 | `code/ir_demo.py` | 从零实现的 BM25 类 + RRF 融合 + 中文 tokenizer 适配 |

---

## 📖 参考资料

1. [论文] Robertson and Zaragoza. "The Probabilistic Relevance Framework: BM25 and Beyond". 2009. https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf — BM25 的权威处理
2. [论文] Karpukhin et al. "Dense Passage Retrieval for Open-Domain QA". EMNLP, 2020. https://arxiv.org/abs/2004.04906 — DPR，经典双编码器框架
3. [论文] Cormack, Clarke, Büttcher. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods". SIGIR, 2009. https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf — RRF 论文
4. [论文] Khattab and Zaharia. "ColBERT: Efficient and Effective Passage Search via Late Interaction". SIGIR, 2020. https://arxiv.org/abs/2004.12832 — 后交互检索

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文检索建议、父文档模式中文适配、工程最佳实践、常见错误、面试考点等均为原创内容。
