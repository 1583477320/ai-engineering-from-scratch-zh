# 综合项目65——混合检索 BM25 与稠密嵌入（Hybrid Retrieval with BM25 and Dense）

> 词汇检索和语义检索在不同查询分布上失败。混合检索使用倒数秩融合——不是插值，是投票——在每个查询类别上都赢。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第64节
**预计时间：** 90分钟

---

## 学习目标

- 从零实现带字段权重、文档长度归一化的 BM25
- 构建基于确定性模拟嵌入的稠密检索器
- 实现 Cormack-Clarke-Buettcher 2009 论文的倒数秩融合（RRF）
- 调优 RRF k 常数和模态权重

---

## 1. 问题

词汇搜索在查询包含字面标识符时获胜——搜索 `AbortMultipartOnFail` 返回正确的 Go 函数。语义搜索在查询被改写时获胜——"如何处理取消上传" 找到提及取消的函数。选择不是静态的——查询分布是变量，生产系统必须同时处理两者。

---

## 2. 核心概念

### 2.1 BM25

```
idf(t) = log((N - df + 0.5) / (df + 0.5) + 1)
tf_norm = (f × (k1+1)) / (f + k1 × (1-b + b × dl/avgdl))
score(d,q) = Σ idf(t) × tf_norm
```

字段权重：索引时重复字段中出现的词元。

### 2.2 稠密检索

嵌入每个块为固定维度向量。查询时嵌入查询，计算余弦排名。算法就是点积+排序。本课使用确定性哈希嵌入。

### 2.3 倒数秩融合（RRF）

```python
def rrf(rankings, k=60, weights=None):
    """rankings: dict of {modality: [(doc_id, rank)]}"""
    scores = {}
    for mod, ranked in rankings.items():
        w = weights.get(mod, 1.0) if weights else 1.0
        for doc_id, rank in ranked:
            scores[doc_id] = scores.get(doc_id, 0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

k=60 是 2009 年论文的默认值。排名越靠前贡献越大，但衰减缓慢——深位候选仍能投票。

### 2.4 RRF 为什么优于分数加权插值

BM25 分数无界且依赖语料库。余弦分数有界在 [-1, 1]。`alpha×bm25 + (1-alpha)×cosine` 需要逐语料库调 alpha。基于秩的融合不需要——两种模态的排名天然可比较。

---

## 3. 从零实现

```python
"""混合检索——BM25 + 稠密嵌入 + RRF。"""
import math, re, hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


def tokenize(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.df = Counter(); self.term_docs = defaultdict(list)
        self.doc_lens = {}; self.avgdl = 0.0; self.N = 0

    def add(self, doc_id, text, field_boost=None):
        tokens = tokenize(text)
        if field_boost:
            tokens = tokens + [t for t in tokenize(field_boost) for _ in range(2)]
        self.doc_lens[doc_id] = len(tokens)
        self.N += 1
        for t, c in Counter(tokens).items():
            self.df[t] += 1
            self.term_docs[t].append((doc_id, c))

    def search(self, query, top_k=5):
        self.avgdl = sum(self.doc_lens.values()) / max(self.N, 1)
        qtokens = tokenize(query)
        scores = {}
        for did in set(did for t in qtokens for did, _ in self.term_docs.get(t, [])):
            dl = self.doc_lens.get(did, 0)
            s = 0.0
            for t in set(qtokens):
                df = self.df.get(t, 0)
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                f = sum(c for d, c in self.term_docs.get(t, []) if d == did)
                tf = (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1)))
                s += idf * tf
            if s > 0: scores[did] = s
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


def mock_embed(text, dim=64):
    h = hashlib.sha256(text.encode()).digest()
    vec = [0.0] * dim
    for i in range(dim):
        byte_idx = i % len(h)
        vec[i] = (h[byte_idx] / 127.5 - 1.0) * ((i * 7 + 13) % 5 / 5.0)
    norm = sum(v**2 for v in vec) ** 0.5 or 1
    return [v / norm for v in vec]


class DenseIndex:
    def __init__(self, dim=64):
        self.dim = dim; self.ids = []; self.vecs = []

    def add(self, doc_id, text):
        self.ids.append(doc_id); self.vecs.append(mock_embed(text, self.dim))

    def search(self, query, top_k=5):
        q = mock_embed(query, self.dim)
        scores = [(self.ids[i], sum(a*b for a, b in zip(q, self.vecs[i])))
                  for i in range(len(self.ids))]
        return sorted(scores, key=lambda x: -x[1])[:top_k]


def rrf(ranked_lists, k=60, weights=None):
    scores = defaultdict(float)
    for mod, ranked in ranked_lists.items():
        w = weights.get(mod, 1.0) if weights else 1.0
        for rank, (did, _) in enumerate(ranked, 1):
            scores[did] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def main():
    bm25 = BM25Index()
    dense = DenseIndex(dim=64)

    corpus = [
        ("doc1", "AbortMultipartOnFail handles upload cancellation gracefully", "Go file upload"),
        ("doc2", "Large file chunking splits uploads into manageable pieces for reliability", "Upload strategy"),
        ("doc3", "The retry policy ensures failed network requests are attempted again", "Network resilience"),
    ]
    for did, text, title in corpus:
        bm25.add(did, text, field_boost=title)
        dense.add(did, text)

    queries = [
        "AbortMultipartOnFail",
        "how do we handle cancelled uploads",
        "network retry logic",
    ]

    print("混合检索对比:")
    for q in queries:
        bm25_rank = bm25.search(q, 3)
        dense_rank = dense.search(q, 3)
        bm25_dict = {"bm25": [(did, i+1) for i, (did, _) in enumerate(bm25_rank)]}
        dense_dict = {"dense": [(did, i+1) for i, (did, _) in enumerate(dense_rank)]}
        both_dict = {"bm25": bm25_dict["bm25"], "dense": dense_dict["dense"]}
        fused = rrf(both_dict)
        print(f"\n  查询: '{q}'")
        print(f"  BM25:   {[(did, f'{s:.3f}') for did, s in bm25_rank[:3]]}")
        print(f"  Dense:  {[(did, f'{s:.3f}') for did, s in dense_rank[:3]]}")
        print(f"  RRF:    {[(did, f'{s:.3f}') for did, s in fused[:3]]}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 系统 | BM25 | 稠密 | RRF | 特点 |
|:----|:----|:-----|:----|:-----|
| Elasticsearch | ✓ | ✓ | 自定义 | 企业标准 |
| Vespa | ✓ | ✓ | ✓ | 可扩展 |
| Weaviate | ✓ | ✓ | ✓ | 云原生 |
| LanceDB | ✓ | ✓ | ✓ | 轻量级 |

---

## 5. 工程最佳实践

- BM25 和稠密检索并行执行，融合是常数时间合并
- RRF k=60 是默认值，除非有基准测试证明其他值更好
- **中文场景建议**：BM25 中文需要先分词；稠密检索直接使用中文句子嵌入

---

## 6. 常见错误

- **分数加权插值而非 RRF**：BM25 分数和余弦分数量纲不同，直接相加无意义
- **RRF k 设置不当**：k 太小 → 只有 Top-1 有效；k 太大 → 所有排名拉平
- **未去重**：BM25 和稠密可能返回同一文档的不同块，需按文档 ID 去重

---

## 7. 面试考点

**Q1：为什么 RRF 优于分数加权融合？**（难度：⭐⭐⭐）

**参考答案：** BM25 分数无界且依赖语料库规模，余弦分数有界在 [-1,1]。直接线性组合需要每个语料库调 α，且重新索引时 α 失效。RRF 基于排名——排名在不同模态间天然可比较，无需对齐量纲，且 k 常数跨语料库稳定。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| RRF | 倒数秩融合：`Σ 1/(k+rank)` 跨排名列表 |
| k=60 | RRF 默认常数——深位候选仍能投票 |
| 模态权重 | 为 BM25 或稠密分配不同投票权重 |

---

## 📚 小结

混合检索通过 RRF 将词汇和语义的优势结合，使检索器在所有查询类别上都赢。你实现了 BM25、稠密检索和 RRF 融合。下一节将对融合结果进行交叉编码重排序。

---

## ✏️ 练习

1. 【实验】将稠密嵌入替换为真实嵌入模型，对比稠密排名变化
2. 【实验】扫描 RRF k 从 10 到 200，观察 recall 变化
3. 【实现】添加第三个模态（摘要索引），三路 RRF 融合

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 混合检索 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Cormack, Clarke, Buettcher. "Reciprocal Rank Fusion". SIGIR 2009.
2. [论文] Robertson et al. "Okapi at TREC-3". BM25 原始论文.
3. [官方文档] Vespa 混合检索. https://docs.vespa.ai/en/tutorials/hybrid-search.html
