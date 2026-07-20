# 综合项目51——文献检索（Literature Retrieval）

> 假设是廉价的。知道是否有人已经证明过它才是昂贵的部分。构建在运行器启动沙箱之前回答这个问题的检索层。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第20-29节
**预计时间：** 90分钟

---

## 学习目标

- 用下游循环将读取的字段建模小型论文记录
- 仅使用标准库数据结构构建 BM25 索引
- 遍历引用图以获取词汇搜索遗漏的论文
- 通过稳定的论文 ID 去重词汇和图遍历的命中结果
- 将两个模拟外部 API 包装在统一客户端后

---

## 1. 问题

一条假设很廉价。知道是否有人已经证明过它才是昂贵的部分。在运行器启动沙箱执行实验之前，你需要一个能回答这个问题的检索层。

对摘要的关键词搜索返回与查询共享词汇的论文。这覆盖了大部分表面。它遗漏了两种情况。第一种是基础论文使用了不同的词汇——例如，查询"稀疏注意力"遗漏了题为"Transformer 路由中的块选择"的论文。第二种是相关论文是引用已知锚点的后续工作——通过找到锚点并向前遍历比暴力搜索摘要池更高效。

---

## 2. 核心概念

### 2.1 两轮检索

课程构建两种检索方法。基于 BM25 的摘要搜索捕获词汇命中。引用图遍历将种子集向前和向后扩展一到两跳。结果按论文 ID 去重并通过一个组合分数排序。

```
论文记录：
  id           : str     (稳定标识符)
  title        : str
  abstract     : str
  year         : int
  authors      : list[str]
  references   : list[str]  (引用的论文 ID)
  citations    : list[str]  (引用本论文的论文 ID)
  source       : str        ("arxiv" 或 "s2")
```

### 2.2 BM25 从零实现

Okapi BM25 算法，默认参数 `k1=1.5`，`b=0.75`。索引是两个字典：`词元→文档频率` 和 `词元→(doc_id, 词元计数)列表`。文档长度是摘要的词元数。平均文档长度在索引构建时计算一次。查询评分是查询词元上 `idf × tf_norm` 的和。

```
idf(t)      = log((N - df + 0.5) / (df + 0.5) + 1.0)
tf_norm(t)  = (f × (k1 + 1)) / (f + k1 × (1 - b + b × dl / avgdl))
score(d, q) = 对 q 中所有 t 求和 idf(t) × tf_norm(t)
```

### 2.3 引用图遍历

图构建自语料库。前向边从论文指向其引文。后向边从论文指向其被引。遍历是广度优先搜索，种子由 Top BM25 命中结果提供，上限为两跳。

两跳是刻意的天花板。一跳太浅，智能体通常需要直接祖先或后代。三跳在连通图上会使结果规模爆炸且容易偏离主题。

### 2.4 去重与排序

两轮检索返回重叠集合。按论文 ID 合并。最终分数是加权混合：

```
final_score = w_bm25 × bm25_score_norm
            + w_graph × graph_score
            + w_recency × recency_score
```

默认权重为 `0.5`，`0.3`，`0.2`。

---

## 3. 从零实现

```python
"""文献检索——BM25 + 引用图遍历 + 去重排序。"""
from __future__ import annotations
import json, math, re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Paper:
    id: str; title: str; abstract: str; year: int
    authors: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    source: str = ""


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1; self.b = b
        self.doc_freq: Dict[str, int] = Counter()
        self.term_docs: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.doc_lens: Dict[str, int] = {}
        self.avgdl = 0.0; self.N = 0

    def build(self, papers: List[Paper]):
        for p in papers:
            tokens = tokenize(p.abstract)
            self.doc_lens[p.id] = len(tokens)
            counts = Counter(tokens)
            for t, c in counts.items():
                self.doc_freq[t] += 1
                self.term_docs[t].append((p.id, c))
        self.N = len(papers)
        self.avgdl = sum(self.doc_lens.values()) / max(self.N, 1)

    def score(self, query: str, doc_id: str) -> float:
        qtokens = tokenize(query)
        dl = self.doc_lens.get(doc_id, 0)
        if not qtokens or dl == 0:
            return 0.0
        total = 0.0
        tf_counts = {t: 0 for t in qtokens}
        for t in qtokens:
            for did, c in self.term_docs.get(t, []):
                if did == doc_id:
                    tf_counts[t] = c
                    break
        for t in set(qtokens):
            df = self.doc_freq.get(t, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            f = tf_counts[t]
            tf_norm = (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            total += idf * tf_norm
        return total


class CitationGraph:
    def __init__(self, papers: List[Paper]):
        self.fwd: Dict[str, List[str]] = {}
        self.bwd: Dict[str, List[str]] = {}
        for p in papers:
            self.fwd[p.id] = list(p.references)
            self.bwd[p.id] = list(p.citations)

    def bfs(self, seeds: List[str], hops: int = 2) -> Dict[str, int]:
        """从种子集出发，多跳遍历引用图，返回 {doc_id: 距离}。"""
        dist: Dict[str, int] = {s: 0 for s in seeds}
        queue = list(seeds)
        for node in queue:
            d = dist[node]
            if d >= hops: continue
            for nb in self.fwd.get(node, []) + self.bwd.get(node, []):
                if nb not in dist:
                    dist[nb] = d + 1
                    queue.append(nb)
        return dist


def mock_papers() -> List[Paper]:
    topics = {
        "p001": ("Attention Sparsity in Transformers", "We analyze attention sparsity patterns in small transformers and propose top-k routing."),
        "p002": ("Block Selection for Efficient Routing", "This paper introduces block selection for routing in transformer-based models."),
        "p003": ("A Survey of Efficient Attention", "Comprehensive survey of efficient attention mechanisms including sparse attention."),
        "p004": ("Retrieval Augmented Generation Survey", "Survey of retrieval augmented generation methods in NLP."),
        "p005": ("Dense Passage Retrieval", "We propose dense passage retrieval for open-domain question answering."),
    }
    papers = []
    for i, (tid, (title, abstract)) in enumerate(topics.items()):
        year = 2020 + i % 5
        refs = ["p00" + str(j) for j in range(1, i) if i > 1]
        papers.append(Paper(id=tid, title=title, abstract=abstract, year=year, references=refs, citations=[]))
    for p in papers:
        for r in p.references:
            if r in topics:
                ref_paper = next(x for x in papers if x.id == r)
                if p.id not in ref_paper.citations:
                    ref_paper.citations.append(p.id)
    return papers


def main():
    papers = mock_papers()
    bm25 = BM25Index()
    bm25.build(papers)
    graph = CitationGraph(papers)

    query = "sparse attention"
    scores = [(p.id, bm25.score(query, p.id)) for p in papers]
    scores.sort(key=lambda x: -x[1])
    print(f"查询: '{query}'")
    print("BM25 结果:")
    for pid, s in scores:
        if s > 0:
            p = next(x for x in papers if x.id == pid)
            print(f"  {pid}: {s:.4f} — {p.title}")

    seeds = [pid for pid, s in scores if s > 0]
    graph_hits = graph.bfs(seeds, hops=2)
    print("\n引用图扩展:")
    for pid, d in sorted(graph_hits.items(), key=lambda x: x[1]):
        p = next(x for x in papers if x.id == pid)
        label = "种子" if pid in seeds else f"距离{d}"
        print(f"  {pid}: {label} — {p.title}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| BM25 | 基于词频和文档长度的概率检索排序函数 |
| 引用图 | 论文引用关系的向图，支持前向和后向遍历 |
| 两跳限制 | BFS 遍历最多两跳，避免结果爆炸 |
| 去重合并 | 按论文 ID 对词汇和图遍历结果求并集 |
| 检索客户端 | 封装两种检索方法的统一接口 |

---

## 5. 工程最佳实践

- **BM25 参数调优**：k1 控制词频饱和度（通常 1.2-2.0），b 控制文档长度归一化（通常 0.65-0.85）。小语料库使用更小的 k1。
- **引用图需要注意循环引用**：遍历时维护 visited 集合防止死循环。
- **中文场景特别建议**：中文检索需要先进行分词（如 jieba），否则 BM25 的 tokenizer 会将句子切成单字。

---

## 6. 常见错误

- **BM25 的 IDF 为负**：当 `df > N/2` 时对数可能变为负数。修复：使用 `+1.0` 平滑。
- **一跳限制太浅**：种子论文可能不直接相关——允许两跳能发现更多相关论文。
- **未去重**：词汇和图遍历可能返回相同论文，需要在合并时按 ID 去重。

---

## 7. 面试考点

**Q1：BM25 与 TF-IDF 的区别是什么？** BM25 增加了词频饱和度（k1 参数）和文档长度归一化（b 参数），避免长文档天然获得更高分数。

---

## 📖 参考资料

1. [论文] Robertson & Zaragoza. "The Probabilistic Relevance Framework: BM25 and Beyond". 2009.
2. [GitHub] Whoosh — Python BM25 实现. https://github.com/mchaput/whoosh
