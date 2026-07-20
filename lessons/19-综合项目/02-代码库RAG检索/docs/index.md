# 综合项目02——代码库RAG检索（跨仓库语义搜索）

> 2026年每个严肃的工程组织都运行内部代码搜索，它理解含义而不仅是字符串。Sourcegraph Amp、Cursor的代码库问答、Augment的企业图谱、Aider的repomap——都是同样的形式。导入多个仓库，用tree-sitter解析，在函数和类级别分块，混合搜索，重排序，带引用回答。本综合项目要求你构建一个能处理跨10个仓库、200万行代码并能在每次git推送后增量重建索引的系统。

**类型：** 综合项目
**编程语言：** Python（导入管道），TypeScript（API + UI）
**前置知识：** 第5章（NLP基础）、第7章（Transformer）、第11章（LLM工程）、第13章（工具）、第17章（基础设施）
**涉及章节：** P5 · P7 · P11 · P13 · P17
**预计时间：** 30小时

---

## 学习目标

- 构建AST感知的代码导入管道，使用tree-sitter解析函数和类
- 实现混合检索（稠密向量 + BM25）和互惠排名融合
- 实现跨编码器重排序和引用增强的答案合成
- 实现git推送触发的增量重建索引

---

## 1. 问题

到2026年，每个前沿编码智能体都配有代码库检索层，因为仅靠上下文窗口无法解决跨仓库问题。Claude的100万token上下文窗口有帮助，但它不能消除排名检索的需求。原始分块的朴素余弦搜索在被生成代码、单体仓库重复和稀有导入符号上效果很差。

生产答案是AST感知分块上的混合检索（稠密 + BM25），配合重排序器，并有一个符号引用图谱支持。

---

## 2. 核心概念

### 2.1 AST感知导入管道

导入管道使用tree-sitter解析每个文件，提取函数和类节点，并在节点边界处分块，而不是按固定token窗口分块。

每个块有三种表示：
- **稠密嵌入**：Voyage-code-3或nomic-embed-code
- **稀疏BM25词项**：field-weighted（符号名权重4、主体权重1、摘要权重2）
- **自然语言摘要**：LLM生成的一句话摘要，增加第三种可检索模态

### 2.2 混合检索

查询同时触发稠密和BM25搜索，合并top-k，然后将并集交给跨编码器重排序器（Cohere rerank-3或bge-reranker-v2-gemma-2b）。

重排序后的列表交给长上下文合成器（Claude Sonnet 4.7带提示缓存，或自托管Llama 3.3 70B），要求每个声明引用文件和行范围。没有引用的答案被后过滤器拒绝。

### 2.3 增量重建索引

git推送触发差异比较：哪些文件变化了，哪些符号变化了。只有受影响的块重新嵌入。受影响的跨文件符号边（导入、方法调用）被重新计算。索引保持一致性，无需每次提交重新处理200万行代码。

---

## 3. 从零实现

`code/main.py`实现混合检索的两个半部分：稠密向量（基于哈希的确定性伪嵌入）和从零实现的真实BM25。融合和重排序逻辑是核心。

```python
"""代码RAG——AST感知分块 + 混合检索脚手架。

核心架构原语是带有排名融合的混合检索：
两个索引结构（稠密向量、BM25）并行运行，结果通过互惠排名融合合并，
然后重排序器选择最终top-k。

运行：python3 code/main.py
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 分块结构——AST感知的函数级分块
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    repo: str
    path: str
    start_line: int
    end_line: int
    symbol: str
    body: str
    summary: str = ""

    def anchor(self) -> str:
        return f"{self.repo}/{self.path}:{self.start_line}-{self.end_line}"


# 示例语料库
SAMPLE_CORPUS = [
    Chunk("uploader", "services/retry.go", 122, 148, "AbortMultipartOnFail",
          "aborts multipart upload and decrements bucket budget",
          "终止S3分段上传并减少每个桶的重试预算"),
    Chunk("uploader", "config/budgets.yaml", 34, 51, "bucket_budget",
          "per_bucket_budget: 64; backoff_ms: [100, 500, 2500]",
          "声明每个S3桶的重试预算和指数退避计划"),
    Chunk("client", "libs/s3client/multipart.ts", 44, 61, "abortUpload",
          "await s3.abortMultipartUpload({Bucket, Key, UploadId})",
          "客户端S3分段终止带指标检测"),
    Chunk("auth", "services/authz/check.py", 12, 38, "check_permission",
          "def check_permission(user, resource, action): return policy.evaluate(...)",
          "中央授权网关，评估OPA策略"),
    Chunk("catalog", "services/search/query.rs", 200, 240, "rank_fusion",
          "pub fn rank_fusion(dense: Vec<Hit>, sparse: Vec<Hit>) -> Vec<Hit>",
          "稠密和稀疏检索结果的互惠排名融合"),
]


# ---------------------------------------------------------------------------
# 稠密索引——用于脚手架测试的确定性伪嵌入
# ---------------------------------------------------------------------------

def fake_embed(text: str, dim: int = 64) -> list[float]:
    """基于哈希的确定性嵌入，代替Voyage-code-3"""
    vec = [0.0] * dim
    for tok in re.findall(r"\w+", text.lower()):
        h = hash(tok)
        vec[h % dim] += 1.0
        vec[(h >> 8) % dim] += 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class DenseIndex:
    vectors: list[tuple[Chunk, list[float]]] = field(default_factory=list)

    def add(self, chunk: Chunk) -> None:
        text = f"{chunk.symbol}\n{chunk.summary}\n{chunk.body}"
        self.vectors.append((chunk, fake_embed(text)))

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        qv = fake_embed(query)
        scored = [(c, cosine(qv, v)) for c, v in self.vectors]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ---------------------------------------------------------------------------
# BM25——从零实现，真实算法
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    docs: list[Chunk] = field(default_factory=list)
    doc_lens: list[int] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    tf: list[Counter] = field(default_factory=list)
    avgdl: float = 0.0

    def add(self, chunk: Chunk) -> None:
        # field-weighted: symbol x4, summary x2, body x1
        tokens = (tokenize(chunk.symbol) * 4 +
                  tokenize(chunk.summary) * 2 +
                  tokenize(chunk.body))
        counts = Counter(tokens)
        self.docs.append(chunk)
        self.doc_lens.append(len(tokens))
        self.tf.append(counts)
        for term in counts:
            self.df[term] += 1
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens)

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        q_terms = tokenize(query)
        n = len(self.docs)
        scores: list[float] = [0.0] * n
        for term in q_terms:
            if term not in self.df:
                continue
            idf = math.log((n - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1.0)
            for i, counts in enumerate(self.tf):
                if term not in counts:
                    continue
                f = counts[term]
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        ranked = sorted(zip(self.docs, scores), key=lambda x: -x[1])
        return [(c, s) for c, s in ranked[:k] if s > 0]


# ---------------------------------------------------------------------------
# 互惠排名融合——混合检索的合并步骤
# ---------------------------------------------------------------------------

def rrf(dense: list[tuple[Chunk, float]], sparse: list[tuple[Chunk, float]],
        k_rrf: int = 60) -> list[tuple[Chunk, float]]:
    score: dict[str, float] = defaultdict(float)
    by_anchor: dict[str, Chunk] = {}
    for rank, (c, _) in enumerate(dense):
        score[c.anchor()] += 1.0 / (k_rrf + rank + 1)
        by_anchor[c.anchor()] = c
    for rank, (c, _) in enumerate(sparse):
        score[c.anchor()] += 1.0 / (k_rrf + rank + 1)
        by_anchor[c.anchor()] = c
    fused = sorted(score.items(), key=lambda x: -x[1])
    return [(by_anchor[a], s) for a, s in fused]


# ---------------------------------------------------------------------------
# 跨编码器重排序——根据查询-符号重叠度重新评分
# ---------------------------------------------------------------------------

def rerank(query: str, candidates: list[tuple[Chunk, float]],
           top_k: int = 5) -> list[tuple[Chunk, float]]:
    q_toks = set(tokenize(query))
    out: list[tuple[Chunk, float]] = []
    for c, prior in candidates:
        symbol_overlap = len(q_toks & set(tokenize(c.symbol))) * 3
        summary_overlap = len(q_toks & set(tokenize(c.summary)))
        out.append((c, prior + 0.3 * symbol_overlap + 0.1 * summary_overlap))
    out.sort(key=lambda x: -x[1])
    return out[:top_k]


# ---------------------------------------------------------------------------
# 编排器——完整的检索 -> 融合 -> 重排序流程
# ---------------------------------------------------------------------------

def answer(query: str, dense: DenseIndex, bm25: BM25Index) -> dict[str, object]:
    dense_hits = dense.search(query, k=10)
    sparse_hits = bm25.search(query, k=10)
    fused = rrf(dense_hits, sparse_hits)
    top = rerank(query, fused, top_k=5)
    citations = [c.anchor() for c, _ in top]
    return {
        "query": query,
        "dense_top": [c.anchor() for c, _ in dense_hits[:3]],
        "sparse_top": [c.anchor() for c, _ in sparse_hits[:3]],
        "fused_top": [c.anchor() for c, _ in fused[:5]],
        "rerank_top": citations,
    }


def main() -> None:
    dense = DenseIndex()
    bm25 = BM25Index()
    for ch in SAMPLE_CORPUS:
        dense.add(ch)
        bm25.add(ch)

    for q in ("how is S3 multipart abort wired into retry budget",
              "where is authorization centralized",
              "how does rank fusion work"):
        result = answer(q, dense, bm25)
        print(f"Q: {result['query']}")
        print(f"  dense  : {result['dense_top']}")
        print(f"  sparse : {result['sparse_top']}")
        print(f"  fused  : {result['fused_top']}")
        print(f"  rerank : {result['rerank_top']}")
        print()


if __name__ == "__main__":
    main()
```

运行结果：

```
Q: how is S3 multipart abort wired into retry budget
  dense  : ['uploader/services/retry.go:122-148', ...
  sparse : ['uploader/services/retry.go:122-148', ...
  fused  : ['uploader/services/retry.go:122-148', ...
  rerank : ['uploader/services/retry.go:122-148', ...

Q: where is authorization centralized
  dense  : ['auth/services/authz/check.py:12-38', ...
  sparse : ['auth/services/authz/check.py:12-38', ...
  fused  : ['auth/services/authz/check.py:12-38', ...
  rerank : ['auth/services/authz/check.py:12-38', ...

Q: how does rank fusion work
  dense  : ['catalog/services/search/query.rs:200-240', ...
  sparse : ['catalog/services/search/query.rs:200-240', ...
  fused  : ['catalog/services/search/query.rs:200-240', ...
  rerank : ['catalog/services/search/query.rs:200-240', ...
```

---

## 4. 工具实践

**技术栈：**
- 解析：tree-sitter（17种语言语法）
- 稠密嵌入：Voyage-code-3（托管）或nomic-embed-code-v1.5（自托管）
- 稀疏索引：Tantivy（Rust）带BM25F，符号名vs主体field-weighted
- 向量数据库：Qdrant 1.12混合搜索或pgvector + pgvectorscale
- 重排序：Cohere rerank-3或bge-reranker-v2-gemma-2b自托管
- 编排：LlamaIndex Workflows（导入）、LangGraph（查询智能体）
- 符号图谱：Neo4j（托管）或kuzu（嵌入）

---

## 5. LLM视角

**检索增强视角**：混合检索在代码检索场景中特别重要，因为代码本身包含结构信息（符号名、调用关系）和语义信息（功能描述），两者都很关键。

**引用视角**：强制引用让你可以验证模型回答的每一个声明。这是构建可信代码问答系统的关键。

**增量视角**：实时更新的索引是生产系统的关键要求。全量重建索引的成本太高——增量更新是唯一可行方案。

---

## 6. 工程最佳实践

**分块策略**：
- 在AST节点边界分块，不在固定token窗口分块
- 每个块包含符号名、主体和LLM生成的摘要
- 摘要增加第三种可检索模态

**混合检索**：
- 同时运行稠密和稀疏搜索
- 互惠排名融合（RRF）合并结果
- 跨编码器重排序优化排名

**引用强制**：
- 要求每个声明引用文件和行范围
- 后过滤器拒绝无引用的答案
- 提供可点击引用

---

## 7. 常见错误

**错误1：固定token窗口分块**
症状：分块切断函数边界，检索质量差
修复：AST节点边界分块

**错误2：仅使用稠密检索**
症状：无法检索符号名精确匹配
修复：稠密+BM25混合检索

**错误3：不强制引用**
症状：模型生成无法验证的声明
修复：引用后过滤器

---

## 8. 面试考点

**Q1：为什么代码RAG需要AST感知分块？**
考察：对代码结构特殊性的理解

**Q2：混合检索如何工作？稠密和稀疏各有什么优势？**
考察：对检索方法的理解

**Q3：为什么增量重建索引很重要？**
考察：对生产系统要求的理解

**Q4：引用强制如何提高答案可信度？**
考察：对可信AI的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| AST感知分块 | "函数级分割" | 在tree-sitter节点边界切分，而非固定token窗口 |
| 混合检索 | "稠密+稀疏" | BM25和向量搜索并行运行，合并top-k，重排序 |
| 跨编码器重排序 | "第二阶段排名" | 模型对每个（查询，候选）对一起评分，比余弦更准确 |
| 提示缓存 | "缓存系统提示" | 2026年Claude/OpenAI功能，重复前缀token最高折扣90% |
| 符号图谱 | "代码图" | 跨文件和仓库的导入、调用、继承边 |
| 引用可信度 | "有依据的回答率" | 用户可通过点击锚点验证的声明比例 |
| 增量重建索引 | "推送到搜索时间" | 从git推送到可查询的墙钟时间 |

---

## 参考文献

- [Sourcegraph Amp](https://ampcode.com)
- [Sourcegraph Cody RAG架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase)
- [Aider repo-map](https://aider.chat/docs/repomap.html)
- [Augment Code企业图谱](https://www.augmentcode.com)
- [Qdrant混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [Voyage AI代码嵌入](https://docs.voyageai.com/docs/embeddings)
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank)
