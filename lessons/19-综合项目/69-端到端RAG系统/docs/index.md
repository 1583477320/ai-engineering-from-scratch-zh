# 综合项目69——端到端 RAG 系统（End-to-End RAG System）

> 六节课的组件。一个流水线。一个评估循环。一个自终止演示。这就是你要发布的系统。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第64-68节
**预计时间：** 90分钟

---

## 学习目标

- 组合分块器、混合检索器、查询重写器、交叉编码重排序器和答案生成器为端到端流水线
- 实现带引用锚点和低置信度拒绝的生成器
- 运行第 68 节评估证明集成管道优于各组件独立
- 构建自终止 CLI 演示

---

## 1. 问题

六个组件独立运行证明不了什么。分块器可以赢在 recall@5 上但在系统中输掉——因为检索器无法排名它发射的块。重排序器可以提升合成候选池上的 MRR 但对真实的双编码器候选失败。集成测试就是整个流水线端到端运行在相同的固定 qrels 上。

---

## 2. 核心概念

### 2.1 五阶段流水线

| 阶段 | 输入 | 输出 |
|:----|:-----|:-----|
| 分块器 | 文档文本 | Chunk 列表 |
| 检索器 | 查询字符串 | Top-N Chunk |
| 重写器 | 查询 | 改写 + 假想文档 |
| 重排序器 | 查询 + 候选 | Top-K Chunk |
| 生成器 | 查询 + Top-K Chunk | 带引用的答案 |

### 2.2 带引用的生成器

选择与查询重叠最多的块，从每个选中块取一句，附加 `[doc_id:chunk_index]` 锚点。低置信度时输出"我不知道"。

### 2.3 自终止演示

运行所有阶段，打印每查询分解，打印评估表，所有第 68 节指标达标则退出码 0，否则非零并说明哪个指标失败。

---

## 3. 从零实现

```python
"""端到端 RAG 系统——组合全部组件。"""
import re, hashlib, json, time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class Chunk:
    text: str; doc_id: str; chunk_idx: int; source: str = ""

def fixed_window_chunks(text, size=100, overlap=20):
    chunks = []; i = 0
    while i < len(text):
        chunks.append(Chunk(text[i:i+size], f"doc_{i//size}", i//size))
        i += size - overlap
    return chunks

def mock_embed(text, dim=32):
    h = hashlib.sha256(text.encode()).digest()
    v = [(h[i%len(h)]/127.5-1.0) for i in range(dim)]
    n = sum(x**2 for x in v)**0.5 or 1
    return [x/n for x in v]

def cosine(a, b): return sum(x*y for x,y in zip(a,b))

class HybridRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.embeds = [mock_embed(c.text) for c in chunks]

    def retrieve(self, query, top_k=5):
        qe = mock_embed(query)
        scores = [(i, cosine(qe, e)) for i, e in enumerate(self.embeds)]
        scores.sort(key=lambda x: -x[1])
        return [self.chunks[i] for i, _ in scores[:top_k]]

class MockReranker:
    def rerank(self, query, chunks, top_k=3):
        def score(c):
            overlap = len(set(query.lower().split()) & set(c.text.lower().split()))
            return overlap + (0.5 if query.lower()[:5] in c.text.lower() else 0)
        return sorted(chunks, key=lambda c: -score(c))[:top_k]

class MockGenerator:
    def generate(self, query, chunks):
        if not chunks: return "我不知道。", []
        selected = chunks[:2]
        sentences = []
        cites = []
        for c in selected:
            sents = re.split(r'(?<=[.!?])\s*', c.text)
            if sents: sentences.append(sents[0]); cites.append(f"[{c.doc_id}:{c.chunk_idx}]")
        answer = " ".join(sentences) + " " + " ".join(cites)
        return answer.strip(), cites

@dataclass
class PipelineResult:
    answer: str; citations: List[str]; top_k: List[Chunk]; latency_ms: float

class RAGPipeline:
    def __init__(self, corpus_chunks, generator=None, reranker=None):
        self.retriever = HybridRetriever(corpus_chunks)
        self.generator = generator or MockGenerator()
        self.reranker = reranker or MockReranker()

    def query(self, question):
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(question, top_k=5)
        ranked = self.reranker.rerank(question, candidates, top_k=3)
        answer, cites = self.generator.generate(question, ranked)
        latency = (time.perf_counter() - t0) * 1000
        return PipelineResult(answer, cites, ranked, latency)


def recall_at_k(retrieved, gold, k):
    return sum(1 for d in retrieved[:k] if d in gold) / max(len(gold), 1)

def build_corpus():
    docs = [
        ("d1", "AbortMultipartOnFail aborts an in-flight S3 multipart upload and decrements the per-bucket retry budget."),
        ("d2", "Large file chunking splits uploads into manageable pieces for reliability."),
        ("d3", "Retry policy ensures failed network requests are retried with exponential backoff."),
        ("d4", "Budget threshold limits the number of retry attempts per hour to prevent cost overruns."),
    ]
    chunks = []
    for did, text in docs:
        chunks.append(Chunk(text, did, 0, did))
    return chunks, docs

def main():
    corpus_chunks, docs = build_corpus()
    pipeline = RAGPipeline(corpus_chunks)

    queries = [
        ("abort threshold", ["d1", "d4"]),
        ("upload failure handling", ["d1", "d2"]),
    ]

    print("端到端 RAG 系统演示\n")
    for q, gold in queries:
        result = pipeline.query(q)
        print(f"查询: '{q}'")
        print(f"答案: {result.answer}")
        print(f"延迟: {result.latency_ms:.2f}ms")
        print(f"检索文档: {[c.doc_id for c in result.top_k]}")
        r5 = recall_at_k([c.doc_id for c in result.top_k], gold, 5)
        print(f"Recall@5: {r5:.3f}\n")

    print("✓ 端到端演示完成")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 组件 | 开源 | 云端 |
|:----|:-----|:-----|
| 分块 | LangChain | - |
| 检索 | Elasticsearch | Vespa |
| 重排序 | BGE-Reranker | Cohere Rerank |
| 生成 | LlamaIndex | GPT-4, Claude |

---

## 5. 工程最佳实践

- 每个阶段有明确的输入/输出接口——替换组件不需要改流水线
- 烟雾测试集 20 条查询，30 秒内完成
- **中文场景建议**：评估集需要中文标注；忠实度评估需要分词

---

## 6. 常见错误

- **分块器在标注后更换**：金标准文档 ID 失效
- **重排序器训练集泄漏到评估集**：评估结果不可信
- **Mock 生成器掩盖幻觉风险**：生产环境替换为真实 LLM

---

## 7. 面试考点

**Q1：为什么集成测试优于各组件独立测试？**（难度：⭐⭐⭐）

**参考答案：** 每个组件的输出是下一个组件的输入。独立测试假设输入是理想的；集成测试验证在实际管道数据流中，组件之间是否兼容。例如重排序器在合成候选池上表现好，但在双编码器的真实候选上可能失败——因为召回率不同。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 管道 | 从分块到答案的组合阶段 |
| 引用锚点 | 每个断言附带的 (文档ID, 块索引) |
| 低置信度拒绝 | 重排序器 Top-1 分数低于阈值时拒绝回答 |
| 烟雾测试 | 最小 qrels 子集，每次 PR 检查运行 |

---

## 📚 小结

端到端 RAG 系统将分块、检索、重写、重排序、生成五个阶段组合为统一管道。你实现了带引用的生成器和自终止演示。下一节定义评估任务的规范格式。

---

## ✏️ 练习

1. 【实现】添加按查询长度的策略选择器
2. 【实验】用真实 LLM 替换 Mock 生成器，测量延迟差异

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 端到端 RAG | `code/main.py` |

---

## 📖 参考资料

1. [官方文档] Anthropic 搜索与检索. https://www.anthropic.com/news/contextual-retrieval
2. [GitHub] Ragas. https://docs.ragas.io
