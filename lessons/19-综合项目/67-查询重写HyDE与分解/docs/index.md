# 综合项目67——查询重写 HyDE 与分解（Query Rewriting: HyDE, Multi-Query, Decomposition）

> 用户输入的查询不是检索器想要的。重写在检索之前弥合差距。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第64-65节
**预计时间：** 90分钟

---

## 学习目标

- 实现 HyDE：生成假想文档，嵌入后检索
- 实现多查询扩展：将一个查询改写为 N 个改述
- 实现查询分解：将复杂问题拆分为子问题
- 三种策略对比并解释各自适用场景

---

## 1. 问题

用户输入"上传失败且预算耗尽时团队怎么做？"。文档说的是"AbortMultipartOnFail 在上传失败时中止 S3 多部分上传并递减重试预算"。查询和文档不共享名词短语。BM25 遗漏，双编码器排名靠后。

修复方法：在检索前重写查询。

---

## 2. 核心概念

### 2.1 HyDE（假想文档嵌入）

让 LLM 写出一个"回答问题的假想文档"，嵌入该文档，用其嵌入向量替代查询向量检索。假想文档包含领域词汇——检索器关心的是词元分布而非事实正确性。

### 2.2 多查询扩展

将查询改写为 N 个改述，每个检索 Top-K，用 RRF 合并。覆盖查询的改述方差。

### 2.3 查询分解

将多主题查询拆为子问题，每个子问题独立检索，结果合并。

### 2.4 三种策略互补

| 策略 | 解决 | 瓶颈 |
|:----|:-----|:-----|
| HyDE | 词元分布不匹配 | LLM 幻觉 |
| 多查询 | 改述方差 | 所有改述收敛 |
| 分解 | 多主题 | 过度拆分 |

---

## 3. 从零实现

```python
"""查询重写——HyDE + 多查询 + 分解。"""
import re, hashlib
from collections import defaultdict


class MockLLM:
    def __init__(self):
        self.hyp = {
            "upload fail budget": "AbortMultipartOnFail handles upload cancellation and decrements retry budget.",
            "how to handle cancelled uploads": "The upload cancellation handler aborts multipart transfers.",
            "network retry": "Failed network requests are retried with exponential backoff.",
        }
        self.para = {
            "upload fail budget": ["upload failure handling", "budget exhausted upload", "cancel upload budget"],
            "how to handle cancelled uploads": ["upload cancel handler", "abort multipart upload", "failed upload retry"],
            "network retry": ["request retry policy", "network failure recovery", "exponential backoff retry"],
        }
        self.decomp = {
            "upload fail budget": ["what happens when upload fails", "what happens when budget is gone"],
            "how to handle cancelled uploads": ["upload cancel handler"],
            "network retry": ["retry policy for failed requests"],
        }

    def get_hypothetical(self, q): return self.hyp.get(q, f"Documentation about {q}")
    def get_paraphrases(self, q, n=3): return self.para.get(q, [f"{q} v{i}" for i in range(n)])
    def get_decomposition(self, q): return self.decomp.get(q, [q])


def mock_embed(text, dim=16):
    h = hashlib.sha256(text.encode()).digest()
    v = [(h[i % len(h)] / 127.5 - 1.0) for i in range(dim)]
    n = sum(x**2 for x in v) ** 0.5 or 1
    return [x / n for x in v]


def cosine_sim(a, b):
    return sum(x*y for x, y in zip(a, b))


def retrieve(query, corpus, corpus_embeds, top_k=3):
    qe = mock_embed(query)
    scores = [(i, cosine_sim(qe, ce)) for i, ce in enumerate(corpus_embeds)]
    scores.sort(key=lambda x: -x[1])
    return [(corpus[i], s) for i, s in scores[:top_k]]


def rrf(ranked_lists, k=60):
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc, _) in enumerate(ranked, 1):
            scores[doc] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hyde_retrieve(llm, query, corpus, corpus_embeds, top_k=3):
    hyp = llm.get_hypothetical(query)
    return retrieve(hyp, corpus, corpus_embeds, top_k)


def multi_query_retrieve(llm, query, corpus, corpus_embeds, top_k=3, n=3):
    paras = llm.get_paraphrases(query, n)
    results = [retrieve(p, corpus, corpus_embeds, top_k) for p in paras]
    return rrf(results)[:top_k]


def decompose_retrieve(llm, query, corpus, corpus_embeds, top_k=3):
    subs = llm.get_decomposition(query)
    results = [retrieve(s, corpus, corpus_embeds, top_k) for s in subs]
    return rrf(results)[:top_k]


def main():
    llm = MockLLM()
    corpus = [
        "AbortMultipartOnFail aborts multipart upload and decrements budget.",
        "Large file chunking splits uploads for reliability.",
        "Retry policy retries failed network requests with backoff.",
    ]
    corpus_embeds = [mock_embed(d) for d in corpus]

    queries = ["upload fail budget", "how to handle cancelled uploads", "network retry"]
    for q in queries:
        print(f"\n查询: '{q}'")
        hyde = hyde_retrieve(llm, q, corpus, corpus_embeds, 2)
        mq = multi_query_retrieve(llm, q, corpus, corpus_embeds, 2)
        dc = decompose_retrieve(llm, q, corpus, corpus_embeds, 2)
        print(f"  HyDE:     {[d[:50] for d,_ in hyde]}")
        print(f"  多查询:   {[d[:50] for d,_ in mq]}")
        print(f"  分解:     {[d[:50] for d,_ in dc]}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 策略 | 工具 | 特点 |
|:----|:-----|:-----|
| HyDE | LlamaIndex HyDE | 标准实现 |
| 多查询 | GraphRAG | 微软 GraphRAG |
| 分解 | DSPy | 斯坦福子查询分解 |

---

## 5. 工程最佳实践

- 三种策略互补——生产系统并行运行，RRF 融合
- 假想文档限制在 2-3 句，太长收集噪音
- **中文场景建议**：HyDE 需要中文领域词汇覆盖，多查询扩展效果好

---

## 6. 常见错误

- **HyDE 幻觉标识符**：假想文档中的错误函数名导致 BM25 分数崩溃
- **多查询改述收敛**：弱模型产生近似改述，RRF 合并无效果
- **分解过度拆分**：原子问题被拆为多个子问题，排名下降

---

## 7. 面试考点

**Q1：HyDE 为什么有效？**（难度：⭐⭐）

**参考答案：** HyDE 将查询向量替换为假想文档向量——假想文档使用领域词汇，其嵌入在语义空间中落在真实文档附近。检索器不关心事实正确性，只关心词元分布的相似性。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| HyDE | LLM 写假想文档，嵌入后替代查询向量 |
| 多查询 | N 个改述，RRF 融合 |
| 查询分解 | 多主题查询拆为子问题，独立检索 |
| 原子查询 | 不可再拆的单主题查询 |

---

## 📚 小结

查询重写在检索前改善查询质量。三种策略分别解决词元分布、改述方差和多主题问题。下一节构建 RAG 全链路评估。

---

## ✏️ 练习

1. 【实现】实现"回退提示"：先问更通用的问题检索，再缩小
2. 【实验】对比有/无查询重写的检索 recall@K

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 查询重写 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE). 2023.
2. [官方文档] LlamaIndex 查询变换. https://docs.llamaindex.ai/
