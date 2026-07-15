# 高级 RAG 策略

> Naive RAG 是检索 + 生成。高级 RAG 是查询理解 + 智能检索 + 融合重排 + 生成。前者是 MVP，后者是产品。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 11 · 06（RAG）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现查询重写——将用户查询转换为更适合检索的形式
- [ ] 解释混合搜索——向量搜索 + BM25 关键词搜索的融合策略
- [ ] 实现检索后重排序——用交叉编码器提升 Top-K 精度
- [ ] 理解 Self-RAG——模型自主决定何时检索、检索什么

---

## 1. 问题

Naive RAG 的三个主要失败点：(1) 用户查询模糊（"那个东西怎么用？"）；(2) 纯向量搜索丢失精确关键词匹配；(3) 检索到太多噪声文档，LLM 无法筛选。

高级 RAG 针对每个失败点提供解决方案。

---

## 2. 概念

### 2.1 查询重写

```
用户原始查询: "那个 API 怎么配置？"
       ↓
LLM 重写: "如何配置 REST API 的认证 token 和端点"
       ↓
更精确的检索结果
```

### 2.2 混合搜索

| 方法 | 优势 | 劣势 |
|------|------|------|
| 向量搜索 | 语义匹配好 | 精确关键词匹配差 |
| BM25 关键词 | 精确匹配好 | 语义理解差 |
| **混合搜索** | 两者互补 | 需要融合策略 |

融合方法：RRF（Reciprocal Rank Fusion）= 1/(k + rank)

### 2.3 检索后重排序

```
初检: Top-100 文档（向量搜索，快速）
  ↓
重排序: Top-10（交叉编码器，慢但准）
  ↓
最终: Top-3（送入 LLM）
```

### 2.4 Self-RAG

模型自主决定：(1) 需不需要检索？(2) 检索到的内容是否有帮助？(3) 基于检索内容的回答是否可靠？

---

## 3. 从零实现

### Step 1：查询重写

```python
def rewrite_query(original_query, llm_fn):
    """用 LLM 将模糊查询重写为检索友好形式。"""
    prompt = f"""将用户问题重写为向量搜索友好的形式。
原始: {original_query}
重写:"""
    return llm_fn(prompt)
```

### Step 2：RRF 融合

```python
def reciprocal_rank_fusion(result_lists, k=60):
    """RRF 融合多路检索结果。"""
    scores = defaultdict(float)
    for results in result_lists:
        for rank, (doc, _) in enumerate(results, 1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

### Step 3：重排序

```python
def rerank(query, candidates, cross_encoder_fn, top_k=5):
    """用交叉编码器重排序候选文档。"""
    scores = []
    for doc in candidates:
        score = cross_encoder_fn(query, doc)  # 返回相关性分数
        scores.append((doc, score))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]
```

---

## 4. 工具

### 4.1 LlamaIndex Advanced RAG

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
# SubQuestionQueryEngine 自动拆分复杂问题为子问题
```

### 4.2 Cohere Rerank

```python
import cohere
co = cohere.Client("your-api-key")
results = co.rerank(query="退货政策", documents=docs, top_n=5)
```

---

## 6. 工程最佳实践

### 6.1 查询重写策略

- **HyDE**：先让 LLM 生成假设性文档，再用假设性文档做向量搜索
- **Sub-Questions**：将复杂查询拆分为子查询
- **Step-back**：先回答抽象问题，再检索具体细节

### 6.2 中文场景

- 中文查询重写用中文 LLM 效果更好
- 混合搜索对中文特别有效——BM25 处理中文分词，向量处理语义

### 6.3 踩坑经验

- **重排序模型太慢**：用 Cross-Encoder 重排序在 Top-100 中选 Top-10
- **融合权重不对**：RRF 的 k=60 是经验默认值，根据场景调整
- **过度重写**：查询重写不能丢失原始意图

---

## 7. 常见错误

### 错误 1：查询重写后丢失原始意图

**现象：** 用户问"退货流程"，重写为"电商物流操作规范"——答案不相关。

**原因：** 过度重写将用户意图过度抽象化。

**修复：** 在重写提示词中强调"保留核心意图"——"将以下问题重写为检索友好形式，但保留原始问题的核心意图"。

### 错误 2：RRF 融合未调参

**现象：** 混合搜索结果不如纯向量搜索。

**原因：** RRF 的 k=60 是经验值——对某些数据集可能过大或过小。

**修复：** 在验证集上调整 k（范围 10-120），选择使 Hit Rate 最大的 k 值。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| RRF 融合实现 | `code/main.py` | 多路检索 + RRF 融合 + 重排序 |

---

## 8. 面试考点

### Q1：混合搜索为什么比纯向量搜索更好？（难度：⭐⭐）

**参考答案：**
向量搜索擅长语义匹配——"如何查询数据库"能找到"Python SQL 操作"。但它对精确关键词匹配不擅长——如果文档中提到"PostgreSQL"而查询用"MySQL"，向量搜索可能匹配到"MySQL 文档"而忽略关键词差异。BM25 关键词搜索擅长精确匹配但缺乏语义理解。RRF 融合两者——向量搜索的 Top-5 和 BM25 的 Top-5 各自评分，合并排序后取最佳。实践中混合搜索的 Hit Rate 比纯向量搜索提升 10-20%。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 查询重写 | "翻译用户问题" | 用 LLM 将模糊查询转为检索友好形式——提升向量搜索命中率 |
| RRF | "融合多路结果" | Reciprocal Rank Fusion——1/(k+rank) 加权合并多路检索结果 |
| 重排序 (Reranking) | "精排" | 在初检 Top-K 中用更精确模型重新评估相关性——选出最佳 Top-5 |
| Self-RAG | "模型自己决定" | 模型自主判断是否需要检索、检索结果是否有帮助、回答是否可靠 |
| HyDE | "先猜再搜" | 先让 LLM 生成假设性回答文档，再用该文档做向量搜索 |

---

## 📚 小结

高级 RAG 在 Naive RAG 基础上加入查询重写、混合搜索、检索后重排序。查询重写提升检索质量，RRF 融合向量搜索和关键词搜索，交叉编码器重排序选出最佳文档。Self-RAG 让模型自主决定检索时机。

---

## ✏️ 练习

1. **【实现】** 实现 RRF 融合——在 100 个文档上对比纯向量 vs 混合搜索的 Hit Rate
2. **【实验】** 对比有/无查询重写时的检索质量差异

---

## 📖 参考资料

1. [论文] Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". NeurIPS, 2020.
2. [论文] Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection". ICLR, 2024. https://arxiv.org/abs/2310.11511
3. [文档] LlamaIndex Advanced RAG: https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/
