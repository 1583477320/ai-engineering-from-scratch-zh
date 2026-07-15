# RAG——检索增强生成

> 模型的知识在训练完成时就冻结了。你的业务数据在持续变化。RAG 用检索桥接这个差距——从外部知识库检索相关内容，塞进上下文窗口，让模型"看到"它没训练过的新数据。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 11 · 04（嵌入与向量搜索）| **时间：** ~90 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现标准 RAG 管道——嵌入文档 + 向量搜索 + LLM 生成
- [ ] 解释分块策略如何影响检索质量——块大小和重叠的权衡
- [ ] 理解检索结果排序和过滤——用元数据过滤提高相关性
- [ ] 对比不同 RAG 架构——Naive RAG vs 高级 RAG vs 模块化 RAG

---

## 1. 问题

你有一个公司内部知识库——2000 个文档，包含产品手册、API 文档、常见问题。你希望 LLM 可以回答"退货政策是什么"或"怎么配置 API"。

三种方案：
1. **不投喂**：模型不知道你公司的文档 → 不知道
2. **全量投喂**：2000 个文档塞不进上下文窗口
3. **RAG**：检索相关文档 → 放入上下文 → 模型基于检索内容回答 ✓

RAG（检索增强生成）= 检索（Retrieval）+ 增强（Augmented）+ 生成（Generation）

---

## 2. 概念

### 2.1 标准 RAG 管道

```
用户查询
    ↓
[查询嵌入] → [向量搜索] → 找到 Top-K 相关文档块
    ↓
[检索结果 + 查询] → 构建提示词
    ↓
[LLM 生成] → 基于检索内容的回答
```

### 2.2 分块策略

| 策略 | 块大小 | 重叠 | 适用场景 |
|------|--------|------|---------|
| 固定长度 | 256-512 token | 10-50 | 通用 |
| 段落分割 | 按段落 | 0 | 结构化文本 |
| 递归分割 | 多层降级 | 10-50 | 通用（推荐） |
| 语义分割 | 语义边界 | 0 | 复杂文档 |

### 2.3 RAG 架构演进

| 阶段 | 架构 | 说明 |
|------|------|------|
| Naive RAG | 查询→检索→生成 | 一次检索，一次生成 |
| 高级 RAG | 查询重写+检索+重排序+生成 | 检索前重写查询，检索后重排序 |
| 模块化 RAG | 多检索+路由+融合 | 多重搜索策略，路由到不同知识源 |

### 2.4 检索质量指标

| 指标 | 含义 | 目标 |
|------|------|------|
| 命中率 (Hit Rate) | 相关文档在前 K 个中的比例 | > 90% |
| MRR (Mean Reciprocal Rank) | 第一个相关文档的排名倒数 | > 0.8 |
| NDCG | 排序质量的归一化折扣累加 | > 0.8 |

---

## 3. 从零实现

### Step 1：文档嵌入和索引

```python
import numpy as np

class SimpleVectorStore:
    """简单向量存储。"""
    def __init__(self):
        self.docs = []
        self.vectors = []
    
    def add_document(self, text, vector):
        self.docs.append(text)
        self.vectors.append(vector)
    
    def search(self, query_vec, top_k=5):
        """余弦相似度搜索。"""
        scores = []
        for doc_vec in self.vectors:
            sim = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-10)
            scores.append(sim)
        ranked = sorted(zip(self.docs, scores), key=lambda x: -x[1])
        return ranked[:top_k]
```

### Step 2：RAG 管道

```python
def rag_pipeline(query, vector_store, embed_fn, llm_fn, top_k=3):
    """RAG 管道：检索 → 增强 → 生成。"""
    # 1. 嵌入查询
    query_vec = embed_fn(query)
    
    # 2. 检索相关文档
    retrieved = vector_store.search(query_vec, top_k=top_k)
    context = "\n".join([doc for doc, _ in retrieved])
    
    # 3. 构建增强提示词
    prompt = f"""基于以下文档回答用户的问题。如果你找不到答案，直接说\"我没有足够的信息\"。

文档：
{context}

问题：{query}
回答："""
    
    # 4. 生成回答
    answer = llm_fn(prompt)
    return answer, retrieved
```

### Step 3：查询重写（高级 RAG）

```python
def rewrite_query(original_query, llm_fn):
    """将用户查询重写为更利于检索的形式。"""
    prompt = f"""将以下用户问题重写为检索友好的形式。
原始问题：{original_query}
检索查询："""
    return llm_fn(prompt)
```

---

## 4. 工具

### 4.1 LangChain RAG

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 加载、分块、嵌入
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.create_documents([text])
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# RAG 链
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
result = qa.invoke("退货政策是什么？")
```

### 4.2 LlamaIndex

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("退货政策是什么？")
```

### 4.3 工具对比

| 工具 | 特点 | 适用场景 |
|------|------|---------|
| LangChain | 模块化，生态大 | RAG 通用 |
| LlamaIndex | 专用 RAG 框架 | 复杂文档索引 |
| Chroma | 轻量级向量数据库 | 快速原型 |
| Cohere Rerank | 重排序器 | 提高检索精度 |

---

## 6. 工程最佳实践

### 6.1 提高检索质量

- **元数据过滤**：按日期、部门、标签过滤减少噪声
- **重排序**：检索后加一个重排序器 (Reranker) 提升 Top-K 精度
- **Hybrid 搜索**：向量搜索 + 关键词搜索互补

### 6.2 中文场景建议

- 中文文档分块推荐按段落或句子分割
- 使用 bge-large-zh-v1.5 做中文嵌入
- 中文查询重写用中文 LLM

### 6.3 踩坑经验

- **检索不到**：分块大小不合适、嵌入模型选错 → 调整分块和嵌入
- **检索到但回答不用**：提示词中检索内容优先级低 → 提示词强调"基于文档"
- **幻觉仍然存在**：检索内容质量差或不相关 → 提高检索精度

---

## 7. 面试考点

### Q1：RAG 与微调的本质区别是什么？（难度：⭐⭐）

**参考答案：**
微调将训练数据中的知识编码进模型权重——需要重新训练，知识永久嵌入。RAG 不修改模型权重——基于外部检索内容动态生成回答。知识更新时微调需要重新训练，RAG 只需要更新索引文档。RAG 更适合需要频繁更新或大量异构知识的场景，微调更适合需要改变模型行为（风格、格式、角色）的场景。

### Q2：Naive RAG 在哪里最容易失败？如何修复？（难度：⭐⭐⭐）

**参考答案：**
三个常见失败点：(1) 检索失败——查询和文档语义不匹配。修复：查询重写（用 LLM 将查询译为检索友好形式）；(2) 上下文过多——检索返回了大量噪声文档。修复：重排序器（Reranker）过滤低质量结果；(3) LLM 忽视检索内容——模型依赖自己的知识而非检索结果。修复：强化提示词"基于文档回答"，并用 logit bias 技术增强。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| RAG | "检索+生成" | 从外部知识库检索相关内容，配合 LLM 生成回答 |
| 分块 (Chunking) | "切文档" | 将长文档切分为适合嵌入和检索的小块 |
| 向量搜索 | "按语义找" | 用嵌入向量在向量数据库中做相似度搜索 |
| 重排序 (Reranking) | "重新排序" | 在首次检索后用更精确的模型重新评估相关性 |

---

## 📚 小结

RAG = 检索 + 增强 + 生成。标准 RAG 管道：查询嵌入 → 向量搜索 → 构建提示词 → LLM 生成。分块大小和策略直接影响检索质量。高级 RAG 在检索前重写查询，检索后重排序。RAG 优于微调——不需训练，知识随时更新。

---

## ✏️ 练习

1. **【实现】** 用 Chroma + OpenAI Embeddings + GPT-4o 构建一个完整 RAG 管道——问答公司知识库
2. **【实验】** 对比 256 vs 512 vs 1024 块大小时的检索准确率

---

## 📖 参考资料

1. [论文] Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". NeurIPS, 2020. https://arxiv.org/abs/2005.11401
2. [官方文档] LangChain RAG: https://python.langchain.com/docs/use_cases/question_answering/
3. [GitHub] LlamaIndex: https://github.com/run-llama/llama_index
