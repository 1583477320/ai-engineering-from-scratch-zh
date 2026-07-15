# 嵌入与向量搜索

> 嵌入将文本转换为向量。向量搜索找到最相似的嵌入。这是 RAG、推荐系统、语义搜索的基础——没有好的嵌入，就没有好的检索，也没有好的生成。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01-04（分词器/数据管道）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释嵌入的数学原理——文本如何被编码为固定长度向量
- [ ] 比较不同嵌入模型的维度、速度、质量权衡
- [ ] 实现向量相似度搜索——余弦相似度和向量数据库
- [ ] 理解语义搜索与关键词搜索的区别

---

## 1. 问题

你想搜索"如何用 Python 连接数据库"。传统搜索用关键词匹配——必须包含"python"和"数据库"才能命中。但用户可能说"如何查询 SQL 表"——语义相同但关键词不同。

嵌入解决了这个问题：将文本转换为向量，用向量距离度量语义相似度。

```
"如何用 Python 查询数据库" → [0.12, 0.34, ..., 0.56]  (768维向量)
"Python SQL 查询"          → [0.11, 0.33, ..., 0.55]  (768维向量)
余弦相似度 ≈ 0.95 → 语义匹配
```

---

## 2. 概念

### 2.1 嵌入是什么

嵌入是一个函数 f(text) → ℝ^d。它将任意长度的文本编码为固定长度的向量。核心属性：**语义相似的文本产生距离近的向量**。

### 2.2 常用嵌入模型

| 模型 | 维度 | 速度 | 质量 | 适用场景 |
|------|------|------|------|---------|
| text-embedding-3-small | 1536 | 快 | 高 | 通用（推荐） |
| text-embedding-3-large | 3072 | 中 | 最高 | 高精度场景 |
| BGE-large | 1024 | 快 | 高 | 多语言（推荐） |
| E5-small | 384 | 极快 | 中 | 边缘/低延迟 |
| GTE-base | 768 | 快 | 高 | 开源替代 |

### 2.3 相似度度量

| 方法 | 公式 | 适用场景 |
|------|------|---------|
| **余弦相似度** | cos(a,b) = a·b / (‖a‖‖b‖) | 推荐（默认） |
| **欧氏距离** | d(a,b) = ‖a-b‖ | 归一化向量等价 |
| **内积** | a·b | 当嵌入已归一化时等价余弦 |

### 2.4 向量数据库

| 数据库 | 特点 | 适用场景 |
|--------|------|---------|
| FAISS | Facebook 开源，简单高效 | 小到中等规模 |
| Pinecone | 全托管，云服务 | 生产环境 |
| Weaviate | 开源，支持混合搜索 | 需要元数据过滤 |
| Chroma | 轻量级，嵌入式 | 原型/小项目 |
| Milvus | 开源，分布式 | 大规模企业 |

---

## 3. 从零实现

### Step 1：简单嵌入（词袋）

```python
import numpy as np
from collections import Counter

def bag_of_words_embedding(text, vocab):
    """最简单的嵌入：词频向量。"""
    words = text.lower().split()
    counts = Counter(words)
    vector = np.array([counts.get(w, 0) for w in vocab], dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-10)
```

### Step 2：余弦相似度搜索

```python
def cosine_similarity(a, b):
    """计算两个向量的余弦相似度。"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def search(query_vec, doc_vecs, top_k=5):
    """向量搜索——返回最相似的 top_k 个文档。"""
    scores = [cosine_similarity(query_vec, dv) for dv in doc_vecs]
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    return ranked[:top_k]
```

---

## 4. 工具

### 4.1 OpenAI Embeddings

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    input="如何用 Python 连接数据库",
    model="text-embedding-3-small",
)
vector = response.data[0].embedding
print(f"维度: {len(vector)}")  # 1536
```

### 4.2 HuggingFace sentence-transformers

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
embeddings = model.encode(["中文查询", "测试文本"])
```

---

## 6. 工程最佳实践

### 6.1 嵌入选型

- **多语言**：BGE-large-zh-v1.5（中文最佳）
- **英文**：text-embedding-3-small（OpenAI 价格合理）
- **开源**：GTE-base（阿里，质量高）

### 6.2 踩坑经验

- **维度不匹配**：查询向量和文档向量必须用同一个模型生成
- **嵌入前未归一化**：某些模型需要手动归一化，某些自动处理
- **文档太长**：嵌入有 token 上限（如 8192），超长文档需分块嵌入

---

## 7. 面试考点

### Q1：为什么余弦相似度适合文本搜索？（难度：⭐⭐）

**参考答案：**
余弦相似度衡量两个向量的方向相似性而非大小。在文本嵌入中，语义由向量方向编码，而非大小。两段语义相同的文本有相似方向但可能长度不同——余弦相似度忽略长度因素，只关注语义方向。另外，余弦相似度对缩放不变，使得不同长度的文档可以公平比较。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 嵌入 (Embedding) | "文本的数字表示" | 将文本编码为固定长度向量的函数——语义相似的文本向量距离近 |
| 余弦相似度 | "向量方向匹配度" | cos(a,b) = a·b / (‖a‖‖b‖)——衡量两个向量的方向相似性 |
| 向量数据库 | "存储向量的库" | 支持高效近似最近邻搜索（ANN）的数据库系统 |
| 语义搜索 | "按意思搜索" | 基于嵌入向量距离的搜索——匹配语义而非关键词 |

---

## 📚 小结

嵌入将文本转换为向量，使语义相似度可用数学计算。向量搜索找到最相似的文档。嵌入模型选择影响 RAG 系统的质量——多语言场景推荐 BGE-large-zh-v1.5。余弦相似度是默认相似度度量。

---

## ✏️ 练习

1. **【实验】** 用 OpenAI embedding API 对 10 个中文查询计算向量，画出相似度热力图
2. **【实现】** 实现基于 FAISS 的简单向量搜索——插入 100 个文档，查询最相似的 5 个

---

## 📖 参考资料

1. [文档] OpenAI Embeddings API: https://platform.openai.com/docs/guides/embeddings
2. [GitHub] sentence-transformers: https://github.com/UKPLab/sentence-transformers
3. [GitHub] FAISS: https://github.com/facebookresearch/faiss
