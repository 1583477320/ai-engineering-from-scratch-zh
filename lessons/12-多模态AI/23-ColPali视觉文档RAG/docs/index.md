# ColPali 与视觉原生文档 RAG

> 传统 RAG 解析 PDF 为文本，切分为块，嵌入块，存储向量。每一步都丢失信号：OCR 丢弃图表数据，分块破坏表格行，文本嵌入忽略图像。ColPali（2024 年 7 月）问了一个更简单的问题：为什么要提取文本？直接用 PaliGemma 嵌入页面图像，用 ColBERT 风格的后期交互做检索——保留布局、图表、字体和格式信息。发布基准：在视觉丰富文档上的端到端准确率比文本 RAG 高 20-40%。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 11（RAG 基础）、阶段 12 · 05（LLaVA）| **时间：** ~180 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释双编码器检索和后期交互检索的区别
- [ ] 实现 ColPali 风格的多向量检索——每页多个嵌入向量
- [ ] 对比文本 RAG 和视觉 RAG 在文档理解任务上的性能差异
- [ ] 设计一个生产级视觉文档 RAG 管道

---

## 1. 问题

传统 RAG 管道：PDF → OCR → 文本切分 → 文本嵌入 → 向量搜索。每一步都丢失信息：OCR 无法处理图表中的数据，分块会把表格行切断，文本嵌入忽略图像内容。

ColPali 的答案：**不提取文本——直接嵌入页面图像。** 用 PaliGemma 编码器提取页面特征，用 ColBERT 风格的后期交互做检索——保留所有布局、图表、字体和格式信息。

---

## 2. 概念

### 2.1 双编码器 vs 后期交互

| 方案 | 方法 | 优势 | 劣势 |
|------|------|------|------|
| **双编码器** | 一个向量/文档 | 快速检索 | 信息压缩严重 |
| **后期交互** | 多个向量/文档 + MaxSim | 保留细节 | 存储更多向量 |

### 2.2 ColPali 架构

```
文档页面图像 → [PaliGemma 编码器] → K 个视觉词元 → 存储为多向量
查询图像/文本 → [PaliGemma 编码器] → K 个查询向量
                                        ↓
                              MaxSim 匹配 → 排序结果
```

### 2.3 MaxSim 相似度

对查询的每个向量，找到文档中最相似的向量（max），然后取平均（sim）。这比单向量余弦相似度更鲁棒。

---

## 3. 从零实现

### Step 1：多向量索引

```python
class SimpleVisualIndex:
    """简化版多向量索引。"""
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def index(self, doc_id, embeddings):
        """索引一个文档的多个向量。"""
        self.documents.append(doc_id)
        self.embeddings.append(embeddings)

    def search(self, query_vecs, top_k=5):
        """MaxSim 检索。"""
        scores = []
        for i, doc_vecs in enumerate(self.embeddings):
            score = self._maxsim(query_vecs, doc_vecs)
            scores.append((self.documents[i], score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _maxsim(self, query_vecs, doc_vecs):
        """MaxSim 相似度。"""
        total = 0
        for q in query_vecs:
            sims = [np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d) + 1e-10)
                    for d in doc_vecs]
            total += max(sims) if sims else 0
        return total / max(len(query_vecs), 1)
```

---

## 4. 工具

### 4.1 HuggingFace

```python
from transformers import AutoModel, AutoProcessor

# ColPali
model = AutoModel.from_pretrained("vidore/colpali-v1.2")
processor = AutoProcessor.from_pretrained("vidore/colpali-v1.2")
```

---

## 6. 工程最佳实践

### 6.1 视觉 RAG vs 文本 RAG

| 方面 | 文本 RAG | 视觉 RAG |
|------|---------|---------|
| 适用文档 | 纯文本文档 | 图表、表格、图像丰富的文档 |
| 检索精度 | 高（精确匹配） | 高（保留布局信息） |
| 索引存储 | 小（一个向量/文档） | 大（K 个向量/文档） |
| 推理成本 | 低 | 中 |

---

## 7. 常见错误

### 错误 1：对纯文本文档使用视觉 RAG

**现象：** 检索效果不如文本 RAG，且存储成本更高。

**修复：** 视觉 RAG 适合图表丰富的文档；纯文本文档仍使用文本 RAG。

---

## 8. 面试考点

### Q1：ColPali 的 MaxSim 和双编码器余弦相似度有什么区别？（难度：⭐⭐）

**参考答案：**
双编码器将整个文档压缩为一个向量——信息损失大。ColPali 保留 K 个向量——每个向量编码页面的不同区域。MaxSim 对查询的每个向量找到文档中最相似的向量，然后取平均。这保留了局部细节（图表、表格的具体位置），而双编码器只能捕捉全局语义。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| ColPali | "视觉文档检索" | 用 PaliGemma 编码页面图像 + ColBERT MaxSim 做多向量检索 |
| MaxSim | "多向量相似度" | 查询的每个向量找文档中最相似的向量（max），然后取平均（sim） |
| 后期交互 | "多向量匹配" | 每个文档保留多个向量，检索时做细粒度匹配 |

---

## 📚 小结

ColPali 用视觉编码器直接嵌入页面图像——保留 OCR 丢失的布局、图表、格式信息。MaxSim 多向量检索比单向量更精确。视觉 RAG 在图表丰富的文档上比文本 RAG 高 20-40%。核心洞察：**不提取文本——直接理解图像。**

---

## ✏️ 练习

1. **【对比】** 对比双编码器和 ColPali 风格多向量检索在文档检索上的准确率
2. **【实现】** 实现 MaxSim 相似度计算——对比单向量余弦相似度

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 多向量索引 | `code/main.py` | ColPali 风格多向量检索实现 |

---

## 📖 参考资料

1. [论文] Faysse et al. "ColPali: Efficient Document Retrieval with Vision Language Models". arXiv, 2024.
2. [论文] Khattab & Zaharia. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT". SIGIR, 2020.
3. [论文] Team ColPali. "ColQwen2: Efficient Document Retrieval with Vision Language Models". arXiv, 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
