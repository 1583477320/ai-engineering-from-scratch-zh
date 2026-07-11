# 嵌入模型——2026 深入

> Word2Vec 每个词给一个向量。现代嵌入模型给每段文本一个向量——跨语言、稀疏+稠密+多向量视角、尺寸适配你的索引。选错了，你的 RAG 检索到错误的内容。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 05 · 03（Word2Vec）、阶段 05 · 14（信息检索）
**预计时间：** ~60 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 23（RAG 分块策略）— 嵌入模型的上下文窗口直接决定了分块策略的上限

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 在五个维度上选择嵌入模型——稠密/稀疏/多向量、语言、上下文长度、维度预算、开源/托管
- [ ] 使用 Matryoshka 截断——1024d 训练，256d 部署，精度损失 1-3%，存储省 75%
- [ ] 运行 BGE-M3 的三合一模式（稠密+稀疏+ColBERT），调整融合权重

---

## 1. 问题

你的 RAG 系统 40% 的时间检索到错误的段落。元凶很少是向量数据库或提示词——是嵌入模型。

2026 年选择嵌入意味着在五个维度上做权衡：

1. **稠密 vs 稀疏 vs 多向量。** 每段一个向量 vs 每词元一个权重 vs 每词元一个向量
2. **语言覆盖。** 单语言英文模型在纯英文任务上仍然赢。多语言模型在混合语料上赢
3. **上下文长度。** 512 vs 8192 vs 32768 token——实际有效容量通常只有标称的 60-70%
4. **维度预算。** 3072 维 × 4 字节 = 12KB/向量。1 亿条向量 = $1300/月存储。Matryoshka 截断砍到 1/4
5. **开源 vs 托管。** 开源 = 完全控制。托管 = 始终最新但要接受数据流出

---

## 2. 概念

### 2.1 三种嵌入模式

| 模式 | 原理 | 优势 | 成本 |
|---|---|---|---|
| **稠密** | 每段一个固定大小向量（384-3072d） | 快、余弦排序 | 漏关键词 |
| **稀疏（SPLADE）** | 每词表 token 一个学到的权重——大部分为 0 | 关键词匹配 + 学到的词权重 | 大词表 → 大索引 |
| **多向量（ColBERT）** | 每词元一个向量。MaxSim 打分 | 长查询、领域语料最佳召回 | 存储 × token 数，贵 |

### 2.2 BGE-M3——一个模型，三种模式

单个模型同时输出稠密、稀疏和多向量表示。各模式可独立查询，分数通过加权和融合。当你想从一个 checkpoint 获得灵活性时——2026 的默认选择。

### 2.3 Matryoshka 表示学习

**训练时：** 损失在 256、512、768、1024 维上同时优化。**部署时：** 截断到你的存储预算维数——精度降 1-3%，存储省 75%（1024→256d）。

### 2.4 MTEB 排行榜——部分真相

大规模文本嵌入基准——发布时 56 个任务跨 8 种类型，v2 扩展到 100+。2026 年初：Gemini Embedding 2 在检索上居首（67.71 MTEB-R）。Cohere embed-v4 在通用上领先（65.2 MTEB）。BGE-M3 在开源多语言上领先（63.0）。**排行榜必要但不充分——始终在你的领域上基准测试。**

---

## 3. 从零实现

### 第 1 步：基线——Sentence-BERT 稠密嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = ["The first iPhone launched in 2007.", "Apple released the iPod in 2001."]
emb = encoder.encode(corpus, normalize_embeddings=True)  # 点积=余弦

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
```

`normalize_embeddings=True` 使点积等于余弦相似度。**始终设置它。**

### 第 2 步：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)  # 截断后重新归一化

emb_256 = truncate(emb, 256)
```

Nomic v1.5、OpenAI text-3、Voyage-4 在训练中已适配——前几层截断几乎无损。非 Matryoshka 模型截断时退化严重。

### 第 3 步：BGE-M3 三合一

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
output = model.encode(corpus, return_dense=True, return_sparse=True, return_colbert_vecs=True)

# 分数融合——在你的领域上调权重
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

三个索引，一次推理调用。

### 第 4 步：MTEB 自定义评估

```python
from mteb import MTEB
evaluation = MTEB(tasks=["ArguAna", "SciFact", "NFCorpus"])
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在**代表你领域的**子集上跑候选模型。不要仅信任排行榜排名——你的领域是关键。

---

## 4. 陷阱

- **查询和文档用了不同的编码路径。** Voyage、Jina-ColBERT 使用非对称编码——查询和文档经过不同的投影。**始终检查模型卡片**
- **缺少前缀。** `bge-*` 模型需要在查询前加上 `"Represent this sentence for searching relevant passages: "`——忘了就是 3-5 个点的召回差距
- **Matryoshka 过度截断。** 1536→256 通常安全。1536→64 不安全。在你的评估集上验证
- **上下文截断。** 大多数模型对超长输入静默截断。长文档需要分块（见阶段 23）
- **忽略延迟尾部。** MTEB 分数隐藏了 p99 延迟。600M 模型可能比 335M 模型高 2 分，但每查询成本高 3 倍

---

## 5. 工业工具——2026 技术栈

| 场景 | 选择 |
|---|---|
| 纯英文、快、API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源、英文 | `BAAI/bge-large-en-v1.5` |
| 开源、多语言 | `BAAI/bge-m3` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4 |
| CPU-only 部署 | Nomic Embed v2（137M 参数，MoE） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词密集型查询 | 加 SPLADE 稀疏，与稠密做 RRF 融合 |

**中文首选 BGE-M3 或 `bge-large-zh-v1.5`（纯中文）。** 2026 模式：从 BGE-M3 或 text-3-large 开始，用 MTEB 在你的领域上评估，如果有领域特化模型赢超过 3 个点才换。

---

## 🔑 关键术语 | 📚 小结

稠密（一个向量）、稀疏（学到的词权重）、多向量（每词元一个向量）——2026 年的嵌入选择是五维矩阵。Matryoshka 训练一次，按存储预算截断——256d 部署省 75% 存储。BGE-M3 三合一是中文 RAG 的默认起点。**MTEB 排行榜是起点——始终在你的领域上基准测试。**

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
