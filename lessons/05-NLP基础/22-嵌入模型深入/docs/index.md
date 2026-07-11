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

### 2.4 三层模式——大多数生产系统三层都用

| 场景 | 模式 |
|---|---|
| 快速第一轮 | 稠密双编码器（BGE-M3、text-3-small） |
| 召回率提升 | 稀疏（SPLADE、BGE-M3 稀疏）+ RRF 融合 |
| Top-50 精确率 | 多向量（ColBERTv2）或 Cross-encoder 重排器 |

大多数生产系统三层都用。第一层负责速度和召回——在不贵的情况下尽可能多捞。第二层补充关键词匹配——捕获稠密漏掉的精确词匹配。第三层负责精确率——在候选池上做最精细的比较。

### 2.5 MTEB 排行榜——部分真相

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

---

## 5. 知识连线

- **阶段 05 · 03（Word2Vec）→** Word2Vec 的"一个词一个向量"是本课稠密嵌入的最简形式——神经嵌入 = Word2Vec + Transformer 编码器 + 对比学习
- **阶段 05 · 14（信息检索）→** 本课的稠密/稀疏/多向量是阶段 14 四层检索架构中"稠密检索层"的模型选择——BM25 + BGE-M3 稠密 + RRF = 2026 默认
- **阶段 05 · 23（RAG 分块策略）→** 嵌入模型的上下文窗口（512/8192/32K）直接决定了分块大小的上限——块不能超过嵌入模型有效窗口的 70%

---

## 6. 常见错误

### 错误 1：查询和文档用了不同的前缀

**现象：** BGE 模型检索召回率比预期低 3-5 个点。

**原因：** `bge-*` 模型在训练时查询加了前缀 `"Represent this sentence for searching relevant passages: "`，文档不加。推理时忘了给查询加前缀 → 查询和文档在嵌入空间中错位。

**修复：** 始终检查模型卡片。BGE 查询加前缀，文档不加。`instructor-xl` 相反——查询和文档都要加不同的前缀。不对称编码模型（Voyage、Jina-ColBERT）查询和文档经过完全不同的投影路径——不能用同一个 `encode()` 调用两者。

### 错误 2：Matryoshka 过度截断

**现象：** 1024→64 维后，检索召回率崩溃——下降了 20+ 个点。

**原因：** 1536→256 通常安全（训练时 256 维度也在损失中）。但 64 维远低于训练时最小维度的范围——精度损失不可控。

**修复：** 在你的评估集上验证截断的效果。Matryoshka 模型支持的最小维度通常在其模型卡片上声明。非 Matryoshka 模型（原始 Sentence-BERT）截断时退化严重——不要在它们上面用截断。

---

## 7. 面试考点

### Q1：稠密、稀疏、多向量——选哪个？（难度：⭐⭐）

**参考答案：**
稠密是默认——快、便宜、在大多数任务上足够好。在关键词密集型查询（专有名词、错误代码、产品 SKU）上，稠密漏掉精确匹配 → 加入稀疏（SPLADE 或 BGE-M3 稀疏模式）做 RRF 融合。在需要 token 级精准匹配的长查询（20+ 词）、或领域语料与其他语料语言差异大时 → 多向量（ColBERT）的 MaxSim 显著优于稠密和稀疏。不存在"最好"的模式——取决于查询类型和领域。

### Q2：MTEB 排名第一的模型在你的领域上一定最好吗？（难度：⭐⭐）

**参考答案：**
不一定。MTEB 平均了 56-100+ 个任务——目标任务在你的领域中可能被"稀释"了。如果 MTEB 排名第一的模型在"分类"上赢了 5 个点但在"检索"上反而输了——而你的任务是检索——排名第一对你是误导。**始终在你的领域上跑 MTEB 子集——而非信任总排名。** 候选模型在你的领域特定任务（如中文法律文档检索）上的 Recall@10 比 MTEB 总排名更能预测生产表现。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 稠密嵌入 | "那个向量" | 每段文本一个固定大小向量。余弦相似度排序 |
| 稀疏嵌入 | "学到的 BM25" | 每词表 token 一个学到的权重——大部分为 0。端到端训练 |
| 多向量 | "ColBERT 风格" | 每词元一个向量。MaxSim 打分。更大索引，更好召回 |
| Matryoshka | "俄罗斯套娃" | 前 N 维本身就是一个有效的更小嵌入。训练一次，按存储截断 |
| MTEB | "排行榜" | 大规模文本嵌入基准——发布时 56 个任务，v2 中 100+ |
| BEIR | "检索基准" | 18 个零样本检索任务；常被引用于跨领域鲁棒性 |
| 非对称编码 | "查询 ≠ 文档路径" | 模型对查询和文档使用不同的投影——不能用同一函数编码两者 |

---

## 📚 小结

稠密（一个向量）、稀疏（学到的词权重）、多向量（每词元一个向量）——2026 年的嵌入选择是五维矩阵。Matryoshka 训练一次，按存储预算截断——256d 部署省 75% 存储。BGE-M3 三合一是中文 RAG 的默认起点。MTEB 排行榜是必要起点——始终在你的领域上基准测试候选模型，而非信任总排名。

---

## ✏️ 练习

1. 【理解】用 `bge-small-en-v1.5` 在完整 384 维和 Matryoshka 128 维分别编码 100 个句子。在 10 条查询上衡量 MRR 下降。

2. 【实现】对比 BGE-M3 的稠密、稀疏和 ColBERT 三种模式在你领域 500 段文本上的 Recall@10。哪个模式赢了？RRF 融合是否击败了最佳单一模式？

3. 【实验】在三个候选模型上跑 MTEB，选取与你领域最相关的 2 个任务。报告 MTEB 分数、100 条查询批次上的 p99 延迟、以及每百万查询的成本。选出帕累托最优的那个。

4. 【思考】你的中文法律文档检索用 `bge-m3` 的 Recall@10 只有 72%。你怀疑是领域词汇（"甲方"、"违约金"、"不可抗力"）没有被通用嵌入空间捕获。你会如何用不到 500 条领域查询-文档对来改进？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 可复用提示词 | `outputs/skill-embedding-model-picker.md` | 按语料/部署/预算选择嵌入模型、维度和模式的系统化方案 |

---

## 📖 参考资料

1. [论文] Reimers, Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". EMNLP, 2019. https://arxiv.org/abs/1908.10084 — 双编码器论文
2. [论文] Muennighoff et al. "MTEB: Massive Text Embedding Benchmark". EACL, 2023. https://arxiv.org/abs/2210.07316 — 排行榜论文
3. [论文] Chen et al. "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation". 2024. https://arxiv.org/abs/2402.03216 — 统一三模式模型
4. [论文] Kusupati et al. "Matryoshka Representation Learning". NeurIPS, 2022. https://arxiv.org/abs/2205.13147 — 维度阶梯训练目标
5. [MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard) — 实时排名

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文嵌入建议、工程最佳实践、常见错误、面试考点等均为原创内容。
