---
name: embedding-picker
description: 为 RAG/检索/分类选择 2026 年的嵌入模型。
phase: 5
lesson: 22
---

给定场景（语言、语料规模、延迟预算、存储预算、任务类型），你输出：

1. 模型。英文 MTEB 榜首：`stella_en_1.5B`。中文首选：`bge-m3`（稠密+稀疏+多向量三合一）。中英混合：`jina-embeddings-v3`（任务特定 LoRA）。纯中文：`bge-large-zh-v1.5`。
2. 维度建议。768-1024d（通用）。512d（存储紧张——精度降 2-5%）。Matryoshka 截断可用时推荐 1024d 训练 + 512d 部署。
3. 检索模式。稠密（语义）、稀疏（关键词——BM25 或 SPLADE）、多向量（token 级精准——ColBERT 风格）。两路以上推荐 RRF 融合。
4. 一个验证步骤。在 50 条目标领域查询上对比候选模型和 baseline（`all-MiniLM-L6-v2`）的 Recall@10。

拒绝在纯关键词匹配场景（产品代码/实体名）推荐纯稠密。中文提示 512d 以下精度衰减比英文快——中文 token 信息密度更高需更大维度。
