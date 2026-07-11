# RAG 分块策略

> 分块配置对检索质量的影响与嵌入模型选择相当（Vectara NAACL 2025）。分块做错了，再多的重排序也救不回来。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 05 · 14（信息检索）、阶段 05 · 22（嵌入模型）
**预计时间：** ~60 分钟
**所处阶段：** Tier 1

---

## 🎯 学习目标

- [ ] 实现六种分块策略——固定、递归、语义、句级、父文档、上下文检索
- [ ] 解释 Vectara 2026 基准：递归 512-token 分块击败语义分块（69% vs 54%）——为什么"显而易见的答案"是错的
- [ ] 按查询类型匹配块大小：事实型 256-512、分析型 512-1024、整段理解 1024-2048

---

## 1. 问题

你把一份 50 页的合同放进 RAG 系统。用户问："终止条款是什么？"检索器返回了封面页。为什么？模型用 512-token 的块训练——终止条款在第 20 页，跨过页码分隔，没有局部关键词绑定到查询。

修复不是"买个更好的嵌入模型"。修复是分块——多长、重叠多少、在哪切、要不要带上下文？

**2026 年 2 月基准的意外结论：**
- Vectara 2026：递归 512-token 分块击败语义分块——69% → 54% 准确率
- SPLADE + Mistral-8B 在 Natural Questions 上：重叠提供零可测量的收益
- 上下文悬崖：响应质量在约 2500 token 的上下文处急剧下降

**"显而易见"的答案（语义分块、20% 重叠、1000 token）经常是错的。**

---

## 2. 概念——六种策略

| 策略 | 做法 | 优势 | 陷阱 |
|---|---|---|---|
| **固定** | 每 N token，重叠 M | 最简单 baseline | 从句子中间截断 |
| **递归** | 按 `\n\n`→`\n`→`。`→` ` 优先级逐级切分 | **2026 默认选择** | — |
| **语义** | 邻句嵌入余弦→低于阈值则切分 | 保留主题连贯 | 可能产生 40 token 碎片——破坏检索 |
| **句级** | 每句一块或 N 句窗口 | 成本极低、与语义效果接近 | 窗口外上下文丢失 |
| **父文档** | 小"子"块检索 + 大"父"块返回 | 精度+不丢上下文 | 索引存储翻倍 |
| **上下文检索** | LLM 为每块生成它在文档中位置的摘要→前缀在块上 | Anthropic 基准提升 35-50% | 构建索引成本高 |

### 查询类型匹配块大小（NVIDIA 2026）

| 查询类型 | 块大小 |
|---|---|
| 事实型（"CEO 叫什么名字？"） | 256-512 token |
| 分析型/多跳 | 512-1024 token |
| 整段理解 | 1024-2048 token |

**核心原则：** 块要大到包含答案+局部上下文，小到检索器的 top-K 聚焦于答案而非上下文噪音。

---

## 3. 从零实现

### 递归分块——2026 默认

```python
def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks, buf = [], ""
        for p in parts:
            if len(p) > size:
                if buf: chunks.append(buf); buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size: buf = candidate
            else: chunks.append(buf); buf = p
        if buf: chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)  # 最后回退：硬切
```

### 父文档模式

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping
```

**关键洞察：去重父块。** 多个子块可能映射到同一个父块——全部返回等于浪费上下文。

### 语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences: return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])  # 邻句余弦相似度
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])   # 相似度骤降 → 此处切分
        else:
            chunks[-1].append(sentences[i])
    # 合并组内句子，超大块递归二次切分
    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

**在你的领域上调 `threshold`。** 太高 → 碎片化。太低 → 一个巨大的块。Vectara 2026 基准中语义分块被递归击败——但它在主题切换检测\*应该\*有理论优势。实际限制：句子嵌入相似度在短句上噪音大，min_chars 的强制是最重要的防护。

### 上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    prompts = [f"<document>{document}</document>\nHere is the chunk: <chunk>{c}</chunk>\n"
               f"用 50-100 字描述此块在文档中的位置。" for c in chunks]
    contexts = llm.batch(prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引上下文化的块。查询时检索受益于额外的周围信号。Anthropic 自身基准提升 35-50% 检索效果。**代价：** 每个块一次 LLM 调用——对 1 万块的语料库，构建索引的成本显著增加。仅在对检索质量有刚性要求（法律/金融/医疗文档）时使用。

### 评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top): hits += 1
    return hits / len(queries)
```

**始终基准测试。** 你的语料库的"最佳"策略可能与任何博客文章都不匹配。在 50 条查询评估集上测量 Recall@5——事实型、分析型、多跳三类分层——然后调参。

---

## 4. 陷阱

- **只按事实型查询评估分块。** 多跳查询揭示截然不同的赢家。用分类型查询评估集
- **语义分块不设最小块大小。** 产生 40 token 碎片——破坏检索。始终强制 `min_tokens`
- **重叠作为 cargo cult。** 2026 研究发现重叠常常提供零收益且翻倍索引成本。测量，不假设
- **无最小/最大强制。** 5 token 或 5000 token 的块都会破坏检索。钳制
- **跨文档分块。** 永远不让一个块跨越两篇文档。始终按文档独立分块，再合并

---

## 5. 工业工具——2026 技术栈

| 场景 | 策略 |
|---|---|
| 首次构建、语料未知 | 递归、512 token、无重叠 |
| 事实型 QA | 递归、256-512 token |
| 分析型/多跳 | 递归、512-1024 token + 父文档 |
| 大量交叉引用（合同/论文） | Late Chunking 或上下文检索 |
| 对话语料 | 按轮次分块 + 说话人元数据 |
| 短文本（微博/评论） | 一篇文档 = 一个块 |

**从递归 512 开始。** 在 50 条查询评估集上测量 Recall@5。从那里开始调参。

- **只按事实型查询评估分块。** 多跳查询揭示截然不同的赢家。用分类型查询评估集
- **语义分块不设最小块大小。** 产生 40 token 碎片——破坏检索。始终强制 `min_tokens`
- **重叠作为 cargo cult。** 2026 研究发现重叠常常提供零收益且翻倍索引成本。测量，不假设
- **无最小/最大强制。** 5 token 或 5000 token 的块都会破坏检索。钳制

---

## 5. 中文分块特别建议

- **中文递归优先级：** `\n\n` → `\n` → `。` → `；` → `，`。最后才按字符数硬切
- **重叠用"句数"而非"token 数"。** 重叠 2-3 句比重叠 200 token 更稳定——中文 token/字比例在不同文体中差异大
- **Markdown 中文文档按 `##` 分块最优。** 本项目 Lesson 文档天然有标题层级——按 `##` 分块 + 保留上级标题信息——最有效的中文技术文档分块方案

---

## 📚 小结

六种分块策略——固定、递归、语义、句级、父文档、上下文检索。递归 512-token 是 2026 默认（击败语义 69% vs 54%）。查询类型决定块大小——事实型 256-512、分析型 512-1024。重叠在 2026 基准中常为零收益。中文按 `。`→`；`→`，` 优先级递归分块，重叠以完整句子为单位。

从递归 512 开始。在 50 条查询评估集上测量 Recall@5。不要假设博客文章中的"最佳实践"适合你的数据——你的领域是最好的裁判。

---

## ✏️ 练习

1. 【理解】用固定(512,0)、递归(512,0)和递归(512,100)在 20 页文档上分块。对比块数量和边界质量。

2. 【实现】在 5 篇文档上构建 30 条查询的评估集。衡量递归、语义和父文档三种策略的 Recall@5。哪个赢了？与博客文章的结论一致吗？

3. 【实验】实现上下文检索。衡量相比基线递归的 MRR 提升。报告索引成本（LLM 调用次数）vs 准确率提升。

4. 【思考】你的中文法律合同 RAG 在"违约责任"查询上召回率很高但在"不可抗力条款"查询上很低。分块策略可能是原因吗？如何仅靠调整分块参数（不改嵌入模型）来改进？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 可复用提示词 | `outputs/skill-chunker.md` | 按语料和查询类型选择分块策略、大小和重叠的系统化方案 |

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 块 (Chunk) | "文档的一片" | 被嵌入、索引和检索的子文档单元 |
| 重叠 (Overlap) | "安全边界" | 相邻块间共享的 N 个 token；2026 基准中常无作用 |
| 语义分块 | "聪明地切" | 在邻句嵌入相似度骤降处切分 |
| 父文档 | "两级检索" | 检索小子块，返回大父块——精度 + 不丢上下文 |
| Late Chunking | "先嵌入后分块" | 在 token 级嵌入整个文档，再池化为块向量——保留跨块上下文 |
| 上下文检索 | "Anthropic 的把戏" | LLM 为每块生成它在文档中位置的摘要→前缀在块上—构建索引前注入周围信号 |
| 上下文悬崖 | "2500 token 的墙" | RAG 响应质量在约 2.5k 上下文 token 处观察到的质量突降（2026 年 1 月） |

---

## 📖 参考资料

1. [官方文档] LangChain — Recursive Character Text Splitter. https://python.langchain.com/docs/how_to/recursive_text_splitter/ — 生产中的默认实现
2. [论文] Vectara. "Chunking configurations analysis". NAACL, 2025. https://arxiv.org/abs/2410.13070 — 分块配置与嵌入选择同等重要
3. [技术博客] Jina AI — Late Chunking in Long-Context Embedding Models. 2024. https://jina.ai/news/late-chunking-in-long-context-embedding-models/ — Late Chunking 论文
4. [技术博客] Anthropic — Contextual Retrieval. 2024. https://www.anthropic.com/news/contextual-retrieval — LLM 生成的上下文前缀提升检索 35-50%
5. [技术博客] NVIDIA / PremAI — RAG Chunking Strategies: The 2026 Benchmark Guide. https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/ — 按查询类型的块大小建议

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文分块建议（递归优先级、句级重叠、Markdown `##` 分块）、工程最佳实践、常见错误等均为原创内容。
