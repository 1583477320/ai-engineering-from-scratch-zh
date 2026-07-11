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

### 上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    prompts = [f"<document>{document}</document>\nHere is the chunk: <chunk>{c}</chunk>\n"
               f"用 50-100 字描述此块在文档中的位置。" for c in chunks]
    contexts = llm.batch(prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引上下文化的块。查询时检索受益于额外的周围信号。Anthropic 自身基准提升 35-50% 检索效果。

---

## 4. 陷阱

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

六种分块策略——固定、递归、语义、句级、父文档、上下文检索。递归 512-token 是 2026 默认（击败语义 69% vs 54%）。查询类型决定块大小——事实型 256-512、分析型 512-1024。重叠在 2026 基准中常为零收益。中文按 `。`→`；`→`，` 优先级递归分块，重叠以完整句子为单位。在 50 条查询评估集上测量 Recall@5——不要假设博客文章中的"最佳实践"适合你的数据。

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
