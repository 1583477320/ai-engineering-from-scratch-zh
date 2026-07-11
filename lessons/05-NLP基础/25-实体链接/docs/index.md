# 实体链接与消歧

> NER 找到了"Jordan"。实体链接决定：迈克尔·乔丹（篮球）？Michael B. Jordan（演员）？Michael I. Jordan（伯克利 ML 教授）？约旦（国家）？没有链接，你的知识图谱永远是模糊的。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05 · 06（NER）、05 · 24（指代消解） | **预计时间：** ~60 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

- [ ] 理解实体链接的两个子任务——候选生成（提及→可能的 KB 条目）+ 消歧（上下文→正确的那个）
- [ ] 实现消歧排序器的三个信号——上下文相似度、先验概率、全局一致性

---

## 1. 问题

"Jordan 击败了媒体。"NER 标注"Jordan"为 PERSON。但**哪个** Jordan？迈克尔·乔丹（篮球）？Michael B. Jordan（演员）？Michael I. Jordan（伯克利 ML 教授——在 ML 论文中这个混淆真实存在）？约旦（国家）？

实体链接（Entity Linking, EL）将每个提及消歧为知识库（Wikidata、Wikipedia、DBpedia 或领域 KB）中的唯一条目。两个子任务：

1. **候选生成。** 给定"Jordan"，哪些 KB 条目是可能的？用名称索引 + 别名词典筛到 5-20 个
2. **消歧。** 给定上下文，哪个候选是正确的？三个信号排序

---

## 2. 消歧的三个信号

| 信号 | 做法 | 示例 |
|---|---|---|
| **上下文相似度** | 文档和实体描述之间的向量相似度 | 文档提到"篮球"→ Michael Jordan (basketball) 的 Wikidata 描述也提到"basketball" |
| **先验概率** | 该提及在 Wikipedia 中链接到某实体的比例 | "Paris" → 90% 链接到 France，8% 到 Hilton，2% 到 Texas |
| **全局一致性** | 同一文档中的所有实体应来自语义一致的世界 | 文档已有"NBA"和"Chicago" → 进一步支持 Michael Jordan (basketball) |

---

## 3. 从零实现

```python
# 候选生成：Wikipedia 名称索引 + 别名词典
def generate_candidates(mention, kb_index, alias_dict, max_candidates=20):
    candidates = kb_index.search(mention)  # 精确+模糊匹配
    candidates.extend(alias_dict.get(mention, []))  # 缩写/"Apple"→"Apple Inc."
    return candidates[:max_candidates]

# 消歧：上下文向量 × 先验概率
def disambiguate(mention, candidates, doc_context, entity_embeddings, priors):
    ctx_emb = embed(doc_context)
    scores = {}
    for c in candidates:
        sim = cosine(ctx_emb, entity_embeddings[c])  # 上下文相似度
        prior = priors.get(mention, {}).get(c, 1e-5)  # 先验概率
        scores[c] = sim * prior                       # 乘积排序
    return max(scores, key=scores.get)
```

---

## 4. 陷阱

- **候选生成太窄。** 别名覆盖不足 → 正确答案不在候选集中 → 后续消歧再好也无用。**先衡量候选召回率**——正确答案出现在 top-20 候选中的比例
- **先验概率在新鲜实体上失效。** "COVID-19"在 2020 年前的 Wikipedia 先验概率为 0。新闻/社交媒体领域需要定期更新先验
- **全局一致性的计算成本。** 全文档级别的联合推断是 NP-hard。生产中用贪心（一次消歧一个提及，固定已消歧的）或整数线性规划

---

## 5. 中文实体链接特殊挑战

- **中文名字歧义度极高。** "张伟"全国重名超过 30 万。消歧几乎完全依赖上下文——同篇文档中提到的其他实体是唯一线索
- **百度百科 + Wikipedia 联合候选库。** Wikipedia 中文条目数远少于英文——大量中国本土实体（企业家、地方景点、网络红人）只在百度百科。中文 EL 生产系统需两者联合
- **中英文混用提及。** 一篇中文新闻中"苹果公司"、"Apple Inc."、"AAPL"三者指代同一实体。需要跨语言别名映射

---

## 🔑 关键术语 | 📚 小结

实体链接 = NER 的下一站——将模糊的提及消歧为 KB 中的唯一条目。候选生成决定上限，消歧决定质量。三个信号互补：上下文相似度（语义）、先验概率（统计）、全局一致性（结构）。中文 EL 需要 Wikipedia + 百度百科联合候选库和中英文别名映射。

---

## ✏️ 练习

1. 【理解】在 Wikipedia 上查找"Jordan"的候选实体——列出所有可能的 Wikidata 条目。哪个候选应该是消歧的最高分？
2. 【实现】用 Wikipedia API 为一个提及生成候选集并基于上下文相似度排名。在 20 条含歧义实体的句子上评估。
3. 【思考】"苹果发布了新 iPhone"——实体链接应该把"苹果"链接到水果还是公司？你的上下文信号从哪来？

---

## 📖 参考资料

1. [论文] Shen et al. "Entity Linking with a Knowledge Base". 2015. — 经典 EL 综述
2. [工具] spaCy + Wikipedia + DBpedia Spotlight

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。中文 EL 挑战为原创内容。
