# 指代消解

> "她打电话给他。他没接。医生在吃午饭。"三个指称、两个人、没有一个是名字。指代消解弄清楚谁是谁。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 05 · 06、05 · 07 | **预计时间：** ~60 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

- [ ] 理解指代消解如何将指向同一实体的所有表达聚类——"Apple"、"该公司"、"他们"、"库比蒂诺的科技巨头"
- [ ] 区分五种提及类型（命名实体/名词性/代词性/同位语/零指代）和五种架构（规则→mention-pair→mention-ranking→span-based→LLM）
- [ ] 理解为什么 CoNLL F1 是 MUC+B³+CEAF 三者平均——单个指标各有盲区

---

## 1. 问题

从一篇 300 字的文章中提取所有关于苹果公司的提及。文章写"Apple"时很好找。写"这家公司"、"他们"、"库比蒂诺的科技巨头"、"乔布斯的公司"时——NER 管道漏掉了 60-80% 的提及。

指代消解将指向同一现实实体的每个表达链接到一个簇中。它是表层 NLP 和下游语义之间的胶水。**2026 年为什么重要：** 摘要中的"The CEO announced..."需要被还原为"Tim Cook announced..."。QA 中的"Who did she call?"需要先消解"she"。知识图谱中"PER1 创立 Apple"和"Jobs 创立 Apple"是同一个事实——消解后将它们合并。

---

## 2. 概念

### 2.1 四种提及类型

| 类型 | 示例 |
|---|---|
| **命名实体** | "Tim Cook" |
| **名词性** | "the CEO"、"the company" |
| **代词性** | "he"、"she"、"they"、"it" |
| **同位语** | "Tim Cook, Apple's CEO," |

### 2.2 五种架构

1. **基于规则（Hobbs, 1978）。** 语法树代词消解——用句法规则。好的基线。在代词上出奇地难以被超越
2. **Mention-pair 分类器。** 每对提及预测是否同指 → 传递闭包聚类。2016 年前的标准
3. **Mention-ranking。** 每个提及对候选先行词排名（含"无先行词"）。选 top-1
4. **Span-based 端到端（Lee et al., 2017）。** Transformer 编码 → 枚举候选 span → 预测提及分数 → 预测先行概率 → 贪心聚类。**现代默认**
5. **LLM（2024+）。** 提示词要求列出代词和先行词。简单案例好，长文档和罕见指称上挣扎

### 2.3 评估——为什么需要五个指标的平均

**MUC、B³、CEAF、BLANC、LEA——因为没有单一指标能完整评价聚类质量。** CoNLL F1 = 前三者平均。2026 年 CoNLL-2012 上 SOTA 约 83 F1。

### 2.4 已知硬案例

- 定指描述——指向在页面前方引入的实体
- 桥接指代（"轮子"→前面提到的车）
- **零指代**——中文和日文中主语被省略
- 前指（代词在指称之前）："When **she** walked in, Mary smiled."

---

## 3. 从零实现

### 第 1 步：预训练神经指代消解

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # 实验性模型
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
# Cluster 1: [Apple, The company, they]
# Cluster 2: [new products]
```

### 第 2 步：基于规则的代词消解器（教学版）

`code/main.py` 中有一个仅用标准库的实现：提取提及（命名实体+代词+定指描述），对每个代词向前 K 个提及打分——性别/数的一致（启发式）、近因（越近越好）、句法角色（偏好主语）→ 链接最高分先行词。不具备与神经模型的竞争力，但展示了搜索空间和端到端模型必须做出的决策。

### 第 3 步：LLM 做指代消解

```python
prompt = f"""Text: {text}
列出所有指向人或公司的代词和名词短语。按指向对象聚类。输出JSON：
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

两个失败模式需要关注：**(1)** LLM 过度合并——将"him"和"her"（指向两个不同的人）错误合并。**(2)** 长文档中 LLM 静默丢弃提及——跨 50+ 段落时一次性 API 调用不可靠。使用滑动窗口 + 合并策略。

### 第 4 步：评估

标准 CoNLL-2012 脚本计算 MUC、B³、CEAF-φ4 并报告三者平均。内部评估从 span 级精确率/召回率开始，再加入提及链接 F1。

---

## 4. 陷阱

- **单例爆炸。** 有些系统将每个提及报告为一个独立簇——B³ 对此宽容，MUC 惩罚。始终检查全部三个指标
- **长上下文中的代词。** 在 2000+ token 的文档上性能下降约 15 F1。谨慎分块
- **性别假设。** 硬编码性别规则在非二元指称、组织和动物上失效。使用学到的模型或中性打分
- **长文档上的 LLM 漂移。** 单次 API 调用无法在 50+ 段落上可靠地聚类提及。使用滑动窗口 + 合并

---

## 5. 工业工具——2026 技术栈

| 场景 | 选择 |
|---|---|
| 英文、单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP neural coref |
| 多语言 | SpanBERT/XLM-R 在 OntoNotes 或 Multilingual CoNLL 上训练 |
| 跨文档事件指代消解 | 专用端到端模型（2025-26 SOTA） |
| 快速 LLM 基线 | GPT-4o/Claude + 结构化输出指代消解 prompt |
| 生产对话系统 | 规则回退 + 神经主模型 + 关键槽位人工复核 |

**2026 集成模式：** 先跑 NER → 跑指代消解 → 将指代消解簇合并进 NER 实体。下游任务看到的是每个簇一个实体——而非每个提及一个实体。

---

## 6. 常见错误

### 错误 1：中文指代消解直接用英文工具

**现象：** 用 spaCy `en_coreference_web_trf` 处理中文文本——返回空簇或随机聚类。

**原因：** 英文指代消解模型依赖英文的词性、句法树和代词体系。中文没有英文式的代词格变化（he/him/his），且存在零指代——英文模型完全无法处理。

**修复：** 中文指代消解需要专门的解决方案。封闭域：规则+距离约束消解"它"/"他"/"她"。开放域：LLM prompt（中文提示词——"这段话中的'他'指的是谁？"）。零指代恢复：上下文窗口 3 句内查找最近的一致主语。

---

## 7. 面试考点

### Q1：为什么指代消解需要五个评估指标？（难度：⭐⭐）

**参考答案：**
因为聚类质量无法用单一指标捕获。MUC（link-based）惩罚过度合并。B³（mention-based）对单例爆炸宽容。CEAF-φ4（entity-based）要求簇的对齐。三者平均（CoNLL F1）是 2026 年报告的标准。任何一个指标单独使用都会给特定类型的系统留下"钻空子"的空间——例如只报告 B³ 的系统可以通过将所有提及分为独立簇获得高分，但实际上什么都没消解。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 提及 (Mention) | "一个指称" | 指向某实体的一段文本（名字、代词、名词短语） |
| 先行词 (Antecedent) | "'it'指的是什么" | 后来的提及与之同指的较早提及 |
| 簇 (Cluster) | "实体的所有叫法" | 全部指向同一现实实体的提及集合 |
| 回指 (Anaphora) | "向后指" | 后来的提及指向较早的提及（"他"→"张三"） |
| 前指 (Cataphora) | "向前指" | 较早的提及指向后来的提及（"When he arrived, John..."） |
| 桥接 (Bridging) | "隐式指代" | "我买了辆车。轮子是坏的。"（那辆车的轮子） |
| CoNLL F1 | "排行榜上的数字" | MUC、B³、CEAF-φ4 F1 分数的平均值 |

---

## 📚 小结

指代消解将"Apple"、"the company"、"they"、"it"聚合到一个簇中——NER 的下游任务因此看到的是实体，而非孤立的提及。Span-based 端到端（Lee et al., 2017）是现代默认。中文零指代是英文工具无法处理的额外挑战——封闭域用规则+距离约束，开放域用 LLM prompt。CoNLL F1 = MUC+B³+CEAF 三者平均——永远不要只报告一个指标。

---

## ✏️ 练习

1. 【理解】在 5 段手写段落上运行基于规则的代词消解器。对照人工标注衡量提及链接准确率。

2. 【实现】使用预训练神经指代消解模型处理一篇新闻文章。与你自己的人工标注对比聚类结果。模型在哪里失败了？

3. 【实验】构建指代消解增强的 NER 流水线：先 NER，再通过指代消解簇合并实体。在 100 篇文章上衡量实体覆盖率相比纯 NER 的提升。

---

## 📖 参考资料

1. [教材] Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking. https://web.stanford.edu/~jurafsky/slp3/26.pdf
2. [论文] Lee et al. "End-to-end Neural Coreference Resolution". EMNLP, 2017. https://arxiv.org/abs/1707.07045 — span-based 端到端
3. [论文] Joshi et al. "SpanBERT: Improving Pre-training by Representing and Predicting Spans". TACL, 2020. https://arxiv.org/abs/1907.10529 — 提升指代消解的预训练
4. [论文] Hobbs. "Resolving Pronoun References". 1978. — 基于规则的经典

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。中文零指代分析、中文指代消解建议为原创内容。
