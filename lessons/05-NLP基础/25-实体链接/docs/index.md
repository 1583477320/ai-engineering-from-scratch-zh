# 实体链接与消歧

> NER 找到了"Paris"。实体链接决定：法国巴黎？Paris Hilton？德克萨斯州 Paris？特洛伊王子 Paris？没有链接，你的知识图谱永远是模糊的。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05 · 06（NER）、05 · 24（指代消解） | **预计时间：** ~60 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现实体链接的两个子任务——候选生成（别名索引）+ 消歧（上下文排序）
- [ ] 理解三种消歧方法——先验+上下文（Milne-Witten）、嵌入（BLINK）、生成式（GENRE）
- [ ] 解释为什么必须分别报告候选召回率和消歧准确率——99% 消歧 × 80% 候选召回 = 80% 流水线
- [ ] 处理 NIL（不在 KB 中的实体）——系统必须学会说"我不知道"，而非猜测错误的实体

---

## 1. 问题

"Jordan 击败了媒体。"NER 标注"Jordan"为 PERSON。但**哪个** Jordan？

- 迈克尔·乔丹（篮球）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利 ML 教授——在 ML 论文中这个混淆真实存在）？
- 约旦（国家）？
- Jordan（希伯来语名字）？

实体链接（Entity Linking, EL）将每个提及消歧为知识库（Wikidata、Wikipedia、DBpedia 或领域 KB）中的唯一条目。两个子任务：

1. **候选生成。** 给定"Jordan"，哪些 KB 条目是可能的？Wikipedia 别名词典覆盖大部分命名实体——"JFK"→ John F. Kennedy / Jacqueline Kennedy / JFK airport / JFK (movie)。典型索引每个提及返回 10-30 个候选
2. **消歧。** 给定上下文，哪个候选是正确的？

两个步骤都可以学习。两个步骤都可以基准测试。组合流水线已经稳定了十年——变化的是消歧器的质量。

---

## 2. 概念

### 2.1 三种消歧方法

1. **先验 + 上下文（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好，快，不需要训练
2. **基于嵌入（BLINK / REL）。** 编码提及+上下文。编码每个候选的描述。选最大余弦相似度。2020-2024 的默认
3. **生成式（GENRE, 2021；LLM-based, 2023+）。** 逐字符解码实体的规范名称。约束解码到有效实体名称的 trie——保证输出是有效的 KB ID

### 2.2 端到端 vs 流水线

现代模型（ELQ、BLINK、ExtEnD、GENRE）在一次前向中运行 NER + 候选生成 + 消歧。但流水线系统在生产中仍然主导——因为你可以单独替换组件。

### 2.3 两个必须分别报告的指标

- **提及召回率（候选生成）。** 正确 KB 条目出现在候选列表中的比例。**整个流水线的天花板**
- **消歧准确率 / F1。** 候选正确的前提下，top-1 正确的比例

**永远分别报告两者。** 一个在 80% 候选召回率上做到 99% 消歧的系统 = 80% 的流水线。消歧数字看起来很好——但 20% 的提及从候选生成阶段就丢了。

---

## 3. 从零实现

### 第 1 步：从 Wikipedia 重定向构建别名索引

```python
# Wikipedia 别名数据：约 1800 万 (别名, 实体) 对。从 Wikidata dump 下载
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

### 第 2 步：基于上下文的消歧（Jaccard 教学版）

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠是教学用的。替换为嵌入余弦相似度（见第 3 步）。

### 第 3 步：基于嵌入（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    """将提及标记为 [MENTION] ... [/MENTION] 后编码整个上下文。"""
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

索引时：每个 KB 实体编码一次。查询时：提及+上下文编码一次，与候选池做点积，选最大。

### 第 4 步：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。约束解码（见阶段 20）确保只能输出有效标题。现代后代是 REL-GEN 和 LLM-prompted EL + 结构化输出。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention. Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`）——这是 2026 年最简单的可上线 EL 流水线。

### 第 5 步：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准 EL 基准：1393 篇 Reuters 文章、3.4 万提及、Wikipedia 实体。报告 in-KB 准确率（`P@1`）和 out-of-KB NIL 检测率。

---

## 4. 陷阱

- **NIL 处理。** 有些提及不在 KB 中（新兴实体、冷门人物）。系统必须预测 NIL 而非猜测错误实体。**单独衡量 NIL 检测率**
- **提及边界错误。** 上游 NER 漏掉了部分 span（"Bank of America"只标注了"Bank"）。EL 召回率下降
- **流行度偏差。** 训练系统过度预测高频实体。一篇 ML 论文中的"Michael I. Jordan"经常被链接到篮球乔丹。**在领域数据上 fine-tune 纠正先验**
- **跨语言 EL。** 将中文文本中的提及映射到英文 Wikipedia 实体。需要多语言编码器或翻译步骤
- **KB 过时。** 新公司、事件、人物不在去年的 Wikipedia dump 中。生产流水线需要刷新循环

---

## 5. 工业工具——2026 技术栈

| 场景 | 选择 |
|---|---|
| 通用英文 + Wikipedia | BLINK 或 REL |
| 跨语言、KB = Wikipedia | mGENRE |
| LLM 友好、少量提及/天 | Claude/GPT-4 + 候选列表 + 约束 JSON |
| 领域特定 KB（医疗、法律） | 自定义 BERT + KB 感知检索 + 领域 AIDA 风格数据 fine-tune |
| 极低延迟 | 纯先验精确匹配（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

**2026 上线模式：** NER → 指代消解 → 每提及 EL → 簇合并为一个规范实体。输出：文档中每个实体一个 KB ID——而非每个提及一个。

---

## 6. 常见错误

### 错误 1：不衡量候选召回率就直接优化消歧

**现象：** 消歧准确率 95%，但端到端只有 70%——20% 的提及正确答案从未进入候选列表。

**原因：** 别名索引覆盖不足——缩写、拼写变体、跨语言提及没有匹配到 KB 条目。消歧器再好也无法修复候选生成阶段的损失。

**修复：** 始终先基准测试候选召回率（正确答案在 top-20 中的比例）。如果召回率 < 85%——先扩展别名词典（加入常见拼写错误、缩写、跨语言映射），再优化消歧器。

### 错误 2：流行度偏差在领域数据上未经纠正

**现象：** 在 ML 论文数据集上，所有"Jordan"都被链接到篮球运动员。

**原因：** 训练数据中 Michael Jordan (basketball) 的先验概率远高于 Michael I. Jordan (professor)。通用 Wikipedia 先验在特定领域中被系统性误导。

**修复：** 在领域标注数据上 fine-tune 消歧器——给上下文的权重增加，给先验的权重降低。或使用**仅上下文**的消歧策略（嵌入方法），在领域数据上彻底忽略通用先验。

---

## 7. 面试考点

### Q1：为什么候选召回率和消歧准确率必须分别报告？（难度：⭐⭐）

**参考答案：**
因为它们是串联的两阶段——候选召回率是整个流水线的天花板。一个在 80% 候选召回率上做到 99% 消歧的系统实际上只有 80% 的端到端准确率。消歧数字看起来很好——但 20% 的提及从候选生成阶段就丢了，消歧器从未有机会处理它们。分别报告让你知道瓶颈在哪里。

### Q2：中文实体链接为什么比英文更难？（难度：⭐⭐⭐）

**参考答案：**
三个结构性原因：**(1)** 中文名字歧义度极高——"张伟"全国重名超过 30 万，消歧几乎完全依赖上下文。**(2)** Wikipedia 中文条目远少于英文——大量中国本土实体只在百度百科——候选库必须跨 KB 联合。**(3)** 中英文混用——同一篇文档中"苹果公司"、"Apple Inc."、"AAPL"指代同一实体——需要跨语言别名映射。三者的叠加使中文 EL 的难度系统性高于英文。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 实体链接 (EL) | "链接到 Wikipedia" | 将提及映射到 KB 中的唯一条目 |
| 候选生成 | "可能是谁？" | 返回一个提及的合理 KB 条目短列表 |
| 消歧 | "选对的那个" | 用上下文对候选打分，选出胜者 |
| 别名索引 | "查找表" | 从表层形式 → 候选实体的映射 |
| NIL | "不在 KB 中" | 显式预测没有任何 KB 条目匹配——而非猜测错误实体 |
| KB | "知识库" | Wikidata、Wikipedia、DBpedia 或你的领域 KB |
| AIDA-CoNLL | "基准" | 1393 篇 Reuters 文章，含标准实体链接标注 |

---

## 📚 小结

实体链接 = NER 的下一站——候选生成（别名索引）+ 消歧（上下文排序）。三种消歧方法：先验+上下文（Milne-Witten，最快）、嵌入（BLINK，默认）、生成式（GENRE/LLM，2026 前沿）。永远分别报告候选召回率和消歧准确率——前者是整个流水线的天花板。

中文 EL 需要 Wikipedia + 百度百科联合候选库。生产模式：NER → 指代消解 → 每提及 EL → 簇合并为一个规范实体。NIL 检测是必备组件——系统必须学会说"我不知道"。

---

## ✏️ 练习

1. 【理解】在 `code/main.py` 中实现先验+上下文消歧器——在 10 个歧义提及（Paris, Jordan, Apple）上测试。手工标注正确实体。衡量准确率。

2. 【实现】用 Sentence-Transformer 编码 50 个歧义提及和每个候选的描述。对比嵌入消歧和 Jaccard 上下文重叠的准确率差异。

3. 【实验】构建一个 1000 实体的领域 KB（如公司员工+产品）。实现 NER + EL 端到端。在 100 句留出句子上衡量精确率和召回率。

4. 【思考】你的 EL 系统在测试集上把 30% 的新兴公司名预测为 NIL——但实际上它们在 KB 中（只是别名索引没覆盖）。你如何在不重新训练整个模型的情况下修复？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 可复用提示词 | `outputs/skill-entity-linker.md` | 按场景选择 KB、候选生成器和消歧器的系统化方案 |

---

## 📖 参考资料

1. [论文] Milne, Witten. "Learning to Link with Wikipedia". CIKM, 2008. https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf — 基础先验+上下文方法
2. [论文] Wu et al. "Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)". ACL, 2020. https://arxiv.org/abs/1911.03814 — 基于嵌入的主力模型
3. [论文] De Cao et al. "Autoregressive Entity Retrieval (GENRE)". ICML, 2021. https://arxiv.org/abs/2010.00904 — 约束解码生成式 EL
4. [论文] Hoffart et al. "Robust Disambiguation of Named Entities in Text (AIDA)". EMNLP, 2011. https://www.aclweb.org/anthology/D11-1072.pdf — 基准论文
5. [论文] van Hulst et al. "REL: An Entity Linker Standing on the Shoulders of Giants". 2020. https://arxiv.org/abs/2006.01969 — 开源生产堆栈

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文 EL 挑战分析（百度百科联合、人名歧义、跨语言混用）、工程最佳实践、常见错误、面试考点等均为原创内容。
