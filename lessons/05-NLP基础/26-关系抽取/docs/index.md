# 关系抽取与知识图谱构建

> NER 找到了实体。实体链接锚定了它们。关系抽取找到了它们之间的边。知识图谱是节点、边及其出处的总和。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05 · 06（NER）、05 · 25（实体链接） | **预计时间：** ~60 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现三种关系抽取方法——规则/模式匹配、有监督分类（REBEL）、LLM 提示提取
- [ ] 理解 AEVS 锚定-验证流水线——Anchor → Extract → Verify → Supplement——如何将 LLM 幻觉率大幅降低
- [ ] 实现关系规范化——将开放 IE 的任意短语映射到 Wikidata 等封闭本体的规范 ID

---

## 1. 问题

"Tim Cook 于 2011 年成为苹果公司 CEO。"四个事实：

- `(Tim Cook, 职位, CEO)`
- `(Tim Cook, 雇主, Apple)`
- `(Tim Cook, 入职时间, 2011)`
- `(Apple, 类型, 公司)`

关系抽取（Relation Extraction, RE）将自由文本转化为结构化三元组 `(主语, 关系, 宾语)`。在语料库上聚合 → 知识图谱。聚合 + 查询 → RAG 的推理基础、分析或合规审计。

**2026 年的问题：LLM 热情地提取关系。太热情了。** 它们幻觉出原文不支持的虚假三元组。没有出处——你无法区分真实三元组和看起来合理的虚构。2026 年的答案是 AEVS 风格的锚定-验证流水线。

---

## 2. 概念

### 2.1 三元组形式

`(subject_entity, relation_type, object_entity)`。关系来自封闭本体（Wikidata 属性、FIBO、UMLS）或开放集合（OpenIE 风格——任何短语都可以）。

### 2.2 三种提取方法

1. **规则/模式匹配。** Hearst patterns："X such as Y" → `(Y, isA, X)`。手工正则。脆弱、精确、可解释
2. **有监督分类。** 给定句子中的两个实体提及，从固定集合中预测关系。在 TACRED、ACE、KBP 上训练。2015-2022 标准
3. **LLM 生成式。** Prompt 模型输出三元组。开箱即用。需要出处——否则幻觉出看起来合理的垃圾

### 2.3 AEVS（Anchor-Extraction-Verification-Supplement, 2026）

当前的幻觉缓解框架：

- **Anchor（锚定）。** 用精确位置标识每个实体 span 和关系短语 span
- **Extract（提取）。** 生成链接到锚定 span 的三元组
- **Verify（验证）。** 将每个三元组元素匹配回原文；拒绝任何不受支撑的内容
- **Supplement（补充）。** 一次覆盖率扫描——确保没有锚定 span 被遗漏

幻觉率大幅下降。需要更多计算，但可审计。

### 2.4 开放 vs 封闭的权衡

- **封闭本体。** 固定属性列表（如 Wikidata 的 11,000+ 属性）。可预测、可查询、难以编造
- **开放 IE。** 任何动词短语都可以成为关系。高召回、低精确率、查询困难

生产 KG 通常混合：开放 IE 做发现 → 将关系规范化为封闭本体 → 合并到主图。

---

## 3. 从零实现

### 第 1 步：基于模式的提取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

Hearst patterns 至今仍在领域特定流水线中部署——因为它们可调试。

### 第 2 步：有监督关系分类（REBEL）

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 Seq2Seq 关系提取器：文本进，三元组出，已映射到 Wikidata 属性 ID。在远程监督数据上 fine-tune。标准开源基线。

### 第 3 步：LLM 提示提取 + 锚定

```python
prompt = f"""从文本中提取 (主语, 关系, 宾语) 三元组。
每个三元组包含在原文中的精确字符位置。

Text: {text}

输出 JSON：
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}]

只包含文本完全支撑的三元组。不做超出原文陈述的推理。
"""
```

验证每个返回的 span 是否匹配原文。拒绝任何 `text[start:end] != triple_entity` 的三元组。**这是 AEVS"验证"步骤的最简形式。**

### 第 4 步：规范化为封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",    # chief executive officer
    "was born in":   "P19",     # place of birth
    "founded":        "P112",   # founded by (注意主宾语反转)
    "works at":       "P108",   # employer
}

def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None  # 丢弃未映射的开放关系，或路由到人工审核
```

**规范化通常占 60-80% 的工程工作。** 为它留足预算。开放 IE 关系表达不一致（"was born in"、"came from"、"is a native of"）——折叠到规范 ID——否则图谱不可查询。

### 第 5 步：构建小图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))

def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, [])
            if relation is None or r == relation]

print(neighbors("Tim Cook", relation="P108"))  # → [(P108, Apple)]
```

这是每个 RAG-over-KG 系统的基本原子。用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强的图存储进行规模化。

---

## 4. 陷阱

- **RE 之前先做指代消解。** "He founded Apple"——RE 需要知道"he"是谁。先跑阶段 24 的指代消解
- **实体规范化。** "Apple Inc"和"Apple"必须解析到同一个节点。先跑阶段 25 的实体链接
- **幻觉三元组。** LLM 输出原文不支撑的三元组。强制执行 span 验证
- **关系规范化漂移。** 开放 IE 关系不一致（"was born in"、"came from"、"is a native of"）。折叠到规范 ID——否则图谱不可查询
- **时间错误。** "Tim Cook is CEO of Apple"——现在正确，2005 年错误。许多关系是有时间范围的。使用限定符（Wikidata 的 `P580` 开始时间、`P582` 结束时间）
- **领域不匹配。** REBEL 在 Wikipedia 上训练。法律、医疗、科学文本通常需要领域 fine-tune 的 RE 模型

---

## 5. 工业工具——2026 技术栈

| 场景 | 选择 |
|---|---|
| 快速生产、通用领域 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 领域特定（生物医学、法律） | SciREX 风格领域 fine-tune + 自定义本体 |
| LLM 提示、可审计输出 | AEVS 流水线：Anchor → Extract → Verify → Supplement |
| 高容量新闻 IE | 模式匹配 + 有监督混合 |
| 从零构建 KG | 开放 IE + 人工规范化 |
| 时序 KG | 带限定符提取（开始/结束时间、时间点） |

**集成模式：** NER → 指代消解 → 实体链接 → 关系抽取 → 本体映射 → 图谱加载。每个阶段都是潜在的质量关卡。

---

## 6. 常见错误

### 错误 1：LLM 提取三元组不作 span 验证

**现象：** 提取的 100 个三元组中约 15-20% 包含原文中不存在的实体或值。KG 中累积了不可追溯的虚假信息。

**原因：** LLM 被训练为"完成模式"——当它看到"X is the CEO of"时，即使原文没有说，它也会倾向性地补全"Apple"。这是语言模型的知识，不是文本中的事实。

**修复：** 每个三元组强制携带出处——源文档 + 字符 span。拒绝任何 `text[span] != entity_text` 的三元组。这是 AEVS 流水线的最低可行版本。

### 错误 2：关系方向反转

**现象：** "A 收购了 B"被提取为 `(B, acquired, A)`——方向反转。图谱查询返回错误的收购关系。

**原因：** 被动语态（"B 被 A 收购"）和主动语态（"A 收购了 B"）中主宾语的位置交换。模型可能从字面上匹配"B...收购"而忽略了"被"字。

**修复：** 加入关系方向验证——对每个提取的三元组，用 NLI 模型验证原文是否蕴含 `(subject, relation, object)` 而非 `(object, relation, subject)`。

---

## 7. 面试考点

### Q1：AEVS 流水线为什么比纯 LLM 提取更可靠？（难度：⭐⭐）

**参考答案：**
因为它在 LLM 生成之后增加了一个独立的验证步骤——LLM 提取可以被幻觉污染，但验证步骤将每个三元组元素匹配回原文的具体字符位置。如果声称的实体在原文的那个位置不存在——三元组被直接拒绝，无论 LLM 多自信。这切断了 LLM 的语言模型知识（可能不准确）和文本实际陈述之间的联系——从"相信 LLM"变为"验证原文"。

### Q2：为什么关系规范化占 RE 工程的 60-80%？（难度：⭐⭐⭐）

**参考答案：**
因为自然语言中同一关系的表达方式极度多样。"is the CEO of"、"runs"、"heads"、"leads"、"serves as chief executive of"——都是同一个 Wikidata 属性 P169。如果每组同义表达在 KG 中作为独立的关系谓词——图谱不可查询（你需要记住所有 5 种表达才能找到"Tim Cook 的雇主"）。规范化将这些表达折叠为一组规范的 ID——使图谱可查询、可合并、可推理。这项工作是纯工程——没有"深度学习"可以替你完成。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 三元组 (Triple) | "主语-关系-宾语" | KG 的原子单元：`(s, r, o)` |
| 开放 IE | "什么都提取" | 开放词汇关系短语——高召回、低精确率 |
| 封闭本体 | "固定 Schema" | 有界的关系类型集合（Wikidata、UMLS、FIBO） |
| 规范化 | "一切标准化" | 将表面名称/关系映射到规范 ID |
| AEVS | "锚定提取" | Anchor-Extraction-Verification-Supplement 流水线（2026） |
| 出处 | "来源链接" | 每个三元组携带来源的文档 ID + 字符 span |
| 远程监督 | "廉价标注" | 将文本与现有 KG 对齐来生成训练数据 |

---

## 📚 小结

三种 RE 方法——规则、有监督（REBEL）、LLM 生成——构成从精确率到召回率的完整光谱。AEVS 流水线（Anchor→Extract→Verify→Supplement）是 2026 幻觉缓解的标配。**关系规范化占工程工作的 60-80%——为它留足预算。** 每个三元组必须携带出处（源文档+char-span）。没有出处 = 不可审计 = 不可上线。

KG + RAG 提供结构化的事实核查——比向量相似度更可靠。一个三元组要么在 KG 中（事实），要么不在（需要验证）。不存在模糊匹配。

---

## ✏️ 练习

1. 【理解】在 `code/main.py` 中运行模式提取器——在 5 句新闻文章上测试。手工检查精确率。

2. 【实现】用 REBEL（或小 LLM）在相同句子上运行。对比三元组。哪个提取器精确率更高？召回率更高？

3. 【实验】构建 AEVS 流水线：LLM 提取 + 对照原文验证 span。在 50 句 Wikipedia 风格句子 上衡量验证前后的幻觉率。

4. 【思考】你的 RE 系统在 Wikidata 本体规范化的覆盖率只有 60%。剩下的 40% 是开放 IE 关系——如何让这些"野生"关系也可以查询？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 可复用提示词 | `outputs/skill-re-designer.md` | 按场景设计 RE 流水线的系统化方案 |

---

## 📖 参考资料

1. [论文] Mintz et al. "Distant supervision for relation extraction without labeled data". ACL, 2009. https://www.aclweb.org/anthology/P09-1113.pdf — 远程监督论文
2. [论文] Huguet Cabot, Navigli. "REBEL: Relation Extraction By End-to-end Language generation". EMNLP, 2021. https://aclanthology.org/2021.findings-emnlp.204.pdf — Seq2Seq RE 主力
3. [论文] Wadden et al. "Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)". EMNLP, 2019. https://arxiv.org/abs/1909.03546 — 联合 IE
4. [框架] AEVS — Anchor-Extraction-Verification-Supplement. 2026. https://www.mdpi.com/2073-431X/15/3/178 — 幻觉缓解设计
5. [教程] Wikidata SPARQL. https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial — 规范图查询

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、AEVS 中文适配分析、关系规范化中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
