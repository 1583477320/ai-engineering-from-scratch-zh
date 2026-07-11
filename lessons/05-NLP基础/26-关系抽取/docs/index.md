# 关系抽取与知识图谱构建

> NER 找到了实体。实体链接锚定了它们。关系抽取找到了它们之间的边。知识图谱是节点、边及其出处的总和。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05 · 06（NER）、05 · 25（实体链接） | **预计时间：** ~60 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

- [ ] 将自由文本转化为结构化三元组 (主语, 关系, 宾语)——预定义 RE vs 开放 RE
- [ ] 理解 2026 生产模式：AEVS 锚定-验证流水线——LLM 提取 + 出处追踪 + 验证

---

## 1. 问题

"Tim Cook 于 2011 年成为苹果公司 CEO。"四个事实：`(Tim Cook, 职位, CEO)`、`(Tim Cook, 雇主, Apple)`、`(Tim Cook, 入职时间, 2011)`、`(Apple, 类型, 公司)`。

**2026 年的问题：LLM 热情地提取关系。太热情了。** 它们幻觉出原文不支持的虚假三元组。没有出处 = 你无法区分真实三元组和看起来合理的虚构。2026 年答案是 AEVS 风格的锚定-验证流水线。

---

## 2. 预定义 RE vs 开放 RE

| | 预定义 RE | 开放 RE |
|---|---|---|
| **关系类型** | 预先定义（雇主、创始人…） | 从文本中提取关系短语本身 |
| **建模** | 多分类或 NLI（蕴含=关系成立） | Seq2Seq 生成三元组 |
| **中文模型** | `bert-base-chinese` + 关系分类头 | HanLP OpenRE、LLM few-shot |

---

## 3. 从零实现

```python
# 预定义 RE：BERT + 关系分类头
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=5)
# 输入：[CLS] 主语 [SEP] 宾语 [SEP] 句子 [SEP] → 输出：关系类型

# AEVS 锚定-验证：先提取，后验证
def extract_and_verify(text, extractor, verifier):
    triples = extractor(text)                     # LLM 或 RE 模型提取候选三元组
    verified = []
    for (s, r, o) in triples:
        evidence = verifier(text, s, r, o)        # NLI: 原文是否蕴含此三元组？
        if evidence: verified.append((s, r, o, evidence))
    return verified                               # 只返回有出处支撑的三元组
```

---

## 4. KG + RAG 模式

```
文档 → NER → 实体链接 → 关系抽取 → 知识图谱 → 结构化事实核查 → LLM 回答
```

KG 提供的验证比向量相似度更可靠——一个三元组要么在 KG 中，要么不在。不存在"50% 相似"的模糊匹配。中文金融/法律知识图谱（CN-DBpedia、OwnThink）是中文 RAG 系统中结构化事实核查的直接来源。

**2026 生产铁律：** 每个提取的三元组必须存储出处（源文档+句子）。没有出处 = 无法验证 = 不可上线。

---

## 5. 陷阱

- **LLM 幻觉三元组。** "Tim Cook 于 2011 年创立了苹果"——看似合理但在原文中不存在的三元组。**始终验证**
- **关系方向反转。** "A 收购了 B" vs "B 被 A 收购"——`(A, acquired, B)` ≠ `(B, acquired, A)`。方向不可反

---

## 🔑 关键术语 | 📚 小结

预定义 RE 用分类处理已知关系类型，开放 RE 用生成发现未知关系。2026 生产 = 提取 + 验证——每个三元组必须携带出处。KG + RAG 提供结构化的事实核查——比向量相似度更可靠。

---

## ✏️ 练习

1. 【理解】在 10 句新闻中手工标注三元组，用 LLM 提取同样的三元组——LLM 幻觉了多少？
2. 【实现】用 `bert-base-chinese` 训练 5 关系的预定义 RE 分类器。衡量 F1。
3. 【实验】构建 AEVS 流水线：LLM 提取 → NLI 验证。对比纯 LLM 提取的精确率提升。

---

## 📖 参考资料

1. [论文] Zeng et al. "Distant Supervision for Relation Extraction". 2014.
2. [知识图谱] CN-DBpedia. https://cn.dbpedia.org/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
