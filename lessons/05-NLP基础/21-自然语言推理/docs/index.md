# 自然语言推理——文本蕴含

> "前提蕴含假设"的意思是：一个典型读者读完前提，会得出结论说假设为真。NLI 判断的是蕴含 / 矛盾 / 中立。表面上枯燥，生产环境中扛重活。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 05 · 05（情感分析）、阶段 05 · 13（问答系统）
**预计时间：** ~60 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 27（LLM 评估框架）— RAGAS 的忠实度指标底层就是 NLI 模型

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 使用 BART/DeBERTa MNLI 模型做三路蕴含推断——理解蕴含 ≠ 严格逻辑蕴含，而是"典型读者会得出的结论"
- [ ] 将 NLI 用于三个生产场景——幻觉检测、事实核查、零样本分类
- [ ] 理解"零样本模板敏感度"——"This text is about X" vs "X" 可以摆动 10+ 个点的准确率

---

## 1. 问题

你做了一个摘要器。它产出了一段摘要。你怎么知道摘要没有幻觉？

你做了一个聊天机器人。它回答"是的"。你怎么知道这个回答有检索到的段落支撑？

你需要分类 1 万篇新闻但没有标注。你能复用一个已有模型吗？

**三个问题都归结为自然语言推理（NLI）。** NLI 问：给定前提 t 和假设 h——h 是被 t 蕴含的、被 t 反驳的、还是完全无关的（中立）？

| 生产场景 | t（前提） | h（假设） | 判断 |
|---|---|---|---|
| 幻觉检测 | 原文 | 摘要中的一句 | 非蕴含 = 幻觉 |
| 事实核查（RAG） | 检索到的段落 | 生成的回答 | 非蕴含 = 编造 |
| 零样本分类 | 待分类文档 | 标签描述："这段文字是关于体育的" | 蕴含 = 预测该类 |

**一个 NLI 任务，三个生产用途。** 这就是为什么每一个 RAG 评估框架（RAGAS、DeepEval）底层都跑着一个 NLI 模型。

---

## 2. 概念

### 2.1 三个标签——不是逻辑蕴含

- **蕴含（Entailment）。** t → h。"猫在垫子上"蕴含"有一只猫"
- **矛盾（Contradiction）。** t → ¬h。"猫在垫子上"矛盾于"没有猫"
- **中立（Neutral）。** 两者皆非。"猫在垫子上"中立与"猫饿了"

**NLI 是自然语言推理——不是严格的逻辑推理。** "John walked his dog"蕴含"John has a dog"——在 NLI 中被认为是蕴含，但严格一阶逻辑只有在你公理化了"拥有"之后才会承认。NLI 问的是"典型人类读者会得出什么结论"——不是"逻辑上必然从哪些公理推出"。

### 2.2 数据集

- **SNLI（2015）。** 57 万人工标注对，图片标题做前提。领域窄
- **MultiNLI（2017）。** 43 万对，跨 10 种体裁。2026 年的标准训练语料
- **ANLI（2019）。** 对抗性 NLI。人类专门写了设计来打破现有模型的例子。更难
- **DocNLI、ConTRoL（2020-21）。** 文档级前提。测试多跳和长距离推理

### 2.3 架构

Transformer 编码器（BERT/RoBERTa/DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 表示馈入三路 Softmax。在 MNLI 上训练，在留出基准上评估——分布内对可达 90%+ 准确率。

---

## 3. 从零实现

### 第 1 步：运行预训练 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli", top_k=None)

premise = "猫在沙发上睡觉。"
hypothesis = "房间里有一只猫。"

result = nli({"text": premise, "text_pair": hypothesis})[0]
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

生产 NLI：`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是开源默认。DeBERTa-v3 在排行榜上居首。

### 第 2 步：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "股市在央行降息后大幅上涨。"
labels = ["金融", "体育", "政治", "科技"]
result = zs(text, candidate_labels=labels)
# {'labels': ['金融', '政治', '科技', '体育'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

模板默认是"This example is about {label}."。用 `hypothesis_template` 自定义。不需要训练数据，不需要 fine-tune。开箱即用。

### 第 3 步：RAG 忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实度的核心。将生成的答案拆分为原子声明。逐条对照检索到的上下文检查。报告蕴含的比例。

---

## 4. 陷阱

- **Hypothesis-only 捷径。** 模型可以单独从假设预测标签——在 SNLI 上约 60% 准确率——因为"not"、"nobody"、"never" 与矛盾类相关。**这是检测标注泄漏的强基线**
- **词汇重叠启发式。** 子序列启发式（"每个子序列都被蕴含"）能通过 SNLI，但过不了 HANS/ANLI。使用对抗基准测试
- **文档级退化。** 单句 NLI 模型在文档级前提上下降 20+ F1。长上下文使用 DocNLI 训练模型
- **零样本模板敏感度。** "This text is about {label}" vs "{label}" vs "The topic is {label}"——准确率可以摇摆 10+ 个点。**调模板**
- **领域不匹配。** MNLI 训练在通用英文上。法律、医疗、科学文本需要领域特定 NLI（SciNLI、MedNLI）

---

## 5. 工业工具——2026 技术栈

| 场景 | 模型 |
|---|---|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速/边缘 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量） | `facebook/bart-large-mnli` |
| 文档级 NLI | DocNLI 训练模型 |
| 多语言（含中文） | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 幻觉检测 | RAGAS/DeepEval 内置 NLI 层 |

**2026 元模式：NLI 是文本理解的万能胶带。** 任何时候你需要"前提是否支撑假设？"或"前提是否与假设矛盾？"——先用 NLI，再考虑再调用一次 LLM。

### 中文 NLI 特别建议

- **中文 NLI 数据集规模远小于英文。** CMNLI（中文多类型 NLI）和 OCNLI 是两个主要选择——两者都比 MNLI（43 万条）小一个数量级。建议从多语言 XLM-R NLI checkpoint 开始 fine-tune，而非从头训练
- **中文零样本模板需要中文化。** "This text is about X"直接翻译为"这段文字是关于X的"——在中文上可能不如"这段文字的主题是X"有效。在 50 条中文标注样本上对比 3-5 个候选模板
- **中文忠实度检测的原子声明拆分需要适配中文语法。** RAGAS 默认的英文拆分对中文逗号和中文连词的支持有限——自定义中文拆分策略（按"。"、"；"和"而且/但是/此外"）

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 蕴含 | "'前提推出假设'" | 典型读者读完前提会得出假设为真 |
| 矛盾 | "'前提排除假设'" | 典型读者读完前提会得出假设为假 |
| 中立 | "'不确定'" | 前提不提供判断假设的依据 |
| 零样本分类 | "NLI 当分类器用" | 标签文字变成假设，选蕴含分数最高的 |
| 忠实度 | "答案有出处吗？" | 生成内容 ↔ 检索上下文对照。不蕴含 = 幻觉 |

---

## 📚 小结 | ✏️ 练习

NLI 是 NLP 中最"安静"的基础设施——摘要验证、RAG 事实核查、零样本分类全部依赖它。2026 年的默认是 DeBERTa-v3-large-mnli（英文）或多语言 XLM-R NLI（中文）。零样本模板敏感度可达 10+ 点的准确率摆动——调模板。

练习：用 `facebook/bart-large-mnli` 在 20 个手写的（前提、假设、标签）三元组上跑三路推断。加入对抗性"子序列启发式"陷阱。比较三个零样本中文模板的准确率摆动。

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
