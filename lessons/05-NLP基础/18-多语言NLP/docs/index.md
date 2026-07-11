# 多语言 NLP

> 一个模型，100+ 种语言，其中大多数没有训练数据。跨语言迁移是 2020 年代的实用奇迹。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 05 · 04（GloVe、FastText、子词）、阶段 05 · 11（机器翻译）
**预计时间：** ~45 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 19（子词分词）— 本课的"分词税"是阶段 19 选择 BPE/Unigram/WordPiece 的直接动机

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 用 XLM-R 做零样本跨语言分类——理解为什么在英文上 fine-tune 后能在乌尔都语上运行
- [ ] 理解"源语言选择"——语言相似度比语料大小更能预测迁移质量；LANGRANK/qWALS 指标的使用
- [ ] 诊断低资源语言的"分词税"——fertility tax、variant recovery tax、capacity spillover tax 三者如何叠加

---

## 1. 问题

英文有数十亿条标注。乌尔都语有几千条。迈蒂利语几乎没有。任何服务全球用户的 NLP 系统必须在任务特定训练数据不存在的语言长尾上工作。

多语言模型用一个模型同时在多种语言上训练来解决这个问题。**共享的表示空间让模型将在高资源语言上学到的技能迁移到低资源语言。** 在英文情感分析上 fine-tune → 推断时直接在乌尔都语上运行 → 得到意外好的情感预测。这就是零样本跨语言迁移——它重塑了 NLP 如何交付给世界。

本课命名的取舍、经典模型、以及初次接触多语言工作的团队最容易踩的坑：选择迁移的源语言。

---

## 2. 概念

### 2.1 三个共享——一个模型

| 层级 | 说明 |
|---|---|
| **共享词表** | SentencePiece/WordPiece 在所有目标语言的文本上训练。同一子词单元代表跨相关语言的同一词素——英文和意大利文的 `anti-` 是同一个 token |
| **共享表示** | 跨多语言的掩码语言模型预训练学到了一件事：不同语言中语义相似的句子产生相似的隐藏状态。mBERT、XLM-R、NLLB 都展示了这一点。"cat"（英文）、"chat"（法文）、"gato"（西班牙文）的嵌入聚集在一起——完整句子的嵌入也一样 |
| **共享任务头** | 分类/标注头对所有语言共享——因为它接收的是一个语言中立的表示 |

### 2.2 零样本 vs 少样本

- **零样本：** 只在英文上 fine-tune → 推断时在任何语言上运行。不需要目标语言标注。对类型相近的语言（德语→荷兰语、西班牙语→葡萄牙语）效果好，差异大时（英语→日语）效果弱
- **少样本：** 目标语言加 100-500 条标注。分类任务准确率跳到英文基线的 95-98%。**这是多语言 NLP 中投资回报率最高的单杠杆**

### 2.3 模型选择矩阵

| 模型 | 年份 | 语言数 | 适用场景 |
|---|---|---|---|
| mBERT | 2018 | 104 | 经典 NLP 基准、学术对比。低资源语言上弱 |
| XLM-R | 2019 | 100 | CommonCrawl 训练（远大于 Wikipedia）。270M Base, 550M Large。跨语言分类的默认基线 |
| XLM-V | 2023 | 100 | 100 万词表（vs XLM-R 25 万）。低资源语言上更好 |
| mT5 | 2020 | 101 | 多语言生成——摘要/翻译/QA |
| NLLB-200 | 2022 | 200 | 翻译。包含 55 种低资源语言。阶段 11 的默认选择 |
| BLOOM | 2022 | 46+13 编程 | 开源 176B 多语言 LLM |
| Aya-23 | 2024 | 23 | Cohere 多语言 LLM。阿拉伯/印地/斯瓦希里语上很强 |

**中文场景选择：** 中英混合任务用 XLM-R（中文预训练量远超 mBERT）。简体+繁体需同时支持→训练 tokenizer 的语料中两种变体都要有足够占比。

---

## 3. 从零实现

### 第 1 步：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")

def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()  # 蕴含=第3类
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))

# 一个模型，三种语言——相同的 API
print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))  # 印地语
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))      # 法语
```

XLM-R 在 NLI 数据上训练后，通过蕴含技巧（entailment trick）很好地迁移到分类——前提是"假设模板"（`this text is about {}`）在所有语言中都是自然的英文。对中文，建议用中文假设模板测试。

### 第 2 步：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),          # 法语
    ("The cat is sleeping.", "El gato está durmiendo."), # 西班牙语
    ("The cat is sleeping.", "Die Katze schläft."),      # 德语
    ("The cat is sleeping.", "猫在睡觉。"),               # 中文
    ("The cat is sleeping.", "The dog is barking."),     # 完全不同
]
for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    print(f"  {eng!r} <-> {other!r}: cos={np.dot(emb_eng, emb_other):.3f}")
```

翻译聚集在嵌入空间中。不同的英文句子离得更远。这就是跨语言检索、聚类和相似度计算的底层机制。

---

## 4. 源语言选择——2026 年的研究结论

大多数团队默认用英文作为 fine-tune 的源语言。最近的研究（2026）表明这往往是错的。

**语言相似度比语料量更能预测迁移质量。** 对斯拉夫语目标——德语或俄语经常打败英文。对印地语目标——印地语经常打败英文。**qWALS** 相似度指标（2026，基于 World Atlas of Language Structures 特征）量化了这一点。**LANGRANK**（Lin et al., ACL 2019）是一个更早的方法，结合语言相似度、语料量和谱系关系对候选源语言排名。

**实用规则：** 如果你的目标语言有一个类型学上相近的高资源近亲——先在那个近亲上 fine-tune，再比较英文 fine-tune。

---

## 5. 低资源语言的"分词税"

多语言模型在所有语言上共享一个 tokenizer。这个词表在以英文、法文、西班牙文、中文、德文为主的语料上训练。对于不在主导集合中的语言——三种税悄然叠加：

- **Fertility tax（生育税）。** 低资源语言文本的每个词被切分成远比英文多的 token。一句印地语可能需要英文等价句 3-5 倍的 token。这 3-5 倍消耗了你的上下文窗口、训练效率和延迟
- **Variant recovery tax（变体恢复税）。** 每一个拼写错误、变音符号变体、Unicode 归一化不匹配或大小写变体都成为嵌入空间中一个冷启动的不相关序列。模型学不会一个母语者视为理所当然的正字法对应
- **Capacity spillover tax（容量溢出税）。** 税 1 和税 2 消耗了上下文位置、层深度和嵌入维度。留给实际推理的空间系统性低于高资源语言从同一模型中获得的空间

**实际症状：** 你的模型在印地语上正常训练，loss 曲线看起来正常，eval 困惑度看起来合理，生产输出却微妙地错了。形态变化在句子中间崩溃。稀有屈折形式无法恢复。**你不能用更多数据规模化地修复一个坏掉的 tokenizer。**

**缓解：** 选择一个对你的目标语言有良好覆盖的 tokenizer（XLM-V 的 100 万词表是对此的直接修复）；在训练前验证留出目标文本上的分词 fertility；对真正的长尾文字使用字节级回退（SentencePiece `byte_fallback=True`）——使任何东西都不会 OOV。

---

## 6. 实际有效的评估

- **每种语言在留出集上的准确率——分别报告。** 永远不聚合。聚合隐藏了长尾
- **与单语言基线对比。** 对数据充足的语言——从头训练的单语言模型偶尔打败多语言模型。测试
- **实体级测试。** 目标语言中的命名实体。多语言模型对远离拉丁字母的文字通常分词覆盖较弱
- **跨语言一致性。** 两种语言中的相同语义应该产生相同的预测。衡量差距

---

## 7. 工业工具——2026 技术栈

| 任务 | 推荐 |
|---|---|
| 分类，100 种语言 | XLM-R-base (~270M) fine-tune |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见阶段 11） |
| 多语言生成 LLM | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V 或相关高资源语言上的领域 fine-tune |

**永远为 fine-tune 目标语言留预算——如果性能要紧。** 零样本是起点，不是最终答案。

---

## 8. 面试考点

### Q1：为什么在英文上 fine-tune 再迁移到印地语可能不是最优的？（难度：⭐⭐）

**参考答案：**
语言相似度比源语言标注数据量更能预测迁移质量。印地语在类型学上更接近马拉地语或孟加拉语（同为印度-雅利安语支）而非英语（日耳曼语支）。LANGRANK/qWALS 指标量化了这一点——对印地语目标，在印地语自身的数据上 fine-tune 或选择一个类型学上更接近的高资源源语言（如孟加拉语如果有足够标注），经常比英文更好。

### Q2：什么是低资源语言的"分词税"？（难度：⭐⭐⭐）

**参考答案：**
三个叠加效应：**(1)** 印地语每个词需要 3-5 倍于英文的 token 来表达——消耗上下文窗口和训练计算。(**2)** 拼写变体和正字法不一致（变音符号、Unicode 归一化）被当作不相关的 token 序列——嵌入质量崩溃。(**3)** 前两者消耗的容量系统性减少了可用于实际语义推理的表示空间。最终结果：模型在 eval 上看起来 OK 但在生产输出中产生微妙但不可恢复的形态变化错误。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 多语言模型 | "一个模型说多种语言" | 跨语言共享词表和参数 |
| 跨语言迁移 | "在一种语言上训练，在另一种上运行" | 在源语言上 fine-tune，在目标语言上评估——无目标语言标注 |
| 零样本 | "没见过目标语言的数据" | 在没有任何目标语言标注的情况下迁移 |
| 少样本 | "见过一点" | 100-500 条目标语言标注用于 fine-tune。ROI 最高的单杠杆 |
| XLM-R | "跨语言的默认基线" | 100 语言 RoBERTa，在 CommonCrawl 上预训练 |
| 分词税 | "说小语言的隐性成本" | 低资源语言文本被切得比英文碎得多——上下文、训练和推理都被消耗 |

---

## 📚 小结

多语言模型的"三个共享"——词表、表示空间、任务头——使零样本跨语言迁移成为可能。中文场景选择 XLM-R 或 XLM-V。源语言选择：语言相似度 > 标注量——LANGRANK/qWALS 帮助选择。低资源语言的"分词税"不能通过加数据修复——选择一个对目标语言有良好覆盖的 tokenizer（或字节回退）是唯一解。

---

## ✏️ 练习

1. 【理解】在英文、法文、印地语、阿拉伯文上分别跑 10 句零样本分类。报告各语言的准确率。法文应该强，印地语中等，阿拉伯文有波动。

2. 【实现】用 `paraphrase-multilingual-MiniLM-L12-v2` 在多语言混合语料库上构建跨语言检索器。英文查询，检索任意语言的文档。衡量 Recall@5。

3. 【实验】对比英文-源和印地语-源 fine-tune 在印地语分类任务上的效果。各用 500 条目标语言标注做少样本 fine-tune。报告哪个源产生更好的印地语准确率及差距——这是 LANGRANK 论断的微型验证。

4. 【思考】你的多语言模型在斯瓦希里语 NER 上表现异常差——但训练 loss 正常。排查哪一层是瓶颈？如何用一句代码确认分词 fertility 是否是根因？

---

## 📖 参考资料

1. [论文] Conneau et al. "Unsupervised Cross-lingual Representation Learning at Scale". NeurIPS, 2019. https://arxiv.org/abs/1911.02116 — XLM-R 论文
2. [论文] Pires, Schlinger, Garrette. "How Multilingual is Multilingual BERT?". ACL, 2019. https://arxiv.org/abs/1906.01502 — 开创跨语言迁移研究线的分析论文
3. [论文] Costa-jussà et al. "No Language Left Behind". 2022. https://arxiv.org/abs/2207.04672 — NLLB-200 论文
4. [论文] Language Similarity Predicts Cross-Lingual Transfer Learning Performance. Machine Learning and Knowledge Extraction, 2026. — qWALS/LANGRANK 源语言选择论文

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文场景建议、分词税分析、工程最佳实践、常见错误、面试考点等均为原创内容。
