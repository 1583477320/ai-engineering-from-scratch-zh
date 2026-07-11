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

### 第 4 步：手写 NLI 分类器（概念演示）

`code/main.py` 中有一个仅用标准库的玩具实现：前提和假设通过词汇重叠 + 否定检测进行比较。不具备与 Transformer 模型的竞争力——但它展示了任务的形态：两个文本输入，三路标签输出，损失 = 对 `{entail, contradict, neutral}` 的交叉熵。

核心洞察：即使是一个统计"not"/"never"/"nobody"出现频率的简单模型，在 SNLI 上仅凭假设就能达到 ~60% 的准确率——因为这些否定词与矛盾类高度相关。这是 **hypothesis-only 基线**——检测标注泄漏和快捷学习的最强工具。

---

## 4. 陷阱

- **Hypothesis-only 捷径。** 模型可以单独从假设预测标签——在 SNLI 上约 60% 准确率——因为"not"、"nobody"、"never"与矛盾类相关。**这是检测标注泄漏的强基线**
- **词汇重叠启发式。** 子序列启发式（"每个子序列都被蕴含"）能通过 SNLI，但过不了 HANS/ANLI。使用对抗基准测试
- **文档级退化。** 单句 NLI 模型在文档级前提上下降 20+ F1。长上下文使用 DocNLI 训练模型
- **零样本模板敏感度。** "This text is about {label}" vs "{label}" vs "The topic is {label}"——准确率可以摆动 10+ 个点。**在你的数据上调模板，不要用默认值**
- **领域不匹配。** MNLI 训练在通用英文上。法律、医疗、科学文本需要领域特定 NLI（SciNLI、MedNLI）

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

---

## 6. 知识连线

- **阶段 05 · 13（问答系统）→** RAG 忠实度检查的 NLI 层是阶段 13 RAGAS 框架的底层引擎
- **阶段 05 · 27（LLM 评估框架）→** RAGAS/DeepEval 的忠实度指标直接调用 NLI 模型——阶段 27 的 LLM-as-Judge 校准方法同样适用于 NLI 分数
- **阶段 05 · 05（情感分析）→** NLI 的三路分类（蕴含/矛盾/中立）和情感分析的三路分类（正面/负面/中性）在模型架构上完全相同——都是 `[CLS]` → 3-way softmax

---

## 7. 常见错误

### 错误 1：用 NLI 分数作为唯一的幻觉判断依据

**现象：** 系统报告忠实度 95%，但人工抽查发现 20% 的答案包含了检索文档中没有的信息。

**原因：** NLI 模型在训练分布外的文本上可能产生高置信度的错误蕴含判断。特别是当生成答案使用了检索文档中的关键词但重新组织了含义时——NLI 容易被词汇重叠误导为"蕴含"。

**修复：** 始终在目标领域的 50 条样本上做人工 vs NLI 一致性校准。对于分数在 0.5-0.9 之间的"灰色地带"答案，标记为需要人工复查。

### 错误 2：零样本分类用英文模板跑中文文本

**现象：** 中文文档 + 英文标签描述"This text is about sports" → 分类准确率接近随机。

**原因：** NLI 模型在跨语言的前提-假设对上没有被训练。英文假设 + 中文前提 → 模型在两种语言之间翻译——引入了翻译偏差。

**修复：** 始终将假设模板的语言与被分类文本保持一致。中文文本用中文标签描述。或使用多语言 NLI 模型（如 `multilingual-MiniLMv2-L6-mnli-xnli`）。

---

## 8. 面试考点

### Q1：为什么 SNLI 上的 90% 准确率不代表模型"理解了语言"？（难度：⭐⭐）

**参考答案：**
因为 hypothesis-only 基线——仅靠阅读假设（不看前提）——就能达到 ~60% 的准确率。"not"、"never"、"nobody"等否定词在 SNLI 中与矛盾标签高度相关。如果模型 90% 中的 60% 来自这些表面关联——剩余的 30% 才是真正的推理能力。ANLI（对抗 NLI）专门设计来打破这些表面关联——人类写手刻意构造了让统计捷径失效的例子——SOTA 在 ANLI 上低于 60%。

### Q2：如何在自己的领域中校准 NLI 忠实度阈值？（难度：⭐⭐⭐）

**参考答案：**
三步：(1) 在 50 条领域样本上同时跑 NLI 和人工标注——得到二元标签（忠实/不忠实）和 NLI 蕴含分数。(2) 绘制 PR 曲线——在不同阈值下的精确率和召回率。(3) 选择使 F1 最大化的阈值——这个值几乎永远不会是默认的 0.5。在金融/法律等对假阳性容忍度低的领域——调高阈值（0.7-0.8）以最小化"把幻觉误判为忠实"的风险。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| NLI | "自然语言推理" | 前提-假设关系的三路分类 |
| RTE | "文本蕴含识别" | NLI 的旧称；同一任务 |
| 蕴含 (Entailment) | "前提推出假设" | 典型读者读完前提会得出结论说假设为真 |
| 矛盾 (Contradiction) | "前提排除假设" | 典型读者读完前提会得出结论说假设为假 |
| 中立 (Neutral) | "不确定" | 前提不提供判断假设真假的依据 |
| 零样本分类 | "NLI 当分类器" | 将标签文字变成假设，选蕴含分数最高的类 |
| 忠实度 (Faithfulness) | "答案有出处吗？" | NLI over (检索上下文, 生成答案)——不蕴含 = 幻觉 |

---

## 📚 小结

NLI 是 NLP 中最"安静"的基础设施——摘要验证、RAG 事实核查、零样本分类全部依赖它。2026 年的默认是 DeBERTa-v3-large-mnli（英文）或多语言 XLM-R NLI（中文）。零样本模板敏感度可达 10+ 个点的准确率摆动——在你的数据上校准模板和阈值，不要信任默认值。

NLI 不等于解释了幻觉——它减少了幻觉，但没有消除。分数在 0.5-0.9 之间的"灰色地带"答案始终需要人工复查。NLI 是文本理解的万能胶带——任何时候你需要"前提是否支撑假设"，先用 NLI，再考虑再调用一次 LLM。

---

## ✏️ 练习

1. 【理解】用 `facebook/bart-large-mnli` 在 20 个手写的（前提, 假设, 标签）三元组上跑三路推断——覆盖所有三类。衡量准确率。加入对抗性"子序列启发式"陷阱（"I did not eat the cake" vs "I ate the cake"），看它是否被破解。

2. 【实现】对比三个零样本模板——`"This text is about {label}"`、`"The topic is {label}"`、`"{label}"`——在 100 条 AG News 标题上的准确率。报告准确率的摆动幅度。

3. 【实验】构建一个 RAG 忠实度检查器：原子声明分解 + 每条声明做 NLI 对照。在 50 条 RAG 生成的带有标准上下文的答案上评估。衡量与人工标注对比的假阳性率和假阴性率。

4. 【思考】你的 NLI 忠实度系统在金融研报上假阳性率很高——它频繁将"公司营收增长了 20%"判为蕴含，但实际原文是"公司营收增长了 12%"。NLI 模型为什么会被精确数字的近似值误导？你会如何修复？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 可复用提示词 | `outputs/skill-nli-picker.md` | 按场景选择 NLI 模型、模板和阈值的系统化方案 |

---

## 📖 参考资料

1. [论文] Bowman et al. "A large annotated corpus for learning natural language inference". EMNLP, 2015. https://arxiv.org/abs/1508.05326 — SNLI
2. [论文] Williams, Nangia, Bowman. "A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference". NAACL, 2018. https://arxiv.org/abs/1704.05426 — MultiNLI
3. [论文] Nie et al. "Adversarial NLI: A New Benchmark for Natural Language Understanding". ACL, 2020. https://arxiv.org/abs/1910.14599 — ANLI 基准
4. [论文] He et al. "DeBERTa: Decoding-enhanced BERT with Disentangled Attention". ICLR, 2021. https://arxiv.org/abs/2006.03654 — 2026 NLI 主力模型
5. [论文] Yin, Hay, Roth. "Benchmarking Zero-shot Text Classification". EMNLP, 2019. https://arxiv.org/abs/1909.00161 — NLI-as-classifier

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文 NLI 建议、工程最佳实践、常见错误、面试考点等均为原创内容。
