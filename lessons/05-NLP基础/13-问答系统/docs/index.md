# 问答系统——抽取式、检索增强与生成式

> 三套系统塑造了现代 QA。抽取式找到了答案片段。检索增强式把它建立在文档基础上。生成式直接从参数记忆里出答案。每一个现代 AI 助手都是这三者的混合。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 05 · 11（机器翻译）、阶段 05 · 10（注意力机制）
**预计时间：** ~75 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 14（信息检索）— 本课的检索层是阶段 14 四层架构的直接消费方

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 用 HuggingFace 的 `question-answering` pipeline 搭建抽取式 QA——理解为什么 `handle_impossible_answer=True` 不是默认行为
- [ ] 搭建两阶段 RAG 流水线——稠密检索（Sentence-BERT）→ 抽取式 reader，理解为什么检索和阅读必须分别评估
- [ ] 使用 RAGAS 评估 RAG 系统的四个维度——忠实度、答案相关性、上下文精确率、上下文召回率

---

## 1. 问题

用户输入"第一代 iPhone 什么时候发布的？"——期望的回答是"2007 年 6 月 29 日"。不是"苹果公司的历史源远流长"，不是孤零零的"2007"。是一个直接、有依据、正确的答案。

过去十年，三套架构统治了 QA：

- **抽取式 QA。** 给一个问题 + 一段已知包含答案的文章，找到答案在文章中的起点和终点。SQuAD 是其标准基准。答案永远是原文片段——从构造上杜绝幻觉，也从构造上拒绝回答文章不包含的问题
- **开放域 QA / RAG。** 不给定文章——先检索相关段落，再抽取或生成答案。这是今天每一个 RAG 流水线的底层范式
- **生成式 / Closed-book QA。** 大语言模型直接从参数记忆中回答。最快，在常识上最可靠，在罕见或最新事实上最不可靠

2026 年的趋势是混合：检索最佳的几个段落，然后让生成式模型在那些段落的约束下回答。这就是 RAG——本课构建 QA 侧，阶段 14 深入检索侧。

---

## 2. 概念

| | 抽取式 | RAG | 生成式 |
|---|---|---|---|
| **答案来源** | 文章中的连续片段 | 检索到的段落 → 生成/抽取 | 模型训练权重 |
| **幻觉风险** | 零（只输出原文） | 低（以检索结果为基础） | 中-高 |
| **需要检索** | 是（文章已给定） | 是 | 否 |
| **覆盖范围** | 1 篇文章 | 整个语料库 | 训练数据中的全部知识 |
| **2026 场景** | 封闭域、法规/合规要求原句引用 | 开放域、多文档 QA | 常识、对话 |

---

## 3. 从零实现

### 第 1 步：抽取式 QA——预训练模型

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "苹果公司于2007年6月29日发布了第一代iPhone。"
    "该设备由Steve Jobs在2007年1月的Macworld大会上宣布。"
)
question = "第一代iPhone是什么时候发布的？"

answer = qa(question=question, context=passage)
print(answer)
# {'score': 0.98, 'start': 10, 'end': 19, 'answer': '2007年6月29日'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练——包含不可回答的问题。默认情况下，`question-answering` pipeline **即使模型的 null score 更高也返回最高分的 span** ——它不会自动返回空答案。要获得显式的"无答案"行为，传 `handle_impossible_answer=True`，此时 pipeline 仅在 null score 超过所有 span score 时返回空答案。两种方式都应检查 `score` 字段。

中文抽取式 QA：`bert-base-chinese` + SQuAD 格式的 CMRC 2018 数据集 fine-tune。

### 第 2 步：检索增强流水线

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "苹果公司于2007年6月29日发布了第一代iPhone。",
    "Macworld 2007大会上Steve Jobs展示了iPhone。",
    "Android于2008年作为Google的移动操作系统推出。",
    "第一代iPod于2001年发布。",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)

def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]

def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)

print(answer("第一代iPhone什么时候发布的？"))
```

两阶段流水线。稠密检索器（Sentence-BERT）按语义相似度找到相关段落。抽取式 reader（RoBERTa-SQuAD）从合并后的 top 段落中拉出答案片段。在小型语料库上工作。百万级文档用 FAISS 或向量数据库。

### 第 3 步：RAG 生成式

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

只用上面的上下文回答。如果上下文中没有答案，说"我不知道"。
"""
    return llm(prompt)
```

**提示词模式本身很重要。** 显式告诉模型以上下文为基础、在上下文不足时返回"我不知道"——相比天真的提示词，降低 40-60% 的幻觉率。更复杂的模式增加引用标注、置信度评分和结构化提取。

### 第 4 步：反映真实世界的评估

SQuAD 使用 **EM（完全匹配）** 和**词元级 F1**。EM 在归一化（小写、去标点、去冠词）后进行严格匹配——预测与标准答案完全一致或得分 0。F1 基于预测与参考之间的词元重叠计算，给部分匹配以部分分数。两者都给 paraphrases 打分过低："2007年6月29日" vs "2007年6月29号"——EM=0（"号" vs "日"的差异），但 F1 仍有高分（大部分词元重叠）。

生产环境 QA 需要更多维度：

- **答案准确率**（LLM 判断或人工判断——因为 EM/F1 无法捕获语义等价）
- **引用准确率**——引用的段落是否确实支持该答案？
- **拒绝校准**——当检索段落中没有答案时，系统是否正确地说"我不知道"？衡量假自信率
- **检索召回率**——在评估 reader 之前，先衡量检索器是否将正确段落排进了 top-k。Reader 无法修复缺失的段落

### RAGAS——2026 年生产评估框架

`RAGAS` 专门为 RAG 系统打造，是 2026 年的部署默认。它无需参考答案，在四个维度上评分：

- **忠实度**——答案中的每个声明是否来自检索的上下文？基于 NLI 蕴含判断。你的主要幻觉指标
- **答案相关性**——答案是否回应了问题？从答案生成假设问题，与真实问题比较
- **上下文精确率**——检索到的块中有多少是真正相关的？低精确率 = prompt 中的噪音
- **上下文召回率**——检索集是否包含所有需要的信息？低召回率 = reader 无法成功

无参考答案评分让你可以在没有人工标注的生产流量上直接评估。对开放式问题叠加 LLM-as-judge。

`pip install ragas`。接入你的检索器 + reader。每个查询四个标量。对回归告警。

---

## 4. 工业工具——2026 技术栈

| 场景 | 推荐 |
|---|---|
| 给定文章、找答案片段 | `deepset/roberta-base-squad2`（英文）/ `bert-base-chinese` + CMRC（中文） |
| 固定语料库、不允许 closed-book | RAG：稠密检索器 + LLM reader |
| 实时文档存储 | RAG + 混合检索（BM25 + 稠密）+ 重排器（阶段 14） |
| 对话式 QA（有追问） | LLM + 对话历史 + 每轮 RAG |
| 高事实性、受监管域 | 抽取式 + 权威语料库。永远不单用生成式 |

抽取式 QA 在 2026 年显得"不时髦"——因为 RAG + LLM 处理了更多情况。但在需要原句引用的上下文中仍然在部署：法律研究、监管合规、审计工具。

---

## 5. 知识连线

QA 是本阶段前 12 课的第二个"整合点"——将检索（02/14）、注意力（10）、Seq2Seq（09）编织成端到端系统：

- **阶段 05 · 14（信息检索）→** 本课的检索层直接消费阶段 14 的四层架构——BM25 + 稠密 + RRF + 重排器可以替换本课的纯稠密检索
- **阶段 05 · 21（NLI）→** RAGAS 的忠实度指标底层就是 NLI 模型（阶段 21 的核心主题）
- **阶段 05 · 27（LLM 评估）→** RAGAS 的四个维度和 LLM-as-Judge 校准（阶段 27）是同一套方法论

---

## 6. 常见错误

### 错误 1：抽取式 QA 不设无答案检测就上线

**现象：** 用户问"iPhone 的电池容量是多少？"——文章中没有答案——模型返回了一个看起来合理的 span（"2007年"）。用户被误导。

**原因：** SQuAD-trained 模型被训练为"总是输出一个 span"。在没有无答案检测的情况下——模型对任何输入都会输出一个答案，无论文章是否包含。

**修复：** 传 `handle_impossible_answer=True`。或在 reader 之前加置信度阈值——当 top-1 检索分数 < 阈值时返回"我不知道"。

### 错误 2：QA 评估只看读者指标不看检索召回率

**现象：** RAG 系统的 EM/F1 很低——团队花了两周调读者模型。实际上检索器在 40% 的查询上根本没找到正确段落。

**原因：** 两个阶段串联评估——读者在垃圾输入上无法产生好的输出。不知道检索召回率就不知道瓶颈在哪里。

**修复：** 始终在上线前单独基准测试检索器的 Recall@5。如果 Recall@5 < 80%——先修检索，再修读者。

---

## 7. 面试考点

### Q1：为什么 EM 给"2007年6月29日"和"2007年6月29号"打 0 分？（难度：⭐⭐）

**参考答案：**
EM 是字符级的严格匹配——归一化只做小写、去标点、去冠词。"号" vs "日"在归一化后仍然不同，所以 EM=0。这说明 EM 是对"形式"的敏感指标，不是对"内容"的——两个答案在语义上完全等价但在形式上有差异。这就是为什么生产 QA 评估必须叠加 LLM-as-Judge 或人工判断——EM/F1 只是第一层过滤器。

### Q2：RAG 的检索召回率为什么必须先于读者评估进行基准测试？（难度：⭐⭐）

**参考答案：**
RAG 是一个串联系统——读者的输入质量完全依赖检索器的输出质量。如果检索器在 40% 的查询上没有排入正确段落（Recall@5 < 60%）——即使完美的读者也只能在 60% 的查询上正确回答。先独立评估每一段：检索器 Recall@k → 读者在给定正确段落上的准确率 → 端到端准确率。串联评估会让你把检索器和读者的问题混在一起，修错了方向。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 抽取式 QA | "从文章里找个片段" | 预测答案在给定文章中的 start/end 索引。输出永远是原文片段 |
| RAG | "先检索再生成" | 检索增强生成——检索器 + 阅读器的联合流水线 |
| SQuAD | "QA 的标准基准" | Stanford Question Answering Dataset。EM + F1 指标 |
| 忠实度 | "有没有胡说" | 答案中的每个声明是否来自检索上下文。RAGAS 的最核心指标 |
| 拒绝校准 | "该闭嘴时就闭嘴" | 系统在无法回答时正确返回"我不知道"的能力 |

---

## 📚 小结

三套 QA 架构覆盖了从"给定文章"到"全互联网"的答案来源范围。抽取式 QA 零幻觉但受限于给定文章，RAG 用检索扩展了信息来源，生成式 QA 最快但最不可靠。2026 的生产系统是混合——先检索，再在检索结果上生成。

RAGAS 四个维度（忠实度、答案相关性、上下文精确率、上下文召回率）让你在无标注的生产流量上直接评估 RAG 质量。永远先基准测试检索召回率再评估读者——串联评估不区分瓶颈。

---

## ✏️ 练习

1. 【理解】在 10 段 Wikipedia 文章上搭建 SQuAD 抽取式流水线。手写 10 个问题。衡量正确率——如果文章和问题都干净，应该能看到 7-9 题正确。

2. 【实现】加入拒绝分类器——当 top 检索分数低于阈值（如 0.3 余弦相似度）时返回"我不知道"而非调用 reader。在留出集上调阈值。

3. 【实验】在你选择的 1 万文档语料库上搭建 RAG 流水线。实现混合检索（BM25 + 稠密）+ RRF 融合（见阶段 14）。衡量有和没有混合步骤的答案准确率。记录哪些问题类型受益最大。

4. 【思考】对于法规/合规敏感问题，为什么生成式 QA——即使带有 RAG——仍然可能不够？你会增加什么额外的验证步骤？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| QA 从零实现 | `code/qa_demo.py` | EM/F1 评估 + 玩具 BM25 检索 + 三种架构对比 |

---

## 📖 参考资料

1. [论文] Rajpurkar et al. "SQuAD: 100,000+ Questions for Machine Comprehension of Text". EMNLP, 2016. https://arxiv.org/abs/1606.05250 — SQuAD 基准论文
2. [论文] Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". NeurIPS, 2020. https://arxiv.org/abs/2005.11401 — RAG 命名论文
3. [论文] Karpukhin et al. "Dense Passage Retrieval for Open-Domain QA". EMNLP, 2020. https://arxiv.org/abs/2004.04906 — DPR, QA 的经典稠密检索器
4. [论文] Gao et al. "Retrieval-Augmented Generation for Large Language Models: A Survey". 2023. https://arxiv.org/abs/2312.10997 — 全面的 RAG 综述

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、工程最佳实践、常见错误、面试考点等均为原创内容。
