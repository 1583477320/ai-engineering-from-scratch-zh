# 问答系统——抽取式、检索增强与生成式

> 三套系统塑造了现代 QA。抽取式从文章中找到答案片段。检索增强式把它建立在文档基础上。生成式直接从参数记忆里出答案。每一个现代 AI 助手都是这三者的混合。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 05 · 11（机器翻译）、阶段 05 · 10（注意力机制）
**预计时间：** ~75 分钟
**所处阶段：** Tier 1

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分三种 QA 架构——抽取式、RAG、生成式——各自的适用场景和失败模式
- [ ] 实现 EM（完全匹配）和词元级 F1——理解为什么 EM 太严格而 F1 给部分分数
- [ ] 理解检索增强（RAG）的"先搜后读"流程——这是现代 QA 的标准范式

---

## 1. 问题

"第一代 iPhone 什么时候发布的？" 用户期望的答案是"2007 年 6 月 29 日"——不是一个模糊的概述，不是脱离了句子的孤立年份。

过去十年，三套架构统治了 QA：

- **抽取式 QA。** 给定一个问题 + 一段已知包含答案的文章，找到答案在文章中的起点和终点。SQuAD 是其标准基准。答案永远是原文中的一段——从根源上杜绝了幻觉，也从根源上限制了"如果文章不包含答案就无解"
- **开放域 QA（RAG）。** 不给定文章——先检索相关段落，再从中抽取或生成答案。这是今天每个 RAG 流水线的底层范式
- **生成式 / Closed-book QA。** 大语言模型直接从参数记忆中回答。最快，在常见知识上最可靠，在罕见或最新事实上最不可靠

2026 年的趋势是混合：检索最佳的几个段落，然后让生成式模型在那些段落的约束下回答。这就是 RAG——阶段 05 · 14 将深入检索部分。

---

## 2. 概念

| | 抽取式 | RAG（检索增强生成） | 生成式（Closed-book） |
|---|---|---|---|
| **答案来源** | 文章中的连续片段 | 检索到的段落 → 生成/抽取 | 模型训练权重 |
| **幻觉风险** | 零（只输出原文） | 低（以检索结果为基础） | 中-高 |
| **需要检索步骤** | 是（文章已给定） | 是 | 否 |
| **覆盖范围** | 1 篇文章 | 整个语料库 | 训练数据中的所有知识 |
| **代表模型** | BERT + QA head | RAG, REALM, Atlas | GPT-4, Claude, Llama |
| **2026 适用** | 封闭域、给定文章 | 开放域、多文档 | 常识、对话 |

### 评估：EM 和 F1

- **EM（Exact Match）= 1** 当且仅当标准化后的预测和真实答案字符串 100% 匹配。太严格——"2007" 和 "2007年6月29日" → EM=0
- **F1（词元级）=** 预测与真实答案在词元上的精确率/召回率的调和平均。给部分匹配以部分分数

生产环境用 `squad` 指标包（`pip install evaluate` + `evaluate.load("squad")`），自动处理大小写、标点、冠词归一化。

---

## 3. 从零实现

### EM 与 F1

```python
def exact_match(pred, gold):
    return 1.0 if normalize(pred) == normalize(gold) else 0.0

def token_f1(pred, gold):
    p_tokens, g_tokens = tokenize(normalize(pred)), tokenize(normalize(gold))
    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)
```

```python
>>> exact_match("2007年6月29日", "2007年6月29日")  # 1.0
>>> exact_match("2007年", "2007年6月29日")          # 0.0 ← EM 不给部分分
>>> token_f1("2007年", "2007年6月29日")             # 0.50 ← F1 给了半分
```

### 玩具检索——先搜后读

```python
def toy_retrieve(question, top_k=2):
    """简单词汇重叠得分——教学用的 BM25 近似。"""
    q_tokens = set(tokenize(question))
    scored = [(Σ BM25_score(token, doc) for token in q_tokens, doc)
              for doc in corpus]
    return top_k of scored
```

完整的 QA 流程 = `retrieve(question) → reader(retrieved_docs, question) → answer`。检索和阅读可以分别训练、分别评估。

完整代码见 `code/qa_demo.py`。

---

## 4. 工业工具

```python
# 抽取式
from transformers import pipeline
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
qa(question="When was the first iPhone released?",
   context="Apple released the first iPhone on June 29, 2007.")

# RAG（检索+生成一体）
from transformers import RagTokenizer, RagTokenForGeneration
# 或直接用 LangChain / LlamaIndex 的检索+LLM组合

# 评估
import evaluate
squad_metric = evaluate.load("squad")
```

---

## 5. 常见错误

**抽取式 QA 在未找到答案时不拒绝回答。** 抽取式模型被训练为"总是输出一个 span"——即使文章中没有答案。生产环境的 QA 系统必须在 reader 之前或之后加上"无答案检测"——通常通过预测的 start/end logits 的置信度阈值。

**中文 QA 中分词不一致导致 EM=0。** "2007年6月29日" vs "2007 年 6 月 29 日"——空格差异导致 EM 完全失败。标准化步骤必须包括去空格、全半角统一、数字归一化。

---

## 🔑 关键术语

| 术语 | 实际含义 |
|---|---|
| 抽取式 QA | 从给定文章中找出答案的 start/end 位置——输出永远是原文片段 |
| RAG | 检索增强生成——先搜相关文档，再基于文档生成答案。2026 开放域 QA 的默认范式 |
| EM / F1 | QA 的两个标准指标。EM 要 100% 匹配，F1 给部分分数 |
| 无答案检测 | 抽取式 QA 的必须组件——模型默认会输出一个 span，即使答案不存在 |

---

## 📚 小结

三种 QA 架构覆盖了从"给定文章"到"全互联网"的答案来源范围。抽取式 QA 零幻觉但受限于给定文章，RAG 用检索扩展了信息来源，生成式 QA 最快但最不可靠。2026 的生产系统是混合——先检索，再在检索结果上生成。

---

## ✏️ 练习

1. 【实现】为玩具检索器加入段落级别切分——长文章切为 3-5 句的段落，每段独立索引和检索
2. 【实验】用 HuggingFace 的 `question-answering` pipeline 在中文 SQuAD 格式数据上测试 10 个问题。报告 EM 和 F1。分析 F1 > 0.8 但 EM = 0 的案例——是什么差异导致完全匹配失败？

---

## 📖 参考资料

1. [论文] Rajpurkar et al. "SQuAD: 100,000+ Questions for Machine Comprehension of Text". EMNLP, 2016. https://arxiv.org/abs/1606.05250
2. [论文] Lewis et al. "RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". NeurIPS, 2020. https://arxiv.org/abs/2005.11401

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
