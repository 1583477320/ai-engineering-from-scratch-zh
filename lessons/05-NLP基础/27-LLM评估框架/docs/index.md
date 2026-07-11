# LLM 评估——RAGAS、DeepEval、G-Eval

> 完全匹配和 F1 漏掉了语义等价。人工审阅无法规模化。LLM-as-Judge 是生产环境的答案——经过足够的校准让你能信任这个数字。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05 · 13（问答系统）、05 · 14（信息检索） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 使用 RAGAS 评估 RAG 系统的四个维度——忠实度、答案相关性、上下文精确率/召回率
- [ ] 理解 LLM-as-Judge 的三个校准检查——评分分布、位置偏差、长度偏差
- [ ] 为 CI/CD 管道搭建 DeepEval + G-Eval 自动化评估——每次 PR 变更自动跑

---

## 1. 问题

你的 RAG 系统回答："2007年6月29日。"参考答案："June 29, 2007。"完全匹配：0。F1：~0%。人工评分：100%。

把这个问题乘以 1 万个测试用例。再乘以检索器、分块、提示词、模型的每一次变更。你需要一个评估器——它理解语义、在规模上廉价运行、不对回归撒谎、并把正确的失败模式浮到表面。

2026 年有三个框架在解决这个问题。**RAGAS**（四个 RAG 专属指标，NLI+LLM-Judge 后端，研究支撑，轻量）。**DeepEval**（LLM 的 Pytest，G-Eval + 幻觉 + 偏差指标，CI/CD 原生）。**G-Eval**（一个方法——也是 DeepEval 的一个指标——LLM-as-Judge + 思维链 + 自定义标准 + 0-1 分数）。三者都依赖 LLM-as-Judge——本课建立对这个方法以及围绕它的信任层的直觉。

---

## 2. 概念

### 2.1 LLM-as-Judge 架构

用 LLM 替代静态指标——给定评分标准对输出打分。给定 `(query, context, answer)`，提示一个法官 LLM："在忠实度上打分 0-1。"返回分数。

```
(查询, 上下文, 答案) → [法官 LLM + 评分标准] → 0-1 分数
```

**为什么这比 EM/F1 更好。** "2007年6月29日" vs "June 29, 2007"——EM=0，F1~0。LLM-as-Judge 正确地给了满分——因为它理解语义等价。**但为什么它仍然需要校准。** LLM 法官本身有偏差——位置偏差（系统性偏好第一个选项）、长度偏差（系统性偏好更长的答案）、评分分布偏差（比人类更慷慨或更苛刻）。

### 2.2 RAGAS——四个无参考答案的 RAG 指标

RAGAS = Retrieval-Augmented Generation ASsessment。四个维度——不需要参考答案：

| 指标 | 衡量什么 | 后端 |
|---|---|---|
| **忠实度** | 答案中每个声明是否来自上下文 | NLI 逐声明蕴含判断 |
| **答案相关性** | 答案是否直接回应问题 | 反向生成假设问题 → 语义相似度 |
| **上下文精确率** | 检索块中多少相关 | 与答案相关的句子占比 |
| **上下文召回率** | 相关上下文被检索到了多少 | 答案声明在上下文中找证据 |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                     context_precision, context_recall])
```

### 2.3 DeepEval——LLM 的 Pytest

```python
from deepeval import assert_test
from deepeval.metrics import GEval, HallucinationMetric, BiasMetric
from deepeval.test_case import LLMTestCase

# G-Eval: 自定义标准 + 思维链 + 0-1 分数
metric = GEval(
    name="Correctness",
    criteria="答案是否在事实层面正确并得到上下文的充分支撑？",
    evaluation_steps=[
        "检查答案中的每个数字/日期/名称是否与上下文一致",
        "检查答案中是否有上下文未提供的额外声称",
        "综合判定一个 0-1 分数",
    ],
    evaluation_params=[],
)
test_case = LLMTestCase(input="...", actual_output="...", context=["..."])
assert_test(test_case, [metric])
```

G-Eval 的评分标准设计比模型选择更重要——模糊的标准 → LLM 猜测你想要什么 → 分数不可复现。具体、可操作的标准 → 跨模型的稳定评分。

---

## 3. LLM-as-Judge 的三个校准检查

在信任 LLM 评分数字之前，先做三件事：

### 检查 1：评分分布

随机抽取 100 个评估结果，手工打分。LLM 和人工之间的 **Pearson 相关系数应 > 0.7**。如果不到——评分标准太模糊或矛盾——在扩大评估规模之前先修改标准。

### 检查 2：位置偏差

交换候选 A 和 B 的位置。分数是否保持一致？LLM 可能系统性偏好第一个出现的选择。如果偏差存在——随机化顺序或取两轮平均。位置偏差比想象中更普遍——即使交换后差异不大，随机化的标准差有时超过 0.1——对于以 0.05 为差异阈值的评估流程是灾难性的。

### 检查 3：长度偏差

LLM 是否系统性给更长的答案打高分？如果是——加入长度惩罚或在评分标准中明确要求"长度不作为评判依据"。长度偏差是 LLM-as-Judge 中最普遍也最容易忽视的系统性偏差。

---

## 4. 陷阱

- **LLM-as-Judge 也会幻觉。** 评分模型在它不理解的领域上可能产生高置信的错误评分。始终在 50 条目标领域样本上做人工一致性校准
- **RAGAS 忠实度 ≠ 零幻觉保证。** NLI 模型可能被词汇重叠误导——将"公司营收增长 20%"判为蕴含，实际原文是"增长 12%"。精确数字的近似值是 NLI 忠实度的盲区
- **G-Eval 的评分标准即文档。** 模糊的标准 → 不可复现的分数。具体、可操作的标准 → 跨模型稳定性。在团队中把评分标准当作代码 review

---

## 5. 工业工具——2026 技术栈

| 框架 | 场景 | 特点 |
|---|---|---|
| RAGAS | RAG 四维度评估 | 轻量、研究支撑、NLI 后端 |
| DeepEval | CI/CD 管道 | Pytest 原生、G-Eval+幻觉+偏差指标 |
| G-Eval（方法） | 自定义评分 | 思维链 + 自定义标准 + 0-1 分数 |
| MT-Bench / Chatbot Arena | 对话质量排行 | 排行榜和基准——不是可嵌入的评估框架 |

### 中文 LLM 评估特别建议

- **中文 LLM-as-Judge 用中文作为评判语言。** 英文 prompt 评估中文输出 → LLM 在翻译/理解过程中引入额外偏差。评分 prompt 语言应与被评估内容一致
- **中文忠实度原子声明拆分适配中文语法。** RAGAS 默认英文拆分（逗号+and）对中文连词支持有限——自定义按"。"、"；"和"而且/但是/此外"等拆分
- **中文同义表达更丰富——LLM-as-Judge 是唯一能捕捉的方案。** "2007 年 6 月 29 日"和"2007 年六月末"在 EM/F1 下全不匹配——只有 LLM 能给满分

---

## 6. 知识连线

- **阶段 05 · 13（问答系统）→** RAGAS 四个维度直接用于评估阶段 13 的 RAG 流水线
- **阶段 05 · 21（NLI）→** RAGAS 忠实度指标的底层引擎就是 NLI 模型——阶段 21 的 NLI 陷阱（词汇重叠、模板敏感度）在 RAGAS 中同样适用

---

## 7. 面试考点

### Q1：为什么 EM/F1 不够用于评估 LLM 输出？（难度：⭐⭐）

**参考答案：**
因为 EM/F1 衡量的是表面形式的匹配——而非语义等价。"2007年6月29日" vs "June 29, 2007"——EM=0，F1=0%——但语义完全相同。当输出从"抽取原文中的精确 span"变为"LLM 用自己的话重述答案"时——EM/F1 系统性低估了真实质量。LLM-as-Judge 用语义理解替代了表面匹配——但代价是引入了 LLM 自己的偏差——需要校准。

### Q2：如何在 CI/CD 中嵌入 LLM 评估而不让 pipeline 慢到不可用？（难度：⭐⭐⭐）

**参考答案：**
三层漏斗——**(1)** 每次 PR 跑 RAGAS 四个指标（不需要参考答案，轻量）——耗时约 1-2 分钟 / 200 条。**(2)** 仅在 RAGAS 指标下降 > 5% 时触发 G-Eval 深度评估（自定义标准，更贵）。**(3)** 对忠实度分数最低的 20 条——人工复核。这样 95% 的 PR 只经过第一层——CI 总延迟控制在 2 分钟内——只有出问题时才自动升级。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| LLM-as-Judge | "用 LLM 评分" | 用 LLM 替代静态指标——给出评分标准和参考答案后打分 |
| RAGAS | "RAG 的四个分数" | 忠实度、答案相关性、上下文精确率、上下文召回率——四个维度 |
| G-Eval | "自定义评分" | 思维链 + 自定义标准 + 0-1 分数。嵌入在 DeepEval 中 |
| 忠实度 | "有没有胡说" | 答案中每个声明是否来自上下文。**你的主要幻觉指标** |
| Pearson 相关系数 | "LLM 和人的一致性" | > 0.7 = LLM 法官可用。在 100 条上校准 |

---

## 📚 小结

LLM-as-Judge 是 2026 年 LLM 评估生产的答案——在成本、规模和语义理解三个维度上优于人工+静态指标。RAGAS 用四个维度量化 RAG 质量。LLM-as-Judge 需要校准——评分分布、位置偏差、长度偏差三者缺一不可。G-Eval 评分标准 = 代码——模糊带来不可复现。**三层 CI/CD 漏斗：RAGAS 快筛 → G-Eval 深度评估 → 人工复核。**

---

## ✏️ 练习

1. 【理解】用 RAGAS 对你自己的 RAG 系统跑 200 条评估。手工复核忠实度最低的 20 条——确认 LLM 法官是否与你的判断一致。

2. 【实现】用 DeepEval 在 CI 管道中加入三个 G-Eval 指标——正确性、忠实度、答案完整性。每次 PR 变更自动跑。

3. 【实验】对比三个 LLM 法官（GPT-4o、Claude Sonnet、本地 Llama）在同一批 RAG 输出上的评分一致性。哪个法官与人工评分最接近？跨法官的评分方差有多大？

4. 【思考】你的 RAG 系统在忠实度上得分 0.95——但人工抽查发现 15% 的答案包含了上下文没有的信息。LLM 法官哪里出了问题？如何修复？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 可复用提示词 | `outputs/skill-eval-architect.md` | 按场景设计评估流水线的系统化方案 |

---

## 📖 参考资料

1. [GitHub] RAGAS. https://github.com/explodinggradients/ragas — RAG 评估框架
2. [GitHub] DeepEval. https://github.com/confident-ai/deepeval — LLM 的 Pytest
3. [论文] Liu et al. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment". EMNLP, 2023. https://arxiv.org/abs/2303.16634 — G-Eval 论文

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文 LLM 评估建议、工程最佳实践、常见错误、面试考点等均为原创内容。
