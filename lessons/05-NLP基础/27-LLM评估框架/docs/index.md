# LLM 评估——RAGAS、DeepEval、G-Eval

> 完全匹配和 F1 漏掉了语义等价。人工审阅无法规模化。LLM-as-Judge 是生产环境的答案——经过足够的校准让你能信任这个数字。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05 · 13、05 · 14 | **预计时间：** ~75 分钟 | **所处阶段：** Tier 3

---

## 🎯 学习目标

- [ ] 理解 LLM-as-Judge 的核心原理——用 LLM 评分替代静态指标，给定评分标准和参考答案
- [ ] 使用 RAGAS 评估 RAG 系统的四个维度——忠实度、答案相关性、上下文精确率、上下文召回率
- [ ] 解释为什么 LLM-as-Judge 需要校准——评分分布、人工一致性、位置偏差检查

---

## 1. 问题

你的 RAG 系统回答："2007年6月29日。"参考答案是："June 29, 2007。"完全匹配打分：0。F1 打分：~0%。人工评分：100%。

把这个问题乘以 1 万个测试用例。再乘以检索器、分块、提示词、模型的每一次变更。你需要一个评估器——它理解语义、在规模上廉价运行、不对回归撒谎、并把正确的失败模式浮到表面。

2026 年有三个框架在解决这个问题。

---

## 2. RAGAS——RAG 系统的四个评估维度

RAGAS = Retrieval-Augmented Generation ASsessment。四个 RAG 专属指标：

| 指标 | 衡量什么 | 怎么算 |
|---|---|---|
| **忠实度 (Faithfulness)** | 答案中的每个声明是否来自上下文 | 将答案拆为原子声明 → 用 NLI 模型逐条对照上下文判断蕴含/矛盾 |
| **答案相关性** | 答案是否直接回答了问题 | 对答案生成反向问题 → 与原始问题的语义相似度 |
| **上下文精确率** | 检索到的上下文中多少是相关的 | 在上下文中找出与答案相关的句子 / 总句子数 |
| **上下文召回率** | 相关上下文被检索到了多少 | 对答案中的每条声明在上下文中查找证据 |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                     context_precision, context_recall])
```

### DeepEval——LLM 的 Pytest

```python
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

metric = GEval(name="Correctness", criteria="..."
               evaluation_steps=[...], evaluation_params=[])
assert_test(test_case, [metric])
```

---

## 3. LLM-as-Judge 的三个校准检查

在使用 LLM 评分的数字之前，先做三件事：

1. **评分分布检查。** 随机抽取 100 个评估结果，手工打分。LLM 和人工之间的 Pearson 相关系数应该 > 0.7。如果不到——评分标准太模糊或矛盾
2. **位置偏差检查。** 交换候选 A 和 B 的位置。分数是否保持一致？LLM 可能系统性地偏好第一个出现的选择——如果存在这种偏差，随机化顺序或取两轮平均
3. **长度偏差检查。** LLM 是否系统性地给更长的答案打高分？如果是——加入长度惩罚或在评分标准中明确要求"长度不作为评判依据"

---

## 4. 中文 LLM 评估特别建议

- **中文 LLM-as-Judge 应该用中文作为评判语言。** 用英文 prompt 评估中文输出 → LLM 在翻译/理解过程中引入额外的偏差。评分 prompt 的语言应与被评估内容一致
- **中文忠实度的原子声明拆分需要适配中文语法。** "苹果很好吃而且很便宜" → RAGAS 默认的英文拆分（逗号 + and）对中文逗号和中文连词的支持有限。自定义中文拆分策略——按"。"、"；"和"而且/但是/此外"等连词拆分
- **中文同义表达比英文更丰富——LLM-as-Judge 是唯一能捕捉这一点的方案。** "2007 年 6 月 29 日" 和 "2007 年六月末" 在严格 EM/F1 下全不匹配——只有 LLM-as-Judge 能给满分

---

## 5. 三个框架的选择

| 框架 | 场景 |
|---|---|
| RAGAS | RAG 系统的四个维度评估。轻量、有研究支撑 |
| DeepEval | CI/CD 管道中的 LLM 输出断言。Pytest 原生支持 |
| G-Eval (方法) | 自定义评分标准 + 思维链 + 0-1 分数。嵌入在 DeepEval 中 |
| MT-Bench / Chatbot Arena | 对话质量评估。不是框架——是排行榜和基准 |

---

## 📚 小结 | ✏️ 练习

LLM-as-Judge 是 2026 年 LLM 评估生产的答案——在成本、规模和语义理解三个维度上优于人工+静态指标。RAGAS 用四个维度量化 RAG 质量。LLM-as-Judge 需要校准——评分分布、位置偏差、长度偏差三者缺一不可。练习：用 RAGAS 对你自己的 RAG 系统做 200 条的评估批次，手工复核忠实度分数最低的 20 条——确认打分是否与你的判断一致。

---

> 本课程参考了 AI Engineering From Scratch 的课程体系，在此基础上进行了重构和原创内容的扩充。中文 LLM 评估建议为原创内容。
