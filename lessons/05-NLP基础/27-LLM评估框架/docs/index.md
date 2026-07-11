# LLM 评估——RAGAS、DeepEval、G-Eval

> 完全匹配和 F1 漏掉了语义等价。人工审阅无法规模化。LLM-as-Judge 是生产环境的答案——经过足够的校准让你能信任这个数字。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05 · 13（问答系统）、05 · 14（信息检索） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 3

---

## 🎯 学习目标

- [ ] 使用 RAGAS 评估 RAG 系统的四个维度——忠实度、答案相关性、上下文精确率/召回率
- [ ] 理解 LLM-as-Judge 的三个校准检查——评分分布、位置偏差、长度偏差
- [ ] 区分三个框架——RAGAS（轻量）、DeepEval（CI/CD）、G-Eval（自定义标准）

---

## 1. 问题

你的 RAG 系统回答："2007年6月29日。"参考答案："June 29, 2007。"EM=0，F1=0%，人工=100%。乘以 1 万条 × 每次变更 = 需要一个理解语义、在规模上廉价运行、对回归不撒谎的评估器。

2026 年三个框架在解决这个问题：**RAGAS**（四个 RAG 专属指标，NLI+LLM-Judge 后端）、**DeepEval**（LLM 的 Pytest，CI/CD 原生）、**G-Eval**（LLM-as-Judge + 思维链 + 自定义标准）。

---

## 2. RAGAS 四个维度

| 指标 | 衡量什么 | 怎么算 |
|---|---|---|
| **忠实度** | 答案中的每个声明是否来自上下文 | 原子声明 → NLI 逐条对照 → 蕴含比例 |
| **答案相关性** | 答案是否直接回应了问题 | 反向生成假设问题 → 与原始问题的语义相似度 |
| **上下文精确率** | 检索块中多少相关 | 与答案相关的句子 / 总句子数 |
| **上下文召回率** | 相关上下文被检索到了多少 | 答案中每条声明在上下文中找证据 |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
```

### DeepEval——LLM 的 Pytest

```python
from deepeval.metrics import GEval
metric = GEval(name="Correctness", criteria="...", evaluation_steps=[...])
assert_test(test_case, [metric])
```

---

## 3. LLM-as-Judge 三个校准检查

1. **评分分布。** 100 条上 LLM vs 人工 Pearson 相关系数应 > 0.7。不到 → 评分标准太模糊
2. **位置偏差。** 交换候选 A/B 顺序——分数是否一致？LLM 可能系统性偏好第一个。偏差存在 → 随机化顺序或取两轮平均
3. **长度偏差。** LLM 是否系统性给长答案打高分？是 → 加长度惩罚或在标准中排除

---

## 4. 陷阱

- **LLM-as-Judge 也会幻觉。** 评分模型可能在它不理解的领域上产生高置信的错误分数。始终在 50 条目标领域样本上做人工一致性校准
- **位置偏差比想象的更普遍。** 即使交换顺序后差异不大——随机化的标准差有时超过 0.1——对于以 0.05 为差异阈值的评估流程是灾难性的
- **G-Eval 的标准设计比模型选择更重要。** 模糊的标准 → LLM 猜测你想要的 → 分数不可复现。具体、可操作的标准 → 稳定的跨模型评分

---

## 5. 中文特别建议

- **中文 LLM-as-Judge 用中文作为评判语言。** 英文 prompt 评估中文输出 = 翻译偏差
- **中文忠实度原子声明拆分适配中文语法。** RAGAS 默认英文拆分对中文连词支持有限——自定义按"。"、"；"和"而且/但是"拆分
- **中文同义表达更丰富——LLM-as-Judge 是唯一能捕捉的方案。** "2007 年 6 月 29 日"和"2007 年六月末"在 EM/F1 下全不匹配——只有 LLM 能给满分

---

## 6. 框架选择

| 框架 | 场景 |
|---|---|
| RAGAS | RAG 四个维度评估。轻量、有研究支撑 |
| DeepEval | CI/CD 管道中的 LLM 输出断言。Pytest 原生 |
| G-Eval | 自定义评分标准 + 思维链 + 0-1 分数 |

---

## 🔑 关键术语 | 📚 小结

RAGAS 四个维度量化 RAG 质量；DeepEval 嵌入 CI/CD；G-Eval 自定义标准。LLM-as-Judge 的核心前提——"语义等价需要语义判断"——是正确的。但 LLM 法官本身需要校准：评分分布、位置偏差、长度偏差三者缺一不可。**永远在目标领域的 50 条样本上做人工 vs LLM 一致性校准。**

---

## ✏️ 练习

1. 【理解】用 RAGAS 对你自己的 RAG 系统跑 200 条评估，手工复核忠实度最低的 20 条——确认打分与你的判断一致。
2. 【实现】用 DeepEval 在 CI 管道中加入三个 G-Eval 指标——每次 PR 变更自动跑。
3. 【思考】你的 LLM-as-Judge 对长答案系统性打高分——如何仅用提示词修改消除这个偏差？

---

## 📖 参考资料

1. [GitHub] RAGAS. https://github.com/explodinggradients/ragas
2. [GitHub] DeepEval. https://github.com/confident-ai/deepeval

---

> 本课程参考了 AI Engineering From Scratch 的课程体系。中文 LLM 评估建议为原创内容。
