# 评估与基准测试

> 你无法改进你无法衡量的东西。LLM 评估不是选一个 benchmark 跑分——而是设计一个测量体系来回答"模型在我的任务上有多好？"

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 10 · 10（评估）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 设计 LLM 评估体系——选择合适的标准 benchmark 和自定义评估
- [ ] 实现 LLM-as-Judge 自动评估流水线
- [ ] 理解人类评估与自动化评估的权衡
- [ ] 构建生产环境的持续监控和回归测试

---

## 1. 问题

你微调了一个 LLM。它"看起来"更好了——但怎么证明？"我感觉"不是评估。你需要一个可量化、可复现、可比较的评估体系。

---

## 2. 概念

### 2.1 标准 Benchmark 分类

| 类型 | Benchmark | 衡量什么 |
|------|-----------|---------|
| **知识** | MMLU, ARC | 多学科知识问答 |
| **推理** | GSM8K, MATH, BBH | 数学和逻辑推理 |
| **代码** | HumanEval, MBPP | 代码生成和修复 |
| **对话** | MT-Bench, Arena | 多轮对话质量 |
| **指令跟随** | IFEval, AlpacaEval | 遵循指令的能力 |

### 2.2 评估方法

| 方法 | 优点 | 缺点 |
|------|------|------|
| **规则匹配** | 快速、确定性 | 无法评估开放式输出 |
| **LLM-as-Judge** | 快速、灵活 | 偏差和不一致 |
| **人类评估** | 金标准 | 昂贵、慢、不可复现 |
| **混合评估** | 兼顾质量和效率 | 配置复杂 |

### 2.3 评估流程设计

```
1. 明确评估目标 → "模型在我的场景下表现如何？"
2. 选择评估集 → 标准 benchmark + 自定义测试集
3. 选择评估方法 → LLM-as-Judge + 规则匹配
4. 运行评估 → 自动化流水线
5. 分析结果 → 对比、统计检验
6. 持续监控 → 生产环境回归测试
```

---

## 3. 从零实现

### Step 1：LLM-as-Judge 评估框架

```python
class LLMJudge:
    """LLM 自动评判者。"""
    def __init__(self, model_fn):
        self.model_fn = model_fn

    def evaluate(self, task, response, reference=None):
        prompt = f"""评估以下回答的质量（1-10分）。

任务：{task}
回答：{response}
{"参考答案：" + reference if reference else ""}

评分（1-10）:"""

        score = self.model_fn(prompt)
        return self._parse_score(score)

    def _parse_score(self, text):
        import re
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else 5
```

### Step 2：批量评估

```python
def batch_evaluate(test_cases, judge_fn, generate_fn):
    results = []
    for task, reference in test_cases:
        response = generate_fn(task)
        score = judge_fn(task, response, reference)
        results.append({"task": task, "response": response, "score": score})
    avg_score = sum(r["score"] for r in results) / len(results)
    return avg_score, results
```

---

## 6. 工程最佳实践

### 6.1 评估集设计

- **代表性**：评估集应反映真实使用场景的分布
- **多样性**：覆盖简单、中等、困难任务
- **足够规模**：至少 100 个样本才能可靠估计
- **定期更新**：避免评估集被"记忆"

### 6.2 中文场景

- 中文 benchmark：CMMLU, C-Eval, CMMLU-Pro
- LLM-as-Judge 用中文模型评估中文输出效果更好

---

## 7. 面试考点

### Q1：为什么 LLM-as-Judge 可能有偏差？如何缓解？（难度：⭐⭐）

**参考答案：**
LLM-as-Judge 的偏差：(1) 位置偏差——倾向选择第一个/最后一个选项；(2) 长度偏差——倾向更长的回答；(3) 自我偏好——GPT-4 偏好 GPT-4 的输出。缓解：随机化选项顺序、使用多个不同的 Judge 模型、强制指定评分标准（rubric）。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| LLM-as-Judge | "用 AI 评估 AI" | 用大模型作为自动评估者——快速但有偏差 |
| 基准测试 (Benchmark) | "标准化考试" | 标准化的评估数据集和指标——可跨模型比较 |
| 人类评估 | "真人打分" | 邀请人类评估者对模型输出进行评分——金标准但昂贵 |

---

## 7. 常见错误

### 错误 1：只用一个 benchmark 评估

**现象：** 模型在 MMLU 上表现好，但在实际任务上表现差。

**原因：** 单一 benchmark 只能评估一个维度——知识好不代表推理好、对话好。

**修复：** 使用多维度评估组合——知识(MMLU) + 推理(GSM8K) + 对话(MT-Bench) + 自定义测试集。

### 错误 2：LLM-as-Judge 没有校准

**现象：** 自动评估分数与人类评估不一致。

**原因：** LLM Judge 有位置偏差（倾向选第一个）和长度偏差（倾向更长的回答）。

**修复：** 随机化选项顺序，使用多个不同的 Judge 模型取平均，定期用人类评估校准。

### 错误 3：评估集被模型"记忆"

**现象：** 模型在评估集上得分高但在生产环境中表现差。

**原因：** 模型在预训练或微调时见过评估集数据——不是真正的能力提升。

**修复：** 定期更换评估集，使用 holdout 评估集做最终验证，记录评估集的发布时间。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| LLM-as-Judge 框架 | `code/main.py` | 自动评估流水线 + 批量测试 |

---

## 📚 小结

LLM 评估需要标准化 benchmark + 自定义测试集 + 人类评估的组合。LLM-as-Judge 快速但有偏差——需要多个 Judge 和标准化 rubric。生产环境需要持续监控和回归测试。

---

## ✏️ 练习

1. **【实现】** 设计一个 LLM-as-Judge 流水线，评估 10 个问答对的质量
2. **【实验】** 对比 GPT-4 和 Claude 作为 Judge 的一致性

---

## 📖 参考资料

1. [论文] Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena". NeurIPS, 2023.
2. [论文] Hendrycks et al. "Measuring Massive Multitask Language Understanding". ICLR, 2021.
