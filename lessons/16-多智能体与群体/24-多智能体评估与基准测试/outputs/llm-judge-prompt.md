# LLM-as-Judge 评估提示词模板

> 用于多智能体系统评估中，用 LLM 作为裁判对比两个方案的输出质量。

---

## 使用说明

将以下提示词发送给裁判 LLM（如 GPT-4o），替换 `{变量}` 部分为实际值。裁判 LLM 应该与被评估的系统使用不同的模型，以减少偏差。

---

## 中文版提示词

```
你是评估专家。请对比以下两个方案的输出质量。

## 任务
{task_description}

## 方案 A（单智能体）
{single_output}

## 方案 B（多智能体）
{multi_output}

## 评估维度
请从以下 4 个维度对两个方案分别评分（1-5 分）：

1. **准确性**：信息是否正确，是否有事实错误
2. **完整性**：是否覆盖了任务要求的所有要点
3. **一致性**：内部是否自相矛盾，多个部分是否协调
4. **可读性**：结构是否清晰，是否易于理解

## 输出格式
请严格返回以下 JSON 格式：

{
  "single": {
    "accuracy": N,
    "completeness": N,
    "consistency": N,
    "readability": N
  },
  "multi": {
    "accuracy": N,
    "completeness": N,
    "consistency": N,
    "readability": N
  },
  "winner": "single" 或 "multi",
  "reason": "一句话说明为什么选择该方案"
}

## 注意事项
- 评分应基于客观标准，不要因为方案 B 更复杂就给更高分
- 如果两个方案差距很小（总分差 ≤ 2 分），winner 应为 "tie"
- reason 应该具体说明关键差异，而不是泛泛而谈
```

---

## 英文版提示词

```
You are an evaluation expert. Compare the output quality of the following two approaches.

## Task
{task_description}

## Approach A (Single Agent)
{single_output}

## Approach B (Multi-Agent)
{multi_output}

## Evaluation Dimensions
Score each approach on the following 4 dimensions (1-5 scale):

1. **Accuracy**: Are the facts correct? Any errors?
2. **Completeness**: Does it cover all required points?
3. **Consistency**: Is it internally coherent? No contradictions?
4. **Readability**: Is the structure clear? Easy to understand?

## Output Format
Return strictly in the following JSON format:

{
  "single": {"accuracy": N, "completeness": N, "consistency": N, "readability": N},
  "multi": {"accuracy": N, "completeness": N, "consistency": N, "readability": N},
  "winner": "single" or "multi" or "tie",
  "reason": "One sentence explaining the key difference"
}

## Notes
- Score based on objective criteria, not complexity
- If the difference is small (total score gap ≤ 2), winner should be "tie"
- reason should be specific, not generic
```

---

## 使用示例

**输入：**
- task_description：分析上季度销售数据并生成报告
- single_output：（单智能体生成的报告）
- multi_output：（多智能体协作生成的报告）

**预期输出：**
```json
{
  "single": {"accuracy": 4, "completeness": 3, "consistency": 4, "readability": 4},
  "multi": {"accuracy": 4, "completeness": 5, "consistency": 3, "readability": 4},
  "winner": "multi",
  "reason": "多智能体方案覆盖了更多数据维度（竞争对手分析、用户画像），但两处数据引用存在轻微不一致"
}
```

---

## 偏差控制

- **模型多样性**：裁判 LLM 应该与被评估系统使用不同的模型（如用 GPT-4o 评估 Claude 系统）
- **位置偏差**：随机交换方案 A 和方案 B 的顺序，运行两次取平均
- **评分校准**：先用 10 个已知质量的样本校准裁判的评分标准
- **成本控制**：每次评估约 1000-2000 词元，100 个测试用例约 10-20 美元
