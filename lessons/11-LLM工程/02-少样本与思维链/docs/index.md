# 少样本、思维链与思维树

> 告诉模型做什么是提示词。展示如何思考是工程。在同一模型、同一任务、同一数据上从 78% 到 91% 的准确率差距不是更好的模型，而是更好的推理策略。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 11 · 01（提示词工程）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现少样本提示——选择和格式化最大化任务准确率的示例
- [ ] 应用思维链（CoT）推理提高多步问题的准确率
- [ ] 构建思维树（ToT）提示——探索多个推理路径并选择最佳
- [ ] 测量零样本 vs 少样本 vs CoT 的准确率改进

---

## 1. 问题

你构建了一个数学辅导应用。提示词说："解答这个问题。" GPT-5 在 GSM8K（标准小学数学基准）上正确率 94%。你觉得已经到顶了——不——思维链（CoT）仍然增加了 3-4 个百分点。

加五个字——"让我们一步步思考"——准确率跳到 91%。加几个工作示例后达到 95%。同样的模型、同样的温度、同样的 API 成本。唯一的区别是你给了模型打草稿的空间。

这不是 hack。这是推理的工作方式。人类不会在一次心理跳跃中解决多步问题。Transformer 也不会。当你强迫模型生成中间词元时，这些词元成为下一个词元的上下文——每一推理步喂养下一步。

---

## 2. 概念

### 2.1 零样本 vs 少样本 vs 思维链

| 方法 | 示例 | 准确率 | 适用场景 |
|------|------|--------|---------|
| **零样本** | 无示例 | 基线 | 简单任务 |
| **少样本 (Few-Shot)** | 2-5 个输入-输出对 | +5-10% | 需要上下文内学习 |
| **思维链 (CoT)** | "让我们一步步思考" | +3-4% | 数学/逻辑推理 |
| **自洽性 (Self-Consistency)** | 采样多个路径，多数投票 | +2-3% | 高价值任务 |

### 2.2 思维链

```
输入：小明有 12 个苹果，给了小红 3 个，然后买了 5 个。他现在有几个？

思维链输出：
1. 小明开始有 12 个苹果
2. 给了小红 3 个，所以还有 12 - 3 = 9 个
3. 又买了 5 个，所以现在有 9 + 5 = 14 个
4. 答案是 14
```

### 2.3 自洽性

思维链的延伸——用温度 > 0 采样多个推理路径，取绝大多数：

```
路径 1: ... 答案是 14
路径 2: ... 答案是 14
路径 3: ... 答案是 13
结论：14（2/3 多数）
```

---

## 3. 从零实现

### Step 1：少样本提示

```python
def few_shot_prompt(query, examples):
    """构建少样本提示。"""
    prompt = ""
    for i, (q, a) in enumerate(examples, 1):
        prompt += f"示例 {i}:\n问题: {q}\n答案: {a}\n\n"
    prompt += f"现在回答:\n问题: {query}\n答案:"
    return prompt


def zero_shot_cot_prompt(query):
    """零样本思维链——加一句话。"""
    return f"{query}\n\n让我们一步步思考。"
```

### Step 2：自洽性

```python
def self_consistency(model, prompt, n_paths=5, temperature=0.7):
    """采样多个路径，提取多数答案。"""
    responses = []
    for _ in range(n_paths):
        response = model.generate(prompt, temperature=temperature)
        responses.append(response)

    answers = [extract_answer(r) for r in responses]
    return max(set(answers), key=answers.count)  # 多数投票
```

### Step 3：思维树（ToT）

```python
def tree_of_thought(prompt, n_branches=3, max_depth=3):
    """思维树——探索多个推理分支。"""
    def evaluate(path):
        """评估推理路径的质量。"""
        return model.score(path)  # 简化版

    best_path = []
    for depth in range(max_depth):
        candidates = []
        for _ in range(n_branches):
            next_step = model.generate(prompt + "\n".join(best_path))
            candidates.append((evaluate(next_step), next_step))

        candidates.sort(reverse=True)  # 选择最好的分支
        best_path.append(candidates[0][1])

    return best_path
```

---

## 4. 工具

### 4.1 LangChain 的 FewShotPromptTemplate

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"question": "2+2=?", "answer": "4"},
    {"question": "3×3=?", "answer": "9"},
]
template = PromptTemplate(input_variables=["question", "answer"],
                          template="问题: {question}\n答案: {answer}")

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=template,
    suffix="问题: {input}\n答案:",
    input_variables=["input"],
)
```

---

## 6. 工程最佳实践

### 6.1 选择少样本示例

- **覆盖多样性**：示例应覆盖不同难度和类型
- **代表性**：示例应类似测试时的输入分布
- **正确性**：一个错误示例会显著降低准确率
- **数量**：2-5 个通常足够；更多可能导致注意力分散

### 6.2 中文场景特别建议

- 数学题使用中文思维链："让我们一步步计算"替代"Let's think step by step"
- 中文示例让模型更好理解中文语境

### 6.3 踩坑经验

- **示例错误**：少样本示例中的错误会被模型复制——始终验证示例
- **温度太高**：自洽性需要温度 > 0，但 > 1 可能产生无意义路径
- **思维链太长**：推理步骤过多会稀释注意力——3-7 步最佳

---

## 7. 常见错误

### 错误 1：少样本示例有错误

**现象：** 模型复制了示例中的错误答案或推理模式。

**原因：** 模型从示例中学习模式——一个错误的示例比十个正确的更"有效"。

**修复：** 始终验证示例的正确性。使用多样化的示例覆盖不同情况。

### 错误 2：CoT 但不验证最终答案

**现象：** 思维链看起来逻辑正确，但最终答案是错的。

**原因：** 模型可以编造看似合理但错误的推理链。CoT 提升了"看起来对"的概率，但不能消除错误。

**修复：** 结合自洽性——采样多个推理路径取多数。或用程序验证数学计算。

### 错误 3：少样本示例太多导致注意力稀释

**现象：** 添加 10 个示例后，模型反而更混乱。

**原因：** 模型在长上下文中的注意力会分散——过多示例的关键模式被淹没。

**修复：** 2-5 个示例通常最优。更多示例应该覆盖不同类型而非重复同一类型。

---

## 8. 面试考点

### Q1：思维链为什么能提升准确率？（难度：⭐⭐）

**参考答案：**
Transformer 通过逐词元预测来生成。没有思维链时，模型必须从一个表示中直接产生答案——"一步到位"。思维链让模型在答案前生成中间推理词元，每个推理步为下一个步提供上下文。模型"一边算一边写"，有效的上下文窗口在推理过程中动态扩展。

### Q2：自洽性和思维树有什么区别？（难度：⭐⭐⭐）

**参考答案：**
自洽性用高温度并行采样多个独立推理路径，然后对最终答案做多数投票——路径之间没有交互。思维树在每个推理步探索多个分支，评估每个分支的质量，选择最佳分支继续探索——路径在每一步都被评估和修剪。思维树更适合分支空间大、中间步骤可以评估的复杂问题，但成本更高。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 少样本 (Few-Shot) | "给例子" | 在提示词中提供输入-输出示例，模型通过上下文内学习完成类似任务 |
| 思维链 (CoT) | "一步步想" | 在生成答案前先生成中间推理步骤——将多步推理分解 |
| 自洽性 | "多数投票" | 用高温度采样多个推理路径，取答案的绝大多数 |
| 思维树 (ToT) | "探索多条路" | 在每个推理步采样多个候选，评估选择最佳路径继续推进 |

---

## 📚 小结

少样本提示提供示例替代指令；思维链添加中间推理步骤提升准确率；自洽性用多数投票降低方差；思维树在每个推理步做选择和剪枝。这些方法逐步增加计算成本和复杂度，但准确率也逐级提升。

---

## ✏️ 练习

1. **【实现】** 在数学推理数据集上比较零样本、少样本、思维链的准确率。
2. **【实验】** 实现自洽性——采样 5 条路径取多数。对比单路径的准确率提升。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| CoT/ToT 实现 | `code/main.py` | 少样本、CoT、自洽性、思维树实现 |

---

## 📖 参考资料

1. [论文] Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models". NeurIPS, 2022. https://arxiv.org/abs/2201.11903
2. [论文] Wang et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models". ICLR, 2023. https://arxiv.org/abs/2203.11171
3. [论文] Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models". NeurIPS, 2023. https://arxiv.org/abs/2305.10601

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
