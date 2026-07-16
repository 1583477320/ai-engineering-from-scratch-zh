# Self-Refine 和 CRITIC：迭代输出改进

> Self-Refine（Madaan 等人，2023）用一个 LLM 扮演三个角色——生成、反馈、改进——循环执行。平均提升：在 7 个任务上 +20 个绝对分。CRITIC（Gou 等人，2023）通过外部工具验证强化了反馈步骤。2026 年这个模式在每个框架中以"评估器-优化器"（Anthropic）或护栏循环（OpenAI Agents SDK）的形式发布。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、03（Reflexion）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 说明 Self-Refine 的三个提示词（生成、反馈、改进）及其历史的作用
- [ ] 实现 Self-Refine 循环——生成→自我批评→改进
- [ ] 理解 CRITIC 如何用外部工具验证反馈
- [ ] 对比 Self-Refine 和 RLHF 的成本和适用场景

---

## 1. 问题

LLM 的第一次输出通常不是最好的——它没有"编辑"过程。Self-Refine 让 LLM 扮演三个角色：

1. **生成器**：产出初始回答
2. **反馈器**：批评这个回答——"哪里不好？为什么？"
3. **改进器**：根据批评修改回答

这个循环平均在 7 个任务上提升 20 个绝对分——证明了"自我反思"对 LLM 输出质量的重要性。

---

## 2. 概念

### 2.1 Self-Refine 三角色

| 角色 | 提示词 | 输出 |
|------|--------|------|
| **生成器** | "写一个关于X的回复" | 初始回答 |
| **反馈器** | "评价这个回复的质量和准确性" | 批评和建议 |
| **改进器** | "根据以下批评改进回复" | 改进后的回答 |

### 2.2 历史在改进中的作用

改进器需要看到**初始回答 + 批评**——否则它不知道要改进什么。这与 Reflexion 不同——Reflexion 跨试验记忆，Self-Refine 在单次循环内记忆。

### 2.3 CRITIC——工具增强的反馈

Self-Refine 的反馈依赖 LLM 的自我判断——可能有偏差。CRITIC 用外部工具验证反馈：

```
生成器: 写一段 Python 代码
反馈器: 运行代码，检查是否通过测试
改进器: 根据测试结果修复代码
```

### 2.4 Self-Refine vs RLHF vs Reflexion

| 方面 | Self-Refine | RLHF | Reflexion |
|------|-------------|------|-----------|
| 反馈来源 | LLM 自我批评 | 人类偏好 | 任务评估 |
| 改进方式 | 重新生成 | 梯度更新 | 语言条件化 |
| 模型更新 | 不更新权重 | 更新权重 | 不更新权重 |
| 成本 | 低（同模型多轮） | 高（训练） | 中（多轮试验） |

---

## 3. 从零实现

### Step 1：Self-Refine 循环

```python
class SelfRefineAgent:
    """Self-Refine 智能体——生成→批评→改进。"""
    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def generate(self, task):
        """步骤 1: 生成初始回答。"""
        return self.llm_fn(f"写一个关于'{task}'的回复")

    def feedback(self, task, response):
        """步骤 2: 自我批评。"""
        return self.llm_fn(f"评价这个回复的质量: {response[:50]}...")

    def refine(self, task, response, feedback):
        """步骤 3: 根据批评改进。"""
        return self.llm_fn(f"根据以下批评改进回复:\n反馈: {feedback}\n原始: {response[:50]}")

    def self_refine_loop(self, task, max_iterations=3):
        """Self-Refine 循环。"""
        # 步骤 1: 初始生成
        response = self.generate(task)
        print(f"  生成: {response[:60]}...")

        for i in range(max_iterations):
            # 步骤 2: 自我批评
            feedback = self.feedback(task, response)
            print(f"  批评 ({i+1}): {feedback[:50]}...")

            # 步骤 3: 改进
            response = self.refine(task, response, feedback)
            print(f"  改进 ({i+1}): {response[:60]}...")

        return response
```

---

## 4. 工具

### 4.1 框架支持

| 框架 | 支持方式 |
|------|---------|
| Anthropic | "evaluator-optimizer" 模式 |
| OpenAI Agents SDK | 护栏循环 |
| LangChain | 自定义链 |

---

## 5. 工程最佳实践

### 5.1 Self-Refine 设计

- **最大迭代次数**：2-3 次——超过后改进边际递减
- **反馈要具体**：不是"不好"而是"逻辑有漏洞，需要补充条件X"
- **结合外部验证**：用测试用例/验证器替代纯 LLM 反馈

### 5.2 踩坑经验

- **过度改进**：3 次以上迭代后质量可能下降——因为每次修改引入新错误
- **反馈不一致**：LLM 的自我评价可能不稳定——用多个评估提示词
- **成本累积**：3 次迭代 = 3 倍 LLM 调用——在延迟敏感场景需权衡

---

## 6. 常见错误

### 错误 1：Self-Refine 变成无限循环

**现象：** 模型在批评→改进之间反复跳转。

**修复：** 设置最大迭代次数（通常 2-3 次）——超过后停止。

### 错误 2：反馈太泛

**现象：** "做得不够好"——无法指导具体改进。

**修复：** 要求反馈包含具体问题和改进建议。

---

## 7. 面试考点

### Q1：Self-Refine 的三个角色是什么？为什么需要分离？（难度：⭐⭐）

**参考答案：**
三个角色：生成器（产出初始回答）、反馈器（批评质量和准确性）、改进器（根据批评修改回答）。分离的原因：(1) 同一个提示词无法同时做"写"和"评估"——需要不同的上下文；(2) 反馈需要看到原始回答——改进需要看到反馈+原始回答；(3) 分离使每一步更专注——生成器专注创意，反馈器专注质量，改进器专注修正。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Self-Refine | "自我改进循环" | 生成→批评→改进的三步循环——平均提升 +20 分 |
| CRITIC | "工具增强的反馈" | 用外部工具（测试、验证器）替代纯 LLM 反馈 |
| 评估器-优化器 | "Anthropic 的 Self-Refine" | Anthropic 框架中的等价模式 |

---

## 📚 小结

Self-Refine 用一个 LLM 扮演三个角色——生成、批评、改进。CRITIC 用外部工具验证反馈。平均提升 +20 分。最大迭代 2-3 次——超过后边际递减。这是 2026 年每个智能体框架的标配——评估器-优化器模式。

---

## ✏️ 练习

1. **【实现】** 构建 Self-Refine 循环——在代码生成任务上测试 3 次迭代的改进效果
2. **【对比】** 对比纯 LLM 生成 vs Self-Refine 在创意写作上的质量差异

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| Self-Refine Agent | `code/main.py` | 生成→批评→改进循环 |

---

## 📖 参考资料

1. [论文] Madaan et al. "Self-Refine: Iterative Refinement with Self-Feedback". NeurIPS, 2023. https://arxiv.org/abs/2303.17651
2. [论文] Gou et al. "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing". ICLR, 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
