# 宪法 AI 与自我改进

> 人类反馈太贵了。宪法 AI 说：给 AI 一套规则，让它自己判断自己的输出好不好——不需要人类介入。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 10 · 06（SFT）、07（RLHF）| **时间：** ~45 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 07（RLHF）— 宪法 AI 是 RLHF 的替代 | 阶段 10 · 08（DPO）— DPO 可以用 AI 偏好作为数据源

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解宪法 AI 的两步流程——监督阶段（自我修正）和 RL 阶段（偏好学习）
- [ ] 解释自我奖励模型——LLM 评估自己的输出
- [ ] 说明 Constitutional AI vs RLHF 的成本差异

---

## 1. 问题

RLHF 需要人类标注偏好对——Anthropic 花了数亿美元标注 RLHF 数据。这个成本对大多数团队不可承受。

宪法 AI（Constitutional AI, Bai et al. 2022）说：**给 AI 一本"宪法"——一套原则性规则——让它自己判断输出是否符合规则，自己修改自己。**

两步流程：
1. **监督阶段**：AI 生成回答 → 根据宪法规则自我修正 → SFT
2. **RL 阶段**：AI 判断自己的输出是否"合宪" → 训练奖励模型 → PPO

关键创新：**人类只需要写规则，不需要标注数据。** AI 同时是生成者、评判者和修正者。

---

## 2. 概念

### 2.1 宪法规则

宪法是一套简短的原则性规则，例如：

```python
CONSTITUTION = [
    "请选择最有帮助且无害的回答。",
    "请选择不包含有害、不道德或非法内容的回答。",
    "请选择不包含种族歧视或性别歧视的回答。",
    "请选择最真实、最准确的回答。",
    "请选择最清晰、最有条理的回答。",
]
```

Anthropic 的宪法约有 16 条规则——涵盖无害性、有帮助性、诚实性。

### 2.2 监督阶段（Critique + Revision）

```
AI 生成回答 R0
  ↓
AI 根据宪法自我批评："这个回答有什么问题？"
  ↓
AI 自我修正："修改后应该是..."
  ↓
用修正后的回答做 SFT
```

### 2.3 RL 阶段（AI 偏好）

```
AI 为每个提示词生成两个回答 R1, R2
  ↓
AI 根据宪法判断："R1 还是 R2 更合宪？"
  ↓
用 AI 偏好训练奖励模型
  ↓
用 PPO + KL 惩罚优化策略
```

### 2.4 宪法 AI vs RLHF

| 方面 | RLHF | 宪法 AI |
|------|------|---------|
| 反馈来源 | 人类标注 | AI 自我判断 |
| 成本 | 极高（数百万美元） | 低（写规则即可） |
| 可扩展性 | 低（标注速度慢） | 高（AI 可并行评判） |
| 一致性 | 人类标注有噪声 | AI 判断更一致 |
| 局限性 | — | AI 可能有自己的偏见 |

---

## 3. 实现思路

```python
# 宪法 AI 的监督阶段
def constitutional_revision(model, prompt, constitution):
    """AI 自我修正。"""
    # 1. 生成初始回答
    response = model.generate(prompt)

    # 2. 批评
    critique_prompt = f"根据以下规则，批评这个回答：{constitution}\n\n回答：{response}"
    critique = model.generate(critique_prompt)

    # 3. 修正
    revision_prompt = f"根据批评修改回答：\n批评：{critique}\n原始回答：{response}"
    revised = model.generate(revision_prompt)

    return response, revised
```

---

## 4. 工具

### 4.1 Anthropic Claude

Claude 是宪法 AI 的主要产品——使用宪法 AI 训练的有帮助、诚实、无害的助手。

### 4.2 RLAIF（AI 反馈替代人类）

```python
# 使用 AI 生成偏好对——替代人类标注
def generate_ai_preferences(model, prompt, num_pairs=10):
    pairs = []
    for _ in range(num_pairs):
        r1 = model.generate(prompt)
        r2 = model.generate(prompt)
        # AI 判断哪个更好
        judge_prompt = f"哪个回答更好？\nA: {r1}\nB: {r2}"
        preference = model.generate(judge_prompt)
        pairs.append((prompt, r1 if "A" in preference else r2,
                           r2 if "A" in preference else r1))
    return pairs
```

---

## 5. LLM 视角

### 5.1 宪法 AI 的影响

- **Claude 的"灵魂"**：Anthropic 用宪法 AI 训练 Claude 的核心道德准则
- **可审计性**：宪法是人类可读的规则——不像 RLHF 的隐式偏好
- **可扩展性**：AI 反馈可以并行处理数百万样本

### 5.2 与 RLHF/DPO 的关系

宪法 AI 不是 RLHF/DPO 的替代——而是数据生成方法。它可以用 AI 生成的偏好对替代人类标注的偏好对，然后用 DPO 或 RLHF 训练。

---

## 6. 工程最佳实践

### 6.1 宪法规则设计

- **具体 > 抽象**：写"不要回答医疗建议"比"要无害"更有效
- **可操作**：规则应该可以被 AI 理解和执行
- **层次化**：安全规则 > 有帮助性规则 > 风格规则

### 6.2 踩坑经验

- **AI 偏见**：AI 的"自我判断"可能有偏见——需要定期用人类评估校准
- **规则冲突**：多条规则可能互相矛盾——优先级需要明确

---

## 7. 常见错误

### 错误 1：规则太模糊

**现象：** AI 无法一致地判断"什么是有帮助的"。

**修复：** 用具体的例子定义规则——"有帮助=回答了用户的问题且提供了相关上下文"。

---

## 8. 面试考点

### Q1：宪法 AI 和 RLHF 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
RLHF 需要人类标注偏好对——成本高但质量可靠。宪法 AI 用 AI 自我判断替代人类标注——成本低但可能有偏见。宪法 AI 的监督阶段让模型自我修正，RL 阶段用 AI 偏好训练奖励模型。最终效果与 RLHF 相当，但数据获取成本低几个数量级。

### Q2：宪法 AI 的主要局限性是什么？（难度：⭐⭐⭐）

**参考答案：**
(1) **AI 偏见**：AI 的"自我判断"继承了其训练数据的偏见——它可能认为有偏见的回答是"合宪"的；(2) **规则覆盖不全**：宪法无法覆盖所有边缘情况；(3) **自我修正循环**：AI 可能在批评和修正之间陷入循环——每次修正引入新的问题。解决方法：定期用人类评估校准 AI 判断。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 宪法 AI | "给 AI 立法" | 用一套可读规则指导 AI 行为——无需人类标注 |
| 宪法 | "AI 的道德准则" | Anthropic 定义的 16 条原则——无害、有帮助、诚实 |
| 自我修正 | "AI 改自己的作业" | AI 批评自己的输出并修正——监督学习数据的来源 |
| RLAIF | "AI 反馈替代人类" | 用 AI 生成偏好对替代人类标注——大幅降低成本 |
| 宪法 AI 阶段 | "自我改进" | 监督修正 + RL 偏好的两阶段循环 |

---

## 📚 小结

宪法 AI 用 AI 自我判断替代昂贵的人类标注——大幅降低对齐训练成本。监督阶段让模型自我修正，RL 阶段用 AI 偏好训练。核心优势：可审计、可扩展、成本低。核心局限：AI 偏见、规则覆盖不全。2026 年 Claude 就是用宪法 AI 训练的。

---

## ✏️ 练习

1. **【实验】** 设计 5 条中文宪法规则（安全/隐私/公平），在 MiniGPT 上实现宪法 AI 监督阶段。
2. **【思考】** 如果 AI 的"自我判断"有偏见，如何检测和修正？设计一个校准流程。

---

## 📖 参考资料

1. [论文] Bai et al. "Constitutional AI: Harmlessness from AI Feedback". arXiv, 2022. https://arxiv.org/abs/2212.08073
2. [论文] Bai et al. "Training a Helpful and Harmless Assistant with RLHF". arXiv, 2022. https://arxiv.org/abs/2204.05862

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
