# Reflexion：语言强化学习

> 基于梯度的 RL 需要数千次试验和 GPU 集群来修复一个失败模式。Reflexion 用自然语言做到这一点：每次失败试验后，智能体写一段反思，存储在情景记忆中，然后基于该记忆进行下一次试验。这是 Letta 的睡眠时计算、Claude Code 的 CLAUDE.md 学习、pro-workflow 的 learn-rule 背后的模式。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、02（ReWOO）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 命名 Reflexion 的三个组件（Actor、Evaluator、Self-Reflector）并说明情景记忆的作用
- [ ] 实现自我反思循环——生成→评估→反思→重试
- [ ] 理解 Reflexion 与 RLHF 的区别——语言反馈 vs 奖励信号
- [ ] 设计一个基于 Reflexion 的错误修复系统

---

## 1. 问题

基于梯度的 RL 需要数千次试验和 GPU 集群来修复一个失败模式。但 LLM 无法通过梯度优化来改进——它是预训练的，权重是固定的。

Reflexion 的解决方案：**用自然语言做"强化学习"**——失败后写反思、存记忆、下一次试验时带上反思。

```
试验 1: 写代码 → 测试失败 → 反思："我忘记了边界情况"
试验 2: [带反思] 写代码 → 测试失败 → 反思："我需要更多错误处理"
试验 3: [带两次反思] 写代码 → 测试通过！
```

---

## 2. 概念

### 2.1 Reflexion 三组件

| 组件 | 功能 | 类比 |
|------|------|------|
| **Actor** | 执行任务（如生成代码） | 演员 |
| **Evaluator** | 评估结果（如运行测试） | 评委 |
| **Self-Reflector** | 生成反思（为什么失败） | 演员的自我批评 |

### 2.2 情景记忆

每次试验的反思存储在情景记忆中——下一次试验时，Actor 可以看到之前所有失败的反思。

```
情景记忆 = [
  "试验1: 忘记了边界情况 → 需要检查空数组",
  "试验2: 整数溢出 → 使用大整数类型",
]
```

### 2.3 Reflexion 与 RLHF 的区别

| 方面 | RLHF | Reflexion |
|------|------|-----------|
| 反馈类型 | 标量奖励 | 自然语言反思 |
| 学习方式 | 梯度更新 | 语言条件化 |
| 适用模型 | 可微分模型 | 所有 LLM |
| 存储 | 权重更新 | 情景记忆 |

---

## 3. 从零实现

### Step 1：Reflexion Agent

```python
class ReflexionAgent:
    """Reflexion 智能体。"""
    def __init__(self, actor_fn, evaluator_fn, reflector_fn):
        self.actor_fn = actor_fn
        self.evaluator_fn = evaluator_fn
        self.reflector_fn = reflector_fn
        self.memory = []  # 情景记忆

    def run(self, task, max_trials=3):
        for trial in range(max_trials):
            # 1. Actor 执行任务
            result = self.actor_fn(task, context=self.memory)

            # 2. Evaluator 评估
            score, feedback = self.evaluator_fn(result)

            if score >= 0.8:
                return result, "成功"

            # 3. Self-Reflector 生成反思
            reflection = self.reflector_fn(task, result, feedback)
            self.memory.append(reflection)
            print(f"  试验 {trial+1}: 评分={score:.2f}, 反思={reflection[:50]}...")

        return None, "达到最大试验次数"
```

### Step 2：自我反思器

```python
def self_reflector(task, result, feedback):
    """生成自我反思。"""
    return f"任务'{task[:20]}...'的结果不理想。{feedback} 下次需要改进：1) 检查更多边界情况 2) 添加错误处理。"
```

---

## 4. 工具

### 4.1 框架支持

| 框架 | Reflexion 支持 |
|------|---------------|
| LangGraph | 通过状态图实现 |
| Letta | 内置情景记忆 |

---

## 5. 工程最佳实践

### 5.1 Reflexion 设计原则

- **反思要具体**：不是"做错了"，而是"忘记了边界条件"
- **记忆有限制**：只保留最近 N 条反思——避免噪声
- **试验有上限**：设置最大试验次数防止无限循环

### 5.2 踩坑经验

- **反思质量差**：LLM 反思太泛——用具体评估结果引导
- **记忆膨胀**：情景记忆无限增长——定期清理旧反思

---

## 6. 常见错误

### 错误 1：反思没有基于具体错误

**现象：** "做得不好"——太泛，无法指导改进。

**修复：** 反思必须基于评估器的具体反馈——"测试3失败，因为缺少空数组检查"。

### 错误 2：最大试验次数设太小

**现象：** 智能体还没收敛就被终止。

**修复：** 通常需要 3-5 次试验——每次试验至少修复一个错误。

---

## 7. 面试考点

### Q1：Reflexion 和 RLHF 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
RLHF 通过梯度更新模型权重——需要可微分的模型和大量数据。Reflexion 通过自然语言反思改进——不需要梯度，只需要情景记忆。RLHF 更新的是模型的"知识"，Reflexion 更新的是模型的"行为策略"。Reflexion 可以用于任何 LLM（包括不更新权重的情况），RLHF 需要可训练的模型。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Reflexion | "自我反思学习" | 每次失败后写反思并存储，下一次试验时带上反思改进 |
| 情景记忆 | "历史记录" | 存储所有失败试验的反思——供后续试验使用 |
| Self-Reflector | "自我批评器" | 分析失败原因并生成自然语言反思的组件 |
| 睡眠时计算 | "离线学习" | Letta 的概念——在不活跃时处理反思和记忆 |

---

## 📚 小结

Reflexion 用自然语言做"强化学习"——失败后写反思、存记忆、重试。三个组件：Actor（执行）、Evaluator（评估）、Self-Reflector（反思）。与 RLHF 的区别：语言反馈 vs 梯度更新。适用于所有 LLM，不需要可训练模型。

---

## ✏️ 练习

1. **【实现】** 构建一个 Reflexion 智能体——在编程任务上测试失败→反思→重试
2. **【实验】** 对比有/无情景记忆的智能体性能——反思记忆提升多少？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| Reflexion Agent | `code/main.py` | Actor + Evaluator + Self-Reflector |

---

## 📖 参考资料

1. [论文] Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning". NeurIPS, 2023. https://arxiv.org/abs/2303.11366
2. [论文] Letta. "Letta: A Framework for Building Stateful LLM Applications". 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
