# 多智能体辩论与协作

> Du 等人（ICML 2024，"Society of Minds"）运行 N 个模型实例，各自独立提出答案，然后迭代地互相批评，经过 R 轮收敛。提高了事实性、规则遵循、推理能力。稀疏拓扑比全连接在 token 成本上更优。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 12（工作流模式）、05（Self-Refine）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释辩论协议：N 个提议者，R 轮迭代，收敛到共享答案
- [ ] 实现多智能体辩论——多个 LLM 互相批评并改进
- [ ] 理解稀疏拓扑 vs 全连接在 token 成本上的差异
- [ ] 设计辩论系统的轮次和代理数量

---

## 1. 问题

单个 LLM 可能产生幻觉、推理错误或遗漏关键细节。**多个 LLM 辩论**让它们互相检查——每个代理提出答案，其他代理批评，经过多轮迭代收敛到更准确的答案。

---

## 2. 概念

### 2.1 辩论协议

```
轮次 1：N 个代理独立生成答案
轮次 2：每个代理看到其他代理的答案，提出批评
轮次 3：每个代理根据批评修改答案
...
收敛：N 个代理收敛到一致答案（或选出最佳）
```

### 2.2 拓扑结构

| 拓扑 | 描述 | Token 成本 |
|------|------|-----------|
| **全连接** | 每个代理看到所有其他代理的答案 | 高（N²） |
| **环形** | 每个代理只看下一个代理的答案 | 低（N） |
| **稀疏** | 每个代理看 K 个代理的答案 | 中（N·K） |

### 2.3 辩论效果

| 方面 | 单代理 | 多代理辩论 |
|------|--------|-----------|
| 事实性 | 有幻觉 | 减少幻觉 |
| 推理 | 可能遗漏 | 互相检查 |
| 一致性 | 单一观点 | 收敛到共识 |

---

## 3. 从零实现

### Step 1：多代理辩论

```python
class MultiAgentDebate:
    """多代理辩论系统。"""
    def __init__(self, agents, rounds=3):
        self.agents = agents
        self.rounds = rounds

    def debate(self, question):
        """执行辩论。"""
        # 轮次 1：独立生成
        answers = [agent.generate(question) for agent in self.agents]
        print(f"  轮次 1: {[a[:20]+'...' for a in answers]}")

        # 后续轮次：互相批评
        for round_num in range(self.rounds - 1):
            new_answers = []
            for i, agent in enumerate(self.agents):
                others = [a for j, a in enumerate(answers) if j != i]
                context = f"其他代理的观点:\n" + "\n".join(f"- {a[:30]}" for a in others)
                new_answer = agent.generate(f"{question}\n{context}\n请给出改进后的答案")
                new_answers.append(new_answer)
            answers = new_answers
            print(f"  轮次 {round_num+2}: {[a[:20]+'...' for a in answers]}")

        return answers[0]  # 返回第一个代理的最终答案


class SimpleAgent:
    def __init__(self, name):
        self.name = name
    def generate(self, prompt):
        return f"[{self.name}] 基于'{prompt[:20]}...'的分析: 答案是 14"
```

---

## 4. 工具

### 4.1 实现框架

| 框架 | 辩论支持 |
|------|---------|
| LangGraph | 自定义多智能体图 |
| LangChain | 对话链 |
| 手写 | 完全控制 |

---

## 5. 工程最佳实践

### 5.1 辩论设计

- **轮次**：2-3 轮通常足够
- **代理数量**：3-5 个最佳（更多不会显著提升质量）
- **拓扑**：稀疏优于全连接——节省 token 成本

---

## 6. 常见错误

### 错误 1：轮次太多导致成本爆炸

**现象：** 5 个代理 × 5 轮 = 25 次 LLM 调用。

**修复：** 2-3 轮 + 3 个代理 = 6 次调用，效果已经很好。

---

## 7. 面试考点

### Q1：多代理辩论为什么比单代理更好？（难度：⭐⭐）

**参考答案：**
单代理容易产生幻觉和遗漏。多代理辩论让多个 LLM 互相检查——一个代理的幻觉会被其他代理指出。经过 2-3 轮迭代，答案收敛到更准确、更一致的版本。事实性、推理完整性、规则遵循度都会提升。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 多代理辩论 | "多人讨论" | 多个 LLM 互相批评并改进答案——提高事实性和推理 |
| 社会思维 | "Society of Minds" | ICML 2024 论文——N 个代理迭代批评收敛 |
| 稀疏拓扑 | "不全连接" | 每个代理只看部分其他代理——节省 token 成本 |

---

## 📚 小结

多代理辩论让多个 LLM 互相检查——2-3 轮后答案收敛。稀疏拓扑节省 token 成本。提高事实性、推理完整性、一致性。

---

## ✏️ 练习

1. **【实现】** 构建 3 个代理的辩论系统——在推理问题上测试
2. **【实验】** 对比全连接和环形拓扑的 token 成本和质量

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 多代理辩论 | `code/main.py` | N 个代理 + R 轮迭代 |

---

## 📖 参考资料

1. [论文] Du et al. "Improving Factuality and Reasoning via Language Model Debate". NeurIPS, 2022.
2. [论文] Liang et al. "Encouraging Divergent Thinking in LLMs through Multi-Agent Debate". ICML, 2024.
