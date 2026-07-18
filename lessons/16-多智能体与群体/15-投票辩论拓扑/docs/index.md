# 投票、自一致性和辩论拓扑

> 最便宜的聚合：采样 N 个独立智能体，多数投票。Wang 等人 2022 年的自一致性用一个模型采样 N 次。多智能体用**异质**智能体扩展它——不同模型、不同提示词、不同温度、不同上下文。超越多数投票，辩论拓扑很重要：MultiAgentBench（arXiv:2503.01935，ACL 2025）评估了星/链/树/图协调，发现**图最适合研究**，超过 ~4 个智能体有"协调税"。AgentVerse（ICLR 2024）记录了两种涌现模式——志愿者行为和从众行为——从众既是特性（找到共识）也是风险（群体思维）。本课映射拓扑空间，构建每种变体，测量协调税。

**类型：** 概念课 + 实现课
**语言：** Python
**前置知识：** 阶段 16 · 07（思维社会与辩论）、阶段 16 · 14（共识与 BFT）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 对比四种拓扑——星型、链型、树型、图型——在研究、快速回答、流水线任务上的表现
- [ ] 理解协调税——超过 ~4 个智能体后，质量增长放缓而成本飙升
- [ ] 实现四种拓扑变体并测量（准确率、词元、模拟延迟）
- [ ] 识别自一致性的局限——它用相关误差做投票

---

## 1. 问题

辩论可以提高准确性（Du 等人，arXiv:2305.14325）。但它也可以降低准确性。辩论是否有帮助取决于四个结构选择：

1. 谁对谁说（拓扑）
2. 多少轮（Du 2023：轮数和智能体数量各自独立贡献）
3. 智能体是否异质（不同基础模型打破单一文化）
4. 是否有对抗性声音存在（钢人 vs 稻草人）

"运行 5 个智能体然后投票"的团队经常比单智能体退化。失败不是随机的——它们与拓扑和异质性相关。

---

## 2. 概念

### 2.1 自一致性——单模型基线

Wang 等人 2022 用一个模型在温度 > 0 时采样 N 次，对推理路径答案进行多数投票。在 GSM8K 上有显著增益。局限：误差被结构化相关——如果模型有系统偏差，所有 N 个样本共享它。

### 2.2 多智能体投票——异质扩展

用 N 个**不同**的智能体替代 N 个样本。不同基础模型、不同提示词、不同工具访问。好处：误差去相关。代价：不同智能体成本不同，协调增加开销。

### 2.3 四种拓扑

```
星型           链型           树型           图型

    ┌─A─┐       A─B─C─D     ┌──A──┐         A───B
    │   │                   │     │         │ × │
    B   C                   B     C         D───C
    │   │                  / \   / \       (全连接)
    D   E                 D   E F   G
```

| 拓扑 | 结构 | 最佳场景 | 最差场景 |
|------|------|---------|---------|
| 星型 | 一个枢纽，其他只与枢纽通信 | 快速事实回答 | 复杂研究 |
| 链型 | 线性，每个智能体看前一个输出 | 流水线式逐步精炼 | 缺点传播 |
| 树型 | 层次结构 | 层级化智能体系统 | 深度 > 2 时可观测性崩溃 |
| 图型 | 任意对任意 | 研究任务 | 协调税 |

### 2.4 协调税（MultiAgentBench）

MultiAgentBench（MARBLE，ACL 2025，arXiv:2503.01935）基准测试了星/链/树/图：

| 发现 | 说明 |
|------|------|
| 图拓扑在研究任务上获胜 | 信息任意对任意流动；智能体可以互相批评 |
| 星型在快速事实回答上获胜 | 枢纽过滤并整合 |
| 链型在逐步流水线上获胜 | 分阶段精炼 |
| 协调税 | 图拓扑超过 ~4 个智能体后出现；词元成本增长快于质量 |

### 2.5 自一致性的局限

自一致性用一个基础模型采样 N 次。误差被结构化相关——如果模型有系统偏差，所有 N 个样本共享它。

异质性是打破单一文化的旋钮：**将 N 中一个智能体换成不同的基础模型，比增加 N 给出更大的准确率提升。**

在极限情况下，3 个不同模型在大多数有干净真实答案的任务上胜过 5 个同一模型的副本。

### 2.6 何时辩论投票有效/无效

**有效：** 问题有真实答案、智能体可以访问不同来源、轮次有界（2-3 轮）、预算允许 3-5 个智能体

**无效：** 问题是意见性的、所有智能体共享基础模型、轮次无界、任务简单（单智能体 + 自一致性就够了）

---

## 3. 从零实现

### 第 1 步：定义模拟智能体

```python
@dataclass
class SimAgent:
    name: str
    base_accuracy: float
    error_bias: str
    tokens_per_call: int = 400

    def answer(self, correct, rng):
        return correct if rng.random() < self.base_accuracy else self.error_bias
```

### 第 2 步：实现四种拓扑

```python
def run_star(agents, correct, rng):
    """星型：枢纽轮询工作者并聚合。"""
    hub = agents[0]
    workers = agents[1:]
    answers = [w.answer(correct, rng) for w in workers]
    tokens = sum(w.tokens_per_call for w in workers) + hub.tokens_per_call
    final = majority(answers) if answers else hub.answer(correct, rng)
    return RunResult("star", len(agents), final, correct, tokens, steps=2)

def run_chain(agents, correct, rng):
    """链型：顺序精炼。"""
    current = agents[0].answer(correct, rng)
    tokens = agents[0].tokens_per_call
    for a in agents[1:]:
        proposal = a.answer(correct, rng)
        current = proposal if proposal != current and rng.random() < a.base_accuracy else current
        tokens += a.tokens_per_call
    return RunResult("chain", len(agents), current, correct, tokens, steps=len(agents))

def run_tree(agents, correct, rng):
    """树型：层次聚合。"""
    root = agents[0]
    leaves = agents[1:]
    mid = len(leaves) // 2
    left = majority([a.answer(correct, rng) for a in leaves[:mid]])
    right = majority([a.answer(correct, rng) for a in leaves[mid:]])
    tokens = sum(a.tokens_per_call for a in leaves) + root.tokens_per_call
    final = majority([left, right])
    return RunResult("tree", len(agents), final, correct, tokens, steps=3)

def run_graph(agents, correct, rng, rounds=2):
    """图型：所有对所有辩论，有界轮次。"""
    positions = [a.answer(correct, rng) for a in agents]
    tokens = sum(a.tokens_per_call for a in agents)
    for _ in range(rounds - 1):
        maj = majority(positions)
        new_positions = []
        for pos, ag in zip(positions, agents):
            if pos != maj and rng.random() < 0.4:
                new_positions.append(maj)
            else:
                new_positions.append(pos)
            tokens += ag.tokens_per_call
        positions = new_positions
    return RunResult("graph", len(agents), majority(positions), correct, tokens, steps=rounds * 2)
```

### 第 3 步：运行基准

```python
def main():
    for heterogeneous in (False, True):
        tag = "异质" if heterogeneous else "同质（单一文化）"
        print(f"\n基准——{tag}")
        print(f"{'拓扑':10s} {'N':>3s} {'准确率':>8s} {'平均词元':>12s} {'步数':>6s}")
        for topology in ("star", "chain", "tree", "graph"):
            for n in (3, 5, 7):
                acc_sum, tok_sum, step_sum = 0, 0, 0
                for t in range(200):
                    agents = make_agents(n, heterogeneous, seed=t)
                    rng = random.Random(t * 31 + 7)
                    r = run_graph(agents, "RIGHT", rng) if topology == "graph" else ...
                    acc_sum += r.accuracy()
                    tok_sum += r.tokens
                    step_sum += r.steps
                print(f"{topology:10s} {n:>3d} {acc_sum/200:>8.2f} {tok_sum//200:>12d} {step_sum//200:>6d}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 拓扑选择指南

| 拓扑 | 最佳场景 | 最差场景 |
|------|---------|---------|
| 星型 | 快速事实回答（枢纽过滤整合） | 复杂研究 |
| 链型 | 分阶段精炼流水线 | 单一文化（缺点传播） |
| 树型 | 层级智能体系统 | 深度 > 2 时可观测性崩溃 |
| 图型 | 研究任务（任意对任意） | 超过 ~4 个智能体的协调税 |

### 4.2 协调税数据

| 智能体数 | 图型词元成本 | 图型准确率 | 协调税 |
|---------|-----------|----------|--------|
| 3 | 基线 | 高 | 低 |
| 5 | 2-3x 基线 | 略增 | 中 |
| 7 | ~7x 基线 | 饱和 | 高 |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 从自一致性 N=5 开始 | 用一个强基础模型，这是便宜基线 |
| 异质性投票 N=3 | 如果准确率重要，换一个智能体为不同模型 |
| 协调税在 ~4 个智能体后出现 | 图型超过 ~4 后词元成本增长快于质量 |
| 记录少数簇 | 少数簇持续正确时是多样性信号 |
| 轮次有界 | 2-3 轮典型；无界辩论奖励从众 |

---

## 6. 常见错误

### 错误 1：图型超过 4 个智能体

**现象：** 7 个智能体的图型辩论词元成本是 3 个智能体的 ~7 倍，但准确率几乎不增。

**修复：** 协调税是经验性的。图型最佳在 N=3-5。超过 5-7 后协调税主导。

### 错误 2：相同基础模型投票

**现象：** 5 个智能体都是同一模型，多数投票在错误答案上——单一文化。

**修复：** 异质性是打破单一文化的旋钮。3 个不同模型在大多数任务上胜过 5 个同一模型的副本。

### 错误 3：无界辩论轮次

**现象：** "辩论到同意"——从众每次获胜。

**修复：** 限制轮数（2-3 轮）。有界轮次防止从众级联。

---

## 7. 面试考点

### Q1：四种拓扑各适合什么任务？（难度：⭐）

**参考答案：**
| 拓扑 | 最佳 |
|------|------|
| 星型 | 快速事实回答（枢纽过滤整合） |
| 链型 | 分阶段精炼流水线 |
| 树型 | 层级化智能体系统 |
| 图型 | 研究任务（任意对任意） |

### Q2：协调税是什么？（难度：⭐⭐）

**参考答案：**
MultiAgentBench 发现图型超过 ~4 个智能体后，词元成本增长快于质量。墙钟时间和词元成本随智能体数超线性增长。

协调税是经验性的（2026 LLM 上下文容量限制），不是根本性的。它反映了每个智能体的上下文被同伴输出填满，边际价值在每个人都能看到每个人之后下降。

### Q3：为什么异质性比智能体数量更重要？（难度：⭐⭐⭐）

**参考答案：**
2024-2026 实用文献中的模式：将 N 个智能体中的一个换成不同基础模型，比增加 N 给出更大的准确率提升。

原因：单一文化——每个新的独立误差源比额外的相关样本更有价值。在极限情况下，3 个不同模型在大多数有干净真实答案的任务上胜过 5 个同一模型的副本。

异质性打破单一文化；智能体数量只是增加更多相关样本。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 自一致性 | "采样 N 次，投票" | Wang 2022。单模型，N 次温度 > 0 采样，推理路径多数投票 |
| 异质性 | "不同模型" | 不同基础模型或提示词族的集成体。打破单一文化 |
| MAD | "多智能体辩论" | 智能体跨轮次交换批评的通用术语。见 Du 2023 |
| A-HMAD | "对抗性异质 MAD" | 强调不同模型 + 对抗结构的 MAD 变体 |
| 拓扑 | "谁对谁说" | 星/链/树/图。决定信息流 |
| 协调税 | "边际收益递减" | 图型超过 ~4 个智能体后，成本增长快于质量 |
| 志愿者行为 | "未请求的帮助" | AgentVerse 涌现模式：一个智能体主动提供帮助 |
| 从众行为 | "压力下的同意" | AgentVerse 涌现模式：一个智能体与批评者对齐 |

---

## 📚 小结

自一致性是单模型基线——采样 N 次投票。多智能体用异质智能体扩展它。四种拓扑：星型（快速事实）、链型（流水线）、树型（层级）、图型（研究最佳但有协调税）。超过 ~4 个智能体后协调税出现。异质性是打破单一文化的旋钮——3 个不同模型胜过 5 个同一模型的副本。

至此第 16 章（多智能体与群体）全部 15 节完成（01-15）。从"为什么需要多智能体"到"投票和辩论拓扑"——覆盖了多智能体系统的完整谱系。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。为图型绘制协调税曲线：准确率 vs N，词元 vs N。N 多少时曲线拐点？

2. **【实现】** 实现 A-HMAD：三个有意不同偏差的智能体。全同偏差基线 vs A-HMAD 在第 14 课单一文化攻击上的对比如何？

3. **【阅读】** 阅读 AgentVerse 论文（ICLR 2024）。识别你的实现最强烈地表现出哪种涌现行为。你能通过提示变更引出相反的行为吗？

4. **【阅读】** 阅读 MultiAgentBench（arXiv:2503.01935）第 4 节。用你的测试工具重现"图型最适合研究"的结果。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 拓扑对比 | `code/main.py` | 四种拓扑 × N=3/5/7 × 同质/异质 |
| 技能提示词 | `outputs/skill-topology-picker.md` | 为任务推荐拓扑、智能体数、异质性配置 |

---

## 📖 参考资料

1. [论文] Wang 等人. "Self-Consistency Improves Chain of Thought Reasoning". https://arxiv.org/abs/2203.11171
2. [论文] Du 等人. "Improving Factuality and Reasoning via Multiagent Debate". https://arxiv.org/abs/2305.14325
3. [论文] MultiAgentBench / MARBLE. https://arxiv.org/abs/2503.01935
4. [论文] "Should we be going MAD?" https://arxiv.org/abs/2311.17371
5. [论文] AgentVerse (ICLR 2024). https://proceedings.iclr.cc/paper_files/paper/2024/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
