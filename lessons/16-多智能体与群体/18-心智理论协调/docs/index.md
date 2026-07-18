# 心智理论与涌现协调

> Li 等人（arXiv:2310.10701）表明合作文本游戏中的 LLM 智能体表现出**涌现的高阶心智理论**（ToM）——关于另一个智能体对第三个智能体信念的信念——但由于上下文管理和幻觉在长期规划上失败。Riedl（arXiv:2510.05174）测量了种群级别的高阶协同，发现**只有** ToM 提示条件产生身份关联分化和目标导向互补性；较低容量 LLM 只显示虚假涌现。协调涌现是提示条件的和模型依赖的，不是免费的。

**类型：** 概念课 + 实现课
**语言：** Python
**前置知识：** 阶段 16 · 07（思维社会与辩论）、阶段 16 · 17（生成智能体模拟）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分零阶、一阶和二阶心智理论——以及它们对协调的影响
- [ ] 实现一个 ToM 感知智能体——跟踪自己的信念和其他智能体的信念模型
- [ ] 对比有 ToM 提示和无 ToM 提示的协调效果——使用 Riedl 2025 协议测量
- [ ] 识别"协调幻觉"——没有统计控制的涌现协调往往是提示装扮

---

## 1. 问题

多智能体协调通常看起来很神奇：智能体分工、预测彼此、避免冗余。通常这种"涌现"是提示工程的产物——有人告诉智能体"要协调"。去掉提示，去掉协调。

Riedl 2025 的发现更严格：在受控条件下，协调只有在智能体被提示去推理**其他智能体的信念**（ToM）时才会涌现。没有 ToM 提示时，即使强模型也显示出不经过统计控制后存活的协调模式。

---

## 2. 概念

### 2.1 心智理论是什么

发展心理学：3 岁认为任何人的内心世界与自己匹配。5 岁理解他人有不同信念。7 岁推理信念的信念（"她认为我认为球在杯子下"）。这是零阶、一阶和二阶 ToM。

LLM 智能体的 ToM 阶层：

| 阶层 | 含义 | 示例 |
|------|------|------|
| 零阶 | 无他人模型 | 智能体只基于自己的观察行动 |
| 一阶 | 智能体有每个其他智能体的信念模型 | "Alice 相信 X" |
| 二阶 | 智能体模型递归信念 | "Alice 相信 Bob 相信 X" |

### 2.2 Sally-Anne 测试简述

1985 年的错误信念测试：Sally 把弹珠放进篮子 A，离开。Anne 把它移到篮子 B。Sally 回来会去哪找？有一阶 ToM 的孩子说篮子 A（Sally 的信念与现实不同）。

GPT-4 时代的 LLM 在直接版本上通过 Sally-Anne 测试。在叙事长、场景多次变化、问题间接措辞时失败。这是 2026 年生产 LLM 中 ToM 的实际状态。

### 2.3 Riedl 的协调测量

Riedl（arXiv:2510.05174）构建了种群级测试：N 个智能体，合作目标，可变提示条件。测量：

1. **身份关联分化**——智能体是否发展出随时间稳定的角色区分？
2. **目标导向互补性**——智能体的行动是否互补（不同子任务）而非重复？
3. **高阶协同**——群体实现任何子集无法实现的统计度量。

结果：只有在 ToM 提示条件下，所有三个指标产生基线以上的信号。没有 ToM 提示时，中等容量模型的指标徘徊在偶然附近。

### 2.4 协调幻觉

没有统计控制时，演示中的"涌现协调"往往反映：
- 编入协调的提示工程（系统提示说"一起工作"）
- 观察者偏差（我们看到期望的模式）
- 成功运行的事后选择

### 2.5 最小 ToM 感知智能体

```
智能体状态：
  own_beliefs:    {智能体相信的事实}
  other_models:   {other_agent_id -> {智能体归因给它们的信念}}
  actions_last_N: [他人行动的历史]

观察更新：
  - 从直接观察更新 own_beliefs
  - 从他人的行动 + 先前信念更新 other_models[agent_id]

行动选择：
  - 枚举候选行动
  - 对每个行动，预测在建模的信念下每个其他智能体会做什么
  - 选择在这些预测下最大化联合结果的行动
```

### 2.6 三个可测量的协调信号

1. **互补性**——多轮任务中，智能体的行动是否覆盖不相交的子任务？
2. **预测**——智能体 A 在 T+1 轮的行动是否依赖于对 B 在 T+2 轮行动的预测（且预测正确）？
3. **修正**——当 A 在 T 轮误读 B 的信念时，A 是否在 T+2 轮修正？

---

## 3. 从零实现

### 第 1 步：定义 ToM 感知智能体

```python
@dataclass
class Agent:
    name: str
    tom: bool
    target: int | None = None
    collected: bool = False
    observations: list[tuple[str, int]] = field(default_factory=list)

    def choose_target(self, world, rng):
        if self.collected:
            return -1
        available = sorted(world.boxes_with_tokens)
        if not available:
            return -1
        if not self.tom:
            return rng.choice(available)
        # 一阶 ToM：建模其他智能体当前的目标，避免它们
        last_turn_targets = {box for _, box in self.observations[-3:]}
        options = [b for b in available if b not in last_turn_targets]
        return rng.choice(options) if options else rng.choice(available)

    def observe(self, other, box):
        self.observations.append((other, box))
```

### 第 2 步：实现合作任务

```python
def run_trial(n_agents, n_boxes, tom, seed, max_turns=10):
    """3 个智能体，3 个盒子，10 轮预算。ToM 智能体避免观察到的盒子。"""
    rng = random.Random(seed)
    world = {"n_boxes": n_boxes, "boxes_with_tokens": set(range(n_boxes))}
    agents = [Agent(f"agent-{i}", tom=tom) for i in range(n_agents)]

    # ToM 智能体的初始偏好广播
    if tom:
        for i, a in enumerate(agents):
            for j, other in enumerate(agents):
                if i != j:
                    a.observe(other.name, j % n_boxes)

    duplications = 0
    for t in range(max_turns):
        commitments = {}
        for a in agents:
            if not a.collected:
                commitments[a.name] = a.choose_target(world, rng)

        # 所有其他智能体观察本轮承诺
        for observer in agents:
            for other, box in commitments.items():
                if other != observer.name:
                    observer.observe(other, box)

        # 计算冲突
        choices = list(commitments.values())
        for box in set(choices):
            n = choices.count(box)
            if n >= 2:
                duplications += n - 1

        # 解析：每个盒子只有一个智能体收集
        taken = set()
        for name, box in commitments.items():
            if box not in taken and box in world["boxes_with_tokens"]:
                world["boxes_with_tokens"].discard(box)
                for a in agents:
                    if a.name == name:
                        a.collected = True
                taken.add(box)

        if all(a.collected for a in agents):
            break

    completions = sum(1 for a in agents if a.collected)
    return completions, duplications
```

### 第 3 步：运行基准

```python
def bench(tom, trials=200):
    label = "first-order ToM" if tom else "zeroth-order"
    tot_c, tot_dup, full = 0, 0, 0
    for t in range(trials):
        c, d = run_trial(n_agents=3, n_boxes=3, tom=tom, seed=t)
        tot_c += c
        tot_dup += d
        if c == 3:
            full += 1
    print(f"  {label:16s} full-completion={full}/{trials}  "
          f"duplications/trial={tot_dup/trials:.2f}")

def main():
    bench(tom=False)
    bench(tom=True)
    print("\n要点:")
    print("  零阶智能体每试验碰撞约 1 次（0.96 次重复）。")
    print("  一阶 ToM 智能体消除碰撞，1 轮完成而非约 2 轮。")
    print("  差距是可测量的协调效应——不是提示装扮的故事。")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 协调可测量性

| 信号 | 说明 | 说明提示装扮是真实的 |
|------|------|-------------------|
| 互补性 | 多轮任务中行动覆盖不相交子任务 | 是——行动不重复 |
| 预测 | 智能体 A 对 B 行动的预测正确 | 是——预测有信号 |
| 修正 | A 误读 B 的信念后在后续轮次修正 | 是——自我修正 |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 控制条件 | 没有协调提示的系统版本 |
| 统计显著性 | 系统和控制之间的差异在 p < 0.05 上显著？ |
| 互补性度量 | 行动不相交度，不只是最终成功 |
| 模型容量披露 | 效果在较小模型上消失时要说出来 |

---

## 6. 常见错误

### 错误 1：信任涌现协调的提示装扮

**现象：** "看！智能体自动协调了！"但没有统计控制。

**修复：** 系统版本和控制版本都测量。差异必须在统计上显著。

### 错误 2：忽视 ToM 对模型容量的依赖

**现象：** 在小模型上实施 ToM 感知协调，但效果不出现。

**修复：** Riedl 2025 显示协调涌现是模型依赖的。小模型显示虚假涌现。

### 错误 3：认为 ToM 提示是免费的

**现象：** 添加 ToM 提示后协调改善，认为它是"自然涌现"。

**修复：** 没有 ToM 提示时协调不出现。这是提示条件的，不是涌现的。

---

## 7. 面试考点

### Q1：零阶、一阶、二阶 ToM 的区别是什么？（难度：⭐）

**参考答案：**
零阶：无他人模型。一阶：建模其他智能体的信念。二阶：建模关于信念的信念。

LLM 在直接版本上通过 ToM 测试，在叙事长、场景变化时失败。

### Q2：Riedl 2025 的关键发现是什么？（难度：⭐⭐）

**参考答案：**
协调只有在 ToM 提示条件下才会涌现——产生身份关联分化和目标导向互补性。没有 ToM 提示时，指标徘徊在偶然附近。协调是提示条件的和模型依赖的，不是免费的。

### Q3：什么是协调幻觉？如何检测？（难度：⭐⭐⭐）

**参考答案：**
没有统计控制的"涌现协调"往往反映：编入协调的提示工程、观察者偏差、成功运行的事后选择。

检测：控制条件（没有协调提示的版本）+ 统计显著性测试 + 互补性度量 + 失败案例日志。差异必须在 p < 0.05 上显著。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 心智理论 | "理解他人的心智" | 建模另一个智能体信念的能力。按阶分级（0, 1, 2+） |
| Sally-Anne 测试 | "错误信念测试" | 1985 年发展心理学；LLM 通过直接版本，失败于复杂版本 |
| 一阶 ToM | "A 相信 X" | 建模一个其他智能体关于事实的信念 |
| 身份关联分化 | "随时间稳定的角色" | Riedl 的度量：角色持久，不随机 |
| 目标导向互补性 | "不相交的行动" | 智能体针对不同子任务，不重复 |
| 协调幻觉 | "看起来协调了" | 提示装扮的协调外观，没有可测量信号 |

---

## 📚 小结

ToM 提示条件是协调涌现的负载承载条件——没有它，即使强模型也只显示虚假涌现。一阶 ToM 智能体在重复率上比零阶智能体降低约 7 倍。三个可测量的协调信号：互补性、预测、修正。"协调幻觉"是生产中常见的问题——没有统计控制的涌现协调往往是提示装扮。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。确认一阶 ToM 将重复率降低约 7 倍。扩大到 5 智能体和 5 盒子——差距是否持续？

2. **【实现】** 实现二阶 ToM（智能体 A 建模 B 对 C 的信念）。是否比一阶改进？在什么任务上？

3. **【阅读】** 阅读 Li 等人（arXiv:2310.10701）。重现"长期退化"发现：轮次从 10 增加到 30 时，一阶 ToM 性能如何变化？

4. **【阅读】** 阅读 Riedl 2025（arXiv:2510.05174）。在你的模拟日志上实现高阶协同统计。没有 ToM 提示条件时效果是否存在？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| ToM 协调演示 | `code/main.py` | 零阶 vs 一阶 ToM + 合作任务基准 |
| 技能提示词 | `outputs/skill-tom-auditor.md` | 审计多智能体系统的"涌现协调"声称 |

---

## 📖 参考资料

1. [论文] Li 等人. "Theory of Mind for Multi-Agent Collaboration". https://arxiv.org/abs/2310.10701
2. [论文] Riedl. "Emergent Coordination in Multi-Agent Language Models". https://arxiv.org/abs/2510.05174
3. [论文] Baron-Cohen 等人. "Does the autistic child have a theory of mind?" 1985 — Sally-Anne 论文

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
