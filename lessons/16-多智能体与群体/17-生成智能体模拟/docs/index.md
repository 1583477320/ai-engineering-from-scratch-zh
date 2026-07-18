# 生成智能体与涌现模拟

> Park 等人 2023 年（UIST '23，arXiv:2304.03442）用三部分架构填充了**Smallville**——一个 25 个智能体的沙箱：**记忆流**（自然语言日志）、**反思**（智能体关于自身流生成的更高层次综合）和**计划**（日级行为，然后子计划）。标志性结果是情人节派对的涌现：一个被植入"想举办情人节派对"的智能体，没有进一步脚本化，产生了通过人口传播的邀请、协调了日期，派对发生了——从 24 个对此一无所知的智能体开始。消融显示三个组件对可信度都是必需的。

**类型：** 概念课 + 实现课
**语言：** Python
**前置知识：** 阶段 16 · 04（原语模型）、阶段 16 · 13（共享记忆）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述 Smallville 架构的三个组件——记忆流、反思、计划——以及每个为什么必需
- [ ] 实现记忆检索（最近性 + 重要性 + 相关性的加权组合）
- [ ] 演示情人节派对涌现——从一个种子智能体的邀请传播到人口协调到达
- [ ] 识别已记录的失败模式：空间规范错误、记忆溢出、反思幻觉

---

## 1. 问题

大多数多智能体系统是紧密脚本化的团队：规划者规划、编码者编码、审查者审查。这适用于定义明确的任务。它不捕捉智能体有记忆、优先级和开放世界时涌现的、未脚本化的行为。

Smallville 架构是其基准。Park 2023 之前，最好的智能体模拟是浅层脚本跟随者；之后，模式成为开放世界中生成智能体的默认。如果你在 2026 年构建智能体模拟，你要么使用 Smallville 的三个组件，要么明确论证为什么不用。

---

## 2. 概念

### 2.1 三个组件

**记忆流。** 观察、动作、反思和计划的追加日志。每个条目有时间戳、类型、描述（自然语言）和派生元数据：**最近性**、**重要性**（智能体自评 1-10）和**相关性**（与当前查询的余弦相似度）。

检索组合三个分数：`score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`。Top-k 条目进入当前提示词。

**反思。** 定期（每 N 条记忆或在重要事件上），智能体从最近的记忆生成更高层次的综合。反思条目回到流中，可像任何其他记忆一样被检索。这是智能体构建"理解"的方式。

**计划。** 从上到下分解。首先，日级计划（"去上班，和 Klaus 吃晚餐"）。然后小时级计划。然后动作级计划。计划是可修改的：当观察与计划矛盾时，智能体重新计划受影响的部分。

### 2.2 三个组件为什么都必需（消融）

Park 等人运行了去除观察、反思和计划中每一个的消融：

| 缺失组件 | 影响 |
|---------|------|
| 无观察 | 智能体错过上下文，基于过时信念行动 |
| 无反思 | 智能体无法形成更高层次信念；交互保持浅层 |
| 无计划 | 行为变成反应性噪音；目标消散 |

所有三个组件的可信度最高；去掉任何一个都会产生可测量的退化。

### 2.3 情人节派对涌现

一个智能体（Isabella Rodriguez）被植入目标"在 2 月 14 日下午 5 点在 Hobbs Cafe 举办情人节派对"。24 个其他智能体没有收到此种子。模拟天数中：

1. Isabella 的计划包括邀请人们
2. 每个邀请成为邻居记忆流中的观察
3. 邻居的反思产生信念："Isabella 要办派对"
4. 邻居的计划加入"2 月 14 日参加派对"
5. 邻居告诉其他邻居。邀请在没有中央协调的情况下传播
6. 2 月 14 日下午 5 点，几个智能体在 Hobbs Cafe 聚集

这是技术意义上的涌现：系统级行为（派对）从局部交互（双边邀请 + 个人规划）中产生，没有中央编排者。

### 2.4 已记录的失败模式

| 失败 | 说明 | 缓解 |
|------|------|------|
| 空间规范错误 | 智能体走进关闭的商店；共用单人浴室 | 需要外部空间约束 |
| 记忆溢出 | 深度模拟导致检索成本增长 | 周期性记忆压缩（摘要+修剪） |
| 反思幻觉 | 反思可以发明记忆流中不存在的关系 | 反思提示词中包含来源记忆 ID |

### 2.5 三组件实现规则

1. **记忆是追加的**——从不修改记忆条目；修正作为新条目
2. **重要性评分很廉价**——写入时让 LLM 评分 1-10，缓存分数
3. **检索是排序的，不是过滤的**——Top-k 按组合分数；不要用硬过滤
4. **反思定期运行**——当未处理记忆的重要性总和超过阈值时触发
5. **计划可修改**——当新观察与计划矛盾时，只重新生成受影响的部分

---

## 3. 从零实现

### 第 1 步：定义记忆和检索

```python
@dataclass
class Memory:
    ts: int
    kind: str
    content: str
    importance: int

@dataclass
class Plan:
    tick: int
    where: str
    note: str

@dataclass
class Agent:
    name: str
    location: str
    stream: list[Memory] = field(default_factory=list)
    plans: list[Plan] = field(default_factory=list)
    beliefs: list[str] = field(default_factory=list)

    def observe(self, tick, content, importance=3):
        self.stream.append(Memory(tick, "observation", content, importance))

    def reflect(self, tick):
        """反思：从最近的高重要性记忆生成信念。"""
        recent_important = [m for m in self.stream if m.importance >= 6 and tick - m.ts <= 5]
        for m in recent_important:
            if "invited" in m.content and "party at" in m.content:
                belief = "there is a party I was invited to"
                if belief not in self.beliefs:
                    self.beliefs.append(belief)
                    self.stream.append(Memory(tick, "reflection", belief, 8))

    def update_plan(self, tick):
        if "there is a party I was invited to" in self.beliefs:
            if not any(p.where == "HobbsCafe" for p in self.plans):
                self.plans.append(Plan(tick=5, where="HobbsCafe", note="attend the party"))

    def act(self, tick):
        for p in self.plans:
            if p.tick == tick:
                self.location = p.where
                return f"{self.name} moves to {p.where} ({p.note})"
        return f"{self.name} remains at {self.location}"

def retrieve_top_k(stream, query, tick, k=3):
    """检索：最近性 + 重要性 + 相关性的加权组合。"""
    def score(m):
        recency = math.exp(-0.3 * (tick - m.ts))
        importance = m.importance / 10.0
        relevance = 0.6 if any(w in m.content.lower() for w in query.lower().split()) else 0.1
        return recency + importance + relevance
    return sorted(stream, key=score, reverse=True)[:k]
```

### 第 2 步：运行涌现模拟

```python
def run_simulation(n_agents=5, ticks=6):
    agents = [Agent(f"agent-{i}", location="home") for i in range(n_agents)]

    # 种子智能体 0 的派对目标
    agents[0].stream.append(Memory(0, "goal", "host a Valentine's party at HobbsCafe at tick 5", 10))
    agents[0].beliefs.append("there is a party I was invited to")

    for tick in range(ticks):
        # 邀请传播
        if tick == 0:
            for i in (1, 2):
                agents[i].observe(tick, "agent-0 invited me to a party at HobbsCafe", 8)
        if tick == 1:
            agents[3].observe(tick, "agent-1 invited me to a party at HobbsCafe", 7)
        if tick == 2:
            agents[4].observe(tick, "agent-2 invited me to a party at HobbsCafe", 7)

        for a in agents:
            a.reflect(tick)
            a.update_plan(tick)
            action = a.act(tick)
            if action.startswith(a.name + " moves"):
                print(f"  {action}")

    at_party = sum(1 for a in agents if a.location == "HobbsCafe")
    print(f"\n{at_party}/{n_agents} agents converged at HobbsCafe.")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 Smallville 架构在 2026 年的应用

| 应用 | 三个组件的用途 |
|------|--------------|
| 多智能体社会模拟 | 策略/市场研究——模拟用户对功能的反应 |
| 游戏 NPC AI | RPG 中涌现故事线而非脚本任务 |
| 生成智能体评估 | 可信度 + 长期行为连贯性作为度量 |

---

## 5. 工程最佳实践

| 原则 | 说明 |
|------|------|
| 记忆是数据库 | 在规模化时用真实存储（向量 DB、Postgres） |
| 记录检索追踪 | 每个动作记录驱动它的 top-k 记忆 |
| 每智能体预算词元 | N 智能体 × T 轮 × 每轮调用可能超过预算 |
| 定期压缩记忆 | 摘要 + 修剪低重要性条目 |
| 显式检测空间/社会规范违规 | 架构不学习它们 |

---

## 6. 常见错误

### 错误 1：记忆不追加

**现象：** 修改历史记忆条目——审计跟踪丢失。

**修复：** 记忆是追加的。修正作为引用被替代条目的新条目。

### 错误 2：不压缩记忆

**现象：** 深度模拟运行导致记忆流无限增长，检索成本爆炸。

**修复：** 定期压缩：摘要低重要性条目，修剪超过 N 条的记忆。

### 错误 3：反思幻觉

**现象：** 反思生成记忆流中不存在的关系——智能体基于虚假信念行动。

**修复：** 反思提示词中包含来源记忆 ID，检索时验证。

---

## 7. 面试考点

### Q1：Smallville 架构的三个组件是什么？每个为什么必需？（难度：⭐）

**参考答案：**
记忆流（观察的追加日志）、反思（更高层次信念综合）、计划（日/时/动作级分解）。

消融显示：无观察→错过上下文；无反思→交互保持浅层；无计划→行为变成反应噪音。所有三个必需。

### Q2：记忆检索如何工作？（难度：⭐⭐）

**参考答案：**
组合三个分数：最近性（指数衰减）、重要性（1-10 自评）、相关性（余弦相似度）。Top-k 条目进入当前提示词。

不是过滤——是排序。过滤会丢失上下文；排序保留所有记忆但优先最相关的。

### Q3：情人节派对涌现说明了什么？（难度：⭐⭐）

**参考答案：**
一个种子智能体的邀请通过双边记忆观察传播到人口，无需中央协调。24 个智能体最终在 Hobbs Cafe 聚集——这是技术意义上的涌现：系统级行为（派对）从局部交互（邀请 + 个人规划）中产生。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 记忆流 | "智能体的日记" | 观察、动作、反思和计划的追加日志 |
| 反思 | "更高层次信念" | 从最近记忆生成的综合，作为新记忆重新摄入 |
| 计划 | "日/时/动作分解" | 从上到下的计划树。观察矛盾时可修改 |
| Smallville | "Park 2023 的沙箱" | 产生情人节派对涌现的 25 智能体模拟 |
| 涌现 | "系统级行为从局部交互中产生" | 没有中央编排者——一个种子派对通过记忆传播 |
| 信念 | "智能体认为的" | 通过反思生成，通过观察更新 |

---

## 📚 小结

Smallville 的三个组件——记忆流、反思、计划——是 2026 年智能体模拟的参考架构。一个种子智能体的邀请在没有中央协调的情况下传播到 24 个智能体并协调了情人节派对。消融显示三个组件对可信度都是必需的。已记录的失败：空间规范错误、记忆溢出、反思幻觉。记忆是数据库，反思是负载承载的，检索是排序的。

下一课：心智理论与涌现协调。

---

## ✏️ 练习

1. **【实验】** 运行 `code/main.py`。确认 3+ 智能体在派对处汇聚。增加到 10 个——涌现仍然发生吗？

2. **【实验】** 移除反思步骤。行为是什么样的？映射到 Park 2023 的消融发现。

3. **【实现】** 引入竞争性种子目标（"Klaus 想在下午 5 点做研究讲座"）。智能体分裂还是一方占优？

4. **【阅读】** 阅读 Park 等人（arXiv:2304.03442）第 6 节。识别你的微型版本不可复现的一种行为。你需要增强架构的哪个组件？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 生成智能体模拟 | `code/main.py` | Smallville 微型：5 智能体 + 记忆/反思/计划 + 派对涌现 |
| 技能提示词 | `outputs/skill-simulation-designer.md` | 设计生成智能体模拟 |

---

## 📖 参考资料

1. [论文] Park 等人. "Generative Agents: Interactive Simulacra of Human Behavior". https://arxiv.org/abs/2304.03442
2. [GitHub] Smallville. https://github.com/joonspk-research/generative_agents
3. [论文] Hayes-Roth 1985. "A Blackboard Architecture for Control"

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
