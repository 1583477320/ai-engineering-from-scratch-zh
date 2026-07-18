# MARL——MADDPG、QMIX、MAPPO

> 多智能体强化学习（MARL）是 LLM 智能体系统训练策略的底层技术。MADDPG 引入了 CTDE（集中训练分散执行）。QMIX 是单调价值分解。MAPPO 是 PPO 加集中值函数——2026 年默认合作 MARL 基线。本课从小型网格世界构建它们，让三个思想落地为肌肉记忆。

**类型：** 概念课 + 实现课
**语言：** Python
**前置知识：** 阶段 09（强化学习）、阶段 16 · 09（并行群体网络）
**预计时间：** ~90 分钟
**所处阶段：** Tier 3

---

## 学习目标

- 区分独立 RL、CTDE 和价值分解——理解非平稳问题
- 实现 CTDE 模式：训练时全局信息，部署时局部策略
- 实现 MADDPG、QMIX、MAPPO 的脚本化策略模拟
- 理解 MAPPO 为什么是 2026 年默认合作 MARL 基线

---

## 1. 问题

LLM 智能体系统越来越多地训练智能体间协调的策略：何时延后、何时行动、调用哪个对等体。MARL 文献告诉你如何训练这样的策略。

独立 RL（每个智能体独立学习）是非平稳的——每个智能体的环境随其他智能体变化而变。集中式 RL（一个智能体控制所有）不缩放且违反执行约束。CTDE 取得两者最佳。

---

## 2. 概念

### 2.1 三个算法

**MADDPG（2017）：** 每个智能体 i 有 actor mu_i(o_i) 映射自身观察到行动。每个还有 critic Q_i(x, a_1..a_n) 在训练时看到所有观察和所有行动。Actor 更新对 critic 评价的梯度。CTDE 模式的开创者。

**QMIX（2018）：** 合作场景。整体 Q_tot 是各智能体 Q_i 的单调函数。单调性保证 argmax 可分解——每个智能体独立选择 argmax_{a_i} Q_i。在星际争霸合作微操（SMAC）上获胜。

**MAPPO（2022）：** PPO 加集中值函数。多智能体 PPO 匹配或击败离策略 MARL 方法。最少超参数调优。2026 年默认合作 MARL 基线。

### 2.2 CTDE 设计模式

即使不训练，CTDE 也是有用的架构模式：
- 设计时假设全团队可见
- 运行时强制分散执行：每个智能体只看到 o_i

迫使你显式处理部分可观察性。

### 2.3 MADDPG 与 QMIX 与 MAPPO 对比

| 维度 | MADDPG | QMIX | MAPPO |
|------|--------|------|-------|
| 类型 | 离策略 | 离策略 | 在策略 |
| 动作 | 连续+离散 | 离散 | 连续+离散 |
| 合作/竞争 | 混合 | 合作 | 合作 |
| Critic | 集中 critic | 单调混合网络 | 集中值函数 |
| 失败模式 | 不缩放 | 单调性约束 | 样本效率 |

## 3. 从零实现

### 第 1 步：定义环境

```python
GRID = 4

@dataclass
class Env:
    agent0, agent1: tuple[int, int]
    pellet0, pellet1: tuple[int, int]
    pellets_remaining: set[tuple[int, int]]

    @staticmethod
    def new(rng):
        positions = set()
        while len(positions) < 4:
            positions.add((rng.randint(0, GRID-1), rng.randint(0, GRID-1)))
        a0, a1, p0, p1 = list(positions)
        return Env(a0, a1, p0, p1, {p0, p1})

    def collect_if_on_pellet(self):
        for pos in (self.agent0, self.agent1):
            self.pellets_remaining.discard(pos)
```

### 第 2 步：独立基线（无协调）

```python
def run_independent(env, max_steps=50):
    steps = 0
    while not env.done and steps < max_steps:
        t0 = min(env.pellets_remaining, key=lambda p: manhattan(env.agent0, p))
        t1 = min(env.pellets_remaining, key=lambda p: manhattan(env.agent1, p))
        env.agent0 = step_toward(env.agent0, t0)
        env.agent1 = step_toward(env.agent1, t1)
        env.collect_if_on_pellet()
        steps += 1
    return steps
```

### 第 3 步：CTDE 集中分配

```python
def _assigned_targets(env):
    pellets = list(env.pellets_remaining)
    if len(pellets) == 1:
        return pellets[0], pellets[0]
    p, q = pellets[0], pellets[1]
    cost_pq = manhattan(env.agent0, p) + manhattan(env.agent1, q)
    cost_qp = manhattan(env.agent0, q) + manhattan(env.agent1, p)
    return (p, q) if cost_pq <= cost_qp else (q, p)

def run_maddpg_style(env, max_steps=50):
    steps = 0
    while not env.done and steps < max_steps:
        t0, t1 = _assigned_targets(env)
        env.agent0 = step_toward(env.agent0, t0)
        env.agent1 = step_toward(env.agent1, t1)
        env.collect_if_on_pellet()
        steps += 1
    return steps
```

### 第 4 步：运行基准

```python
def bench(label, runner):
    total, trials = 0, 500
    for i in range(trials):
        rng = random.Random(i)
        env = Env.new(rng)
        total += runner(env)
    print(f"  {label:20s} avg_steps = {total/trials:.2f}")

def main():
    bench("independent (no coord)", run_independent)
    bench("MADDPG-style (CTDE)", run_maddpg_style)
    bench("QMIX-style (mono decomp)", run_qmix_style)
    bench("MAPPO-style (centralized V)", run_mappo_style)
```

完整代码见 code/main.py。

---

## 4. 工业工具

| 算法 | 2026 用途 | 框架 |
|------|----------|------|
| MAPPO | 默认合作基线 | PyMARL, RLlib |
| QMIX | 离散动作合作 | PyMARL, MARLlib |
| MADDPG | 混合场景 | RLlib, Multi-Agent API |

---

## 5. 常见错误

### 错误 1：独立 RL 导致的非平稳性

现象：智能体上周工作正常，本周上游智能体变了，它就失败。原因是独立 RL 中环境随其他智能体变化。CTDE 的集中 critic 提供平稳值估计。

### 错误 2：忽略 MAPPO

2022 年前，主流是离策略 MARL。MAPPO 表明在策略方法匹配或击败离线方法。2026 年默认基线。

### 错误 3：不把 CTDE 当架构模式

CTDE 即使不训练也是有用的：设计时全团队可见，运行时只看到自己。迫使你思考部分可观察性。

---

## 6. 面试考点

### Q1：CTDE 是什么？（难度：⭐）

集中训练分散执行。训练时有全局信息（所有观察和行动），部署时只有局部策略。

### Q2：MAPPO 为什么是 2026 年默认基线？（难度：⭐⭐）

Yu 等人 2022 年证明 MAPPO 在粒子世界、SMAC、谷歌研究足球、Hanabi 上匹配或击败离策略 MARL。最少调优。稳定训练。跨种子可复现。

### Q3：LLM 工程师为什么需要理解 MARL？（难度：⭐⭐⭐）

三种用途：路由器训练（MARL 问题，MAPPO 适合）、角色涌现（QMIX 价值分解强制互补性）、多智能体工具使用（CTDE 产生尊重资源约束的局部策略）。

实践中，大多数 LLM 智能体系统用提示词而非训练策略。MARL 在你需要大量交互数据、清晰奖励信号、愿意投入训练基础设施时使用。

---

## 关键术语

| 术语 | 含义 |
|------|------|
| MARL | 多智能体强化学习 |
| CTDE | 集中训练分散执行 |
| MADDPG | CTDE 模式创始者 |
| QMIX | 单调价值分解 |
| MAPPO | 多智能体 PPO，2026 默认 |
| 非平稳性 | 每个智能体的环境随其他智能体变化 |
| SMAC | 星际争霸多智能体挑战 |

---

## 小结

MADDPG（CTDE）、QMIX（单调价值分解）、MAPPO（集中值函数）是 MARL 三个核心算法。MAPPO 是 2026 年默认合作 MARL 基线。CTDE 即使不训练也是有用的架构设计模式。非平稳性是核心挑战，CTDE 的集中 critic 解决它。

下一课：智能体经济、代币激励与声誉。

---

## 练习

1. [实验] 运行 code/main.py。测量独立和 CTDE 的步数差距。在 6x6 网格上差距增减？
2. [实现] 竞争变体：两个智能体竞争一个颗粒，先到者得。哪个模式处理竞争？MADDPG 历史上如此。
3. [阅读] 阅读 MAPPO 论文。作者为什么认为集中值+PPO 击败离策略 MARL？列出三个最强主张。

---

## 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| MARL 演示 | code/main.py | 独立/CTDE/QMIX/MAPPO 对比 |
| 技能提示词 | outputs/skill-marl-picker.md | 为任务选 MARL 算法 |

---

## 参考资料

1. Lowe et al. MADDPG. NeurIPS 2017. arXiv:1706.02275
2. Rashid et al. QMIX. ICML 2018. arXiv:1803.11485
3. Yu et al. MAPPO. NeurIPS 2022. arXiv:2103.01955
4. SMAC. https://github.com/oxwhirl/smac
