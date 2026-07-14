# 多智能体强化学习

> 单代理 RL 假设环境是平稳的。把两个学习代理放在同一个世界里——假设就崩了：每个代理都是对方环境的一部分，而两个都在变化。多智能体 RL 是当马尔可夫假设不再成立时让学习收敛的技巧集。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 09 · 04（Q 学习）、06（REINFORCE）、07（A2C）| **时间：** ~45 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 09 · 07（A2C）— CTDE 架构的基础 | 阶段 16（多智能体）— LLM 多智能体系统

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分完全协作、完全竞争、混合动机三种多智能体场景
- [ ] 解释 CTDE（集中训练分散执行）的原理——为什么训练时用全局信息
- [ ] 说明 MAPPO——2026 年协作多智能体的主流算法
- [ ] 理解自对弈和联赛训练的区别——何时用哪种
- [ ] 对比 IPPO、MAPPO、QMIX 的适用场景

---

## 1. 问题

一个机器人在房间导航是单代理问题。一个足球队不是。AlphaStar 对阵星际争霸对手不是。一个竞标代理市场不是。两辆车协商四路停车不是。

在每个多智能体场景中，从任何一个代理的视角看，其他代理**就是环境的一部分**。当它们学习和改变行为时，环境变得非平稳。马尔可夫性质——"下一个状态只取决于当前状态和我的动作"——因为下一个状态还取决于其他代理的选择而被违反。

这打破了表格收敛证明（Q 学习的保证假设平稳环境）。也打破了朴素的深度 RL：代理陷入循环追逐，永远不会收敛到稳定策略。

2026 年应用：机器人集群、交通路由、自动驾驶车队、市场模拟、LLM 多智能体系统、任何有多个智能玩家的游戏。

---

## 2. 概念

### 2.1 形式化：马尔可夫博弈

MDP 的推广：状态 S、联合动作 a=(a₁,...,aₙ)、转移 P(s'|s,a)、每个代理的奖励 Rᵢ(s,a,s')。每个代理最大化自己的回报。

| 类型 | 特征 | 示例 |
|------|------|------|
| 完全协作 | 所有代理共享相同奖励 | 多机器人协作搬运 |
| 完全竞争 | 零和博弈 | 围棋、扑克 |
| 混合动机 | 部分协作、部分竞争 | 谈判、拍卖、交通 |

### 2.2 四种主流架构

**1. 独立学习（IPPO）**

每个代理独立运行 PPO，把其他代理当作环境。简单，有时有效。无收敛保证。

**2. 集中训练分散执行（CTDE）——最常用**

训练时用全局状态的集中评论家；部署时每个代理只用局部观测的独立策略。

| 方法 | 核心思想 |
|------|---------|
| MAPPO | PPO + 集中价值函数 |
| MADDPG | DDPG + 每个代理的集中评论家 |
| QMIX | 值分解——联合 Q 可分拆 |
| COMA | 反事实基线——隔离单个代理的贡献 |

**3. 自对弈（Self-Play）**

两个相同代理的副本互相博弈。AlphaGo/AlphaZero/MuZero。对称训练信号。最适合零和博弈。

**4. 联赛训练（League Play）**

自对弈的推广——保存历史策略池，从中采样对手。处理"石头剪刀"式的策略循环。AlphaStar（星际争霸 II）。

### 2.3 核心挑战

| 挑战 | 描述 | 修复 |
|------|------|------|
| 非平稳性 | 其他代理的策略在变化 | CTDE、自对弈 |
| 信用分配 | 共享奖励下，谁的贡献？ | COMA 反事实基线、QMIX 值分解 |
| 探索协调 | 代理需要互补策略 | 每代理熵奖励、角色条件化 |
| 可扩展性 | 联合动作空间指数增长 | 分解动作空间、函数逼近 |

---

## 3. 从零实现

### 两代理协作 GridWorld

```python
class CoopGridWorld:
    """两代理协作环境——从对角出发，同时到达目标。"""
    def __init__(self, size=6):
        self.size = size
        self.goal = (size-1, size-1)

    def reset(self):
        return ((0, 0), (size-1, 0))  # 两个代理的初始位置

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

### 独立 Q 学习

```python
def independent_q(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q1 = defaultdict(lambda: {a: 0 for a in ACTIONS})
    Q2 = defaultdict(lambda: {a: 0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            # 每个代理用自己的 Q 表，共享奖励
            Q1[s][a1] += alpha * (r + gamma * max(Q1[s_next].values()) - Q1[s][a1])
            Q2[s][a2] += alpha * (r + gamma * max(Q2[s_next].values()) - Q2[s][a2])
            s = s_next
            if done: break
    return Q1, Q2
```

---

## 4. 工具

### 4.1 PettingZoo

```python
from pettingzoo.mpe import simple_spread_v3
env = simple_spread_v3.env()
env.reset()
for agent in env.agent_iter():
    obs, reward, done, trunc, info = env.last()
    action = env.action_space(agent).sample()
    env.step(action)
```

### 4.2 Multi-Agent Gymnasium

```python
# 支持多代理的标准环境
import gymnasium as gym
env = gym.make("MultiAnt-v4")  # 多代理连续控制
```

---

## 5. LLM 视角

### 5.1 多智能体在大语言模型中的体现

- **AlphaStar**：用 PPO + 联赛训练在星际争霸 II 中达到大师级——最复杂的多智能体 RL 胜利
- **OpenAI Five**：PPO + LSTM + 自对弈在 Dota 2 中达到职业水平
- **LLM 多智能体系统（阶段 16）**：多个 LLM 代理协作/竞争——RL 显示在轨迹级输出上，而非词元级

### 5.2 CTDE 在 LLM 系统中的类比

CTDE = 训练时用全局信息，部署时用局部信息。类比 RLHF：训练时用奖励模型（全局信号），部署时 LLM 只能看到上下文（局部观测）。

---

## 6. 工程最佳实践

### 6.1 方法选型

| 场景 | 推荐方法 |
|------|---------|
| 协作任务 | MAPPO + 集中评论家 |
| 零和博弈 | 自对弈 + MCTS（AlphaZero） |
| 复杂多人游戏 | 联赛训练 |
| LLM 多智能体 | 自然语言通信 + 角色条件化 |

### 6.2 踩坑经验

- **非平稳回放**：独立代理的回放缓冲区比单代理更差——旧转移来自已过时的对手
- **信用分配歧义**：长回合后的共享奖励无法判断哪个代理贡献了——用 COMA 反事实基线
- **探索冗余**：两个代理探索相同的状态-动作对——用每代理熵奖励修复

---

## 7. 常见错误

### 错误 1：在紧密协作任务上使用独立 Q 学习

**现象：** 代理不收敛——陷入循环追逐。

**原因：** 非平稳性——代理 A 的最优响应随代理 B 的更新而变化。

### 错误 2：自对弈中的策略循环

**现象：** 策略在"石头-剪刀-布"式的循环中振荡。

**修复：** 使用联赛训练——保存历史策略池，从中采样对手。

---

## 8. 面试考点

### Q1：CTDE 为什么有效？（难度：⭐⭐）

**参考答案：**
训练时集中评论家可以用全局状态做精确的价值估计（解决了部分可观察性的挑战），而分散执行时每个代理只用局部观测做决策（确保了部署时的效率和独立性）。这解决了多智能体中的非平稳性问题——集中评论家稳定了训练目标。

### Q2：自对弈和联赛训练的区别是什么？（难度：⭐⭐⭐）

**参考答案：**
自对弈只有当前策略对抗自己的历史版本——信号对称但可能陷入循环。联赛训练保存多个历史策略（包括专门的"剥削者"——专门学习击败当前最强策略的代理），从整个池中采样对手——处理了"石头-剪刀"式的策略循环。AlphaStar 用联赛训练处理星际争霸的非传递性策略空间。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 马尔可夫博弈 | "多代理 MDP" | (S, A₁,...,Aₙ, P, R₁,...,Rₙ)；每个代理有自己的奖励 |
| CTDE | "集中训练分散执行" | 训练时用联合评论家；部署时用局部策略 |
| MAPPO | "多代理 PPO" | PPO + 集中价值函数 |
| QMIX | "单调值分解" | Q_tot = f(Q₁,...,Qₙ) 单调混合 |
| COMA | "反事实多代理" | 优势 = 我的 Q - 边际化我的动作的期望 Q |
| 自对弈 | "代理 vs 过去的自己" | 零和博弈的标准训练方式 |
| 联赛训练 | "群体训练" | 保存历史策略，采样对手 |

---

## 📚 小结

多智能体 RL 将单代理扩展到社会场景。核心挑战：非平稳性、信用分配、可扩展性。CTDE 是 2026 年的主流范式——MAPPO 是最常用的算法。自对弈适合零和博弈，联赛训练处理策略循环。2026 年最大的增长领域是 LLM 多智能体系统——代理在自然语言中协作/竞争。

---

## ✏️ 练习

1. **【实现】** 在两代理协作 GridWorld 上训练独立 Q 学习。需要多少回合收敛？
2. **【实验】** 添加"同步到达"条件——只有两个代理同时到达才奖励。独立 Q 还能收敛吗？
3. **【思考】** MAPPO 和独立 PPO 在协作任务上差多少？为什么？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 独立 Q 学习 | `code/main.py` | 两代理协作 GridWorld |

---

## 📖 参考资料

1. [论文] Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments". NeurIPS, 2017.
2. [论文] Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games". NeurIPS, 2022.
3. [论文] Vinyals et al. "Grandmaster level in StarCraft II". Nature, 2019.
4. [论文] Silver et al. "Mastering Go without human knowledge". Nature, 2017.
5. [GitHub] PettingZoo: https://pettingzoo.farama.org/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
