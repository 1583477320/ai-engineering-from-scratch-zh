# Q 学习与 SARSA

> 蒙特卡洛等到回合结束才更新。TD(0) 每步自举——用下一个状态的价值估计更新当前状态。Q 学习离策略且乐观；SARSA 在策略且保守。两者都是一行代码。两者都是本阶段所有深度 RL 方法的基础。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 09 · 01（MDP）、02（动态规划）、03（蒙特卡洛）| **时间：** ~75 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 09 · 03（蒙特卡洛）— 从完整回报到单步自举 | 阶段 09 · 05（DQN）— 从表格 TD 到神经网络的扩展

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解 TD(0) 更新——单步自举替代完整轨迹
- [ ] 实现 Q 学习——离策略 TD 用贪心目标更新 Q 值
- [ ] 实现 SARSA——在策略 TD 用实际动作更新 Q 值
- [ ] 在悬崖行走环境中对比 Q 学习和 SARSA 的行为差异
- [ ] 解释最大化偏差（maximization bias）及其修复方法

---

## 1. 问题

蒙特卡洛有效但有两个昂贵的要求：需要终止的回合，且只在最终回报到达后才更新。如果回合是 1000 步，MC 等 1000 步才更新任何东西。高方差、低偏差、实践中很慢。

动态规划有相反的特性——零方差的自举备份——但需要已知模型。

时间差分（TD）学习取两者之问。从一次转移 (s, a, r, s') 中，形成单步目标 r + γV(s') 并推动 V(s) 向它靠拢。不需要模型。不需要完整回合。来自右侧上近似 V 的偏差，但方差远低于 MC，且从第一步开始就在线更新。

这是所有现代 RL——DQN、A2C、PPO、SAC——的转折点。本课程剩下的部分是函数逼近和技巧的层次，构建在这节课你将写的单步 TD 更新之上。

---

## 2. 概念

### 2.1 TD(0)——RL 的核心算子

$$V(s) \leftarrow V(s) + \alpha [r + \gamma V(s') - V(s)]$$

方括号中的量是 TD 误差 δ = r + γV(s') - V(s)。它是 MC 中 G_t - V(s_t) 的在线对应物。收敛要求 α 满足 Robbins-Monro (Σα=∞, Σα²<∞) 且所有状态无限频繁访问。

### 2.2 Q 学习——离策略 TD

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

max 假设从 s' 开始将遵循贪心策略——无论 agent 实际做什么动作。这种解耦使 Q 学习能在 agent 通过 ε-greedy 探索的同时学习 Q*。

**特性：** 离策略、乐观、收敛到最优 Q*，即使行为策略是次优的。

### 2.3 SARSA——在策略 TD

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$$

SARSA 使用 agent 实际采取的下一个动作 a'，而不是贪心的 argmax。收敛到当前 ε-greedy π 的 Q^π，在 ε→0 的极限下成为 Q*。

**特性：** 在策略、保守、收敛到当前策略的 Q^π。

### 2.4 悬崖行走的差异

经典的悬崖行走任务（掉下悬崖 = -100 奖励）：

| 方法 | 行为 |
|------|------|
| Q 学习 | 学习沿着悬崖边缘的最优路径——但探索时偶尔掉下去 |
| SARSA | 学习离悬崖一步的安全路径——因为 Q 值考虑了探索噪声 |
| 最终 | 两者在 ε→0 时都达到最优 |

当部署时仍在探索时，SARSA 的行为更保守。

### 2.5 Expected SARSA

用 π 下的期望替换 Q(s',a')：

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \sum_{a'} \pi(a'|s') Q(s',a') - Q(s,a)]$$

方差低于 SARSA（没有 a' 的采样），在策略性目标相同。

### 2.6 n-step TD 和 TD(λ)

在 TD(0) 和 MC 之间插值——等待 n 步后再自举：n=1 是 TD，n=∞ 是 MC。TD(λ) 用几何权重 (1-λ)λ^{n-1} 平均所有 n。大多数深度 RL 使用 n 在 3 到 20 之间。

---

## 3. 从零实现

### Step 1：SARSA——在策略 TD

```python
from collections import defaultdict
import random

def sarsa(env, episodes=2000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """SARSA：在策略 TD 控制。"""
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

8 行。与 Q 学习的唯一区别在 target 行。

### Step 2：Q 学习——离策略 TD

```python
def q_learning(env, episodes=2000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """Q 学习：离策略 TD 控制。"""
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

max 将 target 从行为中解耦。这一个符号是在策略和离策略的全部区别。

### Step 3：Double Q 学习——修复最大化偏差

```python
def double_q_learning(env, episodes=2000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """Double Q 学习——两个 Q 表，消除最大化偏差。"""
    Q1 = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    Q2 = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    for _ in range(episodes):
        s = env.reset()
        while True:
            # 用 Q1+Q2 的联合策略选动作
            if random.random() < epsilon:
                a = random.choice(ACTIONS)
            else:
                a = max(ACTIONS, key=lambda a: Q1[s][a] + Q2[s][a])
            s_next, r, done = env.step(s, a)
            # 随机选一个 Q 表更新
            if random.random() < 0.5:
                a_star = max(ACTIONS, key=lambda a: Q1[s_next][a])
                target = r + (gamma * Q2[s_next][a_star] if not done else 0.0)
                Q1[s][a] += alpha * (target - Q1[s][a])
            else:
                a_star = max(ACTIONS, key=lambda a: Q2[s_next][a])
                target = r + (gamma * Q1[s_next][a_star] if not done else 0.0)
                Q2[s][a] += alpha * (target - Q2[s][a])
            if done:
                break
            s = s_next
    return Q1, Q2
```

---

## 4. 工具

### 4.1 Gymnasium 中的经典环境

```python
import gymnasium as gym

# 悬崖行走——Q learning vs SARSA 对比
env = gym.make("CliffWalking-v0")
print(f"状态: {env.observation_space.n}")  # 48
print(f"动作: {env.action_space.n}")       # 4

# Taxi——多人间的运输任务
env = gym.make("Taxi-v3")
print(f"状态: {env.observation_space.n}")  # 500
print(f"动作: {env.action_space.n}")       # 6
```

### 4.2 TD 方法选型

| 方法 | 类型 | 方差 | 偏差 | 安全 |
|------|------|------|------|------|
| Q 学习 | 离策略 | 中 | 无 | 探索时危险 |
| SARSA | 在策略 | 中 | 有 | 保守 |
| Expected SARSA | 在策略 | 低 | 有 | 保守 |
| Double Q 学习 | 离策略 | 中 | 无 | 无最大化偏差 |

---

## 5. LLM 视角

### 5.1 TD 学习在大语言模型中的体现

| 概念 | LLM 对应 |
|------|---------|
| Q(s,a) | 策略在上下文 s 下选择 token a 的预期分数 |
| TD 误差 | PPO 中的优势函数 A_t |
| 经验回放 | LLM 训练不使用——但在线 PPO 使用当前策略采样 |
| Double Q | PPO 中的 value head 和 policy head 分离 |

### 5.2 RLHF 与 TD 的联系

RLHF 的 PPO 训练本质上在做 TD 更新——在每个 token 位置计算优势 A_t = G_t - V(s_t)。G_t 是 MC 回报（完整回复的折扣奖励总和），V(s_t) 是价值网络的估计。TD 误差决定了"这个 token 的贡献"。

### 5.3 使用 ChatGPT / Claude 时的直接体验

当你使用 ChatGPT 时，它每一步的 token 选择不是 Q 学习选的——但 PPO 训练过程中的优势估计使用了 TD 技术。A_t = G_t - V(s_t) 告诉模型"在当前上下文中，选这个 token 比平均值好多少"——这就是 Q 学习的 TD 误差在 LLM 规模下的对应。

---

## 6. 工程最佳实践

### 6.1 超参数调优

| 参数 | Q 学习 | SARSA | 说明 |
|------|--------|-------|------|
| α | 0.05-0.3 | 0.05-0.3 | 学习率，太大震荡太小慢 |
| ε 起始 | 1.0 | 1.0 | 充分探索 |
| ε 终点 | 0.01-0.05 | 0.01-0.05 | 最终策略的随机性 |
| ε 衰减 | 指数/线性 | 指数/线性 | GLIE 条件 |

### 6.2 初始化策略

乐观初始化（Q=0 对于负奖励任务）鼓励探索。悲观初始化可能困住贪心策略。

### 6.3 踩坑经验

- **Q 值初始化为 0 vs 正数**：在负奖励环境中，Q=0 是"乐观"的——agent 会探索来验证
- **ε 不衰减**：如果 ε 在训练结束时仍为 1.0，策略永远不会利用——性能不收敛
- **max 偏差**：Q 学习对噪声 Q 值的 max 操作有向上偏差。Double Q 学习修复这个问题

---

## 7. 常见错误

### 错误 1：Q 学习和 SARSA 的 target 行写反

**现象：** Q 学习表现像 SARSA 或反之。

**原因：** target 行用错了——Q 学习用 max，SARSA 用实际 a'。

```python
# ❌ Q 学习写成 SARSA
target = r + gamma * Q[s_next][a_next]  # 这是 SARSA！

# ✓ 正确 Q 学习
target = r + gamma * max(Q[s_next].values())

# ❌ SARSA 写成 Q 学习
target = r + gamma * max(Q[s_next].values())  # 这是 Q 学习！

# ✓ 正确 SARSA
target = r + gamma * Q[s_next][a_next]
```

### 错误 2：忽略终止状态的 bootstrap

**现象：** TD 误差在终止步骤异常大。

**原因：** 如果 done=True，没有 s'——目标应该只有 r，没有 γQ(s',a')。

```python
# ❌ 错误：终止状态也 bootstrap
target = r + gamma * max(Q[s_next].values())

# ✓ 正确：终止时只用 r
target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
```

---

## 8. 面试考点

### Q1：Q 学习和 SARSA 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
Q 学习是离策略的——它用贪心的 max 动作计算目标，与实际采取的动作无关。SARSA 是在策略的——它使用实际采取的动作 a' 计算目标。这意味着 Q 学习学习的是最优 Q*（只要探索充分），而 SARSA 学习的是当前策略的 Q^π。在实践中，Q 学习更乐观（学习最优路径）但探索时有风险（走悬崖边），SARSA 更保守（更安全路径）。

### Q2：什么是最大化偏差？Double Q 学习如何修复它？（难度：⭐⭐⭐）

**参考答案：**
当一个动作的 Q 值被噪声向上扰动时，max 操作会选择这个被高估的动作。因为 max 是在所有动作上取的，即使只有一个动作被高估，max 的结果也是高估的。这就是最大化偏差。Double Q 学习用两个独立的 Q 表：Q1 选择最佳动作（argmax_a Q1(s',a')），Q2 评估这个动作的值（Q2(s',a')）。选择和评估用不同的表，消除了系统性高估。更新时随机选择更新 Q1 或 Q2。

### Q3：TD(0) 和 MC 的核心权衡是什么？什么时候用哪个？（难度：⭐⭐）

**参考答案：**
TD(0) 的核心权衡是偏差-方差权衡。TD(0) 有偏差（因为用估计的 V(s') 自举）但方差低。MC 无偏差（用真实回报）但方差高（完整回报包含所有未来步的噪声）。在实践中：对于长回合任务（100+ 步），MC 的方差太大——TD 更好。对于短回合任务（如棋类），MC 好用。近几年也有混合方案——n 步 TD（用 n 步回报然后自举）。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| TD 误差 | "更新信号" | δ = r + γV(s') - V(s)，自举残差 |
| TD(0) | "单步 TD" | 每次转移更新，只用下一个状态的估计 |
| Q 学习 | "离策略 RL 基础" | 用 max 算目标，learn Q* 无关行为策略 |
| SARSA | "在策略 Q 学习" | 用实际 a' 算目标，learn 当前 ε-greedy 策略的 Q |
| 最大化偏差 | "Q 学习高估了" | 噪声估计的 max 有向上偏差；Double Q 学习修复 |
| 自举 | "用估计更新估计" | TD 区别于 MC 的特点——偏差来源但方差大幅降低 |

---

## 📚 小结

TD(0) 从单步转移中学习——不需要完整回合，不需要环境模型。Q 学习是离策略 TD，用 max 操作学习最优 Q*。SARSA 是在策略 TD，用实际动作学习当前策略的 Q^π。悬崖行走展示了关键区别：Q 学习学最优但探索时危险，SARSA 学安全路径。Double Q 学习修复了 Q 学习的最大化偏差。下一课我们将 TD 从表格扩展到神经网络——DQN。

---

## ✏️ 练习

1. **【实现】** 在 GridWorld 上实现 Q 学习和 SARSA（4×4，确定性）。画学习曲线（每 100 回合的平均回报），对比收敛速度。
2. **【实验】** 在悬崖行走环境上（4×12，悬崖奖励 -100），对比 Q 学习和 SARSA 的最终策略截图。哪个更靠近悬崖？
3. **【实现】** 在有噪声奖励的 GridWorld（每步奖励添加 N(0,5) 噪声）上实现 Double Q 学习。对比 Q 学习和 Double Q 学习的 V*(0,0) 估计值。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| Q 学习 + SARSA 实现 | `code/main.py` | Q 学习、SARSA、Double Q 学习 + 可视化 |

---

## 📖 参考资料

1. [论文] Watkins & Dayan. "Q-learning". Machine Learning, 1992. https://link.springer.com/article/10.1007/BF00992698
2. [教材] Sutton & Barto. "Reinforcement Learning: An Introduction". Ch. 6. 2018.
3. [论文] Hasselt. "Double Q-learning". NeurIPS, 2010. https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html
4. [论文] Rummery & Niranjan. "On-line Q-learning using connectionist systems". 1994.
5. [教材] Sutton & Barto. Ch. 7 — n-step Bootstrapping. 2018.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
