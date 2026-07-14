# DQN——深度 Q 网络

> 2013 年 Mnih 用单一 Q 学习网络在 7 个 Atari 游戏上击败了所有经典 RL agent。2015 年扩展到 49 个游戏，发表在 Nature 上，开启了深度 RL 时代。DQN 是 Q 学习加三个使函数逼近稳定的技巧。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 09 · 04（Q 学习）、阶段 7 · 05（Transformer）| **时间：** ~90 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 09 · 04（Q 学习）— 从表格到函数逼近 | 阶段 09 · 07（A2C）— 从 Q 学习到 Actor-Critic

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现 DQN 的三个核心——经验回放、目标网络、奖励裁剪
- [ ] 解释"致命三要素"为什么使朴素函数逼近发散
- [ ] 实现 Double DQN——修复最大化偏差
- [ ] 理解 DQN 在 CartPole 上的完整训练循环
- [ ] 对比 Rainbow 的各项改进及其作用

---

## 1. 问题

表格 Q 学习需要为每个 (状态, 动作) 对存储一个独立的 Q 值。棋盘游戏有 ~10⁴³ 个状态。Atari 帧是 210×160×3 = 100,800 个特征。表格 RL 在几千个状态时就死了，更不用说数十亿。

修复方案事后看来很明显：用神经网络 Q(s,a;θ) 替代 Q 表。但"事后明显"花了数十年。"致命三要素"——函数逼近 + 自举 + 离策略——使朴素函数逼近发散。Mnih 等人确定了三个稳定学习的工程技巧：

1. **经验回放**（Experience Replay）——去相关转移
2. **目标网络**（Target Network）——冻结自举目标
3. **奖励裁剪**（Reward Clipping）——归一化梯度幅度

DQN 在 Atari 上是首次一个单一架构、单一超参数集从原始像素解决了数十个控制问题。此后的一切"深度 RL"——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都堆叠在这个三技巧之上。

---

## 2. 概念

### 2.1 DQN 训练目标

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} [(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2]$$

- θ：在线网络，每步梯度下降更新
- θ^-：目标网络，每 C 步从 θ 复制
- D：过去转移的经验回放缓冲区

### 2.2 三个核心技巧

**1. 经验回放**

一个 ~10⁶ 条转移的环形缓冲区。每步随机均匀采样小批量。这打破了时间相关性（连续帧几乎相同），让网络可以从罕见奖励转移中多次学习，并去连续梯度更新。没有它，在策略 TD 加神经网络在 Atari 上发散。

```python
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buf = []
        self.capacity = capacity
        self.pos = 0

    def push(self, s, a, r, s_next, done):
        if len(self.buf) < self.capacity:
            self.buf.append((s, a, r, s_next, done))
        else:
            self.buf[self.pos % self.capacity] = (s, a, r, s_next, done)
        self.pos += 1

    def sample(self, batch_size):
        import random
        return random.sample(self.buf, min(batch_size, len(self.buf)))
```

**2. 目标网络**

在贝尔曼方程两侧使用同一个 Q(·;θ) 会使目标每步都移动——"追自己的尾巴"。修复：保留第二个网络 Q(·;θ^-) 冻结权重。每 C 步复制 θ→θ^-。这使回归目标一次稳定数千个梯度步。软更新 θ^- ← τθ + (1-τ)θ^-（DDPG、SAC 中使用）是更平滑的变体。

**3. 奖励裁剪**

Atari 奖励幅度范围从 1 到 1000+。裁剪为 {-1, 0, +1} 防止单个游戏主导梯度。当奖励幅度重要时是错误的——对 Atari 合适，因为只有符号重要。

### 2.3 Double DQN

Hasselt (2016) 修复最大化偏差——用在线网络选择动作，目标网络评估动作：

$$target = r + \gamma Q(s', \arg\max Q(s',a';\theta); \theta^-)$$

即插即用，持续更好。始终默认使用。

### 2.4 Rainbow 的七个改进

| 改进 | 发表 | 效果 |
|------|------|------|
| Double DQN | 2016 | 修复最大化偏差 |
| Prioritized Replay | 2016 | 高 TD 误差的转移采样更多 |
| Dueling Network | 2016 | 分解 V(s) + 优势 A(s,a) |
| n-step Returns | 2017 | 多步自举 |
| Noisy Nets | 2017 | 学习探索 |
| Distributional Q | 2017 | 学习 Q 分布而非均值 |
| C51 / QR-DQN | 2017 | 分位数回归 |

---

## 3. 从零实现

完整代码见 `code/main.py`（在 CartPole 上训练的 DQN）。以下是核心逻辑：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# 第 1 步：Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)

# 第 2 步：经验回放
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s = torch.FloatTensor([x[0] for x in batch])
        a = torch.LongTensor([x[1] for x in batch])
        r = torch.FloatTensor([x[2] for x in batch])
        s2 = torch.FloatTensor([x[3] for x in batch])
        done = torch.FloatTensor([x[4] for x in batch])
        return s, a, r, s2, done

# 第 3 步：DQN 训练步
def train_step(q_net, target_net, buffer, batch_size=64, gamma=0.99, lr=1e-3):
    if len(buffer.buffer) < batch_size:
        return 0.0

    s, a, r, s2, done = buffer.sample(batch_size)

    # 当前 Q 值
    q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze()

    # 目标 Q 值
    with torch.no_grad():
        max_q_s2 = target_net(s2).max(1)[0]
        targets = r + gamma * max_q_s2 * (1 - done)

    loss = F.mse_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# 第 4 步：训练循环
def train_dqn(env, q_net, target_net, num_episodes=500, batch_size=64,
              gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
              epsilon_decay=0.995, target_update=10):
    buffer = ReplayBuffer()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    epsilon = epsilon_start
    all_rewards = []

    for episode in range(num_episodes):
        s, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # ε-贪心选动作
            if random.random() < epsilon:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = q_net(torch.FloatTensor(s))
                    a = q.argmax().item()

            s2, r, done, _, _ = env.step(a)
            buffer.push(s.tolist(), a, max(r, -1.0), s2.tolist(), done)
            total_reward += r
            s = s2

            # 训练
            loss = train_step(q_net, target_net, buffer, batch_size, gamma)

        # 更新目标网络
        if episode % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        # 衰减 ε
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        all_rewards.append(total_reward)

    return all_rewards
```

---

## 4. 工具

### 4.1 Gymnasium 中的经典环境

```python
import gymnasium as gym

# CartPole——最简单的 DQN 入门
env = gym.make("CartPole-v1")
print(f"状态维度: {env.observation_space.shape}")  # (4,)
print(f"动作空间: {env.action_space.n}")           # 2

# LunarLander——更复杂的控制
env = gym.make("LunarLander-v3")
print(f"状态维度: {env.observation_space.shape}")  # (8,)
print(f"动作空间: {env.action_space.n}")           # 4
```

### 4.2 2026 年 DQN 的状态

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 离散动作小规模 | Rainbow DQN | 所有改进叠加 |
| 连续控制 | SAC / TD3 | DQN 没有策略网络 |
| 在策略/高吞吐 | PPO | 无回放缓冲区，易扩展 |
| 离线 RL | CQL / IQL | 保守 Q 目标 |
| LLM RL | PPO / GRPO | 序列级而非步级 |

---

## 5. LLM 视角

### 5.1 DQN 在大语言模型中的体现

经验回放和目标网络的思想在大语言模型训练中也有对应：

- **数据集回放**：LLM 训练中在旧数据上重放——类似于经验回放。但 LLM 通常不做在线 RL 式采样
- **冻结编码器**：微调时冻结编码器类似于目标网络的"冻结"——防止更新破坏表示
- **学习率预热**：预热阶段类似于目标网络的初始稳定期

### 5.2 经验回放 vs 在线学习

DQN 用经验回放（离策略）打破了数据相关性。PPO/GRPO 是在策略的——采样一批→更新策略→丢弃这批。两者各有优劣：DQN 数据效率高（可以重用旧数据），PPO 理论保证好（在策略梯度无偏）。

### 5.3 使用 ChatGPT / Claude 时的直接体验

ChatGPT 不是用 DQN 训练的——它用 PPO + KL 惩罚（RLHF）。但 DQN 中的"三个技巧"（回放、目标网络、梯度归一化）的思想在 LLM 训练中以其他形式存在：批处理、学习率调度、梯度裁剪。理解 DQN 是理解 LLM RL 训练的前置知识。

---

## 6. 工程最佳实践

### 6.1 DQN 超参数

| 参数 | CartPole | Atari | 说明 |
|------|---------|-------|------|
| buffer capacity | 50000 | 1,000,000 | 越大需要越多内存 |
| batch size | 64 | 32 | 小批量更稳定 |
| target update | 10 | 10,000 | 步数而非回合数 |
| ε start | 1.0 | 1.0 | 充分探索 |
| ε end | 0.01 | 0.01 | 最终策略 |
| ε decay | 0.995/步 | 1,000,000 步 | 线性衰减到最终值 |

### 6.2 致命三要素

函数逼近 + 自举 + 离策略三者同时出现时，收敛性不再有保证。DQN 用目标网络 + 经验回放缓解——不要移除任何一个。

### 6.3 踩坑经验

- **回放缓冲区冷启动**：缓冲区存够几批转移前不要训练。~20 个样本上的早期梯度会过拟合
- **目标同步太频繁**：~= 没有目标网络；太不频繁≈ 过时目标。Atari 用 10,000 步
- **奖励不裁剪**：梯度幅度与奖励幅度成正比。始终裁剪或归一化奖励

---

## 7. 常见错误

### 错误 1：在目标 Q 中使用在线网络而非目标网络

**现象：** Q 值训练不稳定，发散。

**原因：** 目标 Q = r + γmax Q(s';θ) 使用同一个 θ——目标每步移动。

```python
# ❌ 错误：使用在线网络
targets = r + gamma * q_net(s2).max(1)[0]

# ✓ 正确：使用目标网络
with torch.no_grad():
    targets = r + gamma * target_net(s2).max(1)[0]
```

### 错误 2：不在缓冲区冷启动后验证样本量

**现象：** 训练开始几回合 loss=0 然后突然变大。

**原因：** 缓冲区未满时 sample() 返回空。

```python
# ❌ 错误：直接采样
s, a, r, s2, done = buffer.sample(batch_size)

# ✓ 正确：检查缓冲区大小
if len(buffer.buffer) < batch_size:
    return
```

### 错误 3：ε 衰减太快

**现象：** 策略收敛到次优解。

**原因：** ε 在探索足够前降到了最小值。

```python
# ❌ 错误：衰减太快
epsilon = max(epsilon_end, epsilon * 0.99)  # 每步衰减 1%

# ✓ 正确：衰减合理
epsilon = max(epsilon_end, epsilon * 0.995)  # 每回合衰减
```

---

## 8. 面试考点

### Q1：DQN 的三个核心技巧是什么？为什么每个都是必要的？（难度：⭐⭐）

**参考答案：**
(1) 经验回放——打破连续转移的时间相关性，使 iid 假设近似成立；(2) 目标网络——冻结贝尔曼目标的自举来源，防止"追尾巴"效应；(3) 奖励裁剪——归一化不同游戏之间的奖励幅度，使单一超参数集在所有 Atari 游戏上有效。三者共同解决了"致命三要素"（函数逼近+自举+离策略）导致的不稳定性。

### Q2：什么是"致命三要素"？DQN 如何解决？（难度：⭐⭐⭐）

**参考答案：**
致命三要素指函数逼近（神经网络）、自举（TD 目标）、离策略（Q 学习）三者同时出现时收敛性丧失。DQN 没有完全解决它——而是用工程技巧缓解：(1) 经验回放使数据接近 iid，缓解了函数逼近与环境非平稳分布之间的矛盾；(2) 目标网络使自举目标在数千步内稳定，减少了目标移动导致的振荡。在极限下 DQN 仍然可能发散（实践中在 Atari 上稳定）。

### Q3：Double DQN 为什么能修复最大化偏差？（难度：⭐⭐⭐）

**参考答案：**
标准 DQN 中，max 操作对噪声 Q 值有系统性向上偏差——即使 Q 的误差均值为零，max(Q) 的期望 > max(True Q)。Double DQN 将"选择"和"评估"分到两个网络：在线网络选动作（argmax），目标网络评估这个动作的值。由于两个网络的噪声近似独立，选择-评估的解耦消除了系统性向上偏差。即插即用、持续更好、无额外计算成本。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| DQN | "深度 Q 学习" | Q 学习 + 神经网络 + 经验回放 + 目标网络 |
| 经验回放 | "打乱的转移" | 从环形缓冲区均匀采样，去相关数据 |
| 目标网络 | "冻结的自举" | Q 的定期副本，稳定训练目标 |
| 致命三要素 | "RL 为什么发散" | 函数逼近 + 自举 + 离策略 = 无收敛保证 |
| Double DQN | "修复最大化偏差" | 在线网络选动作，目标网络评估 |
| Rainbow | "所有技巧叠加" | DDQN + PER + dueling + n-step + noisy + distributional |
| 奖励裁剪 | "梯度归一化" | 将奖励限制在 {-1,0,+1} 防梯度爆炸 |

---

## 📚 小结

DQN 用神经网络替代 Q 表、用经验回放打破时间相关性、用目标网络稳定训练。2013 年在 Atari 上超越人类——深度 RL 的起点。Double DQN 修复最大化偏差。Rainbow 叠加七项改进。DQN 的三个技巧（回放、目标网络、裁剪）是后续 SAC、TD3、PPO 的基础模块。下一课我们将从"学习价值"转向"直接优化策略"——策略梯度 REINFORCE。

---

## ✏️ 练习

1. **【实现】** 在 CartPole 上实现标准 DQN。画出每回合回报曲线，记录 500 回合后的平均回报。
2. **【实验】** 移除目标网络（用在线网络做两侧）。观察训练曲线的变化——是否发散或振荡？
3. **【实现】** 在有噪声奖励的 GridWorld 上实现 Double DQN，对比标准 DQN 的 Q 估计偏差。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| DQN 完整实现 | `code/main.py` | Q 网络、经验回放、目标网络、训练循环 |

---

## 📖 参考资料

1. [论文] Mnih et al. "Playing Atari with Deep Reinforcement Learning". NeurIPS Workshop, 2013. https://arxiv.org/abs/1312.5602
2. [论文] Mnih et al. "Human-level control through deep reinforcement learning". Nature, 2015. https://www.nature.com/articles/nature14236
3. [论文] Hasselt et al. "Deep Reinforcement Learning with Double Q-learning". AAAI, 2016. https://arxiv.org/abs/1509.06461
4. [论文] Wang et al. "Dueling Network Architectures for Deep Reinforcement Learning". ICML, 2016. https://arxiv.org/abs/1511.06581
5. [论文] Hessel et al. "Rainbow: Combining Improvements in Deep Reinforcement Learning". AAAI, 2018. https://arxiv.org/abs/1710.02298
6. [官方文档] OpenAI Spinning Up — DQN: https://spinningup.openai.com/en/latest/algorithms/dqn.html

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
