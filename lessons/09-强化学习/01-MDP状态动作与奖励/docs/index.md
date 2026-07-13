# MDP、状态、动作与奖励

> 马尔可夫决策过程由五个东西组成：状态、动作、转移、奖励、折扣。本阶段的所有 RL 算法——Q-learning、PPO、DPO、GRPO——都在这个形状上优化。学一次，后面免费读。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 1 · 06（概率分布）、阶段 2 · 01（ML 分类学）
**时间：** ~45 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 定义马尔可夫决策过程的五个对象——状态、动作、转移、奖励、折扣
- [ ] 解释贝尔曼方程——将预期回报分解为"这一步的奖励"+"折扣后的未来价值"
- [ ] 实现一个 4×4 GridWorld——可视化状态、动作、奖励的完整 MDP

---

## 1. 问题

你在写一个象棋机器人、一个库存规划器、一个交易代理、或一个训练推理模型的 PPO 循环。四个不同的领域，一个惊人的事实：**四个都坍缩到同一个数学对象。**

监督学习给你 (x, y) 对，要求你拟合一个函数。强化学习没有标签——只有一个状态流、你采取的动作、以及一个标量奖励。移动赢了吗？库存决策省钱了吗？交易赚钱了吗？LLM 刚生成的 token 让 judge 给了更高奖励吗？

你无法从这个流中学习，直到你把它形式化。"我看到了什么"、"我做了什么"、"接下来发生了什么"、"那有多好"——每个都必须变成一个你可以推理的对象。这个形式化就是马尔可夫决策过程。

---

## 2. 概念

### 2.1 五个对象

| 对象 | 符号 | 含义 |
|---|---|---|
| 状态 | `S` | 代理需要的所有信息 |
| 动作 | `A` | 代理的可选动作 |
| 转移 | `P(s'|s,a)` | 给定状态和动作，下一个状态的分布 |
| 奖励 | `R(s,a,s')` | 标量信号：赢=+1，亏=-1 |
| 折扣 | `γ∈[0,1)` | 未来奖励与当前奖励的权重。γ=0.99 → ~100 步视野；γ=0.9 → ~10 步 |

### 2.2 马尔可夫性质

$$P(s_{t+1}|s_t,a_t) = P(s_{t+1}|s_0,a_0,\dots,s_t,a_t)$$

未来只依赖现在。如果不满足——状态表示不完整。

### 2.3 策略与回报

- **策略** `π(a|s)`：状态→动作概率分布
- **回报** `G_t = r_t + γr_{t+1} + γ^2 r_{t+2} + …`：折扣后的未来奖励总和
- **价值函数** `V^π(s) = E[G_t|s_t=s]`：从 s 开始的预期回报
- **Q 值** `Q^π(s,a) = E[G_t|s_t=s, a_t=a)`：从 s 采取 a 的预期回报

### 2.4 贝尔曼方程

$$V^π(s) = \sum_a π(a|s) \sum_{s',r} P(s',r|s,a)[r + γV^π(s')]$$

$$Q^π(s,a) = \sum_{s',r} P(s',r|s,a)[r + γ\sum_{a'} π(a'|s')Q^π(s',a')]$$

**本阶段的每一个算法都在迭代这个方程。**

---

## 3. 从零实现

### Step 1：4×4 GridWorld MDP

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    new_r, new_c = min(GRID-1, max(0, state[0]+dr)), min(GRID-1, max(0, state[1]+dc))
    new_state = (new_r, new_c)
    return new_state, -1.0, new_state == TERMINAL
```

### Step 2：策略评估

```python
def evaluate_policy(policy, gamma=0.99, n_iter=100):
    """迭代贝尔曼方程计算 V^π(s)。"""
    V = [[0.0]*GRID for _ in range(GRID)]
    for _ in range(n_iter):
        V_new = [[0.0]*GRID for _ in range(GRID)]
        for r in range(GRID):
            for c in range(GRID):
                s = (r, c)
                if s == TERMINAL: V_new[r][c] = 0.0; continue
                for a in ACTIONS:
                    s_next, reward, _ = step(s, a)
                    V_new[r][c] += policy(s, a) * (reward + gamma * V[s_next[0]][s_next[1]])
        V = V_new
    return V
```

完整代码见 `code/main.py`。

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| 马尔可夫性质 | 未来只依赖现在——状态必须完整地包含决策所需的所有信息 |
| 折扣因子 γ | 控制未来奖励的权重——γ=0.99 意味着 100 步内的奖励都有意义 |
| 贝尔曼方程 | 将预期回报分解为当前奖励+折扣后的未来价值——所有 RL 算法的基础 |

---

## 📚 小结

MDP = 状态 + 动作 + 转移 + 奖励 + 折扣。贝尔曼方程将回报分解为递归结构。本阶段的所有算法——Q-learning、PPO、DPO、GRPO——都在优化这个形状。学一次 MDP，后面免费读。

---

## ✏️ 练习

1. 手动计算 2×2 GridWorld 中从每个状态到终止状态的最优策略
2. 用代码计算 4×4 GridWorld 在均匀随机策略下的 V(s)——每个位置的预期回报是多少？

---

## 📖 参考资料

1. [教材] Sutton & Barto. "Reinforcement Learning: An Introduction". 2018.
2. [论文] Mnih et al. "Playing Atari with Deep Reinforcement Learning". 2013.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
