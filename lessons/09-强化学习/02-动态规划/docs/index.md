# 动态规划——策略迭代与价值迭代

> 动态规划是 RL 的"完美世界"——假设模型完全已知（状态转移、奖励函数），直接迭代贝尔曼方程求解最优策略。它是所有采样方法（Q-learning、PPO）试图逼近的基准。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 09 · 01（MDP）| **时间：** ~75 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 09 · 01（MDP）— 贝尔曼方程的基础 | 阶段 09 · 03（蒙特卡洛）— 从精确求解到采样估计

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现策略迭代——交替评估当前策略、改进策略直到收敛
- [ ] 实现价值迭代——直接迭代贝尔曼最优方程
- [ ] 比较两种方法的收敛速度和适用场景
- [ ] 解释为什么贝尔曼算子是 γ-压缩映射
- [ ] 使用动态规划作为调试采样方法的黄金标准

---

## 1. 问题

你有一个已知模型的 MDP——可以查询任意状态-动作对的 P(s'|s,a) 和 R(s,a,s')。库存管理器知道需求分布。棋盘游戏有确定性转移。GridWorld 只需四行 Python。

Model-free RL（Q-learning、PPO、REINFORCE）是为不知道模型的情况发明的——你只能从环境中采样。但当你知道模型时，有更快、更好的方法：**动态规划**。Bellman 在 1957 年设计了它们。

2026 年你仍然需要 DP，原因有三：(1) RL 研究中每个表格环境（GridWorld、FrozenLake、CliffWalking）都用 DP 生成金标准策略；(2) 精确的 V* 值让你可以调试采样方法——如果 Q-learning 估算的 V*(s₀) 与 DP 答案偏差 30%，你的 Q-learning 有 bug；(3) 现代离线 RL 和规划方法（MCTS、AlphaZero 的搜索）都在已知或学到的模型上迭代贝尔曼备份。

---

## 2. 概念

### 2.1 策略迭代（Policy Iteration）

交替两步直到策略停止变化：

```
初始化：随机策略 π₀
重复：
  1. 评估：给定 π，计算 V^π（迭代贝尔曼方程直到收敛）
  2. 改进：给定 V^π，令 π 为对 V^π 的贪心策略
直到 π 不再变化
```

**收敛保证：** 每次改进要么保持 π 不变，要么严格增加 V^π。因为确定性策略空间有限，必然在有限步内收敛。4×4 GridWorld 通常 4-6 轮外层迭代。

### 2.2 价值迭代（Value Iteration）

将评估和改进合并为一步——直接迭代贝尔曼最优方程：

```
V(s) ← max_a Σ_{s',r} P(s',r|s,a)[r + γV(s')]
```

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后提取贪心策略。

**比策略迭代少内部评估循环**——每轮更快，但可能需要更多轮才能收敛。

### 2.3 广义策略迭代（GPI）

统一框架：价值函数和策略在双向改进循环中——任何驱动两者趋向互一致的方法（异步价值迭代、修改策略迭代、Q-learning、actor-critic、PPO）都是 GPI 的实例。

### 2.4 为什么 γ<1 很重要

贝尔曼算子 T 是 sup-范数下的 γ-压缩映射：

$$\|TV - TV'\|_\infty \leq \gamma \|V - V'\|_\infty$$

压缩映射保证唯一不动点和几何收敛速度。没有 γ<1——需要有限回合或吸收终止状态。

### 2.5 两种方法对比

| | 策略迭代 | 价值迭代 |
|---|---------|---------|
| 外层迭代 | 评估+改进交替 | 只有贝尔曼最优更新 |
| 内层评估 | 完全收敛 | 无内层循环 |
| 每轮计算量 | 较高（需完全评估） | 较低（单次扫描） |
| 总轮数 | 较少（4-6 轮） | 较多 |
| 总体速度 | 4×4 GridWorld 上差不多 | 4×4 GridWorld 上差不多 |
| 推荐场景 | 大状态空间，策略变化剧烈 | 状态空间小，快速求解 |

---

## 3. 从零实现

### Step 1：策略评估

```python
def policy_evaluation(policy_fn, gamma=0.99, tol=1e-6, max_iter=1000):
    """迭代贝尔曼方程计算 V^π(s)。"""
    V = {s: 0.0 for s in all_states()}
    for _ in range(max_iter):
        delta = 0.0
        for s in all_states():
            v = sum(
                pi_a * sum(p * (r + gamma * V[s_next])
                           for s_next, r, p in [step(s, a)])
                for a, pi_a in policy_fn(s).items()
            )
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
    return V
```

### Step 2：策略改进

```python
def policy_improvement(V, gamma=0.99):
    """根据 V^π 改进策略——对每个状态选使贝尔曼方程最大化的动作。"""
    new_policy = {}
    for s in all_states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_next])
                              for s_next, r, p in [step(s, a)]),
        )
        new_policy[s] = best_a
    return new_policy
```

### Step 3：策略迭代

```python
def policy_iteration(gamma=0.99):
    """策略迭代：评估→改进→重复直到策略不变。"""
    policy = {s: "up" for s in all_states()}  # 任意起始策略

    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy

    return V, policy
```

### Step 4：价值迭代

```python
def value_iteration(gamma=0.99, tol=1e-6, max_iter=1000):
    """价值迭代：直接迭代贝尔曼最优方程。"""
    V = {s: 0.0 for s in all_states()}

    for _ in range(max_iter):
        delta = 0.0
        for s in all_states():
            v = max(
                sum(p * (r + gamma * V[s_next])
                    for s_next, r, p in [step(s, a)])
                for a in ACTIONS
            )
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break

    policy = policy_improvement(V, gamma)
    return V, policy
```

---

## 4. 工具

### 4.1 Gymnasium 中的标准环境

```python
import gymnasium as gym

# FrozenLake——经典表格 RL 环境
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
# 状态空间: 16, 动作空间: 4

# CliffWalking——控制任务基准
env = gym.make("CliffWalking-v0")
# 状态空间: 48, 动作空间: 4
```

### 4.2 Gymnasium 自定义 MDP

```python
import gymnasium as gym
from gymnasium import spaces

class CustomGridWorld(gym.Env):
    """自定义 GridWorld 环境——实现 DP 求解器。"""
    def __init__(self, size=4):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)
        self.state = 0
        self.terminal = size * size - 1

    def reset(self, seed=None, options=None):
        self.state = 0
        return self.state, {}

    def step(self, action):
        # 实现转移逻辑...
        pass
```

---

## 5. LLM 视角

### 5.1 在主流系统中的体现

- **AlphaZero / MuZero 的搜索**：蒙特卡洛树搜索（MCTS）本质上是异步贝尔曼备份——在学到的模型上做 DP。
- **离线 RL（CQL、IQL）**：Conservative Q-Iteration 是在学到的模型上做 DP，并对 OOD 动作加惩罚。
- **模型预测控制（MPC）**：在有限视野内，对学到的动态模型做 DP 规划。

### 5.2 为什么 DP 在 2026 年仍然重要

**调试基准：** 当你的 PPO 实现在 CartPole 上训练不收敛时，用 DP 计算最优 V* 作为基准。如果 V* 本身不合理（γ 配错、奖励定义有误），你的 PPO 实现再正确也没用。

**AlphaZero 的核心：** AlphaZero 的 MCTS + DP 是在学到的模型上做异步贝尔曼备份——DP 的直接应用。

### 5.3 使用 ChatGPT / Claude 时的直接体验

当你使用 ChatGPT 时，它每一步的 token 选择本质上是一个 MDP 中的动作。但 GPT 不是通过 DP 求解的——它使用 PPO 在大量对话数据上训练。DP 告诉我们"最优策略是什么样的"，PPO 告诉我们"如何从经验中学习最优策略"。

---

## 6. 工程最佳实践

### 6.1 收敛诊断

| 指标 | 值 | 含义 |
|------|---|------|
| Δ (sup-范数) | < 1e-6 | 已收敛 |
| 策略变化步数 | 0 | 策略稳定 |
| V*(start) | 预期值 | 与理论值比较 |

### 6.2 中文场景特别建议

- FrozenLake 和 CliffWalking 都可以通过 Gymnasium 中文社区安装
- 国内 RL 研究中常用 Minigrid 做网格世界实验

### 6.3 踩坑经验

- **终止状态处理**：忘记对终止状态设 V=0，导致价值不断增长不收敛
- **就地更新 vs 同步更新**：就地更新（Gauss-Seidel）比同步更新（Jacobi）快得多，但理论分析时用同步更新
- **策略平局打破**：两个动作 Q 值相同时，`argmax` 可能每次迭代选择不同动作，导致"策略稳定"检查振荡。用固定的平局打破规则

---

## 7. 常见错误

### 错误 1：对终止状态也应用 Bellman 方程

**现象：** V 不收敛，终止状态价值不为 0。

**原因：** 终止状态之后没有状态了，但代码仍对它计算"下一步的价值"。

**修复：**

```python
# ❌ 错误
for s in all_states():
    v = max(sum(p * (r + gamma * V[s_next])
                for s_next, r, p in [step(s, a)]) for a in ACTIONS)

# ✓ 正确
for s in all_states():
    if s == TERMINAL:
        V[s] = 0.0
        continue
    v = max(...)
```

### 错误 2：使用平均误差而非最大误差

**现象：** 策略看起来稳定了，但某些状态价值估计不准确。

**原因：** 平均误差可以很小但最大误差很大——理论保证是 sup-范数。

**修复：**

```python
# ❌ 错误：平均误差
delta = np.mean([abs(V_new[s] - V[s]) for s in all_states()])

# ✓ 正确：sup-范数
delta = max(abs(V_new[s] - V[s]) for s in all_states())
```

### 错误 3：在就地更新时引用旧值

**现象：** 价值迭代收敛速度异常慢。

**原因：** 更新 V[s] 时，同一扫描中其他状态可能已经更新了——如果引用 V[s_next] 时它已被更新（就地更新），相当于 Gauss-Seidel 加速；如果没被更新（同步更新），是 Jacobi 风格，慢但理论更干净。

**修复：** 生产代码中使用就地更新加速。调试时用同步更新确保理论正确性。

---

## 8. 面试考点

### Q1：策略迭代和价值迭代的核心区别是什么？什么时候用哪个？（难度：⭐⭐）

**参考答案：**
策略迭代显式维护策略 π，交替进行策略评估（完全收敛到 V^π）和策略改进（贪心更新 π）。价值迭代不显式维护策略——直接迭代贝尔曼最优方程，最后提取贪心策略。策略迭代通常需要更少的外层迭代（4-6 轮），但每轮内部需要完全评估策略。价值迭代每轮更快（单次扫描），但需要更多轮。对于小状态空间，两者差不多；对于大状态空间，价值迭代更实用——因为完全策略评估的内部循环在大状态空间下太昂贵。

### Q2：为什么贝尔曼算子是 γ-压缩映射？压缩映射对收敛有什么保证？（难度：⭐⭐⭐）

**参考答案：**
贝尔曼算子 T 定义为 (TV)(s) = max_a Σ P(s',r|s,a)[r + γV(s')]。对于任意两个价值函数 V₁, V₂，有 ||TV₁ - TV₂||_∞ ≤ γ||V₁ - V₂||_∞。这是因为 max 和加权求和操作不会增加差异——乘以 γ 使得差异严格缩小。压缩映射定理（Banach 不动点定理）保证：(1) 存在唯一不动点 V*；(2) 从任意 V₀ 出发，迭代序列 Vₙ = TVₙ₋₁ 几何收敛到 V*；(3) 收敛速度为 γⁿ——即每轮误差缩小 γ 倍。

### Q3：动态规划与模型无关 RL 的关系是什么？（难度：⭐⭐⭐）

**参考答案：**
DP 和 model-free RL 是同一枚硬币的两面。DP 使用完整模型做精确贝尔曼备份——O(|S|·|A|) 每轮，但需要模型。Model-free RL（TD、Q-learning）从采样中估计贝尔曼方程——不需要模型但方差更高。GPI（广义策略迭代）是统一框架：DP 是 GPI 的精确版本，Q-learning、PPO 等都是 GPI 的采样版本。关键联系：(1) Q-learning 的 Q 收敛到 DP 的 Q*；(2) AlphaZero 的 MCTS 是异步 DP——在学到的模型上做贝尔曼备份；(3) 离线 RL（CQL、IQL）是在有限数据上做保守的 DP。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 策略迭代 | "DP 算法" | 交替评估（V^π）和改进（贪心 π 关于 V^π）直到策略不再变化 |
| 价值迭代 | "更快的 DP" | 贝尔曼最优备份一步扫描；几何收敛到 V* |
| 贝尔曼算子 | "递归" | (TV)(s) = max_a Σ P(r + γV(s'))；sup-范数下的 γ-压缩映射 |
| 压缩映射 | "DP 为什么收敛" | 任何算子 T 满足 ||Tx - Ty|| ≤ γ||x - y|| 就有唯一不动点 |
| GPI | "一切都是 DP" | 广义策略迭代：任何驱动 V 和 π 趋向互一致的方法 |
| 就地更新 | "Gauss-Seidel 风格" | 更新 V[s] 时使用已更新的值——收敛更快 |
| 同步更新 | "Jacobi 风格" | 一次扫描中使用旧值——理论更干净但更慢 |

---

## 📚 小结

动态规划在 MDP 已知时直接求解最优策略。策略迭代交替评估和改进，价值迭代直接迭代贝尔曼最优方程。两者都保证收敛——因为贝尔曼算子是 γ-压缩映射。DP 是所有采样方法（Q-learning、PPO）的基准——当采样方法的估计与 DP 答案偏差过大时，说明实现有 bug。下一课我们将去掉"已知模型"的假设——用蒙特卡洛方法从采样轨迹中学习。

---

## ✏️ 练习

1. **【实现】** 在 4×4 GridWorld 上分别运行策略迭代和价值迭代（γ=0.99）。记录外层迭代次数、内层评估次数、最终 V*(0,0)。对比两者差异。
2. **【实验】** 在随机版 GridWorld（slip=0.1）上，比较策略迭代和价值迭代的收敛速度。哪个在迭代次数上更快？哪个在实际运行时间上更快？
3. **【实现】** 实现修改策略迭代（Modified Policy Iteration）：评估步骤只运行 k 步而非完全收敛。画出 V*(0,0) 误差随 k 的变化曲线。
4. **【思考】** 为什么状态空间超过 ~10⁷ 时 DP 不再可行？这如何导致了函数近似（神经网络）的需求？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| 策略迭代 + 价值迭代实现 | `code/main.py` | 完整的 DP 求解器，含收敛诊断和可视化 |
| DP 求解提示词 | `outputs/skill-dp-solver.md` | 给定 MDP，输出 DP 求解方案的提示词 |

---

## 📖 参考资料

1. [教材] Sutton & Barto. "Reinforcement Learning: An Introduction". Ch. 4 — Dynamic Programming. 2018. http://incompleteideas.net/book/RLbook2020.pdf
2. [教材] Bertsekas. "Reinforcement Learning and Optimal Control". 2019. http://www.athenasc.com/rlbook.html — 压缩映射论证的严格处理
3. [教材] Puterman. "Markov Decision Processes". Wiley, 2005. — 修改策略迭代及其收敛分析
4. [论文] Howard. "Dynamic Programming and Markov Processes". MIT Press, 1960. — 原始策略迭代论文
5. [教材] Bertsekas & Tsitsiklis. "Neuro-Dynamic Programming". 1996. http://www.athenasc.com/ndpbook.html — 从 DP 到近似 DP / 深度 RL 的桥梁

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
