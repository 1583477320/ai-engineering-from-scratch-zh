# 蒙特卡洛方法——从完整回合中学习

> 动态规划需要模型。蒙特卡洛什么都不需要——只需要运行策略、观察回报、取平均。RL 中最简单的想法——也是开启下游一切的钥匙。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 09 · 01（MDP）、02（动态规划）| **时间：** ~75 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 09 · 02（动态规划）— 从精确求解到采样估计 | 阶段 09 · 04（Q-learning 与 SARSA）— MC 到时序差分的跃迁

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现 first-visit 和 every-visit 蒙特卡洛策略评估
- [ ] 实现 ε-greedy 策略的蒙特卡洛控制
- [ ] 实现 off-policy 蒙特卡洛（重要度采样）
- [ ] 对比蒙特卡洛估计与 DP 精确解——理解方差-精度权衡
- [ ] 解释为什么蒙特卡洛方法只能用于回合制任务

---

## 1. 问题

动态规划需要知道 P(s'|s,a)——可以精确计算，但现实中几乎不可能。机器人无法解析计算机力矩后的相机像素分布。定价算法无法积分每个可能的客户反应。LLM 无法枚举一个 token 后的所有可能的延续。

你需要一种只需要**从环境中采样**的方法。运行策略，得到一条轨迹 s₀, a₀, r₁, s₁, a₁, r₂, ..., s_T，用它估计价值。

从 DP 到 MC 的转变在哲学上重要：**从已知模型 + 精确备份**到**采样轨迹 + 平均回报**。方差变大了，但可用性爆炸了。本课之后所有 RL 算法——TD、Q-learning、REINFORCE、PPO、GRPO——本质上都是蒙特卡洛估计器，有时叠加了自举。

---

## 2. 概念

### 2.1 直观理解

动态规划像一个看到迷宫完整地图的人，能直接算出最优路径。蒙特卡洛像一个蒙上眼睛的人，只能通过在迷宫里乱走、记住每条走过的路和最终结果，来猜测"哪条路更好"。

```
DP：精确计算     V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[r + γV^π(s')]
MC：采样平均     V^π(s) ≈ (1/N) Σ_i G^(i)(s)
```

随着 N 增大，MC 估计趋近 DP 精确值。

### 2.2 First-visit vs Every-visit MC

| 方法 | 原理 | 偏差 | 收敛速度 |
|------|------|------|---------|
| First-visit | 每次首次访问计一次 | 无偏 | 较慢（每回合一个样本） |
| Every-visit | 每次访问都计一次 | 略微有偏 | 更快（每回合多个样本） |

两者在极限下都是无偏的。First-visit 更容易分析（iid 样本）。Every-visit 每回合使用更多数据，实践中收敛更快。

### 2.3 增量平均

不存储所有回报，用运行平均更新：

$$V_n(s) = V_{n-1}(s) + (1/n)[G_n - V_{n-1}(s)]$$

改写为：`V_new = V_old + α · (target - V_old)`，α=1/n。用常数 α ∈ (0, 1) 替换 1/n → 非平稳 MC 估计器，能跟踪 π 的变化。这一步就是从 MC 到 TD 到所有现代 RL 算法的跃迁。

### 2.4 探索问题

DP 通过枚举接触所有状态。MC 只看到策略访问的状态。如果 π 是确定性的，整个状态空间区域从未被采样。

三种修复方法：

| 方法 | 优点 | 缺点 |
|------|------|------|
| **Exploring Starts** | 理论保证覆盖所有 (s,a) | 不现实 |
| **ε-greedy** | 实用、渐近覆盖 | 永远有 ε 概率次优 |
| **Off-policy** | 可以复用旧数据 | 方差爆炸 |

### 2.5 蒙特卡洛控制

与策略迭代相同——评估→改进→评估，但评估是基于采样的：

1. 运行 π，得到一个回合
2. 从观测回报更新 Q(s,a)
3. 令 π 为关于 Q 的 ε-greedy 策略
4. 重复

收敛到 Q* 和 π*，条件是每个 (s,a) 无限频繁访问、α 满足 Robbins-Monro。

---

## 3. 从零实现

### Step 1：Rollout——> (s, a, r) 序列

```python
def rollout(env, policy, max_steps=200):
    """运行策略生成一条轨迹。"""
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

两个 API：`env.reset()` 和 `env.step(s, a)`。

### Step 2：计算回报（反向遍历）

```python
def returns_from(trajectory, gamma=0.99):
    """从轨迹计算每个时间步的折扣回报 G_t。"""
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次遍历，O(T)。反向递归 G_t = r_{t+1} + γG_{t+1} 避免了重求和。

### Step 3：First-visit MC 评估

```python
from collections import defaultdict

def mc_policy_evaluation(env, policy, num_episodes=5000, gamma=0.99):
    """First-visit 蒙特卡洛策略评估。"""
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(num_episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, _, _), G in zip(trajectory, returns):
            if s in seen:
                continue  # first-visit
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

### Step 4：ε-greedy MC 控制

```python
def mc_control(env, num_episodes=20000, gamma=0.99, epsilon=0.1):
    """ε-greedy 蒙特卡洛控制——学习最优 Q 值。"""
    Q = defaultdict(lambda: {a: 0.0 for a in ACTION_LIST})
    counts = defaultdict(lambda: {a: 0 for a in ACTION_LIST})
    def policy(s):
        if random.random() < epsilon:
            return random.choice(ACTION_LIST)
        return max(Q[s], key=Q[s].get)
    for _ in range(num_episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

---

## 4. 工具

### 4.1 Gymnasium 环境 API

```python
import gymnasium as gym

env = gym.make("Blackjack-v1", natural=True, sab=True)
# 20,000 回合 MC 控制可以学到接近最优的 Blackjack 策略

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
# 经典表格 RL 环境——适合 MC 评估
```

### 4.2 Tabular RL 库

| 库 | 用途 |
|----|------|
| Gymnasium | RL 环境标准接口 |
| Minigrid | 网格世界研究 |
| Pygame RL | 可视化 RL |

---

## 5. LLM 视角

### 5.1 在主流系统中的体现

- **PPO 的基线估计**：PPO 用 `A_t = G_t - V(s_t)`——MC 估计器的优势目标
- **蒙特卡洛树搜索（MCTS）**：AlphaZero 从树叶执行 MC rollout 指导选择和扩展
- **LLM RL 评估**：对给定策略采样完整回复，计算平均奖励——纯 MC

### 5.2 为什么 MC 在 LLM 训练中重要

RLHF 的训练流程本质上在做 on-policy MC 控制：(1) 用当前策略采样一个完整回复；(2) 奖励模型打分；(3) 用这个分数更新策略。LLM 的"回合"很短（~200 token），MC 估计的方差可以接受。

**GRPO 更进一步**：用验证器给多个采样打 0/1 分，取组内相对优势。这是一个 off-policy MC 控制。

### 5.3 使用 ChatGPT / Claude 时的直接体验

当 RLHF 训练后的 Claude 生成一段回复时，它不是 DP 求解最优的——它是在大量 MC 采样训练中找到的"相对较优"策略。这就是为什么同样的 prompt，Claude 的不同采样温度会给出不同回复。

---

## 6. 工程最佳实践

### 6.1 MC 方法选型

| | First-visit MC | Every-visit MC | Off-policy MC |
|---|---------------|----------------|-------------|
| 偏差 | 无偏 | 略微有偏 | 无偏 |
| 方差 | 低 | 更低 | 极高 |
| 适用 | 评估/教学 | 生产应用 | 数据复用 |

### 6.2 踩坑经验

- **无限回合**：MC 要求回合终止。设 max_steps 上限
- **方差爆炸**：长回合的 MC 方差极大。用 TD 方法通过自举剪枝
- **策略非平稳**：如果 π 变化，旧回合的回报来自旧策略。用常数 α 处理

---

## 7. 常见错误

### 错误 1：MC 用于非回合制任务

**现象：** 价值估计不收敛。

**原因：** MC 需要完整的回合计算回报。

**修复：** 设限或使用 TD 方法。

```python
# ❌ 错误：无终止状态
# ✓ 正确：设 max_steps 截断
```

### 错误 2：off-policy MC 的 IS 权重爆炸

**现象：** 方差极大，估计不稳定。

**原因：** 重要度采样权重沿轨迹乘积随长度指数增长。

**修复：** 使用 per-decision IS 或 weighted IS。

### 错误 3：不设置探索策略

**现象：** Q 值只在少数状态更新。

**原因：** 确定性策略永远不探索。

**修复：** 使用 ε-greedy。

```python
# ❌ 贪心
def policy(s): return argmax Q[s]
# ✓ ε-greedy
def policy(s):
    if random() < 0.1: return random_action()
    return argmax Q[s]
```

---

## 8. 面试考点

### Q1：First-visit 和 Every-visit MC 的区别？（难度：⭐⭐）

**参考答案：**
First-visit MC 每回合为每个状态贡献一个样本（第一次访问），样本独立同分布，理论分析简单。Every-visit MC 使用所有访问，每回合贡献多个样本，但有略微的正偏差。实践中 Every-visit 收敛更快。

### Q2：为什么 MC 只在回合制任务上有效？（难度：⭐⭐）

**参考答案：**
MC 需要完整的回报 G_t = Σγ^k r_{t+k+1}——从当前步到回合结束。无限持续任务的回报无限。DP 和 TD 通过自举（bootstrap）跳过这个问题——用估计值代替未来实际回报。

### Q3：为什么 MC 控制需要 ε-greedy 而不是纯贪心？（难度：⭐⭐⭐）

**参考答案：**
纯贪心只选当前 Q 最高的动作——如果某个 (s,a) 从未被采样，Q(s,a) 保持初始值，永远不会被选择。且 Q 估计有误差，"看起来最优"的动作可能不是真的最优。ε-greedy 以概率 ε 探索，渐近覆盖所有 (s,a)，确保 Q 收敛到 Q*。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 蒙特卡洛 | "随机采样" | 通过平均 iid 样本估计期望 |
| 回报 G_t | "未来奖励" | 从 t 步起的折扣奖励和 |
| First-visit MC | "每个状态用一次" | 只使用首次访问的回报 |
| Every-visit MC | "每次都用" | 每次访问都贡献 |
| ε-greedy | "探索噪音" | 1-ε 贪心，ε 随机 |
| On-policy | "用自己的数据学" | 目标=行为策略 |
| Off-policy | "用别人的数据学" | 目标≠行为策略 |

---

## 📚 小结

蒙特卡洛不需要环境模型——只需要从环境中采样回合。First-visit MC 每次访问用一个样本。ε-greedy 策略确保探索。MC 控制收敛到最优 Q*。局限：只适用于回合制任务、方差大。下一课我们将用 TD 方法突破这些局限。

---

## ✏️ 练习

1. **【实现】** 在 4×4 GridWorld 上实现 first-visit MC 评估均匀随机策略。运行 10,000 回合，画出 V(0,0) 的收敛曲线，对比 DP 值。
2. **【实验】** 用 ε ∈ {0.01, 0.1, 0.3} 运行 MC 控制，对比最终策略质量。
3. **【实现】** 实现 off-policy MC（重要度采样），对比 plain IS vs weighted IS。
4. **【思考】** 如果你有连续控制任务（如倒立摆），如何修改 MC 来使用？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| MC 评估 + 控制实现 | `code/main.py` | First-visit MC、ε-greedy MC 控制、off-policy MC |
| MC 评估提示词 | `outputs/skill-mc-evaluator.md` | 策略蒙特卡洛评估的 prompt 模板 |

---

## 📖 参考资料

1. [教材] Sutton & Barto. "Reinforcement Learning: An Introduction". Ch. 5. 2018.
2. [论文] Precup et al. "Eligibility Traces for Off-Policy Policy Evaluation". 2000.
3. [论文] Mahmood et al. "Weighted Importance Sampling for Off-Policy Learning". 2014.
4. [论文] Tesauro. "TD-Gammon". 1995.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
