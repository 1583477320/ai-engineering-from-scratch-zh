# 策略梯度 REINFORCE

> 别估计价值了。直接参数化策略，计算期望回报的梯度，沿梯度上坡走。Williams (1992) 用一个定理把它写清楚了。这就是 PPO、GRPO 和每个 LLM RL 循环存在的原因。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 09 · 01（MDP）、03（蒙特卡洛）| **时间：** ~75 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 09 · 04（TD 学习）— 价值函数作为基线 | 阶段 09 · 07（A2C）— Actor-Critic 合并策略和价值

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 推导策略梯度定理——∇J(θ) = E[G_t · ∇log π_θ(a_t|s_t)]
- [ ] 实现 REINFORCE——采样轨迹、计算回报、更新策略参数
- [ ] 解释基线（baseline）如何降低方差——从回报中减去平均回报
- [ ] 理解 REINFORCE 与 PPO、GRPO 的联系
- [ ] 诊断策略梯度的高方差问题并应用修复技巧

---

## 1. 问题

Q 学习和 DQN 参数化**价值函数**。你通过 `argmax Q` 选动作。这对离散动作和离散状态很好。但当动作连续（对 10 维扭矩做 argmax？）或你想要随机策略（argmax 天生确定性）时就不行了。

策略梯度直接参数化**策略**。π_θ(a|s) 是一个输出动作分布的神经网络。从它采样来行动。计算期望回报对 θ 的梯度。沿梯度上坡走。没有 argmax。没有贝尔曼递归。只有对 J(θ)=E[G] 的梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：

$$ \nabla J(\theta) = \mathbb{E}_\pi[ G_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) ] $$

运行一个回合。计算回报。乘上每一步的 ∇logπ_θ(a_t|s_t)。平均。梯度上升。完成。

2026 年的每个 LLM RL 算法——PPO、DPO、GRPO——都是 REINFORCE 的改进版本。从指尖理解它是本课程后半部分以及 RLHF 实现（阶段 10 · 07）和 DPO（阶段 10 · 08）的前置条件。

---

## 2. 概念

### 2.1 策略梯度定理

对于任意的策略 π_θ，参数化为 θ：

$$ \nabla J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} G_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) \right] $$

直觉：**回报高的动作增加其概率，回报低的动作降低其概率。** G_t 是从该步开始的折扣回报。

### 2.2 Softmax 策略

对于离散动作，标准选择：

```python
def softmax_policy(features, theta, n_actions):
    """线性 softmax 策略。theta 是 (n_actions, n_features) 的权重。"""
    scores = [sum(theta[a][j] * features[j] for j in range(len(features)))
              for a in range(n_actions)]
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]
```

梯度 ∇logπ(a|s) 的简洁形式：onehot(a) - π(·|s)——选中动作的指示向量减去策略分布。

### 2.3 方差降低技巧

**回报到后（Reward-to-Go）：** 用 G_t^{(from t)} 替代 Σ_t G_t。只有未来回报对当前动作有贡献。

**基线（Baseline）：** 从 G_t 减去 b(s_t)。任何不依赖 a_t 的基线都成立。典型选择：b(s_t) = V̂(s_t)，由 critic 学习——这就成了 Actor-Critic。

```python
# REINFORCE with baseline
advantage = G - baseline  # baseline = 运行平均回报或 V̂(s)
```

**熵奖励：** 添加 β·H(π(·|s)) 防策略过早坍塌为确定性。

### 2.4 REINFORCE vs 后续方法

| 方法 | 与 REINFORCE 的关系 |
|------|-------------------|
| A2C | REINFORCE + 学习到的 V(s) 作为基线 |
| PPO | REINFORCE + 剪切重要度采样比 + KL 惩罚 |
| GRPO | REINFORCE + 组内相对基线（无 critic） |
| DPO | REINFORCE 重写为代表偏好损失——无需采样 |
| RLHF | REINFORCE + KL 惩罚 + 奖励模型 |

---

## 3. 从零实现

### Step 1：Softmax 策略和采样

```python
import random
import math

def softmax(scores):
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]

def sample_action(probs):
    x = random.random()
    cum = 0.0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return math.log(probs[a] + 1e-12)
```

### Step 2：Rollout

```python
def rollout(env, theta, n_actions):
    """运行一个回合，记录轨迹和策略概率。"""
    trajectory = []
    s = env.reset()
    done = False
    while not done:
        features = s  # 表格状态：直接当作特征
        probs = softmax_policy(features, theta, n_actions)
        a = sample_action(probs)
        s_next, r, done = env.step(a)
        trajectory.append((features, a, r, probs))
        s = s_next
    return trajectory
```

### Step 3：计算回报

```python
def compute_returns(trajectory, gamma=0.99):
    """反向遍历计算每步的折扣回报。"""
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

### Step 4：REINFORCE 更新

```python
def reinforce_update(theta, trajectory, gamma=0.99, lr=0.01, baseline=0.0):
    """
    REINFORCE 更新。∇logπ(a|s) = onehot(a) - π(·|s)。
    Args:
        theta: 策略权重 (n_actions, n_features)
        trajectory: rollout 记录的轨迹
        baseline: 基线（运行平均回报或 V̂(s)）
    """
    returns = compute_returns(trajectory, gamma)
    n_actions = len(theta)
    n_features = len(theta[0])

    for (features, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        # ∇log π(a|s) = e_a - π
        grad = [-p for p in probs]
        grad[a] += 1.0
        # 更新 θ (手写 SGD)
        for i in range(n_actions):
            for j in range(n_features):
                theta[i][j] += lr * advantage * grad[i] * features[j]

    return theta
```

### Step 5：REINFORCE 完整循环

```python
def reinforce(env, n_actions, n_features, gamma=0.99, lr=0.01,
              num_episodes=2000):
    """REINFORCE 完整训练循环。"""
    theta = [[0.0] * n_features for _ in range(n_actions)]
    running_baseline = 0.0
    all_returns = []

    for ep in range(num_episodes):
        trajectory = rollout(env, theta, n_actions)
        returns = compute_returns(trajectory, gamma)
        ep_return = returns[0] if returns else 0.0
        all_returns.append(ep_return)

        # 更新基线（指数移动平均）
        running_baseline = 0.9 * running_baseline + 0.1 * ep_return

        theta = reinforce_update(theta, trajectory, gamma, lr, running_baseline)

        if (ep + 1) % 500 == 0:
            avg = sum(all_returns[-100:]) / min(100, len(all_returns[-100:]))
            print(f"Episode {ep+1}: 平均回报 = {avg:.2f}")

    return theta, all_returns
```

---

## 4. 工具

### 4.1 Gymnasium 中的经典环境

```python
import gymnasium as gym

# CartPole——最简单的策略梯度入门
env = gym.make("CartPole-v1")
print(f"状态维度: {env.observation_space.shape}")  # (4,)
print(f"动作空间: {env.action_space.n}")           # 2

# LunarLander——更高维的空间
env = gym.make("LunarLander-v3")
print(f"状态维度: {env.observation_space.shape}")  # (8,)
print(f"动作空间: {env.action_space.n}")           # 4
```

### 4.2 连续动作的策略梯度

```python
# 连续动作：高斯策略
# π_θ(a|s) = N(μ_θ(s), σ_θ(s))
# ∇log π = [(a - μ)/σ²] * ∇μ + [(a - μ)²/σ³ - 1/σ] * ∇σ
```

---

## 5. LLM 视角

### 5.1 REINFORCE 在大语言模型中的体现

当你在 2026 年看到 `loss = -advantage * log_prob` 时，那就是 REINFORCE with baseline。PPO、DPO、GRPO 都是这个单行上的方差降低技巧。

| LLM RL 算法 | REINFORCE 对应 |
|------------|---------------|
| PPO | REINFORCE + 剪切重要度采样比 + KL |
| GRPO | REINFORCE + 组内相对基线（无 critic） |
| DPO | REINFORCE 重写为代表偏好损失 |
| RLOO | REINFORCE + leave-one-out 基线 |

### 5.2 GRPO——REINFORCE 的简化版

DeepSeek 的 GRPO 去掉了 critic 网络，用组内回报的相对优势代替。对同一个 prompt 采样多个回复，计算它们的回报，用每个回复的 G - 组均值 作为优势。这本质上是一个 on-policy REINFORCE with baseline（baseline = 组内平均回报）。

### 5.3 使用 ChatGPT / Claude 时的直接体验

RLHF 训练中的 PPO 步在做 `loss = -A_t * log π_θ(a_t|s_t)`——REINFORCE 的核心。A_t = G_t - V(s_t) 是优势。理解 REINFORCE 就是理解 LLM 训练中最关键的一步。

---

## 6. 工程最佳实践

### 6.1 方差控制金字塔

```
梯度上升 ← 高方差（REINFORCE vanilla）
  ↓ 加基线（running mean G）
  ↓ 回报到后（reward-to-go）
  ↓ 学习 V̂(s)（actor-critic）
  ↓ 剪切 IS 比（PPO）
梯度上升 ← 低方差（但偏差上升）
```

### 6.2 超参数

| 参数 | REINFORCE | PPO | 说明 |
|------|----------|-----|------|
| 学习率 | 1e-3 ~ 1e-2 | 1e-4 ~ 3e-4 | REINFORCE 需要更大 lr |
| 回合数 | 1000+ | 10^6+ | 方差大需要更多采样 |
| γ | 0.99 | 0.99 | 标准折扣 |
| β (熵) | 0.01 | 0.01 | 防策略坍塌 |
| 基线 | 运行平均 | 学习 V̂(s) | 方差降低程度不同 |

### 6.3 踩坑经验

- **梯度爆炸**：始终归一化回报到 N(0,1) 再乘 ∇logπ
- **熵坍塌**：策略过早收敛到确定性，停止探索。添加熵奖励 β·H(π)
- **高方差**：vanilla REINFORCE 需要数千回合。基线 + 回报到后是标准修复

---

## 7. 常见错误

### 错误 1：忽略回报到后

**现象：** 更新噪声极大，收敛极慢。

**原因：** 每一步的回报包含了过去奖励——过去动作的贡献是噪声。

```python
# ❌ 错误：用完整回报
G_total = compute_return(trajectory)[0]
for (s, a, _), _ in zip(trajectory, [G_total] * len(trajectory)):
    update(theta, s, a, G_total)

# ✓ 正确：每步用不同的回报到后
returns = compute_returns(trajectory=trajectory)  # 每步 G_t
for (s, a, _), G in zip(trajectory, returns):
    update(theta, s, a, G)
```

### 错误 2：基线计算方式错误导致有偏梯度

**现象：** 策略不收敛到最优。

**原因：** 基线依赖于当前采样中的动作（违反了"不能依赖 a"的条件）。

### 错误 3：不关心策略熵

**现象：** 训练后期策略变成完全确定性，不再探索。

**原因：** 策略梯度奖励最优动作的概率增长——直到其他动作概率接近 0。

```python
# ❌ 无熵奖励
loss = -advantage * log_prob

# ✓ 加熵奖励
loss = -advantage * log_prob - beta * entropy  # beta = 0.01
```

---

## 8. 面试考点

### Q1：策略梯度定理的核心推导步骤是什么？（难度：⭐⭐⭐）

**参考答案：**
从 J(θ) = E[G] 开始，将其写为 Σ_τ P(τ;θ) G(τ)。对 θ 求梯度：∇J = Σ_τ ∇P(τ;θ) G(τ)。用对数导数技巧 ∇P = P·∇logP，得 ∇J = Σ_τ P(τ;θ) ∇logP(τ;θ) G(τ)。P(τ) 分解为 Ππ(a|s) * ΠP(s'|s,a)。后者的梯度为 0（环境不依赖 θ）。得 ∇J = E[ (Σ ∇logπ(a_t|s_t)) · G ] = E[ Σ G_t · ∇logπ(a_t|s_t) ]。

### Q2：REINFORCE 的高方差问题如何缓解？（难度：⭐⭐）

**参考答案：**
(1) 基线（baseline）——减去不依赖动作的 b(s)，大幅降低方差且无偏；(2) 回报到后（reward-to-go）——只使用未来回报而非完整回报；(3) 熵奖励——防策略坍塌保持探索多样性；(4) Actor-Critic——用学习到的 V̂(s) 替代运行平均基线，更精确；(5) PPO 的剪切 IS 比——限制单步更新幅度防破坏性更新。

### Q3：REINFORCE 和 GRPO 的关系是什么？（难度：⭐⭐⭐）

**参考答案：**
GRPO 本质上是 REINFORCE with group-relative baseline。对一个 prompt 采样 K 个回复，计算每个回复的回报 G_i，然后用 G_i - mean(G) 作为优势。这个组内均值就是基线。GRPO 去掉了 critic 网络（避免训练不稳定），baseline 来自组内统计量而非学习。这等价于 REINFORCE with baseline，其中 baseline 是"组内平均回报"。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 策略梯度 | "直接训练策略" | ∇J(θ)=E[G·∇logπ_θ]；从对数导数技巧导出 |
| REINFORCE | "原始 PG 算法" | Williams (1992)；MC 回报乘 log-策略梯度 |
| 对数导数技巧 | "Score 函数估计器" | ∇P = P·∇logP；使梯度求解可处理 |
| 基线 | "方差降低" | 从 G 中减去的 b(s)；无偏因 E[b·∇logπ]=0 |
| 回报到后 | "只有未来回报重要" | 用 G_t^{(from t)} 而非完整 G_0 |
| 熵奖励 | "鼓励探索" | +β·H(π(·|s)) 防策略坍塌 |
| 优势 | "比平均好多少" | A(s,a)=G(s,a)-V(s)；REINFORCE with baseline 的乘数 |

---

## 📚 小结

REINFORCE 是策略梯度的基础——采样轨迹、计算回报、更新策略。基线降低方差。回报到后进一步降噪。PPO、DPO、GRPO 都是 REINFORCE 的改进。2026 年 LLM 的 RLHF 训练本质上在做 REINFORCE：loss = -A_t * logπ(a_t|s_t)。下一课 Actor-Critic（A2C/A3C）将策略梯度和价值函数合并——用学习到的 V(s) 替代运行平均基线。

---

## ✏️ 练习

1. **【实现】** 在 CartPole 上实现 REINFORCE——无基线版本。测量 2000 回合后的方差。
2. **【实验】** 添加运行平均基线，对比收敛速度和最终性能。
3. **【实验】** 添加熵奖励 β∈{0, 0.01, 0.1}。对比策略熵和最终回报。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| REINFORCE 完整实现 | `code/main.py` | Softmax 策略、rollout、回报计算、基线更新 |

---

## 📖 参考资料

1. [论文] Williams. "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning". Machine Learning, 1992. https://link.springer.com/article/10.1007/BF00992696
2. [论文] Sutton et al. "Policy Gradient Methods for Reinforcement Learning with Function Approximation". NeurIPS, 1999. https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html
3. [教材] Sutton & Barto. "Reinforcement Learning: An Introduction". Ch. 13. 2018.
4. [官方文档] OpenAI Spinning Up — VPG / REINFORCE: https://spinningup.openai.com/en/latest/algorithms/vpg.html
5. [论文] Peters & Schaal. "Reinforcement Learning of Motor Skills with Policy Gradients". 2008. — 方差降低和自然梯度视角

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
