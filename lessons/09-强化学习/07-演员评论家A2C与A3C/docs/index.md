# 演员-评论家——A2C 与 A3C

> REINFORCE 太吵了。加一个学习 V̂(s) 的评论家，从回报中减去它，你就得到一个期望相同但方差低得多的优势函数。这就是演员-评论家。A2C 同步运行；A3C 跨线程运行。两者都是每个现代深度 RL 方法的心理模型。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 09 · 05（DQN）、06（策略梯度）| **时间：** ~75 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 09 · 06（REINFORCE）— 从运行平均基线到学习到的基线 | 阶段 09 · 08（PPO）— 在 A2C 上加裁剪

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解演员-评论家架构——演员（策略 π）和评论家（价值 V）协同工作
- [ ] 实现 GAE（广义优势估计）——λ 参数控制偏差-方差权衡
- [ ] 实现 A2C——同步并行 + n-step 优势
- [ ] 说明 A3C 的异步并行优势——多个 worker 独立采样
- [ ] 解释 2026 年为什么 A2C 比 A3C 更常用

---

## 1. 问题

朴素 REINFORCE 有效，但它的方差糟糕透了。MC 回报 G_t 在不同回合之间可能波动 10 倍。将这个噪声乘上 ∇logπ 再平均——产生的梯度估计器需要数千回合才能将策略移动一点距离。

方差来自使用原始回报。如果减去一个基线 b(s_t)——任何只依赖状态的函数——期望不变，方差降低。最佳可计算基线是 V̂(s_t)。现在乘上 ∇logπ 的量是**优势**：

$$A(s,a) = G - \hat{V}(s)$$

一个动作如果产生了高于平均的回报就是好的；低于平均就是坏的。有学习到的评论家的 REINFORCE 就是**演员-评论家**。评论家给了演员一个低方差的老师。这就是 2015 年后每个深度策略方法（A2C、A3C、PPO、SAC、IMPALA）。

---

## 2. 概念

### 2.1 两个网络，一个共享损失

```
演员 π_θ(a|s)： 策略。采��来行动。用策略梯度训练。
评论家 V_φ(s)： 估计从状态开始的期望回报。最小化 (V_φ(s) - target)² 训练。
```

### 2.2 优势的两种形式

| 形式 | 公式 | 偏差 | 方差 |
|------|------|------|------|
| MC 优势 | A_t = G_t - V_φ(s_t) | 无偏 | 高 |
| TD 残差 | A_t = r + γV_φ(s') - V_φ(s) | 有偏（V̂(s') 自举） | 极低 |

通常使用两者的插值：n-step 优势。

### 2.3 n-step 优势

$$A_t^{(n)} = r_{t+1} + γr_{t+2} + ... + γ^{n-1}r_{t+n} + γ^nV_φ(s_{t+n}) - V_φ(s_t)$$

- n=1 是纯 TD
- n=∞ 是 MC
- 大多数实现 Atari 用 n=5，MuJoCo 用 n=2048

### 2.4 GAE——广义优势估计

Schulman 等人 (2016) 提出了对所有 n-step 优势的指数加权平均：

$$A_t^{GAE} = \sum_{l=0}^{\infty} (γλ)^l δ_{t+l}$$

其中 λ ∈ [0,1]：
- λ=0：纯 TD（低方差，高偏差）
- λ=1：MC（高方差，无偏）
- λ=0.95：2026 年默认

### 2.5 A2C vs A3C

| | A2C（同步） | A3C（异步） |
|---|-----------|-----------|
| 训练方式 | 所有 worker 计算梯度后同步更新 | 每个 worker 独立更新主网络 |
| GPU 利用率 | 高（批量前向/反向） | 低（单线程前向） |
| 确定性 | 确定性的（可重放） | 非确定性的（竞态条件） |
| 2026 年 | 默认选择 | 较旧，GPU 批量化后更好 |

### 2.6 组合损失

$$L(θ, φ) = -\mathbb{E}[A_t · \log π_θ(a_t | s_t)] + c_v · \mathbb{E}[(V_φ(s_t) - G_t)^2] - c_e · \mathbb{E}[H(π_θ(·|s_t))]$$

三项：策略梯度损失、价值回归、熵奖励。c_v ≈ 0.5，c_e ≈ 0.01 是标准起点。

---

## 3. 从零实现

### Step 1：评论家更新

```python
def critic_update(w, x, target, lr):
    """线性评论家 V_φ(s) = w · features(s)。"""
    v_hat = sum(w[j] * x[j] for j in range(len(w)))
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

### Step 2：GAE 优势计算

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    """GAE 优势估计。"""
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

### Step 3：完整 A2C 更新

```python
def a2c_update(theta, w, trajectory, gamma=0.99, lam=0.95, lr_a=0.01, lr_c=0.01):
    """A2C 更新：GAE 优势 + 策略梯度 + 价值回归。"""
    # 提取转移
    s_list = [t[0] for t in trajectory]
    a_list = [t[1] for t in trajectory]
    r_list = [t[2] for t in trajectory]
    probs_list = [t[3] for t in trajectory]

    # 计算 V(s)
    values = [sum(w[j] * s[j] for j in range(len(w))) for s in s_list]
    last_val = 0.0  # 终止状态 V=0

    # GAE 优势
    advantages, returns = compute_gae(r_list, values, gamma, lam, last_val)

    for t in range(len(trajectory)):
        x, a = s_list[t], a_list[t]
        adv = advantages[t]
        target_v = returns[t]

        # 评论家更新
        critic_update(w, x, target_v, lr_c)

        # 演员更新（策略梯度）
        grad_logpi = [-p for p in probs_list[t]]
        grad_logpi[a] += 1.0
        n_actions = len(grad_logpi)
        for i in range(n_actions):
            for j in range(len(x)):
                theta[i][j] += lr_a * adv * grad_logpi[i] * x[j]

    return theta, w
```

---

## 4. 工具

### 4.1 Stable-Baselines3（SB3）

```python
from stable_baselines3 import A2C
import gymnasium as gym

env = gym.make("CartPole-v1")
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# 测试
s, _ = env.reset()
total = 0.0
for _ in range(1000):
    a, _ = model.predict(s, deterministic=True)
    s, r, done, _, _ = env.step(a)
    total += r
    if done: break
```

### 4.2 A2C 与后续方法的关系

| 方法 | 与 A2C 的关系 |
|------|-------------|
| PPO | A2C + 裁剪重要度比率 × 多轮训练 |
| SAC | 离策略 A2C + 软值评论家 |
| GRPO | 无评论家的 A2C（组内相对优势） |
| DPO | A2C 坍缩为偏好排序损失 |

---

## 5. LLM 视角

### 5.1 演员-评论家在大语言模型中的体现

2026 年如果你看到 "advantage"，就想到演员-评论家。LLM 的 RLHF 训练中：

- **演员**：LLM 策略 π_θ——根据上下文生成下一个 token
- **评论家**：价值网络 V_φ——估计完整回复的预期奖励
- **优势**：A_t = G_t - V_φ(s_t)——"这个 token 比平均预期好多少"

GRPO 去掉了评论家网络，用组内相对优势替代——但指导思想（降低方差的基线）与演员-评论家相同。

### 5.2 使用 ChatGPT / Claude 时的直接体验

ChatGPT 的 RLHF 训练使用了 PPO（不是 A2C），但 PPO 的演员-评论家架构与 A2C 完全相同。评论家学习预测人类对不同回复的偏好分数。你可以把 ChatGPT 看作一个庞大无比（70B+ 参数）的 A2C 演员加上评论家网络。

---

## 6. 工程最佳实践

### 6.1 超参数调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| n_step | 5-2048 | 环境越长用越多 |
| GAE λ | 0.95 | 偏差-方差平衡 |
| c_v | 0.5 | 价值损失权重 |
| c_e | 0.01 | 熵奖励权重 |
| lr_a | 1e-4~1e-3 | 演员学习率 |
| lr_c | 1e-3~1e-2 | 评��家学习率（通常更大） |

### 6.2 优势归一化

每批将优势归一化为零均值/单位标准差。以近乎零的成本大幅稳定训练。

### 6.3 踩坑经验

- **评论家冷启动**：评论家随机时，基线无意义。预热评论家几百步再开策略梯度
- **熵坍塌**：没有 c_e>0，策略几百步后变成准确定性的，停止探索
- **共享骨干**：图像输入时用共享特征提取器 + 独立头部。共享特征受益于两种损失

---

## 7. 常见错误

### 错误 1：优势符号搞反

**现象：** 策略不断恶化。

**原因：** 最大化 surrogate = 最小化 -L^{CLIP}。符号翻转是最常见的 bug。

```python
# ❌ 错误：最大化目标写成了最小化
actor_loss = advantage * log_prob  # 梯度下降会降低好动作的概率！

# ✓ 正确：最大化 → 最小化负值
actor_loss = -advantage * log_prob
```

### 错误 2：GAE 计算中使用未来的奖励

**现象：** 优势与策略梯度不对齐。

**原因：** GAE 应该使用当前策略下的回报——如果评论家是用旧数据训练的，优势估计偏差大。

### 错误 3：缺乏熵奖励

**现象：** 训练后策略完全确定性——给定状态下概率分布接近独热。

**原因：** 策略梯度奖励最优动作的概率增长——直到其他动作概率趋零。

```python
# ❌ 无熵
loss = -advantage * log_prob

# ✓ 加熵奖励
loss = -advantage * log_prob - c_e * entropy
```

---

## 8. 面试考点

### Q1：演员-评论家相比朴素 REINFORCE 的核心改进是什么？（难度：⭐⭐）

**参考答案：**
用学习到的 V̂(s) 替代运行平均基线。运行平均基线是单一标量——对所有状态都一样。V̂(s) 对每个状态给出不同的基线——"在这个状态下，平均回报是多少"。这使得优势 A_t = G_t - V̂(s_t) 的方差远低于 G_t - bar{G}，因为 V̂(s) 拟合了状态相关的基线。

### Q2：GAE 的参数 λ 如何控制偏差-方差权衡？（难度：⭐⭐⭐）

**参考答案：**
GAE = Σ(γλ)^l δ_{t+l}。λ 控制着 n-step 回报的指数衰减权重。λ=0 只使用 1 步 TD 残差——偏差大（V̂ 的估计误差）但方差极小。λ=1 使用所有步的 MC 回报——无偏但方差大。λ=0.95 中间——偏差适中，方差适中。一般规律：环境越简单（噪声越小），λ 可以越大。

### Q3：为什么 2026 年 A2C 比 A3C 更受欢迎？（难度：⭐⭐）

**参考答案：**
A3C 的异步架构（每个 worker 独立更新）在 2016 年解决了关键问题——在 CPU 上并行训练。但 A3C 的异步更新有梯度滞后问题，且难以利用 GPU 的批处理能力。A2C 将所有 worker 的梯度同步更新——可以批处理、GPU 利用率更高、结果是确定性的（可 debug）、训练稳定。2026 年 GPU 计算远快于 2016 年，A2C 的同步批处理优势远大于 A3C 的异步优势。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 演员（Actor） | "策略网络" | π_θ(a|s)，策略梯度更新 |
| 评论家（Critic） | "价值网络" | V_φ(s)，MSE 回归更新 |
| 优势（Advantage） | "比平均好多少" | A(s,a)=Q(s,a)-V(s)；乘 ∇logπ |
| GAE | "插值旋钮" | n-step 优势的指数加权和，λ 参数化 |
| TD 残差 | "TD 误差 δ" | δ = r + γV(s') - V(s)；单步优势估计 |
| A2C | "同步演员-评论家" | 批量跨环境；每轮播一个梯度步 |
| A3C | "异步演员-评论家" | worker 线程推送梯度到共享参数服务器 |

---

## 📚 小结

演员-评论家合并策略梯度（演员）和价值函数（评论家）。评论家提供一个低方差的优势基线。GAE 用 λ 控制偏差-方差权衡。A2C 同步且可批处理——2026 年的标准选择。A3C 异步但 GPU 利用率低。演员-评论家的架构（策略 + 价值 + 优势）是所有现代深度 RL 方法的基础。下一课我们在 A2C 上加裁剪——PPO。

---

## ✏️ 练习

1. **【实现】** 在 CartPole 上实现 A2C（MC 优势）。对比 REINFORCE-with-baseline 的样本效率。
2. **【实验】** 切换到 TD 残差优势（r + γV(s') - V(s)）。测量优势批次的方差——下降了多少？
3. **【实现】** 实现 GAE(λ)。尝试 λ ∈ {0, 0.5, 0.95, 1.0}，绘制最终回报 vs 样本效率。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| A2C 完整实现 | `code/main.py` | 演员+评论家、GAE 优势、n-step 更新 |

---

## 📖 参考资料

1. [论文] Mnih et al. "Asynchronous Methods for Deep Reinforcement Learning". ICML, 2016. https://arxiv.org/abs/1602.01783
2. [论文] Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation". ICLR, 2016. https://arxiv.org/abs/1506.02438
3. [教材] Sutton & Barto. "Reinforcement Learning: An Introduction". Ch. 13. 2018.
4. [教材] Konda & Tsitsiklis. "Actor-Critic Algorithms". NeurIPS, 2000. — 双时间尺度演员-评论家分解的收敛性证明
5. [GitHub] Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
