# PPO——近端策略优化

> A2C 每轮更新后就丢弃轨迹。PPO 把策略梯度包裹在一个裁剪过的重要度比率里——这样你可以在同一数据上做 10+ 轮训练而不会策略爆炸。Schulman 等人 (2017)。2026 年仍然是默认的策略梯度算法。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 09 · 07（A2C/A3C）| **时间：** ~90 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 09 · 07（A2C）— A2C 的 n-step 优势 + GAE 是 PPO 的前置 | 阶段 09 · 09（RLHF）— PPO 在 RLHF 中的具体应用

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解 PPO 的 Clipped Surrogate Objective——为什么要裁剪策略更新的幅度
- [ ] 实现 PPO 的训练循环——收集轨迹、计算优势、裁剪目标
- [ ] 解释 PPO 中 KL 散度和裁剪是两种信任区域方法
- [ ] 诊断 PPO 训练的三个指标——平均 KL、裁剪比例、解释方差
- [ ] 说明 PPO 在 RLHF/GRPO 中的角色

---

## 1. 问题

A2C（第 7 课）是在策略的：梯度 E[A·∇logπ] 需要从当前 π_θ 采样的数据。取一次更新后 π_θ 就变了——之前用的数据现在变成离策略的。重复使用梯度就有偏差。

Rollout 很贵。在 Atari 上，8 个环境 × 128 步 = 1024 条转移，需要数十秒环境时间。用一次就扔了是浪费的。

**TRPO**（Trust Region Policy Optimization, Schulman 2015）是第一个修复：约束每次更新，使新旧策略之间的 KL 散度保持低于 δ。理论上干净但需要每轮共轭梯度求解。2026 年没人跑 TRPO 了。

**PPO**（Schulman 等人 2017）用简单的裁剪目标替代了硬的信任区域约束。多一行代码。每轮播 10 轮训练。没有共轭梯度。足够好的理论保证。九年后它仍然是所有领域（从 MuJoCo 到 RLHF）的默认策略梯度算法。

---

## 2. 概念

### 2.1 重要度比率

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

这是新策略与收集数据的策略的似然比。r_t=1 表示无变化。r_t=2 表示新策略对 a_t 的倾向是老策略的 2 倍。

### 2.2 裁剪代理目标

$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

两项：
- 如果优势 A_t > 0 且比率试图超过 1+ε——裁剪展平梯度（不要将一个好动作推得比旧策略多 +ε）
- 如果优势 A_t < 0 且比率试图超过 1-ε——裁剪封顶梯度（不要将一个坏动作推得比旧策略多 -ε）

典型 ε=0.2。目标关于 r_t 的函数是分段线性的——在"好的一侧"有平顶，在"坏的一侧"有平底。

### 2.3 PPO 完整损失

$$L(\theta, \phi) = L^{CLIP}(\theta) - c_v \cdot (V_\phi(s_t) - V_t^{target})^2 + c_e \cdot H(\pi_\theta(\cdot|s_t))$$

与 A2C 相同的演员-评论家结构。三个系数：c_v=0.5, c_e=0.01, ε=0.2。

### 2.4 PPO 训练循环

```
1. 在 N 个并行环境中收集 N × T 条转移
2. 计算优势（GAE），冻结为常数
3. 冻结 π_{θ_old} 为当前 π_θ 的快照
4. 对 K 轮训练，每轮采样小批量：
   - 计算 r_t(θ) = exp(log π_θ - log π_old)
   - 应用 L^{CLIP} + 价值损失 + 熵
   - 梯度步
5. 丢弃轨迹，回到步骤 1
```

K=10，小批量 64 是标准超参数。

### 2.5 PPO-KL（KL 惩罚变体）

原论文提出了另一种使用自适应 KL 惩罚的方案：

$$L = L^{PG} - \beta \cdot KL(\pi_\theta || \pi_{old})$$

β 根据观察到的 KL 调整。裁剪版本成为主流；KL 变体在 RLHF 中存活——因为 KL 到参考模型是你始终想要的独立约束。

### 2.6 诊断指标

| 指标 | 公式 | 目标值 | 问题信号 |
|------|------|-------|---------|
| 平均 KL | E[log π_old - log π_θ] | [0, 0.02] | > 0.1 → 降低 K 或 lr |
| 裁剪比例 | 比率在 [1-ε, 1+ε] 之外的样本比例 | 0.1-0.3 | ~0 → lr 太小；~0.5+ → 过拟合 |
| 解释方差 | 1 - Var(target - pred) / Var(target) | 趋近 1 | 评论家学习质量 |

---

## 3. 从零实现

### Step 1：轨迹收集（冻结 log π_old）

```python
def collect_trajectories(env, policy, n_steps=128):
    """收集轨迹并冻结 log π_old。"""
    buffer = []
    s = env.reset()
    for _ in range(n_steps):
        probs = policy(s)
        a = sample_action(probs)
        s2, r, done, _ = env.step(a)
        buffer.append({
            "s": s, "a": a, "r": r, "done": done,
            "log_pi_old": math.log(probs[a] + 1e-12),
            "probs": probs,
        })
        if done:
            s = env.reset()
        else:
            s = s2
    return buffer
```

快照在 rollout 时采集一次。在更新轮次中不变化。

### Step 2：GAE 优势（同 A2C）

```python
def compute_gae(buffer, gamma=0.99, lam=0.95, last_val=0.0):
    """计算 GAE 优势并归一化。"""
    n = len(buffer)
    rewards = [rec["r"] for rec in buffer]
    values = [sum(w[j] * rec["s"][j] for j in range(len(w))) for rec in buffer]

    advantages = [0.0] * n
    gae = 0.0
    for t in reversed(range(n)):
        next_v = values[t+1] if t+1 < n else last_val
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    # 归一化优势
    mean_adv = sum(advantages) / n
    std_adv = math.sqrt(sum((a - mean_adv)**2 for a in advantages) / n)
    advantages = [(a - mean_adv) / (std_adv + 1e-8) for a in advantages]

    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

### Step 3：PPO 裁剪更新

```python
def ppo_update(theta, w, buffer, gamma=0.99, lam=0.95, lr=0.01,
               epsilon=0.2, k_epochs=4):
    """PPO 更新：裁剪代理 + 价值损失 + 熵。"""
    advantages, returns = compute_gae(buffer, gamma, lam)

    for _ in range(k_epochs):
        for rec, adv, target_v in zip(buffer, advantages, returns):
            x = rec["s"]
            a = rec["a"]

            # 当前策略
            probs = softmax_policy(x, theta, n_actions)
            logp = math.log(probs[a] + 1e-12)
            entropy = -sum(p * math.log(p + 1e-12) for p in probs)

            # 重要度比率
            ratio = math.exp(logp - rec["log_pi_old"])

            # 裁剪代理
            surr1 = ratio * adv
            surr2 = clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
            pg_loss = -min(surr1, surr2)

            # 价值损失
            v_hat = sum(w[j] * x[j] for j in range(len(w)))
            v_loss = (v_hat - target_v) ** 2

            # 熵奖励
            e_loss = -0.01 * entropy

            # 总损失
            total_loss = pg_loss + 0.5 * v_loss + e_loss

            # 更新策略参数（手写 SGD）
            if (adv > 0 and ratio >= 1 + epsilon) or (adv < 0 and ratio <= 1 - epsilon):
                pg_grad = 0.0  # 裁剪——梯度为零
            else:
                pg_grad = ratio * adv

            grad_logpi = [-p for p in probs]
            grad_logpi[a] += 1.0
            for i in range(n_actions):
                for j in range(len(x)):
                    theta[i][j] += lr * pg_grad * grad_logpi[i] * x[j]

    return theta, w
```

---

## 4. 工具

### 4.1 Stable-Baselines3 的 PPO

```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2)
model.learn(total_timesteps=100000)

# 测试
s, _ = env.reset()
for _ in range(1000):
    a, _ = model.predict(s, deterministic=True)
    s, r, done, _, _ = env.step(a)
    if done: break
```

### 4.2 PPO 在 2026 年的应用

| 场景 | PPO 变体 |
|------|---------|
| MuJoCo/机器人控制 | PPO + 高斯策略 + GAE(0.95) |
| Atari/离散游戏 | PPO + 分类策略 + 128 步 Rollout |
| LLM RLHF | PPO + KL 惩罚到参考模型 |
| 推理 LLM | GRPO——无评论家的 PPO 变体 |
| 偏好事数据 | DPO——PPO+KL 的闭式坍缩 |

---

## 5. LLM 视角

### 5.1 PPO 在 RLHF 中的角色

InstructGPT（ChatGPT 的前身）的训练配方：

```
SFT → 奖励模型 → PPO（含 KL 惩罚到 SFT 模型）
```

LLM 的 PPO 与标准 PPO 有两个关键区别：

1. **KL 惩罚**：损失中加 β·KL(π_θ || π_ref) 防模型偏离 SFT 初始太远——不是替代裁剪，而是额外的约束
2. **奖励不是即时的**：只在完整回复结束时给分——GAE 的优势计算仍然一样

### 5.2 GRPO——无评论家的 PPO

DeepSeek 的 GRPO 去掉了 PPO 的评论家网络，用组内相对优势替代。但 PPO 的裁剪机制保留了一一r_t(θ) 的裁剪仍然限制了每步更新幅度。PPO 的"裁剪"是 GRPO 的稳定性的核心来源。

### 5.3 使用 ChatGPT / Claude 时的直接体验

当你使用经过 RLHF 训练的对话模型时，它在生成每个 token 时都在做 PPO 训练中优化的同一件事——最大化累积奖励（人类偏好）。PPO 的裁剪确保策略更新不会太激进——这就是为什么多轮 RLHF 训练后模型不会突然生成乱码。

---

## 6. 工程最佳实践

### 6.1 标准超参数

| 参数 | 标准值 | 说明 |
|------|--------|------|
| ε | 0.2 | 裁剪范围，参考标准 |
| K 轮 | 4-10 | 每轮播的 epochs |
| batch size | 64-256 | 小批量大小 |
| n_step | 128-2048 | 环境越长用越多 |
| GAE λ | 0.95 | 偏差-方差平衡 |
| c_v | 0.5 | 价值损失权重 |
| c_e | 0.01 | 熵奖励权重 |

### 6.2 调优指南

| 现象 | 修复 |
|------|------|
| KL > 0.1 | 降低 K_EPOCHS 或 LR |
| 裁剪比例 ≈ 0 | 提高 LR 或 K_EPOCHS（裁剪从未触发） |
| 裁剪比例 > 0.5 | 降低 LR 或 K_EPOCHS（过拟合 rollount） |
| 解释方差 < 0 | 评论家未学习——检查价值目标计算 |

### 6.3 踩坑经验

- **奖励归一化**：大奖励尺度会侵蚀裁剪范围。归一化奖励（运行标准差）再计算优势
- **LR 衰减**：PPO 受益于线性 LR 衰减到零。常数 LR 通常更差
- **梯度符号**：最大化 surrogate = 最小化 -L^{CLIP}。符号翻转是最常见的 PPO bug

---

## 7. 常见错误

### 错误 1：重要度比率用 `new / old` 而非指数形式

```python
# ❌ 错误：直接除法——数值不稳定
ratio = probs_new[a] / probs_old[a]  # 可能下溢

# ✓ 正确：指数形式
ratio = math.exp(logp_new - rec["log_pi_old"])
```

### 错误 2：裁剪范围的超参数调错

```python
# ❌ 太小：更新太谨慎
epsilon = 0.05  # 策略几乎不更新

# ❌ 太大：邀请不稳定
epsilon = 0.5  # 策略可以跳很远

# ✓ 标准
epsilon = 0.2
```

### 错误 3：RLHF 中忘记 KL 惩罚

```python
# ❌ LLM RLHF 中只用裁剪
loss = -min(surr1, surr2)

# ✓ 必须加 KL 惩罚
kl = KL(π_θ || π_ref)
loss = -min(surr1, surr2) + beta * kl
```

---

## 8. 面试考点

### Q1：PPO 的裁剪机制是如何工作的？为什么它能替代 TRPO 的 KL 约束？（难度：⭐⭐⭐）

**参考答案：**
裁剪机制通过限制重要度比率 r_t(θ) 在 [1-ε, 1+ε] 范围内来控制更新幅度。当优势为正时，比率增长被裁——防止过度利用一个好动作；当优势为负时，比率下降被裁——防止过度惩罚一个坏动作。TRPO 使用 KL 散度约束（需要在每一轮求解共轭梯度），PPO 的裁剪是一行代码的修改，不需要额外的求解器。裁剪的效果"足够像"信任区域——限制每次更新能使策略改变多少。

### Q2：PPO 在 LLM RLHF 中的使用与标准 PPO 有什么不同？（难度：⭐⭐⭐）

**参考答案：**
(1) **奖励结构**：标准 PPO 每步都有奖励（环境反馈），LLM RLHF 只在完整回复结束后才有奖励（奖励模型打分）。(2) **KL 惩罚**：LLM 的 PPO 通常加 β·KL(π_θ || π_ref) 惩罚——π_ref 是 SFT 初始模型，防止 RL 训练忘了语言能力。标准 PPO 没有这个 KL 项（或用它替代裁剪）。(3) **批量大小**：LLM PPO 批量以 token 计（百万级），标准 PPO 以环境步数计。(4) **评论家**：LLM 的评论家是一个独立的 transformer，与策略共享骨干或完全分开。

### Q3：PPO 的三个诊断指标——平均 KL、裁剪比例、解释方差——分别告诉你什么？（难度：⭐⭐）

**参考答案：**
(1) 平均 KL：新旧策略之间的距离。应在 0.01-0.02。超过 0.1 说明更新太大——降低 K_EPOCHS 或 LR。(2) 裁剪比例：被裁剪的样本比例。应约 0.1-0.3。~0 表示裁剪从不触发、LR 太小；~0.5+ 表示太多样本在裁剪区——过拟合。15-20% 是理想值。(3) 解释方差：评论家对价值目标的拟合质量。趋近 1 是好的，<0 表示评论家的预测比简单平均更差——检查价值目标计算。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 重要度比率 | "r_t(θ)" | π_θ/π_old；偏离收集数据的策略 |
| 裁剪代理 | "PPO 的主技巧" | min(r·A, clip(r, 1-ε, 1+ε)·A)；超过裁剪范围后梯度展平 |
| 信任区域 | "TRPO/PPO 意图" | 限制每步更新以保证单调提升 |
| KL 惩罚 | "软信任区域" | PPO 替代方案：L - β·KL，自适应 β |
| 裁剪比例 | "裁剪触发的频率" | 诊断值——应 0.1-0.3 |
| 多轮训练 | "数据重用" | 每轮 rollout 跑 K 轮梯度步 |
| 在策略-ish | "主要在策略" | PPO 名义上在策略但 K>1 安全地用了轻度离策略数据 |

---

## 📚 小结

PPO 通过裁剪重要度比率在稳定性和效率之间取得平衡。GAE 降低方差。K 轮多轮训练重用数据、提高样本效率。三个诊断指标（KL、裁剪比例、解释方差）指导调优。PPO 是 2026 年 RLHF/GRPO 的基础算法——InstructGPT、ChatGPT、Claude 的训练都用了 PPO。下一课我们进入 LLM 与 RL 的真正结合——奖励建模与 RLHF。

---

## ✏️ 练习

1. **【实现】** 在 GridWorld 上实现 PPO（ε=0.2, K=4）。对比 A2C 的样本效率（同环境步数下的最终回报）。
2. **【实验】** 尝试 K ∈ {1, 4, 10, 30}。绘制度数最终回报 vs 环境步数，跟踪平均 KL/kbold> 每轮更新。
3. **【实验】** 用自适应 KL 惩罚替代裁剪（β 在 KL>2×target 时翻倍，KL<target/2 时减半）。对比最终回报和稳定性。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| PPO 完整实现 | `code/main.py` | 裁剪代理、GAE、多轮训练、诊断指标 |
| PPO 训练提示词 | `outputs/skill-ppo-trainer.md` | 给定环境产�� PPO 训练配置的提示词 |

---

## 📖 参考资料

1. [论文] Schulman et al. "Proximal Policy Optimization Algorithms". arXiv, 2017. https://arxiv.org/abs/1707.06347
2. [论文] Schulman et al. "Trust Region Policy Optimization". ICML, 2015. https://arxiv.org/abs/1502.05477
3. [论文] Ouyang et al. "Training language models to follow instructions with human feedback". NeurIPS, 2022. https://arxiv.org/abs/2203.02155 — InstructGPT 的 PPO-in-RLHF 配方
4. [论文] Andrychowicz et al. "What Matters In On-Policy RL?" 2021. https://arxiv.org/abs/2006.05990 — 每�� PPO 超参数消融研究
5. [官方文档] OpenAI Spinning Up — PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
6. [GitHub] CleanRL PPO: https://github.com/vwxyzjn/cleanrl
7. [GitHub] Hugging Face TRL — PPOTrainer: https://huggingface.co/docs/trl/main/en/ppo_trainer

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
