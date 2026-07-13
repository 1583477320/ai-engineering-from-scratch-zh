# 策略梯度 REINFORCE

> 直接优化策略 π(a|s) 的参数——不需要学习价值函数。REINFORCE 是最简单的策略梯度方法：采样一条轨迹，用回报作为权重更新策略。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 09 · 01（MDP）、03（蒙特卡洛）| **时间：** ~75 分钟

---

## 🎯 学习目标

- [ ] 推导策略梯度定理——∇J(θ) = E[G_t · ∇log π_θ(a_t|s_t)]
- [ ] 实现 REINFORCE——采样轨迹、计算回报、更新策略参数
- [ ] 解释基线（baseline）如何降低方差——从回报中减去平均回报

---

## 1. 问题

Q 学习通过学习 Q(s,a) 来间接改进策略。策略梯度说：**直接优化策略 π_θ(a|s) 的参数 θ。**

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[\sum_t G_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

**直觉：** 奖励高的动作，增加其概率；奖励低的动作，降低其概率。G_t 是从该步开始的折扣回报。

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| 策略梯度 | 直接优化策略参数 θ——不需要价值函数 |
| REINFORCE | 用完整轨迹的经验平均估计策略梯度 |
| 基线 | 从回报中减去平均回报——降低方差但不改变期望 |

---

## 📚 小结

REINFORCE 是策略梯度的基础——采样轨迹、计算回报、更新策略。加入基线（减去平均回报）降低方差。但 REINFORCE 有高方差——下一步：Actor-Critic 结合策略梯度和价值函数。

---

## ✏️ 练习

1. 在 CartPole 上实现 REINFORCE——对比有/无基线的收敛速度
2. 绘制每轮回报的方差——基线降低了方差多少？

---

## 📖 参考资料

1. [论文] Williams. "Simple statistical gradient-following algorithms for connectionist reinforcement learning". 1992.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
