# PPO——近端策略优化

> PPO 是 2026 年 RLHF/GRPO 的基础。它的 Clipped Surrogate Objective 在稳定性和效率之间取得了最佳平衡——也是 InstructGPT、DPO 的训练配方。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 09 · 07（A2C/A3C）| **时间：** ~90 分钟

---

## 🎯 学习目标

- [ ] 理解 PPO 的 Clipped Surrogate Objective——为什么要裁剪策略更新的幅度
- [ ] 实现 PPO 的训练循环——收集轨迹、计算优势、裁剪目标、更新策略
- [ ] 说明 PPO 在 RLHF/GRPO 中的角色——作为策略优化的基础算法

---

## 1. 问题

A2C 好在同步稳定，但策略更新可能太大——导致灾难性遗忘。PPO 的答案：**裁剪策略比率**——限制每步更新的幅度，保证训练稳定。

### PPO 的核心公式

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

- $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$：新旧策略的比率
- $\hat{A}_t$：优势函数估计（GAE）
- $\epsilon = 0.2$：裁剪范围——当策略变化过大时，梯度被截断

### 为什么裁剪

```
不裁剪:  策略可能一次更新太大 → 训练崩溃
裁剪后:  策略更新被限制在 (1-ε, 1+ε) 范围内 → 稳定
```

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| Clipped Surrogate Objective | 裁剪策略比率——限制每步更新幅度 |
| 优势函数 GAE | 广义优势估计——结合多步 TD 回报，降低方差 |
| PPO-Clip | 用裁剪替代 KL 约束——实现更简单，效果相当 |
| PPO 在 RLHF 中 | 用于优化 LLM——从人类反馈中学习偏好 |

---

## 📚 小结

PPO 通过裁剪策略比率在稳定性和效率之间取得平衡。GAE（广义优势估计）降低方差。PPO-Clip 是 2026 年 RLHF/DPO/GRPO 的基础算法——InstructGPT、ChatGPT、Claude 的训练都用了 PPO。

---

## ✏️ 练习

1. 在连续控制任务（Pendulum）上实现 PPO——对比 PPO vs A2C 的稳定性和最终性能
2. 改变 ε（0.1, 0.2, 0.5）观察对训练稳定性的影响

---

## 📖 参考资料

1. [论文] Schulman et al. "Proximal Policy Optimization Algorithms". 2017.
2. [博客] Lilian Weng. "Proximal Policy Optimization Algorithms". 2018.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
