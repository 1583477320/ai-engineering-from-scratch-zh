# 游戏中的 RL

> 从 Atari 到 AlphaGo，从 Dota 到 StarCraft——游戏一直是 RL 最耀眼的试验场。它提供了完美的评估标准：赢或输，分数就是你的训练信号。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 09 · 05（DQN）、08（PPO）| **时间：** ~60 分钟

---

## 🎯 学习目标

- [ ] 说明 RL 在游戏中的三种范式——单代理、多代理、世界模型
- [ ] 理解 AlphaGo/AlphaStar 的核心技术——蒙特卡洛树搜索 + PPO
- [ ] 说明 2026 年游戏 AI 的前沿——从"达到人类水平"到"创造新策略"

---

## 1. 问题

游戏为什么是 RL 的理想试验场？三个原因：(1) **明确的奖励信号**——赢=+1，输=-1，分数直接可用；(2) **可以无限并行**——在模拟中快速收集经验；(3) **可以自动评估**——不需要人工判断"这个翻译好不好"。

### RL 在游戏中的三波浪潮

| 时代 | 代表 | 突破 |
|---|---|---|
| 2013-2015 | DQN（Atari） | 从像素直接学习——不需要特征工程 |
| 2016-2019 | AlphaGo/AlphaStar | MCTS + PLO+ 人类水平 |
| 2020-2026 | OpenAI Five, AlphaStar | 多智能体 + 世界模型 → 超人类水平 |

### 核心架构演进

**Atari（2013）：** DQN + 帧堆叠。从像素直接学习 Q 值。单一游戏，单一策略。

**AlphaGo（2016）：** MCTS（蒙特卡洛树搜索）+ PPO。监督学习预训练 + RL 精调。用人类棋谱初始化——再用自我对弈超越人类。

**AlphaStar（2019）：** PPO + 多智能体 + 课程学习。星际争霸的完整游戏操作空间。

**OpenAI Five（2019）：** PPO + LSTM + 自我对弈 + 大规模并行。Dota 2 上达到职业水平。

---

## 🔑 关键术语

| 术语 | 含义 |
|---|---|
| MCTS | 蒙特卡洛树搜索——结合树搜索和随机模拟来规划动作 |
| 自我对弈 | 用当前策略的旧版本和新版本对抗训练 |
| 课程学习 | 先在简单场景训练，逐步增加难度 |
| 世界模型 | 学习环境的内部表示——用想象进行规划 |

---

## 📚 小结

游戏是 RL 的完美试验场——明确奖励、无限并行、自动评估。DQN（2013）→AlphaGo（2016）→AlphaStar（2019）→2026 年的前沿已经超越了人类水平——不再是"达到人类水平"，而是"发现新策略"。

---

## ✏️ 练习

1. 在 Connect4 上实现 PPO——观察 AI 如何发现人类不知道的策略
2. 实现自我对弈训练——对比"自己和旧版本对弈"vs"自己和随机策略对弈"的训练效率

---

## 📖 参考资料

1. [论文] Silver et al. "Mastering the game of Go with deep neural networks and tree search" (AlphaGo). 2016.
2. [论文] Vinyals et al. "Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning" (AlphaStar). 2019.
3. [项目] OpenAI Five. https://openai.com/research/openai-five — Dota 2 多智能体 RL

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系。
