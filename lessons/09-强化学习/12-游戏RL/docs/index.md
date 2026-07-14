# 游戏中的 RL——AlphaZero、MuZero 与 LLM 推理时代

> 1992 年 TD-Gammon 用纯 TD 在西洋双陆棋上击败人类冠军。2016 年 AlphaGo 击败李世石。2017 年 AlphaZero 从零自学统治国际象棋、将棋和围棋。2024 年 DeepSeek-R1 证明同样的配方——GRPO 替代 PPO——在推理任务上有效。游戏是推动本阶段每个突破的基准。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 09 · 05（DQN）、08（PPO）、09（RLHF）、10（多智能体）| **时间：** ~120 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 09 · 09（RLHF）— GRPO 与 RLHF 的关系 | 阶段 10 · 08（DPO）— 推理 LLM 的对齐

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 用统一视角解释 AlphaZero、MuZero、GRPO 的共同循环
- [ ] 描述 AlphaZero 的三大组件——策略价值网络、MCTS、自对弈
- [ ] 说明 MuZero 如何将 AlphaZero 扩展到未知规则的环境
- [ ] 解释 GRPO 如何在 LLM 推理中实现"游戏 RL"
- [ ] 描述 DeepSeek-R1 的四阶段训练流程

---

## 1. 问题

游戏拥有一切 RL 想要的特性。清晰的奖励（赢/输）。无限回合（自对弈重置）。完美模拟（游戏本身就是模拟器）。离散或小连续动作空间。迫使对抗鲁棒性的多智能体结构。

游戏也是每个主要 RL 突破的测试场。TD-Gammon（1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（2019）。AlphaStar（2019）。MuZero（2019）。DeepSeek-R1（2025）。

这个结课单元通过单一统一视角调查三个里程碑架构——AlphaZero、MuZero、GRPO：**自对弈 + 搜索 + 策略改进**。每个都推广了前一个；GRPO 特别是 AlphaZero 配方应用于 LLM 推理——词元是动作，数学验证是胜利信号。

---

## 2. 概念

### 2.1 统一循环

```
while True:
    轨迹 = 自对弈(当前策略, 搜索)           # 和自己下棋
    策略目标 = 搜索.改进策略(轨迹)            # 搜索改进原始策略
    策略网络.更新(策略目标, 价值目标)           # 在搜索输出上监督学习
```

### 2.2 AlphaZero（2017）

给定一个已知规则的游戏（国际象棋、将棋、围棋）：

| 组件 | 功能 |
|------|------|
| 策略-价值网络 f_θ(s) | 输出先验概率 p 和预期价值 v |
| MCTS 搜索 | 用 (p,v) 作为先验和自举构建搜索树 |
| 自对弈 | 两个策略副本对弈——MCTS 访问分布 π_t 作为训练目标 |
| 损失 | L = (v - z)² - π·log p + c·‖θ‖² |

**零人类知识。零手工启发式。** 一个配方在数千万局自对弈后统治了国际象棋、将棋和围棋。

### 2.3 MuZero（2019）

移除了"规则必须已知"的要求。学习一个潜在动态模型：

- h(s)：将观测编码为潜在状态
- g(s_latent, a)：预测下一个潜在状态 + 奖励
- f(s_latent)：预测策略先验 + 价值

MCTS 在**学到的潜在空间**中运行。同样的搜索，同样的训练循环。适用于围棋、国际象棋**和**Atari——一个算法，不需要规则知识。

### 2.4 GRPO——LLM 推理中的 AlphaZero

同样的 AlphaZero 形状循环，应用于语言模型推理：

- **游戏**：回答数学/编码/推理问题。"胜利" = 验证器返回 1
- **策略**：LLM。**动作**：词元。**状态**：提示 + 已生成部分
- **无评论家**：采样 G 个补全，计算组内相对优势 Aᵢ = (rᵢ - mean_r) / std_r
- **KL 惩罚**：防漂移到参考策略

$$L_{GRPO}(\theta) = -\mathbb{E}\left[\frac{1}{G} \sum_i A_i \cdot \log \pi_\theta(o_i|q)\right] + \beta \cdot KL(\pi_\theta || \pi_{ref})$$

无奖励模型、无评论家、无 MCTS。组内基线替代了所有三个。

### 2.5 DeepSeek-R1 配方

两个模型在一篇论文中：

**R1-Zero（从零学习推理）：**
- 从 DeepSeek-V3 基座模型开始。无 SFT。
- 直接用 GRPO + 两个奖励：准确率奖励（答案正确？）和格式奖励（有 <think> 标签？）
- 数千步后，平均回复长度从 ~100 增长到 ~10,000 词元，数学分数攀升到接近 o1-preview 水平

**R1（可读版）：**
1. 冷启动 SFT：收集数千个长 CoT 示范
2. 推理导向 GRPO：准确率 + 格式 + 语言一致性奖励
3. 拒绝采样 + SFT 第 2 轮：保留正确且可读的推理轨迹
4. 全谱 GRPO：推理 + 一般对齐

---

## 3. 从零实现

### GRPO 多臂赌博机

```python
import random, math

def softmax(scores):
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    return [e / sum(exps) for e in exps]

def grpo_step(theta, prompt_idx, G=8, beta=0.01, lr=0.1):
    """GRPO 更新——组内相对优势 + KL 惩罚。"""
    probs = softmax(theta[prompt_idx])
    # 采样 G 个回答
    samples = [random.choices(range(len(probs)), weights=probs)[0] for _ in range(G)]
    # 验证器评分
    rewards = [1.0 if s == correct_answer[prompt_idx] else 0.0 for s in samples]
    # 组内相对优势
    mean_r = sum(rewards) / G
    std_r = (sum((r - mean_r)**2 for r in rewards) / G + 1e-8) ** 0.5
    advantages = [(r - mean_r) / std_r for r in rewards]

    # 策略梯度 + KL
    for a, A in zip(samples, advantages):
        grad = [0.0] * len(probs)
        grad[a] = 1.0
        for i in range(len(probs)):
            theta[prompt_idx][i] += lr * A * (grad[i] - probs[i])

    # KL 惩罚到参考策略
    for i in range(len(probs)):
        theta[prompt_idx][i] -= beta * (theta[prompt_idx][i] - reference[prompt_idx][i])

    return theta
```

---

## 4. 工具

### 4.1 Hugging Face TRL 的 GRPOTrainer

```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def accuracy_reward(completions, prompts):
    """规则验证器——检查数学答案是否正确。"""
    # 解析每个 completion 中的最终答案
    # 与 ground truth 比较
    return [1.0 if correct else 0.0 for correct in answers]

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[accuracy_reward],
    args=GRPOConfig(
        per_device_train_batch_size=4,
        num_generations=8,           # G=8 个采样
        beta=0.01,                   # KL 惩罚
        max_completion_length=2048,
    ),
    train_dataset=math_dataset,
)
trainer.train()
```

### 4.2 Open-SZero / Leela Zero

```python
# AlphaZero 的开源复现
# https://github.com/LeelaChessZero/lc0
```

---

## 5. LLM 视角

### 5.1 AlphaZero → GRPO 的对应

| AlphaZero | GRPO (LLM 推理) |
|-----------|----------------|
| 棋盘状态 | 提词 + 已生成词元 |
| 棋盘动作 | 下一个词元 |
| MCTS 搜索 | 组内采样（G 个补全） |
| MCTS 访问分布 π | 组内相对优势 A |
| 游戏结果 z | 验证器奖励 r |
| 价值网络 V(s) | 无评论家（用组均值替代） |

### 5.2 2026 年推理 LLM 的配方

GRPO 是 DeepSeek-R1 的核心训练算法：

- **验证器**：数学 = 数值答案匹配；代码 = 测试用例通过
- **奖励**：准确率 + 格式 + 语言一致性
- **KL 惩罚**：防止模型忘记语言能力

GRPO 之所以适合推理任务：(1) 无需评论家（推理奖励是稀疏的——只有最终答案有分）；(2) 组内相对归一化使不同难度的问题之间优势可比；(3) 推理过程中的 token 选择本质上是"游戏动作"。

### 5.3 使用 ChatGPT / Claude 时的直接体验

当你让 Claude "一步步思考"数学题时，它在做 GRPO 训练过的同一类事情：生成多个推理路径，验证器（或人类）判断哪个正确，策略向正确路径偏移。G=8 的组内采样 = "尝试 8 种解法，选最好的"。

---

## 6. 工程最佳实践

### 6.1 GRPO 关键超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| G（组大小） | 8-64 | G=8 是最小值，G=32 是标准 |
| β（KL 惩罚） | 0.01-0.05 | 防遗忘 |
| max_completion_length | 2048-8192 | 推理需要长回复 |
| 学习率 | 1e-6~1e-5 | 小学习率防破坏性更新 |

### 6.2 验证器设计

| 任务 | 验证器 | 注意事项 |
|------|--------|---------|
| 数学 | 最终答案匹配 | 答案格式标准化 |
| 代码 | 单元测试通过率 | 测试用例覆盖边缘情况 |
| 推理 | 过程奖励模型（PRM） | 每步打分，更精细 |

### 6.3 踩坑经验

- **奖励攻击**：LLM 找到验证器漏洞（如格式奖励但答案不对）→ 多个测试用例
- **组大小太小**：G=2 时方差极大——G=8 是最低值
- **长度偏差**：不同长度回复的 log-prob 不同——按词元数归一化

---

## 7. 常见错误

### 错误 1：验证器覆盖不足

**现象：** LLM 学到通过测试用例但解题思路错误。

**修复：** 设计多个边缘情况测试用例。

### 错误 2：GRPO 中 G 太小

**现象：** 优势估计噪声极大，策略不稳定。

```python
# ❌ G=2
rewards = [verify(p, s) for s in [sample(p) for _ in range(2)]]
# ✓ G=8 或更大
rewards = [verify(p, s) for s in [sample(p) for _ in range(8)]]
```

---

## 8. 面试考点

### Q1：AlphaZero 和 GRPO 的共同循环是什么？（难度：⭐⭐）

**参考答案：**
两者都遵循"自对弈 + 搜索/采样 + 策略改进"的循环。AlphaZero 用 MCTS 搜索改进策略，然后在搜索输出上监督学习。GRPO 用组内采样（G 个补全）替代 MCTS——用组内相对优势替代 MCTS 访问分布。本质上，GRPO 是 AlphaZero 的"无搜索"简化版——采样替代了树搜索。

### Q2：DeepSeek-R1 为什么用 GRPO 而不是 PPO？（难度：⭐⭐⭐）

**参考答案：**
三个原因（DeepSeekMath 2024）：(1) 无需评论家网络——节省一半显存；(2) 组内基线天然处理推理任务的稀疏轨迹奖励——只有最终答案有分；(3) 每提示归一化使不同难度问题的优势可比——PPO 的单一评论家无法处理难度差异巨大的数学题。

### Q3：MuZero 和 AlphaZero 的关键区别是什么？（难度：⭐⭐⭐）

**参考答案：**
AlphaZero 需要已知的环境规则（可以精确模拟下一步状态）。MuZero 学习了一个潜在动态模型——h 编码观测、g 预测下一步、f 预测价值和策略——MCTS 在这个学到的模型中搜索。这使 MuZero 可以处理未知规则的环境（如 Atari），因为不需要知道"规则"——只需要知道"从这个状态做了这个动作，下一个状态大概长什么样"。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| MCTS | "带神经网络的树搜索" | 蒙特卡洛树搜索；用学到的 (p,v) 做 PUCT 选择 |
| AlphaZero | "自对弈 + MCTS" | 策略-价值网络匹配 MCTS 访问和游戏结果 |
| MuZero | "有模型的 AlphaZero" | 同一循环但在潜在空间中，通过学习到的动态模型运行 |
| GRPO | "无评论家的 PPO" | 组内相对策略优化；REINFORCE + 组均值基线 + KL |
| PUCT | "AlphaZero 的 UCB" | Q + c·p·√N/(1+N_a)——平衡价值估计和先验 |
| 自对弈 | "代理 vs 过去的自己" | 零和博弈的标准训练方式 |
| 验证器奖励 | "可验证的 RL" | 奖励来自确定性检查器（测试通过、答案匹配） |
| 过程奖励模型 | "PRM" | 评估部分推理步骤，而非仅最终答案 |

---

## 📚 小结

AlphaZero → MuZero → GRPO 是同一循环的逐步推广：AlphaZero 需要已知规则 + MCTS，MuZero 用学习模型替代已知规则，GRPO 用组内采样替代 MCTS 且应用于文本。2026 年，GRPO 是推理 LLM 的标准训练算法——DeepSeek-R1 证明了"游戏 RL 配方在文本上有效"。这个自对弈 + 搜索 + 蒸馏的配方跨越了文本、像素和物理控制。

---

## ✏️ 练习

1. **【实现】** 在多臂赌博机上实现 GRPO（2 个提示 × 4 个回答）。用 G=8 训练，应在 1000 步内收敛。
2. **【实验】** 对比 GRPO vs PPO vs 纯 REINFORCE 在同一赌博机上的样本效率。
3. **【思考】** 如果将 GRPO 扩展到多步推理链（生成 2 个词元后验证），组内优势应该如何分配到每个词元位置？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| GRPO 多臂赌博机 | `code/main.py` | 组内相对优势 + KL 惩罚实现 |

---

## 📖 参考资料

1. [论文] Silver et al. "Mastering the game of Go without human knowledge". Nature, 2017.
2. [论文] Silver et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go". Science, 2018.
3. [论文] Schrittwieser et al. "Mastering Atari, Go, chess and shogi by planning with a learned model". Nature, 2020.
4. [论文] DeepSeek-AI. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning". arXiv, 2024.
5. [论文] DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL". arXiv, 2025.
6. [GitHub] Hugging Face TRL — GRPOTrainer: https://huggingface.co/docs/trl/main/en/grpo_trainer
7. [GitHub] QwenLM/Qwen2.5-Math: https://github.com/QwenLM/Qwen2.5-Math

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
