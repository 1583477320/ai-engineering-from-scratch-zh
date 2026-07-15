# RLHF——奖励模型 + PPO

> SFT 教模型遵循指令，但不教模型哪个回答更好。两个语法正确、事实准确的回答可能在有用性上天差地别。RLHF 是将人类判断编码到模型行为中的方法。它让 Claude 有帮助，让 GPT 有礼貌。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 10 · 06（SFT）、阶段 09 · 08（PPO）| **时间：** ~90 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 06（SFT）— RLHF 的起点 | 阶段 10 · 08（DPO）— 2024 年后简化 RLHF 的方法

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 构建奖励模型——从人类偏好对（选择 vs 拒绝）中学习响应质量评分
- [ ] 实现 PPO 训练循环——用奖励模型和 KL 惩罚优化语言模型策略
- [ ] 解释为什么 RLHF 需要三个模型——SFT、奖励、策略——以及 KL 如何防止奖励攻击
- [ ] 评估 RLHF 的效果——对比偏好优化前后的回答质量

---

## 1. 问题

问模型"解释量子计算"，它可能生成：

**回答 A：** "量子计算利用量子比特（qubit）处于叠加态的特性，可以同时是 0、1 或两者。这使量子计算机能指数级加速某些计算。关键算法包括 Shor 的大数分解算法和 Grover 的无序数据库搜索算法。"

**回答 B：** "量子计算是一种使用量子力学现象的计算类型。它在 1980 年代首次被提出。Richard Feynman 建议量子系统可以用量子计算机模拟。IBM、Google 等公司取得了进展。Google 在 2019 年声称实现了量子优越性。"

两个回答都正确、语法通顺。但 A 明显更好——更简洁、更有信息量、结构更好。**人类每次都会选 A。**

SFT 无法捕捉这种区别——它把每个训练样本视为同样好。RLHF 解决了这个问题：训练一个奖励模型预测人类偏好，然后用这个奖励信号推动模型生成更高质量的输出。

InstructGPT（ChatGPT 的前身）用 RLHF 大幅提升了 GPT-3 的有用性、真实性和无害性。OpenAI 内部评估者更偏好 InstructGPT 输出的比例达到 85%——尽管 InstructGPT 只有 GPT-3 的 1/135 大小。

---

## 2. 概念

### 2.1 RLHF 三阶段

```
阶段 1: SFT（监督微调）
  用高质量指令-回答对微调预训练模型 → π_SFT

阶段 2: 奖励模型训练
  人类标注偏好对 (y+, y-) → 训练 R_φ(x, y) 评分

阶段 3: PPO 优化
  用奖励模型打分 + KL 惩罚 → 优化策略 π_θ
```

### 2.2 奖励模型——Bradley-Terry

奖励模型 $R_\phi(x, y)$ 给每个 (提示词, 回答) 对打一个标量分。训练使用 Bradley-Terry 偏好模型：

$$L(\phi) = -\mathbb{E}[\log \sigma(R_\phi(x, y_+) - R_\phi(x, y_-))]$$

直觉：好的回答应该得分比差的回答高。σ 是 sigmoid 函数。

### 2.3 PPO + KL 惩罚

PPO 用奖励模型的分数作为奖励信号，通过裁剪目标优化策略。关键：加 KL 惩罚——防止策略偏离 SFT 太远：

$$R_{\text{total}}(x, y) = R_\phi(x, y) - \beta \cdot KL(\pi_\theta(\cdot|x) \| \pi_{SFT}(\cdot|x))$$

β 是 KL 系数（通常 0.01-0.1）。没有 KL，模型会找到奖励模型的漏洞。

### 2.4 三个模型的角色

| 模型 | 角色 | 冻结？ |
|------|------|--------|
| π_SFT | SFT 后的基础模型 | 冻结——作为参考策略 |
| R_φ | 奖励模型——给回答打分 | 冻结——不参与 PPO 训练 |
| π_θ | 被优化的策略 | **可训练**——PPO 的目标 |

### 2.5 奖励攻击（Reward Hacking）

RLHF 最大的敌人：模型找到给奖励模型高分但人类认为很差的策略。症状：奖励持续上升但人类评估停滞。修复：增大 β、使用更大的奖励模型、提前停止。

---

## 3. 从零实现

### Step 1：奖励模型训练

```python
def train_reward_model(model, preference_data, epochs=2, lr=1e-5):
    """
    训练奖励模型。
    preference_data: [(x, y_chosen, y_rejected), ...]
    """
    for epoch in range(epochs):
        for x, y_chosen, y_rejected in preference_data:
            # 奖励分数
            r_chosen = model.score(x, y_chosen)
            r_rejected = model.score(x, y_rejected)

            # Bradley-Terry 损失
            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### Step 2：PPO + KL 惩罚

```python
def rlhf_step(policy, ref_policy, reward_model, prompt,
               beta=0.05, lr=1e-5):
    """一步 RLHF 更新。"""
    # 1. 生成回答
    response = policy.generate(prompt)

    # 2. 计算奖励 + KL 惩罚
    reward = reward_model.score(prompt, response)

    # 3. 计算 KL 散度
    kl = compute_kl(policy.log_prob(prompt, response),
                    ref_policy.log_prob(prompt, response))

    # 4. 总奖励 = 奖励 - KL 惩罚
    total_reward = reward - beta * kl

    # 5. PPO 裁剪更新
    ppo_update(policy, prompt, response, total_reward)
```

### Step 3：KL 散度计算

```python
def compute_kl(log_prob_policy, log_prob_ref):
    """计算 KL(π_θ || π_ref) 的近似。"""
    ratio = torch.exp(log_prob_policy - log_prob_ref)
    kl = ratio * (log_prob_policy - log_prob_ref)
    return kl.mean()
```

---

## 4. 工具

### 4.1 HuggingFace TRL

```python
from trl import PPOTrainer, PPOConfig, RewardTrainer

# 训练奖励模型
reward_trainer = RewardTrainer(
    model=reward_model,
    tokenizer=tokenizer,
    train_dataset=preference_data,
)

# PPO + KL 惩罚
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=64,
    kl_penalty="kl",
    init_kl_coef=0.05,
    target_kl=6.0,
)
ppo_trainer = PPOTrainer(config=ppo_config, model=policy, ref_model=ref_model)
```

### 4.2 TRL PPOTrainer 关键参数

| 参数 | 说明 |
|------|------|
| init_kl_coef | KL 惩罚系数 β 的初始值 |
| target_kl | 自适应 KL 调整的目标值 |
| kl_penalty | "kl" (固定) 或 "abs" (绝对值) |
| clip_range | PPO 裁剪范围 ε |

---

## 5. LLM 视角

### 5.1 RLHF 与 Claude/GPT 的关系

ChatGPT 的核心训练流程：预训练→SFT→RLHF。InstructGPT 用 RLHF 使 GPT-3 遵循人类偏好。Claude 同样使用了 RLHF（及其变体 DPO）来确保有帮助和无害。

### 5.2 2024 年后的趋势：DPO

DPO（Direct Preference Optimization, 2023）将 RM + PPO 简化为一个监督学习损失——不需要单独的奖励模型和 PPO。2024 年后成为主流。但理解 RLHF 对理解 DPO 仍然必要。

### 5.3 使用 ChatGPT / Claude 时的直接体验

当你觉得 Claude 的回答"更贴心"或"更诚实"时，那就是 RLHF 的效果——模型在训练时被推向人类认为更好的行为方向。

---

## 6. 工程最佳实践

### 6.1 KL 系数 β 的选择

| β | 效果 |
|---|------|
| 0.01 | 强约束——模型几乎不偏离 SFT |
| 0.05 | 推荐默认——平衡对齐和多样性 |
| 0.1 | 弱约束——更多探索但有奖励攻击风险 |
| 0.5+ | 几乎无对齐——RLHF 效果很弱 |

### 6.2 奖励模型规模

奖励模型应该 ≥ SFT 模型的规模。太小的 RM 无法准确评估大模型的输出。

---

## 7. 常见错误

### 错误 1：奖励模型过小

**现象：** RM 分数与人类评估不相关。

**修复：** RM 应至少等于 SFT 模型的规模。

### 错误 2：β 太小导致奖励攻击

**现象：** 奖励持续上升但人类评估下降。

**修复：** 增大 β（如从 0.01 调到 0.1），或提前停止。

### 错误 3：忘记冻结参考策略

**现象：** KL 惩罚失效，策略飞离 SFT。

**修复：** 确保 `ref_policy` 在整个训练中保持冻结。

---

## 8. 面试考点

### Q1：为什么 RLHF 需要三个模型？（难度：⭐⭐）

**参考答案：**
(1) SFT 模型提供行为基线——知道"好回答"的格式和质量；(2) 奖励模型提供训练信号——人类偏好转化为标量奖励；(3) 策略模型是被优化的对象。三者缺一不可：没有 SFT，RLHF 从零开始太慢；没有 RM，无法量化"好"；没有 KL 惩罚，策略会崩溃。

### Q2：奖励攻击是如何发生的？（难度：⭐⭐⭐）

**参考答案：**
奖励模型只在人类标注的偏好对上训练——它对分布外输入的评分不可靠。PPO 优化的是 RM 分数，如果 RM 有漏洞（如对重复文本给高分），PPO 会找到利用这个漏洞的策略。修复：(1) 扩大 RM 训练数据的多样性；(2) 增大 KL 系数 β 约束策略偏离；(3) 使用多个 RM 做集成；(4) 定期用人类评估校准。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| RLHF | "人类反馈强化学习" | 三阶段：SFT + 奖励模型 + PPO 优化 |
| 奖励模型 (RM) | "打分器" | 从人类偏好对学习的标量评分模型 |
| Bradley-Terry | "偏好损失" | P(y+ > y-) = σ(r+ - r-)——奖励差的 sigmoid |
| KL 惩罚 | "防止跑偏" | β · KL(π_θ ‖ π_ref)——约束策略不偏离 SFT 太远 |
| 奖励攻击 | "找漏洞" | 策略找到 RM 漏洞——奖励上升但人类评估下降 |
| PPO-ptx | "带预训练的 PPO" | PPO 训练同时保留下一个词预测损失——防止遗忘 |

---

## 📚 小结

RLHF = SFT + 奖励模型 + PPO + KL 惩罚。InstructGPT/ChatGPT 的核心训练配方。KL 惩罚防止策略偏离 SFT——是最重要的旋钮。2024 年后 DPO 简化了流程：不需要 RM 和 PPO。但理解 RLHF 是理解 LLM 对齐的基础。

---

## ✏️ 练习

1. **【实现】** 从偏好对中训练一个简单的奖励模型——评估其准确率。
2. **【实验】** 实现简化版 RLHF：用你的 MiniGPT + RM + PPO 优化。对比 RLHF 前后的回答质量。
3. **【思考】** β=0.01 和 β=0.5 的 RLHF 训练结果会有什么不同？画出训练曲线。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| RM 训练 + RLHF | `code/rlhf.py` | 奖励模型 + PPO + KL 惩罚完整实现 |

---

## 📖 参考资料

1. [论文] Ouyang et al. "Training language models to follow instructions with human feedback". NeurIPS, 2022. https://arxiv.org/abs/2203.02155
2. [论文] Christiano et al. "Deep Reinforcement Learning from Human Preferences". NeurIPS, 2017. https://arxiv.org/abs/1706.03741
3. [论文] Stiennon et al. "Learning to Summarize with Human Feedback". NeurIPS, 2020. https://arxiv.org/abs/2009.01325

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
