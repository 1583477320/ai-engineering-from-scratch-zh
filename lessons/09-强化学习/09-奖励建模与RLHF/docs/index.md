# 奖励建模与 RLHF

> 人类写不出"好回答"的公式——但可以比较两个回答挑出更好的。用偏好训练奖励模型，再用 PPO 优化语言模型。Christiano 2017，InstructGPT 2022。把 GPT-3 变成 ChatGPT 的配方。2026 年大部分被 DPO 取代，但心智模型仍在。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 09 · 08（PPO）、阶段 10 · 07（大语言模型）| **时间：** ~60 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 09 · 08（PPO）— RLHF 中的策略优化算法 | 阶段 10 · 08（DPO）— 2026 年更优的替代方案

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 RLHF 的三阶段流水线——SFT → 奖励模型 → PPO 微调
- [ ] 实现 Bradley-Terry 奖励模型的训练
- [ ] 说明 KL 惩罚在 RLHF 中的作用——防止奖励攻击
- [ ] 对比 DPO 和 RLHF 的优劣势
- [ ] 诊断奖励攻击（reward hacking）并采取修复措施

---

## 1. 问题

你训练了一个语言模型做下一个词预测。它写出语法正确的英语。它也撒谎、跑题、该拒绝时不拒绝。你不能用更多预训练修复——网络文本本身就是问题。

你想要一个**标量奖励**说"回答 A 比回答 B 更好"。手写那个奖励函数不可能。但人类可以比较两个输出并标记偏好——大规模收集很便宜。

RLHF（Christiano 2017；Ouyang 2022）将偏好转换为奖励模型，然后用 PPO 优化 LLM。三步：SFT → RM → PPO。这是 2023-2025 年 ChatGPT、Claude、Gemini 及所有对齐 LLM 的配方。

2026 年 PPO 步骤大多被 DPO 取代——更便宜且对齐效果几乎一样好。但**奖励模型**仍然支撑每个 Best-of-N 采样器、每个来自可验证奖励的 RL 管道、每个使用过程奖励模型的推理模型。

---

## 2. 概念

### 2.1 RLHF 三阶段

```
阶段 1: SFT（监督微调）
  从预训练基础模型开始，用人工示范微调 → π_SFT

阶段 2: 奖励模型训练
  人类标记偏好对 (y+, y-) → 训练 R_φ(x, y) 打分

阶段 3: PPO + KL 惩罚优化
  π_θ 从 π_SFT 初始化 → 用 PPO + KL 到 π_ref 优化
  R_total = R_φ(x, y) - β·KL(π_θ || π_ref)
```

### 2.2 Bradley-Terry 奖励模型

对偏好对 (y+, y-)，奖励模型最小化：

$$L(\phi) = -\mathbb{E}[\log \sigma(R_\phi(x, y_+) - R_\phi(x, y_-))]$$

σ 是 sigmoid。奖励的差意味着偏好概率。Bradley-Terry 自 1952 年以来就是标准。

**为什么 KL 惩罚至关重要？** 没有 KL，PPO 会找到奖励攻击策略——RM 只在分布内回复上训练。分布外的回复可能得分高于任何人类写的内容。KL 将 π_θ 保持在 RM 训练的流形附近。

### 2.3 DPO——2026 年的替代

DPO（Rafailov 2023）：直接从偏好对中优化——不需要奖励模型、不需要 PPO。

$$L_{DPO}(\theta) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

**让好答案更可能，坏答案更不可能——用参考策略锚定。**

### 2.4 2026 年的生产配方

| 场景 | 方法 | 说明 |
|------|------|------|
| 对齐/无害性 | DPO | 更便宜、更简单 |
| 推理（数学/代码） | GRPO + 验证器 | 奖励来自测试用例/数值答案 |
| 安全/拒绝 | RLHF-PPO + 安全 RM | 需要精细的奖励模型 |
| Best-of-N | RM + 采样 | 推理时用 RM 打分，不需训练 |

---

## 3. 从零实现

### Step 1：Bradley-Terry 奖励模型

```python
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))

def rm_train_step(w, y_pos, y_neg, lr=0.01):
    """奖励模型训练一步。"""
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

### Step 2：PPO + KL 惩罚（简化）

```python
def rlhf_step(theta, ref, w, prompt, beta=0.1, eps=0.2):
    probs = softmax(policy_probs(theta, prompt))
    token = sample(probs)
    rm_score = dot(w, bag([token]))
    # KL 惩罚
    probs_ref = softmax(policy_probs(ref, prompt))
    kl = sum(p * math.log(p / pr + 1e-12)
             for p, pr in zip(probs, probs_ref) if p > 0)
    total_reward = rm_score - beta * kl
    # PPO 裁剪更新...
```

---

## 4. 工具

### 4.1 Hugging Face TRL

```python
# Stage 2: 奖励模型
from trl import RewardTrainer, RewardConfig
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)
trainer = RewardTrainer(model=rm, train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5))

# Stage 3: PPO + KL
from trl import PPOTrainer, PPOConfig
ppo = PPOTrainer(config=PPOConfig(learning_rate=1.41e-5, batch_size=64,
    init_kl_coef=0.05, target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok)
```

### 4.2 DPO

```python
from trl import DPOTrainer
dpo = DPOTrainer(model=policy, ref_model=ref,
    train_dataset=preference_data, beta=0.1)
dpo.train()  # 一步完成！不需要 RM 和 PPO
```

---

## 5. LLM 视角

### 5.1 RLHF 与每个 LLM 的训练

| 模型 | 对齐方法 | 备注 |
|------|---------|------|
| ChatGPT | PPO + RM (InstructGPT) | 首个大规模 RLHF |
| Claude | RLAIF (AI 反馈替代人类) | 自我批评循环 |
| Llama 3.1 | DPO + PPO | Meta 混合方案 |
| DeepSeek-R1 | GRPO + 验证器 | 推理导向 |

### 5.2 KL 惩罚的意义

KL 惩罚 = "别跑太远"。RL 训练的每个 token，计算策略相对于 SFT 模型的 KL 散度。KL 太大 → 模型在说"奇怪的话"；KL 太小 → RL 无效。β 通常 0.01-0.05。

### 5.3 奖励攻击——RLHF 最大的敌人

奖励攻击 = LLM 找到给 RM 打高分但人类认为很烂的策略。症状：奖励持续上升但人类评估停滞。修复：提前停止、增加 β、扩大 RM 训练数据、使用过程奖励模型。

---

## 6. 工程最佳实践

### 6.1 KL 策略

| 策略 | 适用场景 |
|------|---------|
| 固定 β | 简单任务 |
| 自适应 β（推荐） | 生产环境 |
| 目标 KL | InstructGPT |

### 6.2 2026 年推荐流程

```
小规模对齐 → DPO（简单、便宜）
大规模推理 → GRPO + 验证器（无需 RM）
安全关键 → RLHF-PPO + 安全 RM（精细控制）
```

---

## 7. 常见错误

### 错误 1：忘记 KL 惩罚

```python
# ❌ 奖励飙升但质量崩溃
reward = rm_score
# ✓ 加 KL 惩罚
reward = rm_score - beta * kl(π_θ || π_ref)
```

### 错误 2：RM 比策略小

**现象：** RM 分数与人类评估不相关。RM 至少要和策略一样大。

### 错误 3：DPO 中忘记参考策略

**现象：** DPO 训练后语言质量崩溃。参考策略 π_ref 是 KL 正则化的来源。

---

## 8. 面试考点

### Q1：RLHF 三阶段各有什么作用？（难度：⭐⭐）

**参考答案：**
SFT：建立基线能力和格式。RM：学习人类偏好（比精确打分更容易收集偏好对）。PPO+KL：优化策略生成 RM 认为好的回答，同时不偏离 SFT 太远。

### Q2：KL 惩罚为什么是最重要的旋钮？（难度：⭐⭐⭐）

**参考答案：**
RM 只在分布内训练——分布外输出可能得高分。没有 KL 约束，PPO 找到 RM 漏洞（奖励攻击）。KL(π_θ || π_ref) 确保策略在 RM 可靠的流形内。β 太小 → 奖励攻击；β 太大 → RL 无效。

### Q3：DPO 如何绕过奖励模型？（难度：⭐⭐⭐）

**参考答案：**
DPO 的数学洞察：Bradley-Terry RM 的最优解有闭式形式——最优策略可以直接从偏好对中计算，不需要显式 RM。损失 L_{DPO} = -E[log σ(β·log(π_θ(y_w)/π_ref(y_w)) - β·log(π_θ(y_l)/π_ref(y_l)))] 与 RLHF+KL 完全等价，只是用策略比替代了 RM 分数。优势：一步监督训练、无 PPO 方差、计算成本低 10 倍。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| RLHF | "对齐 RL" | 三阶段 SFT + RM + PPO（Christiano 2017, Ouyang 2022） |
| 奖励模型 | "打分网络" | 从偏好对学习的标量函数，Bradley-Terry 目标 |
| KL 惩罚 | "靠近参考" | β·KL(π_θ \|\| π_ref)；防止奖励攻击的正则化器 |
| 奖励攻击 | "古德哈特定律" | 策略利用 RM 漏洞；奖励升但人类评停滞 |
| DPO | "不需要 RM 的 RLHF" | 直接偏好优化——闭式坍缩 RM+PPO |
| GRPO | "无评论家的 PPO" | 组内相对优势 + KL 惩罚 |
| PRM | "过程奖励模型" | 评估部分推理步骤，而非仅最终答案 |

---

## 📚 小结

RLHF = SFT + 奖励模型 + PPO + KL 惩罚。KL 是防止奖励攻击的关键。DPO 直接从偏好对学习——更简单。GRPO 用组内相对优势替代评论家。2026 年：DPO 为主，PPO 用于安全关键场景。

---

## ✏️ 练习

1. **【实现】** 训练 BT 奖励模型 500 对合成偏好。测量保留集准确率。
2. **【实验】** PPO-RLHF 中 β ∈ {0.0, 0.1, 1.0}，画出奖励 vs KL 曲线。
3. **【思考】** DPO 有什么不适用的场景？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| BT RM + KL PPO | `code/main.py` | 奖励模型 + KL 惩罚 PPO 循环 |

---

## 📖 参考资料

1. [论文] Christiano et al. "Deep RL from Human Preferences". NeurIPS, 2017.
2. [论文] Ouyang et al. "InstructGPT". NeurIPS, 2022.
3. [论文] Rafailov et al. "Direct Preference Optimization". NeurIPS, 2023.
4. [GitHub] Hugging Face TRL: https://github.com/huggingface/trl

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
