# DPO——直接偏好优化

> RLHF 需要训练奖励模型+PPO——两个阶段，复杂且不稳定。DPO 说：直接从偏好对学习策略，不需要奖励模型。一个损失函数搞定。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 10 · 06（SFT）、07（RLHF）| **时间：** ~60 分钟
**所处阶段：** Tier 3
**关联课程：** 阶段 10 · 07（RLHF）— DPO 是 RLHF 的简化替代 | 阶段 09 · 09（RLHF 基础）— 偏好建模的理论基础

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现 DPO 损失函数——让好答案更可能、坏答案更不可能
- [ ] 对比 DPO vs RLHF——DPO 的优势（简单）和劣势（需要参考模型）
- [ ] 说明 DPO 在 2026 年成为对齐训练首选的原因

---

## 1. 问题

RLHF 需要：(1) 训练奖励模型——需要大量偏好数据、训练不稳定；(2) PPO 优化——需要三个模型同时在显存中、超参数敏感、训练慢。

DPO（Rafailov 2023）的洞察：**RLHF 的最优策略有闭式解——不需要奖励模型和 PPO。** DPO 的损失函数直接从偏好对中优化策略：

$$L_{DPO}(\theta) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

直觉：**让好答案（$y_w$）的对数概率相对于参考策略提高，坏答案（$y_l$）降低。** β 控制偏离参考策略的程度。

DPO 的核心优势：
- **简单**：一个监督学习损失，不需要 RM 和 PPO
- **稳定**：没有 PPO 的方差问题
- **快速**：训练速度比 RLHF 快 5-10 倍
- **效果相当**：在大多数对齐基准上与 RLHF 效果相当

---

## 2. 概念

### 2.1 DPO 的数学推导（简化版）

RLHF 的优化目标（带 KL 约束）：
$$\max_\pi \mathbb{E}[R(x, y)] - \beta \cdot KL(\pi \| \pi_{ref})$$

DPO 的关键洞察：这个优化问题的最优解有闭式形式——最优策略可以用参考策略和奖励函数直接表示：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{R(x,y)}{\beta}\right)$$

反过来解出 R：$R(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$

将这个表达式代入 Bradley-Terry 偏好模型，$Z(x)$ 项抵消——得到 DPO 损失。

### 2.2 DPO 训练数据格式

```
{
  "prompt": "解释量子计算",
  "chosen": "量子计算利用量子比特处于叠加态的特性...",  # 人类偏好的回答
  "rejected": "量子计算是一种使用量子力学的计算..."   # 人类不偏好的回答
}
```

### 2.3 DPO vs RLHF 对比

| 方面 | DPO | RLHF |
|------|-----|------|
| 需要的模型 | 2（策略+参考） | 3（SFT+RM+策略） |
| 奖励模型 | 不需要 | 需要 |
| PPO | 不需要 | 需要 |
| 训练稳定性 | 高（监督损失） | 中（PPO 方差） |
| 训练速度 | 快 5-10× | 慢 |
| 效果 | 相当 | 相当（略好某些场景） |
| 超参数敏感度 | 低（主要是 β） | 高（PPO 超参多） |

### 2.4 DPO 的劣势

- **分布偏移**：DPO 是离策略的——训练数据来自旧策略，但优化目标是新策略。长时间训练可能累积误差
- **不支持在线数据**：RLHF 可以在训练中持续采样新数据，DPO 通常在固定数据集上训练
- **复杂对齐**：对于多目标、长序列、安全性关键场景，PPO + RM 可能更灵活

---

## 3. 从零实现

### Step 1：DPO 损失函数

```python
def dpo_loss(policy_logps_chosen, policy_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    """
    DPO 损失。
    Args:
        policy_logps_chosen: 当前策略对好回答的 log 概率
        policy_logps_rejected: 当前策略对坏回答的 log 概率
        ref_logps_chosen: 参考策略对好回答的 log 概率
        ref_logps_rejected: 参考策略对坏回答的 log 概率
        beta: 偏离参考策略的强度
    """
    # 策略比的 log
    log_ratio_chosen = policy_logps_chosen - ref_logps_chosen
    log_ratio_rejected = policy_logps_rejected - ref_logps_rejected

    # DPO 损失
    logits = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -F.logsigmoid(logits).mean()

    return loss
```

### Step 2：DPO 训练循环

```python
def train_dpo(policy, ref_policy, dataset, beta=0.1, lr=5e-7, epochs=1):
    """
    DPO 训练——比 RLHF 简单得多。
    """
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in dataset:
            # 计算当前策略的 log 概率
            policy_logps_chosen = policy.log_prob(batch["prompt"], batch["chosen"])
            policy_logps_rejected = policy.log_prob(batch["prompt"], batch["rejected"])

            # 计算参考策略的 log 概率
            with torch.no_grad():
                ref_logps_chosen = ref_policy.log_prob(batch["prompt"], batch["chosen"])
                ref_logps_rejected = ref_policy.log_prob(batch["prompt"], batch["rejected"])

            # DPO 损失
            loss = dpo_loss(
                policy_logps_chosen, policy_logps_rejected,
                ref_logps_chosen, ref_logps_rejected,
                beta=beta,
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return policy
```

### Step 3：评估 DPO 效果

```python
def evaluate_dpo(policy, test_prompts):
    """对比 DPO 前后的回答质量。"""
    for prompt in test_prompts:
        before = policy.generate(prompt, temperature=0.8)
        # 训练后
        after = policy.generate(prompt, temperature=0.8)
        print(f"Prompt: {prompt}")
        print(f"  DPO 前: {before[:100]}...")
        print(f"  DPO 后: {after[:100]}...")
```

---

## 4. 工具

### 4.1 HuggingFace TRL DPOTrainer

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM

policy = AutoModelForCausalLM.from_pretrained("./sft-model")
ref_policy = AutoModelForCausalLM.from_pretrained("./sft-model")  # 冻结

dpo_trainer = DPOTrainer(
    model=policy,
    ref_model=ref_policy,
    train_dataset=preference_data,  # {"prompt", "chosen", "rejected"}
    args=DPOConfig(
        output_dir="./dpo-output",
        beta=0.1,
        learning_rate=5e-7,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
    ),
)
dpo_trainer.train()
```

### 4.2 OpenAI 的 DPO 实现

OpenAI 的 InstructGPT 后续版本使用了 DPO 的变体——更注重数据质量和超参数调优。

---

## 5. LLM 视角

### 5.1 DPO 为何成为 2026 年的默认选择

- **简单**：一个监督损失——不需要 RM、PPO、复杂的奖励塑形
- **稳定**：没有 PPO 的方差问题——训练曲线更平滑
- **效果好**：在对齐基准上与 RLHF 相当，甚至在某些场景更好
- **数据效率**：不需要大量偏好数据——10K 对通常足够

### 5.2 从 RLHF 到 DPO 的演进

```
RLHF (2022)：SFT + RM + PPO → 复杂但有效
  ↓
DPO (2023)：SFT + 偏好对 → 简化为监督损失
  ↓
2026 年：DPO 为主，PPO 仅在 RM 密集场景使用
```

---

## 6. 工程最佳实践

### 6.1 DPO 超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| β | 0.1 | 偏离参考策略的强度 |
| 学习率 | 5e-7 ~ 5e-6 | 比 RLHF 小 |
| epochs | 1-3 | 数据量小时多跑几轮 |
| batch_size | 4-8 | 受显存限制 |

### 6.2 踩坑经验

- **β 太大**：策略几乎不更新——DPO 等于不训练
- **β 太小**：过拟合偏好数据——生成质量下降
- **偏好数据噪声**：~30% 人类标签有噪声——使用共识过滤

---

## 7. 常见错误

### 错误 1：忘记参考策略

**现象：** DPO 训练后模型语言质量崩溃。

**原因：** 参考策略是 KL 正则化的来源——没有它就是无约束优化。

### 错误 2：chosen/rejected 搞反

**现象：** 模型越训越差。

**原因：** DPO 损失函数的方向搞反——好答案的 log 概率应该提高。

---

## 8. 面试考点

### Q1：DPO 和 RLHF 的核心区别是什么？（难度：⭐⭐）

**参考答案：**
RLHF 需要训练奖励模型 + PPO 优化——两阶段，复杂且不稳定。DPO 将 RLHF 坍缩为一个监督学习损失——直接从偏好对优化策略，不需要 RM 和 PPO。DPO 的数学洞察：RLHF 的最优策略有闭式解，代入 Bradley-Terry 偏好模型后 RM 项抵消，得到一个纯偏好损失。

### Q2：DPO 有什么劣势？（难度：⭐⭐⭐）

**参考答案：**
(1) **分布偏移**：DPO 是离策略的——训练数据来自旧策略，但目标是新策略。长时间训练可能累积误差；(2) **不支持在线数据**：RLHF 可以在训练中采样新数据，DPO 通常在固定数据集上训练；(3) **复杂场景灵活性不足**：对于多目标、安全性关键场景，PPO + RM 可以更精细地控制（如使用多个 RM）。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| DPO | "不需要 RM 的 RLHF" | 直接偏好优化——闭式坍缩 RLHF+KL 为一个监督损失 |
| 偏好对 | "好 vs 差的回答" | 人类标注的 (chosen, rejected) 数据对 |
| 参考策略 | "防止跑太远" | DPO 中的 KL 正则化来源——冻结的 SFT 模型 |
| β 参数 | "偏离强度" | 控制策略偏离参考的程度——类似 RLHF 的 KL 系数 |

---

## 📚 小结

DPO 直接从偏好对优化策略——不需要奖励模型和 PPO。一个监督损失搞定。简单、稳定、效果相当。2026 年 DPO 已成为对齐训练的默认选择——取代了复杂的 RLHF 管道。下一课我们看一个不需要人类标注的方法——宪法 AI。

---

## ✏️ 练习

1. **【实现】** 在偏好数据集上实现 DPO 损失——对比不同 β 值的训练效果
2. **【实验】** 对比 DPO vs RLHF 在相同数据上的收敛速度和最终质量
3. **【思考】** DPO 为什么不支持在线数据？在什么场景下 RLHF 仍然更好？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| DPO 完整实现 | `code/dpo.py` | DPO 损失 + 训练循环 |

---

## 📖 参考资料

1. [论文] Rafailov et al. "Direct Preference Optimization". NeurIPS, 2023. https://arxiv.org/abs/2305.18290
2. [论文] Ethayarajh et al. "KTO: Model Alignment as Prospect Theoretic Optimization". arXiv, 2024. https://arxiv.org/abs/2402.01306

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
