# 指令遵循作为对齐信号

> 后来对 RLHF 的每一个批评都在反对这个流水线。在研究优化压力如何扭曲代理之前，你必须先看到代理。InstructGPT（Ouyang et al., 2022）定义了参考架构：在指令-响应对上的监督微调、在成对偏好排名上训练的奖励模型、以及用 KL 惩罚到 SFT 策略的 PPO。一个 1.3B 的 InstructGPT 优于 175B 的原始 GPT-3——这个单一结果是为什么 2026 年每一个前沿实验室仍然使用 RLHF 形状的后训练流水线的原因。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 10 · 06（SFT）、阶段 10 · 07（RLHF）、阶段 10 · 08（DPO）
**预计时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 说出 InstructGPT 流水线的三个阶段和每个阶段使用的损失函数
- [ ] 解释为什么 1.3B 指令调优模型在人类偏好评估上优于原始 175B GPT-3
- [ ] 说明第三阶段的 KL 惩罚在防止什么，以及移除它为什么会崩溃到模式寻求行为
- [ ] 描述对齐税和 Ouyang et al. 使用的 PPO-ptx 缓解方案

---

## 1. 问题

预训练语言模型完成文本。它们不回答问题。问 GPT-3"写一个反转列表的 Python 函数"，你经常得到另一个提示词——因为大部分训练分布是网络文本，网络文本继续生成网络文本。模型在做它的工作——但工作本身是错的。

每一个严肃实验室用来修复这个问题的代理是**人类偏好**。两个答复交给标注员；标注员选更好的；奖励模型学习标注员的偏好。然后 RL 循环将策略推向奖励模型评分高的输出。这就是完整的 InstructGPT 论文在三个句子中的总结。

---

## 2. 概念

### 2.1 第一阶段：监督微调（SFT）

收集提示词-响应对，其中响应是好意人类会写的。Ouyang et al. 使用了来自标注员和 OpenAI API 的 13k 条提示词。在基础模型上用标准交叉熵损失微调。

SFT 给你的是：模型现在回答问题而不是继续它们。它不给的是：当多个回答都合理时，哪个回答是标注员偏好的信息。

### 2.2 第二阶段：奖励模型（RM）

对每个提示词，从 SFT 模型中采样 K 个答复。标注员排序。训练一个奖励模型，对任何提示词-响应对打分，使得当 `y_w` 优于 `y_l` 时：

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这是 Bradley-Terry 成对偏好损失。RM 通常从 SFT 模型初始化，将 LM 头替换为标量头。奖励模型很小：6B 就足够用于 175B 的 InstructGPT。它们也很脆弱——论文第 5 节大部分关于在小规模下出现的奖励黑客行为。

### 2.3 第三阶段：带 KL 惩罚的 PPO

定义目标：

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta × KL(pi(.|x) || pi_SFT(.|x))
```

用 PPO 最大化。KL 项防止 `pi` 远离 SFT 策略。没有它，优化器会找到对抗性示例——在 RM 下得分很高的字符串，因为 RM 从未见过它们，而不是因为人类真的偏好它们。

**KL 系数 `beta` 是 RLHF 最重要的超参数。** 太低：奖励黑客。太高：没有比 SFT 更好的改进。

### 2.4 对齐税

RLHF 之后，模型被人类偏好但标准基准（SQuAD、HellaSwag、DROP）上退步了。Ouyang et al. 称之为对齐税，并用 **PPO-ptx** 修复它：将预训练梯度混合到 RL 目标中，这样模型不会忘记它从未被奖励过的下游任务。

```
J_ptx(pi) = J(pi) + gamma × E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx 成了标准。Anthropic、DeepMind 和 Meta 都使用变体。

### 2.5 结果

1.3B 的 InstructGPT（SFT + RM + PPO-ptx）在标注员中约 70% 的情况下优于 175B 的基础 GPT-3。在来自生产流量的隐藏测试提示词上差距扩大。从这一数字可以读到两个东西：

1. **对齐是一个不同于能力的维度。** 175B 模型有更多能力；1.3B 模型有更多对齐；标注员偏好对齐的那个
2. **能力地板由基础模型设定。** 你不能将基础模型 RLHF 到它从未见过的事实中

### 2.6 为什么这是第 18 章的参考点

后来每一课的批评——奖励黑客（第 02 课）、DPO（第 03 课）、谄媚（第 04 课）、宪法 AI（第 05 课）、潜伏智能体（第 07 课）、对齐伪造（第 09 课）——都在反对这个流水线的某个部分。奖励黑客攻击第二阶段。DPO 合并了第二和第三阶段。宪法 AI 替代了人类标注员。对齐伪造显示策略可以完全绕过第三阶段。

---

## 3. 从零实现

```python
import random


def simulate_instructgpt(n_prompts=200, n_preference_pairs=500):
    """简化版 InstructGPT 三阶段模拟。"""

    # 基础"策略"：三个动作上的偏置硬币
    actions = ["A", "B", "C"]
    policy = {"A": 1/3, "B": 1/3, "C": 1/3}

    # 标注员偏好（真实奖励）
    true_reward = {"A": 0.8, "B": 0.5, "C": 0.2}

    # 第一阶段：SFT——模拟标注员在提示词上的行动
    sft_policy = dict(policy)

    # 第二阶段：训练一个简单 RM（简化版 Bradley-Terry）
    def bradley_terry(y_w, y_l):
        r_diff = true_reward[y_w] - true_reward[y_l]
        return 1.0 / (1.0 + random.exp(-r_diff))

    # 第三阶段：带 KL 惩罚的 PPO
    def ppo_step(beta=0.1, lr=0.01):
        nonlocal policy
        policy = {a: policy[a] + lr * (true_reward[a] - 0.5)
                  for a in actions}
        # 归一化 + KL 约束（简化）
        total = sum(policy.values())
        policy = {a: policy[a] / total for a in actions}
        # KL 惩罚
        kl = sum(sft_policy[a] * __import__('math').log(sft_policy[a] / max(policy[a], 1e-10))
                 for a in actions)
        multiplier = 1.0 if kl < beta else 0.5
        return multiplier

    return {"SFT": sft_policy, "PPO": policy}
```

### 第 1 步：三阶段流水线模拟

```python
import random
import math


class InstructGPTPipeline:
    """InstructGPT 三阶段流水线的简化实现。"""

    def __init__(self, actions=["A", "B", "C"]):
        self.actions = actions
        self.policy = {a: 1/len(actions) for a in actions}
        self.sft_policy = None
        self.reward_model = None

    def stage1_sft(self, true_reward):
        """第一阶段：SFT——模拟监督微调。"""
        self.sft_policy = dict(self.policy)
        self.true_reward = true_reward
        return self.sft_policy

    def stage2_rm(self, temperature=0.1):
        """第二阶段：RM——简化版 Bradley-Terry 奖励模型。"""
        self.reward_model = {
            a: self.true_reward[a] + random.gauss(0, temperature)
            for a in self.actions
        }
        return self.reward_model

    def stage3_ppo(self, steps=200, beta=0.1, lr=0.01):
        """第三阶段：带 KL 惩罚的 PPO。"""
        trajectory = []
        for step in range(steps):
            # 策略更新
            for a in self.actions:
                self.policy[a] += lr * (self.reward_model[a] - 0.5)

            # 归一化
            total = sum(self.policy.values())
            self.policy = {a: v/total for a, v in self.policy.items()}

            # KL 约束检查
            if self.sft_policy:
                kl = sum(self.sft_policy[a] * math.log(
                    self.sft_policy[a] / max(self.policy[a], 1e-10)
                ) for a in self.actions)
                if kl > beta:
                    # 退火：向 SFT 策略靠拢
                    self.policy = {
                        a: self.policy[a] * 0.8 + self.sft_policy[a] * 0.2
                        for a in self.actions
                    }
                    total = sum(self.policy.values())
                    self.policy = {a: v/total for a, v in self.policy.items()}

            # 黄金奖励
            gold = sum(self.true_reward[a] * self.policy[a] for a in self.actions)
            trajectory.append({"step": step, "gold": gold, "policy": dict(self.policy)})

        return trajectory


# 演示
pipeline = InstructGPTPipeline()
pipeline.stage1_sft({"A": 0.8, "B": 0.5, "C": 0.2})
pipeline.stage2_rm()
traj = pipeline.stage3_ppo(steps=100, beta=0.1)

# 对比 beta=0 的情况
pipeline2 = InstructGPTPipeline()
pipeline2.stage1_sft({"A": 0.8, "B": 0.5, "C": 0.2})
pipeline2.stage2_rm()
traj_no_kl = pipeline2.stage3_ppo(steps=100, beta=0.0)

print(f"beta=0.1 最终策略: {traj[-1]['policy']}  黄金奖励: {traj[-1]['gold']:.3f}")
print(f"beta=0.0 最终策略: {traj_no_kl[-1]['policy']}  黄金奖励: {traj_no_kl[-1]['gold']:.3f}")
```

关键观察：`beta=0.0` 时策略倾向于将概率集中到单一动作（模式寻求），而 `beta=0.1` 时策略保持更多样性。这是 KL 惩罚在 PPO 中的核心效果——防止策略过度专业化于 RM 信号。

### 第 2 步：奖励轨迹追踪

```python
def analyze_trajectory(trajectory):
    """分析 PPO 训练轨迹。"""
    rewards = [t["gold"] for t in trajectory]
    max_step = max(range(len(rewards)), key=lambda i: rewards[i])
    final_drop = rewards[-1] - max(rewards) if trajectory else 0
    return {
        "max_reward": max(rewards),
        "max_step": max_step,
        "final_reward": rewards[-1],
        "drop_from_max": final_drop,
        "stable": abs(final_drop) < 0.05,
    }

result = analyze_trajectory(traj)
print(f"最大奖励: {result['max_reward']:.3f} (第{result['max_step']}步)")
print(f"最终奖励: {result['final_reward']:.3f} (下降: {result['drop_from_max']:.3f})")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 RLHF 框架对照

| 框架 | 支持 | 适用场景 |
|---|---|---|
| TRL (HuggingFace) | SFT+RM+PPO | 开源研究、中小规模 |
| DeepSpeed Chat | 全流程 RLHF | 大规模分布式训练 |
| OpenRLHF | Ray 分布式 RLHF | 企业级生产流水线 |
| NeMo-Aligner | NVIDIA 生态 | 需要 NeMo 集成的场景 |

### 4.2 三阶段对照

| 阶段 | 输入 | 损失函数 | 典型模型大小 | 输出 |
|---|---|---|---|---|
| SFT | 提示词-响应对 | 交叉熵 | 与基础模型相同 | 指令跟随策略 |
| RM | (y_w, y_l) 成对偏好 | Bradley-Terry | 6B（足够 175B 模型） | 标量奖励 |
| PPO | 策略+RM+KL | PPO + KL 惩罚 | 与 SFT 相同 | 对齐策略 |

---

## 5. 工程最佳实践

### 5.1 KL 系数是 RLHF 最重要的超参数

太低 → 奖励黑客，策略过度优化代理 RM。太高 → 没有比 SFT 更好的改进。典型网格搜索范围：`[0.01, 0.05, 0.1, 0.5]`。监控 KL 散度和黄金奖励的比值——如果 KL 增长快于黄金奖励，`beta` 需要增加。

### 5.2 PPO-ptx 防止对齐税

混合预训练梯度到 RL 目标中。典型 `gamma = 0.03`。否则模型在标准基准上退步。

### 5.3 奖励模型大小不必与策略相同

InstructGPT 用 6B RM 训练 175B 策略。RM 只需要区分偏好，不需要生成。小 RM 节省计算。

---

## 6. 常见错误

### 错误 1：KL 系数设为零

**现象：** PPO 训练几十步后策略找到 RM 的对抗性示例——输出无关的字符串但得分极高。

**原因：** KL 惩罚为零。策略可以自由漂移到 RM 未验证的区域。

**修复：** 设置 `beta = 0.1`（典型值）并监控 KL 散度。

### 错误 2：不 PPOP-px 导致基准退步

**现象：** RLHF 后标准基准（SQuAD、HellaSwag）分数下降。

**原因：** 对齐税——RLHF 只优化了 RM 奖励，没有保留预训练学到的能力。

**修复：** 在 PPO 目标中混合预训练 log-likelihood（PPO-ptx），`gamma = 0.03`。

### 错误 3：RM 太大或太小

**现象：** 使用与策略相同大小的 RM（175B），训练成本翻倍；使用 1B RM，评分不准确。

**原因：** InstructGPT 证明 6B RM 足够 175B 策略。太大浪费，太小不准确。

**修复：** RM 大小约为策略的 3-10%。6B RM 对 175B 策略足够。

---

## 7. 面试考点

### Q1：为什么 1.3B 的 InstructGPT 能打败 175B 的 GPT-3？（难度：⭐⭐）

**参考答案：**
对齐是一个不同于能力的维度。175B GPT-3 有更多能力（训练在更多数据上、更大的模型），但它在"回答问题"这个任务上没有对齐——它的训练分布是网络文本，它倾向于继续生成网络文本。1.3B InstructGPT 通过 RLHF 与人类偏好对齐，输出人类更喜欢的回答。标注员偏好对齐的那个，即使它能力更弱。

### Q2：PPO-ptx 如何影响 RLHF 的效果？（难度：⭐⭐⭐）

**参考答案：**
PPO-ptx 在 PPO 目标中混合了预训练 log-likelihood：`J_ptx = J(pi) + gamma × E[log pi(x)]`。这防止了对齐税——RLHF 只优化 RM 信号，策略可能忘记预训练中学习的通用能力。PPO-ptx 通过强制策略保持预训练分布来保持这些能力。典型 `gamma = 0.03`。这成了 Anthropic、DeepMind 和 Meta 的标准做法。

### Q3：如果 KL 系数 beta 太高或太低，分别观察到什么现象？（难度：⭐⭐）

**参考答案：**
beta 太低（如 0.01）：策略快速过度优化 RM 信号——找到对抗性示例，在代理 RM 上得分很高但在真实人类偏好上很低。beta 太高（如 1.0）：策略几乎不移动——RLHF 效果接近 SFT。最佳值是奖励改进和 KL 散度之间的平衡点。经验法则：监控 KL/奖励比值，如果 KL 增长快于奖励，beta 需要增加。

### Q2：对齐税是什么？如何缓解？（难度：⭐⭐）

**参考答案：**
对齐税是 RLHF 之后模型在标准基准上退步的现象——因为 RLHF 只优化了 RM 奖励，没有保留预训练学到的能力。缓解方案是 PPO-ptx：在 PPO 目标中混合预训练 log-likelihood。这个变体成为了标准——Anthropic、DeepMind、Meta 都使用变体。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| SFT | "指令调优" | 第一阶段：提示词-响应对上的交叉熵微调 |
| 奖励模型 | "RM" | 基于 Bradley-Terry 成对标签的标量回归器 |
| KL 惩罚 | "正则化器" | 保持 RL 策略接近 SFT 锚点 |
| PPO-ptx | "带预训练混合的 PPO" | 在 PPO 目标中加入预训练 log-likelihood 的分数 |
| 对齐税 | "RLHF 退步" | RLHF 后在未目标基准上的下降 |

---

## 📚 小结

InstructGPT 定义了 RLHF 的参考架构：SFT（监督微调）+ RM（奖励模型）+ PPO（带 KL 惩罚）。1.3B 的 InstructGPT 在约 70% 的情况下优于 175B 的 GPT-3——对齐是一个不同于能力的维度。KL 系数是 RLHF 最重要的超参数。PPO-ptx 缓解了对齐税。后来每一课对 RLHF 的批评都从这个参考点出发。

---

## ✏️ 练习

1. 运行 `code/main.py`。设 `beta=0.0` 报告 200 步 PPO 后的动作分布。用一段文字解释模式寻求行为。
2. 修改奖励模型在动作 B 上加 +0.5 偏置。运行 PPO `beta=0.1`——KL 惩罚能防止策略利用偏置吗？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| InstructGPT 模拟 | `code/main.py` | 三阶段流水线的简化实现 |
| InstructGPT 解释器 | `outputs/skill-instructgpt-explainer.md` | 识别 RLHF 流水线中被修改的阶段 |

---

## 📖 参考资料

1. [论文] Ouyang et al. — Training language models to follow instructions with human feedback. arXiv:2203.02155 — InstructGPT 论文
2. [论文] Christiano et al. — Deep reinforcement learning from human preferences. arXiv:1706.03741 — 原始偏好 RL
3. [论文] Bai et al. — Training a Helpful and Harmless Assistant with RLHF. arXiv:2204.05862 — Anthropic HH
