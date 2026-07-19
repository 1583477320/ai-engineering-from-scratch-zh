# 直接偏好优化家族

> Rafailov et al.（2023）展示了 RLHF 的最优解在偏好数据上有一个封闭形式，所以你可以跳过显式奖励模型，直接优化策略。这个洞察产生了一个家族——IPO、KTO、SimPO、ORPO、BPO——每个修复了 DPO 的一个故障模式。2026 年直接对齐算法在前沿后训练中比 PPO 更多。但第 02 课的过度优化曲线仍然适用：DAA 不逃脱 Goodhart，只是移动了它咬人的位置。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 18 · 01（InstructGPT）、阶段 18 · 02（奖励黑客）、阶段 10 · 08（DPO 基础）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从 RLHF-with-KL 最优解推导 DPO 封闭形式
- [ ] 说出 IPO、KTO、SimPO、ORPO、BPO 各自修复的 DPO 故障模式
- [ ] 区分"隐式奖励差距"和"偏好强度"，解释为什么 IPO 的恒等映射重要
- [ ] 解释为什么 Rafailov et al.（NeurIPS 2024）证明尽管没有显式 RM，DAA 仍然过度优化

---

## 1. 问题

RLHF 目标（第 01 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta × KL(pi || pi_ref)
```

有一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) × pi_ref(y|x) × exp(r(x, y) / beta)
```

所以奖励由最优策略与参考的比率隐含定义：

```
r(x, y) = beta × log(pi*(y|x) / pi_ref(y|x)) + beta × log Z(x)
```

将其代入 Bradley-Terry 偏好似然度——分区函数 `Z(x)` 因为只依赖于 `x` 而消除。剩下的损失只在策略参数中——不需要奖励模型。这就是 DPO。

**问题在于：** 推导假设最优解可达、偏好数据在分布内、参考策略是真正的模态锚点。这些都不精确成立。家族中的每个成员修复了一个不同的被违反的假设。

---

## 2. 概念

### 2.1 DPO（Rafailov et al., 2023）

```
L_DPO = -log sigmoid(
  beta × log(pi(y_w|x) / pi_ref(y_w|x))
  - beta × log(pi(y_l|x) / pi_ref(y_l|x))
)
```

出错的地方：

- 隐式奖励差距 `beta × (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。一个小偏好可以产生任意大的差距
- 损失推动 chosen 和 rejected 的 log-prob 朝相反方向。只要 rejected 掉得更快，chosen 的绝对 log-prob 可以下降——这是"降级 chosen 响应"现象

### 2.2 IPO（Azar et al., 2024）

恒等偏好优化用偏好概率上的恒等映射替换了 log-sigmoid：

```
L_IPO = (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)) - 1/(2×beta))²
```

边界由 `1/(2×beta)` 限制。偏好强度和隐式奖励差距成比例。不会爆炸。

### 2.3 KTO（Ethayarajh et al., 2024）

Kahneman-Tversky 优化完全放弃了成对结构。给定一个单一标记输出和一个二元的"理想"或"不理想"信号，将其映射到前景理论效用。优势：你可以使用非配对数据——后者丰富得多。

### 2.4 SimPO（Meng et al., 2024）

简单偏好优化将对齐信号与生成对齐。**完全移除参考策略**，按长度归一化 log-likelihood。长度归一化消除了利用 DPO 长度偏置故障模式的激励。

### 2.5 ORPO（Hong et al., 2024）

优势比偏好优化在标准 SFT 负对数似然上增加了偏好项。无参考策略——SFT 项就是正则化器。单阶段从基础模型训练到对齐模型。

### 2.6 BPO（ICLR 2026）

降级 Chosen 响应问题：DPO 保持了排序 `y_w > y_l` 但 `y_w` 的绝对 log-prob 可以下降。BPO 添加了单行校正——惩罚 chosen 响应上的下降移动。报道相比 DPO 在数学推理上 Llama 3.1 8B 的准确率提升 10.1%。

### 2.7 通用结论：DAA 仍然过度优化

Rafailov et al. "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms"（NeurIPS 2024）用 DPO、IPO、SLiC 在多个数据集上跨 KL 预算训练策略。黄金奖励-vs-KL 曲线与 Gao et al. 相同的峰值-崩塌形状。隐式奖励在训练期间查询超分布样本。KL 正则化不能稳定这一点。

**DAA 不逃脱 Goodhart。** 它们改变了咬人的表面——从"奖励模型过度优化"变成了"参考策略比率过度优化"。通用修复——更好的数据、集成、早期停止——两者都适用。

### 2.8 2026 年选型指南

- 有大量配对偏好数据：DPO 保守 beta，SimPO（如果长度偏置明显）
- 有非配对二元反馈：KTO
- 想要从基础模型单阶段流水线：ORPO
- 在 DPO 日志中看到降级的 chosen log-prob：BPO
- 偏好强度变化大且 DPO 在饱和：IPO

---

## 3. 从零实现

```python
import math


def dpo_loss(pi_y_w, pi_y_l, pi_ref_w, pi_ref_l, beta=0.1):
    """DPO 损失。"""
    ratio_w = math.log(pi_y_w / pi_ref_w)
    ratio_l = math.log(pi_y_l / pi_ref_l)
    gap = beta * (ratio_w - ratio_l)
    return -math.log(1.0 / (1.0 + math.exp(-gap)))


def bpo_loss(pi_y_w, pi_y_l, pi_ref_w, pi_ref_l, chosen_protect=0.01, beta=0.1):
    """BPO：DPO + chosen 保护。"""
    base = dpo_loss(pi_y_w, pi_y_l, pi_ref_w, pi_ref_l, beta)
    correction = chosen_protect * max(0, pi_ref_w - pi_y_w)
    return base + correction


# 演示
for method in ["DPO", "BPO"]:
    loss = dpo_loss if method == "DPO" else bpo_loss
    l = loss(0.6, 0.3, 0.7, 0.3)
    print(f"{method} 损失: {l:.4f}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 DPO 家族对照

| 方法 | 参考策略 | 配对 | 核心修复 |
|---|---|---|---|
| DPO | 是 | 是 | 基础形式 |
| IPO | 是 | 是 | 有界差距 |
| KTO | 是 | 否 | 非配对 |
| SimPO | 否 | 是 | 长度归一化 |
| ORPO | 否 | 是 | 单阶段 |
| BPO | 是 | 是 | 保护 chosen |

---

## 5. 工程最佳实践

### 5.1 DAA 仍然过度优化

DPO 不逃脱 Goodhart。隐式奖励在训练期间查询超分布样本。用集成 RM + 保守 KL 调度 + 早期停止。

### 5.2 每任务测试所有变体

每个实验室在电池上运行所有五个并选出每任务的最优策略。没有理由数学推理和安全的最优策略相同。

---

## 6. 面试考点

### Q1：DPO 如何从 RLHF 最优解推导？（难度：⭐⭐⭐）

**参考答案：**
RLHF 最优解 `pi*(y|x) ∝ pi_ref(y|x) × exp(r(x,y)/beta)` 可以重写为 `r(x,y) = beta×log(pi*/pi_ref) + beta×log Z(x)`。代入 Bradley-Terry 偏好概率——分区函数 Z(x) 只在 x 上依赖，在比较 y_w 和 y_l 时消除。剩下的损失只在策略参数中。不需要训练或维护一个单独的奖励模型。

### Q2：降级 Chosen 响应问题是什么？哪个变体修复了它？（难度：⭐⭐）

**参考答案：**
DPO 推动 chosen 和 rejected 的 log-prob 朝相反方向。只要 rejected 掉得更快，即使 chosen 的绝对概率也下降——排序保持但 chosen 变差。BPO 在 DPO 损失上增加了单行校正：当 chosen log-prob 下降时增加惩罚。报道在 Llama 3.1 8B 数学推理上 +10.1%。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| DPO | "无奖励模型的 RLHF" | 从 RLHF 最优解封闭形式推导的损失；只策略参数 |
| 隐式奖励 | "log-比率" | DPO 隐含的奖励 |
| IPO | "有界 DPO" | 用恒等映射替换 log-sigmoid；隐式奖励差距有界 |
| KTO | "非配对 DPO" | 前景理论效用，有损失厌恶 |
| SimPO | "无参考 DPO" | 长度归一化 log-likelihood + 边距 |
| ORPO | "单阶段 DPO" | NLL + 优势比偏好项 |
| BPO | "保护 chosen 的 DPO" | DPO + 惩罚 chosen 响应上的下降 |

---

## 📚 小结

DPO 家族派生自 RLHF 最优解的封闭形式——跳过显式 RM 直接优化策略。六个变体各有修复：IPO（有界差距）、KTO（非配对）、SimPO（长度归一化）、ORPO（单阶段）、BPO（保护 chosen）。关键的通用结论：DAA 不逃脱 Goodhart——隐式奖励仍然过度优化。2026 年实践：每任务测试所有变体，用集成+保守调度+早期停止。

---

## ✏️ 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终 chosen log-prob 下降。BPO 应该保持更高的 chosen 绝对概率。
2. 修改偏好数据使所有对强度相等——六种方法中最稳健的和最糟糕的是什么？
3. 读 BPO 论文摘要——单行校正是什么？确认在代码中的实现。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 偏好损失比较器 | `code/main.py` | 六种变体的对比 |
| 偏好损失选型 | `outputs/skill-preference-loss-selector.md` | 根据数据结构推荐损失函数 |

---

## 📖 参考资料

1. [论文] Rafailov et al. — Direct Preference Optimization. NeurIPS 2023, arXiv:2305.18290
2. [论文] Azar et al. — IPO. AISTATS 2024, arXiv:2310.12036
3. [论文] Ethayarajh et al. — KTO. arXiv:2402.01306
4. [论文] Meng, Xia, Chen — SimPO. NeurIPS 2024, arXiv:2405.14734
