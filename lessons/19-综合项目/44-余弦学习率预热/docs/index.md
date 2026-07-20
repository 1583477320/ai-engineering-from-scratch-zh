# 综合项目44——余弦学习率预热（Cosine LR with Linear Warmup）

> 学习率调度是损失函数之后第二重要的决策。AdamW 加余弦衰减和线性预热是语言模型训练的现代默认，因为它让模型在脆弱的前一千步更新中看到较小的有效步长，上升到配置的峰值，然后平滑衰减回零。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 实现连接到余弦学习率调度的 AdamW 优化器
- 在不产生浮点漂移的情况下计算调度的精确值
- 将梯度 L2 范数与学习率并排记录
- 将调度渲染为人眼可读的文本图和工具可消费的 CSV

---

## 1. 问题

前一千次训练更新是最嘈杂的。模型权重仍接近初始化。优化器的运行二阶矩估计未稳定。梯度范数大且嘈杂。如果学习率在这些更新期间处于峰值，模型要么直接发散，要么陷入永远不会逃离的损失平台。

两个已知修复是梯度裁剪（第45节主题）和从低到高提升的学习率调度。

如果没有预热阶段，模型在刚开始训练的几步就会收到最大的更新步长——此时梯度方向还几乎是随机的，相当于在一个完全陌生的地形上迈出最大的一步。这通常会导致损失在训练初期爆炸，然后永远无法恢复。

---

## 2. 核心概念

### 2.1 三个区域

```
学习率
  lr_max ┤╱╲                        ╱╲
        ┤╱  ╲                      ╱  ╲
        ┤╱    ╲                    ╱    ╲
        ┤╱      ╲                ╱        ╲
        ┤╱        ╲            ╱            ╲
  lr_min ┤──────────╲__________╱────────────────
         └────┬─────┬─────────────────────────▶ 步数
             预热   余弦衰减                     地板
```

- **预热**：步骤 0 到 `warmup_steps`，学习率从 0 线性提升到 `lr_max`
- **余弦衰减**：`warmup_steps` 到 `total_steps`，余弦曲线从 `lr_max` 平滑下降到 `lr_min`
- **地板**：`total_steps` 之后，学习率固定在 `lr_min`

### 2.2 余弦调度公式

$$
\text{lr}(t) = \text{lr}_{\text{min}} + \frac{1}{2}(\text{lr}_{\text{max}} - \text{lr}_{\text{min}})\left(1 + \cos\left(\pi \cdot \frac{t - t_{\text{warmup}}}{t_{\text{total}} - t_{\text{warmup}}}\right)\right)
$$

其中 $t$ 是当前步数，$t_{\text{warmup}}$ 是预热步数，$t_{\text{total}}$ 是总步数。

余弦调度的关键特性是**两端连续**：在预热结束时，学习率正好等于 `lr_max`；在总步数结束时，学习率正好等于 `lr_min`。两端的导数也为零，使得过渡平滑。

### 2.3 梯度范数日志

调度是训练健康的一半。梯度范数是另一半。两者每步都记录。发散的训练运行在损失之前显示梯度范数峰值。将学习率和梯度范数写入同一个 CSV，可以快速诊断是学习率问题还是梯度问题。

---

## 3. 从零实现

```python
"""余弦学习率预热——调度+梯度范数日志+文本图。"""
import math, csv
from dataclasses import dataclass

@dataclass
class CosineWithWarmup:
    warmup_steps: int
    total_steps: int
    lr_max: float
    lr_min: float = 0.0

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.lr_max
            return self.lr_max * step / self.warmup_steps
        if step >= self.total_steps:
            return self.lr_min
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))


def main():
    sched = CosineWithWarmup(warmup_steps=5, total_steps=20, lr_max=1e-3, lr_min=1e-5)
    print("学习率调度:")
    for step in range(21):
        lr = sched(step)
        marker = " <-- 预热结束" if step == 5 else " <-- 地板" if step == 20 else ""
        print(f"  step {step:3d}: lr={lr:.6f}{marker}")
    # 文本图
    print("\n文本图:")
    heights = [sched(i) for i in range(21)]
    max_h = max(heights) or 1
    for i, h in enumerate(heights):
        bar_len = int(h / max_h * 40)
        print(f"  {i:3d} |{'#' * bar_len:<40s}| {h:.6f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

---

## 4. 工业工具

### 4.1 HuggingFace Transformers

最常用的方式是 HuggingFace 的 `get_cosine_schedule_with_warmup`：

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=100000,
)
```

### 4.2 PyTorch 原生

PyTorch 提供了 `CosineAnnealingLR`，但需要手动实现预热（PyTorch 没有内置的预热调度器）：

```python
import torch.optim.lr_scheduler as lr_sched

# 余弦退火调度
scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=100000, eta_min=1e-5)

# 预热需要手动实现或使用 LambdaLR
warmup_steps = 1000

def lambda_lr(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

warmup_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lambda_lr)
```

### 4.3 现代实践

| 工具 | 优点 | 缺点 |
|------|------|------|
| HuggingFace Schedule | 开箱即用，支持预热 | 依赖 transformers 库 |
| PyTorch LambdaLR + CosineAnnealingLR | 灵活，无额外依赖 | 需要手动组合 |
| 自定义实现 | 完全可控 | 需要编写和测试 |

---

## 5. 工程最佳实践

### 5.1 预热步数的选择

预热步数通常设置为总步数的 1-5%。对于小型实验，500-1000 步预热足够。对于大型模型（>1B 参数），预热可能需要 2000-5000 步。经验法则：`warmup_steps = min(1000, total_steps // 50)`。

### 5.2 学习率的设置

`lr_max` 通常从 3e-4 到 3e-3 之间扫描。`lr_min` 通常设置为 `lr_max` 的 1% 到 10%。余弦调度的最终学习率不收敛到零是为了保持模型在训练结束时仍有微小的学习能力。

### 5.3 中文场景特别建议

- **学习率与批次大小的线性缩放**：当有效批次大小翻倍时，`lr_max` 也应约翻倍。这在分布式训练中尤其重要。
- **CSV 日志与可视化**：使用 `wandb` 或 `tensorboard` 记录学习率和梯度范数。单独依赖 print 输出在大规模训练中不可持续。
- **余弦调度不适用于短期微调**：如果只训练几百步（如 LoRA 微调），使用常数学习率或线性衰减比余弦调度更稳定。

---

## 6. 常见错误

### 错误 1：预热步数为 0

**现象：** 训练初期损失爆炸，loss 直接变为 NaN。

**原因：** `warmup_steps=0` 时学习率从第一步就是最大值，梯度在未稳定的初始化参数上产生超大更新。

**修复：** 始终设置 `warmup_steps > 0`。即使只热身 50 步，也比没有好。

### 错误 2：余弦调度公式中的浮点漂移

**现象：** 在 `total_steps` 时学习率不等于 `lr_min`。

**原因：** 浮点除法导致的精度损失：

```python
# ❌ 可能在最后一步产生微小偏差
progress = step / total_steps
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
# progress 可能不是精确的 1.0
```

**修复：** 显式判断边界条件，在 `step >= total_steps` 时直接返回 `lr_min`。

### 错误 3：梯度范数与学习率分开记录

**现象：** 训练发散后无法判断是学习率问题还是梯度问题。

**原因：** 学习率和梯度范数分别写入不同文件，时间戳不对齐。

**修复：** 将两者写入同一 CSV 文件，每行包含 `step, lr, grad_norm, loss`。

---

## 7. 面试考点

### Q1：为什么余弦调度在两个端点都连续？（难度：⭐⭐）

**参考答案：** 余弦函数在 0 和 π 处的导数都为零——`cos'(0) = -sin(0) = 0`，`cos'(π) = -sin(π) = 0`。这确保学习率在预热结束和总步数结束时平滑过渡，不会出现"断崖式"的变化。相比之下，指数衰减或分段线性衰减在分段点处导数不连续。

### Q2：梯度范数如何帮助诊断训练问题？（难度：⭐⭐）

**参考答案：** 梯度范数是训练健康的先行指标。在损失上升之前，梯度范数通常会先出现异常行为——突然飙升（坏批次）或持续下降（梯度消失）。结合学习率一起观察，可以快速定位问题是出在优化器设置还是数据质量上。如果梯度范数正常但损失上升，问题可能在数据或标签。如果梯度范数异常，问题可能在优化器或模型架构。

---

## 🔑 关键术语

| 术语 | 含义 |
|------|------|
| 预热 | 学习率从 0 线性提升到 `lr_max` |
| 余弦衰减 | 从 `lr_max` 到 `lr_min` 的余弦曲线衰减 |
| 地板 | `total_steps` 后学习率固定在 `lr_min` |
| 梯度范数 | 所有参数梯度向量的欧几里得范数，每步记录用于诊断 |
| 全局步 | 单调递增的步骤计数器，跨训练重启持续 |

---

## 📚 小结

余弦学习率预热是现代语言模型训练的默认调度策略。预热让模型在最脆弱的前一千步中使用较小的学习率，余弦衰减在训练后期平滑降低学习率以收敛到更优解。你从零实现了完整的调度器，并将其与 AdamW 优化器配合使用。

下一节将把本节的调度器与梯度裁剪和混合精度结合起来，构建一个完整的训练步骤循环。

---

## ✏️ 练习

1. 【理解】用自己的话解释预热为什么对语言模型训练特别重要？如果预热步数设置过大或过小，分别会发生什么？

2. 【实现】修改 `CosineWithWarmup` 的 `__call__` 方法，使其输出在 `warmup_steps=0` 时也能正确工作。

3. 【实验】用不同的 `warmup_steps`（0、100、1000）训练同一个模型 10000 步，比较损失曲线的前 1000 步。

4. 【思考】除了余弦调度，还有哪些常见的学习率调度策略（线性、指数、StepLR、OneCycle）？余弦调度相比它们的优缺点是什么？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 余弦学习率调度器 | `code/main.py` | 从零实现的余弦预热调度器 |
| 可复用调度配置 | `outputs/skill-cosine-lr.md` | 余弦预热调度配置指南 |

---

## 📖 参考资料

1. [论文] Loshchilov & Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts". ICLR 2017. https://arxiv.org/abs/1608.03983
2. [官方文档] HuggingFace `get_cosine_schedule_with_warmup`. https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
3. [官方文档] PyTorch `CosineAnnealingLR`. https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
4. [博客] The Learning Rate Cookbook — 各种调度策略的对比与选择
