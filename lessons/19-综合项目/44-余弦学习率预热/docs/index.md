# 综合项目44——余弦学习率预热（Cosine LR with Linear Warmup）

> 学习率调度是损失函数之后第二重要的决策。AdamW加余弦衰减和线性预热是语言模型训练的现代默认，因为它让模型在脆弱的前一千步更新中看到较小的有效步长，上升到配置的峰值，然后平滑衰减回零。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 实现连接到余弦学习率调度的AdamW优化器
- 在不产生浮点漂移的情况下计算调度的精确值
- 将梯度L2范数与学习率并排记录
- 将调度渲染为人眼可读的文本图和工具可消费的CSV

---

## 1. 问题

前一千次训练更新是最嘈杂的。模型权重仍接近初始化。优化器的运行二阶矩估计未稳定。梯度范数大且嘈杂。如果学习率在这些更新期间处于峰值，模型要么直接发散，要么陷入永远不会逃离的损失平台。

两个已知修复是梯度裁剪（第45节主题）和从低到高提升的学习率调度。

---

## 2. 核心概念

### 2.1 三个区域

- **预热**：步骤0到`warmup_steps`，学习率从0线性提升到`lr_max`
- **余弦衰减**：`warmup_steps`到`total_steps`，余弦曲线从`lr_max`到`lr_min`
- **地板**：`total_steps`之后，学习率固定在`lr_min`

### 2.2 梯度范数日志

调度是训练健康的一半。梯度范数是另一半。两者每步都记录。发散的训练运行在损失之前显示梯度范数峰值。

---

## 3. 从零实现

```python
"""余弦学习率预热——调度+梯度范数日志+文本图。"""
import math, csv
from dataclasses import dataclass

@dataclass
class CosineWithWarmup:
    warmup_steps: int; total_steps: int; lr_max: float; lr_min: float = 0.0

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            if self.warmup_steps == 0: return self.lr_max
            return self.lr_max * step / self.warmup_steps
        if step >= self.total_steps:
            return self.lr_min
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))

def plot_schedule(schedule, steps):
    heights = [schedule(i) for i in range(steps)]
    max_h = max(heights) or 1
    for i, h in enumerate(heights):
        bar_len = int(h / max_h * 40)
        print(f"  {i:3d} |{'#'*bar_len:<40s}| {h:.6f}")

def main():
    sched = CosineWithWarmup(warmup_steps=5, total_steps=20, lr_max=1e-3, lr_min=1e-5)
    print("学习率调度:")
    for step in range(21):
        lr = sched(step)
        marker = " <-- warmup" if step == 5 else " <-- floor" if step == 20 else ""
        print(f"  step {step:3d}: lr={lr:.6f}{marker}")
    print("\n文本图:")
    plot_schedule(sched, 21)
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 3. 关键术语

| 术语 | 含义 |
|------|------|
| 预热 | 线性从0提升到lr_max |
| 余弦衰减 | 从lr_max到lr_min的余弦曲线 |
| 地板 | total_steps后固定在lr_min |
| 梯度范数 | 梯度向量的欧几里得范数，每步记录 |
| 全局步 | 单调步计数器，跨重启持续 |

---

## 4. 面试考点

**Q1：为什么余弦调度在两个端点都连续？**
**Q2：梯度范数如何帮助诊断训练问题？**
