# 综合项目46——梯度累积（Gradient Accumulation）

> 用你买不起的有效批次大小训练，一次一个微批次。缩放损失，保留优化器步骤，让梯度累积起来。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第42-45节
**预计时间：** 90分钟

---

## 学习目标

- 推导有效批次恒等式：`有效批次 = 微批次 × 累积步数`
- 实现逐微批次的损失缩放，使累积梯度与单次全批次反向传播匹配
- 跳过优化器同步直到最后一个微批次（末步同步模式）
- 解读吞吐量与有效批次的关系曲线，解释收益递减的原因

---

## 1. 问题

你想用 512 的有效批次大小训练，因为损失曲线更平滑，优化器步长在该规模下更有意义。你桌上的加速器在耗尽显存前只能容纳 32 个样本。加倍批次大小不是一个选项。减半模型也不是一个选项。该领域在 2017 年想出的、至今仍在使用的方法，是运行 16 次反向传播，让梯度在参数缓冲区中累积，只在计数达到目标时才执行优化器步骤。

风险在于损失不再与更大的批次相同。16 个微批次的交叉熵简单地相加，是单个全批次损失的 16 倍。不做缩放，梯度方向正确但幅度错误，优化器步长是正确值的 16 倍。修复方法是一次除法。这个除法也很容易忘记。

---

## 2. 核心概念

### 2.1 有效批次恒等式

```
有效批次 = 微批次大小 × 累积步数 × 数据并行世界大小
```

其中：
- **微批次**：单次前向传播中能放进显存的样本量
- **累积步数**：在单个优化器步骤前累积的反向传播次数
- **数据并行世界大小**（可选）：分布式训练中的 GPU 数量

### 2.2 等价性证明

以下两段代码在数值精度误差范围内等价：

```python
# 全批次（理想情况，可能撑爆显存）
loss = criterion(model(x_full), y_full)
loss.backward()
optimizer.step()

# 梯度累积（分 N 次微批次）
for x, y in chunks(x_full, y_full, N):
    scaled = criterion(model(x), y) / N
    scaled.backward()
optimizer.step()
```

累积的梯度缓冲区在循环结束时的内容，与单次全批次反向传播产生的内容相同——上浮点求和顺序的差异。课程代码通过 `equivalence_check` 在 `max_abs_diff < 1e-4` 下断言这一点。

### 2.3 成本在哪

每个微批次的成本是一次前向和一次反向。通过累积，你用时间换内存。随着有效批次在固定微批次下增长：

- **小的累积步数**：损失噪声预算低，优化器步骤频繁
- **大的累积步数**：损失平滑，优化器步骤稀少
- **样本/秒**在硬件极限处饱和
- **每个优化器步骤的总样本量**随累积步数线性增长

没有免费午餐。将 `accum_steps` 加倍，每个优化器步的墙上时间也加倍。变化的是梯度估计的方差：在相同的壁钟预算下，你做了更少的优化器步骤，但每一步都在更多样本上取平均。

---

## 3. 从零实现

```python
"""梯度累积从零实现。

有效批次大小 = 微批次大小 × 累积步数。
通过多个前向和反向传播累积梯度，只在最后一个微批次后执行优化器步骤。
"""
from __future__ import annotations
import argparse, json, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List

import torch
from torch import nn

HERE = Path(__file__).parent
OUT_DIR = HERE.parent / "outputs"
LOG_PATH = OUT_DIR / "accum-curve.json"

@dataclass
class CurvePoint:
    effective_batch: int
    accum_steps: int
    micro_batch: int
    avg_loss: float
    samples_per_sec: float
    median_step_ms: float
    sync_calls: int
    steps: int

def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)

def synthetic_batch(batch_size: int, in_dim: int, out_dim: int, gen: torch.Generator):
    x = torch.randn(batch_size, in_dim, generator=gen)
    y = torch.randint(low=0, high=out_dim, size=(batch_size,), generator=gen)
    return x, y

def make_model(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, out_dim))

def zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_().zero_()

def train_one_optimizer_step(model, optimizer, micro_batches, loss_fn, sync_counter) -> tuple[float, float]:
    """运行 accum_steps 个微批次，累积梯度，执行一步优化。"""
    accum_steps = len(micro_batches)
    zero_grads(model)
    total = 0.0
    for i, (x, y) in enumerate(micro_batches):
        logits = model(x)
        loss = loss_fn(logits, y) / accum_steps
        loss.backward()
        sync_counter[0] += 1
        total += float(loss.detach().item()) * accum_steps
    grad_norm = math.sqrt(sum(float(p.grad.detach().pow(2).sum().item())
                              for p in model.parameters() if p.grad is not None))
    optimizer.step()
    return total / accum_steps, grad_norm

def run_config(effective_batch: int, accum_steps: int, *, in_dim: int, hidden: int,
               out_dim: int, num_steps: int, lr: float, seed: int) -> CurvePoint:
    """在给定配置下运行训练并收集指标。"""
    assert effective_batch % accum_steps == 0
    micro_batch = effective_batch // accum_steps
    seed_everything(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    model = make_model(in_dim, hidden, out_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    losses: list[float] = []
    step_times_ms: list[float] = []
    sync_counter = [0]
    total_samples = 0
    wall_start = time.perf_counter()
    for step in range(num_steps):
        t0 = time.perf_counter()
        micro_batches = [synthetic_batch(micro_batch, in_dim, out_dim, gen)
                         for _ in range(accum_steps)]
        avg_loss, _ = train_one_optimizer_step(model, optimizer, micro_batches, loss_fn, sync_counter)
        step_times_ms.append((time.perf_counter() - t0) * 1000.0)
        losses.append(avg_loss)
        total_samples += effective_batch
    sps = total_samples / max(time.perf_counter() - wall_start, 1e-6)
    step_times_ms.sort()
    return CurvePoint(effective_batch, accum_steps, micro_batch,
                      sum(losses) / len(losses), sps,
                      step_times_ms[len(step_times_ms) // 2],
                      sync_counter[0], num_steps)

def sweep_effective_batches(*, micro_batch: int, accum_grid: list[int], num_steps: int = 25,
                            lr: float = 0.05, seed: int = 0) -> list[CurvePoint]:
    """扫描不同的累积步数并收集吞吐量曲线。"""
    return [run_config(micro_batch * a, a, in_dim=64, hidden=128, out_dim=16,
                       num_steps=num_steps, lr=lr, seed=seed) for a in accum_grid]

def equivalence_check(*, in_dim=32, hidden=48, out_dim=8, big_batch=16,
                       accum_steps=4, lr=0.1, seed=7) -> dict:
    """全批次与累积微批次必须匹配。"""
    micro = big_batch // accum_steps
    # 全批次
    seed_everything(seed)
    gen_a = torch.Generator(); gen_a.manual_seed(seed)
    x, y = synthetic_batch(big_batch, in_dim, out_dim, gen_a)
    seed_everything(seed)
    model_full = make_model(in_dim, hidden, out_dim)
    opt_full = torch.optim.SGD(model_full.parameters(), lr=lr)
    zero_grads(model_full)
    loss_full = nn.CrossEntropyLoss()(model_full(x), y)
    loss_full.backward()
    full_grads = [p.grad.detach().clone() for p in model_full.parameters()]
    opt_full.step()
    full_params = [p.detach().clone() for p in model_full.parameters()]
    # 累积版本
    seed_everything(seed)
    model_accum = make_model(in_dim, hidden, out_dim)
    opt_accum = torch.optim.SGD(model_accum.parameters(), lr=lr)
    zero_grads(model_accum)
    for cx, cy in zip(torch.split(x, micro, dim=0), torch.split(y, micro, dim=0)):
        (nn.CrossEntropyLoss()(model_accum(cx), cy) / accum_steps).backward()
    accum_grads = [p.grad.detach().clone() for p in model_accum.parameters()]
    opt_accum.step()
    accum_params = [p.detach().clone() for p in model_accum.parameters()]
    grad_diffs = [float((a - b).abs().max().item()) for a, b in zip(full_grads, accum_grads)]
    param_diffs = [float((a - b).abs().max().item()) for a, b in zip(full_params, accum_params)]
    return {"max_grad_diff": max(grad_diffs), "max_param_diff": max(param_diffs)}

def write_curve(points: list[CurvePoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "accum-curve.v1", "points": [asdict(p) for p in points]}, indent=2) + "\n")

def main() -> int:
    args = argparse.Namespace(micro_batch=4, accum_grid=[1,2,4,8,16], num_steps=25, lr=0.05, seed=0, no_write=False)
    accum_grid = args.accum_grid
    eq = equivalence_check()
    print("等价性检查（全批次 vs 累积）:", json.dumps(eq, indent=2))
    assert eq["max_grad_diff"] < 1e-4, f"梯度分歧: {eq['max_grad_diff']}"
    assert eq["max_param_diff"] < 1e-4, f"参数分歧: {eq['max_param_diff']}"
    print("等价性成立。运行扫描...")
    points = sweep_effective_batches(micro_batch=args.micro_batch, accum_grid=accum_grid,
                                      num_steps=args.num_steps, lr=args.lr, seed=args.seed)
    header = f"{'有效批次':>10}  {'累积步':>5}  {'微批次':>5}  {'样本/秒':>10}  {'中位ms':>10}  {'同步':>6}  {'损失':>8}"
    print(header)
    for p in points:
        print(f"{p.effective_batch:>10}  {p.accum_steps:>5}  {p.micro_batch:>5}  "
              f"{p.samples_per_sec:>10.1f}  {p.median_step_ms:>10.2f}  {p.sync_calls:>6}  {p.avg_loss:>8.4f}")
    if not args.no_write:
        write_curve(points, LOG_PATH)
        print(f"写入 {LOG_PATH}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| 微批次 | 单次前向传播中能放进显存的样本切片 |
| 累积步数 | 一次优化器步骤前累积的反向传播次数 |
| 有效批次 | 微批次 × 累积步数 × 数据并行世界大小 |
| 损失缩放 | 逐微批次除以 N，使求和的梯度与全批次匹配 |
| 末步同步 | 仅在窗口最后一次反向传播时运行梯度集体通信 |

---

## 5. 工程最佳实践

### 5.1 生产中的三个选择

- **微批次大小**的选择目标是占满设备显存。再小浪费加速器周期，再大直接崩溃。
- **有效批次大小**由学习率调度决定。大的有效批次需要缩放的学习率和预热——这就是 2017 年以来讨论的线性缩放规则。
- **累积步数**是两者之间的桥梁，也是唯一可以在运行时自由调整而无需重写数据加载器的旋钮。

### 5.2 末步同步模式

在单设备上这是簿记工作。在多 GPU 集群上，相同的模式将非最终的微批次包裹在 `no_sync` 上下文中，跳过梯度全规约；最后一个微批次一次性规约全部累积梯度，而不是支付 N 次网络通信成本。

```python
# DDP 模式：只在最后一步同步
for i, (x, y) in enumerate(micro_batches):
    if i < accum_steps - 1:
        with model.no_sync():
            (loss_fn(model(x), y) / accum_steps).backward()
    else:
        (loss_fn(model(x), y) / accum_steps).backward()
        # 到这里梯度已全部规约
optimizer.step()
```

### 5.3 中文场景特别建议

- **有效批次不是越大越好**：超过一定阈值后，有效批次的增大会导致优化器步骤质量下降（泛化能力变差）。参考 "large batch" 文献中的泛化差距研究。
- **SGD 和 AdamW 对有效批次大小敏感度不同**：AdamW 的自适应学习率使其对大批次更宽容。切换到 SGD 时，需要更谨慎地调整学习率。
- **累积步数应与数据加载器配合**：确保每个微批次在数据上没有相关性（如来自同一个文档的连续片段）。良好的做法是在数据加载器中打乱后再分微批次。

---

## 6. 常见错误

### 错误 1：忘记损失缩放

**现象：** 梯度范数比预期大 N 倍，优化器步长过大，训练发散。

**原因：** 损失没有除以累积步数，梯度求和后幅值变为 N 倍。

**修复：**
```python
# ❌ 不缩放——梯度 N 倍于应有值
loss = criterion(model(x), y)
loss.backward()

# ✓ 缩放——梯度与全批次匹配
loss = criterion(model(x), y) / accum_steps
loss.backward()
```

### 错误 2：每个微批次都执行优化器步骤

**现象：** 优化器状态（Adam 动量、二阶矩）更新频率错误，学习率调度与步数不对齐。

**原因：** 优化器步骤只在最后一个微批次后执行一次，而不是每个微批次后都执行。

**修复：** 将 `optimizer.step()` 放在微批次循环之外。

### 错误 3：数据并行时每个微批次都触发全规约

**现象：** 网络通信开销与累积步数成比例增长，吞吐量没有提升。

**原因：** DDP 在每个 `backward()` 后自动触发梯度全规约。前 N-1 个微批次的规约是浪费的。

**修复：** 使用 `model.no_sync()` 上下文管理器跳过非最终微批次的规约。

---

## 7. 面试考点

### Q1：为什么梯度累积中损失需要除以累积步数？（难度：⭐⭐）

**参考答案：** PyTorch 的 `backward()` 将梯度累加到 `.grad` 缓冲区（不替换）。N 个微批次的梯度求和后是 N 倍于全批次。除以 N 将累积梯度缩放回正确的量级，使其与单次前向传播一个 N 倍大小的批次等价。

### Q2：有效批次大小如何影响学习率？（难度：⭐⭐⭐）

**参考答案：** 线性缩放规则（Goyal et al., 2017）指出，当批次大小乘以 k 时，学习率也应乘以 k。直觉是：更大的批次包含更多的样本，梯度估计的方差更小，因此可以承受更大的步长。但在实践中，学习率有一个上限——超过这个上限，即使批次再大，模型也无法收敛。因此大多数训练脚本使用平方根缩放或直接扫描找到最优值。

---

## 📚 小结

梯度累积让你在显存受限的情况下训练出任意大的有效批次。你实现了损失缩放、末步同步模式，并通过扫描实验验证了吞吐量与有效批次之间的关系。这是弥合单 GPU 实验与多 GPU 生产训练之间的桥梁。

下一节将学习检查点保存与恢复——当训练因为集群重启或配额限制而中断时，如何从中断处精确恢复。

---

## ✏️ 练习

1. 【理解】用自己的话解释为什么 `(loss / accum_steps).backward()` 与 `loss.backward()` 后再手动缩放梯度不等价。

2. 【实现】添加一个错误的缩放变体（不做除法），对比第 1 步的参数差异。

3. 【实验】重新运行扫描，`--num-steps 100` 并绘制样本/秒与有效批次的关系曲线。曲线在哪里趋于平缓？

4. 【思考】引入真实的 `DistributedDataParallel` 封装，并将 `no_sync_context` 路由到其方法。确认同步调用次数从 N 降为每个有效批次 1 次。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 梯度累积演示 | `code/main.py` | 从零实现的梯度累积，含等价性检查和吞吐量扫描 |
| 吞吐量曲线数据 | `outputs/accum-curve.json` | 可复用的有效批次扫描结果 |

---

## 📖 参考资料

1. [论文] Goyal et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour". arXiv 1706.02677. https://arxiv.org/abs/1706.02677
2. [官方文档] PyTorch `DistributedDataParallel.no_sync`. https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
3. [GitHub] PyTorch 梯度累积示例. https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
