# 综合项目45——梯度裁剪与混合精度（Gradient Clipping & Mixed Precision）

> 优化器和学习率调度假设梯度是正常的。它们通常不是。一个坏批次就能让梯度范数飙升三个数量级。混合精度训练通过引入 FP16 溢出进一步放大了这个问题。本节课构建生产训练不可或缺的两条安全带：梯度裁剪到配置的全局 L2 范数，以及带有 autocast 和 GradScaler 的混合精度循环。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第42-44节
**预计时间：** 90分钟

---

## 学习目标

- 计算所有参数梯度的全局 L2 范数，并在超过配置阈值时就地裁剪
- 在训练步骤中嵌入 autocast 和 GradScaler，使 FP16 前向和反向传播在溢出时存活
- 检测损失或梯度中的 NaN 和 Inf，跳过优化器步骤并记录跳过
- 每步记录 GradScaler 的缩放因子，使连续的跳过行为立即可见

---

## 1. 问题

一个昨天还正常运行的训练任务，在第 8,217 步时损失曲线突然垂直起飞。罪魁祸首是一个梯度范数为 4,200 的批次——是此前峰值的 20 倍。没有裁剪，优化器应用了一个足以抹去模型上一小时所有学习的步长。有了全局 L2 裁剪（阈值为 1.0），同一个批次贡献的是单位范数更新；损失保持在趋势线上；训练存活。

混合精度训练通过在前向传播和大部分反向传播中使用 FP16，将吞吐量提升 2-3 倍。代价是 FP16 的指数范围很窄。一个在 FP16 中溢出的典型梯度会变成 Inf，接着通过后续层传播为 NaN，然后在下一步优化器中把所有权重变成 NaN。PyTorch 的 GradScaler 通过在反向传播前将损失乘以一个大缩放因子，并在优化器步骤前将梯度除以相同因子来解决这个问题。如果在反缩放时任何梯度是 Inf 或 NaN，缩放器跳过该步并将缩放因子减半；如果之前 N 步都正常，缩放器将因子加倍。在整个训练过程中，缩放因子会找到 FP16 范围允许的最高值。

构建问题的关键在于正确连线。在反缩放之前裁剪是在缩放后的梯度上操作；在反缩放之后裁剪则涉及 GradScaler 的操作顺序。正确的顺序是：`scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → `clip_grad_norm_` → `scaler.step(optimizer)` → `scaler.update()`。任何其他顺序都会产生一个静默损坏的训练循环。

---

## 2. 核心概念

### 2.1 全局 L2 范数

全局 L2 范数是所有梯度连接成单一向量后的欧几里得范数，而不是逐参数的范数。PyTorch 通过 `torch.nn.utils.clip_grad_norm_(parameters, max_norm)` 实现这个功能。函数返回裁剪前的范数，因此可以同时记录自然值和裁剪后的值——这是诊断"我们每步都在裁剪"问题的必要条件。

### 2.2 autocast 和 GradScaler

`torch.amp.autocast(device_type)` 是一个上下文管理器，选择性地将符合条件的运算（主要是矩阵乘法类运算）在 FP16 中执行。`torch.amp.GradScaler(device_type)` 是一个辅助工具，在反向传播前放大损失，在优化器步骤前反向缩放梯度。两者是配合设计的；单独使用其中一个而没有另一个是配置错误。

本节课使用 CPU autocast，因为它可以在 CI 中运行；同样的模式通过将 `device_type="cpu"` 改为 `device_type="cuda"` 即可直接迁移到 CUDA。CPU 上的 GradScaler 是一个存根（CPU autocast 默认以 BF16 运行，不需要损失缩放），但课程包含所有调用点，使其与 GPU 循环的接线完全一致。

### 2.3 NaN 和 Inf 检测

检测发生在两个地方。首先，损失本身在反向传播前用 `torch.isfinite` 检查；Inf 或 NaN 的损失不会产生有用的梯度，直接跳过而不进入优化器。其次，在 `scaler.unscale_(optimizer)` 之后，课程用 `has_non_finite_grad(...)` 扫描反缩放后的梯度，将任何 Inf 或 NaN 视为跳过。两个检查共同覆盖了前向传播和反向传播的失败模式。

### 2.4 缩放因子诊断

缩放因子是 GradScaler 的内部状态。每步读取 `scaler.get_scale()` 并记录在学习率和梯度范数旁边。健康的训练运行会显示缩放因子以 2 的幂次增长，直到饱和在 $2^{17}$ 或 $2^{18}$ 附近。异常运行会显示因子在高值和低值之间振荡——这是模型梯度有时在范围内、有时不在的信号。不记录就无法进行诊断。

---

## 3. 从零实现

```python
"""梯度裁剪与混合精度训练步骤。"""
from __future__ import annotations
import csv, math, sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("需要 PyTorch。安装：pip install torch") from exc

DEFAULT_MAX_NORM = 1.0
DEFAULT_DEVICE = "cpu"
NORM_TYPE = 2.0

@dataclass
class StepLog:
    """每步训练日志的一行。"""
    step: int
    lr: float
    grad_l2_pre_clip: float
    grad_l2_post_clip: float
    loss: float
    skipped: bool
    skip_reason: str
    scaler_scale: float

    def to_csv_row(self) -> list[str]:
        return [str(self.step), f"{self.lr:.10f}", f"{self.grad_l2_pre_clip:.10f}",
                f"{self.grad_l2_post_clip:.10f}", f"{self.loss:.10f}",
                "1" if self.skipped else "0", self.skip_reason, f"{self.scaler_scale:.6f}"]

def has_non_finite_grad(parameters: Iterable[torch.nn.Parameter]) -> bool:
    """如果任何梯度包含 NaN 或 Inf 则返回 True。"""
    for param in parameters:
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad.detach()).all().item():
            return True
    return False

def compute_global_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """计算所有梯度的欧几里得范数，不裁剪。"""
    squared_sum = sum(float(p.grad.detach().pow(2).sum().item())
                      for p in parameters if p.grad is not None)
    return math.sqrt(squared_sum)

def clip_global_l2_norm(parameters: list[torch.nn.Parameter], max_norm: float) -> tuple[float, float]:
    """就地裁剪梯度到 max_norm，返回 (裁剪前, 裁剪后)。"""
    if max_norm <= 0:
        raise ValueError("max_norm 必须为正数")
    pre_clip = compute_global_l2_norm(parameters)
    if not math.isfinite(pre_clip) or pre_clip <= max_norm:
        return pre_clip, pre_clip
    scale = max_norm / (pre_clip + 1e-12)
    for param in parameters:
        if param.grad is not None:
            param.grad.detach().mul_(scale)
    return pre_clip, max_norm

class AmpTrainState:
    """带混合精度和梯度裁剪的训练步骤。

    接线顺序：autocast 前向 → loss 有限性检查 → scaler.scale(loss).backward()
    → scaler.unscale_(optimizer) → 梯度有限性检查 → clip → scaler.step → scaler.update。
    """

    def __init__(self, model: nn.Module, lr: float = 1e-2, max_norm: float = DEFAULT_MAX_NORM,
                 device_type: str = DEFAULT_DEVICE, weight_decay: float = 0.01):
        if max_norm <= 0:
            raise ValueError("max_norm 必须为正数")
        self.model = model
        self.max_norm = max_norm
        self.device_type = device_type
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler_enabled = device_type == "cuda"
        self.scaler = torch.amp.GradScaler(device_type, enabled=scaler_enabled)
        self.global_step = 0
        self._log: list[StepLog] = []
        self._skip_log: list[StepLog] = []
        self._loss_fn = nn.functional.mse_loss

    def step(self, inputs: torch.Tensor, targets: torch.Tensor,
             gradient_corruptor: Callable[[nn.Module], None] | None = None) -> StepLog:
        """运行一个训练步骤，支持可选的梯度破坏（用于测试）。"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device_type):
            loss = self._loss_fn(self.model(inputs), targets)
        if not torch.isfinite(loss).all().item():
            return self._record_skip(float(loss.detach().cpu().item()), "非有限损失", 0.0, update_scaler=False)
        self.scaler.scale(loss).backward()
        if gradient_corruptor is not None:
            gradient_corruptor(self.model)
        self.scaler.unscale_(self.optimizer)
        if has_non_finite_grad(self.model.parameters()):
            self.scaler.update()
            return self._record_skip(float(loss.detach().item()), "非有限梯度", float("inf"))
        pre_clip, post_clip = clip_global_l2_norm(list(self.model.parameters()), self.max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        record = StepLog(self.global_step, self._current_lr(), pre_clip, post_clip,
                         float(loss.detach().item()), False, "", float(self.scaler.get_scale()))
        self._log.append(record)
        self.global_step += 1
        return record

    def _record_skip(self, loss_value: float, reason: str, pre_clip: float, update_scaler: bool = True) -> StepLog:
        record = StepLog(self.global_step, self._current_lr(), pre_clip, pre_clip,
                         loss_value, True, reason, float(self.scaler.get_scale()))
        self._log.append(record)
        self.global_step += 1
        if update_scaler:
            self.scaler.update()
        return record

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @property
    def log(self) -> list[StepLog]:
        return list(self._log)

    @property
    def skip_count(self) -> int:
        return sum(1 for r in self._log if r.skipped)

def rolling_skip_rate(log: Iterable[StepLog], window: int = 1000) -> list[float]:
    """返回最近 window 步的滚动跳过率。"""
    rows, rates, skipped = list(log), [], []
    for row in rows:
        skipped = (skipped + [1 if row.skipped else 0])[-window:]
        rates.append(sum(skipped) / len(skipped))
    return rates

def write_step_log_csv(log: Iterable[StepLog], path: Path) -> None:
    """写入规范化的训练步骤 CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "lr", "grad_l2_pre_clip", "grad_l2_post_clip", "loss", "skipped", "skip_reason", "scaler_scale"])
        for row in log:
            writer.writerow(row.to_csv_row())

def build_toy_model(in_dim: int = 16, out_dim: int = 4, seed: int = 7) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, out_dim))
    return model, torch.randn(8, in_dim), torch.randn(8, out_dim)

def inject_inf_into_first_grad(model: nn.Module) -> None:
    """测试用：向第一个参数的梯度中写入 +Inf。"""
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data[...] = float("inf")
            return

def run_demo() -> int:
    """训练 20 步，在第 5 步注入非有限梯度以测试跳过路径。"""
    model, inputs, targets = build_toy_model()
    state = AmpTrainState(model=model, lr=1e-2, max_norm=1.0, device_type="cpu")
    for index in range(20):
        corruptor = inject_inf_into_first_grad if index == 5 else None
        record = state.step(inputs, targets, gradient_corruptor=corruptor)
        marker = "跳过" if record.skipped else "步骤"
        print(f"{marker} step={record.step:>3} lr={record.lr:.6f} "
              f"pre_clip={record.grad_l2_pre_clip:>10.6f} post_clip={record.grad_l2_post_clip:>10.6f} "
              f"loss={record.loss:.6f} scale={record.scaler_scale:.1f} reason={record.skip_reason or '-'}")
    print(f"\n跳过次数={state.skip_count}")
    return 0

if __name__ == "__main__":
    sys.exit(run_demo())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| 全局 L2 范数 | 所有可训练参数梯度的欧几里得范数 |
| autocast | 选择性在 `with` 块内以 FP16（或 BF16）执行符合条件的运算 |
| GradScaler | 在反向传播前放大损失、在优化器步骤前反向缩放梯度的辅助工具 |
| 跳过步 | 因梯度或损失非有限而被拒绝的优化器步骤；缩放器因此减半因子 |
| 缩放因子 | GradScaler 的当前乘数；连续正常步后加倍，每次跳过时减半 |

---

## 5. 工程最佳实践

### 5.1 跳过计数器应该是警报，不是日志行

每次训练中少量跳过步是健康的。每轮次数百次跳过是一个硬警报：模型处于 FP16 无法维持的状态，循环正在静默失败。课程跟踪 1,000 步的滚动跳过率，在生产中会在超过 5% 时告警。

### 5.2 裁剪阈值存在于配置中

`max_norm = 1.0` 是语言模型训练的现代默认值。先在小型模型上扫描。较大的阈值让模型从真正困难的批次中恢复；较小的阈值以更嘈杂的损失曲线为代价约束了最坏情况。该阈值与第 44 节的学习率调度位于同一个 YAML 或 JSON 配置中。

### 5.3 范数日志与调度写入同一个 CSV

CSV 列包括：`step, lr, grad_l2_pre_clip, grad_l2_post_clip, loss, skipped, skip_reason, scaler_scale`。审阅者打开文件就能看到调度、梯度故事、缩放因子和跳过结果（及其原因）在一行中。将列分散到多个文件中是不对齐分析的根源。

### 5.4 `scaler.update()` 每步都运行，即使在跳过时

在正常步上，缩放器读取其无 Inf 计数器，递增它，并可能加倍因子。在跳过步上，缩放器减半因子并重置计数器。忘记在跳过路径上调用 `update()` 是"缩放因子从未改变"的 bug 来源。

### 5.5 中文场景特别建议

- **autocast 设备类型必须与优化器设备匹配**：`torch.amp.autocast(device_type="cuda")` 用于 GPU 训练；`torch.amp.autocast(device_type="cpu")` 用于 CPU。混合设备会产生静默类型错误——损失曲线看起来正常但模型没有在学习。
- **反向传播前检查损失**：`torch.isfinite(loss).all()` 是一个张量规约操作，开销几乎可忽略。在 NaN 损失上节省的是整个训练步骤。始终运行它。
- **`set_to_none=True`** 在 `zero_grad` 中将梯度设为 `None` 而非零，让优化器可以跳过未受影响参数组的计算。这是一个免费的吞吐量提升——可略微减少 bug 面。

---

## 6. 常见错误

### 错误 1：裁剪和反缩放的顺序搞反

**现象：** 裁剪后的梯度范数与预期不符，或者缩放因子永远不变。

**原因：** 如果在 `unscale_` 之前裁剪，你是在缩放后的梯度上应用阈值；如果在 `step` 之后裁剪，梯度已经被消耗了。

**修复：**
```python
# ✓ 正确的顺序
scaler.scale(loss).backward()
scaler.unscale_(optimizer)          # 先反缩放
clip_grad_norm_(params, max_norm)  # 再裁剪真实梯度
scaler.step(optimizer)             # 然后更新参数
scaler.update()                    # 最后更新缩放器
```

### 错误 2：跳过步时忘记调用 `scaler.update()`

**现象：** 缩放因子永远停留在初始值 2^15 或 2^16，从未变化。

**原因：** 在跳过路径中，没有对 GradScaler 调用 `update()`，缩放器无法根据跳过步调整因子。

**修复：** 在每条代码路径上都调用 `scaler.update()`，包括跳过路径。

### 错误 3：使用 CPU autocast 的默认数据类型

**现象：** CPU 上的 float16 精度损失严重，或出现非预期溢出。

**原因：** CPU autocast 默认使用 `bfloat16`，它有更宽的指数范围，很少需要损失缩放。显式指定 `dtype=torch.float16` 会引入 FP16 的限制，但 CPU 上没有硬件加速。

**修复：**
```python
# ✓ CPU 上使用 bfloat16（不需要损失缩放）
with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
    loss = loss_fn(model(x), y)
```

---

## 7. 面试考点

### Q1：为什么优化器步骤之前要检查梯度的有限性？（难度：⭐⭐）

**参考答案：** 一个 Inf 或 NaN 的梯度一旦进入 `optimizer.step()`，会立即将所有参数变为 NaN。参数 NaN 后，后续所有步骤都产生 NaN 输出——训练不可逆转地崩溃。在优化器步骤前检查梯度有限性，可以在参数被污染之前跳过这一步，让训练有机会恢复。

### Q2：GradScaler 的缩放因子如何动态调整？（难度：⭐⭐⭐）

**参考答案：** GradScaler 维护一个内部计数器。每步调用 `update()` 时，如果该步没有跳过，计数器递增；当计数器达到 `growth_interval`（默认 2000），缩放因子加倍。如果该步被跳过（检测到梯度溢出），缩放因子减半，计数器重置。这种策略让缩放因子自动找到 FP16 范围允许的最高安全值——太低会浪费精度，太高会导致频繁跳过。

---

## 📚 小结

梯度裁剪和混合精度是现代大语言模型训练中两个不可或缺的安全机制。裁剪保护训练免受灾难性批次的影响，混合精度在保持数值稳定性的同时将吞吐量提升 2-3 倍。你从零实现了一个完整的训练步骤循环，包括 autocast、GradScaler、全局 L2 裁剪和跳过步检测。

下一节将学习梯度累积——当单批次大小受限于显存时，通过在多个微批次上累积梯度来模拟大批次训练。

---

## ✏️ 练习

1. 【理解】用自己的话解释为什么正确的操作顺序是：`scale` → `backward` → `unscale` → `clip` → `step` → `update`。如果顺序颠倒，会发生什么？

2. 【实现】将合成 Inf 注入替换为真实的损失尖峰（将一个批次的 target 乘以 1e8），验证跳过路径是否触发。

3. 【实验】添加一个 `--bf16` 模式，将 autocast 切换为 BF16。BF16 的指数范围比 FP16 更宽，几乎不需要损失缩放。验证同一演示中的跳过率是否降至零。

4. 【思考】添加一个滚动窗口的跳过率计算和一个 CLI 标志，当跳过率在连续 100 步中超过配置阈值时，使训练失败退出。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 混合精度训练步骤 | `code/main.py` | 带梯度裁剪和 GradScaler 的完整训练循环 |
| 可复用提示词 | `outputs/skill-clip-amp.md` | 将裁剪和混合精度集成到训练脚本中的指南 |

---

## 📖 参考资料

1. [论文] Micikevicius et al. "Mixed Precision Training". arXiv 1710.03740. https://arxiv.org/abs/1710.03740
2. [论文] Pascanu, Mikolov, Bengio. "On the difficulty of training recurrent neural networks". arXiv 1211.5063. https://arxiv.org/abs/1211.5063
3. [官方文档] PyTorch `torch.amp.GradScaler`. https://pytorch.org/docs/stable/amp.html
4. [官方文档] PyTorch `torch.nn.utils.clip_grad_norm_`. https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
