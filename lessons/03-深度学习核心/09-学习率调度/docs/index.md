# 学习率调度

> 学习率是唯一值得你花时间调的超参数。不是架构，不是数据集大小，不是激活函数——是学习率。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 03 · 06（优化器）、阶段 03 · 08（权重初始化）
**预计时间：** ~90 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 03 · 06（优化器）— 学习率调度器与优化器协同工作；阶段 10 · 03（从零构建 GPT）— 预热+余弦退火是大语言模型训练的标准配置

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现常量学习率、阶梯衰减、指数衰减、余弦退火、预热+余弦、单周期六种调度策略
- [ ] 诊断学习率的三种失败模式：过高导致发散、过低导致停滞、不衰减导致振荡
- [ ] 解释为什么 Adam 等自适应优化器需要预热，以及预热如何稳定早期训练
- [ ] 在相同任务上对比不同调度策略的收敛速度，并根据训练预算选择合适的策略
- [ ] 使用 PyTorch 的 `torch.optim.lr_scheduler` 和 HuggingFace 的 `get_cosine_schedule_with_warmup` 配置生产级调度器

---

## 1. 问题

把学习率设成 0.1。三步之后 loss 飙到无穷大，训练发散。设成 0.0001。训练 100 个轮次后，模型几乎没有从随机初始化状态移动。设成 0.01。前 50 个轮次一切正常，然后 loss 在一个最小值附近来回振荡，永远无法收敛——因为每一步都太大了，跨过了最优点。

最优的学习率不是常数。它随训练进程变化。训练初期，你需要大步快速探索；训练末期，你需要极小的步幅精确落入最优解。一个 90% 准确率的模型和一个 95% 准确率的模型之间的差距，往往只是调度策略的不同。

过去三年发布的每一个主流模型都使用了学习率调度。Llama 3 使用峰值学习率 3e-4、2000 步预热、余弦衰减到 3e-5。GPT-3 使用学习率 6e-4、在 3.75 亿个词元上预热。这些不是随意的选择——它们是耗资数百万美元的超参数搜索的结果。

你必须理解调度策略，因为默认值不会适用于你的问题。微调预训练模型时，最优调度不同于从零训练。增大批次大小时，预热周期需要调整。训练在第 10000 步崩溃时，你需要判断这是调度问题还是其他问题。

---

## 2. 概念

### 2.1 直观理解

学习率调度本质上是一个"什么时候走多远"的策略：

```
训练初期（探索期）：
  步子要大 ──→ 快速跳过平坦区域，找到大致方向
  好比开车在高速公路上，油门踩到底

训练中期（学习期）：
  步子适中 ──→ 在最优区域附近稳定下降
  好比进入城区，减速慢行

训练末期（收敛期）：
  步子要小 ──→ 精确落入最小值的最底部
  好比停车入位，一点一点挪
```

### 2.2 常量学习率

最简单的方式。选一个数，每一步都用它。

$$lr(t) = lr_0$$

几乎不可能是最优选择。对训练末期来说太大（振荡），对训练初期来说可能太小（浪费算力）。适合调试和小模型。

### 2.3 阶梯衰减 (Step Decay)

ResNet 时代的经典做法。每隔固定轮次，学习率乘以一个衰减因子（通常是 0.1）。

$$lr(t) = lr_0 \times \gamma^{\lfloor epoch / step\_size \rfloor}$$

其中 $\gamma = 0.1$、$step\_size = 30$ 意味着每 30 个轮次衰减 10 倍。ResNet-50 就用这个——$lr=0.1$，在第 30、60、90 个轮次衰减。

问题在于：最优的衰减时间点取决于数据集和架构。换个任务就需要重新调。而且衰减是突变的——学习率骤降时 loss 可能会短暂飙升。

### 2.4 指数衰减 (Exponential Decay)

每一步都乘以一个小于 1 的系数 $\gamma$，平滑递减：

$$lr(t) = lr_0 \times \gamma^t$$

$\gamma$ 越接近 1，衰减越慢。$\gamma = 0.999$ 意味着每 1000 步学习率衰减到原来的约 37%。相比阶梯衰减，指数衰减的曲线更平滑，不会出现突变。

### 2.5 余弦退火 (Cosine Annealing)

沿余弦曲线从最大值平滑衰减到最小值：

$$lr(t) = lr_{min} + 0.5 \times (lr_{max} - lr_{min}) \times (1 + \cos(\pi \times t / T))$$

其中 $t$ 是当前步数，$T$ 是总步数。

$t=0$ 时余弦值为 1，$lr = lr_{max}$。$t=T$ 时余弦值为 -1，$lr = lr_{min}$。衰减在开始和末尾都很温和，在中间最陡峭。这与"大部分学习发生在训练中期"的经验观察吻合。

这是大多数现代训练的默认选择。除了 $lr_{max}$ 和 $lr_{min}$，几乎没有需要调的超参数。

### 2.6 预热 (Warmup)：为什么要从小开始

Adam 等自适应优化器维护着梯度均值和方差的滑动估计。第 0 步时，这些估计被初始化为零。前几次梯度更新基于的是"垃圾统计量"。如果此时学习率很大，模型会朝着错误方向走大步。

预热解决了这个问题。从一个极小的学习率开始，在前 $N$ 步内线性增长到 $lr_{max}$。等优化器的统计量稳定后，再用正常的学习率训练。

$$lr(t) = lr_{max} \times \frac{t}{warmup\_steps} \quad (t < warmup\_steps)$$

典型的预热量：总训练步数的 1%~5%。Llama 3 训练了约 1.8 万亿个词元，预热了 2000 步。GPT-3 在 3.75 亿个词元上预热。

### 2.7 预热 + 余弦退火 (Linear Warmup + Cosine Decay)

现代训练的默认组合。先线性预热，再余弦衰减：

$$
lr(t) = \begin{cases}
lr_{max} \times \frac{t}{warmup\_steps} & t < warmup\_steps \\
lr_{min} + 0.5 \times (lr_{max} - lr_{min}) \times (1 + \cos(\pi \times \frac{t - warmup\_steps}{T - warmup\_steps})) & t \geq warmup\_steps
\end{cases}
$$

Llama、GPT、PaLM 以及大多数现代 Transformer 模型都使用这个方案。预热防止早期不稳定，余弦衰减让模型稳定收敛。

### 2.8 单周期策略 (1cycle Policy)

Leslie Smith 在 2018 年发现的反直觉方法：前半段把学习率从低升到高，后半段再降回来。

$$
lr(t) = \begin{cases}
\frac{lr_{max}}{25} + (lr_{max} - \frac{lr_{max}}{25}) \times \frac{t}{T/2} & t < T/2 \\
lr_{max} \times (1 - \frac{t - T/2}{T/2}) + \frac{lr_{max}}{10000} \times \frac{t - T/2}{T/2} & t \geq T/2
\end{cases}
$$

为什么要*增加*学习率？高学习率阶段充当正则化——给优化轨迹添加噪声，让模型探索更广阔的损失景观，找到更好的盆地。然后在第二半段精细收敛。

1cycle 通常比余弦退火收敛更快。代价是：你必须提前知道总训练步数。

### 2.9 各调度策略的形状对比

```
常量学习率：
  步 0:    lr=0.050000 ████████████████████████████████████████
  步 250:  lr=0.050000 ████████████████████████████████████████
  步 500:  lr=0.050000 ████████████████████████████████████████

阶梯衰减：
  步 0:    lr=0.050000 ████████████████████████████████████████
  步 125:  lr=0.025000 ████████████████████
  步 250:  lr=0.012500 ██████████
  步 375:  lr=0.006250 █████
  步 500:  lr=0.003125 ██

余弦退火：
  步 0:    lr=0.050000 ████████████████████████████████████████
  步 125:  lr=0.046193 █████████████████████████████████████
  步 250:  lr=0.025001 ████████████████████
  步 375:  lr=0.003808 ███
  步 500:  lr=0.000010

预热+余弦：
  步 0:    lr=0.000000
  步 50:   lr=0.050000 ████████████████████████████████████████
  步 250:  lr=0.025001 ████████████████████
  步 500:  lr=0.000010

单周期(1cycle)：
  步 0:    lr=0.002000 █
  步 250:  lr=0.050000 ████████████████████████████████████████
  步 500:  lr=0.000005
```

### 2.10 已发布的模型使用什么调度策略

| 模型 | 峰值学习率 | 预热 | 调度策略 |
|---|---|---|---|
| Llama 3 (405B) | 3e-4 | 2000 步 | 余弦衰减到 3e-5 |
| GPT-3 (175B) | 6e-4 | 3.75 亿词元 | 余弦衰减到 0 |
| ResNet-50 | 0.1 | 无 | 阶梯衰减 x0.1 (第 30/60/90 轮) |
| BERT (340M) | 1e-4 | 10000 步 | 线性衰减 |

---

## 3. 从零实现

### 第 1 步：最简调度函数

每个函数接收当前步数，返回对应的学习率。

```python
import math


def constant_schedule(step, lr=0.01, **kwargs):
    """常量学习率——始终返回同一个值。"""
    return lr


def step_decay_schedule(step, lr=0.1, step_size=100, gamma=0.1, **kwargs):
    """阶梯衰减：每隔 step_size 步，学习率乘以 gamma。"""
    return lr * (gamma ** (step // step_size))


def cosine_schedule(step, lr=0.01, total_steps=1000, lr_min=1e-5, **kwargs):
    """余弦退火：从 lr_max 沿余弦曲线衰减到 lr_min。"""
    if step >= total_steps:
        return lr_min
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * step / total_steps))
```

### 第 2 步：加入预热 + 余弦

```python
def warmup_cosine_schedule(step, lr=0.01, total_steps=1000,
                           warmup_steps=100, lr_min=1e-5, **kwargs):
    """预热 + 余弦退火：先线性升温，再余弦衰减。"""
    if total_steps <= warmup_steps:
        return lr * (step / max(warmup_steps, 1))
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))
```

### 第 3 步：加入指数衰减和单周期

```python
def exponential_decay_schedule(step, lr=0.01, gamma=0.999, **kwargs):
    """指数衰减：每个步骤都乘以 gamma。"""
    return lr * (gamma ** step)


def one_cycle_schedule(step, lr=0.01, total_steps=1000, **kwargs):
    """单周期策略：前半段升，后半段降。"""
    mid = max(total_steps // 2, 1)
    if step < mid:
        return (lr / 25) + (lr - lr / 25) * step / mid
    else:
        progress = (step - mid) / max(total_steps - mid, 1)
        return lr * (1 - progress) + (lr / 10000) * progress
```

### 第 4 步：在圆环数据集上对比

使用一个简单的两层全连接网络（输入 2 维 → 隐藏层 8 个神经元 → 输出 1 维），在圆环分类数据集上分别用六种调度策略训练 300 个轮次：

```text
  策略                起始 Loss   中期 Loss   最终 Loss   最低 Loss
  --------------------------------------------------------------
  常量学习率           0.249998    0.160315    0.159866    0.159866
  阶梯衰减             0.249998    0.159457    0.105674    0.105674
  指数衰减             0.249998    0.155756    0.090249    0.090249
  余弦退火             0.249998    0.149271    0.056458    0.056458
  预热+余弦            0.249998    0.128899    0.048134    0.048134
  单周期(1cycle)       0.249998    0.133297    0.055068    0.055068
```

### 第 5 步：学习率敏感性实验

用常量学习率分别设为 1.0、0.1、0.05、0.01、0.001、0.0001 训练 100 个轮次：

```text
    学习率     起始 Loss     最终 Loss           状态
  ----------------------------------------------------
      1.0000     0.249998          NaN            发散
      0.1000     0.249998     0.159889        正在学习
      0.0500     0.249998     0.170138        正在学习
      0.0100     0.249998     0.236929      几乎没动
      0.0010     0.249998     0.248519      几乎没动
      0.0001     0.249998     0.249844      几乎没动
```

**三种失败模式：**
- **过高（1.0）**：loss 直接飙升到 NaN，发散
- **过低（0.001/0.0001）**：100 轮后几乎没动，浪费算力
- **不衰减（0.1/0.05）**：前期在学，但无法进一步收敛

---

## 4. 工业工具

### 4.1 PyTorch 内置调度器

PyTorch 在 `torch.optim.lr_scheduler` 中提供了所有常用调度器：

```python
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR,
    OneCycleLR, ReduceLROnPlateau
)

# 定义模型和优化器
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# 阶梯衰减：每 30 个轮次衰减 0.1 倍
scheduler_step = StepLR(optimizer, step_size=30, gamma=0.1)

# 指数衰减：每轮次乘以 0.95
scheduler_exp = ExponentialLR(optimizer, gamma=0.95)

# 余弦退火：1000 步内从当前 lr 衰减到 1e-5
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)

# 单周期：前 500 步升温，后 500 步降温
scheduler_onecycle = OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=1000,
    pct_start=0.3  # 前 30% 的步数用于升温
)

# ReduceLROnPlateau：loss 停滞时自动降低学习率
scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
```

使用示例：

```python
# 训练循环中调用调度器
for epoch in range(100):
    for batch in dataloader:
        loss = train_step(model, batch, optimizer)
        optimizer.step()
        optimizer.zero_grad()
        scheduler_onecycle.step()  # OneCycleLR 每个 batch 后调用

    # ReduceLROnPlateau 需要传入监控指标
    val_loss = validate(model, val_loader)
    scheduler_plateau.step(val_loss)
```

### 4.2 HuggingFace 预热 + 余弦调度器

大语言模型微调的标准选择：

```python
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,       # 前 2000 步线性预热
    num_training_steps=100000,   # 总训练步数
)

# 训练循环
for step in range(100000):
    loss = train_step(model, optimizer)
    loss.backward()
    optimizer.step()
    scheduler.step()             # 每步调用一次
    optimizer.zero_grad()
```

### 4.3 各调度器的适用场景

| 调度器 | 推荐场景 | 注意事项 |
|---|---|---|
| `StepLR` | CNN 经典训练（ResNet 等） | 需要手动设定衰减时间点 |
| `ExponentialLR` | 需要平滑衰减的场景 | gamma 需要调参 |
| `CosineAnnealingLR` | 通用默认选择 | 需要知道总步数 |
| `OneCycleLR` | 固定预算下的快速训练 | 必须提前知道总步数 |
| `ReduceLROnPlateau` | 验证 loss 停滞时自动降 lr | 需要验证集，不适合纯训练场景 |
| `get_cosine_schedule_with_warmup` | 大语言模型训练/微调 | HuggingFace 生态标配 |

---

## 5. 知识连线

本课学习的调度策略，是后续所有深度学习课程的基础设施：

- **阶段 03 · 10（卷积神经网络）**：ResNet-50 的经典训练配方就是阶梯衰减——$lr=0.1$，第 30/60/90 轮衰减 10 倍
- **阶段 07 · Transformer 深入**：Transformer 训练标配预热+余弦退火，理解调度策略才能理解为什么 Llama 3 要预热 2000 步
- **阶段 10 · 大语言模型从零**：从零训练 GPT 时，调度策略的选择直接决定模型能否收敛——你会亲手配置 `get_cosine_schedule_with_warmup`

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 大语言模型从零训练 | Warmup + Cosine | Llama/GPT/BERT 的标准选择 |
| 大语言模型微调 | Warmup + Linear Decay 或 Cosine | 预热 3%~10% 总步数 |
| CNN 经典训练 | Step Decay 或 Cosine | ResNet 用 Step，EfficientNet 用 Cosine |
| 快速实验（< 1 小时） | 1cycle | 收敛最快的方案 |
| 训练时长不确定 | Cosine with Warm Restarts | 周期性重置学习率 |
| 自适应调整 | ReduceLROnPlateau | loss 停滞时自动降 lr |

### 6.2 预热步数的选取

- 从零训练：总步数的 1%~5%
- 微调预训练模型：5%~10%（更保守，防止灾难性遗忘）
- 大批次（batch_size > 1024）：等比例增加预热步数

### 6.3 中文场景特别建议

- 微调中文大语言模型时，学习率通常设为 1e-5 ~ 5e-5，比从零训练低一个数量级
- 中文+英文混合数据训练时，建议使用 CosineAnnealingLR 而非 StepLR，因为混合数据的收敛节奏不均匀
- 使用 HuggingFace 的 `Trainer` 时，默认就支持 `get_cosine_schedule_with_warmup`，无需手动实现

### 6.4 踩坑经验

- `OneCycleLR.step()` 必须在每个 batch 后调用，而不是每个 epoch 后——混淆这点会导致学习率曲线完全错误
- `ReduceLROnPlateau` 需要传入验证 loss，不能传入训练 loss，否则会过早降低学习率
- 使用 `get_cosine_schedule_with_warmup` 时，`num_training_steps` 是按 batch 计算的总步数（`num_epochs * len(dataloader)`），不是 epoch 数
- 微调时学习率不要超过 5e-5——过高的学习率会破坏预训练权重，导致灾难性遗忘

---

## 7. 常见错误

### 错误 1：学习率过高导致发散

**现象：** 训练前几步 loss 就飙升到 NaN 或极大值。

**原因：** 学习率过大时，每一步的参数更新幅度过大，跳过了损失函数的最优区域，越走越远。

**修复：**

```python
# ❌ 错误：学习率过高
optimizer = optim.Adam(model.parameters(), lr=0.1)

# ✓ 正确：使用典型的学习率范围
optimizer = optim.Adam(model.parameters(), lr=3e-4)
```

### 错误 2：OneCycleLR 的 step() 调用频率错误

**现象：** 学习率曲线不是预期的"先升后降"形状，而是阶梯状。

**原因：** `OneCycleLR` 按 batch 粒度调度，必须在每个 batch 后调用 `step()`，而不是在 epoch 末尾。

**修复：**

```python
# ❌ 错误：在 epoch 末尾调用
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(model, batch, optimizer)
        optimizer.step()
    scheduler.step()  # 每 epoch 才调一次，太少了

# ✓ 正确：在每个 batch 后调用
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(model, batch, optimizer)
        optimizer.step()
        scheduler.step()  # 每 batch 调一次
```

### 错误 3：微调时学习率设置过高

**现象：** 微调后模型效果比微调前更差，验证 loss 上升。

**原因：** 微调是在预训练权重基础上继续训练。过高的学习率会大幅修改已经学好的权重，导致灾难性遗忘——模型"忘记"了预训练学到的知识。

**修复：**

```python
# ❌ 错误：用从零训练的学习率微调
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# ✓ 正确：微调学习率比从零训练低 1~2 个数量级
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
```

### 错误 4：忘记配置学习率调度器

**现象：** 训练后期 loss 在一个值附近反复振荡，无法进一步下降。

**原因：** 使用常量学习率时，训练后期步长太大，无法精确落入最小值。每次更新都"跨过"最优点，在两侧来回跳动。

**修复：**

```python
# ❌ 错误：不用调度器，全程常量学习率
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# ✓ 正确：加一个余弦衰减
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
```

### 错误 5：ReduceLROnPlateau 传入训练 loss

**现象：** 学习率过早降低，模型还没充分训练就被迫用极小的学习率。

**原因：** 训练 loss 本身就在持续下降，用它作为监控指标会导致调度器误判"已经停滞"。应该用验证 loss。

**修复：**

```python
# ❌ 错误：监控训练 loss
scheduler.step(train_loss)

# ✓ 正确：监控验证 loss
val_loss = evaluate(model, val_loader)
scheduler.step(val_loss)
```

---

## 8. 面试考点

### Q1：为什么 Adam 等自适应优化器需要预热？（难度：⭐⭐）

**参考答案：**

Adam 维护两个滑动平均：一阶矩（动量）和二阶矩（未中心化方差），均初始化为零。前几步的梯度统计量不可靠——方差估计偏差很大，导致自适应学习率计算不准。如果此时学习率很大，模型会按照错误的缩放因子更新参数，可能直接发散。

预热从接近零的学习率开始线性增长，给优化器足够的时间积累准确的统计量。等预热结束时，Adam 的动量和方差估计已经稳定，此时用正常的学习率训练就安全了。

典型预热量是总步数的 1%~5%。Llama 3 训练了 1.8 万亿词元，预热了 2000 步。

### Q2：比较余弦退火和阶梯衰减的优劣。（难度：⭐⭐）

**参考答案：**

阶梯衰减的优点是实现简单、在 ResNet 等经典架构上验证有效。缺点是需要手动设定衰减时间点（不同的数据集和架构需要不同的时间点），且突变的衰减可能导致 loss 短暂飙升。

余弦退火的优点是几乎没有需要调的超参数（只需设定 $lr_{max}$ 和 $lr_{min}$），曲线平滑不会引起训练不稳定，且在大多数任务上表现优于阶梯衰减。缺点是需要提前知道总训练步数。

现代实践中，余弦退火已基本取代阶梯衰减成为默认选择。

### Q3：如果训练 loss 在第 10000 步突然飙升，如何诊断是调度问题还是其他问题？（难度：⭐⭐⭐）

**参考答案：**

首先检查学习率曲线：如果恰好在第 10000 步有阶梯衰减，这可能是正常的"衰减后短暂调整"。如果不是，检查是否出现了 NaN（梯度爆炸）。

如果学习率在该步没有变化，问题可能出在数据（遇到了异常 batch）、模型（梯度爆炸）或优化器（二阶矩估计异常）。

诊断步骤：
1. 画出学习率随步数的变化曲线，确认是否有突变
2. 检查该步的梯度范数是否异常
3. 检查该 batch 的数据是否有异常值
4. 如果使用 ReduceLROnPlateau，检查是否误用了训练 loss

### Q4：1cycle 策略为什么能加速收敛？（难度：⭐⭐⭐）

**参考答案：**

1cycle 的核心洞察是：高学习率阶段充当正则化。当学习率较高时，参数更新的噪声增大，模型被迫探索更广阔的损失景观，而不是被困在初始的局部最小值中。这类似于模拟退火的思想——高温帮助跳出局部最优。

第二半段将学习率大幅降低，模型在探索阶段找到的最佳盆地中精细收敛。

Leslie Smith 在 2018 年的论文中证明，1cycle 可以在相同计算预算下达到与标准训练相同的甚至更好的性能，且训练速度更快。代价是需要提前知道总训练步数，且需要通过学习率范围测试找到合适的 $lr_{max}$。

### Q5：设计题——为一个 70 亿参数的大语言模型微调任务选择学习率调度策略。（难度：⭐⭐⭐）

**参考答案：**

推荐方案：Warmup + Linear Decay（而非 Cosine），原因如下：

1. **调度器选择**：微调预训练模型时，Linear Decay 比 Cosine 更温和，降低灾难性遗忘的风险
2. **学习率**：2e-5 ~ 5e-5（AdamW 优化器），比从零训练低 1~2 个数量级
3. **预热**：5%~10% 的总步数，比从零训练更保守
4. **最小学习率**：设为 0 即可（Linear Decay 的自然终点）
5. **优化器**：AdamW，weight_decay=0.01
6. **代码**：

```python
from transformers import get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.05),  # 5% 预热
    num_training_steps=total_steps,
)
```

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 学习率 (Learning Rate) | "模型学多快" | 乘以梯度的标量，决定每一步参数更新的幅度 |
| 调度策略 (Schedule) | "学习率随时间变" | 一个将训练步数映射到学习率的函数，优化收敛过程 |
| 预热 (Warmup) | "从小开始" | 在前 $N$ 步内将学习率从接近零线性增长到目标值，稳定优化器的统计量 |
| 余弦退火 (Cosine Annealing) | "平滑衰减" | 沿余弦曲线从 $lr_{max}$ 衰减到 $lr_{min}$ 的调度策略 |
| 阶梯衰减 (Step Decay) | "到时间就降" | 在固定轮次将学习率乘以衰减因子（通常 0.1） |
| 指数衰减 (Exponential Decay) | "每步都降一点" | 每一步将学习率乘以一个小于 1 的系数，平滑递减 |
| 单周期策略 (1cycle Policy) | "先升后降" | Leslie Smith 的方法：前半段升学习率（正则化），后半段降（精细收敛） |
| 峰值学习率 (Peak Learning Rate) | "最大 lr" | 训练过程中达到的最高学习率，通常在预热结束时达到 |
| 最小学习率 (Eta Min) | "学习率地板" | 调度器衰减到的最低值，防止学习率降到零导致停止学习 |
| ReduceLROnPlateau | "自动降 lr" | 监控验证 loss，当 loss 停滞时自动降低学习率的调度器 |

---

## 📚 小结

学习率调度的核心洞察是：最优的学习率不是常数。训练初期需要大步探索，末期需要小步收敛。你从零实现了六种调度策略，通过对比实验验证了预热+余弦退火的优越性，并理解了三种学习率失败模式——过高发散、过低停滞、不衰减振荡。

下一课我们将学习正则化技术——Dropout 和权重衰减如何防止过拟合，让你的模型不仅在训练集上学得好，在未见过的数据上也表现良好。

---

## ✏️ 练习

1. 【实现】实现学习率范围测试（Leslie Smith）：从 $10^{-7}$ 到 1 指数增长学习率，训练几百步，画出 loss vs 学习率的曲线。学习率最优值在 loss 开始上升之前。在圆环数据集上验证。

2. 【实验】用预热+余弦退火训练，分别设预热比例为 0%、1%、5%、10%、20%。找出训练最稳定的那个比例，解释为什么过长或过短的预热都不好。

3. 【实现】实现带热重启的余弦退火（SGDR）：每 $T$ 步将学习率重置为 $lr_{max}$，然后再次余弦衰减。在更长的训练中（500 轮）与标准余弦退火对比。

4. 【思考】阅读 Leslie Smith 的 "Super-Convergence" 论文摘要，用自己的话解释为什么高学习率可以充当正则化。如果高学习率是正则化，那它和 Dropout 有什么区别？

5. 【工程】构建一个"调度器诊断工具"：监控训练 loss，当 loss 连续 10 个轮次未下降时自动降低学习率，当 loss 突然飙升时自动回退到上一个好的检查点。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 六种调度策略实现 | `code/main.py` | 从零实现的学习率调度函数，含对比实验 |
| 学习率调度建议提示词 | `outputs/prompt-lr-scheduler-guide.md` | 根据训练配置推荐最优调度策略和超参数 |

---

## 📖 参考资料

1. [论文] Loshchilov & Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts". ICLR, 2017. https://arxiv.org/abs/1608.03983
2. [论文] Smith. "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates". arXiv, 2018. https://arxiv.org/abs/1708.07120
3. [论文] Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models". arXiv, 2023. https://arxiv.org/abs/2307.09288
4. [论文] Goyal et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour". arXiv, 2017. https://arxiv.org/abs/1706.02677
5. [官方文档] PyTorch. "torch.optim.lr_scheduler". https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
6. [官方文档] Hugging Face. "Scheduler". https://huggingface.co/docs/transformers/main_classes/optimizer_schedulers

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、知识连线分析、工程最佳实践、常见错误、面试考点等均为原创内容。
