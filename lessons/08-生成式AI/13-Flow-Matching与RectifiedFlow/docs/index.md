# Flow Matching 与 Rectified Flow

> 扩散模型需要 20-50 步采样。Flow Matching 用更直的路径——2-5 步就能达到相同质量。Rectified Flow 是 2026 年训练扩散骨干的主流方法。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 08 · 06（DDPM）、07（潜在扩散）| **时间：** ~60 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 08 · 06（DDPM）— Flow Matching 扩散的改进方向 | 阶段 08 · 10（视频生成）— Sora/T2V 采用 Flow Matching 训练

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解 Flow Matching 的核心——学习从噪声到数据的最优传输路径
- [ ] 解释为什么 Flow Matching 比 DDPM 快 4-10 倍——路径更直意味着更少采样步数
- [ ] 说明 Rectified Flow 如何通过"校正"进一步缩短路径
- [ ] 使用 Hugging Face Diffusers 中的 Flow Matching 采样器
- [ ] 区分 DDIM、Flow Matching、Consistency Model 三种加速方法的异同

---

## 1. 问题

DDPM 需要 1000 步采样。DDIM 将其压缩到 20-50 步。但这对于实时应用仍然太慢——50 步 × 每步 100ms = 5 秒。视频生成需要每帧都做一次——更不可接受。

问题的本质：**扩散模型的采样路径是弯曲的**（随机微分方程 SDE 的路径），曲线越长需要的步数越多。如果能找到一条更直的噪声→数据路径，就可以用更少的步数到达终点。

Flow Matching 改变了视角：不再将生成视为"逐步去噪"，而是视为**从噪声分布到数据分布的确定性流**。它学习一个向量场（vector field）将噪声"推动"到数据，而不是预测每一步的噪声。

---

## 2. 概念

### 2.1 直观理解

```
DDPM（马尔可夫链，蜿蜒的路径）：   噪声 ─────────────→ 数据（50 步）
                                     ╱╲    ╱╲    ╱
                                    ╱  ╲  ╱  ╲  ╱
                                   ╱    ╲╱    ╲╱
                                  
Flow Matching（连续流，直线路径）：  噪声 ──────────────→ 数据（5 步）
```

想象你在山顶（噪声）要走到山脚（数据）。DDPM 像随机游走——每一步方向不确定，走得很慢。Flow Matching 像看地图找一条最快路线——直接朝着目标方向走。

### 2.2 连续时间连续性

Flow Matching 关注的是**概率路径**（probability path）——在连续时间 $t \in [0,1]$ 中，从噪声分布 $p_0$ 到数据分布 $p_1$ 的平滑转换：

$p_t(x) = \text{路径在时间 } t \text{ 处的分布}$

每一步学习一个向量场 $v_t(x)$ ，它告诉模型"在位置 $x$、时间 $t$ 时，粒子应该向哪个方向移动"。

损失函数：

$$\mathcal{L} = \mathbb{E}_{t \sim [0,1], x_1 \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0,I)} \left[ \left\| v_t(x_t) - \frac{x_1 - x_0}{1 - 0} \right\|^2 \right]$$

其中 $x_t = (1-t) x_0 + t x_1$ 是噪声 $x_0$ 和数据 $x_1$ 之间的线性插值。

**关键洞察：** 不学习噪声→数据的整条路径，而是学习路径上每个点的"速度"（向量场）。这比 DDPM 的"预测噪声"更直接——因为向量场本身就是"朝哪个方向走"。

### 2.3 Flow Matching vs DDPM

| 方面 | DDPM | Flow Matching |
|------|------|---------------|
| 路径类型 | 随机微分方程（SDE） | 常微分方程（ODE） |
| 路径形状 | 弯曲 | 较直（通过最优传输） |
| 训练目标 | 预测噪声 $\epsilon$ | 预测向量场 $v_t(x)$ |
| 采样步数 | 20-50（DDIM） | 2-5 |
| 推理速度 | 每步一次 U-Net | 每步一次 U-Net（但步数更少） |
| 质量 | 基准 | 相近或更好（2026 年） |
| 理论基础 | 马尔可夫链 | 连续归一化流（CNF） |

### 2.4 Rectified Flow——让路径更直

Rectified Flow 在 Flow Matching 的基础上增加了一个"校正"步骤：

```
第 1 轮：噪声 ─────────────→ 数据（初始 Flow Matching，仍有弯曲）
                     ↓
          采样多组配对 (x_0, x_1)
                     ↓
第 2 轮：噪声 ────────→ 数据（Rectified，路径更直）
                     ↓
          再次配对、校正
                     ↓
第 N 轮：噪声 ───→ 数据（几乎直线，仅需 1-2 步）
```

这个"校正"过程被称为**再配对（re-pairing）**。在采样完成后，将同一起点的路径终点重新匹配到新的起点，训练一个更直接的路径。每次迭代后路径就变直一些。

**为什么 Recified Flow 有效？** 因为噪声分布和数据分布之间总是存在最优传输路径——问题是如何找到它。再配对通过多次迭代逐步逼近这个最优传输。

### 2.5 Consistency Model——极致的加速

Consistency Model（So 等人，2023）不是 Flow Matching，但共享相同的目标——加速采样。它蒸馏一个大扩散模型为一个**一步生成**模型：

```
训练：大扩散模型 → 采样多组 (x_0, x_t) 配对
蒸馏：小模型学习直接映射 x_0 → x_1（一步）
推理：噪声 → 一步生成
```

2026 年的许多视频生成和图像编辑模型使用 Consistency Model 做实时推理。

### 2.6 采样步数演进行进

| 方法 | 发表 | 步数 | 质量（FID，CIFAR-10） | 速度提升 |
|------|------|------|---------------------|---------|
| DDPM | 2020 | 1000 | 3.01 | 1× |
| DDIM | 2020 | 50 | 3.02 | 20× |
| DPM-Solver | 2022 | 15 | 3.06 | 67× |
| Flow Matching | 2023 | 10 | 2.99 | 100× |
| Rectified Flow | 2023 | 4 | 3.04 | 250× |
| Consistency Model | 2023 | 1 | 3.92 | 1000× |

---

## 3. 从零实现

Flow Matching 的核心可以用少量代码理解：

### 第 1 步：线性插值和向量场

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def interpolate(x_0, x_1, t):
    """
    线性插值：x_t = (1-t) * x_0 + t * x_1
    这是 Flow Matching 中最常用的概率路径——简单直接。
    Args:
        x_0: 噪声样本 (B, D)
        x_1: 数据样本 (B, D)
        t: 时间步 (B,)，取值 [0, 1]
    Returns:
        x_t: 插值后的样本 (B, D)
    """
    t = t.view(-1, 1)  # (B,) → (B, 1)
    return (1 - t) * x_0 + t * x_1


def compute_target_vector(x_0, x_1):
    """
    Flow Matching 的目标向量场。
    在最优传输下，目标向量场 = x_1 - x_0（常数速度）。
    即从噪声到数据的"最短路径"方向。
    """
    return x_1 - x_0
```

### 第 2 步：Flow Matching 训练

```python
def train_flow_matching(model, dataloader, num_epochs=10, lr=1e-4):
    """
    Flow Matching 训练循环。
    与 DDPM 训练非常相似——只是预测目标从"噪声"变成了"向量场"。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()

        for x_1, _ in dataloader:  # 数据样本
            B = x_1.size(0)
            x_1 = x_1.to(device)

            # 采样噪声
            x_0 = torch.randn_like(x_1)

            # 随机采样时间步 [0, 1]
            t = torch.rand(B, device=device)

            # 线性插值
            x_t = interpolate(x_0, x_1, t)

            # 目标向量场（最优传输方向）
            target = compute_target_vector(x_0, x_1)

            # 模型预测向量场
            predicted = model(x_t, t)

            # MSE 损失
            loss = F.mse_loss(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    return model
```

### 第 3 步：Flow Matching 采样

```python
@torch.no_grad()
def sample_flow_matching(model, num_samples=16, dim=784, num_steps=10, device="cpu"):
    """
    Flow Matching 采样——求解常微分方程（ODE）。
    从 x_0（噪声）到 x_1（数据），使用 Euler 法求解。
    """
    model.eval()
    # 从标准正态分布开始
    x_t = torch.randn(num_samples, dim, device=device)

    # 时间步离散化
    dt = 1.0 / num_steps

    for t in torch.linspace(0, 1 - dt, num_steps, device=device):
        t_batch = torch.full((num_samples,), t.item(), device=device)

        # 模型预测向量场
        v_t = model(x_t, t_batch)

        # Euler 法积分：x_{t+Δt} = x_t + v_t * Δt
        x_t = x_t + v_t * dt

    return x_t
```

### 第 4 步：Rectified Flow 的再配对

```python
def rectified_flow_repairing(flow_model, noise_sampler, num_pairs=10000,
                              num_steps=10, device="cpu"):
    """
    Rectified Flow 的再配对步骤。
    从当前模型采样一批噪声→数据配对，然后用这些配对训练一个更直的路径。
    """
    # 1. 用当前模型采样
    x_0_samples = torch.randn(num_pairs, 784, device=device)
    x_1_samples = sample_flow_matching(
        flow_model, num_samples=num_pairs,
        num_steps=num_steps, device=device
    )

    # 2. 再配对：用同一批训练数据训练新模型
    # 新模型的训练目标是 x_1_samples → x_0_samples 的直线路径
    # 这意味着第二次训练后，生成的路径会更直（因为已经有参考了）

    return x_0_samples, x_1_samples  # 这些配对将用于下一步训练
```

### 第 5 步：Euler 和 Heun 求解器

```python
def euler_solver(model, x_0, num_steps=10, device="cpu"):
    """Euler 法——最简单的一阶 ODE 求解器。"""
    x_t = x_0
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((x_0.size(0),), i * dt, device=device)
        v_t = model(x_t, t)
        x_t = x_t + v_t * dt

    return x_t


def heun_solver(model, x_0, num_steps=10, device="cpu"):
    """
    Heun 法——二阶 ODE 求解器（更精确）。
    在每个时间步做两次评估，取平均——重估稍高但步数可更少。
    """
    x_t = x_0
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((x_0.size(0),), i * dt, device=device)

        # Euler 预测步
        v_t = model(x_t, t)
        x_pred = x_t + v_t * dt

        # Heun 校正步
        t_next = torch.full((x_0.size(0),), min((i + 1) * dt, 1.0), device=device)
        v_pred = model(x_pred, t_next)
        # 取平均
        x_t = x_t + 0.5 * (v_t + v_pred) * dt

    return x_t
```

---

## 4. 工具

### 4.1 Hugging Face Diffusers 中的 Flow Matching

```python
from diffusers import StableDiffusionPipeline, FlowMatchEulerDiscreteScheduler
import torch

# 使用 Flow Matching 采样器替换默认 DDIM 采样器
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# 替换为 Flow Matching 采样器
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(num_inference_steps=5)  # 仅需 5 步！

# 生成图像
image = pipe(
    prompt="一只在樱花树下的猫",
    num_inference_steps=5,
).images[0]
```

### 4.2 Stability AI 的 SD3——基于 Rectified Flow

```python
# Stable Diffusion 3 使用 Rectified Flow 框架
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# SD3 使用整流流，只需要 10-15 步
image = pipe(
    prompt="一只戴着帽子的柴犬，写实摄影风格",
    num_inference_steps=10,
).images[0]
```

### 4.3 主流采样器对比

| 采样器 | 类型 | 推荐步数 | 适用场景 |
|--------|------|---------|---------|
| DDIM | SDE | 20-50 | 通用使用 |
| Flow Matching | ODE (Euler) | 5-10 | 快速生成 |
| Rectified Flow | ODE (Heun) | 4-8 | SD3 训练框架 |
| DPM-Solver++ | ODE (高阶) | 10-15 | 通用高性能 |
| Consistency | 蒸馏 | 1 | 极致速度 |

---

## 5. LLM 视角

### 5.1 在主流系统中的体现

- **Stable Diffusion 3**：基于 Rectified Flow 训练——是首个大规模使用 Flow Matching 架构的图像生成模型。SD3 用 10-15 步即可生成高质量图像，而 SD 1.5 需要 50 步。
- **Sora / Open-Sora**：视频生成模型使用 Flow Matching 的训练框架。2026 年的视频生成几乎全部采用 Flow Matching / Rectified Flow 替代了原始的 DDPM 训练。
- **一致性蒸馏（Consistency Distillation）**：2025-2026 年，几乎每个主流扩散模型都发布了自己的"一致性版本"——从原始模型蒸馏出一步/几步生成能力，用于实时推理。

### 5.2 大语言模型时代什么变了？

Flow Matching 的"向量场"视角与扩散模型的"去噪"视角都解决了同一个问题——从噪声到数据的转换——但前者更通用。在一个多模态时代，这种通用性变得关键：Flow Matching 不仅能用于图像，还能用于视频、3D、音频，甚至**文本**（Diffusion-LM 的改进版本采用 Flow Matching 训练）。

### 5.3 什么没变？

损失函数仍然是最小均方误差（MSE）——从 DDPM 到 Flow Matching，核心的训练目标没有变。变的是**预测的内容**：DDPM 预测噪声 $\epsilon$，Flow Matching 预测向量场 $v_t$。但两者的数学框架高度相似——都是学习一个条件得分函数。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你使用最新的图像生成工具时，生成速度已经从 2022 年的 ~30 秒降低到 2026 年的 ~1-2 秒。这个加速很大程度上来自 Flow Matching 和 Consistency Model。你输入的提示词仍然通过 CLIP 编码，但模型内部的数学过程已经更快了。

---

## 6. 工程最佳实践

### 6.1 求解器选择

| 求解器 | 阶数 | 相同步数下的精度 | 推荐场景 |
|--------|------|-----------------|---------|
| Euler | 1 阶 | 低 | 快速原型 |
| Heun | 2 阶 | 中 | 通用 |
| DPM-Solver++ | 2-3 阶 | 高 | 高质量生成 |

### 6.2 步数选择

| 步数 | 速度 | 质量 | 推荐场景 |
|------|------|------|---------|
| 1 | 极快 | 低（需要蒸馏） | 实时生成 |
| 2 | 快 | 中 | 预览/草稿 |
| 5-10 | 中 | 高 | 通用 |
| 20+ | 慢 | 极高 | 质量优先 |

### 6.3 踩坑经验

- **Euler 求解器在 <5 步时质量下降明显**：需要使用更高阶求解器（Heun、DPM-Solver++）
- **Flow Matching 训练比 DDPM 训练更敏感**：学习率通常需要 DDPM 的 1/2 到 1/4
- **整流流（Rectified Flow）的再配对会引入额外的训练迭代**：对于小模型可能不值得，但对大模型（如 SD3）效果显著

---

## 7. 常见错误

### 错误 1：将 Flow Matching 的 t 域与 DDPM 的 t 域混淆

**现象：** 使用 Flow Matching 时仍用 DDPM 的 t ∈ [0, 1000] 而不是 [0, 1]。

**原因：** Flow Matching 的连续时间范式与 DDPM 的离散马尔可夫链使用了不同的时间表示。

**修复：**

```python
# ❌ 错误：使用 DDPM 的时间域
t = torch.randint(0, 1000, (B,), device=device)  # DDPM 的范围

# ✓ 正确：使用 Flow Matching 的时间域
t = torch.rand(B, device=device)  # Flow Matching 的范围 [0, 1]
```

### 错误 2：向量场的输出没有归一化

**现象：** 训练初期 loss 不下降，梯度爆炸。

**原因：** 向量场 $v_t(x) = x_1 - x_0$ 的量级取决于数据的分布范围。对于图像在 [-1, 1] 内，向量场量级在 [0, 2] 范围内。但对于归一化不正确的输入，向量场可能很大。

**修复：**

```python
# ❌ 错误：不归一化的目标向量场
target = x_1 - x_0

# ✓ 正确：确保输入在 [-1, 1] 内
x_1 = (x_1 - x_1.min()) / (x_1.max() - x_1.min()) * 2 - 1  # 归一化到 [-1, 1]
x_0 = torch.randn_like(x_1)  # 标准正态分布在 [-3, 3] 范围
target = x_1 - x_0

# 或：对目标向量场使用 tanh 裁剪
target = torch.tanh(x_1 - x_0) * 2
```

### 错误 3：采样步数太少时使用 Euler

**现象：** 2 步采样时生成的图像严重失真。

**原因：** Euler 一阶求解器在非常少的步数时误差累积严重。

**修复：**

```python
# ❌ 错误：2 步中使用 Euler
solver = Euler(model)
x_1 = solver(x_0, num_steps=2)  # 质量差

# ✓ 正确：2 步中使用 Heun（二阶）
solver = Heun(model)
x_1 = solver(x_0, num_steps=2)  # 质量好得多
```

---

## 8. 面试考点

### Q1：Flow Matching 和 DDPM 的核心区别是什么？为什么 Flow Matching 可以更少步数？（难度：⭐⭐）

**参考答案：**
DDPM 是一个离散的马尔可夫链——每一步只关注"当前噪声→下一步"的局部关系，路径是弯曲的随机过程（SDE）。Flow Matching 学习一个连续的向量场（ODE），噪声到数据的整个路径是确定的。Flow Matching 使用线性插值的概率路径，使得向量场的方向更加直接——从起点指向终点而非曲折行进。因此，Flow Matching 可以用更少的步数（2-10 步）求解 ODE 达到相同的质量。

### Q2：Rectified Flow 的"再配对"是如何工作的？为什么能加速？（难度：⭐⭐⭐）

**参考答案：**
再配对是 Rectified Flow 的核心创新。在第一次训练后，我们有了一个初步的 Flow Matching 模型。用这个模型采样一批噪声→数据的配对，然后在这些配对中重新训练——新模型的输入是噪声样���，目标是第一次采样的结果。经过再配对后，模型学习到的路径变得更直了。这是因为第一次采样已经找到了一条"可以用"的路径，再配对这个路径后，新模型相当于在已有路径的基础上"校直"它。这个过程可以重复多次，每次路径都变得更接近最优传输路径。

### Q3：一致性模型如何从扩散模型蒸馏出一步生成能力？与 Flow Matching 有什么区别？（难度：⭐⭐⭐）

**参考答案：**
一致性模型不是在训练时学习向量场，而是直接学习"噪声→数据"的映射。训练过程如下：(1) 从预训练的大扩散模型采样多组 (x_0, x_t) 配对——x_t 是去噪路径中间点的带噪数据；(2) 训练一个小模型，使得它对同一路径上的任意两个点给出相同的输出（约束：f(x_t, t) = f(x_s, s) 对于所有 t, s 在同一路径上）；(3) 约束目标：一致性模型在任何 t 的输出都等于在 t=0 的输出（即无噪声的数据）。与 Flow Matching 的区别：Flow Matching 仍然是多步采样（只是步数少），Consistency Model 是单步或极少量步采样。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Flow Matching | "直线去噪" | 学习从噪声分布到数据分布的最优传输路径（向量场），替代 DDPM 的马尔可夫链扩散过程 |
| Rectified Flow | "反复校直的流" | Flow Matching 的改进版本——通过再配对（re-pairing）反复迭代，使路径逐次变直 |
| 向量场 | "告诉粒子往哪走" | 一个函数 $v_t(x)$ 表示在时间 t、位置 x 时的速度方向——Flow Matching 的训练目标 |
| 最优传输 | "最短路径" | 从噪声分布到数据分布的"最优搬运方案"——路径尽可能直，因此采样步数尽可能少 |
| ODE 求解器 | "积分器" | 将连续的向量场离散化为可执行的采样步骤——Euler（一阶）、Heun（二阶）、DPM-Solver（高阶） |
| 一致性模型 | "一步去噪" | 从扩散模型蒸馏出的单步生成模型——学习直接映射噪声→数据，跳过中间时间步 |

---

## 📚 小结

Flow Matching 学习从噪声到数据的向量场而非马尔可夫链，将采样步数从 20-50 压缩到 2-10 步。Rectified Flow 通过再配对进一步校直路径。Consistency Model 蒸馏出一步生成能力。2026 年，这些方法已成为图像/视频生成的主流训练框架——SD3 基于 Rectified Flow，Sora 基于 Flow Matching，实时生成使用一致性模型。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释 DDPM 的"预测噪声"和 Flow Matching 的"预测向量场"有什么本质区别。对比两者的训练公式和采样公式。

2. **【实现】** 实现 Flow Matching 的完整训练循环和采样循环，在 MNIST 或 CIFAR-10 上训练 20 个轮次，记录 5 步、10 步、20 步采样下的 FID 近似值。

3. **【实验】** 使用 Diffusers 库在同一模型上对比三种采样器（DDIM、FlowMatchEuler、DPM-Solver++），在同一提示词下计算生成 4 张图像的时间和质量差异。

4. **【思考】** Flow Matching 假设噪声分布是标准正态分布。如果噪声分布是其他分布（如均匀分布、泊松分布），向量场需要如何调整？什么时候 Flow Matching 的假设会失效？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| Flow Matching 完整实现 | `code/flow_matching.py` | 包含插值、向量场、训练、采样（Euler/Heun） |
| Rectified Flow 再配对 | `code/rectified_flow.py` | 再配对步骤和路径校直演示 |
| 采样器对比 | `outputs/sampler-comparison.md` | DDIM vs Flow Matching vs DPM-Solver 对比 |

---

## 📖 参考资料

1. [论文] Lipman et al. "Flow Matching for Generative Modeling". ICLR, 2023. https://arxiv.org/abs/2210.02747
2. [论文] Liu et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow". ICLR, 2023. https://arxiv.org/abs/2209.03003
3. [论文] Song et al. "Consistency Models". ICML, 2023. https://arxiv.org/abs/2303.01469
4. [论文] Song et al. "Score-Based Generative Modeling through Stochastic Differential Equations". ICLR, 2021. https://arxiv.org/abs/2011.13456
5. [GitHub] Stable Diffusion 3: https://github.com/Stability-AI/generative-models

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
