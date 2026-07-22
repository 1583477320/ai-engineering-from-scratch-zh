# 迷你框架

> PyTorch 不是魔法。它的核心抽象——张量、自动微分、层、优化器——用 500 行 Python 就能复现。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 03 · 01-09（感知机、多层网络、反向传播、激活函数、损失函数、优化器、正则化、权重初始化、学习率调度）
**预计时间：** ~120 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 03 · 03（反向传播）— 自动微分引擎的核心原理；阶段 07 · 01（Transformer 架构）— 理解框架抽象如何支撑复杂架构

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现一个支持自动微分的 Tensor 类，理解计算图和反向传播如何协同工作
- [ ] 实现 Linear、ReLU、Sigmoid、Dropout 等核心层，解释每层的设计决策和反向传播逻辑
- [ ] 构建 MSELoss、BCELoss 损失函数和 SGD、Adam 优化器，掌握训练循环的完整流程
- [ ] 使用 DataLoader 实现批次加载和数据打乱，理解为什么批量训练比逐样本更新更有效
- [ ] 用自制框架训练一个多层感知机（MLP）解决 XOR 问题，对比 Adam、SGD、Dropout 三种实验配置

---

## 1. 问题

你已经会写反向传播、会用 Sigmoid、会算交叉熵。但当你准备训练一个真实网络时，代码变成了这样：

```python
# 手动管理 6 个权重矩阵的梯度
w1_grad = [0.0] * 16
w2_grad = [0.0] * 8
b1_grad = [0.0] * 16
b2_grad = [0.0] * 8
w3_grad = [0.0] * 1
b3_grad = [0.0] * 1

# 手动写每一层的前向传播
h1 = [max(0, sum(w[i][j] * x[j] for j in range(2)) + b1[i])
      for i in range(16)]
h2 = [max(0, sum(w2[i][j] * h1[j] for j in range(16)) + b2[i])
      for i in range(8)]
output = 1.0 / (1.0 + math.exp(
    -(sum(w3[i][j] * h2[j] for j in range(8)) + b3[0])))

# 手动写每一层的反向传播
# ...几十行梯度计算代码

# 换一个激活函数？重写所有相关代码。
# 加一层？重写所有相关代码。
```

这不是编程，这是在和自己打架。每增加一层，代码量翻倍。每换一个组件，所有相关代码都要改。

框架存在的理由是**抽象**。PyTorch 的 `nn.Linear(16, 8)` 一行代码，背后是权重初始化、前向传播、反向传播、参数管理的完整实现。理解这些抽象的内部工作方式，是区分"调包侠"和"工程师"的分水岭。

本课你将用约 550 行纯 Python（零第三方依赖）构建一个迷你深度学习框架，实现 PyTorch 的核心子集：Tensor 自动微分、Module 基类、Linear 层、ReLU 和 Sigmoid 激活、Dropout 正则化、MSE 和 BCE 损失函数、SGD 和 Adam 优化器、DataLoader 批次加载。然后用它训练一个 MLP，解决困扰了感知机十年的 XOR 问题。

---

## 2. 概念

### 2.1 直观理解

深度学习框架本质上是一套相互协作的抽象：

```
框架的三层抽象：

┌──────────────────────────────────────────────────┐
│  DataLoader               训练循环               │
│  数据 → 分批 → 打乱   数据 → 模型 → 损失 → 反向 | 上层：编排
├──────────────────────────────────────────────────┤
│  Module / Sequential                              │
│  Linear → ReLU → Linear → Sigmoid                │
│  forward ────→ loss.backward() ────→ step       │ 中层：模型定义
├──────────────────────────────────────────────────┤
│  Tensor                                           │
│  数据 + requires_grad + grad + backward_fn        │
│  add, matmul, relu, sigmoid —— 自动微分          │ 底层：计算图
└──────────────────────────────────────────────────┘
```

每一层只关心自己的职责：
- **Tensor 层**：存储数据、跟踪计算图、自动计算梯度
- **Module 层**：封装可学习的参数、定义前向计算
- **训练循环层**：编排数据流、调用前向/反向、更新参数

### 2.2 Module 抽象

PyTorch 中每个层都继承自 `nn.Module`。一个 Module 有三个核心职责：

1. **`forward()`** -- 根据输入计算输出
2. **`backward()`** -- 计算梯度（PyTorch 中由 autograd 自动处理，我们的框架中通过自动微分实现）
3. **`parameters()`** -- 返回所有可训练参数

Linear 是一个 Module、ReLU 是一个 Module、Dropout 是一个 Module。它们都有相同的接口。这就是**组合模式（Composite Pattern）**：一个 Module 序列本身也是一个 Module。

### 2.3 Sequential 容器

`Sequential` 将 Module 链起来。前向传播：数据依次通过 Module 1、Module 2、Module 3。容器本身也是一个 Module。

### 2.4 训练模式 vs 评估模式

Dropout 在训练时随机丢弃神经元，在评估时全部通过。这两种行为通过 `train()` 和 `eval()` 方法切换。每个 Module 都有一个 `training` 标志。

框架不默认假设你在训练还是评估 -- 你必须在训练循环中明确设置 `model.train()`，在评估时设置 `model.eval()`。PyTorch 也是同样的设计。

### 2.5 自动微分

自动微分是框架的核心。每个 Tensor 记录自己是怎么被计算出来的：

```python
x = Tensor([1, 2], requires_grad=True)
y = Tensor([3, 4], requires_grad=True)
z = add(x, y)   # z 记录：我是 x + y 的结果
                 # 我的 backward 是：把梯度传给 x 和 y

z.backward([1, 1])
# 实际上发生的：
#   z._backward_fn([1, 1])
#   -> x.backward([1, 1])    # x.grad = [1, 1]
#   -> y.backward([1, 1])    # y.grad = [1, 1]
```

这个机制沿着计算图链式传播。你的框架中每个算子（linear_forward、relu、sigmoid）都定义了"如何反向传播"的规则。这就是反向传播的模块化实现。

### 2.6 框架架构总览

完整的数据流和梯度流遵循一个标准循环：

```
DataLoader → Model (Sequential) → Loss → Optimizer
                ↑                        |
                |  step() 更新参数        |  backward() 反向传播
                +------------------------+
```

### 2.7 为什么从零构建框架

你可能会想："既然有 PyTorch，为什么还要自己写？"

| | 只调 PyTorch | 自己构建一次 |
|---|---|---|
| 理解 Module 的生命周期 | 表面的 | 深入的 |
| 知道为什么需要 zero_grad | 背的 | 懂的 |
| 能调试训练循环中的梯度问题 | 很难 | 直观的 |
| 迁移到其他框架（JAX、TensorFlow） | 重新学 | 秒通 |

你花 2 小时构建了这个迷你框架，后续学习任何深度学习框架的时间都会从"学会用法"缩短到"只会差异"。

---

## 3. 从零实现

这个框架的核心抽象只有 5 个：Tensor、Module、Optimizer、Loss、DataLoader。我们逐步构建。

### 第 1 步：Tensor 与自动微分

Tensor 是框架中最基础的数据结构。它存储数据和梯度，并支持自动微分。

```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, (int, float)):
            data = [data]
        self.data = list(data)
        self.requires_grad = requires_grad
        self.grad = [0.0] * len(self.data)
        self._backward_fn = None

    def backward(self, grad=None):
        if grad is None:
            grad = [1.0] * len(self.data)
        if not self.requires_grad:
            return
        for i in range(len(self.data)):
            self.grad[i] += grad[i]
        if self._backward_fn is not None:
            self._backward_fn(grad)

    def zero_grad(self):
        self.grad = [0.0] * len(self.data)
```

自动微分操作定义了如何反向传播。以线性层（矩阵-向量乘法）为例，它是训练的核心：

```python
def linear_forward(input_vec, weight_rows, bias):
    """线性层前向传播：W @ x + b（带自动微分）"""
    fan_out = len(weight_rows)
    fan_in = len(input_vec)

    # 前向：计算 W @ x + b
    result_data = []
    for i in range(fan_out):
        val = sum(weight_rows[i].data[j] * input_vec.data[j]
                  for j in range(fan_in))
        val += bias.data[i]
        result_data.append(val)

    requires_grad = (input_vec.requires_grad
                     or any(w.requires_grad for w in weight_rows)
                     or bias.requires_grad)
    c = Tensor(result_data, requires_grad=requires_grad)

    if not requires_grad:
        return c

    # 定义反向传播规则（沿三条路径）
    def _backward(grad):
        # 1. 输入梯度：d(loss)/d(input[j])
        if input_vec.requires_grad:
            input_grad = [0.0] * fan_in
            for i in range(fan_out):
                for j in range(fan_in):
                    input_grad[j] += grad[i] * weight_rows[i].data[j]
            input_vec.backward(input_grad)

        # 2. 权重梯度：d(loss)/d(W[i][j]) = grad[i] * input[j]
        for i in range(fan_out):
            if weight_rows[i].requires_grad:
                for j in range(fan_in):
                    weight_rows[i].grad[j] += grad[i] * input_vec.data[j]

        # 3. 偏置梯度：d(loss)/d(bias[i]) = grad[i]
        if bias.requires_grad:
            for i in range(fan_out):
                bias.grad[i] += grad[i]

    c._backward_fn = _backward
    return c
```

为什么自动微分这么设计？每个算子只关心**自己**的局部梯度。linear_forward 知道如何计算 W @ x + b 对 x、W、b 的梯度，它不需要知道整个网络的形状。这就是模块化 —— 每个算子独立实现，组合成完整网络。

### 第 2 步：Module 基类

```python
class Module:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
```

这个基类定义了所有层（Linear、ReLU、Sigmoid、Dropout 等）的统一接口。`parameters()` 让优化器只需要知道"有哪些参数需要更新"而不需要知道网络结构。`train()/eval()` 让 Dropout、BatchNorm 等层可以在两种模式间切换。

### 第 3 步：Linear 层

```python
class Linear(Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        std = math.sqrt(2.0 / fan_in)  # Kaiming 初始化
        self.weight_rows = [
            Tensor([random.gauss(0, std) for _ in range(fan_in)],
                   requires_grad=True)
            for _ in range(fan_out)
        ]
        self.bias = Tensor([0.0] * fan_out, requires_grad=True)

    def forward(self, x):
        return linear_forward(x, self.weight_rows, self.bias)

    def parameters(self):
        return self.weight_rows + [self.bias]
```

初始化方法的选择很重要。使用 Kaiming 初始化（`√(2/fan_in)`）适配 ReLU 激活函数，让每一层的输出方差保持稳定。如果用 Xavier 初始化，前向传播时深层网络的激活值会逐渐萎缩到 0。

### 第 4 步：激活函数与正则化

ReLU 和 Sigmoid 作为 Module 实现，每个都缓存反向传播需要的信息：

```python
class ReLU(Module):
    def forward(self, x):
        return relu(x)  # 使用自动微分操作


class Sigmoid(Module):
    def forward(self, x):
        return sigmoid(x)  # 使用自动微分操作
```

Dropout 训练时随机丢弃，缩放保留元素；评估时直接通过：

```python
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        scale = 1.0 / (1.0 - self.p)
        self._mask = [
            0.0 if random.random() < self.p else scale
            for _ in x.data
        ]
        # ... 使用掩码计算输出，并记录自动微分
```

每个自动微分操作（`relu`、`sigmoid`）都定义了自己的反向传播规则，这样 Module 的 `forward()` 调用这些操作时，自动微分系统会自动构建计算图。

### 第 5 步：Sequential 容器

```python
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._modules = list(modules)

    def forward(self, x):
        for module in self._modules:
            x = module.forward(x)
        return x

    def parameters(self):
        params = []
        for module in self._modules:
            params.extend(module.parameters())
        return params
    # train/eval/zero_grad 同样递归传播到子模块
```

Sequential 使用组合模式：它本身是一个 Module，但它内部包含多个 Module。前向传播时按顺序执行，参数收集时递归遍历所有子模块。自动微分保证了反向传播无需在 Sequential 中手动实现。

### 第 6 步：损失函数

```python
class BCELoss:
    def __call__(self, predicted, target):
        n = len(predicted)
        eps = 1e-7
        loss_val = 0.0
        pred_grads = [0.0] * n

        for i in range(n):
            p = max(eps, min(1 - eps, predicted.data[i]))
            t = target[i] if isinstance(target, list) else target
            loss_val += -(t * math.log(p) + (1 - t) * math.log(1 - p))
            pred_grads[i] = (-t / p + (1 - t) / (1 - p)) / n

        result = Tensor([loss_val / n],
                        requires_grad=predicted.requires_grad)

        if predicted.requires_grad:
            def _backward(grad):
                scaled_grads = [g * grad[0] for g in pred_grads]
                predicted.backward(scaled_grads)
            result._backward_fn = _backward

        return result
```

损失函数有两个职责：计算标量损失值、定义梯度反向传播到预测值的规则。注意 `predicted.backward(scaled_grads)` 这行 —— 它触发了整个计算图的反向传播链。

### 第 7 步：优化器

SGD 和 Adam 优化器接收参数列表，使用梯度更新参数值：

```python
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.params = list(parameters)
        self.lr = lr

    def step(self):
        for param in self.params:
            for i in range(len(param)):
                param.data[i] -= self.lr * param.grad[i]

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
```

Adam 在此基础上加入动量（一阶矩）和梯度平方（二阶矩）的指数移动平均。

优化器不知道网络的结构 —— 它只看到一个扁平的参数列表。这保持了框架的模块化。

### 第 8 步：DataLoader

```python
class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.data[i] for i in batch_indices]
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            yield inputs, targets
```

分批处理有两个原因：一是大数据集无法一次性装入内存；二是小批次梯度下降的噪声有助于逃离局部最小值。

### 第 9 步：完整的训练循环

```python
for epoch in range(epochs):
    for batch_inputs, batch_targets in loader:
        for x_list, t_list in zip(batch_inputs, batch_targets):
            # 前向传播（自动建立计算图）
            pred = model.forward(Tensor(x_list))
            loss = criterion(pred, t_list)

            # 反向传播与参数更新
            optimizer.zero_grad()
            loss.backward()    # 自动微分遍历整个计算图
            optimizer.step()   # 使用梯度更新参数
```

训练循环的核心规则：

1. **前向传播**：依次通过各层，自动建立计算图
2. `zero_grad()`：清零上一次迭代的梯度
3. `loss.backward()`：自动微分系统遍历计算图，填入每个参数的梯度
4. `optimizer.step()`：使用梯度更新参数

步骤顺序不能错。如果把 `zero_grad()` 放在 `backward()` 之后，梯度永远不会清零 —— 每次迭代的梯度会不断累积。

---

## 4. 工业工具

### 4.1 PyTorch 等价实现

你刚构建的框架可以与 PyTorch 逐行对应：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === 你的框架 ===                   # === PyTorch ===
model = Sequential(                   model = nn.Sequential(
    Linear(2, 16),                        nn.Linear(2, 16),
    ReLU(),                               nn.ReLU(),
    Linear(16, 8),                        nn.Linear(16, 8),
    ReLU(),                               nn.ReLU(),
    Linear(8, 1),                         nn.Linear(8, 1),
    Sigmoid(),                            nn.Sigmoid(),
)                                       )

criterion = BCELoss()                  criterion = nn.BCELoss()
optimizer = Adam(params, lr=0.01)      optimizer = torch.optim.Adam(params, lr=0.01)

for epoch in range(epochs):            for epoch in range(epochs):
    for x, t in loader:                    for x, t in dataloader:
        optimizer.zero_grad()                  optimizer.zero_grad()
        pred = model(x)                        pred = model(x)
        loss = criterion(pred, t)              loss = criterion(pred, t)
        loss.backward()                        loss.backward()
        optimizer.step()                       optimizer.step()
```

结构一模一样。区别是 PyTorch 自动处理了 GPU 加速，并且在生产环境中经过数万小时的验证和优化。但骨架是一样的。

### 4.2 PyTorch 的核心设计差异

| 功能 | 你的框架 | PyTorch |
|------|---------|---------|
| 自动微分 | 每个算子手动定义 backward | 自动构建完整计算图 |
| GPU 支持 | 无（纯 CPU）| CUDA / ROCm 原生支持 |
| 梯度模式 | eager（即时计算）| eager + JIT 编译 |
| 运行时 | Python | C++ 后端 + Python 前端 |
| 分布式训练 | 无 | 原生支持 DDP / FSDP |
| 数据格式 | 1D 列表 | N 维 Tensor（strided）|
| 速度 | ~1000 samples/s | ~1000000 samples/s |
| 代码量 | ~550 行 | ~200 万行 |

### 4.3 其他深度学习框架对比

```python
# TensorFlow / Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_dim=2),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=100, batch_size=32)

# JAX + Haiku
def net_fn(x):
    mlp = hk.Sequential([
        hk.Linear(16), jax.nn.relu,
        hk.Linear(8),  jax.nn.relu,
        hk.Linear(1),  jax.nn.sigmoid,
    ])
    return mlp(x)
```

所有框架都使用相同的抽象：层、激活函数、损失函数、优化器。学会了一个，其他的只是语法差异。

---

## 5. 知识连线

本课构建的迷你框架是深度学习从零到工业应用的关键桥梁：

- **阶段 10（从零构建大语言模型）**：你将直接在这个框架之上构建 GPT —— Token 嵌入层、多层 Transformer、最终的 LM Head。框架的 Module 抽象保持不变，只是层变得更复杂
- **阶段 11（LLM 工程）**：微调、部署、推理优化 —— 所有这些工程实践都建立在本课的框架抽象之上。理解了 Module 和 Optimizer 如何交互，你就能理解 LoRA、ZeRO、混合精度训练为什么有效
- **阶段 12（多模态 AI）**：跨模态模型仍然使用同样的框架抽象 —— 编码器是 Module，优化器是 Adam，数据加载是 DataLoader。架构变了，但训练循环的骨架不变

---

## 6. 工程最佳实践

### 6.1 训练循环的模式

```python
# 标准 PyTorch 训练循环（所有项目通用）
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
```

这个模式适用于任何深度学习项目 —— 从 MNIST 分类到 Llama 预训练。

### 6.2 PyTorch 中的实际使用

```python
# PyTorch 2.0+ 推荐模式
import torch

model = nn.Sequential(...).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# compile：将模型编译为更高效的算子融合
model = torch.compile(model)

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # 混合精度训练
            loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
```

### 6.3 中文训练场景的特别建议

- **数据集重复的模式**：XOR 只有 4 个样，我们的框架通过重复 100 次获得 400 个样本。实际项目中，数据增强（翻折、旋转、加噪）比简单重复更有效
- **损失不下降时**：先检查梯度是否非零。在你的框架中，可以在 `backward()` 后打印 `param.grad` 确认梯度流是否正常
- **框架不可知的学习**：不要把自己绑定在一个框架上。理解了本课的核心抽象（Module/Optimizer/Loss/DataLoader），你在任何框架中都能用同一种思维模式工作

### 6.4 踩坑经验

- **`model.parameters()` 返回的是同一个对象**：确保在创建优化器后仍能访问参数。PyTorch 中 `model.parameters()` 每次调用返回同一个 Parameter 对象列表，但在你的框架中 `Tensor` 是可变对象，必须保持引用一致
- **`zero_grad()` 必须在每次 `step()` 之前调用**：如果忘记清零，梯度会累积，导致参数更新过大。为什么 PyTorch 不自动清零？为了支持梯度累积（gradient accumulation）—— 可以在多个小批次上累积梯度，模拟更大的批次大小
- **`requires_grad=True` 的一致性**：如果你的损失 Tensor 没有 `requires_grad=True`，调用 `.backward()` 不会传播任何梯度。这是调试中最容易忽略的问题之一
- **使用正确姿势调用 `model.eval()`**：评估推理时一定要调用 `model.eval()`，否则 Dropout 会随机丢弃结果。`torch.no_grad()` 也很重要，可以关闭梯度追踪、节省显存

---

## 7. 常见错误

### 错误 1：损失 Tensor 错过 `requires_grad`

**现象：** loss 从第一个轮次开始就完全不变，每一步看到的都是相同的损失值。

**原因：** 损失函数返回的 Tensor 没有设置 `requires_grad=True`，`backward()` 检查到 `requires_grad=False` 后直接返回，不触发任何梯度传播。

**修复：**

```python
# ❌ 错误写法
result = Tensor([loss_val / n])  # requires_grad 默认为 False

# ✓ 正确写法
result = Tensor([loss_val / n], requires_grad=predicted.requires_grad)
```

### 错误 2：优化器的动量/方差共享所有参数元素

**现象：** 训练缓慢，loss 下降极慢，甚至完全不动。

**原因：** Adam 优化器中每个参数元素的动量和二阶矩应该是独立的。如果用一个标量代表整个参数的动量，每个元素会互相覆盖。

**修复：**

```python
# ❌ 错误的一维缓存
self.m = [0.0] * len(parameters)
# 对 param = [8 个元素]，用 self.m[idx] = scalar 覆盖所有元素

# ✓ 正确的二维缓存
self.m = [[0.0] * len(p) for p in self.params]
# 对 param = [8 个元素]，用 self.m[idx][i] = scalar 独立维护
```

### 错误 3：`zero_grad` 和 `backward` 顺序搞反

**现象：** 训练开始后 loss 振荡越来越剧烈，最终发散。

**原因：** 如果 `backward()` 在 `zero_grad()` 之前调用，梯度在上一个轮次的基础上持续累积。几个轮次后，参数更新量以数量级放大，直接发散。

**修复：**

```python
# ❌ 错误顺序
loss.backward()
optimizer.zero_grad()  # 清掉了刚算好的梯度
optimizer.step()       # 无梯度可用

# ✓ 正确顺序
optimizer.zero_grad()  # 清零上一次的梯度
loss.backward()        # 计算当前梯度
optimizer.step()       # 使用当前梯度更新参数
```

### 错误 4：推理时忘记调用 `model.eval()`

**现象：** 同一批数据在训练和推理时输出不同，甚至准确率差异巨大。

**原因：** 没有切换到评估模式时，Dropout 仍然在随机丢弃神经元，导致每次前向传播的结果都不同。

**修复：**

```python
# ❌ 错误
predictions = model(test_data)  # Dropout 还在丢神经元

# ✓ 正确
model.eval()
predictions = model(test_data)  # Dropout 不生效
model.train()  # 恢复训练模式（如果需要继续训练）
```

### 错误 5：前向传播中重复使用同一 Tensor 导致梯度冲突

**现象：** 梯度值异常大或异常小，训练不稳定。

**原因：** 如果你的计算图中同一个 Tensor 被用于多个后续操作，它的 `_backward_fn` 会被覆盖。`backward()` 会沿着最后一次设置的分支传播，丢失了其他路径的梯度分量。

**修复：** Tensor 的 `backward()` 方法采用了累加策略而不是覆盖 —— 调用 `backward(grad)` 时梯度和已有的 `self.grad` 累加。但在有分支的计算图中，你需要确保所有路径都被正确追踪。在 PyTorch 中这由完整的 autograd 引擎自动处理。

---

## 8. 面试考点

### Q1：PyTorch 中 `nn.Module` 是什么？它的三个核心方法是什么？（难度：⭐⭐）

**参考答案：**
`nn.Module` 是所有神经网络模块的基类。它的三个核心方法是：
1. `forward()`：定义前向传播计算
2. `parameters()`：返回模块中所有可训练参数（递归包含子模块的参数）
3. `train()/eval()`：切换训练/评估模式

Module 本身不实现反向传播 —— PyTorch 的 autograd 会自动从 forward 构建计算图并处理反向传播。你的框架中每个 Module 直接使用自动微分操作，原理相同。

### Q2：为什么 `optimizer.zero_grad()` 是一个单独的方法？为什么不在 `step()` 中自动清零？（难度：⭐⭐）

**参考答案：**
将 `zero_grad()` 和 `step()` 分离是为了支持**梯度累积（gradient accumulation）**。在某些场景下，受限于 GPU 显存，无法使用大的批量大小。这时可以在多个小批次上分别前向传播和反向传播，梯度累积后只执行一次参数更新：

```python
optimizer.zero_grad()
for i, mini_batch in enumerate(mini_batches):
    loss = model(mini_batch)
    loss = loss / len(mini_batches)  # 平均梯度
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

如果 `step()` 自动清零梯度，这种模式就无法实现。

### Q3：手写一个简化版自动微分系统的核心逻辑（难度：⭐⭐⭐）

**参考答案：**
自动微分的核心是计算图。每个 Tensor 记录自己是怎么被计算出来的，`backward()` 时沿着图的反向传播梯度：

```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None

    def backward(self, gradient=None):
        if gradient is None:
            gradient = Tensor([1.0])  # 标量损失的初始梯度
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad = self.grad + gradient  # 累积
        self._backward()  # 触发父节点的梯度计算

# 使用示例：
# x = Tensor([2.0], requires_grad=True)
# y = x * 3 + 1   # y._backward = lambda: x.backward(y.grad * 3)
# y.backward()    # x.grad = [3.0]
```

关键设计：每个操作（如 `*`、`+`）创建的新 Tensor 绑定了一个 `_backward` 闭包，该闭包知道如何将当前梯度传递给操作数。

### Q4：为什么 XOR 问题对单层感知机来说是"不可解"的？（难度：⭐⭐）

**参考答案：**
XOR 是一个线性不可分问题。单层感知机只能形成一条直线决策边界，但 XOR 需要两条直线（或者等价地，一个弯曲的决策边界）。Minsky 和 Papert 在 1969 年证明了这一限制，直接导致了第一次"人工智能寒冬"。

多层感知机（MLP）通过非线性激活函数（如 ReLU）和隐藏层解决了这个问题。单层感知机的决策边界是一条直线，而 MLP 可以形成任意复杂的决策边界。在你的框架中，4 层 MLP（2->16->8->1）可以完美解决 XOR，验证了多层网络的表达能力。

### Q5：设计一个自定义层的 Module，需要考虑哪些生命周期？（难度：⭐⭐⭐）

**参考答案：**
1. **初始化**：在 `__init__` 中创建所有子模块和参数
2. **前向传播**：在 `forward` 中定义计算
3. **参数注册**：确保 `parameters()` 返回新创建的可训练参数
4. **训练/评估切换**：如果层在训练和评估时有不同行为（如 Dropout、BatchNorm），正确处理 `training` 标志
5. **设备管理**（PyTorch）：使用 `to(device)` 移动参数
6. **梯度清零**：在 `zero_grad()` 中递归清零所有参数

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Module | "就是一个层" | 所有可学习组件的基类 —— 封装了 forward、parameters、train/eval 的生命周期管理 |
| Sequential | "按顺序堆叠层" | 一个容器 Module，前向从左到右执行子模块，反向从右到左传播梯度 |
| 自动微分 (Autograd) | "框架自动算梯度" | 在计算图上记录每个操作的反向传播规则，调用 `backward()` 时自动遍历图计算所有参数的梯度 |
| 前向传播 (Forward pass) | "运行网络" | 输入数据依次通过所有层，计算输出（同时自动构建计算图） |
| 反向传播 (Backward pass) | "计算梯度" | 从损失出发，沿计算图反向传播梯度，填入每个参数的 `.grad` 属性 |
| 计算图 (Computation Graph) | "框架知道怎么算梯度" | 前向传播过程中自动构建的有向无环图 —— 节点是 Tensor，边是操作，反向遍历它即可计算所有梯度 |
| 参数 (Parameters) | "可训练的权重" | 网络中所有优化器可以更新的值 —— 包括权重矩阵和偏置向量 |
| 优化器 (Optimizer) | "更新权重的东西" | 使用梯度更新参数的算法 —— 实现 SGD、Adam 等更新规则 |
| DataLoader | "喂数据的东西" | 一个迭代器，将数据集分批并可选打乱顺序，供训练循环逐批消费 |
| 训练模式 vs 评估模式 | "`model.train()` / `model.eval()`" | 一个标志位，控制 Dropout 是否随机丢弃、BatchNorm 是否使用批统计量 |

---

## 📚 小结

你从零构建了一个完整的深度学习框架 —— 包括 Tensor 自动微分系统、Module 抽象层、Linear/ReLU/Sigmoid/Dropout 层、Sequential 容器、BCELoss/MSELoss 损失函数、SGD/Adam 优化器，以及 DataLoader 数据加载器。总共约 550 行纯 Python，零第三方依赖。

这个框架成功训练了一个 4 层 MLP 解决了 XOR 异或问题 —— 这个 1969 年曾被证明单层感知机无法解决的问题。每个组件的设计都直接映射到 PyTorch 的对应抽象。

下一课我们将进入计算机视觉领域 —— 用卷积神经网络（CNN）处理图像数据。CNN 的核心 Module 没变，但 Linear 层变成了卷积层，全连接变成了局部连接。

---

## ✏️ 练习

1. 【理解】用自己的话解释自动微分的反向传播机制。如果 `x = Tensor([1, 2], requires_grad=True)` 经过两个操作：`y = x * 2` 然后 `z = y + 1` 然后 `z.backward([1, 1])`，画出计算图并说明每个 Tensor 的梯度值。

2. 【实现】在你的框架中添加 `SoftmaxCrossEntropyLoss` 类用于多分类任务。实现 softmax 前向计算和交叉熵损失的反向传播。在 3 类螺旋数据集上测试。

3. 【实现】给 `Sequential` 添加 `save()` 和 `load()` 方法，将模型权重序列化为 JSON 文件并加载回来。验证加载后的模型在相同输入上产生相同输出。

4. 【实验】在你的框架中实现权值衰减（L2 正则化）。在 Adam 中添加 `weight_decay` 参数。对比 `decay=0` 和 `decay=0.01` 在 XOR 上的训练曲线差异。

5. 【思考】如果将训练循环中的 `optimizer.zero_grad()` 移到 `optimizer.step()` 内部自动调用，会丧失什么能力？在什么场景下你需要手动控制梯度累积？

6. 【实现】修改 DataLoader 和训练循环，从当前逐样本梯度更新改为真正的批量梯度累积：在一个批次内累积所有样本的梯度，除以批次大小后统一更新。对比两种方式的收敛速度。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 深度学习迷你框架 | `code/main.py` | 约 550 行纯 Python，包含 Tensor 自动微分、Module 抽象、全连接网络、损失函数、优化器、DataLoader |
| 框架架构师提示词 | `outputs/prompt-framework-builder.md` | 用框架抽象设计神经网络架构的可复用提示词 |

---

## 📖 参考资料

1. [论文] Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library". NeurIPS, 2019. https://arxiv.org/abs/1912.01703
2. [官方文档] PyTorch `nn.Module`: https://pytorch.org/docs/stable/nn.html#module
3. [官方文档] PyTorch `torch.optim`: https://pytorch.org/docs/stable/optim.html
4. [官方文档] PyTorch `DataLoader`: https://pytorch.org/docs/stable/data.html
5. [GitHub] micrograd — Andrej Karpathy 的微型自动微分引擎: https://github.com/karpathy/micrograd
6. [GitHub] tiny-dnn — 头文件 C++ 深度学习框架: https://github.com/tiny-dnn/tiny-dnn
7. [书籍] Chollet. 《Deep Learning with Python, Second Edition》. Manning, 2021. 第 3 章涵盖 Keras 内部实现的 Module/Layer 抽象
8. [论文] Kingma & Ba. "Adam: A Method for Stochastic Optimization". ICLR, 2015. https://arxiv.org/abs/1412.6980

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
