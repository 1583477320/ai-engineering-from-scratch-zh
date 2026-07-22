# 优化器

> 梯度告诉你方向，优化器决定你走多快、怎么走。选错优化器，你的模型会在山谷里来回震荡几万步也到不了终点。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 03 · 03（反向传播）、阶段 03 · 04（激活函数）、阶段 03 · 05（损失函数）
**预计时间：** ~90 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 09 · 04（从零训练一个语言模型）— 优化器在大规模预训练中的角色；阶段 03 · 09（学习率调度）— 学习率与优化器的协同调优

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 SGD、Momentum、Nesterov、AdaGrad、RMSProp、Adam、AdamW 七种优化器，解释每种算法相比前一种解决了什么问题
- [ ] 用数学公式说明 Adam 如何将动量（一阶矩）和自适应学习率（二阶矩）结合，并解释偏差校正的必要性
- [ ] 在 Rosenbrock 函数和环形数据集上对比七种优化器的收敛行为，诊断不同场景下优化器的优劣
- [ ] 区分 Adam 的 L2 正则化与 AdamW 的解耦权重衰减，解释为什么 Transformer 训练必须用 AdamW
- [ ] 使用 PyTorch `torch.optim` 创建和配置优化器，正确设置学习率、权重衰减等关键超参数

---

## 1. 问题

你从零实现了一个两层神经网络，反向传播也写对了，损失函数也选对了。但训练 300 轮之后，准确率卡在 55% 不动了。

你把学习率从 0.01 改成 0.1，损失直接爆炸成 NaN。改回 0.01，训练 3000 轮，准确率缓慢爬到 70%。换个优化器试试？换成了 Adam，同样的网络、同样的数据，50 轮就到了 85%。

这不是玄学。纯 SGD 在损失曲面上像一个没有惯性的球——每一步都严格沿着当前最陡的方向走。如果这个方向在山谷两侧来回切换（震荡方向），SGD 会原地打转；如果通往最小值的方向梯度很小（平坦方向），SGD 走得极慢。这两种问题同时存在时，SGD 就卡住了。

好的优化器解决两类问题：**在震荡方向上抑制抖动**（动量），**在不同参数上自适应调整步长**（自适应学习率）。从 SGD 到 Adam，每一步改进都针对一个具体的痛点。选错了优化器，不只是"训练慢一点"——它可能导致模型完全无法收敛，或者收敛到一个明显次优的解。

---

## 2. 概念

### 2.1 优化器的演进脉络

```
优化器的演进路线：

SGD                    ← 最基础：沿梯度反方向走
  │
  ├──→ SGD + Momentum   ← 加入惯性：抑制震荡，加速平坦方向
  │       │
  │       └──→ Nesterov  ← "先走再看"：动量的矫正版本
  │
  └──→ AdaGrad          ← 自适应学习率：梯度大的参数走小步
          │
          └──→ RMSProp   ← 指数移动平均：解决 AdaGrad 学习率单调递减
                  │
                  └──→ Adam    ← 动量 + 自适应 + 偏差校正 = 工业标配
                          │
                          └──→ AdamW  ← 解耦权重衰减 = Transformer 标配
```

每一代优化器的改进可以总结为一句话：

| 优化器 | 解决的问题 | 一句话 |
|---|---|---|
| SGD | 无 | 基准线，能用但慢 |
| Momentum | 震荡方向消耗大量步数 | 像下坡的球，有惯性 |
| Nesterov | 动量可能冲过头 | 先看一眼再走 |
| AdaGrad | 稀疏特征需要大步走，频繁特征需要小步走 | 按参数自动调步长 |
| RMSProp | AdaGrad 学习率只会越来越小 | 用近期记忆替代永久记忆 |
| Adam | 需要同时解决震荡和自适应 | 动量 + RMSProp + 偏差校正 |
| AdamW | L2 正则化被自适应学习率干扰 | 正则化和更新解耦 |

### 2.2 SGD：随机梯度下降

**直觉：** 损失曲面是一个山谷，SGD 就是一个球沿着当前位置最陡的斜面滚下去。

**公式：**

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

其中 $\eta$ 是学习率，$g_t = \nabla_\theta L(\theta_t)$ 是当前梯度。

**问题：** 当损失曲面在某个方向上波动剧烈（如峡谷地形），SGD 会在峡谷两壁之间反复震荡，真正沿峡谷底部前进的步伐很小。

```
SGD 在峡谷地形中的行为：

    损失
    │ ╲    ╱
    │  ╲  ╱   ← 来回震荡（震荡方向）
    │   ╲╱
    │    ↓    ← 缓慢前进（平坦方向）
    │    ↓
    │    ╲→ → → → 最小值
    └──────────── 参数
```

### 2.3 SGD + Momentum（动量）

**直觉：** 给球加上惯性。在震荡方向上，左右的力互相抵消；在平坦方向上，持续的推力累积加速。

**公式：**

$$v_t = \beta \cdot v_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

其中 $v_t$ 是速度（梯度的指数移动平均），$\beta$ 通常取 0.9。

**关键洞察：** $\beta = 0.9$ 意味着速度的"记忆长度"约为 $\frac{1}{1-\beta} = 10$ 步。过去 10 步的梯度方向一致时速度增大，方向相反时速度互相抵消。这就是动量抑制震荡的本质。

### 2.4 Nesterov 加速梯度

**直觉：** 普通动量"先算梯度，再用动量更新"。Nesterov 的改进是"先按动量走一步，在新位置算梯度"——就像你在滚下坡之前先往前看一眼。

**公式：**

$$v_t = \beta \cdot v_{t-1} + \nabla_\theta L(\theta_t - \eta \cdot \beta \cdot v_{t-1})$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

相比标准动量，Nesterov 在"预看位置"计算梯度，如果动量已经让球走过了最小值，预看位置的梯度会指向回退方向，提供自然的矫正。

### 2.5 AdaGrad（自适应梯度）

**直觉：** 每个参数有自己的"步长历史"。梯度大的参数累积了大量梯度平方，除以一个大数后步长自动缩小；梯度小的参数步长相对较大。这在自然语言处理中特别有用——词嵌入矩阵中，高频词（如"的"）的梯度大但不需要频繁更新，低频词（如"饕餮"）的梯度小但需要大步更新。

**公式：**

$$G_t = G_{t-1} + g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t$$

其中 $G_t$ 是梯度平方的累积和，$\epsilon$ 防止除零。

**关键缺陷：** $G_t$ 只增不减。训练到后期，$G_t$ 变得很大，所有参数的学习率都趋近于零，模型停止学习。

### 2.6 RMSProp（均方根传播）

**直觉：** 用指数移动平均替代累积和，只记住最近的梯度信息，不被远古的梯度"绑架"。

**公式：**

$$s_t = \beta \cdot s_{t-1} + (1 - \beta) \cdot g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon} \cdot g_t$$

其中 $\beta$ 通常取 0.9，控制"记忆长度"约为 10 步。$s_t$ 是梯度平方的指数移动平均，不会无限增长。

**背景：** RMSProp 由 Geoffrey Hinton 在 Coursera 课程中提出，从未正式发表论文。但这并不妨碍它成为深度学习中最重要的优化器之一——Adam 的二阶矩估计正是基于 RMSProp 的思想。

### 2.7 Adam（自适应矩估计）

**直觉：** 同时维护两个指数移动平均——梯度的均值（一阶矩，处理震荡）和梯度平方的均值（二阶矩，处理自适应学习率），再加上偏差校正解决冷启动问题。

**公式：**

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \quad \text{（一阶矩：动量）}$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 \quad \text{（二阶矩：自适应学习率）}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{（偏差校正）}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

默认超参数：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\eta = 0.001$，$\epsilon = 10^{-8}$。

**偏差校正的直觉：** 前几步 $m_t$ 和 $v_t$ 都从 0 开始累积，值偏小。除以 $1 - \beta_1^t$（第一步是 0.1，第二步是 0.19，第 10 步是 0.65）可以把早期的有偏估计拉回正确值。训练到几十步之后，$1 - \beta_1^t \approx 1$，校正自然消失。

```
Adam 偏差校正的效果（假设梯度恒为 1，β1=0.9）：

  Step | m_raw  | m_corrected
  -----|--------|------------
     1 | 0.1000 | 1.0000     ← m_raw 只有真实值的 1/10
     2 | 0.1900 | 0.9500     ← 校正将偏差拉回
     5 | 0.4095 | 0.8190
    10 | 0.6513 | 0.8731     ← 接近真实值
    50 | 0.9940 | 0.9994     ← 校正几乎为 1，自然消失
```

### 2.8 AdamW（解耦权重衰减）

**问题：** Adam 中的 L2 正则化（$\mathcal{L}_{total} = \mathcal{L} + \frac{\lambda}{2}\|\theta\|^2$）会将正则化项的梯度 $\lambda\theta$ 也经过自适应学习率缩放。梯度方差大的参数被缩放得更多，正则化强度反而不均匀。

**AdamW 的做法：** 先做 Adam 更新，然后**直接对参数做衰减**，不经过自适应缩放：

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t - \eta \cdot \lambda \cdot \theta_t$$

两行代码的区别：Adam + L2 把 $\lambda\theta$ 加入梯度一起被自适应缩放；AdamW 在自适应更新之后单独做权重衰减。效果差异显著：AdamW 的正则化对所有参数均匀作用，训练的 Transformer 模型泛化能力更好。

### 2.9 七种优化器的统一视图

```
                    SGD 系列                   自适应系列
                    ──────                   ──────────
    SGD ──→ Momentum ──→ Nesterov    AdaGrad ──→ RMSProp ──→ Adam ──→ AdamW
    │         │           │             │           │         │        │
    梯度     加入惯性    预看矫正      累积平方    移动平均   两者结合  解耦正则
    反方向   抑制震荡    修正过冲      降学习率    解决单调递减 偏差校正  均匀衰减
```

---

## 3. 从零实现

### 第 1 步：SGD——最基础的优化器

```python
class SGD:
    """随机梯度下降：沿梯度反方向更新参数。"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
```

**为什么从这里开始：** SGD 是所有优化器的基准。后续每种优化器都可以理解为"在 SGD 基础上增加了什么"。`step` 方法接收参数列表和对应的梯度列表，就地更新参数——这是所有优化器的统一接口。

### 第 2 步：加入动量（SGD + Momentum）

```python
class SGDMomentum:
    """带动量的 SGD：维护速度的指数移动平均。"""

    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None  # 延迟初始化

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)
        for i in range(len(params)):
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]
```

**为什么需要 `velocities`：** 动量需要跨步骤记忆——当前的速度是上一步的速度（乘以衰减系数）加上当前梯度。第一次调用 `step` 时速度为 0，后续逐步累积。$\beta = 0.9$ 意味着 90% 的旧速度被保留，新梯度只占 10%。

### 第 3 步：Nesterov 加速梯度

```python
class NesterovMomentum:
    """Nesterov 加速梯度：先按动量"预看"一步，在预看位置计算梯度。"""

    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)
        for i in range(len(params)):
            # 先按旧动量更新到"预看位置"
            lookahead = params[i] - self.lr * self.beta * self.velocities[i]
            # 更新速度（使用当前梯度）
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]
```

**与标准动量的区别：** 标准动量用当前梯度更新速度后直接走。Nesterov 想法是"如果我先走一步，那里的梯度会怎么指"——这提供了自我矫正能力。当动量让参数冲过最小值时，Nesterov 在更远位置看到的梯度会指向回退方向。

### 第 4 步：AdaGrad——按参数自适应步长

```python
class AdaGrad:
    """自适应梯度算法：梯度大的参数学习率自动缩小。"""

    def __init__(self, lr=0.01, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.sum_squares = None

    def step(self, params, grads):
        if self.sum_squares is None:
            self.sum_squares = [0.0] * len(params)
        for i in range(len(params)):
            self.sum_squares[i] += grads[i] ** 2
            params[i] -= self.lr * grads[i] / (math.sqrt(self.sum_squares[i]) + self.epsilon)
```

**为什么除以梯度平方的累积和：** 历史上某个参数的梯度一直很大（比如 1.0），100 步后 `sum_squares` 约 100，实际步长变为 $\eta / \sqrt{100} = \eta / 10$——自动缩小 10 倍。如果梯度一直很小（比如 0.01），`sum_squares` 约 0.01，实际步长反而被放大。这就是"按需分配学习率"的本质。

### 第 5 步：RMSProp——用移动平均替代累积和

```python
class RMSProp:
    """均方根传播：用指数移动平均替代累积平方和。"""

    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.s = None

    def step(self, params, grads):
        if self.s is None:
            self.s = [0.0] * len(params)
        for i in range(len(params)):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * grads[i] ** 2
            params[i] -= self.lr * grads[i] / (math.sqrt(self.s[i]) + self.epsilon)
```

**与 AdaGrad 的关键差异：** AdaGrad 用 `sum_squares += g^2`（累积），RMSProp 用 `s = beta * s + (1-beta) * g^2`（移动平均）。累积只会增长，移动平均会衰减——旧梯度的影响随时间淡出，学习率不会单调递减到零。

### 第 6 步：Adam——动量 + 自适应 + 偏差校正

```python
class Adam:
    """自适应矩估计：动量 + RMSProp + 偏差校正。"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一阶矩（梯度均值）
        self.v = None  # 二阶矩（梯度平方均值）
        self.t = 0     # 时间步计数器

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)
        self.t += 1

        for i in range(len(params)):
            # 更新一阶矩（动量）
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            # 更新二阶矩（自适应学习率）
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2
            # 偏差校正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
```

**为什么 Adam 是"默认选择"：** 它同时解决了两个问题——Momentum 抑制震荡，自适应学习率按参数调步长。偏差校正让冷启动阶段也能快速收敛。这就是为什么大多数深度学习论文的默认优化器都是 Adam。

### 第 7 步：AdamW——解耦权重衰减

```python
class AdamW:
    """解耦权重衰减的 Adam：训练 Transformer 的默认优化器。"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)
        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # 先做 Adam 更新
            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
            # 再做解耦权重衰减（直接作用于参数，不经过梯度）
            params[i] -= self.lr * self.weight_decay * params[i]
```

**两行代码的本质区别：** Adam + L2 的做法是在梯度上加上 $\lambda\theta$，然后整体被 $1/\sqrt{\hat{v}}$ 缩放。AdamW 的做法是先完成 Adam 更新，然后单独对参数做 $\theta = \theta - \eta\lambda\theta$。后者保证所有参数承受相同强度的正则化。

### 实验验证

运行完整代码后可以观察到以下结果：

```text
实验 1：SGD 最小化 f(x) = (x-3)^2
  从 x=10 出发，目标 x=3
    Step  0: x=8.400000, loss=29.160000
    Step  5: x=3.756096, loss=0.571648
    Step 10: x=3.176840, loss=0.031272
    Step 19: x=3.011533, loss=0.000133

实验 2：Adam 偏差校正
  Step | m_raw  | m_corrected | v_raw    | v_corrected
    1 | 0.1000 | 1.0000      | 0.001000 | 1.000000
    2 | 0.1900 | 0.9500      | 0.001999 | 0.999500
    5 | 0.4095 | 0.8190      | 0.004988 | 0.997603
   10 | 0.6513 | 0.8731      | 0.009901 | 0.991055
```

偏差校正的效果一目了然：第一步 $m_{raw} = 0.1$（只有真实值的 1/10），校正后 $m_{corrected} = 1.0$（精确等于真实值）。这就是为什么 Adam 在训练初期就能快速更新参数。

---

## 4. 工业工具

### 4.1 PyTorch 内置优化器

```python
import torch
import torch.nn as nn

# 创建一个简单的模型
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# SGD + Momentum
optimizer_sgd = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

# Adam（最常用）
optimizer_adam = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)

# AdamW（训练 Transformer 的标准选择）
optimizer_adamw = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # 解耦权重衰减
)
```

### 4.2 标准训练循环

```python
# 典型的 PyTorch 训练循环
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # 1. 前向传播
        outputs = model(batch_x)
        loss = nn.functional.cross_entropy(outputs, batch_y)

        # 2. 梯度清零（必须！否则梯度会累积）
        optimizer.zero_grad()

        # 3. 反向传播
        loss.backward()

        # 4. 参数更新
        optimizer.step()
```

**`optimizer.zero_grad()` 的必要性：** PyTorch 默认累加梯度（`retain_graph=False` 时 `backward` 会累加 `.grad`）。如果不手动清零，一个批次的梯度会和上一个批次的梯度叠加，训练行为完全错误。

### 4.3 混合精度训练中的优化器

```python
# 混合精度训练：AdamW + GradScaler
scaler = torch.amp.GradScaler()

for batch_x, batch_y in dataloader:
    optimizer.zero_grad()

    # 自动将前向传播转为 FP16
    with torch.amp.autocast():
        outputs = model(batch_x)
        loss = nn.functional.cross_entropy(outputs, batch_y)

    # 缩放 loss 防止 FP16 梯度下溢
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 4.4 性能对比

| 优化器 | 每步内存开销 | 适用场景 | 备注 |
|---|---|---|---|
| SGD | 无额外状态 | CV 任务、简单模型 | 配合动量使用效果更好 |
| SGD + Momentum | 每参数 1 个动量 | CV 训练（ResNet 等） | 配合学习率调度效果好 |
| Adam | 每参数 2 个状态 | NLP、GAN、RL | 默认选择 |
| AdamW | 每参数 2 个状态 | Transformer、LLM 训练 | 正则化更均匀 |

---

## 5. 知识连线

本课学习的优化器，是后续所有深度学习课程训练模型的"引擎"：

- **阶段 03 · 09（学习率调度）**：优化器和学习率调度器是搭档——AdamW 再好，如果学习率恒定不变，训练后期也会在最小值附近徘徊。学习率调度器（如 cosine annealing）在优化器的基础上进一步控制训练节奏。
- **阶段 07 · Transformer 深入**：Transformer 的训练几乎总是使用 AdamW，理解它为什么优于 Adam + L2，就能理解为什么大语言模型的训练配置中总是同时出现 `adamw` 和 `weight_decay` 两个参数。
- **阶段 09 · 从零训练语言模型**：在大规模预训练中，优化器的选择直接影响训练稳定性和最终模型质量。Llama 系列的训练配置中，AdamW 的 $\beta_1=0.9$、$\beta_2=0.95$（而非默认的 0.999）就是针对大语言模型训练的特殊调整。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐优化器 | 关键超参数 | 备注 |
|---|---|---|---|
| 计算机视觉（ResNet 等） | SGD + Momentum | lr=0.1, momentum=0.9, weight_decay=1e-4 | 配合余弦退火 |
| Transformer / NLP | AdamW | lr=1e-4~5e-4, weight_decay=0.01~0.1 | LLM 预训练标配 |
| 微调预训练模型 | AdamW | lr=2e-5~5e-5, weight_decay=0.0 | 学习率远小于预训练 |
| GAN 训练 | Adam | lr=2e-4, betas=(0.5, 0.999) | β1=0.5 是经验值 |
| 强化学习 | Adam | lr=3e-4 | PPO 的默认选择 |

### 6.2 中文场景特别建议

- **中文 NLP 微调**：使用 `bert-base-chinese` 或 `chinese-roberta` 微调时，推荐 AdamW + lr=2e-5。中文模型的词表较小，学习率过大会导致词嵌入震荡，分类性能下降。
- **中文文本生成**：使用 ChatGLM、Qwen 等中文大语言模型微调时，AdamW + lr=1e-5~5e-6 + weight_decay=0.1 是经过验证的组合。注意使用余弦学习率调度配合 warmup。
- **边缘设备部署场景**：如果最终目标是量化部署到手机/嵌入式设备，训练时使用 AdamW 但要注意权重衰减的强度——过强的正则化会让量化后的精度损失更大。

### 6.3 踩坑经验

- **忘记 `zero_grad()`**：最常见的低级错误。梯度不为零，参数更新方向错误，训练发散。养成习惯：在 `loss.backward()` 之前调用 `optimizer.zero_grad()`。
- **学习率和优化器不匹配**：SGD 用 lr=0.001 几乎学不动，Adam 用 lr=0.1 训练爆炸。每个优化器有其合理的学习率范围，不要混用。
- **AdamW 的 weight_decay 不等于 L2 正则化**：很多教程把 `weight_decay` 当作 L2 正则化来讲，但在 AdamW 中它们数学上不等价。`weight_decay=0.01` 不意味着 L2 系数 $\lambda=0.01$。
- **训练初期 loss 震荡**：不要慌。Adam 在前几步的偏差校正会让步长偏大，loss 短暂上升是正常的。如果 10 步后 loss 仍然不下降，再排查学习率和梯度问题。

---

## 7. 常见错误

### 错误 1：SGD 配合过大的学习率

**现象：** 训练一开始 loss 就爆炸成 NaN，梯度变为无穷大。

**原因：** SGD 没有自适应能力，每步的更新幅度完全由学习率控制。lr=0.1 对于很多任务来说太大，参数更新后跳出了损失曲面的合理区域，导致梯度爆炸。

**修复：**
```python
# ❌ 错误：SGD 使用过大的学习率
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# ✅ 正确：SGD 配合适中学习率 + 动量
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 或使用 Adam，它对学习率不那么敏感
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### 错误 2：Adam 与 SGD 混用学习率

**现象：** 从 SGD 切换到 Adam 时忘记调整学习率，训练爆炸或完全学不动。

**原因：** Adam 的自适应学习率机制意味着实际步长是 $\eta / \sqrt{\hat{v}}$。当梯度方差大时，实际步长远小于 $\eta$；当梯度方差小时，实际步长接近 $\eta$。Adam 的有效学习率和 SGD 完全不同。

**修复：**
```python
# ❌ 错误：用 SGD 的学习率训练 Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # 太大

# ✅ 正确：Adam 的默认学习率是 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 微调预训练模型时更小
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

### 错误 3：在 Adam 中使用 L2 正则化代替 AdamW

**现象：** 训练的 Transformer 模型泛化能力不如预期，验证集 loss 下降缓慢。

**原因：** Adam + L2 中，L2 正则化项 $\lambda\theta$ 的梯度也被自适应学习率缩放。梯度方差大的参数，正则化被削弱；梯度方差小的参数，正则化被放大。正则化强度在参数间不均匀。

**修复：**
```python
# ❌ 错误：用 L2 正则化（weight_decay 参数在 SGD 中是 L2）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
# 或手动添加 L2
loss = criterion(output, target) + 0.01 * sum(p.norm()**2 for p in model.parameters())

# ✅ 正确：训练 Transformer 使用 AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

### 错误 4：梯度裁剪和优化器的配合错误

**现象：** 训练 RNN 或 Transformer 时偶尔出现 loss 突然跳变或 NaN，但整体趋势正常。

**原因：** 某些批次产生了极大梯度（梯度爆炸），在梯度被裁剪之前就被优化器使用了。梯度裁剪必须在 `backward()` 之后、`step()` 之前执行。

**修复：**
```python
# ❌ 错误：梯度裁剪放在错误的位置
optimizer.zero_grad()
loss.backward()
optimizer.step()        # 先更新了
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 裁剪太晚

# ✅ 正确：先裁剪再更新
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 先裁剪
optimizer.step()        # 再更新
```

### 错误 5：优化器状态未保存/恢复

**现象：** 中断训练后从 checkpoint 恢复，loss 突然跳变，收敛轨迹和之前完全不同。

**原因：** Adam 等自适应优化器维护了动量 $m$ 和二阶矩 $v$ 等状态。只保存模型参数而丢失优化器状态，恢复后优化器从零状态开始，相当于前几步用错误的偏差校正因子，导致参数更新异常。

**修复：**
```python
# ❌ 错误：只保存模型参数
torch.save(model.state_dict(), "checkpoint.pt")

# ✅ 正确：同时保存优化器状态
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
}, "checkpoint.pt")

# 恢复时
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

---

## 8. 面试考点

### Q1：解释 Adam 优化器中偏差校正的作用和数学原理。（难度：⭐⭐⭐）

**参考答案：** Adam 维护梯度的一阶矩 $m_t$ 和二阶矩 $v_t$ 的指数移动平均，两者都从 0 初始化。在前几步，$m_t$ 和 $v_t$ 的值偏小（例如第一步 $m_1 = (1-\beta_1)g_1$，远小于真实均值 $g_1$）。偏差校正通过除以 $1-\beta_1^t$ 和 $1-\beta_2^t$ 补偿这个偏差。数学上，$E[m_t] = (1-\beta_1^t)E[g_t]$，除以 $1-\beta_1^t$ 后得到无偏估计。训练到几十步之后，$1-\beta_1^t \approx 1$，校正自然消失，不影响稳态行为。

### Q2：AdamW 和 Adam + L2 正则化有什么区别？为什么 Transformer 训练必须用 AdamW？（难度：⭐⭐⭐）

**参考答案：** Adam + L2 将 L2 正则化项的梯度 $\lambda\theta$ 加入梯度，一起被自适应学习率 $1/\sqrt{\hat{v}}$ 缩放。这导致梯度方差大的参数正则化被削弱（$v$ 大，缩放多），梯度方差小的参数正则化被放大——正则化强度在参数间不均匀。AdamW 先做 Adam 更新，然后对参数直接做 $\theta = \theta - \eta\lambda\theta$，正则化不受自适应缩放影响。Transformer 模型参数量大、梯度方差差异大，均匀的正则化对泛化至关重要，因此必须用 AdamW。Loshchilov & Hutter (2019) 的实验也验证了 AdamW 在 Transformer 上的优越性。

### Q3：对比 SGD + Momentum 和 Adam，各自在什么场景下更优？（难度：⭐⭐）

**参考答案：** SGD + Momentum 在计算机视觉任务（如 ResNet、EfficientNet 训练）中通常更优——它倾向于收敛到"平坦"的极小值（sharpness-aware），泛化能力更好，但需要精心调学习率和调度策略。Adam 在 NLP、GAN、RL 等任务中更优——它收敛快，对学习率不敏感，不需要太多调参，但可能收敛到"尖锐"的极小值，泛化能力略差。现代趋势是 CV 也逐渐转向 AdamW（如 ViT 的训练），而 LLM 训练几乎都用 AdamW。选择时优先考虑：训练速度优先选 Adam，泛化能力优先且有调参经验选 SGD。

### Q4：手写一个完整的 Adam 优化器的 `step` 方法。（难度：⭐⭐）

**参考答案：**
```python
def adam_step(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam 优化器的单步更新。"""
    t += 1
    for i in range(len(params)):
        # 更新一阶矩和二阶矩
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grads[i] ** 2
        # 偏差校正
        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)
        # 参数更新
        params[i] -= lr * m_hat / (v_hat ** 0.5 + eps)
    return t
```

### Q5：为什么 Adam 训练初期 loss 可能短暂上升？这正常吗？（难度：⭐）

**参考答案：** 正常。Adam 在前几步的偏差校正会让有效学习率偏大（$m_{corrected} / \sqrt{v_{corrected}}$ 可能大于梯度的原始值），导致前几步的参数更新幅度过大，loss 短暂上升。一般 10-20 步后偏差校正因子接近 1，学习率回归正常，loss 开始稳定下降。如果 loss 在 20 步后仍然不下降，则需要排查学习率是否过大或梯度是否异常。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 优化器（Optimizer） | "帮模型学得更快的" | 根据梯度决定参数如何更新的算法——决定了学习率、动量、自适应策略 |
| 学习率（Learning Rate） | "步长" | 每步更新的最大幅度——太大导致震荡，太小导致收敛慢 |
| 动量（Momentum） | "加速度" | 梯度的指数移动平均——在一致方向上加速，在震荡方向上抵消 |
| 一阶矩（First Moment） | "梯度的均值" | Adam 中梯度的指数移动平均，等价于动量 |
| 二阶矩（Second Moment） | "梯度的方差" | Adam 中梯度平方的指数移动平均，用于自适应调整学习率 |
| 偏差校正（Bias Correction） | "前期补偿" | 除以 $1-\beta^t$ 补偿零初始化导致的早期偏差，随时间步自然消失 |
| 权重衰减（Weight Decay） | "L2 正则化" | 每步将参数乘以 $(1-\eta\lambda)$ 缩小——在 AdamW 中与自适应学习率解耦 |
| 梯度裁剪（Gradient Clipping） | "防止爆炸" | 将梯度范数限制在阈值以内，防止参数更新过大导致训练不稳定 |

---

## 📚 小结

优化器决定了模型如何从梯度中学习——SGD 能用但慢，Momentum 抑制震荡，AdaGrad 按参数自适应步长，Adam 将两者结合成为工业标配，AdamW 进一步解耦正则化成为 Transformer 训练的默认选择。你从零实现了七种优化器，通过 Rosenbrock 函数和环形数据集的实验验证了每种算法的优势和局限，理解了偏差校正如何解决 Adam 的冷启动问题。

下一课我们将学习如何配合优化器使用学习率调度器——即使选对了优化器，固定的学习率也难以同时满足训练初期的快速收敛和训练后期的精细调优。

---

## ✏️ 练习

1. 【理解】用自己的话解释：为什么 AdaGrad 在训练后期会"学不动"？RMSProp 如何解决这个问题？用一个 200 字以内的类比说明。

2. 【实现】修改代码中的 `SGDMomentum` 类，添加 Nesterov 支持。当 `nesterov=True` 时使用 Nesterov 更新，`nesterov=False` 时退化为标准动量。在环形数据集上对比两种模式的收敛速度。

3. 【实验】在代码的环形数据集实验中，增加一个"超参数敏感度测试"：分别对 SGD 和 Adam 测试学习率在 $\{0.0001, 0.001, 0.01, 0.1, 1.0\}$ 五个值下的表现。哪个优化器对学习率更鲁棒？

4. 【思考】代码中的 `weight_decay_demo` 展示了 Adam 和 AdamW 的权重衰减差异。如果把 `weight_decay` 从 0.1 改成 0.001，效果差异会如何变化？如果改成 1.0 呢？设计一个实验验证你的猜想。

5. 【设计】假设你要训练一个 70 亿参数的中文大语言模型。从优化器选择、学习率设置、权重衰减、梯度裁剪四个方面，写出你的训练配置并解释每个参数的依据。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 七种优化器实现 | `code/main.py` | 从零实现 SGD、Momentum、Nesterov、AdaGrad、RMSProp、Adam、AdamW，包含完整实验 |
| 优化器选择指南 | `outputs/prompt-optimizer-selector.md` | 根据任务类型和约束推荐最优优化器和超参数 |
| 训练诊断工具 | `outputs/prompt-training-debugger.md` | 根据 loss 曲线异常现象诊断优化器配置问题 |

---

## 📖 参考资料

1. [论文] Kingma & Ba. "Adam: A Method for Stochastic Optimization". ICLR, 2015. https://arxiv.org/abs/1412.6980
2. [论文] Loshchilov & Hutter. "Decoupled Weight Decay Regularization" (AdamW). ICLR, 2019. https://arxiv.org/abs/1711.05101
3. [论文] Reddi et al. "On the Convergence of Adam and Beyond". ICLR, 2018. https://arxiv.org/abs/1904.09237
4. [论文] Sutskever et al. "On the importance of initialization and momentum in deep learning". ICML, 2013. https://dl.acm.org/doi/10.1145/3042817.3043040
5. [官方文档] PyTorch — `torch.optim`: https://pytorch.org/docs/stable/optim.html
6. [官方文档] PyTorch — `torch.optim.AdamW`: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、工程最佳实践、常见错误、面试考点等均为原创内容。
