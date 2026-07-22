# 权重初始化

> 初始化错了，训练永远不会开始。初始化对了，50 层网络训练得和 3 层一样顺畅。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 03 · 04（激活函数）— 理解 Sigmoid、Tanh、ReLU 的行为；阶段 03 · 05（反向传播）— 理解梯度如何流动
**预计时间：** ~90 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 07 · 01（Transformer 深入）— Transformer 的残差缩放初始化；阶段 10 · 03（从零构建大语言模型）— GPT-2/Llama 的权重初始化策略

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现零初始化、随机初始化、Xavier、Kaiming、正交初始化五种策略，并在 50 层网络上验证其效果
- [ ] 推导 Xavier 初始化的方差公式 Var(w) = 2/(fan_in + fan_out) 和 Kaiming 的 Var(w) = 2/fan_in
- [ ] 诊断零初始化的对称性问题，并解释为什么随机初始化的"尺度"决定了网络能否训练
- [ ] 根据激活函数选择正确的初始化策略：Sigmoid/Tanh 用 Xavier，ReLU/GELU 用 Kaiming

---

## 1. 问题

你把所有权重初始化为零。训练一万轮后，你的 512 个隐藏层神经元——全部输出相同值。你花了 512 个参数的计算量，只得到了 1 个神经元的效果。

你把权重初始化为标准正态分布 N(0, 1)。三层网络训练正常。但换成 50 层——信号在第 5 层就爆炸到 1e15，第 10 层直接溢出为无穷大。梯度走了一条相反的毁灭之路。

你把权重的标准差设为 0.01。50 层之后，信号缩小到 8.8e-16。你的网络从第一个隐藏层开始就是聋子——什么都听不到，什么都学不了。

权重初始化是深度学习中最被低估的决策。架构设计能发论文，优化器能写博客，初始化只配一个脚注。但做错了，其他一切都白搭——你的网络在训练开始之前就已经死了。

---

## 2. 概念

### 2.1 对称性问题

层中的每个神经元结构相同：输入乘以权重、加偏置、过激活函数。如果所有权重都从同一个值开始（零是极端情况），每个神经元计算相同的输出。反向传播时，每个神经元收到相同的梯度。更新时，每个神经元变化相同的量。

```
零初始化的 4 个神经元：

  输入: [0.5, -0.3]
       │       │
       ▼       ▼
  ┌─────────────────────────┐
  │ Neuron 0: w=[0, 0] → z=0 → σ(0)=0.500000 │
  │ Neuron 1: w=[0, 0] → z=0 → σ(0)=0.500000 │
  │ Neuron 2: w=[0, 0] → z=0 → σ(0)=0.500000 │
  │ Neuron 3: w=[0, 0] → z=0 → σ(0)=0.500000 │
  └─────────────────────────┘

  结果：4 个神经元 = 1 个有效神经元
  无论隐藏层有多宽，有效参数始终为 1。
```

随机初始化是打破这种对称性的暴力手段。每个神经元从不同的起点出发，自然会学到不同的特征。但"随机"本身不够——**随机的尺度**决定了网络能否训练。

### 2.2 方差传播

考虑一个有 fan_in 个输入的层：

```
z = w₁x₁ + w₂x₂ + ... + w_n × x_n
```

如果每个权重 w_i 从方差为 Var(w) 的分布中采样，每个输入 x_i 的方差为 Var(x)，那么输出的方差为：

$$\text{Var}(z) = \text{fan\_in} \times \text{Var}(w) \times \text{Var}(x)$$

如果 Var(w) = 1，fan_in = 512，输出方差就是输入方差的 512 倍。10 层后：512^10 = 1.2e27。信号爆炸了。

如果 Var(w) = 0.001，每层方差缩小 0.001 × 512 = 0.512 倍。10 层后：0.512^10 = 0.00013。信号消失了。

目标：选择 Var(w)，使得 Var(z) = Var(x)。信号幅度在层间保持恒定。

```
方差传播的三种结局：

  Var(w) × fan_in > 1  →  信号指数增长  →  爆炸
  Var(w) × fan_in = 1  →  信号保持稳定  →  正确 ✓
  Var(w) × fan_in < 1  →  信号指数衰减  →  消失
```

### 2.3 Xavier/Glorot 初始化

Glorot 和 Bengio（2010）推导出了 Sigmoid 和 Tanh 的最优解。为了在前向和反向传播中同时保持方差恒定：

$$\text{Var}(w) = \frac{2}{\text{fan\_in} + \text{fan\_out}}$$

实际使用时，权重从以下分布中采样：

$$w \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}\right)$$

或均匀分布版本：

$$w \sim U\left(-\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}},\ \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}\right)$$

这之所以有效，是因为 Sigmoid 和 Tanh 在零附近近似线性——正确初始化的激活值恰好生活在这一区域。方差在数十层内保持稳定。

### 2.4 Kaiming/He 初始化

ReLU 将一半的输出杀死（所有负值变为零）。有效输入数减半，因为平均一半的输入被归零。Xavier 初始化没有考虑这一点——它低估了所需的方差。

He 等人（2015）修正了公式：

$$\text{Var}(w) = \frac{2}{\text{fan\_in}}$$

权重采样：

$$w \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in}}}\right)$$

因子 2 补偿了 ReLU 将一半激活归零的效果。没有它，信号每层缩小约 0.5 倍。50 层后：0.5^50 = 8.8e-16。Kaiming 初始化防止了这种情况。

### 2.5 正交初始化

正交初始化通过 SVD 分解生成正交矩阵，确保每层的输入和输出方差严格相等：

$$w = U \Sigma V^T \rightarrow w_{\text{init}} = U$$

正交矩阵的特性是 $W^T W = I$，因此输出方差恰好等于输入方差，不依赖于 fan_in 的大小。这在 RNN/LSTM 中特别有用，因为循环结构对方差变化非常敏感。

缺点是计算开销较大（需要 SVD 分解），通常只在循环网络中使用。

### 2.6 Transformer 的残差缩放

GPT-2 引入了一种不同的初始化模式。残差连接将每个子层的输出加到输入上：

$$x = x + \text{sublayer}(x)$$

每次加法都会增加方差。经过 N 个残差层，方差增长约 N 倍。GPT-2 将残差层的权重缩放 $\frac{1}{\sqrt{2N}}$，其中 N 是层数，保持累积信号幅度稳定。

Llama 3（4050 亿参数，126 层）使用了类似的方案。没有这种缩放，残差流会在 126 层注意力和前馈块中无限增长。

### 2.7 初始化策略选择指南

```
用什么激活函数？
       │
       ├── Sigmoid / Tanh ──→ Xavier/Glorot
       │                      Var(w) = 2/(fan_in + fan_out)
       │
       ├── ReLU / Leaky ReLU ──→ Kaiming/He
       │                         Var(w) = 2/fan_in
       │
       ├── GELU / Swish ──→ Kaiming/He（同 ReLU）
       │
       └── Transformer 残差 ──→ 缩放 1/sqrt(2N)
                                N = 层数

验证：初始化后检查激活幅度是否在 0.5~2.0 之间。
```

---

## 3. 从零实现

### 第 1 步：四种基本初始化策略

```python
import math
import random


def zero_init(fan_in, fan_out):
    """零初始化：所有权重为 0。"""
    return [[0.0 for _ in range(fan_in)] for _ in range(fan_out)]


def random_init(fan_in, fan_out, scale=1.0):
    """随机初始化：从 N(0, scale) 采样。"""
    return [[random.gauss(0, scale) for _ in range(fan_in)] for _ in range(fan_out)]


def xavier_init(fan_in, fan_out):
    """Xavier 初始化：Var(w) = 2/(fan_in + fan_out)。"""
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def kaiming_init(fan_in, fan_out):
    """Kaiming 初始化：Var(w) = 2/fan_in。"""
    std = math.sqrt(2.0 / fan_in)
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
```

### 第 2 步：对称性问题验证

```python
def symmetry_demo():
    weights = zero_init(2, 4)
    biases = [0.0] * 4

    inputs = [0.5, -0.3]
    outputs = []
    for neuron_idx in range(4):
        z = sum(weights[neuron_idx][j] * inputs[j] for j in range(2)) + biases[neuron_idx]
        outputs.append(sigmoid(z))

    for i, out in enumerate(outputs):
        print(f"  神经元 {i}: 输出 = {out:.6f}")
    # 输出: 全部 0.500000
```

### 第 3 步：50 层前向传播实验

```python
def forward_deep(init_fn, activation_fn, n_layers=50, width=64, n_samples=100):
    """将数据通过 n_layers 层，记录每层的平均激活幅度。"""
    random.seed(42)
    layer_magnitudes = []
    inputs = [[random.gauss(0, 1) for _ in range(width)] for _ in range(n_samples)]

    for layer_idx in range(n_layers):
        weights = init_fn(width, width)
        biases = [0.0] * width
        new_inputs = []
        for sample in inputs:
            output = []
            for neuron_idx in range(width):
                z = sum(weights[neuron_idx][j] * sample[j]
                        for j in range(width)) + biases[neuron_idx]
                output.append(activation_fn(z))
            new_inputs.append(output)
        inputs = new_inputs
        magnitudes = [sum(abs(v) for v in s) / width for s in inputs]
        layer_magnitudes.append(sum(magnitudes) / len(magnitudes))

    return layer_magnitudes
```

### 第 4 步：完整实验

运行 `code/main.py` 可以看到所有组合的实验结果。关键结论：

```text
  策略                           L1         L5        L10        L25        L50
  Zero + Sigmoid             0.5000     0.5000     0.5000     0.5000     0.5000
  Random N(0,1) + ReLU       3.1305  3614.0573   EXPLODED   EXPLODED   EXPLODED
  Random N(0,0.01) + ReLU    0.0313   VANISHED   VANISHED   VANISHED   VANISHED
  Xavier + Sigmoid           0.4994     0.5036     0.5029     0.5037     0.4982
  Xavier + Tanh              0.5465     0.2692     0.1967     0.1494     0.0576
  Kaiming + ReLU             0.5534     0.6239     0.4946     0.3603     0.0971
```

观察：
- **Zero + Sigmoid**：信号恒定为 0.5，但所有神经元输出相同，网络无法学习
- **Random(1) + ReLU**：第 5 层就爆炸到 3614，50 层后达到 4e36
- **Random(0.01) + ReLU**：第 5 层就消失为 0
- **Xavier + Sigmoid**：信号在 0.5 附近稳定，50 层后依然健康
- **Kaiming + ReLU**：信号在合理范围内波动

---

## 4. 工业工具

### 4.1 PyTorch 内置实现

```python
import torch
import torch.nn as nn

layer = nn.Linear(512, 256)

# Xavier/Glorot 初始化
nn.init.xavier_uniform_(layer.weight)   # 均匀分布版本
nn.init.xavier_normal_(layer.weight)    # 正态分布版本

# Kaiming/He 初始化
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

# 偏置初始化为零（这是标准做法）
nn.init.zeros_(layer.bias)
```

当你调用 `nn.Linear(512, 256)` 时，PyTorch 默认使用 Kaiming 均匀分布初始化。这就是为什么大多数简单网络"直接能用"——PyTorch 已经做了正确的选择。但当你构建自定义架构或网络超过 20 层时，你需要理解底层发生了什么，并可能需要覆盖默认设置。

### 4.2 自定义初始化函数

```python
def init_weights(model):
    """根据层类型和激活函数自动选择初始化策略。"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # ReLU/GELU 用 Kaiming，Sigmoid/Tanh 用 Xavier
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                    nonlinearity='relu')
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

# 应用自定义初始化
model.apply(init_weights)
```

### 4.3 HuggingFace Transformers

HuggingFace 模型通常在 `_init_weights` 方法中处理初始化。GPT-2 的实现将残差投影缩放 $\frac{1}{\sqrt{N}}$。如果你从零构建 Transformer，需要自己添加这一缩放。

```python
# HuggingFace 模型的初始化通常自动处理
from transformers import GPT2Model

model = GPT2Model.from_pretrained("gpt2")
# 模型已经用 GPT-2 的初始化策略加载完毕
```

### 4.4 性能对比

| 实现方式 | 速度 | 内存 | 适用场景 |
|---|---|---|---|
| 我们的纯 Python 版 | 慢 | 低 | 学习理解 |
| PyTorch `nn.init` | 快 | 低 | 训练 / 研究 |
| HuggingFace `_init_weights` | 快 | 低 | 预训练模型加载 |
| 自定义初始化函数 | 快 | 低 | 特殊架构（如残差缩放） |

---

## 5. 知识连线

本课学习的权重初始化，是后续所有深度学习课程的基础：

- **阶段 03 · 09（批归一化）**：批归一化可以缓解初始化不当带来的问题，但它不能完全替代正确的初始化——批归一化 + Kaiming 初始化的组合效果远好于单独使用
- **阶段 07 · 01（Transformer 深入）**：Transformer 中的残差缩放初始化（GPT-2 风格）直接建立在方差传播的理解之上——理解了 Var(w) × fan_in 的关系，你就能理解为什么残差层需要额外缩放
- **阶段 10 · 03（从零构建大语言模型）**：你会看到 GPT-2 和 Llama 3 如何在数百层的网络中使用精心设计的初始化策略，确保信号在 126 层残差流中保持稳定

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| CNN / 视觉模型 | Kaiming + ReLU | PyTorch 默认即此 |
| Transformer（BERT/GPT/ViT） | Kaiming + GELU | 加残差缩放 1/sqrt(2N) |
| LSTM / GRU | 正交初始化 | 循环连接对方差敏感 |
| Embedding 层 | N(0, 0.02) | GPT 约定 |
| 归一化层 (BN/LN) | weight=1, bias=0 | 标准做法 |
| 输出层 | 无特殊要求 | 通常用默认 |

### 6.2 中文场景特别建议

- 中文 BERT 模型（`bert-base-chinese`）使用 GELU + Kaiming 初始化——微调时不要修改初始化策略
- 中文大语言模型（Qwen、DeepSeek）的 FFN 层使用 SiLU + Kaiming 初始化——保持与预训练一致
- 中文 OCR 或语音识别中的 LSTM 层应使用正交初始化——循环结构对方差变化非常敏感

### 6.3 踩坑经验

- 在超过 20 层的 ReLU 网络中使用 Xavier 而非 Kaiming——信号每层缩小约 0.5 倍，50 层后梯度消失
- 自定义 Transformer 时忘记残差缩放——100 层后残差流方差增长 100 倍，训练不稳定
- 使用 `nn.init.uniform_` 替换 PyTorch 默认的 Kaiming 初始化——均匀分布的方差与 Kaiming 不匹配
- 嵌入层使用 Kaiming 初始化而非 N(0, 0.02)——嵌入层的输入是 one-hot，与线性层的方差传播规律不同
- 在微调预训练模型时重新初始化所有权重——破坏了预训练学到的权重分布，模型需要从头学习

---

## 7. 常见错误

### 错误 1：零初始化导致对称性

**现象：** 训练 loss 停留在随机基线，所有神经元输出相同值。

**原因：** 零权重使每个神经元计算相同函数、收到相同梯度、更新相同量。无论隐藏层多宽，有效参数始终为 1。

**修复：**

```python
# ❌ 零初始化
layer = nn.Linear(256, 512)
nn.init.zeros_(layer.weight)  # 所有神经元相同

# ✓ 随机初始化（PyTorch 默认）
layer = nn.Linear(256, 512)  # 默认 Kaiming 均匀分布
```

### 错误 2：随机初始化尺度不当

**现象：** 训练初期 loss 为 NaN，或 loss 完全不下降。

**原因：** 标准正态分布 N(0,1) 对于 fan_in=512 的层，输出方差为 512 倍输入方差。10 层后方差达到 512^10，激活值溢出。反之，N(0, 0.01) 方差为 0.512 倍，10 层后信号缩小到 0.00013。

**修复：**

```python
# ❌ 手动设定不合理的标准差
layer = nn.Linear(256, 512)
nn.init.normal_(layer.weight, mean=0, std=1.0)  # 太大
nn.init.normal_(layer.weight, mean=0, std=0.01)  # 太小

# ✓ 使用自适应的 Xavier 或 Kaiming
nn.init.xavier_normal_(layer.weight)  # Sigmoid/Tanh
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # ReLU/GELU
```

### 错误 3：激活函数与初始化不匹配

**现象：** 训练初期 loss 下降极慢，或某些神经元永远不激活。

**原因：** Xavier 为 Sigmoid/Tanh 设计，假设激活函数在零附近近似线性。ReLU 将一半输出归零，有效 fan_in 减半，Xavier 低估了所需方差。

**修复：**

```python
# ❌ ReLU 配合 Xavier
layer = nn.Linear(256, 512)
nn.init.xavier_normal_(layer.weight)  # 方差不够

# ✓ ReLU 配合 Kaiming
layer = nn.Linear(256, 512)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

# ❌ Sigmoid 配合 Kaiming
layer = nn.Linear(256, 512)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # 方差过大

# ✓ Sigmoid 配合 Xavier
layer = nn.Linear(256, 512)
nn.init.xavier_normal_(layer.weight)
```

### 错误 4：Transformer 中忘记残差缩放

**现象：** 深层 Transformer（>50 层）训练不稳定，loss 震荡或不收敛。

**原因：** 残差连接 `x = x + sublayer(x)` 每层增加方差。经过 N 层，方差增长约 N 倍。不缩放时，信号在深层网络中无限增长。

**修复：**

```python
# ❌ 无缩放的残差连接
class Sublayer(nn.Module):
    def forward(self, x):
        return x + self.layer(x)  # 方差逐层增长

# ✓ GPT-2 风格：缩放 1/sqrt(2N)
class Sublayer(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layer = layer
        self.scale = 1.0 / math.sqrt(2.0 * n_layers)

    def forward(self, x):
        return x + self.layer(x) * self.scale
```

---

## 8. 面试考点

### Q1：为什么零初始化的网络无法学习？（难度：⭐）

**参考答案：**

零初始化导致每个神经元计算相同输出、收到相同梯度、更新相同量。这就是"对称性"——无论隐藏层有多宽（512 个神经元还是 1024 个），有效参数始终为 1。网络失去了表达能力。随机初始化通过给每个神经元不同的起点来打破对称性。

### Q2：为什么 Xavier 初始化的方差公式是 Var(w) = 2/(fan_in + fan_out)？（难度：⭐⭐）

**参考答案：**

前向传播中，$\text{Var}(z) = \text{fan\_in} \times \text{Var}(w) \times \text{Var}(x)$。反向传播中，$\text{Var}(g_{\text{in}}) = \text{fan\_out} \times \text{Var}(w) \times \text{Var}(g_{\text{out}})$。要同时保持前向和反向方差恒定，需要 $\text{Var}(w) = 2/(\text{fan\_in} + \text{fan\_out})$——因子 2 来自于同时约束两个方向。

### Q3：为什么 Kaiming 初始化比 Xavier 多了一个因子 2？（难度：⭐⭐）

**参考答案：**

ReLU 将一半的激活归零。对于 fan_in=512 的层，平均只有 256 个输入有效。如果用 Xavier（Var(w) = 2/(fan_in + fan_out)），有效方差为 $256 \times 2/(512+512) \approx 0.5$——信号每层缩小一半。Kaiming 用 Var(w) = 2/fan_in，补偿了 ReLU 的一半归零效果：$256 \times 2/512 = 1.0$。

### Q4：GPT-2 为什么将残差层权重缩放 1/sqrt(2N)？（难度：⭐⭐⭐）

**参考答案：**

残差连接 x = x + sublayer(x) 每层做一次加法，每次加法使方差增加 Var(sublayer_out)。经过 N 层，方差增长约 N 倍。缩放 sublayer 输出为 1/sqrt(2N)，使得每次加法增加的方差为 Var(sublayer_out)/(2N)。N 次累积后总增长为 N × 1/(2N) = 0.5，信号幅度保持稳定。因子 2 来自每个 Transformer 块有两个残差连接（注意力 + 前馈）。

### Q5：在生产环境中，你会如何验证初始化是否正确？（难度：⭐⭐）

**参考答案：**

前向传播一次，检查每层激活值的均值和标准差：

```python
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        hooks.append(module.register_forward_hook(
            lambda m, i, o, n=name:
                print(f"{n} | mean: {o.abs().mean():.4f} | std: {o.std():.4f}")
        ))
```

健康指标：所有层激活均值在 0.1~2.0 之间，无全零层，标准差在层间大致一致。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 权重初始化 | "随机设个初值" | 选择初始权重值的策略——决定了网络能否训练、训练多快 |
| 对称性问题 | "神经元都一样" | 零初始化导致所有神经元计算相同输出、收到相同梯度、永远无法分化 |
| Fan-in | "输入数量" | 一个神经元的输入连接数——决定了输入方差在加权求和中如何累积 |
| Fan-out | "输出数量" | 一个神经元的输出连接数——决定了反向传播中梯度方差如何传播 |
| Xavier/Glorot 初始化 | "Sigmoid 的初始化" | Var(w) = 2/(fan_in + fan_out)，为 Sigmoid/Tanh 设计，保持前向和反向方差稳定 |
| Kaiming/He 初始化 | "ReLU 的初始化" | Var(w) = 2/fan_in，补偿 ReLU 将一半激活归零的效果 |
| 方差传播 | "信号怎么变大变小" | 激活方差在层间如何根据权重尺度和激活函数变化的数学分析 |
| 残差缩放 | "GPT-2 的初始化技巧" | 将残差层权重缩放 1/sqrt(2N) 以防止信号在 N 层 Transformer 中无限增长 |
| 死网络 | "训练不动" | 初始化不当导致梯度为零或激活饱和，网络完全无法学习 |
| 信号爆炸 | "值变成无穷大" | 权重方差过大导致激活幅度在层间指数级增长 |

---

## 📚 小结

权重初始化决定了网络能否训练——做错了，其他一切都白搭。你从零实现了五种初始化策略（零初始化、随机初始化、Xavier、Kaiming、正交初始化），通过 50 层前向传播实验直观验证了方差传播的数学分析，并理解了 GPT-2 残差缩放初始化的原理。

下一课我们将学习批归一化——它可以缓解初始化不当带来的问题，但不能完全替代正确的初始化。批归一化 + Kaiming 初始化的组合效果远好于单独使用。

---

## ✏️ 练习

1. 【理解】用自己的话解释：为什么 Xavier 初始化的公式中包含 fan_out，而 Kaiming 不包含？写 150 字以内的说明。（提示：思考反向传播中梯度的方差传播）

2. 【实现】在 `code/main.py` 中增加 LeCun 初始化（Var(w) = 1/fan_in，为 SELU 激活函数设计）。用 50 层网络对比 LeCun + Tanh 和 Xavier + Tanh 的效果。

3. 【实验】将 `forward_deep` 的 fan_in 从 64 改为 16 和 1024，观察 Xavier 和 Kaiming 如何自适应不同宽度，而固定尺度的随机初始化在不同宽度下的表现差异。

4. 【思考】GPT-2 的残差缩放是 1/sqrt(2N)，其中 2 来自每个块有两个残差连接。如果你的 Transformer 块有 3 个残差连接（注意力 + 两个前馈），缩放因子应该改为多少？为什么？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 初始化策略完整实现 | `code/main.py` | 五种初始化策略 + 50 层实验 + 残差缩放演示 |
| 初始化诊断提示词 | `outputs/prompt-weight-init-guide.md` | 根据架构和训练行为诊断初始化问题的提示词 |

---

## 📖 参考资料

1. [论文] Glorot, Bengio. "Understanding the difficulty of training deep feedforward neural networks". AISTATS, 2010. https://proceedings.mlr.press/v9/glorot10a.html
2. [论文] He, Zhang, Ren, Sun. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification". ICCV, 2015. https://arxiv.org/abs/1502.01852
3. [论文] Radford, Wu, Child, Luan, Amodei, Sutskever. "Language Models are Unsupervised Multitask Learners". OpenAI, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
4. [论文] Mishkin, Matas. "All You Need is a Good Init". ICLR, 2016. https://arxiv.org/abs/1511.06422
5. [官方文档] PyTorch `torch.nn.init`: https://pytorch.org/docs/stable/nn.init.html

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、工程最佳实践、常见错误、面试考点等均为原创内容。
