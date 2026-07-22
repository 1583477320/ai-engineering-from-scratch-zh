# 激活函数

> 没有激活函数，再深的网络也只是一个线性变换。激活函数是神经网络"学非线性"的唯一来源。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 01（数学基础）— 导数与链式法则；阶段 02（机器学习基础）— 神经网络基本结构
**预计时间：** ~90 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 03 · 05（反向传播）— 激活函数的导数直接参与梯度计算；阶段 07 · 01（Transformer 深入）— GELU 是 Transformer 的默认选择

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么没有激活函数的深层网络等价于单层线性变换
- [ ] 从零实现 Sigmoid、Tanh、ReLU、Leaky ReLU、GELU、SiLU 六种激活函数及其导数
- [ ] 诊断梯度消失和死亡 ReLU 问题，并说明如何用合适的激活函数和初始化修复
- [ ] 根据任务类型（CNN / Transformer / MLP / 输出层）选择正确的激活函数
- [ ] 使用 PyTorch 在生产代码中配置激活函数，避免常见陷阱

---

## 1. 问题

你的神经网络有 100 层，每一层都认真训练了权重。训练完成后你发现——它跟一层网络的表现一模一样。

这不是 bug，这是数学事实。

一个线性层做的事情是 $y = Wx + b$。两个线性层叠加：

$$y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W'x + b'$$

不管堆多少层，结果永远等价于一个线性变换。100 层的表达能力 = 1 层。

**激活函数**就是来打破这个"线性诅咒"的。它在每个神经元输出上加了一道"门"：信号达到阈值才能通过，否则被抑制。正是这道非线性门，让深层网络能够逼近任意复杂函数。

但选择哪道门，不是随意的。选错了：

- Sigmoid 让梯度在深层网络中指数级衰减——**梯度消失**，前几层几乎不学习
- ReLU 让一部分神经元永久沉默——**死亡神经元**，网络容量白白浪费
- 在 Transformer 里用 ReLU 而不是 GELU——模型收敛更慢，效果更差

你在 ChatGPT 中输入的每一个词元、模型生成的每一个词元——都经过了数十次激活函数的处理。激活函数不是神经网络的"配件"。它是神经网络能够学习的**前提条件**。

---

## 2. 概念

### 2.1 直观理解

把激活函数想象成神经元之间的"阀门"：

```
没有激活函数（线性）：
  输入 ──→ [ 加权求和 ] ──→ 输出
  信号原样通过，100 层 = 1 层

有激活函数（非线性）：
  输入 ──→ [ 加权求和 ] ──→ [ 阀门：够强才通过 ] ──→ 输出
  每层可以选择性地放大、抑制、变形信号
```

每个激活函数回答一个不同的问题：

| 激活函数 | 核心问题 | 回答方式 |
|---|---|---|
| Sigmoid | "这个信号有多强？" | 压缩到 0~1，像概率 |
| Tanh | "这个信号是正是负？" | 压缩到 -1~1，零中心化 |
| ReLU | "这个信号够格吗？" | 负值归零，正值保留 |
| Leaky ReLU | "负值就完全没用吗？" | 负值保留一点点 |
| GELU | "按概率保留多少？" | 平滑的、概率性的 ReLU |
| SiLU | "信号自己决定放大多少？" | 信号 × Sigmoid(信号) |

### 2.2 形式化定义

**Sigmoid：**

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

输出范围 $(0, 1)$。当 $x = 0$ 时输出 $0.5$；$x \to +\infty$ 时趋近 1；$x \to -\infty$ 时趋近 0。

**Tanh：**

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

输出范围 $(-1, 1)$。与 Sigmoid 形状相同，但零中心化（均值为 0），收敛通常更快。

**ReLU（Rectified Linear Unit）：**

$$\text{ReLU}(x) = \max(0, x)$$

负值输出 0，正值原样输出。计算极快（一次比较），但负区间梯度为 0。

**Leaky ReLU：**

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

其中 $\alpha$ 通常取 0.01。负区间保留微小梯度，避免神经元永久死亡。

**GELU（Gaussian Error Linear Unit）：**

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。直觉上：输入 $x$ 越大，被保留的概率越高。这是 BERT、GPT、ViT 等 Transformer 模型的默认激活函数。

**SiLU（Sigmoid Linear Unit，也称 Swish-1）：**

$$\text{SiLU}(x) = x \cdot \sigma(x)$$

信号乘以自身的 Sigmoid 门控。当 $x$ 很大时趋近于 $x$（类似 ReLU）；当 $x$ 为负时平滑趋近于 0。

**Softmax：**

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

将向量转换为概率分布——所有输出在 $(0, 1)$ 之间且求和为 1。仅用于输出层。

### 2.3 激活函数全景对比

```
         Sigmoid          Tanh            ReLU
     1.0 ┤    ╭────    1.0┤   ╭──      2.0┤      ╱
         │   ╱             │  ╱            │     ╱
     0.5 ┤  ╱          0.0┤─╱────     0.0┤────╱
         │ ╱               │╱             │   ╱
     0.0 ┤╱           -1.0┤╱           -0.0├──╱
         └──┬──┬──         └──┬──┬──         └──┬──┬──
           -5  0  5         -5  0  5         -5  0  5

       Leaky ReLU         GELU            SiLU
     2.0 ┤      ╱      2.0┤      ╱      2.0┤      ╱
         │     ╱           │     ╱           │     ╱
     0.0 ┤────╱        0.0┤────╱        0.0┤────╱
         │   ╱             │  ╱              │ ╱
    -0.05├──╱          -0.2┤╱            -0.3┤╱
         └──┬──┬──         └──┬──┬──         └──┬──┬──
           -5  0  5         -5  0  5         -5  0  5
```

### 2.4 导数为什么重要

训练神经网络时，反向传播通过链式法则计算梯度。激活函数的导数是链式乘积中的一环：

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial a} \cdot \underbrace{\frac{\partial a}{\partial z}}_{\text{激活函数导数}} \cdot \frac{\partial z}{\partial W}$$

如果激活函数的导数在很多位置接近 0，整个梯度就会指数级缩小——这就是**梯度消失**。

| 激活函数 | 导数公式 | 最大导数 | 导数为 0 的区域 |
|---|---|---|---|
| Sigmoid | $\sigma(x)(1-\sigma(x))$ | 0.25（在 $x=0$） | $\|x\| > 5$ 时趋近 0 |
| Tanh | $1 - \tanh^2(x)$ | 1.0（在 $x=0$） | $\|x\| > 3$ 时趋近 0 |
| ReLU | $\mathbb{1}_{x>0}$ | 1.0 | 所有 $x < 0$ |
| Leaky ReLU | $\mathbb{1}_{x>0} + \alpha \cdot \mathbb{1}_{x \leq 0}$ | 1.0 | 无（负区间为 $\alpha$） |
| GELU | $\Phi(x) + x \cdot \phi(x)$ | ~1.0 | 无（平滑，负区间有微小梯度） |
| SiLU | $\sigma(x) + x \cdot \sigma(x)(1-\sigma(x))$ | ~1.0 | 无 |

---

## 3. 从零实现

### 第 1 步：最简版本——Sigmoid 和 ReLU

```python
import math

def sigmoid(x):
    """Sigmoid：将输入压缩到 (0, 1)。"""
    x = max(-500, min(500, x))  # 防止 exp 溢出
    return 1.0 / (1.0 + math.exp(-x))

def relu(x):
    """ReLU：负值归零，正值保持不变。"""
    return max(0.0, x)

# 测试
print(f"sigmoid(0) = {sigmoid(0):.4f}")  # 0.5000
print(f"sigmoid(2) = {sigmoid(2):.4f}")  # 0.8808
print(f"relu(-1) = {relu(-1):.4f}")       # 0.0000
print(f"relu(3)  = {relu(3):.4f}")        # 3.0000
```

### 第 2 步：加入导数

```python
def sigmoid_derivative(x):
    """Sigmoid 导数：σ'(x) = σ(x)(1 - σ(x))。"""
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    """ReLU 导数：正区间为 1，负区间为 0。"""
    return 1.0 if x > 0 else 0.0

# 验证：Sigmoid 在 x=0 处导数最大，为 0.25
print(f"σ'(0) = {sigmoid_derivative(0):.4f}")  # 0.2500
print(f"σ'(5) = {sigmoid_derivative(5):.6f}")   # 0.006631 — 接近 0
```

### 第 3 步：GELU 和 SiLU

```python
def gelu(x):
    """GELU：高斯误差线性单元。使用 tanh 近似。"""
    inner = math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1 + math.tanh(inner))

def silu(x):
    """SiLU（Sigmoid Linear Unit）：x * σ(x)。"""
    return x * sigmoid(x)

# 对比：在 x = -2 时，ReLU 输出 0，GELU 和 SiLU 保留微小负值
x = -2.0
print(f"x = {x}")
print(f"  ReLU({x}) = {relu(x):.4f}")   # 0.0000
print(f"  GELU({x}) = {gelu(x):.4f}")   # -0.0454
print(f"  SiLU({x}) = {silu(x):.4f}")   # -0.2384
```

### 第 4 步：梯度死区扫描

```python
def gradient_scan(name, derivative_fn, start=-5, end=5, n=100):
    """统计梯度接近 0 的比例——这就是"死区"。"""
    step = (end - start) / n
    near_zero = sum(1 for i in range(n)
                    if abs(derivative_fn(start + i * step)) < 0.01)
    pct_dead = near_zero / n * 100
    print(f"  {name:15s}: {pct_dead:.0f}% 死区")

gradient_scan("Sigmoid", sigmoid_derivative)    # 9% 死区
gradient_scan("Tanh", tanh_derivative)          # 41% 死区
gradient_scan("ReLU", relu_derivative)          # 51% 死区
gradient_scan("Leaky ReLU", leaky_relu_derivative)  # 0% 死区
gradient_scan("GELU", gelu_derivative)          # 20% 死区
gradient_scan("SiLU", silu_derivative)          # 1% 死区
```

### 第 5 步：死亡神经元检测

```python
def dead_neuron_detector(n_inputs=5, hidden_size=20, n_samples=1000):
    """检测 ReLU 网络中从未被激活的神经元。"""
    import random
    random.seed(0)
    weights = [[random.gauss(0, 1) for _ in range(n_inputs)]
               for _ in range(hidden_size)]
    biases = [random.gauss(0, 1) for _ in range(hidden_size)]

    fire_counts = [0] * hidden_size
    for _ in range(n_samples):
        inputs = [random.gauss(0, 1) for _ in range(n_inputs)]
        for i in range(hidden_size):
            z = sum(w * x for w, x in zip(weights[i], inputs)) + biases[i]
            if relu(z) > 0:
                fire_counts[i] += 1

    dead = sum(1 for c in fire_counts if c == 0)
    print(f"死亡神经元: {dead}/{hidden_size} ({dead/hidden_size*100:.1f}%)")
```

### 第 6 步：完整训练对比

运行 `code/main.py` 可以看到三种激活函数在圆形数据集上的完整训练对比。关键结论：

```
=== 最终损失对比 ===
  Sigmoid  : 起始=0.2222 → 最终=0.0319 (改善: 85.6%)
  ReLU     : 起始=0.2232 → 最终=0.0102 (改善: 95.4%)
  GELU     : 起始=0.2225 → 最终=0.0056 (改善: 97.5%)
```

GELU 收敛最快、最终损失最低。Sigmoid 改善幅度最小——梯度消失限制了它的学习能力。

---

## 4. 工业工具

### 4.1 PyTorch 内置实现

```python
import torch
import torch.nn as nn

# 方式 1：作为独立模块
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
gelu = nn.GELU()
silu = nn.SiLU()  # 也称 Swish-1

x = torch.randn(2, 3)
print(f"ReLU:    {relu(x)}")
print(f"GELU:    {gelu(x)}")

# 方式 2：在 Sequential 中直接使用
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.GELU(),          # Transformer 风格
    nn.Linear(512, 256),
    nn.GELU(),
    nn.Linear(256, 10),
)

# 方式 3：在 Transformer 的 FFN 中使用（这是标准做法）
class FeedForward(nn.Module):
    """Transformer 中的前馈网络。"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()  # BERT/GPT 都用 GELU
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

### 4.2 输出层激活函数选择

```python
# 二分类：Sigmoid 输出概率
binary_head = nn.Sequential(
    nn.Linear(768, 1),
    nn.Sigmoid()
)

# 多分类：CrossEntropyLoss 内部包含 Softmax，不需要显式添加
multi_class_head = nn.Linear(768, 10)  # 直接输出 logits

# 多标签分类：每个类别独立用 Sigmoid
multilabel_head = nn.Sequential(
    nn.Linear(768, 20),
    nn.Sigmoid()
)

# 回归：不使用激活函数（线性输出）
regression_head = nn.Linear(768, 1)
```

### 4.3 性能对比

| 实现方式 | 速度 | 内存 | 适用场景 |
|---|---|---|---|
| 我们的纯 Python 版 | 慢 | 低 | 学习理解 |
| PyTorch `nn.ReLU()` | 快 | 低 | CNN、MLP 隐藏层 |
| PyTorch `nn.GELU()` | 快 | 低 | Transformer 隐藏层 |
| PyTorch `nn.SiLU()` | 快 | 低 | EfficientNet、YOLO |
| `torch.nn.functional` 版 | 最快 | 最低 | 生产环境（无状态） |

---

## 5. 知识连线

本课学习的激活函数，是后续所有深度学习课程的核心组件：

- **阶段 03 · 05（反向传播）**：激活函数的导数直接参与链式法则计算——理解了 $\sigma'(x)$ 和 $\text{ReLU}'(x)$，你就能理解梯度如何在网络中流动
- **阶段 07 · 01（Transformer 深入）**：GELU 是 Transformer 前馈网络的标准选择——理解了 GELU 为什么比 ReLU 平滑，你就能理解为什么 Transformer 训练更稳定
- **阶段 10（从零构建大语言模型）**：你会看到 GPT 和 LLaMA 的每一层都使用 GELU 或 SiLU——激活函数的选择直接影响大语言模型的收敛速度和最终性能

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| CNN / 视觉模型 | ReLU | 计算快，配合批归一化效果好 |
| EfficientNet / MobileNetV3 | SiLU / Swish | 自门控特性在轻量模型上效果显著 |
| Transformer（BERT/GPT/ViT） | GELU | 平滑梯度，训练更稳定 |
| LSTM / GRU 门控 | Sigmoid | 门控天然需要 0~1 输出 |
| LSTM / GRU 隐藏状态 | Tanh | 需要 -1~1 的对称输出 |
| 二分类输出层 | Sigmoid | 输出概率 |
| 多分类输出层 | 无（CrossEntropyLoss 内含 Softmax） | 不要重复加 Softmax |
| 回归输出层 | 无（线性输出） | 除非有明确上下界 |

### 6.2 中文场景特别建议

- 中文文本分类任务中，BERT 的 `bert-base-chinese` 模型使用 GELU 作为激活函数——微调时不要替换为 ReLU，会破坏预训练权重
- 在中文 OCR 或语音识别等序列任务中，LSTM 的门控激活函数（Sigmoid）和隐藏状态激活函数（Tanh）是固定的，不要修改
- 中文大语言模型（如 Qwen、DeepSeek）的 FFN 层使用 SiLU——这是与 GPT 的 GELU 不同的设计选择，微调时保持原样

### 6.3 踩坑经验

- 在 `nn.Sequential` 中同时使用 `nn.Softmax()` 和 `nn.CrossEntropyLoss()`——Softmax 被应用了两次，梯度异常，训练不收敛
- 在 Transformer 中用 ReLU 替换 GELU——模型能训练，但收敛速度慢 20%~30%，最终困惑度更高
- 使用 ReLU 时未配合恰当的初始化（应使用 Kaiming 初始化）——大量神经元在训练初期就死亡
- 在 RNN 隐藏层使用 ReLU——梯度爆炸风险极高，RNN 应使用 Tanh
- 输出层使用 Tanh 做分类——Tanh 输出范围是 $(-1, 1)$，与 0/1 标签不匹配，应使用 Sigmoid

---

## 7. 常见错误

### 错误 1：深层网络使用 Sigmoid 作为隐藏层激活函数

**现象：** 训练 loss 下降极慢，前几层权重几乎不变，浅层网络反而比深层网络效果好。

**原因：** Sigmoid 的最大导数仅为 0.25（在 $x=0$ 处）。反向传播时每经过一层，梯度至少乘以 0.25。经过 10 层后，梯度缩小到 $0.25^{10} \approx 10^{-6}$——前几层几乎收不到更新信号。

**修复：**

```python
# ❌ 深层网络使用 Sigmoid
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.Sigmoid(),  # 梯度消失！
    nn.Linear(512, 512),
    nn.Sigmoid(),
    nn.Linear(512, 10),
)

# ✓ 深层网络使用 GELU 或 ReLU
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.GELU(),     # 梯度流畅
    nn.Linear(512, 512),
    nn.GELU(),
    nn.Linear(512, 10),
)
```

### 错误 2：ReLU 网络出现大量死亡神经元

**现象：** 训练过程中部分神经元输出始终为 0，对应权重不再更新，网络有效容量下降。

**原因：** 如果某个神经元的加权输入 $z$ 始终为负，ReLU 输出为 0，梯度也为 0。权重永远无法更新——神经元"死亡"了。学习率过大或初始化不当会加剧这个问题。

**修复：**

```python
# ❌ 标准 ReLU + 大学习率
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),     # 可能大量死亡
    nn.Linear(512, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # 学习率过大

# ✓ Leaky ReLU + 合理学习率 + Kaiming 初始化
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.LeakyReLU(0.01),  # 负区间保留微小梯度
    nn.Linear(512, 10),
)
# Kaiming 初始化：适配 ReLU 系列的初始化方法
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 错误 3：输出层重复应用 Softmax

**现象：** 训练 loss 不下降或下降异常慢，模型输出概率过于"尖锐"（接近 one-hot）。

**原因：** `nn.CrossEntropyLoss` 内部已经包含了 Softmax。如果模型输出层再加一次 Softmax，相当于做了两次概率归一化，梯度信号被压缩。

**修复：**

```python
# ❌ 重复 Softmax
model = nn.Sequential(
    nn.Linear(768, 10),
    nn.Softmax(dim=-1),  # 多余！
)
criterion = nn.CrossEntropyLoss()  # 内部已有 Softmax

# ✓ 输出层不加 Softmax
model = nn.Sequential(
    nn.Linear(768, 10),  # 直接输出 logits
)
criterion = nn.CrossEntropyLoss()
```

### 错误 4：激活函数与初始化不匹配

**现象：** 训练初期 loss 为 NaN，或者模型输出全部相同。

**原因：** ReLU 系列激活函数需要 Kaiming 初始化（He 初始化），Sigmoid/Tanh 需要 Xavier 初始化（Glorot 初始化）。错配会导致信号在前向传播中指数级增大或缩小。

**修复：**

```python
# ❌ ReLU 配合默认初始化（均匀分布）
layer = nn.Linear(256, 512)
nn.init.uniform_(layer.weight, -0.01, 0.01)  # 不匹配 ReLU

# ✓ ReLU 配合 Kaiming 初始化
layer = nn.Linear(256, 512)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

# ✓ Sigmoid/Tanh 配合 Xavier 初始化
layer = nn.Linear(256, 512)
nn.init.xavier_normal_(layer.weight)
```

---

## 8. 面试考点

### Q1：为什么没有激活函数的多层神经网络等价于单层？（难度：⭐⭐）

**参考答案：**

线性层执行 $y = Wx + b$。两层叠加：$y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W'x + b'$。无论堆叠多少层，复合函数仍然是线性的。激活函数引入非线性，打破了这种复合等价性，使深层网络能够表达更复杂的函数。

### Q2：ReLU 的"死亡神经元"问题是怎么产生的？如何修复？（难度：⭐⭐）

**参考答案：**

当 ReLU 神经元的加权输入始终为负时，输出为 0，梯度也为 0。由于梯度为 0，权重无法更新，神经元永远无法被再次激活——这就是"死亡"。修复方法：(1) 使用 Leaky ReLU 或 GELU，保留负区间的微小梯度；(2) 使用 Kaiming 初始化；(3) 降低学习率。

### Q3：为什么 Transformer 使用 GELU 而不是 ReLU？（难度：⭐⭐⭐）

**参考答案：**

GELU 是平滑的激活函数（处处可导），而 ReLU 在 $x=0$ 处不可导。平滑性带来更稳定的梯度流，这对 Transformer 这种深层（12~96 层）架构至关重要。此外，GELU 是概率性的——按输入被保留的概率进行加权，比 ReLU 的硬阈值更符合自然信号的连续特性。实验表明 GELU 在语言模型上的效果优于 ReLU。

### Q4：手写 GELU 的近似计算公式（难度：⭐⭐）

**参考答案：**

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)$$

或等价地使用误差函数形式：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

### Q5：Softmax 为什么通常只用于输出层？（难度：⭐⭐）

**参考答案：**

Softmax 将向量归一化为概率分布（所有值求和为 1）。输出层需要概率分布来表示类别置信度。但隐藏层需要保留和传递信息——如果对隐藏层输出做 Softmax，会强制所有神经元的输出竞争（一个增大其他必须减小），限制了网络的表达能力。此外，隐藏层使用 Softmax 会导致梯度在神经元间分配不均，训练困难。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 激活函数 | "给神经网络加个非线性" | 对神经元加权求和后的输出进行非线性变换的函数——没有它，多层网络等价于单层 |
| 梯度消失 | "深层网络训练不动" | 反向传播时梯度逐层乘以一个小于 1 的数，指数级缩小，导致前几层几乎不更新 |
| 死亡 ReLU | "神经元死了" | ReLU 神经元因输入始终为负而输出恒为 0、梯度恒为 0，永远无法恢复 |
| GELU | "Transformer 用的那个" | 高斯误差线性单元——按概率保留输入，平滑可导，是 BERT/GPT/ViLU 的默认选择 |
| 死区 | "梯度为 0 的地方" | 激活函数导数接近 0 的输入范围——Sigmoid 在 $|x|>5$，ReLU 在 $x<0$ |
| 零中心化 | "输出均值为 0" | Tanh 的输出范围 $(-1, 1)$ 均值为 0，比 Sigmoid 的 $(0, 1)$ 更有利于梯度流动 |
| Kaiming 初始化 | "ReLU 专用的初始化" | 根据 ReLU 的特性调整初始化方差，防止信号在前向/反向传播中指数级变化 |
| 自门控 | "信号自己控制自己" | SiLU/Swish 中信号 $x$ 乘以 $\sigma(x)$——信号越大，门控越开 |

---

## 📚 小结

激活函数是神经网络能够学习非线性关系的根本原因。你从零实现了六种激活函数及其导数，理解了梯度消失和死亡 ReLU 的产生机制，并掌握了根据架构类型选择激活函数的方法。

下一课我们将学习反向传播算法——激活函数的导数将在链式法则中发挥核心作用，梯度如何从输出层一路传回每一层的权重。

---

## ✏️ 练习

1. 【理解】用自己的话解释：为什么没有激活函数的 100 层神经网络等价于 1 层？写 150 字以内的说明，让一个只学过线性代数的程序员也能听懂。

2. 【实现】修改 `code/main.py` 中的 `gradient_scan` 函数，增加一个"梯度爆炸"检测——统计梯度绝对值大于 10 的比例。对 Sigmoid 和 Tanh 运行，观察结果。

3. 【实验】在 `code/main.py` 中增加一个实验：对比 ReLU 和 Leaky ReLU 在死亡神经元检测中的表现（`dead_neuron_detector`），记录两者的死亡比例差异。

4. 【思考】GELU 的公式是 $x \cdot \Phi(x)$，其中 $\Phi(x)$ 是正态分布的 CDF。请解释：为什么这个公式可以理解为"按概率保留输入"？当 $x$ 很大时，GELU 趋近于什么？当 $x$ 很小时呢？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 激活函数完整实现 | `code/main.py` | 从零实现 6 种激活函数及其导数，含梯度扫描、死亡神经元检测、训练对比 |
| 激活函数选择提示词 | `outputs/prompt-activation-selector.md` | 根据架构类型和任务选择最优激活函数的决策提示词 |
| 激活函数可视化 | `code/main.py`（第 6 步） | 生成激活函数曲线和导数曲线对比图 |

---

## 📖 参考资料

1. [论文] Nair, Hinton. "Rectified Linear Units Improve Restricted Boltzmann Machines". ICML, 2010. https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf
2. [论文] Hendrycks, Gimpel. "Gaussian Error Linear Units (GELUs)". arXiv, 2016. https://arxiv.org/abs/1606.08415
3. [论文] Ramachandran, Zoph, Le. "Searching for Activation Functions". arXiv, 2017. https://arxiv.org/abs/1710.05941
4. [官方文档] PyTorch Activation Functions: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
5. [论文] Glorot, Bengio. "Understanding the difficulty of training deep feedforward neural networks". AISTATS, 2010. https://proceedings.mlr.press/v9/glorot10a.html

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
