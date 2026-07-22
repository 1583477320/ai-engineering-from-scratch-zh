# JAX 入门

> PyTorch 修改张量。TensorFlow 构建计算图。JAX 编译纯函数。最后这种范式会改变你对深度学习编程的理解。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 03 · 01-10（感知机、多层网络、反向传播、激活函数、损失函数、优化器、正则化、权重初始化、学习率调度、迷你框架）；基本的 NumPy 使用经验
**预计时间：** ~90 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 03 · 10（迷你框架）— 理解 PyTorch 风格的面向对象框架；阶段 07 · 01（Transformer 架构）— JAX 是训练大规模 Transformer 的主要框架之一

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 JAX 的函数式编程哲学，对比它与 PyTorch 面向对象方式在状态管理上的根本差异
- [ ] 使用 `jax.numpy`、`jax.grad`、`jax.jit`、`jax.vmap` 编写函数式神经网络代码
- [ ] 解释 JIT 编译的工作原理——追踪（tracing）如何将 Python 函数转化为 XLA 机器码
- [ ] 使用 `jax.pmap` 在多设备上实现数据并行，理解其与 PyTorch `DataParallel` 的差异
- [ ] 用 JAX + Optax 训练一个 3 层多层感知机在 MNIST 上达到 97% 以上准确率

---

## 1. 问题

你已经会用 PyTorch 构建神经网络了。定义一个 `nn.Module`，调用 `.backward()`，更新优化器。它能用。全世界数百万人在用。

但 PyTorch 的设计里有一个无法绕过的约束：它在 Python 中急切地（eagerly）逐个追踪操作。每次 `tensor + tensor` 都是一次独立的内核启动。每次训练步骤都要重新解释同一段 Python 代码。对于小模型和单 GPU，这完全没问题。但当你需要在 2048 块 TPU 上训练 5400 亿参数的模型时，这些开销就成了瓶颈。

Google DeepMind 用 JAX 训练 Gemini。Anthropic 用 JAX 训练 Claude。这不是小团队——它们是地球上最大的神经网络训练项目的运行方。它们选择 JAX，是因为 JAX 把你的训练循环视为一个可编译的程序，而不是一系列 Python 调用。

JAX 本质上是 NumPy 加上了三个超能力：自动微分、即时编译（JIT）到 XLA、以及自动向量化。你写一个处理单个样本的函数，JAX 帮你得到一个处理批次、计算梯度、编译为机器码、在多个设备上运行的版本。原始函数一行都不用改。

---

## 2. 概念

### 2.1 JAX 的函数式哲学

JAX 是函数式框架。没有类，没有可变状态，没有 `.backward()` 方法。它的核心变换只有一条规则：**纯函数——相同输入永远产生相同输出，没有副作用。**

```
┌─────────────────────────────────────────────────────────┐
│                   PyTorch（面向对象）                    │
│                                                         │
│  model = nn.Linear(10, 1)     ← 创建对象，状态在内部     │
│  model.weight[0] = 5          ← 原地修改权重             │
│  output = model(input)        ← 对象内部状态参与计算     │
│  model.zero_grad()            ← 手动重置内部梯度         │
│  model.parameters()           ← 从对象中提取所有参数     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   JAX（函数式）                          │
│                                                         │
│  params = init_params(key)    ← 参数是普通字典           │
│  params['w'] = new_w          ← 不行！数组不可变         │
│  output = predict(params, x)  ← 参数显式传入函数         │
│  params = update(params, g)   ← 函数返回新参数，不修改旧的│
└─────────────────────────────────────────────────────────┘
```

这不是风格偏好，而是编译器的约束。JIT 编译要求纯函数——如果你的函数偷偷修改外部变量，编译器就无法安全地追踪和优化它。这个限制恰恰是 100 倍加速的代价。

以下是 JAX 与 PyTorch 核心概念的对照表：

| PyTorch | JAX |
|---------|-----|
| `nn.Module` 类（封装状态） | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 急切执行 | JIT 编译为 XLA |
| `for x in batch:` 手动循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| `model.parameters()`（可变） | 不可变的 pytree 数组结构 |

### 2.2 jnp 数组：熟悉的 API，不同的规则

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)  # 结果: 32.0
```

函数名相同，广播规则相同，切片语义相同。但数组生活在 GPU/TPU 上，每个操作都能被编译器追踪。

关键差异在于：JAX 数组是不可变的。不能写 `a[0] = 5`。必须写 `a = a.at[0].set(5)`——返回一个新数组，而不是修改原数组。这看起来很笨拙，但一周后你会理解——不可变性是 `grad`、`jit`、`vmap` 等变换能够组合工作的基础。

```
JAX 数组不可变：

  不可写:  a[0] = 5       ← 会报错
  必须写:  a = a.at[0].set(5)  ← 返回新数组

PyTorch 可变:

  可以写:  a[0] = 5       ← 原地修改
```

### 2.3 jax.grad：函数式自动微分

PyTorch 把梯度附加到张量上（`.grad`）。JAX 把梯度附加到函数上。

```python
import jax

def f(x):
    return x ** 2

# jax.grad 将 f 变换为 f 的导函数
df = jax.grad(f)
print(df(3.0))  # 6.0 (即 2*3)
```

`jax.grad` 接收一个函数，返回一个新函数——新函数计算原函数的梯度。没有 `.backward()` 调用。没有计算图存储在张量上。梯度只是另一个函数，你可以调用、组合、JIT 编译。

这种组合性没有上限：

```python
# 二阶导数：对导函数再求导
d2f = jax.grad(jax.grad(f))
print(d2f(3.0))  # 2.0 (常数)

# 三阶导数
d3f = jax.grad(jax.grad(jax.grad(f)))
print(d3f(3.0))  # 0.0
```

PyTorch 也能计算高阶导数（`torch.autograd.functional.hessian`），但那是后期补丁式的支持。在 JAX 中，`grad` 是基础——它天然支持任意阶导数的组合。

约束条件：`grad` 只能用于纯函数。不能在函数内打印（print 在追踪时执行，不在实际运行时执行）。不能修改外部状态。不能使用不带显式密钥管理的随机数。

### 2.4 jit 编译：追踪 → XLA → 机器码

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss
```

第一次调用时，JAX **追踪**（trace）函数——记录发生了哪些操作，但不实际执行。然后把追踪结果交给 XLA（Accelerated Linear Algebra，加速线性代数编译器），它会融合操作、消除冗余内存拷贝、生成优化的机器码。

后续调用完全跳过 Python 解释器。编译后的代码以 C++ 的速度在加速器上运行。

JIT 什么时候有帮助：

- 训练步骤（相同计算重复数千次）
- 推理（相同模型，不同输入）
- 任何被多次调用且输入形状相似的函数

JIT 什么时候会有问题：

- 控制流依赖于被追踪的值（如 `if x > 0`，其中 `x` 是数组）
- 只运行一次的函数（编译开销超过运行时间）
- 调试时（追踪机制会隐藏实际执行过程）

控制流的限制是真实的。`jax.lax.cond` 替代 `if/else`。`jax.lax.scan` 替代 `for` 循环。这不是可选的——这是编译的代价。

### 2.5 vmap：自动向量化

你写一个处理单个样本的函数：

```python
def predict_single(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 将其提升为处理批次的函数：

```python
batch_predict = jax.vmap(predict_single, in_axes=(None, 0))
```

`in_axes=(None, 0)` 表示：`params` 不需要批量（所有样本共享），`x` 沿第 0 维批量。不需要手写循环。不需要重塑形状。不需要手动管理批次维度。JAX 自动推断并生成融合的向量化代码。

这不是语法糖。`vmap` 生成的融合向量化代码比 Python 循环快 10-100 倍。而且它可以与 `jit` 和 `grad` 自由组合：

```python
# 一行代码：计算每个样本的梯度
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

在 PyTorch 中做同样的事情极其困难，通常需要手动拆分批次、逐个求梯度、再拼接。

### 2.6 pmap：跨设备数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 将函数复制到所有可用设备（GPU/TPU）上，并自动分割批次。函数内部通过 `jax.lax.pmean` 和 `jax.lax.psum` 同步梯度。

Google 用 `pmap`（及其后继 `shard_map`）在数千块 TPU v5e 芯片上训练 Gemini。编程模型极其简洁：写单设备版本，用 `pmap` 包装，完成。

### 2.7 PRNG：显式随机数管理

JAX 没有全局随机状态。每个随机操作都需要一个显式的 PRNG 密钥：

```python
key = jax.random.PRNGKey(42)  # 种子
key1, key2 = jax.random.split(key)  # 分出两个独立子密钥
w = jax.random.normal(key1, shape=(784, 256))
```

一开始很烦。但它保证了跨设备、跨编译的可复现性——PyTorch 的 `torch.manual_seed` 在多 GPU 环境下无法保证这一点。

```
PyTorch 随机数管理（隐式全局状态）：

  torch.manual_seed(42)
  a = torch.randn(3)    # 由全局种子决定
  b = torch.randn(3)    # 由全局状态推进决定
  # 多 GPU 时行为不可预测

JAX 随机数管理（显式密钥）：

  key = jax.random.PRNGKey(42)
  key1, key2 = jax.random.split(key)
  a = jax.random.normal(key1, shape=(3,))  # 由 key1 决定
  b = jax.random.normal(key2, shape=(3,))  # 由 key2 决定
  # 永远可复现
```

### 2.8 lax 底层操作

`jax.lax` 是 JAX 的底层操作原语集合，提供了更精细的控制。`lax` 操作通常比 `jax.numpy` 对应物更快，因为它们直接映射到 XLA 内核：

```python
# jax.lax 比 jnp 更底层、更高效
from jax import lax

x = lax.conv_general_dilated(...)     # 卷积操作
x = lax.dynamic_slice(...)            # 动态切片
x = lax.while_loop(cond, body, init)  # 条件循环
```

在实际使用中，你大多数时候用 `jax.numpy` 和 `jax.nn`。`lax` 用于需要极致控制的场景——如自定义控制流（`lax.cond`、`lax.scan`、`lax.while_loop`）和底层线性代数操作。

### 2.9 Flax：JAX 的神经网络库

JAX 提供原语。Flax（Google 出品）提供人体工学——类似 PyTorch `nn.Module` 的层级抽象：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与 PyTorch 类似，但 `params` 与 `model` 是分离的。`model.init()` 创建参数，`model.apply(params, x)` 运行前向传播。模型对象本身没有状态。

Flax 是 Google 内部训练大模型的标准选择。如果你要用 JAX 做严肃的深度学习工作，Flax 几乎是必经之路。

### 2.10 Pytree：通用数据结构

JAX 操作的对象称为 "pytree"——列表、元组、字典和数组的嵌套组合。你的模型参数就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 变换——`grad`、`jit`、`vmap`——都知道如何遍历 pytree。`jax.tree.map(f, tree)` 对每个叶子节点应用 `f`。优化器就是这样一次性更新所有参数的：

```python
# 对每个参数和对应梯度执行：新参数 = 旧参数 - 学习率 × 梯度
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

不需要 `.parameters()` 方法。不需要参数注册。树结构就是模型本身。

### 2.11 什么时候用 JAX，什么时候用 PyTorch

| 因素 | JAX | PyTorch |
|------|-----|---------|
| TPU 支持 | 一等公民（Google 同时构建了两者） | 社区维护（torch_xla） |
| GPU 支持 | 良好（通过 XLA 的 CUDA 支持） | 最佳（原生 CUDA） |
| 调试体验 | 困难（追踪 + 编译） | 简单（急切执行，逐行调试） |
| 生态系统 | 研究导向（Flax、Equinox） | 庞大（HuggingFace、torchvision 等） |
| 人才市场 | 小众（Google/DeepMind/Anthropic） | 主流（到处都有） |
| 大规模训练 | 优势明显（XLA、pmap、mesh） | 良好（FSDP、DeepSpeed） |
| 原型开发速度 | 较慢（函数式开销） | 更快（修改即运行） |
| 生产推理 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |

诚实的答案：**除非你有明确理由，否则用 PyTorch。** 这些理由包括——使用 TPU、需要逐样本梯度、需要在超大规模上多设备训练、在 Google/DeepMind/Anthropic 工作。

---

## 3. 从零实现

本课的代码实践分为三个递进阶段，逐步展示 JAX 的核心变换。

### 第 1 步：从函数式自动微分开始

```python
import jax
import jax.numpy as jnp
from jax import random

# 定义一个纯函数
def f(x):
    return x ** 3

# jax.grad 将函数变换为其导函数
df = jax.grad(f)
d2f = jax.grad(df)

x_val = 2.0
print(f"f({x_val})   = {f(x_val)}")     # 8.0
print(f"f'({x_val})  = {df(x_val)}")     # 12.0
print(f"f''({x_val}) = {d2f(x_val)}")    # 12.0
```

关键点：`jax.grad` 返回的梯度本身是一个函数。你可以对它再调用 `jax.grad`，得到二阶导数、三阶导数……这种组合性是 PyTorch 的 `.backward()` 无法直接提供的。

### 第 2 步：用 vmap 实现自动向量化

```python
key = random.PRNGKey(42)
k1, k2 = random.split(key)

params = {'w': random.normal(k1, (4,)), 'b': 0.0}

# 写一个处理单个样本的函数
def predict_single(params, x):
    return jnp.dot(params['w'], x) + params['b']

# vmap: 自动将其提升为批次处理
batch_x = random.normal(k2, (8, 4))
batch_predict = jax.vmap(predict_single, in_axes=(None, 0))
results = batch_predict(params, batch_x)

print(f"输入形状: {batch_x.shape}")   # (8, 4)
print(f"输出形状: {results.shape}")   # (8,)
```

### 第 3 步：用 JIT 编译加速计算

```python
import time

x = random.normal(random.PRNGKey(0), (1000, 1000))

def heavy_computation(x):
    for _ in range(10):
        x = jnp.dot(x, x)
        x = x / jnp.linalg.norm(x)
    return x

fast_fn = jax.jit(heavy_computation)
_ = fast_fn(x)  # 预热（第一次调用触发编译）

# 对比速度
start = time.perf_counter()
for _ in range(10):
    _ = heavy_computation(x)
eager_time = time.perf_counter() - start

start = time.perf_counter()
for _ in range(10):
    _ = fast_fn(x).block_until_ready()
jit_time = time.perf_counter() - start

print(f"Python 解释执行: {eager_time:.4f}s")
print(f"JIT 编译执行:    {jit_time:.4f}s")
print(f"加速比:          {eager_time / jit_time:.1f}x")
```

JIT 的加速在计算密集型操作上最显著。`block_until_ready()` 是必须的——JAX 使用异步分发，不等待就无法测量真实执行时间。

### 第 4 步：用 JAX + Optax 训练 MNIST

完整代码见 `code/main.py`。以下是最关键的片段——函数式训练步骤：

```python
import optax

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),   # 梯度裁剪
    optax.adam(learning_rate=1e-3),   # Adam 优化器
)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

注意这段代码中没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新过程是一个组合函数调用：计算梯度、Optax 变换梯度、应用到参数——全部在 `train_step` 内完成。而且整个函数被 `@jax.jit` 编译为单个 XLA 内核。

---

## 4. 工业工具

### 4.1 Flax — JAX 的标准神经网络库

```python
import flax.linen as nn
import jax

class TransformerBlock(nn.Module):
    """Flax 风格的 Transformer 块。"""
    num_heads: int = 8
    d_model: int = 512

    @nn.compact
    def __call__(self, x):
        # 多头注意力
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
        )
        x = x + attn(x, x)

        # 前馈网络
        x = x + nn.Dense(self.d_model * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model)(x)

        # 层归一化
        x = nn.LayerNorm()(x)
        return x

model = TransformerBlock()
# 初始化参数
variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 128, 512)))
params = variables['params']

# 前向传播（参数显式传入）
output = model.apply({'params': params}, jnp.ones((1, 128, 512)))
print(f"输出形状: {output.shape}")  # (1, 128, 512)
```

Flax 的 `nn.Module` 与 PyTorch 类似，但有一个关键差异：`params` 是分离的。`model.init()` 创建参数，`model.apply({'params': params}, x)` 运行前向传播。模型对象本身是无状态的。

### 4.2 Optax — 可组合的梯度变换

Optax 是 JAX 生态的标准优化器库。它将梯度变换（Adam、SGD、裁剪等）与参数更新解耦，使组合变得极其简单：

```python
import optax

# 预热 + 余弦衰减学习率
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=2000,
    decay_steps=50000,
    end_value=1e-6,
)

# 优化器链：梯度裁剪 + 权重衰减 + Adam
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减——全部作为链条中的变换组合。每个变换接收梯度、修改梯度、传递给下一个变换。不需要一个庞大的优化器类。

### 4.3 性能对比

| 操作 | PyTorch | JAX | JAX 优势 |
|------|---------|-----|----------|
| 单次前向传播 | 快 | 快 | 平手 |
| 训练循环（1000 步） | 正常 | 更快（JIT 编译后） | 2-10x |
| 梯度计算 | `loss.backward()` | `jax.grad(loss)(params)` | 平手 |
| 逐样本梯度 | 极慢（手写循环） | 快（vmap + grad 组合） | 100x+ |
| 多 GPU 训练 | `DataParallel` | `pmap` | 可比 |
| TPU 训练 | 有限支持（torch_xla） | 原生支持 | 显著优势 |

---

## 5. 知识连线

本课学习的 JAX 函数式框架，是后续深度学习课程中理解现代训练基础设施的关键：

- **阶段 04（计算机视觉）**：CNN 的训练可以理解为一个纯函数变换——输入图片、参数、标签，输出损失和梯度。JAX 的函数式视角让你更清晰地看到数据流过卷积层、池化层、全连接层的过程。
- **阶段 07（Transformer 深入）**：大规模 Transformer（如 Gemini）几乎都用 JAX 训练。理解 `jit` 编译如何将整个 Transformer 前向+反向传播融合为单个内核，是理解这些模型为什么能高效训练的基础。
- **阶段 10（大语言模型从零）**：当你从零实现 GPT 时，用 JAX 的函数式方式写一个纯函数版本的 Transformer 训练步骤，会让你对"训练循环本质上是一个可编译的程序"有深刻体会。

---

## 6. 工程最佳实践

### 6.1 JIT 编译的正确使用

- 训练步骤函数**必须**加 `@jax.jit`——同样的计算重复数千次，编译开销可以忽略
- 推理函数也建议加 `@jax.jit`
- 一次性计算（如只调用一次的初始化函数）**不要**加 JIT——编译开销超过运行时间
- 用 `jax.debug.print("{}", x)` 替代 `print(x)`——JIT 内的 `print` 在追踪时执行，不在运行时执行

### 6.2 PRNG 密钥管理

```python
# 每个轮次使用独立子密钥
for epoch in range(n_epochs):
    key, subkey = jax.random.split(key)
    perm = random.permutation(subkey, len(X_train))

# 需要多个随机操作时，一次性 split
key, k1, k2, k3 = jax.random.split(key, 4)
w1 = jax.random.normal(k1, shape)
w2 = jax.random.normal(k2, shape)
noise = jax.random.normal(k3, shape)
```

**永远不要重用密钥。** 两个操作使用相同密钥会产生完全相同的"随机"数。

### 6.3 Pytree 参数管理

```python
# 用 jax.tree.map 批量更新参数
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)

# 统计参数数量
n_params = sum(p.size for p in jax.tree.leaves(params))
print(f"总参数量: {n_params:,}")
```

### 6.4 中文场景特别建议

- JAX 的 `jax.numpy` 完全兼容 NumPy 的中文数组处理方式——中文字符串在 JAX 中以 `jnp.array` 形式存储时需要先编码为字节
- 如果你在中国大陆使用 GPU，安装 JAX 时可能需要设置镜像源：`pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`
- JAX 默认预分配 GPU 75% 的显存。如果需要更精细控制，设置环境变量 `XLA_PYTHON_CLIENT_PREALLOCATE=false`

### 6.5 踩坑经验

- **第一次 JIT 调用慢是正常的**——编译需要时间。在基准测试前务必先"预热"
- **JIT 内不能用 Python 的 `if/else` 处理数组值**——用 `jax.lax.cond` 替代
- **JIT 内不能用 Python 的 `for` 循环处理数组值**——用 `jax.lax.scan` 或 `jax.lax.fori_loop` 替代
- **忘记 `.block_until_ready()`**——JAX 使用异步分发，不等待就无法测量真实执行时间
- **全局变量在追踪时被捕获**——JIT 编译后修改全局变量不会生效。一切必须通过参数传递

---

## 7. 常见错误

### 错误 1：在 JIT 函数内修改数组

**现象：** 运行时报 `ConcreteArray` 错误，提示不允许原地修改。

**原因：** JAX 数组是不可变的。JIT 编译基于"相同输入相同输出"的假设。如果允许原地修改，这个假设就不成立。

**修复：**

```python
# 错误写法
x[0] = 5.0

# 正确写法
x = x.at[0].set(5.0)
```

### 错误 2：在 JIT 函数内使用 Python 的 print

**现象：** `print` 语句只在第一次调用（追踪时）执行一次，之后不再执行。看起来像"消失了"。

**原因：** `print` 是 Python 的副作用操作。JIT 追踪函数时执行一次 `print`，但编译后的代码中不会包含它。

**修复：**

```python
# 错误写法（追踪时执行一次，之后不再执行）
@jax.jit
def f(x):
    print(f"x = {x}")
    return x ** 2

# 正确写法（每次调用都打印）
@jax.jit
def f(x):
    jax.debug.print("x = {}", x)
    return x ** 2
```

### 错误 3：在 JIT 函数内使用 Python 的 if/for 处理数组

**现象：** 报 `ConcretizationTypeError`，提示"在追踪时遇到了抽象的数组值"。

**原因：** `if x > 0` 中的 `x` 在追踪时是抽象的（没有具体值），Python 的 `if` 无法处理抽象值。

**修复：**

```python
# 错误写法
@jax.jit
def f(x):
    if x > 0:
        return x
    else:
        return -x

# 正确写法：用 jax.lax.cond
@jax.jit
def f(x):
    return jax.lax.cond(x > 0, lambda _: x, lambda _: -x, None)
```

### 错误 4：重用 PRNG 密钥

**现象：** 两个"随机"操作产生了完全相同的输出。调试时很难发现。

**原因：** JAX 的 PRNG 是确定性的——相同密钥产生相同序列。如果你重用同一个密钥，就会得到相同的结果。

**修复：**

```python
# 错误写法：同一个密钥生成两次"随机"数
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (3,))  # [0.497, -0.139, 0.648]
b = jax.random.normal(key, (3,))  # [0.497, -0.139, 0.648] 完全相同！

# 正确写法：先 split，再使用
key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)
a = jax.random.normal(k1, (3,))  # [0.497, -0.139, 0.648]
b = jax.random.normal(k2, (3,))  # [-0.008, 0.289, -0.813] 不同
```

### 错误 5：忘记 JAX 默认预分配 GPU 显存

**现象：** 导入 JAX 后，PyTorch 或其他框架无法分配显存。

**原因：** JAX 默认预分配 GPU 75% 的显存，导致其他框架没有足够空间。

**修复：**

```bash
# 在 Python 代码之前设置环境变量
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# 或者限制预分配比例
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

---

## 8. 面试考点

### Q1：解释 JAX 的"纯函数"概念，以及为什么 JIT 编译要求纯函数？（难度：⭐⭐）

**参考答案：**

JAX 的纯函数是指输出完全由输入决定的函数——没有全局状态修改、没有副作用、没有不带显式密钥的随机数。JIT 编译要求纯函数是因为：编译器在第一次调用时"追踪"函数，用抽象值代替真实数据构建计算图。如果函数依赖外部状态，追踪结果就无法代表后续调用的真实行为，编译出的代码就是错误的。纯函数的"相同输入相同输出"保证了追踪一次就足以代表所有调用。

### Q2：`jax.vmap` 和 PyTorch 的批处理方式有什么根本区别？（难度：⭐⭐）

**参考答案：**

PyTorch 中，通常在写函数时就考虑批次维度（输入形状为 `(batch, ...)`），或者用 `torch.utils.data.DataLoader` 自动添加批次。`jax.vmap` 的方式完全不同：你写一个处理单个样本的函数（不考虑批次维度），然后用 `vmap` 自动提升为批次版本。这意味着：(1) 单样本函数更易测试和调试；(2) `vmap` 生成的融合向量化代码比手写 Python 循环快 10-100 倍；(3) `vmap` 可以与 `grad` 组合，一行代码计算逐样本梯度——在 PyTorch 中这几乎不可能优雅实现。

### Q3：为什么 JAX 数组是不可变的？这带来了什么好处？（难度：⭐⭐）

**参考答案：**

不可变性是 JAX 函数式设计的根基。好处有三：(1) `grad`、`jit`、`vmap` 等变换依赖于"函数无副作用"的保证——如果允许原地修改，这些变换就无法安全组合；(2) 编译器可以自由地重排、融合、并行化操作，因为没有状态依赖；(3) 并发安全——多个设备可以同时读取同一个数组而不需要加锁。代价是修改数组时必须用 `x.at[i].set(v)` 创建新数组，但这在 XLA 编译后会被优化为等价的原地操作。

### Q4：用 JAX 训练一个模型时，参数更新的流程是什么？与 PyTorch 有何不同？（难度：⭐⭐⭐）

**参考答案：**

JAX 中的参数更新流程：(1) `jax.value_and_grad(loss_fn)(params, x, y)` 计算损失和梯度；(2) `optimizer.update(grads, opt_state, params)` 计算参数更新量（Optax 内部处理 Adam 动量、方差估计等）；(3) `optax.apply_updates(params, updates)` 应用更新，返回新参数。关键区别：整个过程是一个纯函数，参数和优化器状态都是显式传递的 pytree。PyTorch 中，`model.parameters()` 返回可变引用，`optimizer.step()` 原地修改模型权重，`optimizer.state` 存储在优化器内部。JAX 的方式使得整个训练步骤可以被编译为单个 XLA 内核。

### Q5：如果你要在 JAX 中实现一个需要根据输入值选择不同计算路径的函数（如条件丢弃），如何处理 JIT 的限制？（难度：⭐⭐⭐）

**参考答案：**

JIT 内不能用 Python 的 `if/else` 处理数组值，因为追踪时数组是抽象的。解决方案是使用 `jax.lax` 的控制流原语：(1) `jax.lax.cond(pred, true_fn, false_fn)` 替代 `if/else`；(2) `jax.lax.select(pred, true_val, false_val)` 类似三元运算符；(3) 对于条件丢弃，可以用 `jax.lax.cond` 在训练和推理模式之间切换。更实际的做法是：在训练循环外部决定是否启用丢弃，通过参数传递给 JIT 函数，避免在 JIT 内处理动态控制流。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| JIT | "即时编译" | JAX 在第一次调用时追踪函数，编译为 XLA 机器码，后续调用跳过 Python 直接运行编译版本 |
| XLA | "让 JAX 变快的东西" | Accelerated Linear Algebra——Google 的编译器，将计算图融合为优化的 GPU/TPU 内核 |
| 纯函数 | "没有副作用" | 输出完全由输入决定——没有全局状态修改、没有副作用、没有不带显式密钥的随机数 |
| vmap | "自动批处理" | 将处理单个样本的函数自动转换为处理批次的函数，无需重写代码 |
| pmap | "自动并行" | 将函数复制到多个设备并分割输入批次，实现数据并行 |
| Pytree | "嵌套字典" | 列表、元组、字典和数组的嵌套结构，JAX 可以遍历和变换它 |
| 追踪 (Tracing) | "记录计算过程" | JAX 用抽象值执行函数，记录操作序列构建计算图，不计算真实结果 |
| 函数式自动微分 | "函数的 grad" | 通过变换函数来计算导数，而不是在张量上附加梯度存储 |
| PRNG 密钥 | "随机种子" | JAX 中随机数操作的显式密钥——每次使用前必须 split，不能重用 |
| Flax | "JAX 的 nn.Module" | Google 的 JAX 神经网络库，提供层级抽象，同时保持状态显式管理 |

---

## 📚 小结

JAX 将深度学习编程从面向对象转向函数式范式：参数是显式传递的 pytree，梯度是函数变换而非张量属性，训练步骤是可编译的纯函数。这种设计的代价是更陡峭的学习曲线和更严格的状态管理，但回报是 JIT 编译的极致加速、`vmap` 的优雅向量化、以及跨设备的天然可复现性。

你通过构建一个 3 层 MLP 的完整训练流程，体验了 JAX 从数据加载、参数初始化、前向传播、损失计算、梯度变换到参数更新的完整管道。对比你在第 10 课用 PyTorch 迷你框架实现的版本，两种范式各有其工程哲学。

下一课我们将深入 Transformer 架构——理解注意力机制如何让序列中的每个位置都能直接与其他位置交互，这是取代循环神经网络的根本原因。

---

## ✏️ 练习

1. 【理解】用自己的话向一个只用过 PyTorch 的同事解释：为什么 JAX 要求数组不可变？用一个具体的场景（如训练循环中的梯度更新）说明不可变性如何使 JIT 编译成为可能。写 200 字以内。

2. 【实现】修改 `code/main.py` 中的 MLP，为每层添加 Dropout。JAX 的 Dropout 需要一个 PRNG 密钥——在前向传播中传递密钥，用 `jax.random.split` 为每层生成独立的丢弃掩码。对比有无 Dropout 的测试准确率。

3. 【实验】用 `jax.vmap` 实现一个函数，批量计算 32 张 MNIST 图片的梯度范数。分析哪些样本的梯度最大，思考这与样本的"难度"有什么关系。

4. 【思考】在 JAX 中，`@jax.jit` 装饰器将整个训练步骤编译为单个 XLA 内核。这与 PyTorch 的急切执行模式相比，在哪些场景下优势最大？在哪些场景下反而不如急切执行？列举至少 2 个场景并解释原因。

5. 【实验】用 `jax.profiler` 或 TensorBoard 对比 JIT 编译前后 `train_step` 的性能。追踪操作的耗时分布——XLA 融合了哪些操作？哪些操作是性能瓶颈？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| JAX 入门完整代码 | `code/main.py` | 包含 jax.grad、vmap、jit、PRNG、MNIST 训练的完整可运行示例 |
| JAX 优化器配置指南 | `outputs/prompt-jax-guide.md` | 根据场景推荐 Optax 优化器链和学习率调度的提示词模板 |

---

## 📖 参考资料

1. [论文] Bradbury et al. "JAX: composable transformations of Python+NumPy programs". 2018. https://github.com/google/jax
2. [官方文档] JAX Documentation. https://jax.readthedocs.io/
3. [官方文档] Flax Documentation. https://flax.readthedocs.io/
4. [官方文档] Optax Documentation. https://optax.readthedocs.io/
5. [论文] Kidger. "On Neural Differential Equations". 2022. https://github.com/patrick-kidger/equinox
6. [论文] Vaswani et al. "Attention Is All You Need". NeurIPS, 2017. https://arxiv.org/abs/1706.03762

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
