# 卷积从零实现

> 卷积不是"更好的全连接层"。它是唯一一种同时具备平移等变性和参数共享的运算——这两个性质让图像识别从不可能变为可能。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 03（深度学习核心）、阶段 04 · 01（图像基础）
**预计时间：** ~75 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 03 · 11（PyTorch 入门）— PyTorch 中的 `nn.Conv2d` 就是本课实现的加速版本

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零用 NumPy 实现 2D 卷积，包括嵌套循环版和向量化 `im2col` 版
- [ ] 对任意输入尺寸、卷积核大小、填充和步长的组合，手算输出空间尺寸并验证公式 `(H - K + 2P) / S + 1`
- [ ] 手工设计卷积核（边缘检测、锐化、模糊、Sobel），并解释每个核产生特定激活模式的原因
- [ ] 将多层卷积堆叠为特征提取器，理解感受野大小与网络深度的关系
- [ ] 实现最大池化和平均池化，比较两者的下采样特性差异

---

## 1. 问题

一张 224×224 的 RGB 图像有 224 × 224 × 3 = 150,528 个像素值。如果用一个全连接层处理——每个神经元需要 150,528 个输入权重。一个包含 1000 个隐藏单元的层就是 1.5 亿个参数，这还没开始学习任何有用的特征。

更致命的问题不只是参数量。全连接层把每个像素位置当作独立的输入——左上角的猫和右下角的猫对它来说是完全不同的模式。但图像天然具有**平移不变性**：把猫向右移动 3 个像素，它仍然是猫。全连接层不具备这个能力。

图像处理需要两个关键性质：

- **平移等变性（Translation Equivariance）**：输入平移，输出也平移。检测到边缘的卷积核在图像的任何位置都能检测到边缘。
- **参数共享（Parameter Sharing）**：同一个卷积核在所有位置复用。无论猫在图像的哪个位置，同一个 3×3 卷积核都能检测到它。

全连接层一个都不具备。卷积两个都给。

卷积并不是为深度学习发明的——JPEG 压缩、Photoshop 的高斯模糊、工业视觉的边缘检测、所有音频滤波器，底层都是卷积。2012 年到 2020 年 CNN 统治 ImageNet 的根本原因很简单：对于"相邻像素相关、同一模式可以出现在任何位置"的数据，卷积就是正确的归纳偏置。

---

## 2. 概念

### 2.1 一个核，滑动窗口

2D 卷积的操作很直觉：拿一个小型权重矩阵（**卷积核**，也叫滤波器），在输入上滑动，每个位置计算逐元素乘积之和。这个和就是一个输出像素。

```
输入 X (5×5):                  卷积核 W (3×3):

  1  2  0  1  2                 1  0 -1
  0  1  3  1  0                 2  0 -2
  2  1  0  2  1                 1  0 -1
  1  0  2  1  3
  2  1  1  0  1

卷积核在每个 3×3 窗口上滑动，逐元素相乘再求和。
输出 Y 是 3×3:

  Y[0,0] = sum( W × X[0:3, 0:3] ) = 1×1+0×2+(-1)×0 + 2×0+0×1+(-2)×3 + 1×2+0×1+(-1)×0 = 1+0+0+0+0-6+2+0+0 = -3
  Y[0,1] = sum( W × X[0:3, 1:4] ) = ...
  ...
```

整个卷积就一句话：**共享权重、局部连接、滑动窗口**。其余都是配套工程。

### 2.2 输出尺寸公式

给定输入空间尺寸 `H`、卷积核大小 `K`、填充 `P`、步长 `S`：

$$
H_{out} = \left\lfloor \frac{H - K + 2P}{S} \right\rfloor + 1
$$

这个公式会伴随你整个 CNN 架构设计生涯。下面是一些典型场景：

| 场景 | H | K | P | S | H_out |
|---|---|---|---|---|---|
| 无填充卷积 | 32 | 3 | 0 | 1 | 30 |
| Same 卷积（保持尺寸） | 32 | 3 | 1 | 1 | 32 |
| 步长 2 下采样 | 32 | 3 | 1 | 2 | 16 |
| 2×2 池化 | 32 | 2 | 0 | 2 | 16 |
| 大感受野下采样 | 32 | 7 | 3 | 2 | 16 |

**"Same 填充"** 指选择 P 使得当 S=1 时 H_out == H。对于奇数 K，P = (K-1)/2。这就是 3×3 卷积核如此流行的原因——它是拥有中心像素的最小奇数核。

### 2.3 填充（Padding）

不做填充时，每次卷积都会缩小特征图。堆叠 20 层卷积后，224×224 的图像会缩小到 184×184——不仅浪费了边界像素的计算，残差连接还需要匹配输入输出的形状。

```
在 5×5 输入上做零填充（P=1）：

  0  0  0  0  0  0  0
  0  1  2  0  1  2  0
  0  0  1  3  1  0  0
  0  2  1  0  2  1  0    填充后，卷积核可以居中在
  0  1  0  2  1  3  0    边界像素上，仍然有足够的
  0  2  1  1  0  1  0    邻域值参与计算。
  0  0  0  0  0  0  0
```

常见的填充模式：

| 模式 | 做法 | 使用场景 |
|---|---|---|
| `zero` | 边界补零 | 最常见，绝大多数 CNN |
| `reflect` | 镜像翻转边界 | 生成模型，避免硬边界 |
| `replicate` | 复制最近的边界值 | OpenCV 默认 |
| `circular` | 环绕填充 | 周期性数据 |

### 2.4 步长（Stride）

步长是卷积核每次滑动的步数。`stride=1` 是默认值，`stride=2` 将空间尺寸减半。现代架构（ResNet、ConvNeXt、MobileNet）普遍用步长卷积替代最大池化来做下采样。

```
步长 1（5×5 输入，3×3 核）:

  起始位置: (0,0) (0,1) (0,2)     → 输出第 0 行
            (1,0) (1,1) (1,2)     → 输出第 1 行
            (2,0) (2,1) (2,2)     → 输出第 2 行

  输出: 3×3

步长 2（同一输入）:

  起始位置: (0,0) (0,2)            → 输出第 0 行
            (2,0) (2,2)            → 输出第 1 行

  输出: 2×2
```

### 2.5 多通道输入

真实图像有 3 个通道（RGB）。对 RGB 输入做 3×3 卷积，实际是 3×3×3 的体积运算：每个通道一个 3×3 的切片，在每个空间位置上跨所有通道求和，再加偏置。

```
输入:    (C_in,  H,  W)        3 × 5 × 5
卷积核:  (C_in,  K,  K)        3 × 3 × 3（一个卷积核）
输出:    (1,     H', W')       2D 特征图

要产生 C_out 个输出通道，就需要 C_out 个卷积核：

权重:    (C_out, C_in, K, K)   例如 64 × 3 × 3 × 3
输出:    (C_out, H', W')       64 × 3 × 3

参数量: C_out × C_in × K × K + C_out   （+C_out 是偏置项）
```

最后一行公式是设计模型时必算的。一个 64 通道的 3×3 卷积层，输入 3 通道，参数量为 `64 × 3 × 3 × 3 + 64 = 1,792`。非常轻量。对比一下：如果用全连接层处理同样的输入输出，参数量约为 224 × 224 × 3 × 64 ≈ 960 万——相差 5000 倍。

### 2.6 im2col 技巧

嵌套循环好读但慢。GPU 擅长大规模矩阵乘法。核心思路：把输入中每个感受野窗口展开为大矩阵的一列，把卷积核展平为一行，整个卷积就变成了一次矩阵乘法（GEMM）。

```
im2col 变换流程:

  输入 X            im2col 提取所有窗口
  (C_in, H, W)  →  列矩阵 (C_in×K×K, H_out×W_out)
                                          ↓
  权重 W            展平为二维
  (C_out, C_in, K, K) → (C_out, C_in×K×K)  ↓
                                          ↓
                              矩阵乘法 w_flat @ cols
                                          ↓
                              输出 (C_out, H_out×W_out)
                              → reshape 为 (C_out, H_out, W_out)
```

所有生产级卷积库（cuDNN、oneDNN、MIOpen）的底层实现都是 im2col 的某种变体，加上缓存分块、Winograd 变换、FFT 卷积等优化。理解了 im2col，就理解了卷积加速的核心。

### 2.7 感受野（Receptive Field）

单个 3×3 卷积看 9 个输入像素。堆叠两个 3×3 卷积，第二层的一个神经元实际看的是 5×5 的输入区域。三个 3×3 卷积覆盖 7×7。通式：

```
L 层 K×K 卷积（步长 1）后的感受野:
  RF = 1 + L × (K - 1)

考虑步长时:
  感受野沿各层的累积步长乘积增长
```

这就是"3×3 一路到底"（VGG、ResNet、ConvNeXt）的设计依据：两个 3×3 卷积看的区域和一个 5×5 卷积一样大，但参数更少（2 × 9 = 18 vs 25），而且多了一次非线性变换，表达能力更强。

---

## 3. 从零实现

### 第 1 步：零填充

最基本的原语——在数组周围补零。

```python
def pad2d(x, p):
    """在最后两个维度周围填充 p 圈零。"""
    if p == 0:
        return x
    h, w = x.shape[-2:]
    out = np.zeros(x.shape[:-2] + (h + 2 * p, w + 2 * p), dtype=x.dtype)
    out[..., p:p + h, p:p + w] = x
    return out

x = np.arange(9).reshape(3, 3)
print(x)
print()
print(pad2d(x, 1))
```

```text
[[0 1 2]
 [3 4 5]
 [6 7 8]]

[[0 0 0 0 0]
 [0 0 1 2 0]
 [0 3 4 5 0]
 [0 6 7 8 0]
 [0 0 0 0 0]]
```

`x.shape[:-2]` 这个切片技巧使得同一函数兼容 `(H, W)`、`(C, H, W)`、`(N, C, H, W)` 三种形状。

### 第 2 步：嵌套循环卷积

参考实现——慢但逻辑透明，是 `torch.nn.functional.conv2d` 的原理版本。

```python
def conv2d_naive(x, w, b=None, stride=1, padding=0):
    c_in, h, w_in = x.shape
    c_out, c_in_w, kh, kw = w.shape
    assert c_in == c_in_w

    x_pad = pad2d(x, padding)
    h_out = output_size(h, kh, padding, stride)
    w_out = output_size(w_in, kw, padding, stride)

    out = np.zeros((c_out, h_out, w_out), dtype=np.float32)
    for oc in range(c_out):           # 遍历每个输出通道
        for i in range(h_out):        # 遍历输出的每一行
            for j in range(w_out):    # 遍历输出的每一列
                hs = i * stride
                ws = j * stride
                patch = x_pad[:, hs:hs + kh, ws:ws + kw]
                out[oc, i, j] = np.sum(patch * w[oc])
        if b is not None:
            out[oc] += b[oc]
    return out
```

四层嵌套循环（输出通道、行、列，加上隐式的通道求和）。这是检验所有更快实现的基准真相。

### 第 3 步：用手工设计的核验证

构建一个垂直 Sobel 核，应用到合成的阶跃图像上，观察垂直边缘被激活。

```python
def synthetic_step_image(size=16):
    """生成左黑右白的阶跃图像。"""
    img = np.zeros((1, size, size), dtype=np.float32)
    img[:, :, size // 2:] = 1.0
    return img

sobel_x = np.array([
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]
], dtype=np.float32)[None]  # 形状 (1, 1, 3, 3)

x = synthetic_step_image()
y = conv2d_naive(x, sobel_x, padding=1)
print(y[0].round(1))
```

在列 7（从暗到亮的边界）处应该出现较大的正值，其余位置为零。这一行输出就是你验证数学是否正确的检查点。

### 第 4 步：im2col 变换

把输入的每个感受野窗口展平为列矩阵的一列。

```python
def im2col(x, kh, kw, stride=1, padding=0):
    c_in, h, w = x.shape
    x_pad = pad2d(x, padding)
    h_out = output_size(h, kh, padding, stride)
    w_out = output_size(w, kw, padding, stride)

    cols = np.zeros((c_in * kh * kw, h_out * w_out), dtype=x.dtype)
    col = 0
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            patch = x_pad[:, hs:hs + kh, ws:ws + kw]
            cols[:, col] = patch.reshape(-1)
            col += 1
    return cols, h_out, w_out
```

虽然外层仍然是 Python 循环，但接下来的重活是一次向量化矩阵乘法。

### 第 5 步：im2col + 矩阵乘法快速卷积

```python
def conv2d_im2col(x, w, b=None, stride=1, padding=0):
    c_out, c_in, kh, kw = w.shape
    cols, h_out, w_out = im2col(x, kh, kw, stride, padding)
    w_flat = w.reshape(c_out, -1)        # 展平卷积核
    out = w_flat @ cols                   # 一次 GEMM 完成
    if b is not None:
        out += b[:, None]
    return out.reshape(c_out, h_out, w_out)
```

验证两种实现的等价性：

```python
rng = np.random.default_rng(0)
x = rng.normal(0, 1, (3, 16, 16)).astype(np.float32)
w = rng.normal(0, 1, (8, 3, 3, 3)).astype(np.float32)
b = rng.normal(0, 1, (8,)).astype(np.float32)

y_naive = conv2d_naive(x, w, b, padding=1)
y_im2col = conv2d_im2col(x, w, b, padding=1)
print(f"max abs diff: {np.max(np.abs(y_naive - y_im2col)):.2e}")
```

`max abs diff` 应该在 `1e-5` 量级——差异来自浮点累加顺序，不是 bug。

### 第 6 步：手设计卷积核库

五个经典滤波器，展示卷积层在训练前就能表达什么。

```python
KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    "blur_3x3": np.ones((3, 3), dtype=np.float32) / 9.0,
    "sharpen":  np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "sobel_x":  np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "sobel_y":  np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
}

def apply_kernel(img2d, kernel):
    """将单通道核应用到灰度图上。"""
    x = img2d[None].astype(np.float32)   # (1, H, W)
    w = kernel[None, None]                # (1, 1, K, K)
    return conv2d_im2col(x, w, padding=1)[0]
```

对任何灰度图像：blur 使图像柔和，sharpen 增强边缘对比度，Sobel-X 激活垂直边缘，Sobel-Y 激活水平边缘。这些恰好是 AlexNet 和 VGG 第一层训练后学到的模式——因为无论任务是什么，好的图像特征提取器都需要边缘和色块检测器。

### 第 7 步：池化操作

最大池化和平均池化是最常用的下采样手段。

```python
def max_pool2d(x, pool_size=2, stride=None):
    """最大池化：在窗口内取最大值。"""
    if stride is None:
        stride = pool_size
    c, h, w = x.shape
    h_out = output_size(h, pool_size, 0, stride)
    w_out = output_size(w, pool_size, 0, stride)

    out = np.zeros((c, h_out, w_out), dtype=x.dtype)
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            window = x[:, hs:hs + pool_size, ws:ws + pool_size]
            out[:, i, j] = np.max(window, axis=(1, 2))
    return out

def avg_pool2d(x, pool_size=2, stride=None):
    """平均池化：在窗口内取平均值。"""
    if stride is None:
        stride = pool_size
    c, h, w = x.shape
    h_out = output_size(h, pool_size, 0, stride)
    w_out = output_size(w, pool_size, 0, stride)

    out = np.zeros((c, h_out, w_out), dtype=x.dtype)
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            window = x[:, hs:hs + pool_size, ws:ws + pool_size]
            out[:, i, j] = np.mean(window, axis=(1, 2))
    return out
```

| 池化类型 | 特点 | 适用场景 |
|---|---|---|
| 最大池化 | 保留最强激活，引入平移不变性 | 分类网络的中间层下采样 |
| 平均池化 | 平滑局部响应，保留全局趋势 | 全局平均池化（GAP）替代全连接层 |

---

## 4. 工业工具

### 4.1 PyTorch 内置实现

```python
import torch
import torch.nn as nn

# 创建一个标准的 2D 卷积层
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
print(conv)
print(f"权重形状: {tuple(conv.weight.shape)}   # (C_out, C_in, K, K)")
print(f"偏置形状: {tuple(conv.bias.shape)}")
print(f"参数量:   {sum(p.numel() for p in conv.parameters())}")

x = torch.randn(8, 3, 224, 224)   # batch=8, RGB, 224×224
y = conv(x)
print(f"\n输入形状: {tuple(x.shape)}")
print(f"输出形状: {tuple(y.shape)}")
```

```text
Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
权重形状: (64, 3, 3, 3)   # (C_out, C_in, K, K)
偏置形状: (64,)
参数量:   1792

输入形状: (8, 3, 224, 224)
输出形状: (8, 64, 224, 224)
```

把 `padding=1` 换成 `padding=0`，输出变为 222×222。把 `stride=1` 换成 `stride=2`，输出变为 112×112。就是上面的公式。

### 4.2 PyTorch 池化层

```python
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

x = torch.randn(1, 64, 112, 112)
print(f"最大池化: {tuple(max_pool(x).shape)}")   # (1, 64, 56, 56)
print(f"平均池化: {tuple(avg_pool(x).shape)}")   # (1, 64, 56, 56)
```

### 4.3 性能对比

| 实现方式 | 速度 | 内存 | 适用场景 |
|---|---|---|---|
| 我们的 NumPy 嵌套循环版 | 极慢 | 低 | 学习理解原理 |
| 我们的 NumPy im2col 版 | 慢 | 中 | 理解 GEMM 技巧 |
| PyTorch `nn.Conv2d` | 快 | 中 | 研究 / 一般训练 |
| cuDNN 卷积 | 极快 | 低 | 生产环境 |
| FlashDepthwiseConv（MobileNet） | 极快 | 极低 | 移动端推理 |

生产环境中，`nn.Conv2d` 调用 cuDNN 的底层实现，使用 Winograd 变换（对 3×3 核）或 FFT 卷积（对大核），自动选择当前硬件上的最优算法。

---

## 5. 知识连线

本课学习的卷积操作，是后续所有计算机视觉和多模态课程的基础：

- **阶段 04 · 03（CNN 从零）**：你将把本课的 `Conv2d`、池化层组装成完整的卷积神经网络，从零训练图像分类器
- **阶段 12 · 01（视觉编码器）**：ViT 虽然用注意力替代了卷积，但其 Patch Embedding 层本质上就是步长卷积的一种特殊形式——理解卷积是理解 ViT 的对比参照
- **阶段 07 · 02（自注意力从零）**：自注意力的 QKV 投影和卷积的权重共享形成对比——一个是全局交互，一个是局部窗口

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 学习 / 实验 | PyTorch `nn.Conv2d` | 开箱即用，有自动微分 |
| 大规模训练 | `nn.Conv2d` + cuDNN | 自动选择 Winograd 或 FFT |
| 移动端部署 | `nn.Conv2d(groups=C_in)` + `nn.Conv2d(1×1)` | 深度可分离卷积，参数量降为 1/K² |
| 高分辨率图像 | 空洞卷积（Dilated Conv） | 不降采样即可扩大感受野 |

### 6.2 中文场景特别建议

- 处理中文文档扫描件时，3×3 卷积核对汉字笔画的细节捕捉能力有限，可考虑在第一层使用 5×5 或 7×7 的大核
- OCR 场景（如 PaddleOCR）中，卷积特征提取器的输出直接接入序列模型（CTC/Attention），确保特征图高度合理（通常为原图高度的 1/4 到 1/8）
- 使用预训练的 ResNet/EfficientNet 作为中文文档理解的特征提取器时，注意 ImageNet 预训练权重的均值方差（mean=[0.485, 0.456, 0.406]），中文文档图像可能需要调整

### 6.3 踩坑经验

- 忘记 padding 导致逐层缩小——堆叠 10 层 3×3 卷积后 224×224 缩到 204×204，残差连接报形状不匹配
- `nn.Conv2d` 默认有偏置（`bias=True`），如果你后面接了 BatchNorm，偏置会被冗余吸收，应该设 `bias=False`
- 步长卷积替代池化时，输出尺寸计算要格外小心——`stride=2, kernel=3, padding=1` 给出 `floor(H/2)`，但 `stride=2, kernel=2, padding=0` 也是 `floor(H/2)`，不要混用
- 用 `groups` 参数实现深度可分离卷积时，`C_in` 和 `C_out` 都必须能被 `groups` 整除，否则直接报错

---

## 7. 常见错误

### 错误 1：忘记做填充导致尺寸缩小

**现象：** 堆叠多层卷积后，输出尺寸逐层缩小，残差连接报 `RuntimeError: size mismatch`。

**原因：** 默认 `padding=0`（valid 卷积）时，每层卷积会缩减 2 个像素。堆叠 10 层后缩小 20 像素。残差连接要求输入输出形状完全一致。

**修复：**

```python
# ❌ 错误：无填充，输出缩小
conv = nn.Conv2d(64, 64, kernel_size=3)       # padding=0

# ✓ 正确：Same 填充，保持空间尺寸
conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # (3-1)/2 = 1
```

### 错误 2：输出尺寸公式用错——步长导致非整数

**现象：** `output_size` 返回非整数或负数，后续索引报 `IndexError`。

**原因：** 公式 `floor((H + 2P - K) / S) + 1` 中，`(H + 2P - K)` 必须能被 S 整除。否则向下取整后输出尺寸比预期小，但代码不会报错——只会产生不对齐的特征图。

**修复：**

```python
# ❌ 静默的尺寸错误
h_out = (H + 2*P - K) // S + 1   # 向下取整，可能丢弃了最后一列

# ✓ 先断言，再计算
remainder = (H + 2 * P - K) % S
assert remainder == 0, f"尺寸不对齐: ({H} + 2×{P} - {K}) % {S} = {remainder}"
h_out = (H + 2 * P - K) // S + 1
```

### 错误 3：混淆卷积核形状约定

**现象：** `np.sum(patch * w[oc])` 报 `ValueError: operands could not be broadcast together`。

**原因：** PyTorch 的权重形状是 `(C_out, C_in, K, K)`，但某些教程（包括 OpenCV）用 `(K, K, C_in, C_out)`。混用约定会导致形状不匹配。

**修复：**

```python
# ❌ 两种约定混用
w_opencv = np.random.randn(3, 3, 3, 64)      # (K, K, C_in, C_out)
c_out, c_in, kh, kw = w_opencv.shape           # 解读错误！

# ✓ 统一使用 PyTorch 约定
w_torch = w_opencv.transpose(3, 2, 0, 1)      # → (C_out, C_in, K, K)
```

### 错误 4：池化时步长与窗口大小不匹配

**现象：** 池化输出出现重叠区域或遗漏区域，特征图出现周期性伪影。

**原因：** 池化窗口大小和步长通常设置为相同值（如都为 2），如果不一致，相邻窗口会重叠，可能导致梯度被多次计算。

**修复：**

```python
# ❌ 窗口和步长不一致，产生重叠
pool = nn.MaxPool2d(kernel_size=3, stride=2)   # 重叠 1 像素

# ✓ 通常让步长等于窗口大小（无重叠下采样）
pool = nn.MaxPool2d(kernel_size=2, stride=2)   # 无重叠，输出减半
```

### 错误 5：im2col 的列矩阵形状计算错误

**现象：** 矩阵乘法报维度不匹配错误。

**原因：** `im2col` 输出列矩阵的行数应该是 `C_in × K_h × K_w`（一个感受野窗口的所有值），而不是 `C_in × H × W`。

**修复：**

```python
# ❌ 错误的列矩阵形状
cols = np.zeros((c_in * h * w, h_out * w_out))     # 行数太多

# ✓ 正确：每个感受野窗口展平为一列
cols = np.zeros((c_in * kh * kw, h_out * w_out))    # 一行 = 一个感受野
```

---

## 8. 面试考点

### Q1：为什么 CNN 使用 3×3 卷积核而不是更大的核？（难度：⭐⭐）

**参考答案：**

两个 3×3 卷积堆叠的感受野等价于一个 5×5 卷积，但参数量从 25 降到 18（2 × 9），且中间多了一次非线性变换（ReLU），表达能力更强。VGG 网络最早证明了这一点。同理，三个 3×3 等价于一个 7×7，参数从 49 降到 27。

更深层的原因：小核的参数效率更高。3×3 核有 9 个参数，每个参数都参与所有空间位置的计算（参数共享），这是大核无法比拟的效率。

### Q2：计算一个卷积层的参数量和 FLOPs。（难度：⭐⭐）

**参考答案：**

参数量 = `C_out × C_in × K × K + C_out`（偏置项）。

FLOPs（每个输出像素）= `2 × C_in × K × K`（乘加各算一次）。总 FLOPs = `2 × C_out × C_in × K × K × H_out × W_out`。

例如：`Conv2d(3, 64, 3)` 处理 224×224 输入：
- 参数量 = 64 × 3 × 3 × 3 + 64 = 1,792
- FLOPs = 2 × 64 × 3 × 3 × 3 × 224 × 224 ≈ 1.74 亿

### Q3：im2col 为什么比嵌套循环快？（难度：⭐⭐）

**参考答案：**

嵌套循环有四层 Python for 循环，每次只处理一个感受野窗口的 9 次乘加。im2col 把所有感受野窗口排列成矩阵的列，将卷积转化为一次 GEMM（通用矩阵乘法）。GEMM 在 BLAS 库和 GPU 上有极致优化（缓存分块、向量化、Tensor Core），吞吐量比 Python 循环高 3-4 个数量级。这就是为什么所有生产级卷积库底层都是某种 im2col 变体。

### Q4：最大池化和平均池化各有什么优缺点？（难度：⭐⭐）

**参考答案：**

最大池化保留局部最强激活，对微小平移有鲁棒性（特征在池化窗口内移动不影响输出），适合分类任务的中间层下采样。但它会丢失精确的空间位置信息，不适合需要精确定位的任务（如语义分割）。

平均池化保留全局趋势，用于全局平均池化（GAP）可以替代全连接层（ResNet 的做法），减少参数量。但在中间层使用时，它会稀释强激活信号。

现代趋势：用步长卷积（`stride=2`）替代池化，因为步长卷积是可学习的，而池化是固定操作。

### Q5：深度可分离卷积（Depthwise Separable Conv）为什么能大幅减少参数？（难度：⭐⭐⭐）

**参考答案：**

标准卷积的参数量是 `C_out × C_in × K × K`。深度可分离卷积把它拆成两步：

1. **深度卷积（Depthwise）**：每个通道单独做一个 K×K 卷积。参数量 = `C_in × K × K`。
2. **逐点卷积（Pointwise）**：1×1 卷积混合通道信息。参数量 = `C_out × C_in × 1 × 1`。

总参数 = `C_in × K × K + C_out × C_in`。以 `(C_in=3, C_out=64, K=3)` 为例：标准卷积 1,792 个参数，深度可分离卷积 3 × 9 + 64 × 3 = 219 个参数，减少约 8 倍。这就是 MobileNet 的核心思想。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 卷积 (Convolution) | "滑动一个滤波器" | 在每个空间位置上执行共享权重的点积运算；数学上实际是互相关（cross-correlation），但约定俗成称为卷积 |
| 卷积核 / 滤波器 | "特征检测器" | 形状为 (C_in, K, K) 的小型权重张量，与输入窗口做点积产生一个输出像素 |
| 步长 (Stride) | "跳多远" | 卷积核相邻放置位置之间的步数；步长 2 将每个空间维度减半 |
| 填充 (Padding) | "边界补零" | 在输入周围添加额外值，使卷积核能居中在边界像素上；Same 填充保持输出尺寸等于输入尺寸 |
| 感受野 (Receptive Field) | "神经元看多大范围" | 某个输出激活值所依赖的原始输入区域大小，随网络深度和步长增长 |
| im2col | "GEMM 技巧" | 将每个感受野窗口展开为矩阵的列，使卷积转化为一次矩阵乘法——所有快速卷积实现的核心 |
| 深度可分离卷积 | "每个通道一个核" | `groups == C_in` 的卷积，每个输出通道只从对应的输入通道计算；是 MobileNet 和 ConvNeXt 的基础 |
| 平移等变性 | "输入平移，输出跟着平移" | 输入平移 k 个像素，输出也平移 k 个像素；这是卷积共享权重带来的天然属性 |
| 最大池化 | "取窗口最大值" | 在池化窗口内取最大值进行下采样，保留最强激活，引入平移不变性 |
| 平均池化 | "取窗口平均值" | 在池化窗口内取平均值进行下采样，平滑局部响应，常用于全局平均池化替代全连接层 |

---

## 📚 小结

卷积的核心只有三个词：共享权重、局部连接、滑动窗口。你从零实现了嵌套循环版和 im2col 加速版两种卷积，手工设计了边缘检测和模糊核来验证数学正确性，还实现了最大池化和平均池化两种下采样操作。

下一课我们将把这些组件组装成完整的卷积神经网络——从零构建一个能识别手写数字的分类器。

---

## ✏️ 练习

1. **【理解】** 给定一个 128×128 的灰度输入，依次经过 `[Conv3x3(s=1,p=1), Conv3x3(s=2,p=1), Conv3x3(s=1,p=1), Conv3x3(s=2,p=1)]`，手算每层的输出尺寸和感受野大小。用 PyTorch 的 `nn.Sequential` 和随机权重验证你的计算。

2. **【实现】** 扩展 `conv2d_naive` 和 `conv2d_im2col`，增加 `groups` 参数。展示 `groups=C_in=C_out` 时的行为等价于深度可分离卷积中的深度卷积步骤，并验证参数量从 `C_out × C_in × K × K` 降低到 `C_in × K × K`。

3. **【实验】** 找一张真实的灰度照片（如建筑或风景），分别用 identity、blur、sharpen、sobel_x、sobel_y 核处理，打印输出并描述每种核的效果。思考：如果把这五个核作为第一层卷积的初始权重，训练后它们会变成什么样？

4. **【思考】** FlashAttention 通过 IO 感知的分块策略将标准注意力从 O(n²) 内存降到 O(n)。类比思考：im2col 技巧本质上做了什么"数据重排"来让 GEMM 替代循环？这种重排的内存开销是什么？在什么情况下 im2col 反而不如嵌套循环？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 2D 卷积从零实现 | `code/main.py` | 嵌套循环版 + im2col 加速版，含池化操作和感受野计算 |
| CNN 架构设计提示词 | `outputs/prompt-convolution-guide.md` | 给定输入尺寸、参数预算和目标感受野，自动设计 Conv2d 层堆叠方案 |

---

## 📖 参考资料

1. [论文] Dumoulin & Visin. "A guide to convolution arithmetic for deep learning". arXiv, 2016. https://arxiv.org/abs/1603.07285
2. [论文] Simonyan & Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition". ICLR, 2015. https://arxiv.org/abs/1409.1556
3. [论文] Howard et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications". arXiv, 2017. https://arxiv.org/abs/1704.04861
4. [官方文档] PyTorch `nn.Conv2d`: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
5. [官方文档] PyTorch `nn.MaxPool2d`: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
6. [GitHub] CS231n: Convolutional Neural Networks for Visual Recognition. https://cs231n.github.io/convolutional-networks/

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、知识连线分析、工程最佳实践、常见错误、面试考点等均为原创内容。
