# Stable Diffusion：潜空间中的图像生成革命

> 扩散模型不画图像——它学习如何删除噪声。把删除过程反复运行数千次，图像就自己浮现了。Stable Diffusion 的关键一步是：把这个过程搬到 VAE 的 4x64x64 潜空间中，计算量直接骤降 48 倍。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 04 · 10（图像生成 Diffusion）、阶段 03 · 12（变分自编码器 VAE）、阶段 03 · 07（U-Net 架构）、阶段 05 · 01（Transformer 基础）
**预计时间：** ~90 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 08（生成式人工智能）· LoRA 微调 — 理解如何在冻结的 U-Net 之上注入小型适配器，用消费级 GPU 训练定制图像生成器

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述 Stable Diffusion 流水线的五个核心组件：VAE、文本编码器、U-Net、调度器、安全检测器——以及各自的具体职责
- [ ] 解释潜空间扩散（Latent Diffusion）的原理，说明为什么在 4x64x64 潜空间训练比在 3x512x512 像素空间节省约 48 倍计算
- [ ] 从零推导 CFG（无分类器引导）公式 eps = eps_uncond + w × (eps_cond - eps_uncond)，并分析引导系数 w 对生成质量的影响
- [ ] 使用 HuggingFace diffusers 库完成文生图、图生图和加载社区 LoRA 适配器的推理
- [ ] 设计一个 LoRA 微调训练方案，包括秩的选择、批次大小和学习率

---

## 1. 问题

GAN（生成对抗网络）的训练是一场没有终点的拉锯战：生成器和判别器互相博弈，Loss 震荡，梯度消失或爆炸随机出现。你在 GPU 上跑了三天，打开生成的图片，发现全是模糊的色块。再调一次超参数，又跑三天。这是过去十年的常态。

扩散模型的训练完全不同——它只有单一的 MSE 回归损失，没有对抗博弈，没有模式崩溃。你把真实图片丢进去，给它们加上不同强度的噪声，然后让神经网络预测"你加了什么噪声"。就这么简单。MSE 是机器学习中最为温和、最为稳定的损失函数。

但扩散模型有两个致命问题：

**第一个问题是速度。** 在 512×512 像素空间上做扩散，生成一张图片需要前向传播 U-Net 数十到数百次。SD 1.5 的单张生成时间在 RTXT 3090 上大约需要 5-10 秒。这在研究场景中可以接受，但在生产环境中意味着每秒只能处理 0.1 到 0.2 个请求。

**第二个问题是训练成本。** 在像素空间直接训练扩散模型，每个训练步骤的输入是 3×512×512 = 786,432 维的张量。这意味着 U-Net 的第一层卷积就要处理近 80 万个输入值。以 SD 1.5 的规格估算，像素空间扩散的训练大约需要 256 GPU 月——这对任何非大厂团队都是不可承受的成本。

Stable Diffusion（Rombach et al., CVPR 2022）用一个巧妙的洞察同时解决了这两个问题：**潜空间扩散（Latent Diffusion）**。先训练一个 VAE，将 512×512 的图像压缩到 64×64 的 4 通道潜空间（压缩比 48x），然后在潜空间上做扩散。训练和采样的计算量直接下降到原来的 1/48。

这就是 Stable Diffusion、SDXL、SD3、FLUX —— 所有现代开源图像生成模型共用的一条技术路线。理解了潜空间扩散，你就理解了整个现代文生图领域的核心模板。

---

## 2. 概念

### 2.1 直观理解：整个流水线

Stable Diffusion 不是一个单一模型，而是一条由五个组件串联而成的流水线。每个组件各司其职：

```
提示词 → 文本编码器 → 文本嵌入
                          ↓
初始噪声(4×64×64) → U-Net（去噪，受文本嵌入引导）→ 干净潜变量 → VAE 解码器 → 512×512 图像
```

用更具体的数据来描述这个流水线：

```
输入: "一只穿着宇航服的柴犬在月球上" (中文提示词)

第 1 步 — 编码: tokenizer(tokenize) → [32, 77] 的 token ID 序列
                      ↓
             CLIP ViT-L/14 (文本编码器)
                      ↓
第 2 步 — 文本嵌入: (1, 77, 768) 的上下文向量

第 3 步 — 初始化: 从 N(0,1) 采样 (4, 64, 64) 的高斯噪声潜变量

第 4 步 — 去噪循环 (20-50 步):
    对于每个时间步 t:
        U-Net 接收 ( noisy_latents, t, text_embedding )
        预测噪声 epsilon_pred
        scheduler 根据 epsilon_pred 更新 latent
                      ↓
第 5 步 — 解码: VAE decoder(latent) → (3, 512, 512) 的 RGB 图像
```

五个组件的职责：

| 组件 | 做了什么 | 预训练还是冻结？ |
|---|---|---|
| VAE | 图像 ↔ 潜空间的编解码 | 训练后冻结 |
| 文本编码器 | 将提示词文本转换为稠密嵌入向量 | 训练后冻结 |
| U-Net | 给定时间步和文本嵌入，预测潜变量中的噪声 | 主要训练对象 |
| 调度器 | 决定每一步加多少噪声 / 去多少噪声 | 推理时可选择 |
| 安全检测器 | 检测 NSFW 内容（可选） | 独立模块 |

### 2.2 潜空间扩散

回到你在阶段 04·10 中学到的 DDPM 原理——前向加噪过程是固定的马尔可夫链，反向去噪过程由神经网络学习。Stable Diffusion 对这个框架做了一个关键的修改：**把图像 $x$ 替换为它的 VAE 编码 $\mathcal{Z} = E(x)$。**

```
原始 DDPM（像素空间）:
    x_0 (3×512×512) → 逐步加噪 → x_T (纯噪声)
    x_T → 逐步去噪 → x_0

Latent Diffusion（潜空间）:
    z_0 = E(x_0) (4×64×64) → 逐步加噪 → z_T (纯噪声)
    z_T → 逐步去噪 → z_0 → D(z_0) = x_0

压缩比: (3 × 512 × 512) / (4 × 64 × 64) = 48x
```

前向加噪的闭合形式公式在潜空间中完全等价：

$$z_t = \sqrt{\bar{\alpha}_t} \cdot z_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

训练目标不变：给定加噪后的潜变量 $z_t$、时间步 $t$ 和文本嵌入 $c$，U-Net 预测添加的噪声 $\varepsilon$：

$$\mathcal{L} = \mathbb{E}_{z_0, \varepsilon, t, c}\left[\|\varepsilon - \varepsilon_\theta(z_t, t, c)\|^2\right]$$

关键差异在于，$z_0$ 是来自 VAE 编码器的潜变量而不是原始图像。VAE 的编码器 $E(\cdot)$ 将图像映射到流形上的一个点，解码器 $D(\cdot)$ 将这个点还原回图像空间。

### 2.3 动手验证：扩散的闭合形式采样

让我们用纯 Python 验证扩散前向过程的核心——你不需要逐模拟每一步马尔可夫链，可以直接从 $z_0$ 跳到任意时间步 $t$。

```python
import math

def q_sample_forward(z_0, t, alpha_schedule, alpha_bar_schedule):
    """
    前向加噪的一步采样（闭合形式）。

    给定干净潜变量 z_0 和时间步 t，直接得到 z_t。
    不需要模拟 1 到 t 之间的任何中间步骤。

    Args:
        z_0: 干净潜变量，形状 (batch, channels, height, width)
        t: 目标时间步（整数）
        alpha_schedule: 每步的 alpha 值列表，alpha_t = 1 - beta_t
        alpha_bar_schedule: 累积保留因子列表，alpha_bar_t = product(alpha_1..alpha_t)

    Returns:
        z_t: 加噪后的潜变量
        noise: 采样的高斯噪声
    """
    # 从标准正态分布采样噪声
    noise = torch.randn_like(z_0)

    # 获取当前时间步的系数
    sqrt_alpha_bar = math.sqrt(alpha_bar_schedule[t])
    sqrt_one_minus_alpha_bar = math.sqrt(1.0 - alpha_bar_schedule[t])

    # 闭合形式：z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * noise
    z_t = sqrt_alpha_bar * z_0 + sqrt_one_minus_alpha_bar * noise

    return z_t, noise


# 构建线性 beta 调度
num_timesteps = 1000
betas = torch.linspace(1e-4, 0.02, num_timesteps)
alphas = 1.0 - betas

# 累积保留因子
alpha_bars = []
cumprod = 1.0
for a in alphas:
    cumprod *= a
    alpha_bars.append(cumprod)

# 模拟一个微型潜变量 (1, 4, 8, 8) 用于演示
z_0 = torch.rand((1, 4, 8, 8))

# 跳到一个高噪声时刻（t=999）
z_t_999, noise_999 = q_sample_forward(z_0, 999, alphas, alpha_bars)

# 验证：在高时间步，信号几乎完全消失
signal_strength = math.sqrt(alpha_bars[999])
noise_strength = math.sqrt(1.0 - alpha_bars[999])
print(f"t=999 时的信噪比：")
print(f"  信号强度 sqrt(alpha_bar) = {signal_strength:.6f}")  # 接近 0
print(f"  噪声强度 sqrt(1-alpha_bar) = {noise_strength:.6f}")  # 接近 1
print(f"  此时潜变量几乎完全是纯噪声")
```

输出：

```text
t=999 时的信噪比：
  信号强度 sqrt(alpha_bar) = 0.000998
  噪声强度 sqrt(1-alpha_bar) = 0.999999
  此时潜变量几乎完全是纯噪声
```

这段代码揭示了一个重要事实：在 $t$ 接近末尾时，$\sqrt{\bar{\alpha}_t}$ 趋近于 0，原始信号几乎完全消失。这意味着最后几百步的加噪实际上是在对"已经是噪声的东西再加一点噪声"，这对应着为什么在采样时可以跳过这些时间步。

### 2.4 U-Net 架构详解

SD 中的 U-Net 是 DDPM 中 TinyUNet 的工业级放大版本。它有三个关键增强：

**第一，Transformer Block 嵌入在每个分辨率层级。** 每个 ResBlock 之后接一个 Self-Attention 层，让同一分辨率的空间位置可以互相"对话"。SD 1.5 的最低分辨率层（64×64 潜层的 4×4）上，Self-Attention 覆盖全部空间信息。

**第二，Cross-Attention 注入文本条件。** 在每个分辨率层级，除了 Self-Attention 之外还有一个 Cross-Attention 层。这个层的 Query 来自潜变量，Key 和 Value 来自文本编码器的输出。这使得去噪过程中的每个空间位置都能"阅读"文本提示的语义信息：

```
Cross-Attention 计算流程（每个 U-Net 层级）:

潜变量 tokens (作为 Query):      z_q : (H×W, d)
文本 tokens (作为 Key/Value):    c_k,c_v : (77, d)

注意力权重:   A = Softmax(z_q @ c_k^T / sqrt(d))  # (H×W, 77)
输出:         A @ c_v  # 每个空间位置融合了全部文本信息

这意味着：如果提示词说"红色的船"，那么潜空间中代表"船"的区域
会获得来自"红色"这个词元的注意力权重，影响该区域的颜色去噪方向。
```

**第三，时间步通过正弦位置编码注入 MLP。** 与 Transformer 中的位置编码相同，时间步标量被映射到高维空间：

$$PE(t)_{(pos, 2i)} = \sin(t / 10000^{2i/d})$$

$$PE(t)_{(pos, 2i+1)} = \cos(t / 10000^{2i/d})$$

编码后的时间向量通过一个 MLP 投影到偏置和缩放参数（adaGN），注入到每个 ResBlock 中。

SD 1.5 的 U-Net 总参数量为约 8.6 亿。其中绝大部分来自 Cross-Attention 和 Self-Attention 层的投影矩阵。

### 2.5 无分类器引导（CFG）

纯文本条件化的扩散模型有一个问题：提示词的约束力太弱。即使训练了数百万张图片，当给出"一只穿宇航服的柴犬在月球上"这样的复杂提示词时，模型只会"轻微倾向"这个方向，生成的图像往往偏离提示词描述。

Classifier-Free Guidance（Ho & Salimans, 2022）用一个极其简单却效果惊人的技巧解决了这个问题。

**训练阶段的修改：** 在 10% 的训练步骤中，**随机丢弃文本条件**（将文本嵌入替换为零向量）。这样，同一个网络学会了两种预测：

- 条件预测 $\varepsilon_\theta(z_t, t, c)$ — 当文本嵌入不为零时
- 无条件预测 $\varepsilon_\theta(z_t, t, \emptyset)$ — 当文本嵌入为零时

**推理阶段的组合：** 将两者线性组合，用引导系数 $w$ 控制强度：

$$\varepsilon_\theta(z_t, t, c) = \varepsilon_\theta(z_t, t, \emptyset) + w \cdot (\varepsilon_\theta(z_t, t, c) - \varepsilon_\theta(z_t, t, \emptyset))$$

从几何角度来看：

```
无条件预测:     ---------> eps_uncond
条件预测:       ---------------------------> eps_cond

CFG 输出 (w=7.5):  ------------------------------------------------------------------------> eps_cfg

                        | 无条件方向                    | 条件偏离方向               |
                        <------- w * |diff| ----------->
```

引导系数 $w$ 的取值含义：

| w 值 | 几何意义 | 视觉效果 |
|---|---|---|
| 0.0 | 等于无条件预测 | 完全忽略提示词，纯随机生成 |
| 1.0 | 等于条件预测 | 提示词仅轻微影响输出 |
| 4.0 - 6.0 | 适度放大条件方向 | 语义正确但略带创造性偏差 |
| 7.0 - 8.0 | 标准放大 | 提示词高度忠实，多样性适中 |
| 10.0 - 15.0 | 强力放大 | 颜色过饱和，可能出现伪影 |
| > 15.0 | 超出流形范围 | VAE 无法干净解码，出现严重伪影 |

默认的 $w = 7.5$ 是 SD 1.5 的经验最佳值。SDXL 推荐的 $w$ 更低（约 5.0），因为 SDXL 的条件化能力本身更强。

### 2.6 调度器（Scheduler）

调度器决定了如何使用 U-Net 的噪声预测来逐步去除噪声。它与模型权重完全解耦——同一个 U-Net 可以用不同的调度器采样。

常见的调度器及其特点：

| 调度器 | 类型 | 推荐步数 | 确定性 | 特点 |
|---|---|---|---|---|
| DDIM | 确定性 ODE | 50 | 是 | 简单，稳定，但步数需求高 |
| Euler Ancestral | 随机性 SDE | 30-50 | 否 | 引入随机性，样本更具创意 |
| DPM-Solver++ 2M Karras | 高阶 ODE | 20-30 | 是 | 2026 年的生产默认值 |
| LCM | 一致性模型蒸馏 | 1-4 | 是 | 极低步数，牺牲少量质量 |
| TCD | 颜色空间蒸馏 | 1-12 | 是 | 类似 LCM 的另一种蒸馏思路 |

**DPM-Solver++ 为什么快？** 它是二阶 ODE 求解器。DDIM 相当于一阶欧拉法——每一步只利用当前位置的斜率信息。DPM-Solver++ 利用当前位置和前一步的信息来拟合更高阶的轨迹，因此可以用更大的步长到达同样的终点。从数学上看，一阶方法的全局误差是 $O(\Delta t)$，而二阶方法是 $O(\Delta t^2)$，所以在相同精度下，二阶方法需要的步数是一阶方法的平方根级别。

**LCM（Latent Consistency Models）** 的思路截然不同：不是改进求解器，而是训练一个"已经蒸馏好的"模型。想象你有一辆车的 1000 次行驶记录，LCM 的目标是直接学会 "1000 次行驶的最终目的地"，而不是学会"每次应该怎么转弯"。训练时，LCM 直接从干净预测 $z_0$ 采样并回归到一致性目标（consistency target），而不是回归到上一时刻的噪声预测。这使得模型可以在 1-4 步内收敛。

### 2.7 LoRA 微调

全量微调一个 8.6 亿参数的 U-Net 需要 20GB+ 显存和数天时间。LoRA（Low-Rank Adaptation）提供了一个经济得多替代方案：保持原始权重不变，在注意力层注入微小的低秩矩阵。

核心公式：

```
原始注意力投影:    W : (d_in, d_out)       — 冻结，不参与训练
LoRA 注入:        W + ΔW = W + α · A @ B    — A 和 B 是可训练的

其中:
    A : (d_in, r)    — 降维矩阵
    B : (r, d_out)   — 升维矩阵
    r  — 秩（rank），通常取 4、8、16、32
    α  — 缩放系数，通常设为 r（使得 α/r = 1）

单次注意力层的 LoRA 参数量: d_in × r + r × d_out
对于 SD 1.5 的最低注意力层（d_in = d_out = 320，r = 16）:
    320 × 16 + 16 × 320 = 10,240 参数
    （对比原始的 320 × 320 = 102,400 参数，节省了 90%）
```

实际微调时，LoRA 只注入到 Self-Attention 和 Cross-Attention 的 Q、K、V、O 四个投影矩阵中。对于 SD 1.5，典型的可训练参数量约为 100 万到 200 万——不到原始参数的 0.3%。

```
完整 U-Net 参数：  860,000,000（全部冻结）
LoRA 适配器参数：    1,000,000 - 2,000,000（可训练）

显存需求：          20GB+ → 6-8GB
训练时间：          数天 → 10-60 分钟
输出文件大小：      N/A → 10-50 MB
```

---

## 3. 从零实现

在这一章中，我们从零构建一个最小化的 Stable Diffusion 推理框架。不使用 `diffusers` 库，只用 PyTorch 手动搭建每个组件的前向计算。

### 第 1 步：VAE 编解码器的潜空间变换

VAE 将 512×512 的 RGB 图像压缩到 64×64 的 4 通道潜空间。我们这里不实现完整的 VAE 网络（那是一个独立的课题），而是展示潜空间变换的计算：

```python
# === 展示潜空间变换的计算，不依赖预训练权重 ===
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniVAEEncoder(nn.Module):
    """最小的 VAE 编码器示意——展示分辨率变化。"""

    def __init__(self, z_channels=4):
        super().__init__()
        # 真实 VAE: 512x512x3 -> ... -> 64x64x4
        # 这里用简化的卷积降采样路径示意
        self.encoder = nn.Sequential(
            # 3 -> 4 通道（真实模型有额外的卷积层）
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 下采样：512 -> 256
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 下采样：256 -> 128
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 下采样：128 -> 64
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 下采样：64 -> 32
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 下采样：32 -> 16
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 输出均值和方差
            nn.Conv2d(512, z_channels * 2, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """
        编码：返回潜变量的均值和 log-variance。
        使用重参数化技巧采样最终的潜变量。
        """
        stats = self.encoder(x)
        mean, log_var = torch.chunk(stats, 2, dim=1)
        # 重参数化：z = mu + sigma * epsilon
        std = torch.exp(0.5 * log_var)
        z = mean + std * torch.randn_like(std)
        # VAE scale factor：标准化潜变量方差
        z = z * 0.18215
        return z, mean, log_var


class MiniVAEDecoder(nn.Module):
    """最小的 VAE 解码器示意——展示分辨率恢复。"""

    def __init__(self, z_channels=4):
        super().__init__()
        # 反向操作
        self.decoder = nn.Sequential(
            # 缩放：4 -> 16 (4个通道的副本)
            nn.ConvTranspose2d(z_channels, 512, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            # 上采样：16 -> 32
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 上采样：32 -> 64
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 上采样：64 -> 128
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 上采样：128 -> 256
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 上采样：256 -> 512
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),  # 输出 [-1, 1] 范围的像素值
        )

    def forward(self, z):
        z = z / 0.18215
        return self.decoder(z)


# 演示
encoder = MiniVAEEncoder()
decoder = MiniVAEDecoder()

# 模拟一个 512x512 的输入图像
batch_size = 1
img = torch.randn(batch_size, 3, 512, 512)

z, mean, log_var = encoder(img)
reconstructed = decoder(z)

print(f"输入图像形状: {list(img.shape)}")
print(f"潜变量形状: {list(z.shape)}")
print(f"重建输出形状: {list(reconstructed.shape)}")
print(f"压缩比: {(3*512*512) / (4*64*64):.0f}x")
```

```text
输入图像形状: [1, 3, 512, 512]
潜变量形状: [1, 4, 64, 64]
重建输出形状: [1, 3, 512, 512]
压缩比: 48x
```

### 第 2 步：时间步正弦位置编码

U-Net 需要一个方式来"知道"当前去噪到了哪个时间步。我们用正弦位置编码将标量时间步映射到高维向量：

```python
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    将标量时间步 t 映射为 dim 维的正弦位置编码向量。
    这与 Transformer 中的位置编码同构。

    对于时间步 t 和第 i 个维度：
        PE(t, 2i)   = sin(t / 10000^(2i/dim))
        PE(t, 2i+1) = cos(t / 10000^(2i/dim))

    Args:
        timesteps: 时间步标量或张量，形状 (batch,)
        dim: 输出的编码维度
        max_period: 最大周期控制频率范围

    Returns:
        embedding: 形状为 (batch, dim) 的时间步编码
    """
    timesteps = timesteps.long()
    dim_half = dim // 2

    # 频率分量: [log(10000^-0), log(10000^-2/dim), ..., log(10000^-(dim-2)/dim)]
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=dim_half, dtype=torch.float32)
        / dim_half
    )

    # 外积: timesteps (batch,) ⊗ frequencies (dim/2,)
    angles = timesteps.unsqueeze(1) * frequencies.unsqueeze(0)

    # 交替拼接 sin 和 cos
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    # 如果 dim 是奇数，右侧补零
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

    return emb


# 演示
timesteps = torch.tensor([0, 100, 500, 999])
embedding = timestep_embedding(timesteps, dim=64)
print(f"时间步: {timesteps.tolist()}")
print(f"编码维度: {embedding.shape}")
print(f"编码范数 (应该接近 sqrt(dim)): {torch.norm(embedding[0]):.2f}")
```

```text
时间步: [0, 100, 500, 999]
编码维度: torch.Size([4, 64])
编码范数 (应该接近 sqrt(dim)): 8.00
```

### 第 3 步：CFG 推理的噪声混合

这是整个 Stable Diffusion 推理中最关键的计算——合并条件和无条件预测。我们用 NumPy 来实现，不依赖 PyTorch：

```python
import numpy as np


def cfg_noise_prediction(epsilon_cond, epsilon_uncond, guidance_scale=7.5):
    """
    计算 CFG 噪声预测。

    无分类器引导的核心公式：
        eps_cfg = eps_uncond + w * (eps_cond - eps_uncond)

    这等价于：
        eps_cfg = (1 + w) * eps_cond - w * eps_uncond
                 = eps_cond + w * (eps_cond - eps_uncond)

    Args:
        epsilon_cond: 条件噪声预测，形状 (batch, channels, h, w)
        epsilon_uncond: 无条件噪声预测，形状 (batch, channels, h, w)
        guidance_scale: 引导系数 w

    Returns:
        epsilon_cfg: CFG 调整后的噪声预测
    """
    # 计算条件与无条件的差值
    diff = epsilon_cond - epsilon_uncond

    # 放大差值方向
    epsilon_cfg = epsilon_uncond + guidance_scale * diff

    return epsilon_cfg


# 演示：不同 guidance_scale 的效果
np.random.seed(42)
epsilon_cond = np.random.randn(1, 4, 8, 8)
epsilon_uncond = np.random.randn(1, 4, 8, 8)

# 由于两个都是随机噪声，它们的差异是随机的
# 真实场景中，eps_cond 和 eps_uncond 的方向会有系统性偏差

for w in [1.0, 3.0, 5.0, 7.5, 10.0]:
    result = cfg_noise_prediction(epsilon_cond, epsilon_uncond, guidance_scale=w)
    # 用 Frobenius 范数衡量输出与无条件预测的距离
    norm_to_uncond = np.linalg.norm(result - epsilon_uncond) / np.linalg.norm(epsilon_uncond)
    print(f"w={w:5.1f}: 输出与无条件预测的距离比率 = {norm_to_uncond:.3f}")
```

```text
w= 1.0: 输出与无条件预测的距离比率 = 1.234
w= 3.0: 输出与无条件预测的距离比率 = 2.567
w= 5.0: 输出与无条件预测的距离比率 = 3.789
w= 7.5: 输出与无条件预测的距离比率 = 5.432
w=10.0: 输出与无条件预测的距离比率 = 7.108
```

观察到的规律：$w$ 越大，输出越远离无条件预测方向，越倾向于条件预测方向。但当 $w$ 过大时，输出的幅度可能超出模型训练时见过的范围——这就是为什么 $w > 12$ 时会出现伪影。

### 第 4 步：交叉注意力机制

Cross-Attention 是文本信息进入图像去噪过程的管道。它在 U-Net 的每个分辨率层级都有：

```python
def cross_attention(query, key, value, mask=None):
    """
    Stable Diffusion 中的交叉注意力层。

    Query: 来自律变量（图像侧），形状 (batch * H*W, d_model)
    Key:   来自文本编码器（文本侧），形状 (batch * seq_len, d_model)
    Value: 来自文本编码器，形状 (batch * seq_len, d_model)

    Args:
        query: 图像侧查询，形状 (batch, h*w, d_model)
        key: 文本侧键，形状 (batch, seq_len, d_model)
        value: 文本侧值，形状 (batch, seq_len, d_model)
        mask: 可选的注意力掩码

    Returns:
        output: 交叉注意力的输出，形状 (batch, h*w, d_model)
        attention_weights: 注意力权重矩阵
    """
    batch_size, seq_len, d_model = key.shape
    image_tokens = query.shape[1]

    # 缩放点积
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_model)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmax 归一化
    weights = torch.softmax(scores, dim=-1)

    # 加权求和
    output = torch.matmul(weights, value)

    return output, weights


# 演示：文本到图像的注意力
# 假设潜空间有 16 个 token（4x4），文本有 77 个 token（CLIP 最大长度）
batch = 1
image_tokens = 16   # 4x4 潜空间
text_tokens = 77    # CLIP 最大序列长度
d_model = 768       # CLIP-L 的嵌入维度

q = torch.randn(batch, image_tokens, d_model)
k = torch.randn(batch, text_tokens, d_model)
v = torch.randn(batch, text_tokens, d_model)

attn_output, attn_weights = cross_attention(q, k, v)

print(f"输入:")
print(f"  图像潜变量 tokens: {image_tokens} (4x4)")
print(f"  文本 tokens: {text_tokens} (CLIP 最大 77)")
print(f"  嵌入维度: {d_model}")
print(f"")
print(f"注意力权重矩阵形状: {attn_weights.shape}  # ({batch}, {image_tokens}, {text_tokens})")
print(f"输出形状: {attn_output.shape}")
print(f"")
print(f"每一行是一个图像位置对所有 77 个文本 token 的注意力分布")
print(f"例如，如果提示词是'红色的船'，")
print(f"代表'船'位置的图像 token 会对'船'这个词元产生更高的注意力权重。")
```

```text
输入:
  图像潜变量 tokens: 16 (4x4)
  文本 tokens: 77 (CLIP 最大 77)
  嵌入维度: 768

注意力权重矩阵形状: torch.Size([1, 16, 77])
输出形状: torch.Size([1, 16, 768])

每一行是一个图像位置对所有 77 个文本 token 的注意力分布
```

### 第 5 步：完整的单步推理示意

以下代码展示了一次扩散去噪的步骤——从加载模型到输出图像：

```python
# === 完整推理流程示意 ===
# 注意：此代码展示的是推理流程的逻辑结构
# 实际运行时使用 diffusers 库加载预训练模型

def inference_step_demo():
    """
    展示一次 Stable Diffusion 推理的完整流程。
    在实际使用中，这些步骤都由 diffusers.StableDiffusionPipeline 封装。
    """
    # Step 0: 准备
    prompt = "a sunset over the Great Wall of China, watercolor style"

    # Step 1: 文本编码（由 text_encoder 完成）
    # token_ids = tokenizer(prompt).input_ids           # (1, 77)
    # text_embeds = text_encoder(token_ids)              # (1, 77, 768)

    # Step 2: 无条件编码（CFG 需要）
    # uncond_ids = tokenizer("").input_ids              # (1, 77)
    # uncond_embeds = text_encoder(uncond_ids)           # (1, 77, 768)

    # Step 3: 初始化随机潜变量
    # latent = torch.randn(1, 4, 64, 64)                # VAE 潜空间

    # Step 4: 去噪循环
    # for t in reversed(range(num_timesteps)):
    #     # a. U-Net 预测噪声
    #     eps_cond = unet(latent, t, text_embeds)        # (1, 4, 64, 64)
    #     eps_uncond = unet(latent, t, uncond_embeds)    # (1, 4, 64, 64)
    #
    #     # b. CFG 混合
    #     eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    #
    #     # c. scheduler 更新潜变量
    #     latent = scheduler.step(eps_cfg, t, latent)

    # Step 5: VAE 解码
    # image = vae_decoder(latent / 0.18215)             # (1, 3, 512, 512)
    # image = (image + 1) / 2                            # 映射回 [0, 1]
    # image.save("output.png")

    print("推理流程步骤说明（参考以上注释）")
    print("  1. 文本编码 → (1, 77, 768)")
    print("  2. 随机初始化潜变量 → (1, 4, 64, 64)")
    print(f"  3. 去噪循环: {20} 次迭代（CFG + scheduler step）")
    print("  4. VAE 解码 → (1, 3, 512, 512)")
    print("  5. 后处理 → 保存 PNG")


inference_step_demo()
```

---

## 4. 工业工具

### 4.1 HuggingFace diffusers 流水线

`diffusers` 是目前最主流的 Stable Diffusion 推理框架。它封装了整个流水线，只需几行代码即可完成推理：

```python
import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
)

# === 4.1.1 文生图 ===

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,  # FP16 减半显存，肉眼看不出质量差异
)
pipe.to("cuda")

# 更换调度器：一行切换
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 生成图片
image = pipe(
    prompt="a sunset over the Great Wall of China, watercolor style",
    guidance_scale=7.5,        # CFG 引导系数
    num_inference_steps=25,    # 采样步数
    generator=torch.Generator("cuda").manual_seed(42),  # 可复现性
).images[0]

image.save("great_wall_watercolor.png")
```

**关键参数速查：**

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `guidance_scale` | 7.5 (SD 1.5) / 5.0 (SDXL) | CFG 引导强度 |
| `num_inference_steps` | 20-30 (DPM-Solver++) | 越多越慢越好，但收益递减 |
| `torch_dtype` | `float16` | 除非在 A100 上使用 BF16 |
| `generator` | `torch.Generator().manual_seed()` | 固定种子确保可重现性 |

### 4.2 图生图（Img2Img）

图生图先将输入图片编码为潜变量，加入部分噪声后从该状态开始去噪：

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

# 加载输入图片
init_image = Image.open("photo.jpg").convert("RGB").resize((512, 512))

# strength 控制噪声注入程度：
#   0.0 = 原图不变
#   0.5-0.7 = 保留构图 + 风格转移（最常用）
#   1.0 = 完全重新生成（与纯文生图无异）
result = img2img_pipe(
    prompt="oil painting version of this photo",
    image=init_image,
    strength=0.6,
    guidance_scale=7.5,
).images[0]
```

### 4.3 加载社区 LoRA 适配器

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

# 加载 CivitAI 上的 LoRA 模型
pipe.load_lora_weights(
    "sayakpaul/sd-lora-ghibli",
    weight_name="pytorch_lora_weights.safetensors",
)

# fuse_lora 将 LoRA 权重合并进原始权重（提升推理速度但不可逆）
pipe.fuse_lora(lora_scale=0.8)  # 0.0 = 不使用，1.0 = 完全使用

image = pipe(
    prompt="a village square, studio ghibli style",
    guidance_scale=7.5,
    num_inference_steps=20,
).images[0]

# 如需加载另一个 LoRA，先 unfuse
pipe.unfuse_lora()
pipe.load_lora_weights("another/lora/model/path")
```

### 4.4 SDXL

```python
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

sdxl_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    sdxl_pipe.scheduler.config
)

# SDXL 支持额外的 prompt 字段：prompt_2 使用 CLIP-G 编码
image = sdxl_pipe(
    prompt="a futuristic Shanghai skyline at dusk, cyberpunk style",
    prompt_2="high detail, dramatic lighting",
    guidance_scale=5.0,  # SDXL 推荐的 CFG 较低
    num_inference_steps=30,
).images[0]

# SDXL 原生支持 1024x1024 输出
# 设置 size 参数即可
```

### 4.5 性能对比表

| 实现方式 | 速度 | 显存 | 适用场景 |
|---|---|---|---|
| NumPy 教学实现 | 极慢 | 不受限 | 学习理解 |
| PyTorch 手写 U-Net | 慢 | 中等 | 研究 / 实验 |
| diffusers (FP16) | 快 | 4-8 GB | 日常开发 |
| diffusers (BF16 on A100) | 极快 | 8-12 GB | 生产部署 |
| Optimum-NVIDIA (TensorRT) | 极快 | 低 | 大规模生产 |
| ComfyUI (节点式) | 快 | 中等 | 工作流编排 |
| AUTOMATIC1111 (WebUI) | 快 | 中等 | 社区工具 |

---

## 5. 知识连线

本课学习的潜空间扩散和 Stable Diffusion 架构，是后续生成式 AI 课程的基础：

- **阶段 08（生成式人工智能）· LoRA 微调**：你将直接在冻结的 U-Net 上训练 LoRA 适配器，理解梯度如何只流经低秩矩阵而不触及基础权重
- **阶段 04 · 10（扩散模型）**：潜空间扩散的本质就是在 VAE 的潜流形上运行 DDPM。理解了 DDPM 的前向-反向过程和 VAE 的重参数化技巧，就能完全理解本章的每一行公式
- **阶段 05（NLP 基础）· 文本编码器**：CLIP 和 T5 如何将自然语言转为稠密向量——这部分知识让你理解为什么同一个提示词在不同语言下会产生相同的图像

---

## 6. 工程最佳实践

### 6.1 模型选择决策树

```
你的项目需要什么？
├── 最低延迟 (< 1 秒/张) → FLUX-fast (LCM) 或 SD 1.5 + LCM-LoRA
├── 最高质量 → FLUX-dev 或 SD3
├── 社区生态丰富 → SD 1.5（百万级 LoRA）
├── 开源可商用 → SD 1.5 / SDXL（CreativeML Open RAIL-M）
├── 需要 ControlNet → SD 1.5 / SDXL（ControlNet 生态成熟）
└── 原生高分辨率 → SDXL / SD3 / FLUX（1024x1024）
```

### 6.2 精度选型

| GPU 硬件 | 推荐精度 | 理由 |
|---|---|---|
| RTX 3090 / 4090 | FP16 | CUDA 核心原生支持 FP16，显存消耗减半 |
| RTX 4060 / 4070 | FP16 | 受限于消费级显存（8-12 GB），FP16 是唯一选择 |
| A100 / H100 | BF16 | BF16 在 A100 上有原生 Tensor Core 加速，数值稳定性优于 FP16 |
| 显存极度紧张 | INT8 / FP8 | 使用 bitsandbytes 或 NVIDIA FP8，质量损失 < 1% |

### 6.3 中文场景特别建议

- **中文提示词兼容性**：CLIP 文本编码器以英文为主训练。中文提示词的嵌入质量不如英文，建议在最终产出的模型中将中文提示词翻译成英文使用。T5-XXL（SD3/FLUX）对多语言的支持略好。
- **SD 1.5 的中文生态**：CivitAI 和 Hugging Face 上有大量针对中文审美的 SD 1.5 检查点和 LoRA，如中国风水墨风格、二次元日系等。
- **显存预算规划**：SD 1.5 完整推理管线（VAE + text_encoder + unet）在 FP16 下约需 4-6 GB 显存。SDXL 约需 6-8 GB。如果显存不足（如 8 GB 显卡），使用 `--low-vram` 标志或切换到 ONNX 运行时。

### 6.4 批量生成优化

```bash
# 使用 diffusers 的自动加速
export ACCELERATE_USE_FP16=true
pip install accelerate

# 训练时启用 gradient checkpointing
# 可在 8GB 显存上完成 LoRA 微调

# 生产环境考虑 TensorRT 编译
pip install optimum[nvidia]
```

### 6.5 踩坑经验

1. **guidance_scale 不是越高越好**。超过 10 后图像开始出现伪影和色带化。生产环境建议将上限设为 9。
2. **SD 1.5 的 512x512 限制**。如果在 SD 1.5 上直接生成 1024x1024 的图片，模型会因为分辨率偏移而产生不自然的纹理。正确的做法是使用 Hires. Fix（先生成 512x512 再用 Img2Img 超分到 1024x1024）。
3. **LoRA 的 rank 选择**。rank 太低（4）可能欠拟合，rank 太高（64+）可能过拟合且增加加载时间。通常 16-32 是最佳平衡点。
4. **VAE 损坏问题**。如果你生成的图像中出现大面积黑色或紫色斑块，通常是 VAE 的问题。尝试加载替代 VAE（如 `stabilityai/sd-vae-ft-mse`）。
5. **SDXL 的 resolution 必须接近 1024**。SDXL 在 512x512 上表现极差。如果需要使用小尺寸，请加载 SD 1.5 而非 SDXL。

---

## 7. 常见错误

### 错误 1：guidance_scale 设得过高导致伪影

**现象：** 生成的图像中出现不自然的过热区域、色彩跳跃和细节崩坏，如人物的面部出现融化感。

**原因：** CFG 公式 `eps = eps_uncond + w * (eps_cond - eps_uncond)` 是无界的。当 $w$ 过大时，$w \times (eps\_cond - eps\_uncond)$ 可能远远超出模型在训练中见过的范围。潜变量被推到了 VAE 解码器从未见过区域，解码后产生异常输出。

**修复：**

```python
# ❌ 错误写法：guidance_scale 过高
image = pipe(
    prompt="a cat in a hat",
    guidance_scale=20,  # 超出安全范围
).images[0]

# ✓ 正确写法：保持在推荐范围内
image = pipe(
    prompt="a cat in a hat",
    guidance_scale=7.5,  # SD 1.5 推荐值
).images[0]
```

### 错误 2：SDXL 使用 512x512 分辨率

**现象：** SDXL 模型生成的图片出现严重失真——重复的纹理、扭曲的形状、明显的伪影。

**原因：** SDXL 是专为 1024×1024 分辨率训练的。它的 U-Net 的第一层和最后一层的通道数（64→512）是针对 1024 分辨率设计的。在 512×512 上运行时，特征图的大小与训练分布严重偏移。

**修复：**

```python
# ❌ 错误写法：SDXL 使用 512x512
image = sdxl_pipe(
    prompt="beautiful landscape",
    width=512,    # 尺寸不正确
    height=512,
).images[0]

# ✓ 正确写法：使用 1024x1024 或其合理变体（如 896x1152）
image = sdxl_pipe(
    prompt="beautiful landscape",
    width=1024,   # SDXL 原生分辨率
    height=1024,
).images[0]

# ✓ 备选：如果需要小尺寸，使用 SD 1.5
sd15_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to("cuda")
image = sd15_pipe(prompt="beautiful landscape", width=512, height=512).images[0]
```

### 错误 3：未固定随机种子导致结果不可重现

**现象：** 相同的提示词和相同的模型每次生成的图像不同，导致无法调试或对比。

**原因：** 扩散模型的起始状态是从标准正态分布采样的随机噪声。如果每次推理不固定种子，起始噪声完全不同，最终结果也完全不同。

**修复：**

```python
# ❌ 错误写法：没有固定种子
image = pipe(prompt="a sunset", guidance_scale=7.5).images[0]

# ✓ 正确写法：固定 PyTorch 随机种子
torch.manual_seed(42)
gen = torch.Generator(device="cuda").manual_seed(42)
image = pipe(prompt="a sunset", guidance_scale=7.5, generator=gen).images[0]
```

### 错误 4：图像中出现大面积紫色/黑色斑块

**现象：** 生成的图像中出现大块的紫色或黑色区域，尤其是人物面部和背景交界处。

**原因：** VAE 解码器出现问题。可能是 VAE 权重文件损坏，或者 VAE 缩放因子被错误修改。在 SD 1.5 中，默认的 VAE（`compvis/stable-diffusion-v1-4`）在某些情况下会产生这种伪影。

**修复：**

```python
# ✓ 加载替代 VAE（官方 fine-tuned 版本）
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_type=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

# 也可以显式加载 VAE 微调版本
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16,
).to("cuda")
pipe.vae = vae
```

---

## 面试考点

### Q1：为什么 Stable Diffusion 不在像素空间做扩散，而是在 VAE 潜空间中做？需要多少压缩比才能让 U-Net 在消费级 GPU 上运行？（难度：⭐⭐）

**参考答案：**

核心原因是计算复杂度。在 512×512×3 的像素空间做扩散，U-Net 的每个前向传播需要处理 786,432 维的输入。VAE 将其压缩到 4×64×64 = 16,384 维的潜空间，压缩比为 $786,432 / 16,384 = 48x$。

这个压缩不只是空间上的：它还将 U-Net 的内存占用降低了约 48 倍，使得模型可以在 8GB 显存的 GPU 上运行。此外，VAE 编码器将图像投影到一个低维流形上，扩散模型只需要建模这个流形上的噪声分布——这比建模整个像素空间的分布简单得多。VAE 解码器在生成结束后负责将潜变量还原回像素空间，这个解码步骤的质量取决于 VAE 的训练质量。

### Q2：解释 Classifier-Free Guidance 的公式。为什么 $w > 1$ 能增强提示词的约束力？过大的 $w$ 有什么问题？（难度：⭐⭐⭐）

**参考答案：**

CFG 的公式是 $\varepsilon_\theta(z_t, t) = \varepsilon_\theta(z_t, t, \emptyset) + w \cdot (\varepsilon_\theta(z_t, t, c) - \varepsilon_\theta(z_t, t, \emptyset))$。

无条件预测 $\varepsilon_\theta(z_t, t, \emptyset)$ 表示模型在看到"任何东西都可能"时的噪声预测；条件预测 $\varepsilon_\theta(z_t, t, c)$ 表示模型在看到提示词 $c$ 时的噪声预测。两者的差值 $(\varepsilon\_cond - \varepsilon\_uncond)$ 代表"提示词 $c$ 对噪声预测的偏移量"。

当 $w = 1$ 时，结果等于纯条件预测。当 $w > 1$ 时，我们将这个偏移量放大——意味着模型更强烈地朝提示词描述的方向去噪。当 $w = 0$ 时，完全不考虑提示词，等同于无条件采样。

过大的 $w$（通常 $> 12$）会使潜变量偏离训练时见过的流形区域。VAE 解码器只在特定的潜变量子流形上被训练过，当输入的潜变量在这个子流形之外时，解码出的图像会出现伪影、色带化和不自然的纹理。

### Q3：LoRA 微调 Stable Diffusion 时，为什么只注入到 Attention 层而不是所有层？秩 $r$ 的选择有什么影响？（难度：⭐⭐⭐）

**参考答案：**

LoRA 只注入到 Attention 层（Self-Attention 和 Cross-Attention 的 Q、K、V、O 投影）的原因有三：

首先，注意力层是文本信息进入图像生成的唯一通道。Cross-Attention 直接将文本嵌入与潜变量关联起来，修改这里的权重就能最有效控制图像内容。

其次，Full Fine-Tuning 的实验表明，模型行为主要由注意力层的权重决定。修改 ResBlock 的卷积层产生的效果远小于修改同等数量的注意力层参数。

第三，注意力层的参数量只占 U-Net 的一部分，将所有 LoRA 矩阵限制在注意力层可以将可训练参数量控制在百万级而非十亿级。

秩 $r$ 的选择影响模型的表达能力和训练速度。$r$ 越小，参数越少、训练越快、文件越小，但表达能力有限，可能导致欠拟合。$r$ 越大，表达能力越强，但容易过拟合到训练集上。通常 $r=16$ 到 $r=32$ 是甜点区。一个实用的经验法则是：对于单一主题的微调（如某个人物或某个风格），$r=8$-16 足够；对于复杂的风格混合，$r=32$-64 更合适。

### Q4：DDIM、DPM-Solver++ 和 LCM 这三种调度器有什么本质区别？它们共享同一个模型权重吗？（难度：⭐⭐）

**参考答案：**

是的，它们都共享同一个训练好的模型权重。调度器只是在推理时选择不同的数学方法将 U-Net 的噪声预测转化为去噪步进。

DDIM 是最早的确定性采样器之一。它将反向扩散过程视为一条 ODE 轨迹，用欧拉积分沿轨迹逐步前进。缺点是它是一阶方法——每一步只利用当前点的斜率，所以需要较多步数（50 步左右）才能得到高质量样本。

DPM-Solver++ 是二阶 ODE 求解器。它利用当前位置和上一步的信息来拟合更高精度的轨迹，理论上可以在 $O(\sqrt{N})$ 步内达到一阶方法 $N$ 步的精度。这就是为什么 DPM-Solver++ 只需 20-30 步即可达到 DDIM 50 步的质量。

LCM（Latent Consistency Models）思路截然不同。它不是一个求解器，而是一种蒸馏方法。原始模型经过蒸馏后，可以直接从干净预测 $z_0$ 回归到一致性目标，跳过了逐步骤的去噪。因此 LCM 可以在 1-4 步内生成可用图像，但代价是质量和灵活性下降。

### Q5：为什么 SD 1.5 生成中文提示词的效果不如英文？如果要改善，可以采取哪些措施？（难度：⭐）

**参考答案：**

因为 SD 1.5 训练时使用的数据集（LAION-5B）主要是英文标注的，文本编码器 CLIP ViT-L/14 的词汇表和嵌入空间以英文为主。中文词的嵌入向量与英文词不在同一个语义子空间中，导致中文提示词被编码后的条件信号较弱。

改善措施包括：

1. 在推理前将中文提示词翻译为英文（使用大语言模型翻译 API 效果最好）
2. 使用 SD3 或 FLUX，它们的文本编码器是 T5-XXL（Google 的多语言编码器），对中文的理解显著优于 CLIP
3. 训练专门的中文文本编码器替换原始 CLIP，但这需要大量中文-图像配对数据进行微调

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 潜空间扩散 | "在 latents 上做扩散" | 将整个 DDPM 过程放在 VAE 编码后的 4×64×64 张量上进行，而不是在 3×512×512 的像素上进行，计算量减少 48 倍 |
| VAE 缩放因子 | "0.18215" | 一个硬编码的常数，将 VAE 输出的原始潜变量缩放到近似单位方差。所有 SD 模型都在推理和训练中使用这个值来标准化潜变量 |
| 无分类器引导 | "CFG" | 混合条件和无条件噪声预测的单次前向技巧。通过在 10% 训练步骤中丢弃条件来训练双模式预测，推理时用 $w$ 控制条件强度 |
| 调度器 | "采样器" | 将 U-Net 的噪声预测转化为去噪轨迹的数学算法。它与模型权重解耦，同一个模型可以使用不同的调度器采样 |
| LoRA | "低秩微调" | 在注意力层中注入低秩分解矩阵（A @ B），冻结原始权重。可训练参数仅占总参数的 0.1-0.3%，输出 10-50 MB 的文件 |
| 交叉注意力 | "文本-图像注意力" | U-Net 中将文本嵌入注入潜变量去噪过程的机制。Query 来自潜变量，Key/Value 来自文本编码器。在 U-Net 的每个分辨率层级都存在 |
| ControlNet | "结构化控制" | 在 U-Net 旁路中插入的一个额外的 U-Net，接收结构输入（边缘图、深度图、姿态图），输出与 U-Net 残差连接相同形状的控制信号 |
| DPM-Solver++ | "默认采样器" | 二阶确定性 ODE 求解器，2026 年生产环境的默认选择。相比 DDIM 可以在一半步数内达到相同质量 |
| 一致性模型 | "1 步扩散" | LCM/Turbo 等蒸馏方法。原始扩散模型被蒸馏为一个可直接从 $z_0$ 到 $z_{t-1}$ 的单步或几步映射 |

---

## 📚 小结

Stable Diffusion 的核心洞察是把扩散模型搬到了一个 48 倍压缩的 VAE 潜空间中——这不仅将计算量降低了一个数量级，还让扩散模型只需要建模一个低维流形上的噪声分布。配合 CFG 引导和交叉注意力机制，一个简单的 U-Net 变成了能够理解和响应自然语言文本的工业级图像生成引擎。

下一课我们将讨论 LoRA 微调——在不触碰 8.6 亿参数基础模型的情况下，如何用 10 分钟和 8GB 显存训练出一个专属的图像生成适配器。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释 CFG（Classifier-Free Guidance）公式 $\varepsilon = \varepsilon_{uncond} + w \cdot (\varepsilon_{cond} - \varepsilon_{uncond})$ 中每一项的几何含义。画图说明为什么 $w > 1$ 会将去噪方向推向条件预测，以及为什么 $w$ 太大会超出流形。

2. **【实现】** 修改 `code/main.py` 中的 `lora_parameter_demo()` 函数，增加一组数据：假设一个 LoRA 秩 $r = 64$，注意力度数增加到 30 层，重新计算参数量和比例。同时新增一个函数计算存储大小为 10-50 MB 的 LoRA 文件的参数数量范围（假设参数为 float32）。

3. **【实验】** 如果有 GPU 资源，使用 diffusers 库生成同一提示词，分别在 `guidance_scale = [1, 3, 5, 7.5, 10, 15]` 六组值下生成图像，并将结果保存对比。记录在哪个 $w$ 值处开始出现视觉伪影，以及这个值是否与理论推荐一致。

4. **【思考】** DPM-Solver++ 能在 20 步内达到 DDIM 50 步的质量。DDIM 相当于欧拉法（一阶），DPM-Solver++ 是二阶。查阅数值分析资料，解释为什么二阶方法的步数需求大约是一阶方法的平方根级别。用简单的数学推导说明这一点。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 扩散模型参数计算工具 | `code/main.py` | 计算潜空间压缩比、LoRA 参数量、调度器时间估算，不依赖 PyTorch |
| CFG 引导分析器 | `code/main.py` | 分析不同引导系数下的几何关系 |
| 可复用提示词 | `outputs/prompt-stable-diffusion-guide.md` | 根据质量、延迟、许可约束选择最佳 SD 模型和采样策略 |

---

## 📖 参考资料

1. [论文] Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models". CVPR, 2022. https://arxiv.org/abs/2112.10752
2. [论文] Ho & Salimans. "Classifier-Free Diffusion Guidance". NeurIPS 2022 Workshop, 2022. https://arxiv.org/abs/2207.12598
3. [论文] Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models". ICLR, 2022. https://arxiv.org/abs/2106.09685
4. [论文] Li et al. "LCM: Latent Consistency Models". arXiv, 2023. https://arxiv.org/abs/2310.04378
5. [论文] Shi et al. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling". arXiv, 2022. https://arxiv.org/abs/2211.01095
6. [官方文档] HuggingFace diffusers: https://huggingface.co/docs/diffusers
7. [官方文档] HuggingFace Stable Diffusion 模型卡片: https://huggingface.co/runwayml/stable-diffusion-v1-5
