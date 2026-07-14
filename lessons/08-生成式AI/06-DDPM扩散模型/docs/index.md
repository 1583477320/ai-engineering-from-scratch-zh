# DDPM 扩散模型

> 扩散模型将图像逐步加噪到纯高斯噪声，再学会逆向去噪。这个简单的想法在 2020 年后主导了图像、视频和 3D 生成。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 08 · 02（VAE）| **时间：** ~90 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 08 · 07（潜在扩散）— 理解扩散如何在更小的潜空间运行

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 DDPM 的前向扩散过程——逐步向图像添加高斯噪声
- [ ] 实现去噪网络 U-Net——预测每步的噪声残差
- [ ] 解释采样过程——如何从纯噪声逐步恢复出图像
- [ ] 使用 DDIM 将采样步数从 1000 压缩到 50
- [ ] 在 MNIST 上训练一个完整的 DDPM 模型

---

## 1. 问题

GAN 训练不稳定（模式坍塌、训练不收敛）。VAE 生成质量有限（模糊）。有没有一种方法既能像 GAN 一样生成锐利图像，又能像 VAE 一样稳定训练？

2020 年 Ho 等人的 DDPM 论文给出了答案：**不直接学习数据分布，而是学习去噪过程。** 训练一个网络从噪声中恢复原始图像——然后在推理时从随机噪声开始，反复去噪，直到生成新图像。

这个方法没有对抗训练的博弈。它只有一个简单的 MSE 损失函数。它在 ImageNet 上生成了当时最好的样本质量。更重要的是，它的数学非常清晰——每一步都有严格的概率推导。

---

## 2. 概念

### 2.1 直观理解

想象你在看一部电影，但每一帧都被不同程度地加上了雪花噪点。

```
第 0 帧：清晰画面（原始数据）
第 1 帧：轻微噪点
第 2 帧：较多噪点
...
第 1000 帧：纯雪花（高斯噪声）
```

扩散模型要做两件事：

1. **训练**：看每一帧"清晰画面"和对应的"噪点画面"，学习"如果我知道加了什么噪声，就能把它减掉"
2. **生成**：从纯雪花开始，一步步"减去预测的噪声"，直到画面清晰

关键洞察：**生成 = 逆向的去噪过程。** 你不需要理解"什么是猫"——你只需要学会一步步去掉噪声。

### 2.2 前向过程（加噪）

DDPM 的前向过程是固定的（不需要学习），通过 T 步逐步添加高斯噪声：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$，$\beta_s$ 是预设的噪声调度（noise schedule）。

**噪声调度的作用：** 决定每一步加多少噪声。常见的有线性调度（DDPM 默认）和余弦调度（更接近均匀加噪）。

### 2.3 反向过程（去噪）

训练一个网络 $\epsilon_\theta$ 预测每步添加的噪声。损失函数：

$$L = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2 \right]$$

即预测噪声与真实噪声的均方误差。注意这里用了**条件均方误差**——网络还需要看到时间步 $t$。

### 2.4 采样过程

从随机噪声 $x_T \sim \mathcal{N}(0, I)$ 开始，迭代去噪：

$$x_{t-1} = \frac{1}{\sqrt{1 - \beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t \cdot z$$

其中 $z \sim \mathcal{N}(0, I)$，$\sigma_t^2 = \frac{(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)} \beta_t$。

**T=1000 步的 DDPM 需要 1000 次 U-Net 前向传播——很慢。** 这就是为什么后续出现了 DDIM（确定性采样，可压缩到 50 步）、DPM-Solver（高阶数值求解器，10 步即可）等方法。

### 2.5 U-Net 架构

扩散模型的去噪网络通常基于 U-Net——一种编码器-解码器结构，带有跳跃连接：

```
输入 x_t (带噪图像)
    ↓
[时间嵌入] ──────────────┐
    ↓                     ↓
[Encoder]  →  [Bottleneck]  →  [Decoder]
    ↑                     ↑
[Skip Connections] ←─────┘

每层都注入时间步信息 t
```

跳跃连接让浅层的空间信息（边缘、纹理）能够直达深层，这对保留图像细节至关重要。

---

## 3. 从零实现

完整代码见 `code/ddpm.py`。这里逐步讲解核心逻辑。

### 第 1 步：噪声调度

```python
class NoiseScheduler:
    """DDPM 噪声调度器——管理 beta、alpha、alpha_bar 序列。"""

    def __init__(self, num_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_steps = num_steps
        # 线性调度：beta 从 beta_start 线性增长到 beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample(self, t, shape, device):
        """
        闭式解采样：直接跳到时间步 t，无需逐步加噪。
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        alpha_bar = self.get_alpha_bar(t).to(device).view(-1, 1, 1, 1)
        noise = torch.randn(shape, device=device)
        return torch.sqrt(alpha_bar) * noise, torch.sqrt(1 - alpha_bar), noise
```

**为什么用闭式解？** 如果逐步加噪，每一步都要做一次矩阵乘法，复杂度 O(T)。闭式解直接计算 $x_t$，复杂度 O(1)。训练时每次随机采样一个时间步 $t$，所以只需要一次计算。

### 第 2 步：U-Net 去噪网络

```python
class SimpleUNet(nn.Module):
    """简化版 U-Net——预测噪声残差。"""

    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=256):
        super().__init__()
        # 时间嵌入：将标量时间步 t 映射到高维向量
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        # 编码器 + 解码器（含跳跃连接）
        # ... 详见 code/ddpm.py 中的完整实现
```

时间嵌入使用正弦位置编码（与 Transformer 中的位置编码同源），因为时间步是连续值而非离散类别，正弦编码能更好地泛化到未见过的时间步。

### 第 3 步：训练循环

```python
def forward_diffuse(x_0, scheduler, device):
    """前向扩散：随机采样时间步 t，向 x_0 添加噪声。"""
    batch_size = x_0.size(0)
    t = torch.randint(0, scheduler.num_steps, (batch_size,), device=device)
    noise = torch.randn_like(x_0)
    scaled_noise, scale_noise, _ = scheduler.sample(t, x_0.shape, device)
    x_t = scaled_noise + x_0 * (1 - scale_noise)
    return x_t, t, noise

def train_step(model, x_0, scheduler, device):
    """一步训练：预测噪声。"""
    x_t, t, noise = forward_diffuse(x_0, scheduler, device)
    pred_noise = model(x_t, t)
    return F.mse_loss(pred_noise, noise)
```

### 第 4 步：DDIM 加速采样

```python
@torch.no_grad()
def sample_ddim(model, scheduler, num_samples=16, num_steps=50, device="cpu"):
    """
    DDIM 采样——确定性去噪，步数从 1000 压缩到 50。
    与 DDPM 的关键区别：不使用随机采样 z，而是确定性地计算 x_{t-1}。
    """
    model.eval()
    stride = scheduler.num_steps // num_steps
    timesteps = list(range(0, scheduler.num_steps, stride))

    x = torch.randn(num_samples, 1, 28, 28, device=device)

    for i, t in enumerate(reversed(timesteps)):
        t_batch = torch.full((num_samples,), t, device=device)
        predicted_noise = model(x, t_batch)

        alpha_bar = scheduler.get_alpha_bar(t).to(device).view(-1, 1, 1, 1)
        alpha_next = scheduler.get_alpha_bar(timesteps[min(i + 1, len(timesteps) - 1)]).to(device)
        alpha_next = alpha_next.view(-1, 1, 1, 1)

        # 确定性公式：x_{t-1} = sqrt(alpha_bar_next) * x_hat_0
        #            + sqrt(1 - alpha_bar_next) * epsilon_theta
        coefficient = torch.sqrt(alpha_bar)
        pred_original = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / coefficient
        x = torch.sqrt(alpha_next) * pred_original + torch.sqrt(1 - alpha_next) * predicted_noise

    return torch.clamp(x, -1.0, 1.0)
```

DDIM 的核心洞察：**前向加噪过程可以看作一个确定性映射**。如果固定 $x_0$，那么无论加噪过程是否有随机性，只要知道预测的噪声 $\epsilon_\theta$，就能确定性地计算出 $x_{t-1}$。这使得步数压缩成为可能。

---

## 4. 工业工具

### 4.1 Hugging Face Diffusers

工业界最常用的扩散模型库。封装了 DDPM、DDIM、Stable Diffusion 等几乎所有主流扩散算法。

```python
from diffusers import DDPMPipeline

# 一行代码加载预训练 DDPM 管道
pipeline = DDPMPipeline()

# 生成 16 张 32x32 CIFAR-10 图像
images = pipeline(num_samples=16).images
```

支持的采样器包括：`DDIMScheduler`、`PNDMScheduler`、`EulerDiscreteScheduler`、`DPMMultistepScheduler` 等。

### 4.2 PyTorch 内置实现

```python
import torch.nn as nn

# 使用 PyTorch 构建自定义 U-Net
class CustomUNet(nn.Module):
    def __init__(self, channel=3):
        super().__init__()
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(516, 512),
        )
        # 编码器
        self.enc1 = nn.Conv2d(channel, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        # 解码器
        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.out = nn.Conv2d(64, channel, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = F.silu(self.enc1(x))
        h = F.silu(self.enc2(h))
        h = F.silu(self.dec1(h))
        return self.out(h)
```

### 4.3 性能对比

| 实现方式 | 采样步数 | ImageNet 512x512 速度 | FID | 适用场景 |
|---------|---------|---------------------|-----|---------|
| DDPM (1000 步) | 1000 | ~30 秒/张 | 3.1 | 基准、论文复现 |
| DDIM (50 步) | 50 | ~2 秒/张 | 3.5 | 快速原型 |
| DPM-Solver++ (15 步) | 15 | ~0.5 秒/张 | 3.2 | 实时生成 |
| Consistency Model (4 步) | 4 | ~0.1 秒/张 | 4.0 | 极致速度 |

---

## 5. LLM 视角

扩散模型虽然主要用于图像生成，但其核心思想正在渗透到语言模型领域。

### 5.1 在主流大语言模型中的体现

扩散的思想已经在多个方向影响大语言模型：

- **Diffusion-LM**（Ho et al., 2022）：将语言建模也视为扩散过程——在词元嵌入空间逐步加噪，然后去噪生成文本。证明了扩散模型不仅适用于连续空间（图像），也适用于离散空间（文本）的近似处理。
- **流匹配（Flow Matching）**：扩散模型的后续发展——将扩散过程建模为从噪声到数据的连续流。2024-2025 年的视频生成模型（如 Sora 类架构）广泛采用这种思路。
- **去噪自编码器在预训练中的应用**：BERT 的掩码语言建模本质上是去噪自编码器的一种形式——输入被"污染"（部分词元被掩码），模型学习恢复原始输入。

### 5.2 大语言模型时代什么变了？

扩散模型从"学术好奇心"变成了"工业基础设施"。Stable Diffusion 2022 年开源后，整个 AI 图像生成生态围绕它建立：LoRA 微调、ControlNet 控制、DreamBooth 个性化——这些都是建立在 DDPM 数学基础之上的工程创新。

### 5.3 什么没变？

扩散模型的核心数学——前向加噪的马尔可夫链和反向去噪的条件分布——自 2020 年以来几乎没有变化。变化的是规模（从 32x32 到 4096x4096）、速度（从 1000 步到 4 步）、和应用领域（从图像到视频到 3D 到音频到文本）。但那个简单的 MSE 损失函数仍然在运行。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你让 ChatGPT 的 DALL-E 3 或 Claude 的图像生成功能生成图片时，背后很可能就是 Stable Diffusion 或其变体。你输入的提示词通过 CLIP 编码器转化为条件信号，注入到扩散模型的 U-Net 中。你感受到的"生成速度"取决于后端使用的采样器——DALL-E 3 使用的是经过大量优化的定制采样器，通常在 50-250 步之间，平衡了质量和速度。

---

## 6. 工程最佳实践

### 6.1 噪声调度选择

| 场景 | 推荐调度 | 原因 |
|------|---------|------|
| 图像生成 | 线性调度（DDPM 默认） | 简单稳定，论文基准 |
| 高质量生成 | 余弦调度 | 均匀分配信噪比，感知质量更好 |
| 快速采样 | 轨迹调度 | 与流匹配兼容 |

### 6.2 训练技巧

- **学习率预热**：前 5000 步使用线性预热，避免早期梯度爆炸
- **EMA（指数移动平均）**：维护模型权重的 EMA 副本，采样质量显著提升
- **混合精度训练**：使用 FP16 可将显存占用减半，但需注意梯度缩放因子
- **数据增强**：对图像做随机翻转和缩放，能提升生成多样性

### 6.3 中文场景特别建议

- 在中文文本到图像场景中，使用支持中文的文本编码器（如 mCLIP），否则提示词中的中文会被错误分词
- 国内部署建议使用魔搭社区（ModelScope）的预训练模型，下载速度远快于 Hugging Face

### 6.4 踩坑经验

- **损失不下降**：检查数据是否归一化到 [-1, 1] 范围——DDPM 假设输入在此范围内，不归一化会导致噪声水平估计错误
- **采样图像全黑/全白**：通常是 `alpha_bar` 计算错误，确认 `torch.cumprod` 的维度是否正确
- **显存溢出**：U-Net 的跳跃连接会累积中间特征图。使用梯度检查点（`torch.utils.checkpoint.checkpoint`）可将显存占用降低 40%

---

## 7. 常见错误

### 错误 1：在采样时忘记设 `eval()` 模式

**现象：** 生成的图像出现随机斑块或不连贯区域。

**原因：** 训练时 Dropout 和 Batch Normalization 的行为与推理时不同。如果在采样时模型处于训练模式，Dropout 会引入随机性，Batch Normalization 会使用批次统计量而非训练时的全局统计量。

**修复：**

```python
# ❌ 错误写法
pred_noise = model(x_t, t)

# ✓ 正确写法
model.eval()
with torch.no_grad():
    pred_noise = model(x_t, t)
```

### 错误 2：alpha_bar 计算方向搞反

**现象：** 加噪过程直接得到纯噪声，而不是逐步加噪的效果。

**原因：** `torch.cumprod` 计算的是累积乘积。如果 beta 的定义是"保留比例"而非"噪声比例"，公式需要相应调整。混淆 $\beta$（噪声添加比例）和 $\alpha = 1 - \beta$（保留比例）是最常见的错误。

**修复：**

```python
# ❌ 错误：混淆了 beta 和 alpha
alphas = betas  # 错！beta 是噪声比例，alpha 是保留比例

# ✓ 正确
betas = torch.linspace(beta_start, beta_end, num_steps)
alphas = 1.0 - betas  # alpha 是保留比例
alpha_bars = torch.cumprod(alphas, dim=0)  # 累积乘积
```

### 错误 3：采样时时间步顺序错误

**现象：** 生成的图像模糊，质量远低于训练时的预期。

**原因：** 采样需要从 $t=T$ 递减到 $t=0$。如果顺序搞反（从 0 到 T），实际上是在"加噪"而不是"去噪"。

**修复：**

```python
# ❌ 错误：从 0 到 T（这是在加噪！）
for t in range(scheduler.num_steps):

# ✓ 正确：从 T 到 0（这才是去噪）
for t in reversed(range(scheduler.num_steps)):
```

---

## 8. 面试考点

### Q1：为什么扩散模型不需要对抗训练？它的损失函数是什么？（难度：⭐⭐）

**参考答案：**
扩散模型将生成问题转化为去噪估计问题。训练目标是预测每一步添加的噪声：$L = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$。这是一个标准的 MSE 回归损失，没有对抗博弈。这与 GAN 形成鲜明对比——GAN 需要同时训练生成器和判别器，两者相互博弈导致训练不稳定。扩散模型的优势在于训练简单稳定，劣势在于采样慢（需要多次前向传播）。

### Q2：DDPM 和 DDIM 采样的核心区别是什么？（难度：⭐⭐⭐）

**参考答案：**
DDPM 的反向过程是一个马尔可夫链，每一步都引入随机噪声 $z \sim \mathcal{N}(0, I)$，因此从相同噪声出发多次采样会得到不同结果。DDIM 则发现前向扩散过程可以视为一个确定性映射，因此在反向时可以不引入随机噪声，直接由 $x_t$ 确定性地计算 $x_{t-1}$。这使得 DDIM 允许跳过中间时间步——可以从 1000 步压缩到 50 步甚至更少，而 DDPM 严格要求 1000 步（因为每一步的噪声调度都是精心设计的）。代价是 DDIM 的样本多样性略低。

### Q3：如何设计一个噪声调度，使得扩散过程在早期加噪慢、晚期加噪快？（难度：⭐⭐⭐）

**参考答案：**
DDPM 默认使用线性调度 $\beta_t = \beta_{\text{start}} + t \cdot (\beta_{\text{end}} - \beta_{\text{start}}) / T$。如果要早期加噪慢、晚期加噪快，可以使用二次调度：$\beta_t = \beta_{\text{start}} + (\frac{t}{T})^2 \cdot (\beta_{\text{end}} - \beta_{\text{start}})$。但实践中更常用的是余弦调度（Sohl-Dickstein et al., 2015 提出），它通过在积分空间中均匀分配信噪比（SNR）来实现更均匀的加噪分布。余弦调度的采样质量通常优于线性调度，尤其在高质量图像生成中。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 扩散模型 | "慢慢加噪声再减掉" | 通过马尔可夫链逐步加噪到高斯分布，再学习逆向去噪过程的生成模型 |
| 噪声调度 (Noise Schedule) | "每步加多少噪声" | 预定义的 $\beta_t$ 序列，决定每一步的噪声强度——线性、余弦或轨迹调度 |
| U-Net | "扩散模型的网络" | 编码器-解码器架构，带有跳跃连接和时间步嵌入，是扩散模型的核心去噪网络 |
| DDIM | "更快的采样方法" | Deterministic Implicit Likelihood Modeling——确定性采样，可将步数从 1000 压缩到 50 |
| 闭式解 (Closed-form) | "一步到位加噪" | 利用高斯分布的解析性质直接计算 $x_t$，无需逐步加噪——训练时的关键优化 |
| 采样步数 | "生成一张图要跑多少次" | 去噪迭代的次数——DDPM 默认 1000 步，DDIM 可压缩到 50，DPM-Solver 可压缩到 15 |

---

## 📚 小结

DDPM 将图像生成简化为"学习去噪"：训练 U-Net 预测每步添加的噪声，然后从随机噪声开始逐步去噪。稳定训练、高质量样本、清晰的数学基础。代价：1000 步去噪需要 1000 次 U-Net 前向——太慢。DDIM 通过确定性采样将步数压缩 20 倍。下一课我们将进入潜在扩散——在 VAE 的潜空间做扩散，速度再提升 10-50 倍。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释为什么扩散模型不需要对抗训练。对比 GAN 的博弈损失和扩散模型的 MSE 损失，说明各自的优劣势。写 200 字以内的说明。

2. **【实现】** 修改 `sample_ddpm` 函数，将采样步数从 1000 减少到 100（每隔 10 步采样一次），对比质量变化。记录不同步数下的 FID 近似值（可通过重建误差衡量）。

3. **【实验】** 尝试三种不同的噪声调度（线性、余弦、二次），在相同的训练条件下训练 10 个轮次，对比收敛速度和最终损失值。

4. **【思考】** DDIM 为什么能将步数压缩？它的"确定性"假设在什么情况下会失效？阅读 DDIM 原文（arXiv:2010.02502）的 Section 3，用你自己的话解释。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| DDPM 完整实现 | `code/ddpm.py` | 包含噪声调度、U-Net、训练、DDPM 采样、DDIM 加速采样 |
| 噪声调度分析器 | `outputs/prompt-ddpm-analyzer.md` | 分析任意噪声调度参数对扩散过程的影响 |

---

## 📖 参考资料

1. [论文] Ho et al. "Denoising Diffusion Probabilistic Models". NeurIPS, 2020. https://arxiv.org/abs/2006.11239
2. [论文] Song et al. "Denoising Diffusion Implicit Models". arXiv, 2020. https://arxiv.org/abs/2010.02502
3. [论文] Dhariwal and Nichol. "Diffusion Models Beat GANs on Image Synthesis". NeurIPS, 2021. https://arxiv.org/abs/2105.05233
4. [官方文档] Hugging Face Diffusers: https://huggingface.co/docs/diffusers
5. [GitHub] Hugging Face diffusers: https://github.com/huggingface/diffusers

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
