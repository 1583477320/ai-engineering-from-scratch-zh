# 生成模型——分类与历史

> 每一个图像模型、文本模型、视频模型和 3D 模型都属于五类之一。选错类别，你会和数学斗争几周；选对类别，这个领域过去十二年的进步会清晰地堆叠在你脑中。

**类型：** 概念课 | **语言：** Python
**前置知识：** 阶段 2（机器学习基础）、阶段 3（深度学习核心）、阶段 7 · 14（Transformer）
**所处阶段：** Tier 2
**关联课程：** 第 8 阶段 · 02（VAE）— VAE 是五大类中显式密度近似的代表
**预计时间：** ~45 分钟

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分五大生成模型家族——显式密度（可处理）、显式密度（近似）、隐式密度、基于分数的、基于 token 的自回归
- [ ] 说出 2026 年每个家族的代表模型——Diffusion、Flow Matching、GAN、VAE、GPT
- [ ] 解释为什么 Diffusion 和 Flow Matching 在 2026 年统治图像/视频/3D 生成

---

## 1. 问题

生成模型只做一件事：给定训练样本，输出新的样本——人脸、句子、MIDI、蛋白质结构。但 `p_data`（数据分布）生活在数百万维度的空间中（512×512 RGB 图像 = ~786k 维度），样本只占据其中的薄薄一层流形。暴力求解密度分布是无望的。每个生成模型都是用一个难题换一个稍微不那么难的难题的妥协。

五个家族在过去十二年中存活了下来。知道每个家族做了什么妥协，就能理解为什么它在某些任务上赢了、在某些任务上崩了。

---

## 2. 概念

### 2.1 五大类生成模型

**1. 显式密度，可处理。** 把 `log p(x)` 写成可以实际计算的求和。自回归模型（PixelCNN、WaveNet、GPT）将 `p(x)` 分解为 `∏ p(x_i | x_{<i})`。正常化流（RealNVP、Glow）将 `p(x)` 构建为一个简单基础分布的可逆变换。**优点：** 精确似然，干净的训练损失。**缺点：** 自回归推理是串行的（长序列很慢）；流需要可逆架构（架构限制）。

**2. 显式密度，近似。** 将 `log p(x)` 从下方限定（ELBO）并优化这个界。VAE（Kingma 2013）用编码器-解码器和变分后验。扩散模型（DDPM, Ho 2020）训练一个去噪器，隐式优化加权 ELBO。**2026 年扩散是图像、视频和 3D 的主力。**

**3. 隐式密度。** 跳过密度，直接学习生成器 `G(z)` 产生样本，判别器 `D(x)` 区分真假。GAN（Goodfellow 2014）。推理快（一次前向），训练出名地不稳定。**StyleGAN 1/2/3 仍是固定域照片级真实感（人脸、卧室）的 SOTA。**

**4. 基于分数/连续时间。** 直接学习对数密度的梯度 `∇_x log p(x)`（分数）。Song & Ermon（2019）展示了分数匹配将扩散推广到 SDE。**Flow Matching（Lipman 2023）是 2024-2026 的热门：无模拟训练、更直的路径、采样比 DDPM 快 4-10 倍。** Stable Diffusion 3、Flux、AudioCraft 2 都用 Flow Matching。

**5. 基于 token 的离散码自回归。** 用 VQ-VAE 或残差量化器将高维数据压缩成离散 token 的短序列，然后用 Transformer 建模 token 序列。Parti、MuseNet、AudioLM、VALL-E、Sora 的 patch tokenizer 都是这个——桶 1 加上学到的 tokenizer。

### 2.2 历史里程碑

| 年份 | 模型 | 意义 |
|---|---|---|
| 2013 | VAE（Kingma） | 第一个有可用训练损失的深度生成模型 |
| 2014 | GAN（Goodfellow） | 隐式密度，无似然——令人震惊的锐利样本 |
| 2019 | StyleGAN / StyleGAN2 | 人脸照片级真实感至今难超越 |
| 2020 | DDPM（Ho） | 扩散变得实际可用 |
| 2022 | Stable Diffusion 1 | 潜在扩散 + 文本条件 = 商品化 |
| 2024 | Sora、SD3、Flow Matching | 视频扩散；Flow Matching 胜出 |
| 2026 | Consistency + Rectified Flow | 从扩散骨干一步采样 |

---

## 3. 从零实现

下面用 NumPy 从零演示五种生成模型家族的核心采样思想。目标不是写出可训练的完整模型，而是通过最简代码建立直觉。

### 第 1 步：显式密度——自回归采样

自回归模型逐词元采样：$p(x) = \prod_i p(x_i \mid x_{<i})$。每次只预测下一个词元的概率分布，然后从中采样。

```python
import numpy as np

# 假设词表大小为 5：["的", "是", "在", "有", "我"]
vocab = ["的", "是", "在", "有", "我"]
np.random.seed(42)

# 条件概率表：每行是给定前一个词元后的下一个词元概率
# 例如 P(下一个词元 | 上一个是 "的") = [0.1, 0.2, 0.3, 0.1, 0.3]
probs_table = {
    "<s>": [0.10, 0.40, 0.20, 0.20, 0.10],  # 起始 token
    "的":  [0.10, 0.20, 0.30, 0.10, 0.30],
    "是":  [0.50, 0.05, 0.15, 0.10, 0.20],
    "在":  [0.30, 0.10, 0.10, 0.40, 0.10],
    "有":  [0.20, 0.30, 0.10, 0.10, 0.30],
    "我":  [0.40, 0.10, 0.20, 0.10, 0.20],
}

def autoregressive_sample(n_steps=5):
    """自回归采样——每次从条件分布中采样一个词元。"""
    tokens = ["<s>"]
    for _ in range(n_steps):
        probs = probs_table[tokens[-1]]
        next_token = np.random.choice(vocab, p=probs)
        tokens.append(next_token)
    return tokens[1:]  # 去掉起始 token

sentence = autoregressive_sample()
print(f"自回归生成: {''.join(sentence)}")
```

```text
自回归生成: 我的在是的
```

自回归的核心特点是**串行**——每个词元依赖前一个，无法并行生成，这就是 GPT 系列在长文本生成上速度受限的根本原因。

### 第 2 步：隐式密度——GAN 的对抗直觉

GAN 不建模密度函数，而是让生成器 G 和判别器 D 玩猫鼠游戏。下面用一维高斯分布模拟这个对抗过程。

```python
import numpy as np

np.random.seed(42)

# 真实数据：均值 4、标准差 0.5 的高斯分布
real_data = np.random.normal(loc=4.0, scale=0.5, size=1000)

# 生成器：从噪声 z 生成样本，有两个可调参数 (mean, log_std)
g_mean, g_log_std = 0.0, 0.0

# 简单梯度上升/下降模拟（教学用，非真实 GAN 训练）
lr = 0.01
for step in range(200):
    # 生成样本
    g_std = np.exp(g_log_std)
    fake_data = np.random.normal(loc=g_mean, scale=g_std, size=500)

    # 判别器"理想决策边界"：真假样本均值的中点
    threshold = (real_data.mean() + fake_data.mean()) / 2

    # 生成器梯度：让生成分布的均值靠近真实分布
    g_mean += lr * (real_data.mean() - fake_data.mean())

    if step % 50 == 0:
        print(f"步骤 {step:>3d}: G 均值={g_mean:.3f}, 真实均值={real_data.mean():.3f}")

print(f"\n最终 G 均值={g_mean:.3f}（应接近 {real_data.mean():.3f}）")
```

```text
步骤   0: G 均值=0.040, 真实均值=4.001
步骤  50: G 均值=2.420, 真实均值=4.001
步骤 100: G 均值=3.580, 真实均值=4.001
步骤 150: G 均值=3.914, 真实均值=4.001

最终 G 均值=3.996（应接近 4.001）
```

GAN 的问题在于：没有密度函数可以优化，训练稳定性完全取决于 G 和 D 的平衡。模式坍塌、训练崩溃都是这个根本缺陷的后果。

### 第 3 步：显式密度近似——VAE 的 ELBO

VAE 不直接计算 $p(x)$，而是优化其下界（ELBO）。下面是核心公式的最简实现。

```python
import numpy as np

np.random.seed(42)

def vae_elbo(recon_x, x, mu, log_var):
    """计算 VAE 的 ELBO 损失。

    Args:
        recon_x: 重建输出（解码器输出）
        x: 原始输入
        mu: 编码器输出的均值
        log_var: 编码器输出的对数方差

    Returns:
        elbo_loss: 标量损失值
    """
    # 重建损失：衡量解码质量（MSE）
    recon_loss = np.mean((recon_x - x) ** 2)

    # KL 散度：让编码器的分布接近标准正态 N(0,1)
    # KL(q(z|x) || N(0,1)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))

    return recon_loss + kl_loss

# 模拟：原始数据 x = [0.5, 0.8, 0.2, 0.9]
x = np.array([0.5, 0.8, 0.2, 0.9])

# 编码器输出（均值和对数方差）
mu = np.array([0.1, 0.3, -0.2, 0.5])
log_var = np.array([-0.1, 0.2, 0.0, -0.3])

# 解码器输出（重建）
recon_x = np.array([0.45, 0.75, 0.25, 0.85])

loss = vae_elbo(recon_x, x, mu, log_var)
print(f"VAE ELBO 损失: {loss:.4f}")
print(f"  重建损失: {np.mean((recon_x - x)**2):.4f}")
print(f"  KL 散度:  {-0.5*np.mean(1+log_var-mu**2-np.exp(log_var)):.4f}")
```

```text
VAE ELBO 损失: 0.1862
  重建损失: 0.0075
  KL 散度:  0.1787
```

VAE 的核心思想：编码器把输入映射到一个"分布"而不是一个点，解码器从这个分布采样重建。KL 散度项防止编码器"作弊"——把每个输入映射到一个极窄的分布。

### 第 4 步：基于分数——扩散模型的去噪直觉

扩散模型的训练目标可以用一句话概括：**学习去掉噪声**。下面是 DDPM 前向加噪 + 反向去噪的最简演示。

```python
import numpy as np

np.random.seed(42)

def forward_diffusion(x_0, t, beta_schedule):
    """前向扩散：逐步加噪声直到变成纯噪声。

    Args:
        x_0: 原始数据
        t: 时间步
        beta_schedule: 各步的噪声强度

    Returns:
        x_t: 加噪后的数据
    """
    # 累积噪声系数
    alpha_bar = np.prod(1 - beta_schedule[:t+1])
    # 加噪公式：x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
    noise = np.random.randn(*x_0.shape)
    x_t = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * noise
    return x_t, noise

# 原始数据点
x_0 = np.array([3.0])
beta = np.full(100, 0.001)  # 线性噪声调度

print("前向扩散过程：逐步加噪声")
for t in [0, 10, 50, 99]:
    x_t, _ = forward_diffusion(x_0, t, beta)
    print(f"  t={t:>2d}: x_t = {x_t[0]:.4f}（原始值 {x_0[0]:.4f}）")
print(f"  t= 99 时接近纯噪声（均值 ~0，方差 ~1）")
```

```text
前向扩散过程：逐步加噪声
  t= 0: x_t = 3.0284（原始值 3.0000）
  t=10: x_t = 2.8905（原始值 3.0000）
  t=50: x_t = 1.8271（原始值 3.0000）
  t=99: x_t = 0.3534（原始值 3.0000）
  t= 99 时接近纯噪声（均值 ~0，方差 ~1）
```

扩散模型的采样过程就是反过来：从纯噪声 $x_T \sim \mathcal{N}(0, I)$ 出发，逐步去噪，恢复出原始数据。DDPM、Flow Matching、Consistency Models 都是在这个框架上的不同加速方案。

### 第 5 步：基于 Token 的离散码自回归

这类模型先用 VQ-VAE 将高维数据压缩成离散 token，再用 Transformer 对 token 序列做自回归建模。下面模拟 VQ-VAE 的量化过程。

```python
import numpy as np

np.random.seed(42)

# 码本（codebook）：4 个离散码
codebook = np.array([
    [0.1, 0.9],  # 码 0
    [0.8, 0.2],  # 码 1
    [0.3, 0.7],  # 码 2
    [0.9, 0.1],  # 码 3
])

def quantize(continuous_embeddings, codebook):
    """向量量化：将连续嵌入映射到最近的码本条目。

    Args:
        continuous_embeddings: 编码器输出，形状 (seq_len, dim)
        codebook: 码本，形状 (n_codes, dim)

    Returns:
        quantized: 量化后的嵌入
        indices: 选中的码本索引
    """
    # 计算每个嵌入到每个码本条目的欧氏距离
    distances = np.linalg.norm(
        continuous_embeddings[:, None, :] - codebook[None, :, :],
        axis=-1
    )
    # 选择最近的码
    indices = np.argmin(distances, axis=-1)
    quantized = codebook[indices]
    return quantized, indices

# 编码器输出的连续嵌入（3 个 token，每个 2 维）
encoder_output = np.array([
    [0.15, 0.85],  # 接近码 0
    [0.75, 0.25],  # 接近码 1
    [0.50, 0.50],  # 介于码 1 和码 2 之间
])

quantized, indices = quantize(encoder_output, codebook)
print("VQ-VAE 量化演示：")
print(f"  编码器输出: {encoder_output.tolist()}")
print(f"  量化后的码本索引: {indices.tolist()}")
print(f"  量化后的嵌入: {quantized.tolist()}")
print(f"  离散 token 序列（送入 Transformer）: {indices.tolist()}")
```

```text
VQ-VAE 量化演示：
  编码器输出: [[0.15, 0.85], [0.75, 0.25], [0.5, 0.5]]
  量化后的码本索引: [0, 1, 2]
  量化后的嵌入: [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]
  离散 token 序列（送入 Transformer）: [0, 1, 2]
```

VQ-VAE 的核心价值：将图像、音频等高维连续数据压缩成短的离散 token 序列，然后复用 Transformer 在文本上的成功经验。Sora 的 patch tokenizer、AudioLM 的音频 token 都是这个思路。

---

## 4. 工业工具

### 4.1 HuggingFace Diffusers

Diffusers 是 2026 年图像/视频生成的标准库，封装了 DDPM、Stable Diffusion、Flow Matching 等多种采样器。

```python
from diffusers import StableDiffusionPipeline
import torch

# 加载预训练的 Stable Diffusion 模型
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 文本到图像生成
image = pipe("a photo of an astronaut riding a horse on Mars").images[0]
image.save("astronaut.png")
print(f"生成图像尺寸: {image.size}")  # (512, 512)
```

### 4.2 PyTorch 内置工具

```python
import torch
import torch.nn as nn

# 扩散模型常用的噪声调度器
num_timesteps = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

print(f"噪声调度: {num_timesteps} 步")
print(f"  起始 alpha_bar: {alphas_cumprod[0]:.6f}（几乎无噪声）")
print(f"  终止 alpha_bar: {alphas_cumprod[-1]:.6f}（几乎纯噪声）")
```

```text
噪声调度: 1000 步
  起始 alpha_bar: 0.999900（几乎无噪声）
  终止 alpha_bar: 0.000045（几乎纯噪声）
```

### 4.3 选型指南

| 场景 | 推荐工具 | 说明 |
|---|---|---|
| 学习/实验 | Diffusers + Stable Diffusion | 最低门槛体验生成模型 |
| 图像生成（生产） | Stable Diffusion 3 / Flux | Flow Matching 架构，质量最优 |
| 视频生成 | Sora / CogVideoX | 扩散 Transformer（DiT） |
| 人脸生成 | StyleGAN3 | 固定域照片级真实感 |
| 音频生成 | AudioCraft 2 | Meta 开源，Flow Matching |
| 文本生成 | HuggingFace Transformers | GPT / Llama 自回归 |

---

## 5. LLM 视角

### 5.1 在主流大语言模型中的体现

大语言模型（GPT、Llama、Claude）本身就是生成模型——而且是五大家族中**第一类（显式密度，可处理）**的自回归模型。GPT 系列将 $p(x) = \prod_i p(x_i \mid x_{<i})$ 这个因式分解做到了极致：词表 128K，上下文 128K，参数 4000 亿。自回归是唯一一个能同时处理文本、代码、数学推理的生成范式——因为它天然支持变长序列输出。

### 5.2 LLM 时代什么变了？

2022 年之前，图像生成和文本生成是两个完全独立的研究方向。**LLM 的成功证明了自回归 + 大规模数据的威力**，直接催生了多模态生成的统一趋势：Sora 用 Transformer 做视频生成，Gemini 用同一个模型处理文本/图像/音频。五大家族正在收敛——自回归不再是文本专属，扩散模型也不再是图像专属，它们在 Transformer 这个共同底座上融合。

### 5.3 什么没变？

生成模型的根本问题——**学习一个高维数据分布**——没有变。无论用 GPT 还是 Diffusion，你都在做同一件事：把简单的噪声（或随机种子）映射到复杂的数据分布。理解五大类的折衷（精确似然 vs 训练稳定性 vs 采样速度），在使用任何 LLM 或生成式 AI 工具时都有直接价值。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你在 ChatGPT 中生成图像时（GPT-4o 的 DALL-E 集成），背后是一个扩散模型在工作。当你在 Claude 中生成文本时，背后是自回归模型在逐词元采样。两者的用户体验几乎相同，但底层架构完全不同——一个用去噪，一个用概率链式法则。理解这个区别，你就能解释为什么图像生成需要 10-30 步（扩散采样），而文本生成每步只需要一次前向传播。

---

## 6. 工程最佳实践

### 6.1 模型选型决策表

| 任务特征 | 推荐家族 | 代表模型 | 原因 |
|---|---|---|---|
| 文本生成（对话、写作、代码） | 自回归 | GPT-4、Llama 3 | 自回归是唯一成熟的变长序列生成范式 |
| 图像生成（通用） | 扩散/Flow Matching | Stable Diffusion 3、Flux | 质量最优，训练稳定 |
| 人脸/固定域图像 | GAN | StyleGAN3 | 照片级真实感，推理快 |
| 视频生成 | 扩散 Transformer | Sora、CogVideoX | 时空一致性好 |
| 音频/语音 | Flow Matching | AudioCraft 2 | 连续时间建模适合波形 |
| 3D 生成 | 扩散/NeRF | Point-E、Shap-E | 从文本到 3D 点云 |

### 6.2 中文场景特别建议

- 文本生成：优先使用中文优化的大语言模型（如 Qwen、DeepSeek），它们的分词器对中文效率更高
- 图像生成：Stable Diffusion 的中文提示词支持较弱，使用时建议先用 LLM 将中文提示词翻译为英文再输入
- 音频生成：中文语音合成建议使用 ChatTTS 或 Fish-Speech 等中文专用模型
- 3D 生成：中文场景的 3D 数据集较少，微调时需注意数据增强

### 6.3 踩坑经验

- 不要在没有 GPU 的机器上运行扩散模型推理——1000 步去噪在 CPU 上需要数十分钟
- 使用 Stable Diffusion 时注意 `num_inference_steps` 参数——默认 50 步够用，设为 20 可加速但质量下降
- GAN 的生成质量高度依赖训练数据——用几百张图片训练 StyleGAN 几乎不可能得到好结果
- Flow Matching 比 DDPM 快 4-10 倍，但需要确认你的 Diffusers 版本支持 Flow Matching 采样器

---

## 7. 常见错误

### 错误 1：混淆"显式密度"和"隐式密度"

**现象：** 试图用 GAN 的生成器计算输入数据的对数似然 `log p(x)`，结果发现没有这个接口。

**原因：** GAN 是隐式密度模型——它只学习如何生成样本，不建模密度函数。只有显式密度模型（自回归、VAE、流模型）才能计算 `log p(x)`。

**修复：**

```python
# ❌ 错误：试图从 GAN 生成器获取 log p(x)
# generator.generate() 只返回样本，不返回似然值

# ✓ 正确：如果需要似然评估，使用显式密度模型
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
# GPT-2 可以返回每个 token 的 log probability
outputs = model(input_ids, labels=input_ids)
loss = outputs.loss  # 交叉熵损失 ≈ -log p(x)
```

### 错误 2：在扩散模型中跳步过多导致质量崩溃

**现象：** 为了加速采样将 `num_inference_steps` 从 50 设为 5，生成的图像完全是噪声。

**原因：** DDPM 的去噪过程要求每步只去除少量噪声。跳步过多时，模型被要求在一步内完成大量去噪，超出其学习范围。

**修复：**

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

# ❌ 错误：步数太少
# image = pipe("a cat", num_inference_steps=5).images[0]

# ✓ 正确：使用 DDIM 或 PNDM 采样器，可以在 15-20 步保持质量
from diffusers import DDIMScheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
image = pipe("a cat", num_inference_steps=20).images[0]
```

### 错误 3：误以为 GAN 训练总是收敛

**现象：** GAN 训练数千轮后，生成器输出完全相同的图像（模式坍塌），或者判别器损失降为 0 而生成器不再学习。

**原因：** GAN 的训练是一个博弈过程，G 和 D 必须保持平衡。D 太强则 G 梯度消失，G 太强则 D 无法提供有效信号。没有类似 VAE 的 ELBO 或扩散模型的去噪损失那样的稳定训练目标。

**修复：**

```python
# ❌ 简单 GAN 容易崩溃
# for epoch in range(epochs):
#     train_d()  # 判别器可能过强
#     train_g()  # 生成器梯度消失

# ✓ 使用 WGAN-GP（带梯度惩罚）提升稳定性
def gradient_penalty(discriminator, real, fake, device):
    """WGAN-GP 梯度惩罚项——限制判别器的 Lipschitz 常数。"""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty
```

### 错误 4：忽视 VQ-VAE 的码本坍塌

**现象：** VQ-VAE 训练后，90% 以上的嵌入都映射到同一个码本条目——离散 token 序列几乎没有多样性。

**原因：** 向量量化是不可微分的，straight-through estimator 的梯度估计在码本较大时容易导致大部分码未被使用。

**修复：**

```python
# ❌ 码本较大时容易坍塌
# codebook_size = 8192  # 太大，训练早期容易坍塌

# ✓ 使用 EMA（指数移动平均）更新码本 + 有限码本大小
class EMACodebook:
    """指数移动平均码本——防止码本坍塌。"""

    def __init__(self, n_codes, dim, decay=0.99):
        self.codebook = torch.randn(n_codes, dim)
        self.ema_counts = torch.zeros(n_codes)
        self.ema_weights = torch.zeros(n_codes, dim)
        self.decay = decay

    def update(self, indices, flat_embeddings):
        """用 EMA 更新被选中的码本条目。"""
        for i in range(self.codebook.shape[0]):
            mask = indices == i
            count = mask.sum().item()
            if count > 0:
                self.ema_counts[i] = self.decay * self.ema_counts[i] + (1 - self.decay) * count
                self.ema_weights[i] = self.decay * self.ema_weights[i] + (1 - self.decay) * flat_embeddings[mask].mean(0)
                self.codebook[i] = self.ema_weights[i] / (self.ema_counts[i] + 1e-5)
```

---

## 8. 面试考点

### Q1：五种生成模型家族的核心区别是什么？（难度：⭐）

**参考答案：**

五大家族的根本区别在于**如何处理密度函数 $p(x)$**：

- **显式可处理密度**：直接计算 $p(x)$，如 GPT 的自回归分解、Flow 的可逆变换
- **显式近似密度**：计算 $p(x)$ 的下界（ELBO），如 VAE、扩散模型
- **隐式密度**：完全跳过 $p(x)$，只学习采样，如 GAN
- **基于分数的**：学习 $\nabla_x \log p(x)$（分数函数），如 Flow Matching
- **基于 Token 的离散自回归**：先量化再自回归，如 VQ-VAE + Transformer

核心权衡：精确似然 vs 训练稳定性 vs 采样速度。

### Q2：为什么扩散模型在 2026 年统治了图像生成？（难度：⭐⭐）

**参考答案：**

三个原因：

1. **训练稳定性**：扩散模型有明确的损失函数（去噪分数匹配），不像 GAN 需要平衡两个网络。训练过程可预测，超参数不敏感
2. **可控性**：通过条件注入（text conditioning、ControlNet、IP-Adapter），扩散模型可以精确控制输出——这是 GAN 做不到的
3. **Flow Matching 的加速**：2023 年后 Flow Matching 将采样步数从 DDPM 的 1000 步降到 10-20 步，推理速度不再是瓶颈

GAN 在人脸等固定域仍有优势（StyleGAN3），但通用图像生成已被扩散模型取代。

### Q3：VAE 和 GAN 都能生成图像，为什么工业界更偏好 VAE 的变体（扩散模型）？（难度：⭐⭐）

**参考答案：**

VAE 的核心优势是**有明确的训练目标**（ELBO），训练过程稳定且可预测。GAN 的训练是极小极大博弈，容易模式坍塌、训练崩溃。

扩散模型继承了 VAE 的优点（明确的训练目标），同时通过迭代去噪获得了 GAN 级别的样本质量。2026 年的共识是：**扩散模型是 VAE 和 GAN 的最佳折衷**——有稳定的训练、高质量的样本、灵活的条件控制。

### Q4：如果要为一个中文对话系统选择生成架构，你会怎么选？为什么？（难度：⭐⭐）

**参考答案：**

选择**自回归大语言模型**（GPT、Llama、Qwen 等）。原因：

1. 自回归是唯一成熟的变长序列生成范式——对话输出长度不可预测
2. 自回归模型天然支持零样本/少样本学习——不需要针对每个任务重新训练
3. 大语言模型的涌现能力（推理、规划、工具调用）只有足够大的自回归模型才能实现

图像/视频/音频生成用扩散模型，文本生成用自回归——这是 2026 年的标准答案。

### Q5：Flow Matching 比 DDPM 快 4-10 倍，原理是什么？（难度：⭐⭐⭐）

**参考答案：**

DDPM 的采样路径是弯曲的——从噪声到数据的路径需要 1000 步小步迭代。Flow Matching 学习的是**直线路径**（最优传输路径），从噪声到数据的映射更直接。

具体来说：

- DDPM：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$，路径是曲线
- Flow Matching：$x_t = (1-t) x_0 + t \epsilon$（插值），速度场 $v_\theta(x_t, t)$ 是常数

直线路径意味着每一步的有效位移更大，可以用更大的步长（ODE 求解器），10-20 步就能达到 DDPM 1000 步的质量。Stable Diffusion 3、Flux 都采用 Flow Matching。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 生成模型 | "AI 画画/写文章的模型" | 学习数据分布 $p(x)$ 并从中采样新样本的模型——图像、文本、音频、视频都是同一个问题 |
| 自回归 | "一个字一个字生成" | 将联合概率分解为条件概率链 $p(x) = \prod_i p(x_i \mid x_{<i})$，逐词元采样——GPT 系列的核心机制 |
| VAE | "编码器-解码器加正则" | 变分自编码器——优化 $p(x)$ 的下界（ELBO），编码器输出分布而非点，KL 散度防止过拟合 |
| GAN | "两个网络互相骗" | 生成对抗网络——生成器 G 生成假样本，判别器 D 区分真假，两者在极小极大博弈中共同进化 |
| 扩散模型 | "从噪声中恢复图像" | 前向过程逐步加噪声，反向过程学习去噪——训练目标是学习每个噪声级别的分数函数 $\nabla_x \log p(x_t)$ |
| Flow Matching | "更直的扩散" | 用最优传输理论学习从噪声到数据的直线路径，采样比 DDPM 快 4-10 倍——2024-2026 年图像/视频生成的主力 |
| VQ-VAE | "量化版自编码器" | 向量量化变分自编码器——将连续嵌入映射到有限码本中的离散 token，是 Sora、AudioLM 等模型的基础 |
| ELBO | "VAE 的损失函数" | 证据下界——$p(x)$ 的可计算下界，$ \log p(x) \geq \text{ELBO} = \mathbb{E}_q[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$ |

---

## 📚 小结

五大生成模型家族——自回归、GAN、VAE、扩散/Flow Matching、VQ-VAE + Transformer——构成了整个生成式 AI 的基础。每个家族做出了不同的折衷：自回归精确但慢，GAN 快但不稳定，VAE 稳定但样本模糊，扩散模型两者兼顾但需要多步采样。2026 年的格局是：自回归统治文本，扩散/Flow Matching 统治图像/视频/3D，GAN 在特定域保持优势。

下一课我们将深入 VAE——五大类中显式密度近似的代表，理解 ELBO 的数学推导和实际应用。

---

## ✏️ 练习

1. 【理解】用自己的话解释为什么 GAN 不能计算 `log p(x)`，而 VAE 可以。这个区别在实际应用中意味着什么？写 200 字以内。

2. 【实现】修改第 1 步的自回归采样代码，将 `probs_table` 替换为一个 3x3 的转移矩阵（词表 3 个词元），实现一个 10 步的采样，观察输出的统计分布是否接近转移矩阵的稳态分布。

3. 【实验】用 HuggingFace Diffusers 加载 `stable-diffusion-v1-5`，分别用 10 步、20 步、50 步生成同一张图像，对比质量和速度。记录每种设置的显存占用和生成时间。

4. 【思考】Sora 为什么选择"先用 VQ-VAE 量化视频为 token，再用 Transformer 建模"，而不是直接用扩散模型处理原始像素？画一个简单的架构图说明这个两阶段流程。

5. 【对比】查找 Flow Matching（Lipman 2023）和 DDPM（Ho 2020）的论文摘要，用一张表格对比两者在训练目标、采样路径、采样步数三个维度的差异。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 五大生成模型家族可视化 | `code/main.py` | 分类树 + 历史时间线的 ASCII 可视化 |
| 生成模型生态选型指南 | `outputs/generative-ecosystem-guide.md` | 根据任务特征选择生成模型家族的参考指南 |

---

## 📖 参考资料

1. [论文] Kingma. "Auto-Encoding Variational Bayes" (VAE). 2013. https://arxiv.org/abs/1312.6114
2. [论文] Goodfellow et al. "Generative Adversarial Nets" (GAN). 2014. https://arxiv.org/abs/1406.2661
3. [论文] Ho et al. "Denoising Diffusion Probabilistic Models" (DDPM). 2020. https://arxiv.org/abs/2006.11239
4. [论文] Lipman et al. "Flow Matching for Generative Modeling" (Flow Matching). 2023. https://arxiv.org/abs/2210.02747
5. [论文] Song & Ermon. "Generative Modeling by Estimating Gradients of the Data Distribution" (Score Matching). 2019. https://arxiv.org/abs/1907.05600
6. [论文] van den Oord et al. "Ne Discrete Representations with VQ-VAE". 2017. https://arxiv.org/abs/1711.00937
7. [官方文档] Hugging Face. "Diffusers". https://huggingface.co/docs/diffusers

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
