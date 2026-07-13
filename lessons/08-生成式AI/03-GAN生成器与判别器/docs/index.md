# GAN——生成器与判别器

> Goodfellow 2014 年的技巧是完全跳过密度估计。两个网络。一个造假，一个抓假。它们对抗直到假货无法区分真货。这不该有效。它经常失败。但当它有效时，样本仍然是文献中窄域上最锐利的。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 3 · 02（反向传播）、阶段 3 · 08（优化器）、阶段 08 · 02（VAE）
**预计时间：** ~75 分钟
**所处阶段：** Tier 2
**关联课程：** 第 8 阶段 · 04（条件 GAN）——本课的无条件 GAN 是条件 GAN 的基础

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 GAN 的 minimax 训练循环——生成器和判别器的交替更新
- [ ] 解释模式坍塌和梯度消失的机制——GAN 训练的两个经典失败模式
- [ ] 使用归一化熵检测模式坍塌——通过统计生成样本的像素分布来判断训练是否健康
- [ ] 说明 GAN 在 2026 年的实际用途——快速单步采样（SDXL-Turbo）和对抗蒸馏
- [ ] 比较 GAN 和扩散模型的优劣——在不同场景下做出合理的技术选型

---

## 1. 问题

VAE 产生模糊样本——因为 MSE 解码损失对多个合理重建取平均值。你想奖励**可信度**而非像素级接近。没有可信度的封闭形式——你必须学习它。

Goodfellow 的想法：训练一个分类器 `D(x)` 区分真假图像。训练生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号是 `D` 当前认为什么是"真实的"。这个信号随 `G` 的改进而变化——追逐一个移动的目标。如果两个网络都收敛了，`G` 在没有显式写出 `log p(x)` 的情况下学会了数据分布。

**GAN 的核心悖论：** 两个对抗的网络同时训练，理论上应该达到纳什均衡。但实践中它们几乎从不收敛——要么 D 太强（梯度消失），要么 G 只生成一种样本（模式坍塌）。2026 年的共识是：不要让 GAN 从零训练大模型——而是用 GAN 做"最后一公里"的蒸馏加速。

---

## 2. 概念

### 2.1 Minimax 博弈

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

**训练 D：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`
**训练 G：** `loss_G = -log D(G(z))`（非饱和形式）

非饱和形式的直觉：如果用原始的 `log(1-D(G(z)))` 作为 G 的损失，当 D 非常确信时梯度接近 0——G 无法学习。换成 `-log D(G(z))` 后，D 越确信假样本是假的，G 收到的梯度越大。

### 2.2 关键失败模式

| 模式 | 机制 | 修复 |
|---|---|---|
| **模式坍塌** | `G` 找到一个 `D` 无法判别的模式并永远生成它 | 噪声注入 + minibatch 判别 + WGAN-GP |
| **梯度消失** | `D` 训练太快，`log D` 饱和 | 降低 D 的学习率 2-5 倍 + 谱归一化 |
| **训练不稳定** | 学习率、batch size 任何变化都可能崩溃 | WGAN-GP + 谱归一化 + 标签平滑 |

### 2.3 2026 年 GAN 的真正用途

GAN 不再是 SOTA 生成器（扩散/Flow Matching 夺走了王冠）。但 GAN 在 2026 年仍然在使用：

- **SDXL-Turbo / SD3-Turbo：** 用对抗训练将 20-50 步扩散管道蒸馏为 1-4 步。**快速单步文本到图像的主导技术**
- **感知损失：** 扩散训练中使用 GAN 判别器作为感知损失
- **人脸：** StyleGAN3 仍然是固定域照片级真实感的基准

---

## 3. 从零实现

### Step 1：生成器——从噪声生成图像

```python
class Generator(nn.Module):
    """生成器：将随机噪声 z 映射为 28x28 图像。"""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )

    def forward(self, z):
        return self.net(z)
```

BatchNorm 加速训练收敛。Sigmoid 将输出压缩到 [0, 1]，与像素值范围匹配。

### Step 2：判别器——区分真假

```python
class Discriminator(nn.Module):
    """判别器：将图像映射为"真"或"假"的概率。"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
```

LeakyReLU(0.2) 而非 ReLU——避免判别器中"死神经元"问题，保持负半轴梯度流通。

### Step 3：训练循环——交替更新 D 和 G

```python
for epoch in range(NUM_EPOCHS):
    for real_images, _ in dataloader:
        batch_size = real_images.size(0)

        # --- 训练判别器（1 步）---
        # D 的目标：真图像输出接近 1，假图像输出接近 0
        z = torch.randn(batch_size, LATENT_DIM)
        fake_images = generator(z).detach()  # detach：不回传梯度到 G

        loss_d = criterion(discriminator(real_images), real_labels)
        loss_d += criterion(discriminator(fake_images), fake_labels)
        loss_d /= 2

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # --- 训练生成器（1 步）---
        # G 的目标：让 D 认为假图像是真的
        z = torch.randn(batch_size, LATENT_DIM)
        fake_images = generator(z)
        loss_g = criterion(discriminator(fake_images), real_labels)

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
```

关键细节：`fake_images.detach()` 阻止梯度回传到 G——训练 D 时不需要更新 G 的参数。

### Step 4：监测模式坍塌

```python
def check_mode_collapse(generator, n_samples=1000, n_bins=10):
    """用归一化熵检测模式坍塌。"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM)
        fake_images = generator(z).numpy().flatten()

    hist, _ = np.histogram(fake_images, bins=n_bins, range=(0, 1), density=True)
    hist = hist / hist.sum()  # 归一化为概率分布

    # 归一化熵：越接近 1 越多样，越接近 0 越坍塌
    nonzero = hist[hist > 0]
    entropy = -(nonzero * np.log(nonzero)).sum()
    return entropy / np.log(n_bins)
```

**直觉：** 健康的 GAN 生成的像素值应该接近均匀分布（熵高）。模式坍塌时，像素值集中在少数几个值附近（熵低）。

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 PyTorch 内置实现

```python
import torch.nn as nn

# 生成器——从噪声生成图像
generator = nn.Sequential(
    nn.Linear(100, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(True),
    nn.Linear(256, 784),
    nn.Sigmoid(),
)

# 判别器——判断真假
discriminator = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid(),
)

# 一步前向：毫秒级推理
z = torch.randn(16, 100)
fake_images = generator(z)  # (16, 784)——16 张 28x28 图像
scores = discriminator(fake_images)  # (16, 1)——真假概率
```

### 4.2 HuggingFace Diffusers——GAN 作为蒸馏目标

```python
from diffusers import StableDiffusionXLPipeline

# SDXL-Turbo：用 GAN 判别器蒸馏的扩散模型，1 步生成
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
).to("cuda")

image = prompt_to_image("a futuristic city at sunset", num_inference_steps=1)
# 1 步生成——这就是 GAN 对抗蒸馏的威力
```

### 4.3 StyleGAN3——固定域生成

```python
# NVIDIA 官方 StyleGAN3（人脸生成的标杆）
# https://github.com/NVlabs/stylegan3

# 推理流程：
# 1. 从潜在空间采样
# 2. 通过映射网络生成风格向量
# 3. 通过合成网络生成图像
# 全程单次前向传播——适合实时应用
```

### 4.4 性能对比

| 实现方式 | 速度 | 多样性 | 训练稳定性 | 适用场景 |
|---|---|---|---|---|
| 我们的 PyTorch 版 | 快（毫秒） | 中等 | 需调参 | 学习理解 |
| StyleGAN3 | 极快（毫秒） | 高（固定域） | 稳定 | 人脸/固定域 |
| SDXL-Turbo（1 步） | 快（~0.1s） | 高 | 稳定（蒸馏） | 快速文生图 |
| SDXL（50 步） | 慢（~5s） | 极高 | 稳定 | 高质量文生图 |
| DALL-E 3 | 中等（~2s） | 极高 | 闭源 | 商业文生图 |

---

## 5. LLM 视角

### 5.1 在大语言模型中的体现

GAN 的对抗训练思想在大语言模型时代并未消失——它以新形式回归：

- **RLHF 中的判别器角色**：奖励模型（Reward Model）本质上是一个"判别器"——判断生成的回复是好是坏。PPO 算法中的策略网络是"生成器"，奖励模型是"判别器"。这与 GAN 的 minimax 博弈结构高度相似。
- **对抗蒸馏**：SDXL-Turbo 用 GAN 判别器将 50 步扩散蒸馏为 1 步——同样的思想可以用于加速大语言模型的推理（将多次采样蒸馏为一步生成）。

### 5.2 什么变了？

GAN 从"独立生成器"变成了"蒸馏工具"。2024-2026 年最有影响力的 GAN 相关工作不是"训练一个更好的 GAN"，而是"用 GAN 判别器加速扩散模型"。这意味着：

- GAN 的训练技巧（谱归一化、标签平滑、WGAN-GP）在大语言模型的 RLHF 训练中仍然有用
- 理解 GAN 的失败模式（模式坍塌、梯度消失）有助于理解 RLHF 的训练不稳定

### 5.3 什么没变？

GAN 的核心博弈论思想——两个网络对抗、纳什均衡、非饱和损失——在任何涉及"生成 vs 判别"的场景中都适用。从 GAN 到 RLHF，到未来可能出现的 AI 对 AI 的自对齐，对抗训练的直觉是通用的。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你使用 ChatGPT 的语音模式或 DALL-E 生成图像时，背后可能有 GAN 的参与——许多实时语音合成和图像加速管道使用对抗训练。SDXL-Turbo 的 1 步文生图就是对抗蒸馏的产物：你以为扩散模型在一步步去噪，实际上是 GAN 生成器在一步到位。

---

## 6. 工程最佳实践

### 6.1 工业界常用方案

| 场景 | 推荐方案 | 备注 |
|---|---|---|
| 学习/实验 | PyTorch `nn.Sequential` 搭建 GAN | 简单直观 |
| 固定域生成 | StyleGAN3 | 人脸、特定风格的标杆 |
| 快速文生图 | SDXL-Turbo / SD3-Turbo | 对抗蒸馏的扩散模型 |
| 高质量文生图 | SDXL / SD3 / DALL-E 3 | 不需要 GAN，用扩散模型 |
| RLHF 训练 | PPO + Reward Model | GAN 博弈思想的变体 |

### 6.2 中文场景特别建议

- GAN 在中文文本生成上几乎没有应用——文本的离散性不适合 GAN 的连续梯度
- 中文人脸生成（如证件照、虚拟主播）可以使用 StyleGAN3 + 中文人脸数据集
- 如果需要中文文本生成的加速，考虑投机采样（Speculative Decoding）而非 GAN 蒸馏

### 6.3 踩坑经验

- **训练 D 太快导致 G 梯度消失**：D 的学习率应设为 G 的 1/2 ~ 1/5，这是最常见的 GAN 训练失败原因
- **忘记 `detach()` 导致梯度泄漏**：训练 D 时 `fake_images = generator(z).detach()` 是必须的，否则梯度会回传到 G
- **BatchNorm 在 G 和 D 中的行为不同**：训练 G 时用 BatchNorm，训练 D 时可以用 InstanceNorm 或谱归一化
- **不要在 D 中使用 ReLU**：使用 LeakyReLU(0.2) 避免负半轴梯度消失
- **标签平滑比不平滑好**：真标签用 0.9 而非 1.0，假标签用 0.1 而非 0.0，几乎总是提升训练稳定性

---

## 7. 常见错误

### 错误 1：训练判别器过快导致生成器梯度消失

**现象：** 训练初期 D 的损失快速下降到接近 0，G 的损失持续上升，生成的样本质量没有任何改善。

**原因：** D 的学习率过高（与 G 相同），D 过快地学会了完美区分真假，使得 `log D(G(z))` 饱和到 0，G 收不到有效梯度。

**修复：**
```python
# ❌ 错误：D 和 G 使用相同学习率
opt_d = optim.Adam(discriminator.parameters(), lr=2e-4)
opt_g = optim.Adam(generator.parameters(), lr=2e-4)

# ✓ 正确：D 的学习率是 G 的 1/2 ~ 1/5
opt_d = optim.Adam(discriminator.parameters(), lr=1e-4)
opt_g = optim.Adam(generator.parameters(), lr=2e-4)
```

### 错误 2：训练 D 时忘记 detach 生成的假图像

**现象：** 训练过程极不稳定，G 的损失不收敛，有时甚至变成 NaN。

**原因：** `fake_images = generator(z)` 没有调用 `.detach()`，导致训练 D 时梯度回传到了 G——D 和 G 的梯度混合在一起，两个网络都无法正确学习。

**修复：**
```python
# ❌ 错误：梯度泄漏到生成器
fake_images = generator(z)
loss_d = criterion(discriminator(fake_images), fake_labels)

# ✓ 正确：detach 阻止梯度回传到 G
fake_images = generator(z).detach()
loss_d = criterion(discriminator(fake_images), fake_labels)
```

### 错误 3：在判别器中使用 ReLU 激活函数

**现象：** 判别器的梯度在负半轴完全消失，D 对某些输入模式无法学习。

**原因：** ReLU 将所有负值设为 0，导致 D 的某些神经元永远不激活（"死神经元"）。LeakyReLU 保留负半轴的小梯度，让 D 能够持续学习。

**修复：**
```python
# ❌ 错误：ReLU 导致死神经元
nn.Linear(256, 128),
nn.ReLU(),  # 负半轴梯度为 0

# ✓ 正确：LeakyReLU 保留负半轴梯度
nn.Linear(256, 128),
nn.LeakyReLU(0.2),  # 负半轴有 0.2 的斜率
```

### 错误 4：没有模式坍塌检测，训练结束才发现问题

**现象：** 训练了 100 个轮次，损失曲线看起来正常，但生成的样本全部一样。

**原因：** 只看 D 和 G 的损失无法发现模式坍塌——损失可能看起来"正常"，但 G 已经坍塌到只生成一种样本。

**修复：**
```python
# ❌ 只看损失
print(f"D: {loss_d:.4f} G: {loss_g:.4f}")  # 损失正常不代表样本多样

# ✓ 同时检测多样性
if (epoch + 1) % 5 == 0:
    diversity = check_mode_collapse(generator)
    print(f"D: {loss_d:.4f} G: {loss_g:.4f} 多样性: {diversity:.4f}")
    if diversity < 0.1:
        print("  [!] 模式坍塌！考虑降低 G 学习率或增加噪声")
```

### 错误 5：生成器和判别器容量不匹配

**现象：** 生成的样本模糊，或者 D 太强导致 G 无法学习。

**原因：** D 太小（容量不足）无法区分真假，或者 D 太大（容量过剩）轻松碾压 G。理想情况下 D 和 G 的容量应该接近。

**修复：**
```python
# ❌ D 比 G 大太多——D 轻松碾压 G
generator = nn.Sequential(nn.Linear(64, 128), ...)     # G: 64→128
discriminator = nn.Sequential(nn.Linear(784, 1024), ...) # D: 784→1024

# ✓ D 和 G 容量接近
generator = nn.Sequential(nn.Linear(64, 256), ...)     # G: 64→256→256→784
discriminator = nn.Sequential(nn.Linear(784, 256), ...) # D: 784→256→256→1
```

---

## 8. 面试考点

### Q1：GAN 的训练目标是什么？为什么叫 minimax 博弈？（难度：⭐）

**参考答案：**

GAN 的训练目标是一个零和博弈：生成器 `G` 想要最大化判别器 `D` 将假样本判为真的概率，判别器 `D` 想要最大化区分真假的能力。数学上表示为 `min_G max_D V(D,G)`——D 试图最大化目标函数，G 试图最小化它。这与博弈论中的零和博弈结构一致。

### Q2：什么是模式坍塌？如何检测和缓解？（难度：⭐⭐）

**参考答案：**

模式坍塌是指生成器只学会生成少数几种样本，尽管训练数据有多种模式。例如在 MNIST 上训练 GAN，生成器可能只输出数字"1"和"7"，完全忽略其他数字。

**检测方法：** 统计生成样本的像素分布归一化熵——熵接近 0 表示坍塌，接近 1 表示多样。也可以计算 FID 或 Inception Score。

**缓解方法：** 标签平滑（真标签 0.9）、降低 D 的学习率、WGAN-GP（用 Wasserstein 距离替代 BCE）、minibatch 判别。

### Q3：为什么 GAN 使用非饱和损失 `-log D(G(z))` 而非原始的 `log(1-D(G(z)))`？（难度：⭐⭐）

**参考答案：**

当 D 训练得很好时，`D(G(z))` 接近 0——此时 `log(1-D(G(z)))` 的梯度接近 0（饱和），G 无法学习。而 `-log D(G(z))` 在 `D(G(z))` 接近 0 时梯度很大——给 G 强学习信号。这称为"非饱和"形式。

### Q4：RLHF 中的 PPO 和 GAN 有什么相似之处？（难度：⭐⭐⭐）

**参考答案：**

两者都涉及对抗/博弈结构：GAN 中 G 和 D 对抗训练；RLHF 中策略网络（生成器）和奖励模型（判别器）的博弈。两者都面临训练不稳定的问题——GAN 有模式坍塌和梯度消失，RLHF 有奖励黑客（Reward Hacking）。非饱和损失的直觉在 RLHF 中也有应用——当奖励模型过于自信时，PPO 的 clipped objective 防止策略更新过大。

### Q5：2026 年，什么情况下还会选择 GAN 而非扩散模型？（难度：⭐⭐）

**参考答案：**

1. **推理延迟是硬约束**：边缘设备、实时视频流——GAN 单步推理（毫秒级），扩散模型需要多步去噪
2. **固定域生成**：人脸（StyleGAN3）、特定风格的图像——GAN 在固定域上质量仍然最高
3. **作为蒸馏目标**：SDXL-Turbo 用 GAN 判别器加速扩散模型——这是 GAN 在 2026 年最主流的新角色

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 生成器 | "造假的网络" | 从随机噪声生成样本的网络 `G: z → x̂`，目标是骗过判别器 |
| 判别器 | "打假的网络" | 二分类器 `D: x → [0,1]`，区分真实样本和生成样本 |
| 非饱和损失 | "换了个损失函数" | G 用 `-log D(G(z))` 而非 `log(1-D(G(z)))`——避免 D 太强时 G 梯度消失 |
| 模式坍塌 | "生成的东西都一样" | G 只学会生成少数几种样本，尽管数据有多种模式——训练失败的标志 |
| 对抗训练 | "两个网络互相对打" | G 和 D 交替更新，各自优化自己的目标——理论目标是纳什均衡 |
| WGAN | "更好的 GAN" | 用 Wasserstein 距离替代 BCE——损失更平滑，训练更稳定，不需要模式坍塌检测 |
| 谱归一化 | "限制权重大小" | 将 D 的每层权重矩阵除以最大奇异值——防止 D 训练过快，提升稳定性 |
| 对抗蒸馏 | "用 GAN 加速扩散" | 用 GAN 判别器作为额外损失，将多步扩散模型蒸馏为 1-4 步——2026 年 GAN 的主流用途 |

---

## 📚 小结

GAN 的核心是对抗训练：生成器学会造假，判别器学会抓假，两者博弈直到假货无法区分。两个经典失败：模式坍塌和梯度消失——WGAN-GP 和谱归一化分别解决了它们。2026 年 GAN 不再是 SOTA 生成器——但对抗训练作为蒸馏工具（SDXL-Turbo）仍然主导快速推理。理解 GAN 的博弈论思想，对理解 RLHF 的训练机制也有直接帮助。

下一课我们将学习条件 GAN——如何用标签控制生成器输出指定类别的图像。

---

## ✏️ 练习

1. **【实现】** 运行 `code/main.py`（默认参数），观察训练过程中 D 和 G 损失的变化。然后设 `D_LR = 5 * G_LR`，对比两种设置下的生成质量和训练稳定性。

2. **【实验】** 将 `BATCH_SIZE` 从 128 分别改为 32 和 256，观察模式坍塌检测的归一化熵变化。解释 batch size 对 GAN 训练的影响。

3. **【实现】** 修改 `Generator` 的隐藏层大小为 64（原来是 256），训练并对比生成质量。再改为 512，继续对比。解释生成器容量与生成质量的关系。

4. **【分析】** 阅读 `outputs/gan-guide.md` 中的选型矩阵。针对你的一个具体项目（如证件照生成、商品图片生成、虚拟主播），选择 GAN 还是扩散模型，并给出理由。

5. **【思考】** RLHF 中的 PPO 算法与 GAN 的训练有哪些相似之处？写出 3 点类比，并解释为什么 RLHF 没有"模式坍塌"这个概念（提示：想想输出空间的差异）。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| GAN 训练代码 | `code/main.py` | PyTorch 实现的完整 GAN 训练循环，含模式坍塌检测和 WGAN 损失 |
| 选型指南 | `outputs/gan-guide.md` | GAN vs 扩散模型的实用选型对比，含迁移建议 |

---

## 📖 参考资料

1. [论文] Goodfellow et al. "Generative Adversarial Nets". NeurIPS, 2014. https://arxiv.org/abs/1406.2661
2. [论文] Radford et al. "Unsupervised Representation Learning with Deep Convolutional GAN" (DCGAN). ICLR, 2016. https://arxiv.org/abs/1511.06434
3. [论文] Arjovsky et al. "Wasserstein GAN". ICML, 2017. https://arxiv.org/abs/1701.07875
4. [论文] Miyato et al. "Spectral Normalization for Generative Adversarial Networks". ICLR, 2018. https://arxiv.org/abs/1802.05957
5. [论文] Karras et al. "Analyzing and Improving the Image Quality of StyleGAN" (StyleGAN2). CVPR, 2020. https://arxiv.org/abs/1912.04958
6. [论文] Sauer et al. "Adversarial Diffusion Distillation" (SDXL-Turbo). 2023. https://arxiv.org/abs/2311.17042

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
