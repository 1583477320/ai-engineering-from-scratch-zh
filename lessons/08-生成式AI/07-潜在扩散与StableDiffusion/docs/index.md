# 潜在扩散与 Stable Diffusion

> DDPM 在像素空间做扩散太慢。潜在扩散在 VAE 的潜空间做扩散——用 64×64 的"压缩图"代替 512×512 的原始图——速度提升 10-50 倍，质量几乎不变。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 08 · 02（VAE）、06（DDPM）| **时间：** ~45 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 08 · 06（DDPM）— 理解扩散如何在压缩的潜空间运行 | 阶段 08 · 08（ControlNet 与 LoRA）— 在 Stable Diffusion 之上添加控制

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解潜在扩散的两阶段架构——VAE 编码、扩散去噪、VAE 解码
- [ ] 说明 Stable Diffusion 的三大组件及其交互方式
- [ ] 解释交叉注意力如何将文本条件注入 U-Net
- [ ] 解释为什么在潜空间做扩散比在像素空间快 10-50 倍
- [ ] 使用 Hugging Face Diffusers 加载 Stable Diffusion 生成图像

---

## 1. 问题

DDPM 在 512×512 的像素空间上做扩散——需要 1000 步去噪，每步都在 512×512 的维度上计算——极其缓慢。即使在 RTX 4090 上，生成一张 512×512 的图像也需要 30 秒以上。

解决方案很直接：**先用 VAE 将图像压缩到 64×64 的潜空间，在那里做扩散。** 计算量从 512² = 262,144 降到 64² = 4,096——理论上快 64 倍。加上 VAE 编码和解码的开销，实际加速约 10-50 倍。

2022 年 Rombach 等人的 "High-Image-Resolution Synthesis with Latent Diffusion Models" 论文将这个想法商品化——Stable Diffusion。它在消费级 GPU 上生成 1024×1024 的高质量图像，引发了开源图像生成的爆发。

---

## 2. 概念

### 2.1 潜在扩散的两阶段

```
训练阶段:
  原始图像 → [VAE 编码器 E] → 潜向量 z → [扩散加噪] → 预测噪声 ε_θ(z_t, t)

生成阶段:
  随机噪声 z_T → [U-Net 去噪 T 步] → 潜向量 z_0 → [VAE 解码器 D] → 图像
```

关键区别：扩散过程在 **潜空间（latent space）** 而非 **像素空间（pixel space）** 运行。VAE 编码器 $E$ 将 512×512×3 的图像压缩为 64×64×4 的潜向量，扩散模型在这个 4 通道的潜向量上运行，VAE 解码器 $D$ 再将潜向量还原为图像。

### 2.2 Stable Diffusion 的三大组件

| 组件 | 作用 | 参数量 | 维度变化 |
|------|------|--------|---------|
| **VAE 编码器** | 将图像压缩到潜空间 | ~60M | 512×512×3 → 64×64×4 |
| **U-Net + 交叉注意力** | 在潜空间去噪 + 接收文本条件 | ~2B | 64×64×4 → 64×64×4 |
| **VAE 解码器** | 将潜向量解码回图像 | ~60M | 64×64×4 → 512×512×3 |

**U-Net 是绝对的主力**——占总参数的 97% 以上。它包含约 35 个下采样块、35 个上采样块、以及连接它们的交叉注意力层。

### 2.3 交叉注意力——文本如何控制生成

文本到图像的桥接通过 **交叉注意力（Cross-Attention）** 实现：

```
文本 → CLIP Text Encoder → 文本嵌入序列 [c_1, c_2, ..., c_L]
                                              ↓
                          U-Net 的每个去噪块中：
                              Q 来自图像特征
                              K, V 来自文本嵌入
                              Attention(Q, K, V) = 图像"查询"文本描述
```

具体计算：

$$\text{CrossAttn}(H, C) = \text{Softmax}\left(\frac{H W_Q}{\sqrt{d}} (C W_K)^T\right) (C W_V)$$

其中 $H$ 是 U-Net 的图像特征（查询），$C$ 是 CLIP 编码的文本嵌入（键和值）。每一层交叉注意力的输出维度与输入相同，但内容已经被"文本调制"过。

**关键洞察：** 交叉注意力让 U-Net 在去噪的每一步都能"参考"文本描述。图像特征 $H$ 是"我在找什么"，文本嵌入 $C$ 是"我有什么"——注意力机制将它们配对。

### 2.4 计算量对比

| 空间 | 单步计算量（FLOPs） | 1000 步总计算量 | 512×512 图像生成时间（RTX 4090） |
|------|-------------------|---------------|-------------------------------|
| 像素空间 | ~2×10¹² | ~2×10¹⁵ | ~30 秒 |
| 潜空间（64×64×4） | ~3×10¹⁰ | ~3×10¹³ | ~1 秒 |

潜空间的加速来自两个方面：(1) 空间分辨率从 512×512 降到 64×64（64 倍）；(2) 通道数从 3 降到 4（略增），但影响微乎其微。

---

## 3. 从零实现

Stable Diffusion 的核心架构可以用不到 100 行代码理解：

```python
import torch
import torch.nn as nn

class LatentDiffusionSimple(nn.Module):
    """简化版潜在扩散模型——理解 Stable Diffusion 的三阶段架构。"""

    def __init__(self, latent_dim=4, latent_h=64, latent_w=64, embed_dim=768):
        super().__init__()
        # 1. VAE 编码器：将 3x256x256 压缩到 4x32x32
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, padding=1),   # 256->128
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 128->64
            nn.SiLU(),
            nn.Conv2d(256, latent_dim, 3, padding=1),     # 64->32 (近似 64x64)
        )
        # 2. VAE 解码器：将 4x32x32 还原到 3x256x256
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 256, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32->64
            nn.SiLU(),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),    # 64->128->256
        )
        # 3. 简化 U-Net（潜空间去噪）
        self.unet = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, latent_dim, 3, padding=1),
        )
        # 4. 文本嵌入投影（简化版交叉注意力）
        self.text_proj = nn.Linear(embed_dim, latent_dim * latent_h * latent_w)

    def encode(self, image):
        """VAE 编码：图像 → 潜向量。"""
        return self.encoder(image)

    def decode(self, latent):
        """VAE 解码：潜向量 → 图像。"""
        return self.decoder(latent)

    def forward(self, image, text_embed):
        """
        完整前向流程：
        1. 编码图像到潜空间
        2. 在潜空间中添加噪声（简化：直接传入带噪潜向量）
        3. U-Net 去噪，注入文本条件
        """
        # 编码
        z = self.encode(image)
        # 文本条件投影
        text_condition = self.text_proj(text_embed).view(
            text_embed.size(0), -1, z.size(2), z.size(3)
        )
        # U-Net 去噪（简化：单层卷积）
        noise_pred = self.unet(z)
        # 注入文本条件（简化：加法注入）
        noise_pred = noise_pred + text_condition
        return noise_pred, z
```

这个简化版本抓住了 Stable Diffusion 的核心架构：**编码→扩散→解码** 的三段式流水线。真实的 Stable Diffusion 将 U-Net 替换为一个完整的带跳跃连接的编码器-解码器，并将交叉注意力嵌入到每个残差块中。

---

## 4. 工业工具

### 4.1 Hugging Face Diffusers——一行加载 Stable Diffusion

```python
from diffusers import StableDiffusionPipeline
import torch

# 加载预训练 Stable Diffusion v1.5
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,  # 使用 FP16 节省显存
)
pipe = pipe.to("cuda")

# 生成图像
image = pipe(
    prompt="一只坐在键盘上的猫，赛博朋克风格",
    negative_prompt="模糊的，低质量的",
    num_inference_steps=50,       # 50 步采样
    guidance_scale=7.5,           # 条件强度（越高越遵循提示词）
).images[0]

image.save("cat_cyberpunk.png")
```

### 4.2 不同版本的 Stable Diffusion

| 版本 | 训练数据 | 分辨率 | 参数量 | 推荐用途 |
|------|---------|--------|--------|---------|
| SD 1.4 | LAION-400M | 512×512 | 860M | 经典、社区模型最多 |
| SD 1.5 | LAION-400M | 512×512 | 860M | 生态最成熟 |
| SD 2.1 | LAION-5B | 768×768 | 1.5B | 更高分辨率 |
| SDXL 1.0 | 10 亿级 | 1024×1024 | 2.6B | 高质量、主流选择 |
| SD3 | LAION 子集 | 1024×1024 | 2B+ | 流匹配架构，文本渲染更好 |

### 4.3 加速方案

```python
# 方案 1：使用 DPMSolver++ 采样器（15 步即可高质量生成）
from diffusers import DPMSolverMultistepScheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 方案 2：使用 torch.compile 加速（PyTorch 2.0+）
pipe.unet = torch.compile(pipe.unet)

# 方案 3：使用 xformers 注意力优化（节省显存，加速 2-3 倍）
pipe.enable_xformers_memory_efficient_attention()

# 方案 4：全量优化（适合生产环境）
pipe.enable_sequential_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
```

---

## 5. LLM 视角

### 5.1 在主流系统中的体现

扩散模型的架构思想深刻影响了大语言模型的多个方面：

- **DALL-E 2/3**：Google 的 Imagen 和 OpenAI 的 DALL-E 2 都采用了潜在扩散架构。DALL-E 3 进一步将文本编码器从 CLIP 换为 GPT-4 的文本编码器，使得图像生成对复杂提示词的理解能力大幅提升。
- **Midjourney**：基于 Stable Diffusion 架构，但在训练数据、损失函数和采样策略上做了大量工程优化。其 v6 版本使用了类似 SDXL 的更大容量模型。
- **Imagen（Google）**：引入了分层扩散——先在低分辨率生成，再逐步超分辨率。这种"粗到细"的策略显著提升了生成质量。

### 5.2 大语言模型时代什么变了？

Stable Diffusion 的出现让图像生成从"大厂专利"变成了"人人可用"。2022 年之前，只有 OpenAI、Google 等顶级实验室能训练图像生成模型。Stable Diffusion v1.5 开源后，全球开发者在其之上构建了数万种微调模型——从动漫风格到建筑渲染，从产品设计到医学可视化。

LoRA 和 ControlNet 的加入更是让定制成本降到几乎为零——训练一个特定风格的 LoRA 只需 200 张图片、一张消费级 GPU、30 分钟。

### 5.3 什么没变？

潜在扩散的三阶段架构——编码、扩散、解码——自 2022 年以来基本保持不变。变化的是每个阶段的规模和质量：VAE 从 64×64 潜空间升级到 SDXL 的 128×128，U-Net 从 860M 参数升级到 2.6B，文本编码器从 CLIP ViT-L/14 升级到 CLIP ViT-bigG/14。但那个核心公式——在潜空间做扩散——仍然是同一个。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你使用 ChatGPT 的 Image Generation 功能时，它使用的是 DALL-E 3，底层是潜在扩散架构。当你使用 Claude 的图像生成功能时，它同样基于扩散模型。你输入的提示词会被 LLM 先"润色"（扩展和细化），然后再送入扩散模型的文本编码器。这就是为什么有时候同样的提示词在 ChatGPT 中生成的效果比直接在 Stable Diffusion WebUI 中更好——因为 LLM 帮你优化了提示词。

---

## 6. 工程最佳实践

### 6.1 采样器选择

| 采样器 | 推荐步数 | 质量 | 速度 | 适用场景 |
|--------|---------|------|------|---------|
| Euler A | 20-40 | 中 | 快 | 创意探索 |
| Euler | 20-50 | 中 | 快 | 通用 |
| DPM++ 2M | 15-30 | 高 | 快 | 质量优先 |
| DPM++ SDE | 20-50 | 最高 | 中 | 极致质量 |
| DDIM | 20-100 | 中 | 快 | 确定性生成 |

### 6.2 Guidance Scale 调优

`guidance_scale` 控制提示词对生成的约束强度：

- **1-3**：自由度高，但可能偏离提示词
- **5-9**（推荐）：平衡质量和提示词遵循度
- **12+**：严格遵循提示词，但可能出现伪影和色彩过度饱和

### 6.3 中文场景特别建议

- SD 1.5 和 SDXL 的文本编码器主要针对英文训练。中文提示词效果通常不如英文，建议使用英文提示词或通过 LLM 将中文翻译为英文后再输入
- 国内可用的替代方案：通义万相（阿里）、文心一格（百度）、LiblibAI（社区平台）

### 6.4 踩坑经验

- **显存不足**：SDXL 在 FP16 下生成 1024×1024 图像需要约 6GB 显存。如果显存小于 8GB，使用 SD 1.5（512×512）或启用 `enable_sequential_cpu_offload()`
- **提示词不生效**：检查 `negative_prompt` 是否写错。SD 系列对负面提示词非常敏感，写错会导致生成质量大幅下降
- **生成的人脸畸形**：SD 1.5 在人脸细节上表现不佳。使用 SDXL 或安装 ControlNet-OpenPose 来固定人体姿态

---

## 7. 常见错误

### 错误 1：混淆潜空间维度和像素空间维度

**现象：** 计算显存占用时严重低估，导致 OOM。

**原因：** 认为"64×64×4 的潜空间和 512×512×3 的像素空间差不多"。实际上 U-Net 在处理潜空间特征时，特征图的尺寸是 64×64，而在像素空间是 512×512——面积相差 64 倍。

**修复：**

```python
# ❌ 错误：按像素空间估算显存
# 512x512x3 的特征图，假设 U-Net 中间层有 128 通道
# 128 * 512 * 512 * 2 bytes = 64 MB 每层

# ✓ 正确：按潜空间估算显存
# 64x64x4 的特征图
# 128 * 64 * 64 * 2 bytes = 1 MB 每层
# 显存占用降低了 64 倍
```

### 错误 2：guidance_scale 设置过高

**现象：** 生成的图像色彩过度饱和、出现黑色斑块、细节扭曲。

**原因：** `guidance_scale=15` 意味着条件预测和无条件预测的差距被放大 15 倍。这会将 U-Net 的注意力权重推到极端值，导致过曝和伪影。

**修复：**

```python
# ❌ 错误：guidance_scale 过高
image = pipe(prompt, guidance_scale=15).images[0]

# ✓ 推荐：从 7.5 开始调试
image = pipe(prompt, guidance_scale=7.5).images[0]
```

### 错误 3：在 CPU 上运行 Stable Diffusion

**现象：** 生成一张图像需要 5-10 分钟。

**原因：** Stable Diffusion 的 U-Net 包含约 8.6 亿参数，在 CPU 上的矩阵乘法极其缓慢。

**修复：**

```python
# ❌ 错误：默认在 CPU 上运行
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# ✓ 正确：明确指定设备
pipe = pipe.to("cuda")
# 或使用 MPS（Apple Silicon）
pipe = pipe.to("mps")
```

---

## 8. 面试考点

### Q1：为什么潜在扩散比像素空间扩散快这么多？量化说明。（难度：⭐⭐）

**参考答案：**
假设原始图像为 H×W×C，潜空间为 h×w×c。Stable Diffusion 中 H=512, W=512, C=3，h=64, w=64, c=4。空间分辨率的比为 (H×W)/(h×w) = 262144/4096 = 64。U-Net 的计算量与特征图面积成正比，因此单步计算量降低约 64 倍。加上通道数从 3 略增到 4，总体加速约 50 倍。实际中 VAE 编码/解码也有开销，所以综合加速约 10-50 倍。

### Q2：交叉注意力和自注意力的区别是什么？在 Stable Diffusion 中它们分别在哪里使用？（难度：⭐⭐⭐）

**参考答案：**
自注意力（Self-Attention）是 Q=K=V 来自同一输入，用于捕捉输入内部的依赖关系。在 Stable Diffusion 中，U-Net 的每个块内部都使用自注意力来建模图像特征的空间关系。交叉注意力（Cross-Attention）是 Q 来自一个输入（图像特征），K 和 V 来自另一个输入（文本嵌入），用于跨模态对齐。在 Stable Diffusion 中，U-Net 的每个块的中间层插入交叉注意力层，将文本条件注入图像生成过程。

### Q3：如果要在潜空间做扩散，VAE 的训练和扩散模型的训练应该如何协调？（难度：⭐⭐⭐）

**参考答案：**
有两种策略：(1) 先验训练（Stable Diffusion 的做法）：先独立训练好 VAE，冻结 VAE 权重，然后在 VAE 的潜空间上训练扩散模型。优点是简单，缺点是 VAE 的潜空间可能不是扩散模型的最优表示。(2) 联合训练：同时优化 VAE 和扩散模型，让 VAE 的潜空间专门为扩散任务优化。这更复杂但效果更好——SDXL 就采用了改进的联合训练策略，使用了一个感知压缩损失的 VAE，使得潜空间既紧凑又保留足够的生成细节。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 潜在扩散 | "在压缩空间做扩散" | 先用 VAE 将图像压缩到低维潜空间（如 64×64×4），在潜空间上做扩散过程，最后用 VAE 解码回像素空间 |
| 交叉注意力 | "文本怎么告诉模型画什么" | 一种注意力机制，Q 来自图像特征，K 和 V 来自文本嵌入——让去噪过程"参考"文本描述 |
| Guidance Scale | "提示词的权重" | 控制条件预测和无条件预测的差距放大倍数。7.5 是默认值，过高会导致伪影，过低会偏离提示词 |
| VAE | "压缩-解压图像的工具" | 变分自编码器——将图像编码为连续潜向量，并从潜向量解码回图像。在潜在扩散中充当"压缩器"和"解压���" |
| Sampler | "采样器" | 控制从噪声到图像的去噪路径。Euler 是最基础的，DPM++ 系列更快更好，DDIM 是确定性的 |
| Latent | "潜变量" | 潜空间中的表示——比像素更紧凑、更语义化。64×64×4 的潜向量等价于 512×512×3 的像素图像 |

---

## 📚 小结

潜在扩散通过 VAE 将图像压缩到 64×64 的潜空间再做扩散，速度提升 10-50 倍。Stable Diffusion 将这个架构商品化——VAE 编码、U-Net 去噪（含交叉注意力注入文本条件）、VAE 解码。文本通过 CLIP 编码器和交叉注意力控制生成。下一课我们将学习 ControlNet 和 LoRA——在 Stable Diffusion 之上添加精确控制和低成本微调。

---

## ✏️ 练习

1. **【理解】** 画出 Stable Diffusion 的完整数据流图——从文本提示词和随机噪声开始，到最终图像输出结束。标注每个模块的输入输出维度。

2. **【实现】** 修改 `LatentDiffusionSimple` 类，将简化版的文本条件注入（加法）替换为真正的交叉注意力机制。使用 `torch.nn.functional.scaled_dot_product_attention` 实现。

3. **【实验】** 使用 `diffusers` 库加载 Stable Diffusion，分别用 Euler、DPM++ 2M、DDIM 三种采样器生成同一提示词的图像，对比 10、30、50 步下的生成质量。

4. **【思考】** 如果 VAE 的压缩比进一步提高（如从 64×64×4 压缩到 16×16×2），对扩散模型的生成质量和速度分别有什么影响？是否存在一个最优压缩比？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| 潜在扩散简化实现 | `code/latent_diffusion.py` | 包含 VAE 编码/解码、U-Net 去噪、交叉注意力注入 |
| Stable Diffusion 使用模板 | `outputs/sd-prompt-template.md` | 面向中文用户的 Stable Diffusion 提示词工程模板 |

---

## 📖 参考资料

1. [论文] Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models". CVPR, 2022. https://arxiv.org/abs/2112.10752
2. [论文] Kingma & Welling. "Auto-Encoding Variational Bayes". ICLR, 2014. https://arxiv.org/abs/1312.6114
3. [论文] Radford et al. "Learning Transferable Visual Models from Natural Language Supervision". ICML, 2021. https://arxiv.org/abs/2103.00020
4. [官方文档] Hugging Face Diffusers: https://huggingface.co/docs/diffusers
5. [GitHub] Stability AI stable-diffusion: https://github.com/Stability-AI/StableDiffusion

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
