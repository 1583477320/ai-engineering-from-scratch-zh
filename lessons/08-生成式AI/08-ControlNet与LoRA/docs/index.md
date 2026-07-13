# ControlNet 与 LoRA

> ControlNet 让扩散模型遵循结构化控制（边缘图、深度图、姿态）。LoRA 让微调只更新 1% 的参数。两者结合 = 用极少计算获得精确控制的生成。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 08 · 07（潜在扩散）| **时间：** ~60 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 08 · 06（DDPM）— 理解扩散模型的基础架构 | 阶段 08 · 07（潜在扩散）— ControlNet 和 LoRA 都建立在 Stable Diffusion 之上

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解 ControlNet 的零卷积初始化——为什么它能"附加"到预训练模型而不破坏原始权重
- [ ] 解释 LoRA 的低秩分解——如何用 1-2% 的参数实现微调
- [ ] 说明 ControlNet 支持的多种控制信号类型及其应用场景
- [ ] 使用 Diffusers 加载预训练的 ControlNet 模型生成受控图像
- [ ] 训练并加载一个 LoRA 模型实现特定风格的微调

---

## 1. 问题

Stable Diffusion 的文本条件化很好——但它是"盲画"。你输入"一只猫坐在沙发上"，它会画一只猫和沙发，但你不知道猫在画面的哪个位置、什么姿势、什么角度。

ControlNet（Yu et al., 2023）解决了这个问题。它的思路很简单：**给扩散模型一份"参考图"**。这份参考图可以是边缘检测图、深度图、姿态骨架、涂鸦甚至分割图。ControlNet 学习将这份参考图的结构信息注入到 U-Net 的去噪过程中。

但 ControlNet 有个问题：它需要克隆整个 U-Net 的编码器部分，参数量翻倍。训练一个 ControlNet 需要大量 GPU 时间和海量标注数据。

LoRA（Hu et al., 2021）提供了另一种思路：**不训练整个模型，只训练一小部分参数。** 将权重矩阵分解为两个低秩矩阵的乘积，只训练这两个小矩阵。对于 Stable Diffusion 的 8.6 亿参数模型，LoRA 只需要训练约 1000 万个参数——不到 2%。

**ControlNet + LoRA = 最小成本的定制生成。** ControlNet 控制"画什么、在哪里画"，LoRA 控制"什么风格"。

---

## 2. 概念

### 2.1 ControlNet——给扩散模型一双"眼睛"

ControlNet 的核心思想：**冻结预训练的 Stable Diffusion，附加一个可训练的副本网络来学习控制信号。**

```
原始 Stable Diffusion 流程:
  文本 → CLIP → 交叉注意力 → U-Net(去噪) → 潜向量 → VAE解码 → 图像

ControlNet 增强流程:
  文本 → CLIP → 交叉注意力 → U-Net(去噪) → 潜向量 → VAE解码 → 图像
                                ↑
  控制信号 → [ControlNet 副本] → 零卷积 → 注入 U-Net 中间层
```

**三个关键设计：**

1. **克隆 U-Net 的编码器部分**：ControlNet 复制了原始 U-Net 的降采样分支（down-blocks），但冻结原始 U-Net 的所有权重
2. **零卷积初始化**：ControlNet 的输出层初始化为零矩阵——训练开始时 ControlNet 的输出为零，相当于没有 ControlNet，不破坏预训练模型
3. **中间层注入**：ControlNet 在每个降采样块的输出处通过零卷积注入特征偏移量

**为什么零卷积很重要？** 如果没有零初始化，ControlNet 从一开始就会大幅改变 U-Net 的行为，导致预训练知识被破坏。从零开始训练意味着模型可以渐进地学习控制信号，而不会突然"忘记"如何生成合理的图像。

### 2.2 控制信号类型

| 控制信号 | 生成方式 | 适用场景 | 代表模型 |
|---------|---------|---------|---------|
| Canny 边缘 | OpenCV 边缘检测 | 精确控制物体轮廓 | `controlnet-canny-sdxl-1.0` |
| 深度图 | MiDaS / ZoeDepth | 控制场景透视和距离 | `controlnet-depth-sdxl-1.0` |
| 姿态骨架 | OpenPose | 控制人物动作和姿势 | `controlnet-openpose-sdxl-1.0` |
| 涂鸦 | 用户手绘 | 草图到图像 | `controlnet-scribble` |
| 分割图 | SAM / 手动标注 | 控制物体布局和颜色 | `controlnet-segmentation` |
| 法线图 | 深度图推导 | 控制表面朝向和材质 | `controlnet-normal` |

### 2.3 LoRA——低秩自适应

LoRA 的核心洞察：**模型权重的更新矩阵通常具有低秩特性。** 也就是说，微调时权重变化的有效维度远小于参数总量。

```
原始层：y = Wx + b,    W ∈ ℝ^(d×d)    → 固定，不训练
LoRA 层：y = Wx + Δy = Wx + BAx + b
其中 A ∈ ℝ^(d×r), B ∈ ℝ^(r×d), r << d

可训练参数 = d×r + r×d = 2dr
总参数比例 = 2dr / d² = 2r/d

当 d=2048, r=16 时：
  可训练参数 = 2 × 2048 × 16 = 65,536
  参数比例 = 65,536 / 4,194,304 ≈ 1.56%
```

**α 参数（缩放因子）：** LoRA 的实际更新为 $\frac{\alpha}{r} \cdot A \cdot B$。α/r 控制了 LoRA 的强度。通常设 α = r（缩放因子为 1），但可以通过调节 α 来增强或减弱 LoRA 的效果。

**多 LoRA 切换：** 因为 LoRA 只是权重的小增量，可以在推理时轻松加载/卸载不同的 LoRA。这让你可以用同一个基础模型切换多种风格。

### 2.4 ControlNet + LoRA 的组合

```
文本提示词 → CLIP → 交叉注意力 → U-Net（含 LoRA 权重）
                                    ↑
控制信号 → ControlNet → 零卷积 → 注入 U-Net 中间层
```

ControlNet 控制空间布局（物体的位置、姿态、深度），LoRA 控制视觉风格（艺术风格、角色外观、材质质感）。两者互不干扰，可以独立训练和切换。

---

## 3. 从零实现

### 第 1 步：LoRA 的低秩分解

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """LoRA 层——低秩分解的微调模块。"""

    def __init__(self, original_layer, rank=16, alpha=1.0):
        """
        Args:
            original_layer: 原始线性层（冻结）
            rank: 低秩 r
            alpha: 缩放因子 α
        """
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.requires_grad_(False)  # 冻结原始层

        d = original_layer.in_features  # 输入维度
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 低秩矩阵 A 和 B
        # A: 降维 (d -> r)，B: 升维 (r -> d)
        self.A = nn.Parameter(torch.randn(d, rank) * 0.02)  # 高斯初始化
        self.B = nn.Parameter(torch.zeros(rank, d))         # B 初始化为零

    def forward(self, x):
        """
        前向传播：y = Wx + (α/r)·BAx
        由于矩阵乘法结合律，先计算 B@A（rank x rank），再加到 W 上。
        """
        # 方法 1：分别计算（节省显存）
        # output = self.original_layer(x) + self.scaling * (self.B @ self.A) @ x

        # 方法 2：合并权重（推理时更快）
        # 训练时将 BA 的增量加到原始权重上
        weight = self.original_layer.weight + self.scaling * self.B @ self.A
        return F.linear(x, weight, self.original_layer.bias)


class LinearWithLoRA(nn.Module):
    """带 LoRA 的线性层——更清晰的实现。"""

    def __init__(self, in_features, out_features, rank=16, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))
        self.scaling = alpha / rank
        # 冻结原始权重
        self.linear.requires_grad_(False)

    def forward(self, x):
        # 原始输出 + LoRA 增量
        return self.linear(x) + self.scaling * (self.lora_B @ self.lora_A @ x.t()).t()
```

### 第 2 步：零卷积初始化

```python
class ZeroConv(nn.Module):
    """零卷积层——ControlNet 的核心设计。
    初始化所有权重和偏置为零，确保开始时输出恒为零。
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        # 关键：权重和偏置都初始化为零
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)  # 初始时始终返回零张量
```

### 第 3 步：简化版 ControlNet 注入

```python
class SimpleControlNetInjection(nn.Module):
    """简化版 ControlNet 注入机制。"""

    def __init__(self, in_channels, controlnet_channels=32):
        super().__init__()
        # ControlNet 副本：克隆 U-Net 的下采样分支
        self.controlnet_down_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, controlnet_channels, 1),
            nn.Conv2d(in_channels, controlnet_channels, 1),
            nn.Conv2d(in_channels, controlnet_channels, 1),
            nn.Conv2d(in_channels, controlnet_channels, 1),
        ])
        # 零卷积
        self.controlnet_zero_convs = nn.ModuleList([
            ZeroConv(controlnet_channels) for _ in range(4)
        ])

    def forward(self, controlnet_cond, down_block_additional_residuals):
        """
        Args:
            controlnet_cond: 控制信号（如边缘图）经过 ControlNet 编码后的特征
            down_block_additional_residuals: 原始 U-Net 下采样块的输出

        Returns:
            注入到 U-Net 每个下采样块的零卷积输出
        """
        condensed = [c.clone() for c in down_block_additional_residuals]
        idx = 0
        for block in self.controlnet_down_blocks:
            if idx < len(condensed):
                condensed[idx] = block(controlnet_cond)
            idx += 1
        # 通过零卷积（初始时输出为零）
        return [zc(c) for zc, c in zip(self.controlnet_zero_convs, condensed)]
```

---

## 4. 工业工具

### 4.1 Diffusers 中的 ControlNet

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image
import cv2
import numpy as np

# 加载 ControlNet 模型
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16,
)

# 加载带有 ControlNet 支持的管道
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# 准备控制信号（Canny 边缘检测）
def prepare_control_image(image_path, width=512, height=512):
    img = np.array(cv2.imread(image_path))
    img = cv2.resize(img, (width, height))
    img = cv2.Canny(img, 100, 200)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    return Image.fromarray(img)

control_image = prepare_control_image("input.jpg")

# 生成受控图像
image = pipe(
    prompt="一只坐在键盘上的猫，赛博朋克风格",
    negative_prompt="模糊的，低质量的",
    image=control_image,
    num_inference_steps=50,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.8,  # 控制信号强度（0-1）
).images[0]
```

### 4.2 Diffusers 中的 LoRA

```python
from diffusers import StableDiffusionPipeline

# 加载基础模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

# 加载 LoRA 权重（从 Hugging Face Hub）
pipe.load_lora_weights(
    "custom-art-style-lora",  # LoRA 模型 ID
    adapter_name="art_style"    # 给 LoRA 起个名字
)

# 使用 LoRA 生成
pipe.set_adapter("art_style")
image = pipe(
    prompt="一只猫，毕加索立体主义风格",
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]

# 切换回无 LoRA 模式
pipe.set_adapter(None)
image = pipe(
    prompt="一只猫，写实照片风格",
    num_inference_steps=50,
).images[0]

# 同时使用多个 LoRA（加权混合）
pipe.load_lora_weights("style-a", adapter_name="style_a")
pipe.load_lora_weights("style-b", adapter_name="style_b")
pipe.set_adapter(["style_a", "style_b"])
pipe.adapter_weights = {"style_a": 0.6, "style_b": 0.4}
```

### 4.3 常用 LoRA 模型来源

| 平台 | 说明 | URL |
|------|------|-----|
| CivitAI | 最大的社区模型分享平台 | https://civitai.com |
| Hugging Face | 开源模型托管平台 | https://huggingface.co/models?search=loras |
| 魔搭社区 | 国内模型平台 | https://modelscope.cn |

---

## 5. LLM 视角

### 5.1 在主流系统中的体现

- **DALL-E 3**：虽然没有公开使用 ControlNet 或 LoRA，但其"图像编辑"功能本质上做了类似的事——通过额外的条件输入（编辑指令 + 原图）来控制生成过程
- **Midjourney v6**：引入了 "Style Reference" 功能，允许用户上传一张图片作为风格参考。这与 LoRA 的理念相似——用少量样本注入特定视觉风格
- **Stable Diffusion 3**：引入了更高效的微调方式——Dream Booth 的改进版本，可以在更少数据和更短训练时间内获得更好的风格迁移效果

### 5.2 大语言模型时代什么变了？

LoRA 已经从图像生成"溢出"到大语言模型领域。2023 年后，**LoRA 成为微调大语言模型的事实标准**：

- 全量微调一个 70 亿参数的模型需要 8 张 A100 80GB 和数天时间
- 用 LoRA（r=16）微调同一个模型只需要 1 张 A100 80GB 和几小时
- 微调后的 LoRA 权重文件通常只有 10-100MB，而全量模型权重为 140GB

这意味着个人开发者和中小团队也可以定制自己的大语言模型。

### 5.3 什么没变？

ControlNet 的核心架构——克隆编码器 + 零卷积注入——自 2023 年以来几乎没有变化。变化的只是支持的控制信号类型越来越多（从最初的 Canny、OpenPose 扩展到深度、法线、涂鸦、分割、混合控制等）。LoRA 的低秩分解数学也没有变化，但秩的选择（r=4, 8, 16, 64, 128）和应用的范围（从注意力层扩展到 FFN 层）在不断演进。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你使用 ChatGPT 的图像编辑功能（"修改这张图中的猫为狗"）时，背后可能用的是 InstructPix2Pix 或类似的指令微调模型。这些模型本质上是在 LoRA 思路上更进一步——不是用低秩分解，而是全量微调一个小模型来学习"图像到图像"的编辑操作。理解 LoRA 的低秩分解思想，有助于理解为什么有些编辑操作效果好（因为编辑的本质是权重的小幅度扰动），而有些操作做不到（因为需要大幅度改变模型行为）。

---

## 6. 工程最佳实践

### 6.1 ControlNet 条件强度调优

`controlnet_conditioning_scale` 控制 ControlNet 对生成的约束力度：

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 0.0-0.3 | 弱控制，风格自由 | 仅需大致布局参考 |
| 0.4-0.8 | 适中控制（推荐） | 大多数场景 |
| 0.9-1.2 | 强控制，严格遵循 | 需要精确对齐 |
| 1.3+ | 可能过度约束，出现伪影 | 极少使用 |

### 6.2 LoRA 训练参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `rank` (r) | 4-128 | 秩越大表达能力越强，但也更容易过拟合 |
| `alpha` | 等于 rank | 缩放因子。alpha/rank = 1 是默认 |
| `epochs` | 10-50 | 取决于数据集大小，200 张图片通常 20 轮 |
| `learning_rate` | 1e-4 - 1e-5 | 越低越稳定 |
| `network_dropout` | 0.1-0.5 | 防止过拟合，训练时随机丢弃部分 LoRA 层 |

### 6.3 中文场景特别建议

- 国内训练 LoRA 推荐使用 LiblibAI 的在线训练平台，无需自备 GPU
- ControlNet 的 OpenPose 模型对中外国人姿态识别差异较大，亚洲人脸部特征可能需要调整阈值
- 魔搭社区提供了中文优化的 ControlNet 模型，对中文提示词和 Asian 风格人物有更好的支持

### 6.4 踩坑经验

- **ControlNet 输出全黑**：`controlnet_conditioning_scale` 设为 0 了，或者控制信号图像本身是全黑的
- **LoRA 加载后图像质量下降**：LoRA 是在 SD 1.5 上训练的，但加载到了 SDXL 上——版本必须匹配
- **多个 LoRA 冲突**：不同风格的 LoRA 叠加使用时可能互相抵消。建议先测试两个 LoRA 的混合权重，再逐步增加
- **显存不足**：ControlNet + LoRA 同时使用时，显存占用约为 SD 基础的 1.5 倍。8GB 显存的 GPU 可能不够，建议 12GB+

---

## 7. 常见错误

### 错误 1：ControlNet 的 conditioning_scale 设置过高

**现象：** 生成的图像僵硬、缺乏细节、颜色失真。

**原因：** `controlnet_conditioning_scale=1.5` 意味着 ControlNet 的特征偏移量被放大了 1.5 倍，超过了 U-Net 原始特征的承载能力。

**修复：**

```python
# ❌ 错误：控制强度过高
image = pipe(prompt, image=control_image, controlnet_conditioning_scale=1.5).images[0]

# ✓ 推荐：从 0.8 开始，逐步调整
image = pipe(prompt, image=control_image, controlnet_conditioning_scale=0.8).images[0]
```

### 错误 2：LoRA 和基础模型版本不匹配

**现象：** 加载 LoRA 后生成质量严重下降，或 LoRA 完全不生效。

**原因：** LoRA 是在特定版本的基础模型上训练的。SD 1.5 的 LoRA 不能用在 SDXL 上，反之亦然。

**修复：**

```python
# ❌ 错误：版本不匹配
# LoRA 在 SD 1.5 上训练，但加载到 SDXL
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.load_lora_weights("sd15-trained-lora", adapter_name="style")

# ✓ 正确：确保版本一致
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.load_lora_weights("sd15-trained-lora", adapter_name="style")
```

### 错误 3：零卷积初始化被意外修改

**现象：** 训练 ControlNet 时一开始就生成奇怪的结果，损失震荡剧烈。

**原因：** 如果在加载预训练权重后重新初始化了 ControlNet 的零卷积层，初始为零的特性就被破坏了。

**修复：**

```python
# ❌ 错误：重新初始化破坏了零卷积
for module in controlnet.modules():
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)  # 破坏了零初始化！

# ✓ 正确：只初始化非零卷积层
for name, module in controlnet.named_modules():
    if "zero_conv" not in name:
        # 正常初始化
        pass
```

---

## 8. 面试考点

### Q1：为什么 ControlNet 要用零卷积初始化？不用零初始化会怎样？（难度：⭐⭐）

**参考答案：**
零卷积初始化确保 ControlNet 在训练开始时输出为零张量，这意味着 ControlNet 不会对预训练的 U-Net 产生任何影响。如果不用零初始化，ControlNet 从一开始就会改变 U-Net 的行为，而预训练的 U-Net 权重是针对"无控制信号"的场景优化的，突然加入随机初始化的控制信号会导致生成的图像完全不合理，训练也会难以收敛。从零初始化保证了模型可以渐进地从"无控制"过渡到"有控制"，就像给预训练模型穿上一件逐渐合身的衣服。

### Q2：LoRA 的秩 r 如何选择？r 太大或太小有什么后果？（难度：⭐⭐⭐）

**参考答案：**
秩 r 决定了低秩分解的表达能力。r 太小（如 r=1）：可训练参数极少（< 0.1%），训练速度快，但表达能力受限，无法学习复杂的风格变化，容易欠拟合。r 太大（如 r=256）：可训练参数增多（可达 5-10%），接近全量微调，失去了 LoRA 的轻量化优势，且更容易过拟合。实践中 r=16 是最常用的选择，参数占比约 1-2%，在表达能力和效率之间取得良好平衡。对于简单风格（如单一材质）可以用 r=4-8，对于复杂风格（如多角色多场景）可以用 r=32-64。

### Q3：ControlNet 和 LoRA 可以同时使用吗？如果可以，它们的权重如何合并？（难度：⭐⭐⭐）

**参考答案：**
可以同时使用，且这是最常见的组合方式。ControlNet 和 LoRA 的工作方式不同：ControlNet 通过零卷积在 U-Net 的中间层注入特征偏移量（additive residual），LoRA 通过修改 U-Net 的权重矩阵来改变模型行为（weight modification）。两者作用于不同的层面，互不冲突。在推理时，先计算 LoRA 修改后的权重进行前向传播，同时在每个下采样块注入 ControlNet 的特征偏移量。控制信号强度由 `controlnet_conditioning_scale` 调节，LoRA 的强度由 `adapter_weights` 调节。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| ControlNet | "给 AI 一张参考图" | 通过克隆并附加到预训练 U-Net 上，学习将空间控制信号（边缘/深度/姿态）注入扩散过程 |
| 零卷积 | "一开始没有作用的卷积" | ControlNet 的输出层初始化为零矩阵——训练开始时 ControlNet 不影响模型，随着训练渐进地学会控制 |
| LoRA | "只训练一点点参数" | 低秩自适应——将权重更新分解为两个小矩阵 A(r×d) 和 B(d×r) 的乘积，只训练这两个小矩阵 |
| 秩 (Rank) | "LoRA 的能力上限" | 低秩分解的内维维度。r=16 是默认值，越大表达能力越强但也越容易过拟合 |
| Conditioning Scale | "ControlNet 有多听指挥" | 控制信号对生成的约束力度。0.8 左右通常是最优的，过高会导致图像僵硬 |
| Adapter | "LoRA 的名字" | Diffusers 库中对 LoRA 权重的称呼。一个模型可以有多个 adapter，推理时切换 |

---

## 📚 小结

ControlNet 通过零卷积初始化和 U-Net 克隆副本，向预训练扩散模型注入空间控制信号，让生成不再"盲画"。LoRA 用低秩分解将微调参数降到 1-2%，让定制风格变得极其轻量。两者组合——ControlNet 控制布局，LoRA 控制风格——是当前成本最低的定制生成方案。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释为什么 ControlNet 的零卷积初始化是必要的。如果去掉零初始化，训练过程会发生什么变化？画一个对比图说明。

2. **【实现】** 实现一个带 LoRA 的线性层 `LinearWithLoRA`，支持训练模式和推理模式。推理模式下应将 LoRA 增量合并到原始权重中（`W_merged = W + (α/r)·BA`），以加速前向传播。

3. **【实验】** 从 CivitAI 下载一个公开的 LoRA 模型（如某种艺术风格），使用 Diffusers 加载并生成 4 张不同提示词的图像，记录不同 `adapter_weights` 值（0.5、1.0、1.5）对生成结果的影响。

4. **【思考】** ControlNet 只克隆了 U-Net 的编码器（下采样分支），而没有克隆解码器（上采样分支）。为什么？如果也要克隆解码器，会有什么好处和代价？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| LoRA 实现 | `code/lora.py` | 从零实现的 LoRA 低秩分解模块，含训练和推理两种模式 |
| ControlNet 注入示例 | `code/controlnet_injection.py` | 简化版 ControlNet 零卷积注入机制实现 |
| LoRA 调参指南 | `outputs/lora-tuning-guide.md` | LoRA 训练参数选择和调优指南 |

---

## 📖 参考资料

1. [论文] Yu et al. "ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models". arXiv, 2023. https://arxiv.org/abs/2302.05543
2. [论文] Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models". ICLR, 2022. https://arxiv.org/abs/2106.09685
3. [论文] Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models". CVPR, 2022. https://arxiv.org/abs/2112.10752
4. [官方文档] Hugging Face Diffusers ControlNet: https://huggingface.co/docs/diffusers/using-diffusers/controlnet
5. [官方文档] Hugging Face Diffusers LoRA: https://huggingface.co/docs/diffusers/using-diffusers/loading_adapters

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
