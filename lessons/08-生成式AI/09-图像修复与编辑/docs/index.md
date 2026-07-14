# 图像修复与编辑

> 修复（inpainting）填补缺失区域；外扩（outpainting）扩展边界；编辑修改特定区域。三者都是"条件扩散+掩码"的变体。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 08 · 07（潜在扩散）| **时间：** ~45 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 08 · 07（潜在扩散）— 修复与编辑建立在 Stable Diffusion 之上 | 阶段 08 · 08（ControlNet 与 LoRA）— 结构化控制与风格微调

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 理解 inpainting 的掩码机制——在二进制掩码下，只对未遮盖区域进行去噪
- [ ] 解释 outpainting 如何通过上下文扩展实现——从已知像素向外生成
- [ ] 说明 InstructPix2Pix 如何用文本指令编辑图像——"把天空变蓝"
- [ ] 使用 Diffusers 库对图像进行修复、外扩和编辑
- [ ] 区分 inpainting、outpainting、图像编辑三种任务的技术差异

---

## 1. 问题

你拍了一张完美的照片，但背景里有一根碍眼的电线杆。你想把它删掉，但不想重拍。传统做法是用 Photoshop 手动修补——需要专业技能和大量时间。

扩散模型改变了这一切。**给模型一张图和一个掩码，它就能自动"脑补"缺失区域的内容。** 而且效果远好于传统方法——因为它理解"场景的语义"，知道电线杆后面应该是什么。

修复（inpainting）、外扩（outpainting）、编辑（editing）是三种最常见的图像修改任务，它们共享同一个核心原理：**在扩散过程中，用掩码或条件信号告诉模型"哪些区域需要生成、哪些区域保持原样"。**

---

## 2. 概念

### 2.1 直观理解

想象你有一张被撕了一角的照片。修复的过程就像"看着照片其余部分，猜测撕掉的那块应该是什么内容"。

```
原始图像:    [完整照片]
掩码:        [白色区域=缺失，黑色区域=保留]
修复后:      [模型"脑补"出缺失区域]
```

关键：模型不是随机填充——它利用已知像素的上下文信息来生成连贯的内容。

### 2.2 Inpainting——修复缺失区域

Stable Diffusion 的 inpainting 流程：

```
输入:
  原始图像 (512×512×3)  +  二进制掩码 (512×512×1)
      ↓                        ↓
  [VAE 编码]              [保持不变]
      ↓                        ↓
  带噪潜向量 z_t         掩码潜向量
      ↓                        ↓
  [U-Net 去噪] ← 掩码告诉模型哪些区域需要生成
      ↓
  [VAE 解码] → 修复后的完整图像
```

核心机制：**将原始图像（含已知区域）作为条件输入到 U-Net 的第一个卷积层，掩码作为额外通道一起输入。** 去噪时，已知区域的特征保持与原始图像一致，未知区域由模型生成。

### 2.3 Outpainting——向外扩展

Outpainting 是 inpainting 的特例——掩码覆盖图像的边缘区域。关键是**上下文窗口**：模型需要"看到"原始图像的边缘内容，才能向外生成连贯的内容。

```
原始图像  [512×512]
扩展画布  [1024×1024]
掩码: 原图区域=0，空白区域=1
生成: 模型从原图边缘"向外延伸"
```

### 2.4 InstructPix2Pix——文本指令编辑

InstructPix2Pix（Brooks et al., 2023）的创新：**用自然语言指令指导图像编辑**。"把天空变蓝"、"让猫戴帽子"、"添加一个太阳"。

```
输入:
  原始图像 + 文本指令 "把天空变蓝"
      ↓
  [CLIP 编码指令] → 条件嵌入
      ↓
  [条件 U-Net 去噪] → 只修改天空区域
      ↓
  编辑后的图像
```

关键技术：使用一个 **instruction strength** 参数（$t_{\text{inst}}$）控制编辑强度。$t_{\text{inst}}=0$ 表示不修改，$t_{\text{inst}}=1$ 表示完全重绘。通常设为 0.5-0.8 以保持原图结构。

### 2.5 三种任务的技术差异

| 任务 | 掩码位置 | 条件信号 | 典型应用 |
|------|---------|---------|---------|
| Inpainting | 图像内部 | 原始图像 + 掩码 | 去除物体、修复损坏照片 |
| Outpainting | 图像边缘 | 原始图像 + 掩码 | 扩展画布、生成背景 |
| 图像编辑 | 自动定位（由指令决定） | 原始图像 + CLIP 文本嵌入 | 改颜色、换背景、加元素 |

---

## 3. 从零实现

### 第 1 步：Inpainting 掩码处理

```python
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

def create_inpainting_mask(image_size=(512, 512), mask_region="center",
                           mask_ratio=0.3):
    """
    创建 inpainting 掩码。
    Args:
        image_size: 图像尺寸 (H, W)
        mask_region: 掩码区域 "center" / "random" / "manual"
        mask_ratio: 掩码占图像的比例
    Returns:
        mask: 二进制掩码 (1=需要生成, 0=保留)
    """
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.uint8)

    if mask_region == "center":
        # 中心区域掩码
        h_start = int(H * (1 - mask_ratio) / 2)
        h_end = int(H * (1 + mask_ratio) / 2)
        w_start = int(W * (1 - mask_ratio) / 2)
        w_end = int(W * (1 + mask_ratio) / 2)
        mask[h_start:h_end, w_start:w_end] = 1
    elif mask_region == "random":
        # 随机区域掩码
        num_masks = np.random.randint(1, 4)
        for _ in range(num_masks):
            cy, cx = np.random.randint(0, H), np.random.randint(0, W)
            ry, rx = np.random.randint(20, H//4), np.random.randint(20, W//4)
            Y, X = np.ogrid[:H, :W]
            mask[((Y - cy)**2 / ry**2 + (X - cx)**2 / rx**2) < 1] = 1

    return torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)


def prepare_inpainting_input(image, mask, noise_level=0.5):
    """
    准备 inpainting 输入：将原始图像、掩码、噪声组合为模型输入。
    """
    # 将掩码扩展为与图像相同的通道数
    mask_3ch = mask.unsqueeze(1).expand_as(image)  # (B, 3, H, W)

    # 在掩码区域添加噪声
    noise = torch.randn_like(image)
    noisy_image = (1 - mask_3ch) * image + mask_3ch * noise

    # 模型输入：[带噪图像, 掩码]
    model_input = torch.cat([noisy_image, mask_3ch], dim=1)  # (B, 6, H, W)
    return model_input
```

### 第 2 步：Outpainting 画布扩展

```python
def create_outpainting_canvas(image, target_size=(1024, 1024)):
    """
    创建 outpainting 画布：将原始图像放在中心，周围填充噪声。
    """
    B, C, H, W = image.shape
    target_H, target_W = target_size

    # 创建空白画布
    canvas = torch.randn(B, C, target_H, target_W)

    # 将原始图像放在左上角（或任意位置）
    canvas[:, :, :H, :W] = image

    # 创建掩码：原始图像区域=0，空白区域=1
    mask = torch.ones(B, 1, target_H, target_W)
    mask[:, :, :H, :W] = 0

    return canvas, mask


def outpaint_with_overlap(image, overlap=64):
    """
    带重叠区域的 outpainting——避免接缝。
    """
    B, C, H, W = image.shape
    new_H = H * 2
    new_W = W * 2

    canvas = torch.randn(B, C, new_H, new_W)
    mask = torch.ones(B, 1, new_H, new_W)

    # 原始图像放在左上角
    canvas[:, :, :H, :W] = image
    mask[:, :, :H, :W] = 0

    # 重叠区域：允许模型微调以避免接缝
    mask[:, :, H-overlap:H, :W] = 0.5  # 半掩码=混合

    return canvas, mask
```

### 第 3 步：InstructPix2Pix 的条件注入

```python
class InstructPix2PixConditioning(nn.Module):
    """简化版 InstructPix2Pix 条件注入——展示指令如何控制编辑。"""

    def __init__(self, text_dim=768, image_dim=4):
        super().__init__()
        # 指令嵌入投影
        self.instruction_proj = nn.Linear(text_dim, 768)
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 768),
        )
        # 指令强度调制
        self.strength_proj = nn.Linear(1, 768)

    def forward(self, text_embed, timestep, instruction_strength=0.7):
        """
        将文本指令和时间步融合为条件信号。
        instruction_strength 控制编辑强度（0=不编辑, 1=完全重绘）。
        """
        # 投影文本嵌入
        cond = self.instruction_proj(text_embed.mean(dim=1))  # (B, 768)

        # 注入时间步信息
        t_emb = self.time_embed(torch.zeros(cond.size(0), 256))
        cond = cond + t_emb

        # 指令强度调制
        strength = torch.full((cond.size(0), 1), instruction_strength, device=cond.device)
        strength_emb = self.strength_proj(strength)
        cond = cond * strength_emb

        return cond
```

---

## 4. 工业工具

### 4.1 Diffusers 中的 StableDiffusionInpaintPipeline

```python
from diffusers import StableDiffusionInpaintPipeline
import torch

# 加载 inpainting 管道（专门为修复任务微调的 SD 模型）
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# 修复图像
from PIL import Image
image = Image.open("damaged_photo.jpg").resize((512, 512))
mask = Image.open("mask.png").resize((512, 512))

result = pipe(
    prompt="一个干净的客厅，没有电线杆",
    image=image,
    mask_image=mask,
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]
result.save("repaired_photo.png")
```

### 4.2 InstructPix2Pix

```python
from diffusers import StableDiffusionInstructPix2PixPipeline

# 加载指令编辑模型
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# 用文本指令编辑图像
image = Image.open("landscape.jpg").resize((512, 512))
result = pipe(
    prompt="把天空变成日落的橙色",
    image=image,
    image_guidance_scale=1.5,  # 图像与指令的平衡
    guidance_scale=7.5,
).images[0]
result.save("sunset_landscape.png")
```

### 4.3 性能对比

| 方法 | 模型 | 分辨率 | 推理时间 | 适用场景 |
|------|------|--------|---------|---------|
| SD Inpainting | Stable Diffusion Inpaint | 512×512 | ~5 秒 | 内部修复 |
| InstructPix2Pix | instruct-pix2pix | 512×512 | ~5 秒 | 文本指令编辑 |
| DALL-E 3 Edit | DALL-E 3 | 1024×1024 | ~10 秒 | ChatGPT 集成 |
| Photoshop Generative Fill | 自研模型 | 1024×1024 | ~3 秒 | 专业修图 |
| Flux Fill | Flux.1 | 1024×1024 | ~4 秒 | 2026 主流 |

---

## 5. LLM 视角

### 5.1 在主流系统中的体现

- **DALL-E 3 + ChatGPT**：当你说"把这张图的天空换成日落"时，ChatGPT 先用 LLM 理解指令，再调用 DALL-E 的 inpainting/editing 功能执行。LLM 负责"理解意图"，扩散模型负责"生成图像"。
- **Midjourney 的 Vary (Region)**：允许用户选择图像的一个区域并用提示词重新生成该区域。底层使用 inpainting 技术。
- **Stable Diffusion WebUI 的 inpainting**：开源社区最常用的修复工具，支持多种采样器和参数调节。

### 5.2 大语言模型时代什么变了？

2024-2025 年，图像编辑从"手动调参"变成了"对话式交互"。你不需要自己画掩码、调 guidance_scale——只需要用自然语言告诉系统你想要什么修改。这背后是 LLM 和扩散模型的深度集成：LLM 将你的自然语言指令转化为结构化的编辑命令（掩码区域、修改内容、强度参数），然后交给扩散模型执行。

### 5.3 什么没变？

掩码机制是 inpainting 的核心——从 2020 年的 DeepFill 到 2026 年的最新模型，"掩码=1 的区域生成，掩码=0 的区域保留"这个基本范式没有变。变化的是生成的质量和速度——从模糊的 GAN 输出变成了清晰的扩散模型输出，从需要 30 秒变成了 5 秒。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你让 ChatGPT "修改这张照片的背景"时，它实际上在做：(1) 理解你的修改意图（LLM 能力）；(2) 生成掩码（可能是自动的，也可能是你手动选择）；(3) 调用 inpainting 模型生成新内容。理解掩码机制有助于你更精准地描述修改需求——告诉系统"只修改背景，保留人物"比"修改这张照片"效果好得多。

---

## 6. 工程最佳实践

### 6.1 掩码边缘处理

| 策略 | 效果 | 适用场景 |
|------|------|---------|
| 硬边掩码 | 修复区域与原图有明显边界 | 简单修复 |
| 羽化掩码 | 边缘渐变过渡，更自然 | 大多数场景（推荐） |
| 带重叠的掩码 | 修复区域向外扩展若干像素 | 高质量修复 |

### 6.2 提示词策略

- **Inpainting**：描述修复后应该是什么样子（不是描述原始图像）
- **Outpainting**：描述扩展区域应该包含什么内容
- **编辑**：用简洁的指令描述修改（"把天空变成蓝色"而非"让天空变成蓝色的晴天"）

### 6.3 中文场景特别建议

- 中文提示词在 SD inpainting 模型中的效果通常不如英文，建议先用 LLM 翻译为英文
- 国内可用的替代方案：魔搭社区的 Inpainting 模型、百度文心一格的图像编辑功能

### 6.4 踩坑经验

- **修复区域模糊**：将 `num_inference_steps` 从默认的 50 增加到 75-100，或降低 `guidance_scale` 到 5-7
- **修复内容与原图风格不一致**：在提示词中加入风格描述（如"与原图一致的写实风格"）
- **Outpainting 接缝明显**：使用带重叠区域的掩码（overlap=32-64 像素），让模型在接缝处做混合

---

## 7. 常见错误

### 错误 1：掩码的 0 和 1 定义搞反

**现象：** 修复了你不想修改的区域，保留了你想修复的区域。

**原因：** 不同框架对掩码的定义不同。Diffusers 中 1=修复区域（需要生成），0=保留区域。但有些自定义实现可能反过来。

**修复：**

```python
# ❌ 错误：掩码定义搞反
mask = torch.zeros(1, 1, 512, 512)  # 全零 = 不修复任何区域

# ✓ 正确：需要修复的区域设为 1
mask = torch.ones(1, 1, 512, 512)  # 全一 = 整张图重新生成
mask[:, :, 100:400, 100:400] = 0   # 中间区域保留
```

### 错误 2：使用通用 SD 模型做 inpainting

**现象：** 修复区域出现明显的边界伪影，内容与原图不连贯。

**原因：** 通用的 Stable Diffusion 模型没有在 inpainting 任务上微调。它不知道如何利用掩码信息——只是把整张图当作条件输入。

**修复：**

```python
# ❌ 错误：使用通用模型
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# ✓ 正确：使用专门的 inpainting 模型
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting"
)
```

### 错误 3：Outpainting 时原图放在画布边缘

**现象：** 扩展区域与原图在接缝处有明显不连贯。

**原因：** 将原图直接放在画布边缘，模型没有足够的上下文来推断边缘处的内容。

**修复：**

```python
# ❌ 错误：原图紧贴画布边缘
canvas[:, :, :H, :W] = image

# ✓ 正确：原图放在中心，周围留出空间
offset_h = (target_H - H) // 2
offset_w = (target_W - W) // 2
canvas[:, :, offset_h:offset_h+H, offset_w:offset_w+W] = image
```

---

## 8. 面试考点

### Q1：Inpainting 中，掩码是如何在 U-Net 中发挥作用的？（难度：⭐⭐）

**参考答案：**
在 Stable Diffusion 的 inpainting 模型中，掩码被作为额外的输入通道拼接到潜向量上。原始潜向量是 4 通道，掩码是 1 通道，所以 inpainting 模型的第一个卷积层接受 5 通道输入（4 通道潜向量 + 1 通道掩码）。在去噪过程中，U-Net 通过这个掩码通道知道哪些区域需要生成（掩码=1）、哪些区域保持原样（掩码=0）。具体实现中，还会将原始图像的潜向量拼接为 9 通道输入（4 通道带噪潜向量 + 4 通道原始图像潜向量 + 1 通道掩码），让模型同时看到原图和掩码。

### Q2：InstructPix2Pix 和传统 inpainting 有什么本质区别？（难度：⭐⭐⭐）

**参考答案：**
传统 inpainting 需要用户提供明确的二进制掩码来指定修复区域。InstructPix2Pix 不需要显式掩码——它通过 CLIP 文本编码器理解编辑指令（如"把天空变蓝"），然后由模型自动定位需要修改的区域。它的训练数据是通过 GPT-4 生成指令-图像对，然后用 Stable Diffusion 生成编辑前后的对比数据来训练的。`image_guidance_scale` 参数控制编辑的忠实度——值越大越接近原始图像，值越小越倾向于完全遵循文本指令。

### Q3：如何处理大面积修复（掩码覆盖超过 50% 的图像）？（难度：⭐⭐⭐）

**参考答案：**
大面积修复的主要挑战是：(1) 缺乏足够的上下文信息——已知像素太少，模型难以推断缺失区域的内容；(2) 生成内容可能不连贯。解决方案包括：(1) 分块修复——将大掩码拆分为多个小掩码，逐块修复，重叠区域用加权混合；(2) 多次迭代——先用粗略的提示词修复大区域，再用细化的提示词修复细节；(3) 使用 ControlNet 提供结构约束——在大面积修复时，用边缘图或深度图控制生成的结构。工业实践中，Photoshop 的 Generative Fill 采用了类似策略：先粗后细，逐步填充。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Inpainting | "给照片补洞" | 给定图像和二进制掩码，用扩散模型在掩码区域生成新内容，其余区域保持不变 |
| Outpainting | "扩展画布" | 在图像边缘区域生成新内容，使图像"向外生长"——本质上是边缘区域的 inpainting |
| InstructPix2Pix | "用文字修图" | 用自然语言指令（如"把天空变蓝"）指导图像编辑，不需要手动绘制掩码 |
| 掩码 (Mask) | "遮住要改的部分" | 二进制张量：1=需要生成的区域，0=保留原样的区域 |
| Image Guidance Scale | "图片和指令的权重" | InstructPix2Pix 专用参数——控制编辑结果对原始图像的忠实度 |
| 重叠区域 | "修复边缘时的过渡带" | 在 inpainting 中，掩码边缘向外扩展若干像素，让模型在接缝处做自然混合 |

---

## 📚 小结

Inpainting/Outpainting/Edit 都是"条件扩散+掩码"的变体——掩码告诉模型哪些区域需要生成。InstructPix2Pix 用 CLIP 文本编码实现了文本指令编辑，无需手动绘制掩码。2026 年这些功能已被集成进 DALL-E 3、Midjourney、Stable Diffusion 的 UI 中，成为图像编辑的标准能力。下一课我们将进入视频生成——将扩散模型从二维图像扩展到三维视频。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释 inpainting 中"掩码=1 的区域生成，掩码=0 的区域保留"是如何在 U-Net 中实现的。为什么需要将掩码作为额外通道输入？

2. **【实现】** 修改 `create_inpainting_mask` 函数，支持生成羽化掩码（边缘渐变而非硬边）。羽化宽度设为 10-20 像素，对比硬边和羽化掩码的修复效果差异。

3. **【实验】** 使用 Diffusers 的 `StableDiffusionInpaintPipeline`，对同一张图片分别尝试：(a) 10% 掩码面积；(b) 30% 掩码面积；(c) 60% 掩码面积。观察修复质量随掩码面积的变化规律。

4. **【思考】** 如果要修复一张视频中的某一帧（视频修复），与单张图像修复有什么额外挑战？如何保证修复后的帧与相邻帧的时间一致性？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| Inpainting 掩码工具 | `code/inpainting_utils.py` | 创建掩码、准备输入、羽化处理 |
| Outpainting 画布工具 | `code/outpainting_canvas.py` | 画布扩展、重叠区域处理 |
| 编辑指令注入示例 | `code/instruct_editing.py` | InstructPix2Pix 条件注入实现 |

---

## 📖 参考资料

1. [论文] Brooks et al. "InstructPix2Pix: Learning to Follow Image Editing Instructions". CVPR, 2023. https://arxiv.org/abs/2211.09800
2. [论文] Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models". CVPR, 2022. https://arxiv.org/abs/2112.10752
3. [论文] Yu et al. "Scaling Autoregressive Models for Content-Rich Text-to-Image Generation". arXiv, 2022. https://arxiv.org/abs/2206.10789
4. [官方文档] Hugging Face Diffusers Inpainting: https://huggingface.co/docs/diffusers/using-diffusers/inpaint
5. [GitHub] Stability AI inpainting: https://github.com/Stability-AI/stablediffusion

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
