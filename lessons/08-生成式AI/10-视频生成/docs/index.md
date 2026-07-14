# 视频生成

> 图像是二维的；视频是三维的（时间+空间）。扩散模型加上时间维度——变成视频扩散——就是 Sora、Kling、Runway 的核心技术。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 08 · 07（潜在扩散）| **时间：** ~45 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 08 · 06（DDPM）— 扩散模型基础 | 阶段 08 · 07（潜在扩散）— 视频扩散建立在潜在扩散之上

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 说明视频扩散与图像扩散的关键差异——时间维度、帧间一致性、计算开销
- [ ] 区分 Sora、Kling、Runway Gen-3 的架构差异
- [ ] 解释 2026 年视频生成的两个前沿——Consistency Model 和 Rectified Flow
- [ ] 使用 Diffusers 加载视频生成模型并生成短视频
- [ ] 解释时空 patches 如何将视频编码为 Transformer 可处理的序列

---

## 1. 问题

图像扩散在 512×512 上做 20-50 步已经很快了。但视频是 3D 的（H×W×T）——T=24 帧/秒 × 10 秒 = 240 帧——计算量增长了 240 倍。而且视频有**时间一致性**要求——连续帧必须看起来连贯，不能闪烁或跳变。

2024-2025 年，OpenAI 的 Sora、快手的 Kling、Runway 的 Gen-3 Alpha 证明了视频扩散模型可以生成电影级质量的视频。2026 年，一致性和流匹配模型将推理速度从分钟级压缩到秒级。

---

## 2. 概念

### 2.1 视频扩散的核心挑战

| 挑战 | 图像 | 视频 |
|------|------|------|
| 维度 | 2D (H×W) | 3D (H×W×T) |
| 一致性 | 无时间约束 | 连续帧必须连贯 |
| 计算量 | 基线 | T 倍（T=帧数） |
| 光流 | 无 | 物体运动必须平滑 |

### 2.2 三种架构

#### 架构 1：3D U-Net

```
视频帧序列 → [3D 卷积] → 时空特征 → [3D 反卷积] → 输出视频
```

将时间维度当作第三个空间维度——用 3D 卷积替代 2D 卷积。简单直接，但计算量随帧数线性增长。早期 Video Diffusion Model 使用此架构。

#### 架构 2：帧级扩散 + 时序模块（主流）

```
每帧独立扩散 → [空间特征] → [时序 Transformer] → 时间一致性 → 输出视频
```

这是 Sora、Kling、Runway Gen-3 的共同思路。先在空间维度处理每帧，再用 Transformer 在时间维度建模帧间关系。

#### 架构 3：一致性模型

```
噪声 → [一致性映射] → 视频（1步或几步）
```

从预训练的扩散模型蒸馏出一步生成能力。2026 年的前沿——将视频生成从分钟级压缩到秒级。

### 2.3 Sora 的核心创新

Sora（OpenAI, 2024）的核心创新有三个：

**时空 patches（Space-Time Patches）：** 将视频切分为固定大小的时空块，每个 patch 包含若干帧的局部空间区域：

```
视频 (3s, 24fps, 1080p)
    ↓
切分为时空 patches (每 patch: 2帧 × 16×16 像素)
    ↓
展平为序列 [patch_1, patch_2, ..., patch_N]
    ↓
Transformer 处理这个序列
```

**DiT（Diffusion Transformer）：** 用 Transformer 替代 U-Net 作为去噪网络。每个 patch 同时关注空间邻居（同一帧的相邻 patch）和时间邻居（相邻帧的同一位置）。

**可变时长：** 因为 Transformer 处理的是序列，可以处理任意长度的 patch 序列——支持 5-60 秒的可变长度视频。

### 2.4 主要视频生成模型对比

| 模型 | 发布 | 分辨率 | 最长时长 | 架构 | 开源 |
|------|------|--------|---------|------|------|
| Sora | 2024.02 | 1080p | 60s | DiT + 时空 patches | ❌ |
| Kling | 2024.06 | 1080p | 2min | 类 Sora 架构 | ❌ |
| Runway Gen-3 | 2024.07 | 1080p | 10s | 帧级扩散 | ❌ |
| Open-Sora | 2024.04 | 720p | 16s | DiT（开源复现） | ✅ |
| CogVideoX | 2024.08 | 720p | 6s | Expert Transformer | ✅ |
| Wan-Video | 2025 | 720p | 5s | DiT（开源） | ✅ |

### 2.5 2026 年的突破

- **一致性模型（Consistency Model）：** 从扩散模型蒸馏出一步生成。训练时学习一个"一致性映射"——将噪声轨迹上的任意点直接映射到起点（无噪声的视频）。推理时只需 1 步前向传播。
- **Rectified Flow：** Flow Matching 的加速版本。将扩散过程重新参数化为一条直线（rectified），使得 2-5 步采样即可获得接近完整采样的质量。
- **Sora / Kling / Runway：** 商业级视频生成。1080p 分辨率，5-60 秒时长，可用于广告、电影预览、社交媒体内容。

---

## 3. 从零实现

### 第 1 步：视频数据预处理

```python
import torch
import torch.nn as nn
import numpy as np

def preprocess_video(frames, target_size=(256, 256), num_frames=16):
    """
    将视频帧序列预处理为模型输入。
    Args:
        frames: 原始帧列表 [frame_1, frame_2, ...]，每个 (H, W, 3)
        target_size: 目标尺寸 (H, W)
        num_frames: 采样的帧数
    Returns:
        video_tensor: (C, T, H, W) 归一化到 [-1, 1]
    """
    # 均匀采样帧
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]

    # 调整尺寸并归一化
    processed = []
    for frame in frames:
        frame = frame.resize(target_size)
        frame_np = np.array(frame).astype(np.float32) / 127.5 - 1.0
        processed.append(frame_np)

    # 堆叠为 (T, H, W, C) 然后转置为 (C, T, H, W)
    video = np.stack(processed, axis=0)
    return torch.from_numpy(video.transpose(3, 0, 1, 2))
```

### 第 2 步：时空 Patches 编码

```python
class SpaceTimePatchEmbedding(nn.Module):
    """将视频切分为时空 patches 并嵌入为向量——Sora 的核心组件。"""

    def __init__(self, in_channels=4, patch_size=2, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        # 3D 卷积：同时在时间和空间维度切分 patch
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: 视频潜向量 (B, C, T, H, W)
        Returns:
            patches: (B, num_patches, embed_dim)
        """
        # 3D 卷积切分 patches
        # (B, C, T, H, W) -> (B, embed_dim, T//ps, H//ps, W//ps)
        x = self.proj(x)
        B, D = x.shape[0], x.shape[1]
        # 展平为序列: (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
```

### 第 3 步：简化版视频扩散模型

```python
class SimpleVideoDiffusion(nn.Module):
    """简化版视频扩散模型——展示时空 patches + Transformer 的架构。"""

    def __init__(self, in_channels=4, embed_dim=768, num_heads=8, num_layers=6):
        super().__init__()
        # 时空 patches 嵌入
        self.patch_embed = SpaceTimePatchEmbedding(in_channels, patch_size=2, embed_dim=embed_dim)

        # 位置编码（空间 + 时间）
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)

        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影：还原为视频潜向量形状
        self.output_proj = nn.Linear(embed_dim, in_channels * 8)  # patch_size^3 = 8

    def forward(self, noisy_video, timestep_emb):
        """
        前向传播：预测噪声。
        Args:
            noisy_video: 带噪视频 (B, C, T, H, W)
            timestep_emb: 时间步嵌入 (B, embed_dim)
        Returns:
            predicted_noise: 预测的噪声 (B, C, T, H, W)
        """
        # 时空 patches 嵌入
        x = self.patch_embed(noisy_video)  # (B, num_patches, embed_dim)

        # 加入时间步信息
        x = x + timestep_emb.unsqueeze(1)

        # 加入位置编码
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer 处理
        x = self.transformer(x)

        # 输出投影
        x = self.output_proj(x)
        return x
```

---

## 4. 工具

### 4.1 Diffusers 中的视频生成

```python
from diffusers import DiffusionPipeline
import torch

# 加载 Stable Video Diffusion（图像到视频）
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# 从图像生成视频
from PIL import Image
image = Image.open("input.jpg").resize((1024, 576))
frames = pipe(image, num_frames=25).frames[0]

# 保存为 GIF
frames[0].save("output.gif", save_all=True, append_images=frames[1:],
               duration=83, loop=0)  # 12fps
```

### 4.2 开源视频生成工具

| 工具 | 说明 | 显存需求 | 推荐用途 |
|------|------|---------|---------|
| Stable Video Diffusion | 图像到视频 | 12GB+ | 短视频生成 |
| CogVideoX | 文本到视频 | 16GB+ | 中文提示词友好 |
| Open-Sora | 文本到视频 | 24GB+ | 研究和复现 |
| Wan-Video | 文本到视频 | 16GB+ | 2025 开源标杆 |
| AnimateDiff | 图像到视频 | 8GB+ | 轻量级动画 |

### 4.3 加速方案

```python
# 方案 1：减少生成帧数
# 从 25 帧减到 14 帧，速度提升约 40%
frames = pipe(image, num_frames=14).frames[0]

# 方案 2：使用 torch.compile
pipe.unet = torch.compile(pipe.unet)

# 方案 3：使用 FP8 量化（需 Hopper GPU）
pipe.enable_model_cpu_offload()
```

---

## 5. LLM 视角

### 5.1 在主流系统中的体现

- **Sora（OpenAI）**：用 DiT + 时空 patches 生成 1080p 视频。虽然未完全开源，但其架构思想影响了整个领域——Open-Sora、CogVideoX、Wan-Video 都采用了类似的时空 patches 设计。
- **Kling（快手）**：国内领先的视频生成模型。支持中文提示词、人物一致性控制、镜头运动控制。2025 年已集成进快影 App。
- **Runway Gen-3**：专业视频生成工具。支持从文本/图像生成视频，以及视频编辑（风格转换、物体替换）。

### 5.2 大语言模型时代什么变了？

视频生成正在从"生成短视频"变成"生成长视频故事"。2026 年的趋势是**多模态理解与生成的统一**：一个模型同时理解视频内容（用于问答、摘要）和生成视频内容（用于创作）。这类似于 GPT-4 同时理解和生成文本——未来的视频模型可能同时理解视频和生成视频。

### 5.3 什么没变？

扩散模型的核心——逐步去噪——在视频生成中仍然有效。变化的只是处理的维度从 2D 扩展到 3D。但那个简单的 MSE 损失函数（预测噪声 vs 真实噪声）仍然在运行。一致性模型和流匹配也没有改变这个核心——它们只是改变了去噪的路径（从曲线变成直线）。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当你让 ChatGPT "生成一段视频"时，它可能调用 Sora 或 DALL-E 3 的视频扩展功能。你输入的提示词会被 LLM 先"润色"（添加细节、指定镜头语言），然后送入视频扩散模型。理解时空 patches 有助于你写出更好的视频提示词——指定"镜头从左向右平移"比"一个动态的场景"更有效，因为前者对应了 Transformer 可以建模的明确时空模式。

---

## 6. 工程最佳实践

### 6.1 帧数与时长选择

| 帧数 | 时长（12fps） | 时长（24fps） | 显存需求 | 适用场景 |
|------|-------------|-------------|---------|---------|
| 14 帧 | ~1.2 秒 | ~0.6 秒 | 8-12 GB | 快速原型 |
| 25 帧 | ~2.1 秒 | ~1.0 秒 | 12-16 GB | 短视频 |
| 50 帧 | ~4.2 秒 | ~2.1 秒 | 16-24 GB | 中等时长 |
| 100 帧 | ~8.3 秒 | ~4.2 秒 | 24-32 GB | 较长视频 |

### 6.2 提示词策略

- **指定镜头语言**：推拉摇移、特写/远景、固定/运动镜头
- **描述运动**："一只猫从左向右走过画面"比"一只猫"更有效
- **指定风格**："电影质感、浅景深、自然光"提升质量
- **避免矛盾**：不要同时写"静止"和"快速移动"

### 6.3 中文场景特别建议

- CogVideoX 和 Kling 对中文提示词支持最好
- 国内可用的替代方案：可灵（Kling）、通义万相视频版、智谱清影

### 6.4 踩坑经验

- **帧间闪烁**：增加时序 Transformer 的层数，或使用更强的时序一致性损失
- **人物变形**：SD 系列对人物面部的时序一致性较弱，使用 OpenPose 姿态控制可缓解
- **显存溢出**：50 帧以上的视频生成需要 24GB+ 显存。使用 `pipe.enable_model_cpu_offload()` 可降低显存占用

---

## 7. 常见错误

### 错误 1：将图像扩散模型直接用于视频

**现象：** 生成的每帧图像质量很高，但帧间完全不连贯——像幻灯片而非视频。

**原因：** 图像扩散模型没有时间建模能力——每帧独立生成，没有帧间依赖。

**修复：**

```python
# ❌ 错误：用图像模型逐帧生成
for i in range(num_frames):
    frame = image_pipe(prompt).images[0]  # 每帧独立，无时间一致性

# ✓ 正确：使用视频扩散模型
from diffusers import DiffusionPipeline
video_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt")
frames = video_pipe(image, num_frames=25).frames[0]  # 一次生成整个视频
```

### 错误 2：提示词太短太模糊

**现象：** 生成的视频内容随机、不符合预期。

**原因：** 视频扩散模型需要更详细的提示词来描述运动和场景变化——比图像生成需要更多细节。

**修复：**

```python
# ❌ 错误：太模糊
prompt = "一只猫"

# ✓ 正确：描述运动和场景
prompt = "一只橘猫从左向右走过木地板，阳光从窗户照进来，浅景深，电影质感"
```

### 错误 3：忽略显存限制

**现象：** 生成过程中 OOM（显存溢出）。

**原因：** 视频的 3D 结构导致显存占用远高于图像。512×512×25 帧的视频潜向量占用显存约为单张图像的 25 倍。

**修复：**

```python
# ❌ 错误：直接生成高分辨率长视频
frames = pipe(image, num_frames=100, height=1024, width=1024)  # OOM!

# ✓ 正确：先低分辨率预览，再高分辨率精修
frames = pipe(image, num_frames=14, height=256, width=256)  # 快速预览
# 确认满意后再用高分辨率生成
```

---

## 8. 面试考点

### Q1：视频扩散模型如何保证帧间时间一致性？（难度：⭐⭐）

**参考答案：**
主要有三种方式：(1) 3D 卷积——在时间和空间维度同时卷积，天然捕捉帧间关系；(2) 时序 Transformer——在空间特征提取后，用 Transformer 在时间维度建模帧间依赖；(3) 光流引导——显式估计帧间光流，用光流对齐相邻帧的特征。Sora 采用了时空 patches + DiT 的方案——将视频切分为时空块后，Transformer 的自注意力同时在空间和时间维度运作，自动学习帧间关系。

### Q2：Sora 的时空 patches 和 ViT 的图像 patches 有什么区别？（难度：⭐⭐⭐）

**参考答案：**
ViT 的图像 patches 是 2D 的——将图像切分为固定大小的方块（如 16×16），展平为序列。Sora 的时空 patches 是 3D 的——将视频切分为时空块（如 2×16×16），每个 patch 包含 2 帧的 16×16 像素区域。区别在于：(1) 时空 patches 额外编码了时间信息——同一个 patch 内的像素来自不同帧；(2) Transformer 的注意力机制需要在时间和空间两个维度建模；(3) 位置编码需要同时编码空间位置和时间位置。

### Q3：一致性模型是如何从扩散模型蒸馏出一步生成能力的？（难度：⭐⭐⭐）

**参考答案：**
一致性模型的核心思想是学习一个"一致性映射"——将扩散过程中噪声轨迹上的任意点直接映射到起点（无噪声的数据点）。训练时：(1) 从预训练的扩散模型采样噪声轨迹；(2) 在轨迹上随机采样两个点 t1 和 t2；(3) 训练一致性模型将 t1 处的点映射到数据点，同时将 t2 处的点也映射到同一个数据点；(4) 使用"一致性损失"确保映射的一致性。推理时：只需一步前向传播，直接从噪声映射到数据。这将视频生成从 50 步压缩到 1 步，速度提升 50 倍。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 时空 patches | "把视频切成小块" | 将视频切分为固定大小的 3D 块（时间×空间×空间），每个块编码为一个向量，展平为序列供 Transformer 处理 |
| DiT | "用 Transformer 做扩散" | Diffusion Transformer——用 Transformer 替代 U-Net 作为去噪网络，在时空 patches 上运行自注意力 |
| 时间一致性 | "视频不能闪烁" | 连续帧之间物体位置、外观、光照必须连贯——这是视频生成区别于图像生成的核心约束 |
| 一致性模型 | "一步出视频" | 从扩散模型蒸馏出的一步生成模型——学习噪声轨迹的起点映射，推理时只需 1 步前向传播 |
| Rectified Flow | "直线去噪" | 将扩散过程的曲线轨迹重新参数化为直线，使得 2-5 步采样即可获得接近完整采样的质量 |
| 帧率 (FPS) | "每秒多少帧" | 每秒播放的帧数——12fps 适合动画，24fps 是电影标准，30fps 是视频标准 |

---

## 📚 小结

视频扩散 = 图像扩散 + 时间维度。三大架构：3D U-Net、帧级扩散+时序模块、一致性模型。Sora 用 DiT + 时空 patches 实现了 5-60 秒的高质量视频生成。2026 年的前沿是一致性模型和 Rectified Flow——从扩散骨干蒸馏出一步/几步生成，大幅降低推理成本。下一课我们将进入音频生成——将扩散模型从视觉扩展到听觉。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释为什么视频扩散不能简单地对每帧独立运行图像扩散。画一个对比图说明有/无时间建模的区别。

2. **【实现】** 修改 `SpaceTimePatchEmbedding` 类，支持可变帧数输入。测试输入为 (B, 4, 8, 64, 64) 和 (B, 4, 32, 64, 64) 时的输出形状变化。

3. **【实验】** 使用 `diffusers` 的 Stable Video Diffusion，分别用 14 帧和 25 帧生成同一输入图像的视频，对比质量差异和生成时间。

4. **【思考】** 如果要生成一段 60 秒的 1080p 视频（1440 帧），当前的技术瓶颈在哪里？从显存、计算量、时间一致性三个角度分析。

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| 视频预处理工具 | `code/video_utils.py` | 视频帧采样、时空 patches 编码 |
| 简化版视频扩散模型 | `code/video_diffusion.py` | 时空 patches + Transformer 的简化实现 |
| 视频提示词指南 | `outputs/video-prompt-guide.md` | 面向中文用户的视频生成提示词模板 |

---

## 📖 参考资料

1. [论文] Brooks et al. "Video Generation Models as World Simulators". OpenAI, 2024. https://openai.com/research/video-generation-models-as-world-simulators
2. [论文] Blattmann et al. "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets". arXiv, 2023. https://arxiv.org/abs/2311.15127
3. [论文] Song et al. "Consistency Models". ICML, 2023. https://arxiv.org/abs/2303.01469
4. [GitHub] Hugging Face diffusers SVD: https://github.com/huggingface/diffusers
5. [GitHub] PKU-YuanGroup/Open-Sora: https://github.com/PKU-YuanGroup/Open-Sora

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
