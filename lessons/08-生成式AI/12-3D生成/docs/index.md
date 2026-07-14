# 3D 生成

> 从文本或图像生成三维模型、场景、纹理——3D 生成是视频生成的下一个前沿。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 08 · 07（潜在扩散）、10（视频生成）| **时间：** ~45 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 08 · 06（DDPM）— 扩散模型在 3D 生成中的应用 | 阶段 08 · 10（视频生成）— 视频的时空建模与 3D 的体素建模相通

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分 3D 生成的三种表示——点云、体素、网格（Mesh）
- [ ] 解释 3D Gaussian Splatting——为什么它在 2024-2026 年成为 3D 重建的首选
- [ ] 说明 2026 年文本到 3D 的两条路线——基于扩散和基于原生 3D
- [ ] 使用 Shap-E 或 TripoSR 从文本/图像生成 3D 模型
- [ ] 区分重建（从图像→3D）和生成（从文本/噪声→3D）的技术差异

---

## 1. 问题

图像是 2D 像素数组，视频是 3D 数组（空间 + 时间）。3D 场景是 4D 数据——空间三维（H×W×D）+ 颜色/材质。维度的增加意味着：

- **计算量**：256³ 体素网格 ≈ 1600 万个点，而 512² 像素 ≈ 26 万个点——60 倍
- **表示多样性**：点云、体素、网格、隐式函数（NeRF）、高斯椭球——每种表示都有独特的数学特性和渲染方式
- **数据稀缺**：3D 模型的标注数据远少于 2D 图像——2025 年最大 3D 数据集 Objaverse 只有约 10 万条，而图像数据集 LAION-5B 有 50 亿张

但在游戏、影视、建筑、医疗、机器人等领域的 3D 内容需求远超过 2D。2024-2026 年，扩散模型和 Transformer 从 2D 扩展到 3D——文本/图像到 3D 的技术正在快速成熟。

---

## 2. 概念

### 2.1 三种 3D 表示

| 表示 | 形式 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|---------|
| **点云 (Point Cloud)** | (x, y, z) 点集 | 简单直接，易于采集 | 缺少表面/纹理信息 | LiDAR、3D 扫描 |
| **体素 (Voxel)** | 3D 像素网格 256³ | 完全对称，易于扩散 | 内存巨大（256³×4bytes≈512MB） | 扩散模型训练 |
| **网格 (Mesh)** | 顶点+面+UV 纹理 | 游戏/影视标准 | 拓扑复杂，生成难 | 3D 建模、渲染 |
| **NeRF（隐式）** | MLP 权重 | 高质量渲染 | 渲染慢 | 场景重建 |
| **Gaussian Splatting** | 3D 高斯椭球集合 | 实时渲染，高质量 | 训练略慢 | 实时光栅化 |

### 2.2 3D Gaussian Splatting（2023）

Kerbl 等人的 3D Gaussian Splatting（3DGS）在 2024-2025 年彻底改变了 3D 重建领域：

```
输入：多视角 2D 图像（几十到几百张）
   ↓
初始化 3D 高斯椭球体（位置、不透明度、协方差、颜色各向异性）
   ↓
优化：可微分光栅化（tile-based rasterization）→ 逐像素误差
   ↓
输出：数百万个高斯椭球 → 任意视角实时渲染（>30fps）
```

**为什么 3DGS 优于 NeRF？** NeRF 需要逐像素光线步进（~10-100ms/像素），3DGS 使用可微分光栅化（<1ms/帧）。对于 1080p 分辨率，3DGS 比 NeRF 快约 1000 倍。

### 2.3 重建 vs 生成

| | 重建（Reconstruction） | 生成（Generation） |
|------|---------------------|------------------|
| 输入 | 多视角 2D 图像 | 文本/单张图像/噪声 |
| 输出 | 同一个场景的 3D | 任意新 3D 模型 |
| 代表 | 3DGS, NeRF | Shap-E, Point-E, TripoSR |
| 典型时间 | 5-30 分钟训练 | 1-30 秒推理 |
| 应用 | 摄影测量、VR/AR | 3D 内容创作 |

### 2.4 2026 文本/图像到 3D 的路线

#### 路线 1：基于扩散

Shap-E（OpenAI, 2023）：在 3D 数据的潜空间上运行扩散模型。

```
文本/图像 → [CLIP 编码器] → 条件 → [扩散生成] → 3D 潜向量 → [隐式解码器] → 网格/纹理
```

Point-E（OpenAI, 2023）：先生成点云，再由第二个模型从点云转换为网格。两步管道降低训练难度，但质量有限。

#### 路线 2：基于原生 3D

TripoSR（Stability AI, 2024）：一步从单张图像生成 3D 网格。基于 LRM（Large Reconstruction Model）——在 Objaverse 上训练的大型 Transformer，将图像直接映射为 3D 表示。

```
图像 → [Transformer 编码器] → [3D 解码器] → 网格 + 纹理（<1秒）
```

Meshy：商业级文本到 3D 网格平台。支持 PBR 材质、四边形网格、多种风格。

### 2.5 2026 年的主要 3D 生成模型

| 模型 | 发布 | 输入 | 输出 | 速度 | 开源 |
|------|------|------|------|------|------|
| Point-E | 2023 | 文本 | 点云 | ~10s | ✅ |
| Shap-E | 2023 | 文本/图像 | 隐式 3D | ~10s | ✅ |
| TripoSR | 2024 | 图像 | 网格 | <1s | ✅ |
| Zero-1-to-3 | 2023 | 图像 | 多视角 | ~5s | ✅ |
| LGM | 2024 | 多视角 | Gaussian | ~7s | ✅ |
| Meshy v4 | 2024 | 文本/图像 | 网格+材质 | ~30s | ❌ |
| 3DTopia | 2024 | 文本 | 网格 | ~20s | ✅ |

---

## 3. 从零实现

3D 生成的核心概念——3D Gaussian 的可微分渲染——可以用简化代码理解：

### 第 1 步：3D Gaussian 椭球体

```python
import torch
import torch.nn as nn
import numpy as np

class Gaussian3D:
    """3D 高斯椭球体的数据结构。"""

    def __init__(self, position=(0, 0, 0), scale=(0.1, 0.1, 0.1), 
                 rotation=(1, 0, 0, 0), opacity=0.5, color=(1.0, 0.5, 0.2)):
        """
        Args:
            position: 3D 位置 (x, y, z)
            scale: 各向异性缩放 (sx, sy, sz)
            rotation: 四元数 (w, x, y, z)
            opacity: 不透明度 [0, 1]
            color: RGB 颜色
        """
        self.position = torch.tensor(position, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)
        self.rotation = torch.tensor(rotation, dtype=torch.float32)
        self.opacity = torch.tensor(opacity, dtype=torch.float32)
        self.color = torch.tensor(color, dtype=torch.float32)

    def compute_covariance_matrix(self):
        """计算 3D 协方差矩阵——决定高斯椭球的形状和方向。"""
        # 将缩放转换为对角矩阵
        S = torch.diag(self.scale)
        # 四元数转旋转矩阵（简化版）
        R = quaternion_to_rotation_matrix(self.rotation)
        # 协方差 = R·S·S^T·R^T
        return R @ S @ S.T @ R.T


def quaternion_to_rotation_matrix(quat):
    """四元数转旋转矩阵。"""
    w, x, y, z = quat
    return torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


class Gaussian3DScene(nn.Module):
    """3D 高斯场景——一组高斯椭球的集合。"""

    def __init__(self, num_gaussians=10000):
        super().__init__()
        # 所有高斯参数都是可训练的
        self.positions = nn.Parameter(torch.randn(num_gaussians, 3) * 0.5)
        self.scales = nn.Parameter(torch.randn(num_gaussians, 3))
        self.rotations = nn.Parameter(torch.randn(num_gaussians, 4))
        self.opacities = nn.Parameter(torch.sigmoid(torch.randn(num_gaussians)))
        self.colors = nn.Parameter(torch.rand(num_gaussians, 3))
```

### 第 2 步：体素扩散概念

```python
class SimpleVoxelDiffusion(nn.Module):
    """简化版体素扩散模型——将 DDPM 扩展到 3D 体素空间。"""

    def __init__(self, in_channels=4, voxel_size=32):
        super().__init__()
        # 使用 3D 卷积替代 2D 卷积
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, strain=2),  # 降采样
            nn.ReLU(),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.mid = nn.Conv3d(256, 256, 3, padding=1)
        self.up = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv3d(128, in_channels, 3, padding=1),
        )

    def forward(self, voxels, t):
        """
        预测 3D 噪声。
        Args:
            voxels: 带噪体素 (B, C, D, H, W)
            t: 时间步
        Returns:
            noise_pred: 预测的噪声 (B, C, D, H, W)
        """
        h = self.down(voxels)
        h = self.mid(h)
        return self.up(h)
```

### 第 3 步：Shap-E 的条件注入

```python
class SimpleShapEConditioning(nn.Module):
    """简化版 Shap-E 条件注入——文本/图像条件 + 3D 潜空间生成。"""

    def __init__(self, latent_dim=512, text_dim=768, num_layers=4):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, latent_dim)

        # Transformer 在 3D 潜空间上做扩散
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出头：预测 3D 表示参数
        self.output_proj = nn.Linear(latent_dim, 256)  # 简化：256 维 3D 表示

    def forward(self, text_encoding, step_encoding):
        """
        Args:
            text_encoding: 文本嵌入 (B, text_dim)
            step_encoding: 时间步嵌入 (B, latent_dim)
        Returns:
            latent: 生成的 3D 潜向量
        """
        # 条件投影
        cond = self.text_proj(text_encoding) + step_encoding
        cond = cond.unsqueeze(1)  # (B, 1, latent_dim)

        # Transformer 处理
        latent = self.transformer(cond)
        output = self.output_proj(latent.squeeze(1))
        return output
```

---

## 4. 工具

### 4.1 Diffusers 中的 3D 生成

```python
# Point-E：文本到点云
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DPMSolverMultistepScheduler
import torch

# Shap-E：文本/图像到 3D
from transformers import pipeline

pipe = pipeline("text-to-3d", model="shap-e")
images = pipe("一只蓝色的小鸟")

# 或者使用 Diffusers
pipe = ShapEPipeline.from_pretrained("openai/shap-e")
latents = pipe("一只坐在树枝上的猫")
# latents → 隐式解码器 → 网格
```

### 4.2 TripoSR——一步图像到 3D

```python
# 使用 TripoSR（依赖 PyTorch3D）
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
import numpy as np
from PIL import Image

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

image = Image.open("input.jpg")
image = remove_background(image)
image = resize_foreground(image, 1.0)

with torch.no_grad():
    scene_codes = model([image])
    mesh = model.extract_mesh(scene_codes, resolution=256)
```

### 4.3 3D 渲染和后端

| 工具 | 类型 | 说明 |
|------|------|------|
| PyTorch3D | 渲染器 | Meta 的 3D 深度学习库 |
| Three.js | Web 渲染 | 浏览器端展示 3D 模型 |
| Blender | 3D 编辑 | 开源 3D 内容制作 |
| Open3D | 点云处理 | 开源 3D 数据处理库 |
| Meshlab | 网格处理 | 网格修复、简化、转换 |
| Unity/Unreal | 游戏引擎 | 3D 实时渲染和交互 |

---

## 5. LLM 视角

### 5.1 在主流系统中的体现

- **Shap-E / Point-E（OpenAI）**：将扩散模型从 2D 扩展到 3D——证明了同一个数学框架（噪声→数据）可以应用于任意维度的数据。Shap-E 的潜空间编码了一个隐式 3D 函数，可以用文本条件控制生成。
- **Meshy / Tripo（商业）**：3D 内容创作平台，输入文本或图像 → 输出可编辑的 3D 网格。2025-2026 年广泛应用于游戏资产生成、电商商品展示、VR/AR 内容制作。
- **Luma AI / Koloro**：基于 NeRF/3DGS 的 3D 重建应用，手机拍摄即可得到高精度 3D 模型。

### 5.2 大语言模型时代什么变了？

3D 生成的"ChatGPT 时刻"还没有到来，但趋势已经很明确：2023 年你还需要专业软件和数小时做 3D 建模；2026 年，输入一段文字描述就能在 30 秒内得到一个可用的 3D 模型。关键转变是：**从"学习 3D 形状"变成了"学习 2D→3D 的映射"**。前者需要大量 3D 数据（稀缺），后者可以利用 2D 图像扩散模型的先验知识。

### 5.3 什么没变？

3D 表示的多样性仍然是核心挑战——没有一个表示方法在所有场景中都最优。点云简单但不带表面，体素对称但浪费计算，网格高质量但难生成，NeRF 漂亮但渲染慢，3DGS 快但内存消耗大。这个"表示挑战"自 2010 年代以来一直没有被根本解决。

### 5.4 使用 ChatGPT / Claude 时的直接体验

当 ChatGPT 生成 DALL-E 图像时，它是 2D 的。如果需要 3D（例如产品展示），目前多数 LLM 还不能直接输出 3D 模型。但你可以让 ChatGPT **生成 2D 的多视角图像"草图"**（正面、侧面、背面）——然后将这些图像输入 TripoSR 或 Meshy 得到 3D 模型。这是一个 2D→3D 的管道，在 2026 年是最实用的方案。

---

## 6. 工程最佳实践

### 6.1 3D 生成场景选型

| 需求 | 推荐工具 | 原因 |
|------|---------|------|
| 文本到 3D | Meshy / Shap-E | 文本条件化质量好 |
| 图像到 3D | TripoSR / LGM | 一步输出，速度快 |
| 场景重建 | 3D Gaussian Splatting | 实时渲染，质量最高 |
| 多视角生成 | Zero-1-to-3 | 可控视角生成 |

### 6.2 提示词策略

- **文本到 3D**: 描述形状 + 材质 + 颜色 + 风格
  - "一把带有木纹扶手的蓝色扶手椅，布艺坐垫，现代风格"
- **图像到 3D**: 用干净、无背景的正面图像
  - 使用背景移除工具预处理

### 6.3 中文场景特别建议

- Meshy 和 Tripo 支持中文提示词
- 国内 3D 生成平台：影眸科技（Rodin）、魔珐科技、Zego 3D
- 3DGS 训练需要 NVIDIA GPU（推荐 12GB+ 显存）

### 6.4 踩坑经验

- **网格质量差**：TripoSR 输出的网格需要 Blender 后处理（减面、平滑、拓扑优化）
- **纹理缺失/模糊**：3DGS 的纹理是从输入图像提取的，确保输入图像有足够的分辨率和覆盖角度
- **显存不足**：256³ 体素网格 ≈ 512MB 显存。使用更小的分辨率（128³）或渐进式训练

---

## 7. 常见错误

### 错误 1：混淆 3D 重建和 3D 生成

**现象：** 认为 TripoSR（一步图像到 3D）可以像 NeRF 一样精确重建场景。用单张照片 TripoSR 做室内场景重建，得到的结果完全不对。

**原因：** TripoSR 是**生成**模型——它根据训练数据（Objaverse 模型）猜测"单张图像背后的 3D 样子"。它不是**重建**——它不保证几何精确性。重建需要多视角输入。

**修复：**

```python
# ❌ 错误：用生成做重建
mesh = triposr_pipeline(single_image)  # 只有正面信息，背面是"猜的"

# ✓ 正确：区分使用场景
# 重建 → 多个视角的图像 → 3DGS 或 NeRF
# 生成 → 文本/单张图像 → TripoSR 或 Shap-E（背面不一定精确）
```

### 错误 2：忽略 3D 坐标系的差异

**现象：** 从 Shap-E 导出的模型在 Three.js 中渲染时是倒立的。

**原因：** 不同框架使用不同的坐标系（Y-up vs Z-up、右手系 vs 左手系）。

**修复：**

```python
# ❌ 错误：不转换坐标系
mesh.vertices = shap_e_output.vertices

# ✓ 正确：转换坐标系（Shap-E 是 Z-up，Three.js 是 Y-up）
import numpy as np
mesh.vertices = shap_e_output.vertices @ np.array([
    [1, 0, 0],
    [0, 0, 1],  # Z → Y
    [0, 1, 0],  # Y → Z
])
```

### 错误 3：在低端 GPU 上训练 3DGS

**现象：** 训练过程中 OOM 或速度极慢。

**原因：** 3DGS 需要在 GPU 上维护数百万个高斯椭球的可微计算图。

**修复：**

```python
# ❌ 8GB 显存 GPU 训练完整 3DGS
# OOM!

# ✓ 使用 smaller scene 或 lower resolution
# 降低输入图像分辨率（1K → 512）
# 减少迭代次数（30000 → 10000）
# 使用分块训练（tile-based training）
```

---

## 8. 面试考点

### Q1：为什么 3D Gaussian Splatting 比 NeRF 快这么多？（难度：⭐⭐）

**参考答案：**
NeRF 使用隐式表示（MLP 权重），渲染时需要逐像素光线步进——每个像素沿光线采样约 64-128 个点，每点做一次 MLP 前向，对于 1080p 分辨率约 200 万像素 × 128 点 × 1 MLP = 2.56 亿次 MLP 调用。3DGS 使用显式表示（高斯椭球集合），渲染时使用 tile-based 可微分光栅化——先将高斯椭球投影到 2D，然后对每个 tile 排序，只渲染该 tile 覆盖的高斯。这使得 3DGS 可以在 <1ms 内完成一帧渲染，而 NeRF 需要 10-100ms。

### Q2：文本到 3D 生成的主要技术瓶颈是什么？（难度：⭐⭐⭐）

**参考答案：**
(1) 3D 数据稀缺——最大的开源 3D 数据集 Objaverse 约 10 万条，而图像数据集有 50 亿张。这使得直接训练大型 3D 生成模型困难。(2) 表示复杂——3D 的表示方式（点云、体素、网格）没有统一的"最佳表示"。(3) 评测困难——2D 图像有 FID 和 CLIP Score，3D 模型目前主要依赖人类评估和 2D 渲染投影的 FID。解决思路包括：利用 2D 扩散模型的先验知识（SDS 损失——Score Distillation Sampling）来训练 3D 模型，以及使用更大规模合成 3D 数据集。

### Q3：什么是 SDS（Score Distillation Sampling）？它在 3D 生成中有什么用？（难度：⭐⭐⭐）

**参考答案：**
SDS（Score Distillation Sampling）是 DreamFusion（Google, 2022）提出的技术，用于解决 3D 数据稀缺的问题。核心思路：用一个预训练的 2D 扩散模型作为"教师"，来指导 3D 表示的优化。具体做法：(1) 从任意视角渲染 3D 表示的 2D 投影；(2) 用 2D 扩散模型评估这个投影的质量（与提示词的对齐度）；(3) 将扩散模型的误差信号反向传播到 3D 表示的参数上。这样，即使没有 3D 训练数据，也可以利用 2D 模型的知识生成 3D 内容。Zero-1-to-3、DreamFusion、Magic3D 都基于这个技术。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 3D Gaussian Splatting | "3D 高斯喷溅" | 将 3D 场景表示为一组带颜色的高斯椭球体，通过可微分光栅化渲染——比 NeRF 快 1000 倍 |
| 点云 (Point Cloud) | "一堆3D点" | 三维坐标点的集合，每个点包含 (x, y, z) 坐标和可能的颜色信息——最原始的 3D 表示 |
| 体素 (Voxel) | "3D 像素" | 三维空间中的正立方体网格单元——类似 2D 图像中的像素，但扩展到深度维度 |
| NeRF | "神经辐射场" | Neural Radiance Fields——用 MLP 表示 3D 场景，从多视角图像重建任意视角的新视图 |
| Shap-E | "Shape 的 Embedding" | OpenAI 的文本/图像到 3D 模型——在 3D 潜空间上做扩散生成，再解码为网格 |
| SDS (Score Distillation) | "用 2D 模型教 3D" | 使用预训练 2D 扩散模型指导 3D 表示优化的技术——解决 3D 数据稀缺问题 |

---

## 📚 小结

3D 生成 = 从图像/文本生成三维表示。3D Gaussian Splatting 是 2023-2026 的重建 SOTA——比 NeRF 快 1000 倍。文本/图像到 3D 有两条路线：基于扩散（Shap-E）和基于原生 3D（TripoSR）。SDS 技术利用 2D 扩散模型指导 3D 生成，解决了 3D 数据稀缺问题。2026 年的前沿：一步生成（TripoSR）、带 PBR 材质的网格（Meshy）、场景级 3D 生成。

---

## ✏️ 练习

1. **【理解】** 画一张对比 NeRF 和 3D Gaussian Splatting 的流程图，标注两者的核心差异（表示方式、渲染速度、训练时间）。

2. **【实现】** 修改 `Gaussian3D` 类，加入批量计算支持。定义一个 `Gaussian3DScene` 类，管理多个高斯椭球的位置、缩放、旋转、不透明度和颜色。

3. **【实验】** 使用 Meshy 或 TripoSR，对同一张输入图像生成 3D 网格，分别尝试不同参数（网格面数、纹理分辨率），对比输出质量。

4. **【思考】** 3D 生成的"ChatGPT 时刻"还没到来。思考要实现一个通用的 3D 生成模型（像 GPT-4 理解文本一样理解 3D 结构），需要克服哪些关键障碍？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| 3D Gaussian 实现 | `code/gaussian_3d.py` | 3D 高斯椭球体表示和协方差矩阵计算 |
| 体素扩散模型 | `code/voxel_diffusion.py` | 简化版 3D 体素扩散模型 |
| 3D 提示词模板 | `outputs/3d-prompt-guide.md` | 文本到 3D 的提示词编写指南 |

---

## 📖 参考资料

1. [论文] Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering". ACM TOG, 2023. https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
2. [论文] Jun and Nichol. "Shap-E: Generating Conditional 3D Implicit Functions". OpenAI, 2023. https://arxiv.org/abs/2305.02463
3. [论文] Tochilkin et al. "TripoSR: Fast 3D Object Reconstruction from a Single Image". Stability AI, 2024. https://arxiv.org/abs/2403.02151
4. [论文] Poole et al. "DreamFusion: Text-to-3D using 2D Diffusion". ICLR, 2023. https://arxiv.org/abs/2209.14988
5. [GitHub] graphicconception/gaussian-splatting: https://github.com/graphicconception/gaussian-splatting

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
