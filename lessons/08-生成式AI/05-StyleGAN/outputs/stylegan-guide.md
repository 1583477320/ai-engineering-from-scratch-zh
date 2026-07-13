# StyleGAN 实践参考指南

> 快速查阅 StyleGAN 系列模型的核心概念、架构选型和工程要点。

---

## 1. 核心概念速查

### 映射网络（Mapping Network）

```
标准 GAN:   z ──────────────────→ [生成器] → 图像
StyleGAN:   z → [映射网络 8层] → w → [生成器 + AdaIN] → 图像
```

**作用：** 将噪声向量 z 映射到中间空间 w，解纠缠不同语义因素。

- z 空间：不同因素纠缠在一起，改变一个会影响多个
- w 空间：因素更独立，可以单独控制年龄、性别、表情等

### AdaIN（自适应实例归一化）

```
输入特征 x → [实例归一化] → [用 w 生成 scale 和 bias] → 调制输出
```

**公式：**

$$\text{AdaIN}(x, y) = y_{s,i} \cdot \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$

- $y_s$：缩放因子（控制通道强度/对比度）
- $y_b$：偏移因子（控制通道偏移/亮度）
- $\mu(x_i)$、$\sigma(x_i)$：第 i 个通道的均值和标准差

### 样式混合（Style Mixing）

不同层使用不同的 w 向量，混合不同"脸"的属性：

| 层级 | 分辨率 | 控制的属性 |
|---|---|---|
| 粗粒度（层 0-1） | 4x4 ~ 8x8 | 姿势、脸型、眼镜 |
| 中粒度（层 2-3） | 16x16 ~ 32x32 | 五官位置、发型 |
| 细粒度（层 4+） | 64x64+ | 纹理、颜色、光照 |

---

## 2. 模型选型

| 模型 | 发布年份 | 适用场景 | 特点 |
|---|---|---|---|
| StyleGAN | 2019 | 学习理解 | 原始论文，架构清晰 |
| StyleGAN2 | 2020 | 人脸生成 | 去除水滴伪影，改进正则化 |
| StyleGAN3 | 2021 | 人脸生成（推荐） | 消除别名，平移/旋转等变 |
| StyleGAN-XL | 2022 | 非人脸域 | 支持 ImageNet 级别的多样性 |

**2026 年推荐：** 人脸生成首选 StyleGAN3；非人脸域考虑 StyleGAN-XL 或扩散模型。

---

## 3. 快速上手

### 3.1 使用 NVIDIA 官方预训练模型

```python
# 依赖：torch, ninja, Pillow
# 安装：pip install torch ninja Pillow

# 加载预训练的 StyleGAN3 生成器
import torch

# 从 NVIDIA 下载预训练权重
# https://github.com/NVlabs/stylegan3

generator_url = (
    "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/"
    "pretrained/ffhq.pkl"
)

# 注意：实际使用需要安装 stylegan3 包并按其 API 调用
# 这里展示概念流程
print("加载预训练 StyleGAN3 模型...")
print("输入：512 维随机噪声向量 z")
print("输出：1024x1024 人脸图像")
```

### 3.2 使用 HuggingFace Diffusers

```python
# 扩散模型生态中也有 StyleGAN 的集成
# 但 StyleGAN 本身不通过 diffusers 使用
# 推荐直接使用 NVIDIA 的官方实现

# HuggingFace 上可以找到社区微调的 StyleGAN 模型
# 搜索 "stylegan" 可以找到人脸编辑、属性操控等应用
```

### 3.3 样式混合的代码模式

```python
# 伪代码：展示样式混合的核心逻辑
def style_mixing(w_source, w_target, generator):
    """
    w_source: 控制粗粒度层的 w（姿势、脸型）
    w_target: 控制细粒度层的 w（纹理、颜色）
    """
    outputs = []
    for layer_idx, (conv, adain) in enumerate(generator.layers):
        # 前 4 层用 source 的 w，后面用 target 的 w
        if layer_idx < 4:
            w = w_source
        else:
            w = w_target
        feat = conv(feat)
        feat = adain(feat, w)
        outputs.append(feat)
    return generator.to_image(outputs[-1])
```

---

## 4. 常见应用场景

### 4.1 人脸生成与编辑

- **人脸生成：** 随机采样 z → 映射到 w → 生成高质量人脸
- **属性编辑：** 在 w 空间中沿特定方向移动 → 改变年龄/性别/表情
- **人脸混合：** 两个 w 的插值 → 平滑过渡

### 4.2 属性操控

```python
# 概念：在 w 空间中找到"年龄方向"
# w_old = mapping(z_old)      # 年轻的脸
# w_new = mapping(z_new)      # 年老的脸
# direction = w_new - w_old   # "年龄方向"
# w_edit = w_source + alpha * direction  # 调整年龄
```

### 4.3 数据增强

- 用 StyleGAN 生成训练数据，增强小样本数据集
- 特别适用于人脸、医疗影像等数据稀缺领域

---

## 5. 工程要点

### 5.1 训练技巧

| 技巧 | 说明 |
|---|---|
| 使用 ADA（自适应判别器增强） | 数据量不足时防止过拟合 |
| 渐进式训练 | 从低分辨率逐步增加到高分辨率 |
| R1 正则化 | 对判别器的梯度做正则化，稳定训练 |
| 路径长度正则化 | 鼓励 w 空间的均匀分布，改善样式混合 |

### 5.2 显存需求

| 分辨率 | 显存（推理） | 显存（训练） |
|---|---|---|
| 256x256 | ~2 GB | ~8 GB |
| 512x512 | ~4 GB | ~12 GB |
| 1024x1024 | ~6 GB | ~16 GB |

### 5.3 推理加速

- **TensorRT：** 将 StyleGAN 转换为 TensorRT 引擎，推理速度提升 3-5x
- **量化：** INT8 量化可在保持质量的前提下减少 50% 显存
- **批量推理：** GPU 利用率随 batch size 增大而提高

---

## 6. 与扩散模型的对比

| 特性 | StyleGAN3 | 扩散模型（SDXL） |
|---|---|---|
| 生成速度 | 极快（1 次前向传播） | 慢（20-50 步迭代） |
| 图像质量 | 窄域顶级 | 广域顶级 |
| 可控性 | w 空间插值、属性操控 | 提示词、ControlNet |
| 多样性 | 受限于训练域 | 极高 |
| 训练稳定性 | 需要技巧 | 相对稳定 |
| 适用场景 | 人脸、固定域 | 通用图像生成 |

**结论：** 如果任务是人脸或固定域生成，StyleGAN3 仍是最佳选择。如果需要通用图像生成，扩散模型更合适。

---

## 7. 参考资源

- [NVIDIA StyleGAN3 官方仓库](https://github.com/NVlabs/stylegan3)
- [NVIDIA StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [StyleGAN 论文](https://arxiv.org/abs/1812.04948)
- [StyleGAN2 论文](https://arxiv.org/abs/1912.04958)
- [StyleGAN3 论文](https://arxiv.org/abs/2104.12476)
- [InterFaceGAN](https://github.com/genforce/interfacegan) — 人脸属性编辑工具
