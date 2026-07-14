# 评估指标——FID 与 CLIP Score

> 生成模型的评估不能只靠人类看——FID 衡量"生成分布与真实分布的距离"，CLIP Score 衡量"生成图像与文本描述的对齐度"。两个指标共同定义了生成质量的量化基准。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 08 · 01（生成模型分类）| **时间：** ~45 分钟
**所处阶段：** Tier 2
**关联课程：** 阶段 08 · 01（生成模型分类）— 理解评估场景 | 阶段 08 · 06（DDPM）— 评估扩散模型生成质量

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现 FID（Fréchet Inception Distance）——理解均值向量和协方差矩阵的作用
- [ ] 解释 CLIP Score 如何评估文本-图像对齐——对比学习的推理时应用
- [ ] 说明 2026 年生成评估的最佳实践——FID + CLIP Score + LPIPS + 人类评估
- [ ] 计算 FID 时区分"分布距离"和"样本质量"的区别
- [ ] 分析 FID 和 CLIP Score 在不同生成任务中的适用性和局限性

---

## 1. 问题

你训练了一个扩散模型，生成了一些图像。它们看起来不错——但有多好？如何**客观地、可复现地**衡量生成质量？

"看起来不错"是不够的——因为(1) 不同人的判断标准不同；(2) 你需要在论文/报告中量化对比；(3) 你需要知道模型在哪个方面需要改进。

工业界形成了两个最常用的客观指标：
- **FID**：衡量生成图像的**整体质量**——锐利度、自然度、多样性
- **CLIP Score**：衡量生成图像与提示词的**语义对齐度**

两者互补——一个好的生成模型需要在两个指标上都表现优秀。

---

## 2. 概念

### 2.1 FID——Fréchet Inception Distance

FID 的核心思想：**用 Inception-v3 模型提取图像特征，然后在特征空间中比较真实图像和生成图像的分布。**

```
真实图像 → Inception-v3 提取特征 → 均值和协方差
                                       ↓
生成图像 → Inception-v3 提取特征 → 均值和协方差
                                       ↓
                              计算两个分布的距离
```

公式：

$$\text{FID} = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

其中：
- $\mu_r, \Sigma_r$：真实图像在 Inception-v3 特征空间的**均值向量**和**协方差矩阵**
- $\mu_g, \Sigma_g$：生成图像的均值和协方差
- $\text{Tr}$：矩阵的迹（对角线元素之和）

**FID 越低越好**——FID=0 表示两个分布完全相同。通常：
- FID < 10：生成质量优秀（如 Sora、DALL-E 3）
- FID 10-30：质量良好（如早期扩散模型）
- FID > 50：质量较差（如简单 GAN）

### 2.2 为什么用 Inception-v3？

Inception-v3 是在 ImageNet 上预训练的图像分类模型。用它提取特征有三个原因：
1. **语义特征**：Inception-v3 的倒数第二层（≈2048 维向量）编码了丰富的语义信息——不仅仅是像素统计
2. **标准化的基准**：所有论文都使用 Inception-v3，使得 FID 值可跨论文比较
3. **对自然图像有效**：Inception-v3 在 ImageNet 上训练，对自然图像的分布有很好的建模

**重要警告：** FID 只在 ImageNet 类自然图像上有效。对于非自然图像（医学图像、卫星图像、抽象艺术），Inception-v3 的特征可能不准确。

### 2.3 FID 的局限性

| 局限性 | 说明 | 缓解方法 |
|--------|------|---------|
| 小样本偏差 | 样本数 < 1000 时，FID 估计不准（协方差矩阵不稳定） | 使用 ≥ 10000 样本 |
| 不检测模式坍塌 | 一个模型只生成一张高质量图像也能获得好 FID | 配合多样性指标 |
| 不衡量语义对齐 | FID 不关心图像是否符合提示词 | 配合 CLIP Score |
| 对模糊不敏感 | FID 测量分布距离，不直接测量清晰度 | 配合 LPIPS |
| 数据漂移 | 在非 Inception 域上失效 | 使用域适配的特征提取器 |

### 2.4 CLIP Score

CLIP Score 衡量文本提示词与生成图像的语义对齐度：

$$\text{CLIP Score} = \max(100 \times \cos(E_{\text{image}}, E_{\text{text}}), 0)$$

其中 $E_{\text{image}}$ 和 $E_{\text{text}}$ 是 CLIP 模型的图像编码器和文本编码器输出的嵌入向量。

**CLIP Score 越高越好**（通常在 20-40 之间）：

| CLIP Score | 意义 |
|-----------|------|
| < 25 | 文本-图像对齐差（生成内容与提示词不匹配） |
| 25-30 | 基本对齐（主体正确但细节偏差） |
| 30-35 | 良好对齐（主体和大多数细节正确） |
| > 35 | 优秀对齐（生成图像高度符合提示词描述） |

### 2.5 LPIPS——感知相似度

LPIPS（Learned Perceptual Image Patch Similarity，学习型感知图像块相似度）衡量两张图像的**感知相似度**——基于 VGG 网络的特征差异：

$$\text{LPIPS}(x, x') = \sum_{l} \frac{1}{H_l W_l} \sum_{h,w} ||\hat{y}_{l,h,w} - \hat{y'}_{l,h,w}||^2$$

越低越好。LPIPS 更符合人类对"这两张图看起来像不像"的判断。

### 2.6 2026 年的四维评估组合

| 维度 | 指标 | 衡量内容 | 越高/越低 | 数据需求 |
|------|------|---------|----------|---------|
| **质量** | FID | 生成分布 vs 真实分布的距离 | 越低越好 | 真实图像 → |
| **语义** | CLIP Score | 文本-图像语义对齐度 | 越高越好 | 提示词 + 图像 |
| **感知** | LPIPS | 生成 vs 参考的感知相似度 | 越低越好 | 参考图像配对 |
| **人类** | 人类偏好评分 | 用户主观评价 | 越高越好 | 标注员 |

---

## 3. 从零实现

### 第 1 步：FID 计算

```python
import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
# scipy 用于矩阵平方根计算


def compute_fid(real_features, gen_features, eps=1e-6):
    """
    计算 FID（Fréchet Inception Distance）。
    Args:
        real_features: 真实图像的特征 (N_r, D) 矩阵
        gen_features: 生成图像的特征 (N_g, D) 矩阵
        eps: 数值稳定性常数
    Returns:
        fid: FID 值
    """
    # 计算均值向量
    mu_real = real_features.mean(dim=0)
    mu_gen = gen_features.mean(dim=0)

    # 计算协方差矩阵
    sigma_real = torch.cov(real_features.T)
    sigma_gen = torch.cov(gen_features.T)

    # 均值差平方
    diff = mu_real - mu_gen
    diff_sq = (diff * diff).sum()

    # 协方差矩阵迹部分
    # Tr(σ_r + σ_g - 2*(σ_r·σ_g)^(1/2))
    covmean, _ = linalg.sqrtm(
        sigma_real.cpu().numpy() @ sigma_gen.cpu().numpy(),
        disp=False
    )
    # 如果 sqrtm 返回复数，取实部
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    trace = (
        torch.trace(sigma_real)
        + torch.trace(sigma_gen)
        - 2 * torch.trace(torch.from_numpy(covmean))
    )

    return diff_sq + trace


def extract_inception_features(images, model, device="cpu"):
    """
    使用 Inception-v3 提取图像特征。
    Args:
        images: 图像张量 (B, 3, H, W)，归一化到 [0, 1]
        model: Inception-v3 模型
    Returns:
        features: (B, 2048) 特征矩阵
    """
    model.eval()
    with torch.no_grad():
        # 前向传播到倒数第二层
        features = model(images.to(device))
        # 假设 model 返回的是融合池化层输出 (B, 2048)
        return features.cpu()


def compute_fid_small_sample(features_real, features_gen):
    """
    小样本 FID 计算（用于演示，不推荐正式使用）。
    小样本（N < 5000）时使用修正的协方差估计。
    """
    N_r = features_real.size(0)
    N_g = features_gen.size(0)

    # 使用收缩估计（shrinkage estimation）稳定协方差
    # 当样本数小于特征维度时，协方差矩阵会奇异的
    shrinkage = 0.1
    sigma_real = (1 - shrinkage) * torch.cov(features_real.T) \
                 + shrinkage * torch.eye(features_real.size(1))
    sigma_gen = (1 - shrinkage) * torch.cov(features_gen.T) \
                + shrinkage * torch.eye(features_gen.size(1))

    mu_real = features_real.mean(dim=0)
    mu_gen = features_gen.mean(dim=0)

    diff = mu_real - mu_gen
    diff_sq = (diff * diff).sum()

    covmean, _ = linalg.sqrtm(
        sigma_real.cpu().numpy() @ sigma_gen.cpu().numpy(),
        disp=False
    )
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    trace = torch.trace(sigma_real) + torch.trace(sigma_gen) \
            - 2 * torch.trace(torch.from_numpy(covmean))

    return diff_sq + trace
```

### 第 2 步：CLIP Score 计算

```python
def compute_clip_score(image_embeds, text_embeds):
    """
    计算 CLIP Score——图像嵌入和文本嵌入的余弦相似度。
    Args:
        image_embeds: 图像嵌入 (N, D)
        text_embeds: 文本嵌入 (N, D)
    Returns:
        clip_score: 平均 CLIP Score (标量)
    """
    # 归一化嵌入
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # 计算余弦相似度
    similarity = (image_embeds * text_embeds).sum(dim=-1)

    # CLIP Score = max(100 * similarity, 0)
    clip_score = torch.clamp(100 * similarity, min=0)

    return clip_score.mean().item()


def use_pretrained_clip(images, texts):
    """
    使用预训练的 CLIP 模型计算 Score。
    """
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(text=texts, images=images, return_tensors="pt",
                       padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        score = compute_clip_score(
            outputs.image_embeds, outputs.text_embeds
        )

    return score
```

### 第 3 步：LPIPS 计算

```python
def compute_lpips(img1, img2, model="alex"):
    """
    计算 LPIPS 感知相似度。
    使用预置的 LPIPS 网络（基于 AlexNet 或 VGG）。
    Args:
        img1: 图像 1 (B, C, H, W)，值域 [0, 1]
        img2: 图像 2 (B, C, H, W)
    Returns:
        lpips: 平均感知距离 (标量)
    """
    try:
        import lpips
    except ImportError:
        # 如果 lpips 未安装，使用简化版
        return compute_lpips_simple(img1, img2)

    loss_fn = lpips.LPIPS(net=model)  # "alex" 或 "vgg"
    d = loss_fn(img1, img2)
    return d.mean().item()


def compute_lpips_simple(img1, img2):
    """
    简化版 LPIPS——使用 Sobel 边缘检测 + MSE 近似感知差异。
    不是真实 LPIPS，仅用于演示。
    """
    import torch.nn.functional as F

    # 用 Sobel 滤波器提取边缘
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3)

    gray1 = img1.mean(dim=1, keepdim=True)
    gray2 = img2.mean(dim=1, keepdim=True)

    edge1_x = F.conv2d(gray1, sobel_x, padding=1)
    edge1_y = F.conv2d(gray1, sobel_y, padding=1)
    edge2_x = F.conv2d(gray2, sobel_x, padding=1)
    edge2_y = F.conv2d(gray2, sobel_y, padding=1)

    # 边缘差异 + 像素差异
    edge_diff = F.l1_loss(edge1_x, edge2_x) + F.l1_loss(edge1_y, edge2_y)
    pixel_diff = F.l1_loss(gray1, gray2)

    return (edge_diff + pixel_diff).item() / 2
```

---

## 4. 工具

### 4.1 标准评估套件

```python
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
import torchvision.models as models
import torchvision.transforms as T

# 1. FID
fid = FrechetInceptionDistance(feature=2048)
# 真实图像
fid.update(real_images, real=True)
# 生成图像
fid.update(gen_images, real=False)
fid_value = fid.compute()
print(f"FID: {fid_value:.2f}")

# 2. CLIP Score
clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
score = clip_score(gen_images, ["提示词"] * len(gen_images))
print(f"CLIP Score: {score:.2f}")

# 3. LPIPS
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
d = lpips(img1, img2)
```

### 4.2 完整评估流水线

```python
def evaluate_generation(model, dataloader, prompts, num_samples=10000, device="cpu"):
    """评估生成模型的完整流水线。"""
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    # 收集真实图像特征
    for batch_x, _ in dataloader:
        fid_metric.update(batch_x, real=True)

    # 收集生成图像特征
    model.eval()
    with torch.no_grad():
        generated = sample_ddim(model, num_samples=num_samples, device=device)
        fid_metric.update(generated, real=False)

    fid_score = fid_metric.compute()
    return {"fid": fid_score.item()}
```

### 4.3 常用评估指标库

| 库 | 功能 | 安装 |
|----|------|------|
| torchmetrics | FID、CLIP Score、LPIPS | `pip install torchmetrics` |
| clean-fid | 优化的 FID 计算（更快、更准） | `pip install clean-fid` |
| lpips | 感知相似度 | `pip install lpips` |
| CLIP | 文本-图像对齐 | `pip install clip-anytorch` |

---

## 5. LLM 视角

### 5.1 在主流系统中的体现

- **论文评估**：每篇生成模型的论文都需要报告 FID 和 CLIP Score。没有 FID，审稿人不会接受。指标对比表是论文的"标准配置"。
- **模型排行榜**：PapersWithCode、Hugging Face 的排行榜使用 FID 和 CLIP Score 排序。用户根据这些指标选择模型。
- **Midjourney / DALL-E 的实际评估**：商业模型不仅看 FID/CLIP Score，还看用户在 A/B 测试中的偏好率。Midjourney 曾经使用基于用户选择的 Elo 评分。

### 5.2 大语言模型时代什么变了？

FID 和 CLIP Score 在文本生成评估中也找到了对应物——BLEU/ROUGE 对应 FID（分布距离），BERTScore 对应 CLIP Score（语义对齐）。但 LLM 评估更复杂——除了正确性还有真实性、安全性、一致性等维度。2025-2026 年的趋势是使用 LLM-as-Judge（如 GPT-4 评估文本质量），类似于 CLIP 评估图像-文本对齐。

### 5.3 什么没变？

FID 的数学定义（Wasserstein-2 距离）自 2017 年提出以来没有变化。CLIP Score 的余弦相似度计算也没有变化。变化的是(1) 特征提取器的网络架构——从 Inception-v3 到 CLIP/DINOv2；(2) 计算效率——从单 CPU 到 GPU 批量计算。但"分布距离+语义对齐"的评估范式没有变。

### 5.4 使用 ChatGPT / Claude 时的直接体验

FID 和 CLIP Score 直接影响最终用户的使用体验：较高的 CLIP Score 意味着你输入的提示词被更准确地执行。较低的 FID 意味着生成图像更真实、更少伪影。当你在两个模型之间选择时，这些指标是很好的参考——但在选择之前，最好看看两者生成的示例图像是否符合你的审美偏好。

---

## 6. 工程最佳实践

### 6.1 样本数量要求

| 指标 | 最小样本数 | 推荐样本数 | 原因 |
|------|-----------|-----------|------|
| FID | 1000 | 10000+ | 协方差矩阵需要足够样本稳定估计 |
| CLIP Score | 100 | 500+ | 语义对齐不需要大量样本 |
| LPIPS | 1 对 | 50+ | 单对也能算，但统计意义有限 |

### 6.2 中文场景特别建议

- CLIP 模型对中文提示词的支持有限，中文 CLIP Score 可能不准。使用 mCLIP 或 Chinese-CLIP 评估中文提示词
- FID 不需要文字信息，对中文/英文提示词生成的结果同样适用

### 6.3 踩坑经验

- **FID 计算中未归一化图像**：Inception-v3 期望输入在 [0, 1] 或 [-1, 1]，未归一化会导致 FID 异常高
- **小样本 FID**：1000 样本以下，FID 的方差很大，结果不可靠。使用 10000+ 样本
- **CLIP Score 对提示词长度敏感**：长提示词的 CLIP Score 往往高于短提示词——这不是模型更好，而是 CLIP 编码了更多信息

---

## 7. 常见错误

### 错误 1：用小样本计算 FID

**现象：** 同一模型用 100 样本和 10000 样本计算出的 FID 相差 3-5 分。

**原因：** 样本量少时协方差矩阵估计不稳定——尤其在 2048 维空间中，100 个样本远不足以估计一个 2048×2048 的协方差矩阵。

**修复：**

```python
# ❌ 错误：100 样本算 FID
features_real = extract_features(real_images[:100])
features_gen = extract_features(gen_images[:100])
fid = compute_fid(features_real, features_gen)  # 不稳定！

# ✓ 正确：至少 10000 样本
features_real = extract_features(real_images[:10000])
features_gen = extract_features(gen_images[:10000])
fid = compute_fid(features_real, features_gen)  # 稳定
```

### 错误 2：FID 报告未使用标准预处理

**现象：** 同一生成图像在不同实现中计算的 FID 相差很大。

**原因：** 不同的预处理方式（中心裁剪 vs 缩放、双线性 vs 最近邻）导致输入到 Inception-v3 的图像不同。

**修复：**

```python
# ❌ 错误：自定义预处理
transform = T.Compose([T.Resize(512), ...])

# ✓ 正确：使用 FID 计算的标准预处理
transform = T.Compose([
    T.Resize(299),  # Inception-v3 期望 299×299
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 错误 3：混淆 FID 和 IS（Inception Score）

**现象：** 将 FID 标准套用到 IS 上，或反之。例如认为 FID 越高越好（其实是 IS 越高越好）。

**原因：** IS 衡量条件标签分布的熵（高 IS 表示模型覆盖了所有类别），FID 衡量分布距离（低 FID 更好），两个指标的"好坏方向"完全不同。

**修复：**

```python
# ✗ 错误
fid = compute_fid(real_features, gen_features)
if fid > 10:
    print("生成质量好")  # ❌ 错误！FID 越低越好！

# ✓ 正确
fid = compute_fid(real_features, gen_features)
if fid < 10:
    print("生成质量优秀")
elif fid < 30:
    print("生成质量良好")
else:
    print("生成质量需要改进")
```

---

## 8. 面试考点

### Q1：FID 的公式用到了 Frobenius 范式的平方。解释为什么 FID 被广泛使用？（难度：⭐⭐）

**参考答案：**
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2))。第一部分度量均值差——生成图像的平均特征是否与真实图像一致。第二部分度量协方差差异——生成图像的多样性（特征分布）是否与真实图像一致。FID 之所以被广泛使用，是因为(1) 同时捕捉了质量和多样性两个维度；(2) 使用 Inception-v3 的高层语义特征，而非像素级比较——更接近人类对"真实感"的判断；(3) 结果是可复现的数值，方便论文间对比；(4) 对轻微的图像退化有合理的单调性（轻微模糊 ≈ 略微增加的 FID）。

### Q2：CLIP Score 的局限性是什么？在什么情况下它可能给出误导性的结果？（难度：⭐⭐⭐）

**参考答案：**
CLIP Score 的局限性包括：(1) 对提示词措辞敏感——"一只橘色的猫"和"一个橙色的猫"可能得到不同的分数；(2) 对图像构图不敏感——一张内容正确但构图糟糕的图像仍然可以得高 CLIP Score；(3) 不衡量图像质量——充满伪影但语义正���的图像得分可能很高；(4) 长提示词偏向——CLIP 编码器对长文本有编码优势，导致长提示词的生成的图像自然得高分；(5) 对细粒度差异不敏感——CLIP 区分"狗"和"猫"很好，但区分"哈士奇"和"阿拉斯加"可能不准确；(6) 对非英文提示词表现下降——CLIP 主要针对英文训练。

### Q3：如果要在 Inception 域之外（如医学 CT 图像）评估生成质量，应该如何调整 FID？（难度：⭐⭐⭐）

**参考答案：**
FID 使用 Inception-v3 提取特征，而 Inception-v3 是在 ImageNet（自然图像）上训练的。直接应用于医学 CT 图像时，Inception 的特征可能不包含 CT 图像的关键信息（如病灶纹理、解剖结构）。调整方法：(1) 替换特征提取器——使用在相似域上预训练的模型（如医疗图像的 CheXNet）；(2) 微调特征提取器——用真实/生成的数据对特征提取器进行线性探测（linear probing）微调；(3) 使用域无关特征——使用自监督模型（如 DINO、iBOT）的特征，不需要特定域标注；(4) 增加域相关指标——除了 FID，还应包含域特定的量化指标（如 CT 图像的信噪比、分割准确率）。实践中(2)和(3)最常用。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| FID | "生成图像和真实图像像不像" | Fréchet Inception Distance——在 Inception-v3 特征空间中计算两个高斯分布（真实和生成）之间的 Wasserstein-2 距离 |
| CLIP Score | "图和文字对得上吗" | CLIP 模型对图像和文本的嵌入向量的余弦相似度——衡量文本-图像语义对齐度 |
| LPIPS | "两张图看起来一不一样" | Learned Perceptual Image Patch Similarity——基于 VGG 特征差异的感知相似度指标 |
| Inception Score (IS) | "生成的类别多不多" | 基于 Inception-v3 分类置信度的指标——高 IS 表示模型生成多样且清晰的图像（但已被 FID 取代） |
| 协方差 (Covariance) | "特征的分布宽度" | 反映生成图像在特征空间的多样性——FID 公式中用协方差矩阵表示分布的形状 |
| 分布距离 | "两个集合有多像" | 在特征空间中对真实分布和生成分布之间差异的定量衡量——FID 的核心 |

---

## 📚 小结

FID 衡量生成图像与真实图像在 Inception 特征空间中的分布距离——越低越好。CLIP Score 衡量生成图像与文本提示词的语义对齐度——越高越好。LPIPS 衡量两张图像的感知差异。2026 年的三标准：FID + CLIP Score + LPIPS + 人类评估的四维组合。FID 的局限性：对非 Inception 域数据不适用、小样本估计不稳定、不衡量语义对齐。CLIP Score 的局限性：对措辞敏感、不衡量图像质量。

---

## ✏️ 练习

1. **【理解】** 用自己的话解释 FID 公式的每一部分的物理含义：(a) 均值差平方；(b) 协方差矩阵的迹；(c) 为什么需要矩阵平方根项。在什么情况下 FID 会失去其物理意义？

2. **【实现】** 从零实现 `compute_fid` 函数。使用 PyTorch 的 `torch.cov` 计算协方差矩阵，使用 SciPy 的 `linalg.sqrtm` 计算矩阵平方根。在生成的随机数矩阵上验证 FID 计算正确性。

3. **【实验】** 训练一个简单的 GAN 或扩散模型（或使用预训练模型），生成 10000 张图像。分别计算：(a) FID vs 真实图像；(b) CLIP Score vs 原始提示词；(c) 分析 FID 和 CLIP Score 之间的相关性。

4. **【思考】** FID 使用 Inception-v3 的特征。如果用 CLIP 的视觉编码器替代 Inception-v3，会有什么后果？FID→"CLIP-FID" 会测量什么？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|------|------|------|
| FID 从零实现 | `code/fid_implementation.py` | 包含协方差计算、矩阵平方根、完整 FID 流水线 |
| CLIP Score 计算 | `code/clip_score.py` | 图像-文本嵌入的余弦相似度计算 |
| 评估流水线 | `code/evaluation_pipeline.py` | 四维评估组合（FID+CLIP+LPIPS+人类偏好） |
| 指标参考表 | `outputs/metrics-reference.md` | FID / CLIP Score / LPIPS 的数值参考表 |

---

## 📖 参考资料

1. [论文] Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium". NeurIPS, 2017. https://arxiv.org/abs/1706.08500
2. [论文] Radford et al. "Learning Transferable Visual Models from Natural Language Supervision". ICML, 2021. https://arxiv.org/abs/2103.00020
3. [论文] Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric". CVPR, 2018. https://arxiv.org/abs/1801.03924
4. [GitHub] PyTorch Image Quality (PIQ): https://github.com/photosynthesis-team/piq
5. [GitHub] torchmetrics: https://github.com/Lightning-AI/torchmetrics
6. [官方文档] Clean-FID: https://github.com/GaParmar/clean-fid

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
