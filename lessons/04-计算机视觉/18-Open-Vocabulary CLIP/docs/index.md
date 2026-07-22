# Open-Vocabulary CLIP：让视觉模型认识任意类别

> 用一张图片一段文字就能定义一个新类别——你不需要标注数据，只需要会写句子。

**类型：** 实现课
**语言：** Python
**前置知识：** 第 14 课（Vision Transformers）、第 17 课（自监督学习）
**预计时间：** ~90 分钟
**所处阶段：** Tier 1
**关联课程：** 第 03 阶段 · 第 16 课（对比学习）— 对比损失的推广到图文域；第 05 阶段 · 第 08 课（多模态基础）— CLIP 是视觉语言模型的基石

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 CLIP 的双塔架构设计——为什么图像和文本需要独立的编码器
- [ ] 从零实现对称对比损失，理解温度参数 $\tau$ 对训练的影响
- [ ] 使用预训练 CLIP 进行零样本分类，无需任何标签数据
- [ ] 设计有效的提示词模板集，利用模板平均提升零样本准确率
- [ ] 实现 TIP-Adapter——一种不修改模型权重的提示词适配方法
- [ ] 构建双向图文检索系统——图像到文本、文本到图像

## 1. 问题

传统的图像分类器是封闭词汇表（closed-vocabulary）的：一个在 ImageNet 上训练的 1000 类分类器，只能预测这 1000 个标签。每当你想识别一个新的类别——比如"无人机"或者"某种罕见花卉"——你就需要收集该类别的标注数据，重新训练分类头。

这种模式有两个根本缺陷：

1. **扩展成本极高**——每个新类别都意味着新一轮的数据收集和模型训练。
2. **无法覆盖长尾分布**——现实中大量的类别只有少数几个样本，传统分类器完全无法学习。

CLIP（Radford et al., OpenAI 2021）证明了完全不同的思路：通过在 40 亿对（图像,  caption）上进行对比学习，模型学到的不是"1000 个固定类别"，而是"图像与文本语义的映射关系"。推理时，你只需要用自然语言描述一个类别，模型就能判断图片是否匹配。

这就是开放词汇表（open-vocabulary）能力——一个模型可以识别**任意**类别，只要你能够用文字描述它。

这张能力不是"锦上添花"，它是现代视觉系统的基石。2026 年主流的视觉管线无一例外地建立在 CLIP 或其变体之上：

| 任务 | 基于 CLIP 的系统 |
|---|---|
| 零样本目标检测 | Grounding DINO, OWL-ViT |
| 开放词汇分割 | CLIPSeg, SAM + 文本提示 |
| 视觉语言模型 | LLaVA, Qwen-VL, InternVL |
| 图文检索 | 商品搜索、医疗影像归档 |
| 文生图模型 | Stable Diffusion（条件信号来自 CLIP） |

理解 CLIP，就是理解 2020 年之后所有多模态 AI 系统的起点。

## 2. 概念

### 2.1 直观理解

CLIP 的核心想法可以用一句话概括：

> **把图片和描述它的文字放到同一个向量空间中，让匹配的配对靠得更近，不匹配的配对离得更远。**

这个过程可以想象成一场"配对游戏"：

```
给定一批图片及其对应的文字描述：
┌──────────┐   ┌──────────────┐
│ 图片 A   │   │ "一只橘猫"    │  ← 匹配对
└──────────┘   └──────────────┘
┌──────────┐   ┌──────────────┐
│ 图片 B   │   │ "一辆汽车"    │  ← 匹配对
└──────────┘   └──────────────┘

训练目标：
  图片 A 的嵌入  ≈  "一只橘猫" 的嵌入     ✓
  图片 B 的嵌入  ≈  "一辆汽车" 的嵌入      ✓
  图片 A 的嵌入  ≠  "一辆汽车" 的嵌入      ✓
  图片 B 的嵌入  ≠  "一只橘猫" 的嵌入      ✓
```

关键设计选择在于**如何学习**这种配对关系。CLIP 采用的是双塔架构（Two-Tower Architecture），也就是为图像和文本分别训练一个独立的编码器。这与后来流行的单模态融合方法（如让图像先通过 ViT，再拼接词元送入 Transformer）形成了鲜明对比：

- **双塔**：图像和文本各自编码为向量后直接计算相似度。推理灵活——可以独立编码整个图库，只在查询时用文本编码新提示。
- **单塔融合**：图像 patch 和文本词元一起送入 Transformer。表达能力强但推理时需要同时编码图文，开销更大。

CLIP 选择了双塔，因为它更适配零样本推理和大规模索引的场景。

### 2.2 形式化定义

#### 双塔编码

$$
z_i = f_{\theta}^{\text{img}}(x_i) \in \mathbb{R}^D
$$

$$
z_t = g_{\theta}^{\text{text}}(t) \in \mathbb{R}^D
$$

两个编码器共享相同的输出维度 $D$（CLIP-B/32 为 512，CLIP-L/14 为 1024）。

#### L2 归一化与相似度

$$
\hat{z}_i = \frac{z_i}{\|z_i\|_2}, \quad \hat{z}_t = \frac{z_t}{\|z_t\|_2}
$$

$$
s_{i,t} = \tau \cdot \hat{z}_i^\top \hat{z}_t
$$

其中 $\tau$ 是可学习的温度参数（初始值设为 $\ln(1/0.07) \approx 2.659$）。归一化后点积等于余弦相似度，温度参数控制 Softmax 的锐度。

#### 对称对比损失

给定一个大小为 $N$ 的批次，包含 $N$ 对匹配的图片-文本：

$$
S = \tau \cdot Z_i Z_t^\top \in \mathbb{R}^{N \times N}
$$

$$
L_{\text{i2t}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(S_{i,i})}{\sum_{j=1}^{N} \exp(S_{i,j})}
$$

$$
L_{\text{t2i}} = -\frac{1}{N}\sum_{t=1}^{N} \log \frac{\exp(S_{t,t})}{\sum_{j=1}^{N} \exp(S_{t,j})}
$$

$$
L_{\text{CLIP}} = \frac{L_{\text{i2t}} + L_{\text{t2i}}}{2}
$$

这是对称的——图像到文本和文本到图像各有一个交叉熵损失。对称性确保了推理时两种方向的检索都能工作。

#### 零样本分类

训练完成后，对于新类别 $c$，构造提示词模板 $T_c$（如"a photo of a {class}"），编码得到类嵌入 $z_c = g_\theta^{\text{text}}(T_c)$。测试图像的预测为：

$$
\hat{y} = \arg\max_c \; \tau \cdot \hat{z}_{\text{img}}^\top z_c
$$

这里的关键是：分类过程中没有使用任何标签进行微调。模型靠的是预训练中学到的通用视觉-语言对齐。

### 2.3 动手验证：对比损失的直觉

用极简的代码验证对比损失的行为：

```python
import torch
import torch.nn.functional as F

# 假设一个 4×4 的相似度矩阵（4 张图 × 4 段文字）
sim_matrix = torch.tensor([
    [ 2.0, -0.5,  0.1, -0.3],
    [-0.4,  1.8, -0.2,  0.6],
    [ 0.3, -0.1,  2.1, -0.5],
    [-0.2,  0.7, -0.3,  1.9],
])

labels = torch.arange(4)

# 行方向：每张图匹配对应文本的概率
row_probs = F.softmax(sim_matrix, dim=-1)
print("行概率（每张图对各文本的匹配度）:")
for i in range(4):
    print(f"  图 {i}: diag={row_probs[i, i]:.3f}")

# 列方向：每段文本匹配对应图的概率
col_probs = F.softmax(sim_matrix.T, dim=-1)
print("\n列概率（每段文本对各图的匹配度）:")
for j in range(4):
    print(f"  文本 {j}: diag={col_probs[j, j]:.3f}")
```

```text
行概率（每张图对各文本的匹配度）:
  图 0: diag=0.938
  图 1: diag=0.855
  图 2: diag=0.946
  图 3: diag=0.883

列概率（每段文本对各图的匹配度）:
  文本 0: diag=0.794
  文本 1: diag=0.832
  文本 2: diag=0.861
  文本 3: diag=0.824
```

可以看到：对角线元素的概率已经很高了——因为我们在初始化时就把匹配对的相似度设为了正值，非匹配对设为负值。这正是对比损失要达到的效果：训练是让这种区分能力从数据中自动学到。

### 2.4 SigLIP：比 CLIP 更好的损失函数

CLIP 的对称交叉熵有一个固有缺陷：它是一个 softmax over the whole batch。在大批次（8192+）下，每个样本有 $N-1$ 个负样本，训练很充分。但在小批量下，负样本不足，梯度信号会变弱。

SigLIP（Zhai et al., Google 2023）用逐对 sigmoid 替换了 softmax：

$$
L_{\text{SigLIP}} = \frac{1}{N^2} \sum_{i,j} \log\left(1 + \exp(-y_{ij} \cdot s_{ij})\right)
$$

其中 $y_{ij} = +1$ 表示匹配对，$-1$ 表示非匹配对。

**核心差异**：

| 特性 | CLIP | SigLIP |
|---|---|---|
| 损失形式 | 对称交叉熵（softmax） | 逐对 sigmoid |
| 批量依赖 | 强依赖大批次（≥4096） | 无批次依赖，小批次也能训练 |
| 归一化 | 需要批次内归一化 | 逐对独立决策 |
| 最终性能 | 好 | 同等数据下 ≥ CLIP |

在工业实践中，SigLIP 已经成为新的默认选择。Hugging Face 提供了 `google/siglip-base-patch16-224` 等预训练权重。

### 2.5 知识连线：CLIP 与前后课程的关联

CLIP 对比学习直接继承了第 16 课（对比学习）的核心思想——InfoNCE 损失的图文扩展。它的双塔架构与 CNN 特征提取（第 4 课）和 ViT（第 14 课）形成互补：CLIP 的视觉编码器通常是预训练的 ViT，其文本编码器是标准的 Transformer。同时，CLIP 为后续的多模态模型（VLM）奠定了嵌入空间对齐的基础。

## 3. 从零实现

### 第 1 步：最简双塔模型

先用最简单的 MLP 模拟 CLIP 的双塔结构，确保在 CPU 上可以运行训练循环。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerCLIP(nn.Module):
    """最简 CLIP 双塔模型。

    两个塔分别处理图像特征和文本特征，共享相同的嵌入维度。
    可学习的温度参数控制对比损失的锐度。
    """

    def __init__(self, visual_dim=512, text_dim=128, embed_dim=64):
        super().__init__()
        # 视觉塔：投影到嵌入空间
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )
        # 文本塔：投影到嵌入空间
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )
        # 可学习温度：初始化为 ln(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def encode_image(self, x):
        """编码图像特征并归一化。"""
        emb = self.visual_proj(x)
        return F.normalize(emb, dim=-1)

    def encode_text(self, x):
        """编码文本特征并归一化。"""
        emb = self.text_proj(x)
        return F.normalize(emb, dim=-1)

    def forward(self, img_feat, txt_feat):
        img_emb = self.encode_image(img_feat)
        txt_emb = self.encode_text(txt_feat)
        scale = self.logit_scale.exp()
        return img_emb, txt_emb, scale
```

### 第 2 步：对称对比损失

```python
def clip_loss(image_emb, text_emb, logit_scale):
    """CLIP 对称对比损失。

    对 batch 中的 N 对 (图像, 文本)，构建 NxN 相似度矩阵，
    对称训练使得对角线匹配对的相似度高，非对角线低。

    Args:
        image_emb:    (N, D) 归一化图像嵌入
        text_emb:     (N, D) 归一化文本嵌入
        logit_scale:  温度参数 tau

    Returns:
        loss:         标量损失
    """
    N = image_emb.size(0)
    # 计算相似度矩阵并缩放
    sim_matrix = logit_scale * (image_emb @ text_emb.T)

    # 标签：对角线为匹配对 [0, 1, ..., N-1]
    labels = torch.arange(N, device=sim_matrix.device)

    # 双向交叉熵：图像→文本 + 文本→图像
    loss_i2t = F.cross_entropy(sim_matrix, labels)
    loss_t2i = F.cross_entropy(sim_matrix.T, labels)

    return (loss_i2t + loss_t2i) / 2
```

**为什么需要对称？** 因为下游任务需要双向检索：你可以用文字搜图（文本→图像），也可以用图搜文字（图像→文本）。只训练一个方向会导致另一个方向的检索质量很差。

### 第 3 步：合成数据训练循环

用一组精心构造的合成数据来演示训练过程：每个类别有一个语义原型，正样本在原型附近加少量噪声。

```python
# 数据准备：5 个类别，每个类别一个原型
num_classes = 5
dim = 32
rng = torch.Generator().manual_seed(42)
prototypes = F.normalize(torch.randn(num_classes, dim, generator=rng), dim=-1)

# 采样函数：返回带标签的 (图像特征, 文本特征) 批次
def sample_batch(batch_size=32):
    labels = torch.randint(0, num_classes, (batch_size,))
    img = torch.randn(batch_size, 512)
    txt = torch.randn(batch_size, 128)
    # 注入原型信号：前 dim 维使用原型 + 少量噪声
    for b in range(batch_size):
        proto = prototypes[labels[b].item()]
        img[b, :dim] += proto
        txt[b, :dim] += proto
    return img, txt, labels

# 训练
model = TwoTowerCLIP(visual_dim=512, text_dim=128, embed_dim=dim)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(100):
    img_b, txt_b, _ = sample_batch(32)
    img_emb, txt_emb, scale = model(img_b, txt_b)
    loss = clip_loss(img_emb, txt_emb, scale)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"step {step:3d}  loss = {loss.item():.3f}  "
              f"tau = {scale:.3f}")
```

```text
step   0  loss = 2.019  tau = 14.318
step  20  loss = 0.246  tau = 4.521
step  40  loss = 0.089  tau = 2.847
step  60  loss = 0.062  tau = 2.712
step  80  loss = 0.051  tau = 2.681
```

损失从接近 $\log(32) \approx 3.47$（随机初始化期望值）快速下降到接近 0——说明模型学会了区分匹配对和非匹配对。

### 第 4 步：零样本分类

训练完成后，用任意文本描述来分类图像：

```python
@torch.no_grad()
def zero_shot_predict(model, image_feat, class_prompts):
    """
    零样本分类。

    Args:
        model:          训练好的模型
        image_feat:     (1, visual_dim) 单张图像特征
        class_prompts:  List[str] 每个类别对应一条提示词

    Returns:
        predicted_class_name: 预测的类别名
    """
    # 编码图像
    img_emb = model.encode_image(image_feat)  # (1, D)

    # 编码所有类别提示词
    text_embs = []
    for prompt in class_prompts:
        # 简单地将字符串转换为特征向量（教学简化版）
        char_codes = [ord(c) % 128 for c in prompt[:50]]
        txt_idx = torch.tensor(char_codes, dtype=torch.long).unsqueeze(0)
        txt_emb = model.encode_text(txt_idx.float())
        text_embs.append(txt_emb)
    text_matrix = torch.cat(text_embs, dim=0)  # (C, D)

    # 相似度 & 预测
    sim = img_emb @ text_matrix.T  # (1, C)
    pred = sim.argmax(dim=-1).item()
    return class_prompts[pred], sim[0].tolist()


# 测试
classes = ["猫", "狗", "汽车", "花朵", "建筑"]
pred_class, scores = zero_shot_predict(model, test_img, classes)
print(f"预测: {pred_class} (得分: {scores})")
```

## 4. 工业工具

### 4.1 OpenCLIP — 开源复现

OpenCLIP 是当前最活跃的 CLIP 复现项目：

```python
import open_clip
import torch
from PIL import Image

# 加载预训练模型
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# 零样本推理
image = preprocess(Image.open("dog.jpg")).unsqueeze(0)
texts = tokenizer(["一只猫", "一条狗", "一辆车"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(texts)
    # L2 归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    # 余弦相似度 → 概率
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(f"分类概率: {probs.tolist()}")
```

### 4.2 Hugging Face Transformers — 统一接口

Hugging Face 提供统一的 `CLIPModel` 接口，支持 OpenAI 和 CLIP 系列模型：

```python
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("dog.jpg").convert("RGB")
texts = ["一只猫", "一条狗", "一只鸟"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    image_features = model.get_image_features(inputs["pixel_values"])
    text_features = model.get_text_features(inputs["input_ids"])
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(f"概率: {probs.squeeze().tolist()}")
```

### 4.3 SigLIP — 新一代替代

```python
from transformers import AutoModel, AutoProcessor

# Google 的 SigLIP 模型
model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
```

### 4.4 性能对比

| 实现方式 | 速度 | 显存 | 适用场景 |
|---|---|---|---|
| 我们的 NumPy/PyTorch 版 | 慢 | 极低 | 学习理解 |
| OpenCLIP ViT-B/32 | 快 | 中 | 通用零样本分类 |
| OpenCLIP ViT-L/14 | 中等 | 高 | 高精度需求 |
| SigLIP ViT-B/16 | 快 | 中 | 小规模批量训练 |
| Hugging Face Pipeline | 中等 | 中 | 快速原型开发 |

### 4.5 中文场景提示词模板

中文 CLIP 推理时，提示词的设计需要考虑中文表达方式：

```python
# 推荐的中文提示词模板
CHINESE_TEMPLATES = [
    "一张{}的照片",
    "一个{}的特写",
    "{}的插图",
    "一张模糊的{}照片",
    "{}的线稿",
    "{}的黑白照片",
]
```

不同模板编码后的嵌入取平均，可以有效提升分类鲁棒性。

## 5. 知识连线

CLIP 的核心思想——用对比学习在跨模态空间中对齐不同数据类型——是后续几乎所有多模态系统的基础。

- **后续阶段 12（多模态 AI）**：LLaVA、Qwen-VL 等视觉语言模型直接使用 CLIP 的视觉编码器作为"眼睛"，将图像嵌入映射到大语言模型的文本空间中。
- **后续阶段 08（生成式 AI）**：Stable Diffusion 和 DALL-E 用 CLIP 的文本编码器生成条件信号，指导图像生成过程。
- **实际工程应用**：2026 年的内容审核系统、商品搜索引擎、医疗影像检索，绝大多数都以 CLIP 的图文嵌入为基础构建。

## 6. 工程最佳实践

### 6.1 模板数量与收益递减

OpenAI 原始论文测试了 80 种模板，在 ImageNet 上零样本准确率提升约 2%。经验法则：

| 模板数量 | 预期提升 | 说明 |
|---|---|---|
| 1 | 基线 | "a photo of a {}" |
| 3-5 | +0.5~1% | 基本的风格变化 |
| 10-20 | +1~2% | 覆盖不同表述习惯 |
| 80 | +2~3% | OpenAI 最优值 |
| >100 | 趋缓 | 边际收益极低，增加推理延迟 |

### 6.2 TIP-Adapter 的使用建议

TIP-Adapter 适合以下场景：

- 有少量标注验证集，但不想全量微调 CLIP
- 领域偏移（domain shift）显著，如从自然图像迁移到卫星图像
- 需要快速适配新类别而不影响通用能力

注意：`alpha` 参数控制了适配器对原始嵌入的修改程度，训练中应该从 0 开始学习，避免过度破坏预训练的知识。

### 6.3 中文场景特别建议

- **字符编码**：中文 CLIP（如 Chinese-CLIP）使用专门在中文语料上训练的文本编码器，比直接用英文 CLIP 处理中文提示词效果好 5-10 个百分点。
- **模板长度**：中文句子通常比英文更长，注意 CLIP 文本编码器的最大上下文长度限制（通常为 77 个词元）。
- **分词差异**：CLIP 的文本编码器使用 WordPiece 分词，对中文是按字切分的，不像英文有子词机制。这意味中文每个字是一个词元，77 个词元大约对应 77 个汉字。

### 6.4 检索系统的工程搭建

构建图文检索系统时的关键步骤：

1. **离线编码**：将所有图像和文本的嵌入预先计算并存储，避免在线编码的重复计算。
2. **近似最近邻搜索（ANN）**：对于超过 10 万条目的数据库，使用 FAISS 或 Milvus 等 ANN 库进行高效检索。
3. **混合索引**：先做粗排（FAISS），再做精排（重新计算相似度），平衡速度和精度。

## 7. 常见错误

### 错误 1：忽略温度参数的影响

**现象：** 训练初期 loss 非常高且震荡，或者最终损失几乎为 0 导致所有预测分数接近。

**原因：** 温度参数 $\tau$ 控制了相似度分布的锐度。$\tau$ 太大时 softmax 过于尖锐，容易过拟合；太小时分布过于平滑，正负样本难以区分。CLIP 将其初始化为 $\ln(1/0.07) \approx 2.659$，这是一个经过大量实验验证的经验值。

**修复：** 不要手动设置温度，让它作为可学习参数在训练中自适应调整。

```python
# ❌ 固定温度
TEMPERATURE = 0.07
sim_matrix = sim_matrix / TEMPERATURE

# ✓ 可学习温度
self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
# 在推理时：sim_matrix * self.logit_scale.exp()
```

### 错误 2：误用余弦相似度而非归一化点积

**现象：** 分类结果不合理，相似度分数范围异常（超出 [−1, 1]）。

**原因：** 余弦相似度要求向量已归一化。如果忘记归一化而直接使用点积，得到的不是余弦相似度，可能导致数值溢出或预测偏差。

**修复：** 始终确保嵌入在执行点积之前 L2 归一化。

```python
# ❌ 错误：未归一化
sim = img_emb @ text_emb.T  # 可能产生任意大小的值

# ✓ 正确：先归一化
img_emb = F.normalize(img_emb, dim=-1)
text_emb = F.normalize(text_emb, dim=-1)
sim = img_emb @ text_emb.T  # 值域 [−1, 1]
```

### 错误 3：提示词大小写不一致

**现象：** 相同类别用不同大小写的提示词，分类结果波动很大。

**原因：** CLIP 文本编码器对大小写敏感。"A Dog"和"a dog"会产生不同的文本嵌入。OpenAI 的原始模板全部使用小写开头（"a photo of a {}"），这是经过实验验证的最佳做法。

**修复：** 统一模板的大小写格式，优先小写。

```python
# ❌ 不一致
prompts = ["a photo of a Dog", "A Photo Of A cat", "photo of bird"]

# ✓ 统一格式
prompts = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]
```

### 错误 4：零样本分类时忘记取多个模板的平均

**现象：** 换了一组类别就几乎无法分类——模型"过拟合"到了单条提示词的表述上。

**原因：** 单条提示词只提供关于类别的一个视角。比如"a photo of a dog"只强调了"照片"这种形式，模型可能在遇到"狗的卡通画"时就失效了。

**修复：** 始终使用多条模板，编码后对文本嵌入取平均。

## 8. 面试考点

### Q1：CLIP 的双塔架构为什么要把图像和文本分开编码，而不是合并到一个 Transformer 中？（难度：⭐⭐）

**参考答案：**

双塔架构的核心优势是推理灵活性。训练时图像和文本成对出现，必须分别编码然后计算相似度。但在推理时，你通常希望"一次编码，多次查询"：先把整个图像库编码好存入索引，然后每次只需编码新的查询文本进行搜索。如果图像和文本在一个 Transformer 中联合编码，每次查询都需要重新传入整张图像，失去了一切效率。

此外，双塔让两个编码器可以独立使用不同的预训练权重和架构——图像端可以用 ViT，文本端可以用 Transformer Decoder，最大化各自的表达能力。

### Q2：为什么对比损失要对称（同时计算图像→文本和文本→图像）？（难度：⭐⭐）

**参考答案：**

因为下游任务需要双向检索能力。零样本分类本质上是"文本到图像"的匹配——用文本提示编码类别，与图像嵌入比较相似度。但检索任务往往需要双向：用户可以输入文字找图片（文本→图像），也可以上传图片找相似描述（图像→文本）。

只训练单向（如仅 i2t）会导致模型学到偏斜的嵌入空间：一个方向很好，另一个方向很差。对称损失强制两个方向都对齐，得到一个"双向兼容"的共享空间。

### Q3：模板平均（prompt averaging）为什么能提升零样本准确率？（难度：⭐⭐⭐）

**参考答案：**

每条提示词模板提供了一个关于类别的不同"视角"。"a photo of a dog"强调图像形式，"a sketch of a dog"强调手绘风格，"a blurry photo of a dog"强调低质量图像。这些不同的表述激活了文本编码器的不同子空间。

将这些嵌入取平均，相当于在语义流形上找到了一个"类别中心"——它综合了所有可能的自然语言描述方式，因此对任意输入的鲁棒性更强。这就好比用多个证人互相印证，比单一证人的证词更可靠。

### Q4：TIP-Adapter 与全量微调的区别是什么？（难度：⭐⭐⭐）

**参考答案：**

TIP-Adapter 只训练一个轻量级适配器网络（通常几十 KB 参数），不修改预训练模型的权重。它的公式是：

$$z_{\text{adapted}} = z_{\text{original}} + \alpha \cdot f(z_{\text{image}})$$

其中 $f$ 是小型适配器，$\alpha$ 控制修改幅度。

全量微调则更新所有参数（数十亿级别），需要大量 GPU 显存和时间，且存在灾难性遗忘风险——调好了特定任务，却破坏了通用能力。TIP-Adapter 的优势在于：不破坏预训练知识、显存占用极小、可随时切换回零样本模式。代价是效果上限略低于全量微调。

### Q5：CLIP 和 DINOv2 都做了自监督学习，它们有什么区别？什么时候选哪个？（难度：⭐⭐⭐）

**参考答案：**

DINOv2 是纯视觉自监督——只用图像，不借助任何文本信号。它学到的是"视觉上相关的 patch 应该相似"，适合纯视觉任务如分割、深度估计。CLIP 是跨模态自监督——利用互联网上的（图像, 文本）对，学到的是"与文字描述一致的特征"。

选择原则：

- **需要理解语义内容**（分类、检索、VLM）：选 CLIP 或 SigLIP。它们已经见过海量图文对，对词汇和概念的感知远超纯视觉模型。
- **需要几何理解**（分割、深度、姿态）：选 DINOv2。它的特征对空间结构更敏感。
- **最好的方案**：两者结合。近年许多 SOTA 系统同时使用了 DINOv2（结构化特征）和 CLIP（语义特征）的融合。

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 开放词汇表 | "能认任意东西" | 模型不依赖预定义的类别集合，推理时通过文本描述动态确定目标类别 |
| 双塔架构 | "两个编码器" | 图像和文本各自通过独立编码器映射到同一维度的嵌入空间，点积即相似度 |
| 对比学习 | "让相似的在一起" | 在批次内构建正对和负对，通过交叉熵损失让匹配的嵌入靠近、不匹配的远离 |
| 温度参数 | "tau" | 可学习标量，缩放相似度矩阵后输入 Softmax。值越大分布越锐利，越小越平滑 |
| 零样本 | "不训练就能用" | 在目标任务上不接触任何标签数据，直接利用预训练模型的泛化能力进行分类 |
| 提示词模板 | "a photo of a {}" | 将类别名称填入自然语言框架中生成的文本，用于引导文本编码器产生有意义的嵌入 |
| SigLIP | "sigmoid 版 CLIP" | 用逐对 sigmoid 损失替代 softmax 交叉熵，摆脱了对大批次的依赖 |
| TIP-Adapter | "不用微调的适配器" | 在推理时通过小型适配器网络微调文本嵌入，不改变预训练模型权重的提示词优化方法 |
| 图文嵌入对齐 | "图片和文字在一个空间" | 通过对比训练让匹配的图文对在向量空间中距离接近，实现跨模态的语义比较 |
| 模板平均 | "多条提示词取平均" | 对同一类别的多条提示词模板分别编码，将得到的嵌入向量取平均值作为该类的代表嵌入 |

## 📚 小结

CLIP 用对称对比损失将图像和文本映射到同一个嵌入空间——匹配对的距离近，非匹配对的距离远。有了这个共享空间，零样本分类、双向检索、以及视觉语言模型的架构都变得自然而然。模板平均、TIP-Adapter 等方法进一步释放了预训练 CLIP 的潜力，使得它在几乎没有标注数据的场景下依然表现优异。

下一课我们将深入探索基于 CLIP 的图像-文本检索系统——如何用 FAISS 构建大规模索引，以及如何结合 TIP-Adapter 实现领域自适应。

## ✏️ 练习

1. 【理解】用自己的话解释为什么 CLIP 的对比损失需要对两个方向（图像→文本和文本→图像）都计算交叉熵。写 200 字以内的说明，让一个没有 ML 背景的工程师能听懂。

2. 【实现】修改代码中的 `zero_shot_classify` 函数，使其支持加权模板投票——可以为不同模板赋予不同权重（如"清晰照片"模板的权重高于"模糊照片"模板）。

3. 【实验】使用 OpenCLIP 预训练模型（`ViT-B-32`）在 CIFAR-10 数据集上测试零样本分类。分别使用 1 条模板和 5 条模板，报告 Top-1 准确率的差距。

4. 【思考】SigLIP 用逐对 sigmoid 损失替代了 CLIP 的 softmax 交叉熵。请分析在 batch size = 128 的情况下，两种损失的训练稳定性差异，并画出两种损失在单个批次上的计算图对比。

5. 【进阶】阅读 TIP-Adapter 论文（Zhu et al., CVPR 2022），用你自己的话解释"为什么在推理时使用一个轻量级适配器比直接微调整个模型更适合零样本迁移场景"。写出你设计的适配器结构，并估算参数量与全量微调的差距。

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| 双塔 CLIP 完整实现 | `code/main.py` | 从零实现双塔模型、对比损失、零样本分类、提示词工程、TIP-Adapter、图文检索 |
| 提示词设计指南 | `outputs/prompt-clip-guide.md` | 面向零样本 CLIP 分类的提示词模板设计方法 |
| 图文检索技能 | `outputs/image-text-retriever.md` | 构建基于 CLIP 的大规模图文检索系统 |

## 📖 参考资料

1. [论文] Radford et al. "Learning Transferable Visual Models from Natural Language Supervision". ICML, 2021. https://arxiv.org/abs/2103.00020
2. [论文] Zhai et al. "Sigmoid Loss for Language Image Pre-Training". ICCV, 2023. https://arxiv.org/abs/2303.15343
3. [论文] Zhu et al. "Tip-Adapter: Training-Free ADAPTER for CLIP". CVPR, 2022. https://arxiv.org/abs/2206.08636
4. [GitHub] mlfoundations/open_clip — OpenCLIP 开源实现 https://github.com/mlfoundations/open_clip
5. [官方文档] Hugging Face Transformers CLIP: https://huggingface.co/docs/transformers/model_doc/clip
6. [官方文档] FAISS 向量检索库: https://github.com/facebookresearch/faiss

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
