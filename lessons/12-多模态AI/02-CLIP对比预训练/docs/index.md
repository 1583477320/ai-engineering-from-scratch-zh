# CLIP 与对比视觉语言预训练

> OpenAI 的 CLIP（2021）证明了一个足够大的想法能驱动未来五年：用噪声网络图文对和对比损失，在同一向量空间中对齐图像编码器和文本编码器。零监督标签，4 亿对数据。由此产生的嵌入空间实现了零样本分类、图文检索，并插入到每个 2026 年 VLM 的视觉塔中。SigLIP 2（2025）用 sigmoid 替代 softmax 并以更低的成本超过了 CLIP。本课从 InfoNCE 到 sigmoid 成对损失推导数学，并在纯 Python 中构建训练步。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 12 · 01（ViT 图块）、阶段 07（Transformer）| **时间：** ~180 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从互信息推导 InfoNCE 损失并实现数值稳定的向量化版本
- [ ] 解释为什么 sigmoid 成对损失（SigLIP）能在不进行全 gather 的情况下扩展到 batch 32768+
- [ ] 通过构建文本模板（"a photo of a {class}"）并取余弦相似度的 argmax 进行零样本 ImageNet 分类
- [ ] 说出 CLIP / SigLIP 预训练提供的四个杠杆：batch size、temperature、prompt template、数据质量

---

## 1. 问题

预 CLIP 的视觉是监督的。收集标注数据集（ImageNet：120 万图像，1000 个类），训练 CNN，部署。标注昂贵、偏向标注者能达成一致的任务，且无法迁移到新任务无需微调。

网络上的图文对有超过 10 亿对免费的弱标注数据。一张金毛犬的照片加上替代文本"my dog Max in the park"就携带了监督信号——文本描述图像。问题：你能把这些变成有用的训练吗？

CLIP 的答案：将图文对视为匹配任务。给定一批 N 张图像和 N 个标题，学习将每张图像与其自身标题匹配，对抗 N-1 个干扰项。监督信号是"这两个东西属于一起；这 N-1 个不属于。" 没有类别标签。没有人类标注。只有对比损失。

由此产生的嵌入空间做比 CLIP 训练更多的事。ImageNet 零样本有效是因为"a photo of a cat"嵌入在从未被标注为猫的猫图片附近。这就是催生 2026 年每个 VLM 的赌注。

---

## 2. 概念

### 2.1 双塔编码器

CLIP 有两个塔：
- **图像编码器 f**：ViT 或 ResNet，输出 D 维向量
- **文本编码器 g**：小型 Transformer，输出 D 维向量

两者都归一化输出到单位长度。相似度 = cos(f(x), g(y)) = f(x)ᵀg(y)。

### 2.2 InfoNCE 损失

CLIP 使用行和列上的对称交叉熵：

```
loss_i2t = CE(S, labels=identity)     # 每张图像的正例是自己的标题
loss_t2i = CE(S^T, labels=identity)   # 每个标题的正例是自己的图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 迫使每张图像匹配自己的标题而非其他标题。"负例"是批次中的所有其他项目。批次越大 = 负例越多 = 信号越强。CLIP 在 batch 32k 上训练；规模很重要。

### 2.3 温度 τ

τ 控制 softmax 的锐度。低 τ → 锐分布，硬负例挖掘效果。高 τ → 柔和，所有样本贡献。CLIP 学习 log(1/τ)，裁剪防止坍缩。

### 2.4 为什么 sigmoid 能更好地扩展（SigLIP）

Softmax 需要整个相似矩阵同步。在分布式训练中必须将每个嵌入全 gather 到每个副本，然后做 softmax。这在通信上对 world size 是二次的。

SigLIP 用逐元素 sigmoid 替代 softmax：对于每对 (i, j)，损失是"这些是匹配对吗？"的二分类。对角线是正类，其余是负类。每对的损失独立——不需要全 gather。每个 GPU 计算自己的局部块并求和。

SigLIP 2 能以低成本扩展到 batch 32k-512k，而 CLIP 需要成比例更多的通信。

### 2.5 零样本分类

用文本模板构建每个类别的嵌入——"a photo of a {class_name}"。对每张图像计算所有类别的余弦相似度，取 argmax 作为预测。这就是 CLIP 的零样本分类能力。

---

## 3. 从零实现

### Step 1：InfoNCE 损失

```python
import torch
import torch.nn.functional as F

def info_nce_loss(image_embeds, text_embeds, temperature=0.07):
    """InfoNCE 对比损失。"""
    # 归一化
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # 相似度矩阵 (N, N)
    logits = image_embeds @ text_embeds.T / temperature

    # 标签：对角线是正例
    labels = torch.arange(len(logits), device=logits.device)

    # 对称损失：图像→文本 + 文本→图像
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
```

### Step 2：Sigmoid 成对损失（SigLIP）

```python
def sigmoid_contrastive_loss(image_embeds, text_embeds, temperature=10.0):
    """SigLIP 的 sigmoid 成对损失——无需全 gather。"""
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # 相似度矩阵
    logits = image_embeds @ text_embeds.T * temperature

    # 标签：对角线为 1，其余为 0
    N = len(logits)
    labels = torch.eye(N, device=logits.device)

    # 逐元素 sigmoid 交叉熵
    loss = -labels * F.logsigmoid(logits) - (1 - labels) * F.logsigmoid(-logits)
    return loss.mean()
```

### Step 3：零样本分类

```python
def zero_shot_classify(image_embed, class_names, text_encoder, template="a photo of a {}"):
    """零样本分类——构建文本模板并匹配。"""
    text_embeds = text_encoder([template.format(name) for name in class_names])
    text_embeds = F.normalize(text_embeds, dim=-1)

    # 余弦相似度
    similarities = image_embed @ text_embeds.T
    predicted_class = similarities.argmax().item()
    return class_names[predicted_class], similarities
```

---

## 4. 工具

### 4.1 OpenCLIP

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("cat.jpg")).unsqueeze(0)
text = tokenizer(["a photo of a cat"])
image_embed = model.encode_image(image)
text_embed = model.encode_text(text)
similarity = (F.normalize(image_embed) @ F.normalize(text_embed).T).item()
```

### 4.2 工具对比

| 工具 | 特点 | 适用场景 |
|------|------|---------|
| OpenCLIP | 开源，多架构 | 研究、自定义训练 |
| CLIP (OpenAI) | 官方实现 | 推理、基线 |
| HuggingFace | 易用集成 | 快速实验 |

---

## 6. 工程最佳实践

### 6.1 训练超参数

| 参数 | CLIP 推荐 | SigLIP 推荐 |
|------|----------|------------|
| batch size | 32768 | 65536+ |
| temperature | 0.07（可学习） | 10.0（固定） |
| 学习率 | 5e-4 → 0 | 1e-3 |
| epochs | 32 | 32 |

### 6.2 中文场景

- CLIP 的中文支持有限——中文图文对较少
- 推荐使用 mCLIP（多语言 CLIP）或 Chinese-CLIP 处理中文
- 零样本分类的文本模板需要中文化

### 6.3 踩坑经验

- **温度梯度爆炸**：温度用 log 空间学习并裁剪——不要直接优化 τ
- **批次太小**：对比学习的负例数量 = 批次大小-1——batch<1024 效果差
- **嵌入维度不够**：CLIP ViT-L 的 768 维比 ViT-B 的 512 维好很多

---

## 7. 常见错误

### 错误 1：忘记对嵌入归一化

**现象：** 损失不收敛——余弦相似度计算错误。

**原因：** 对比学习要求嵌入是单位向量——否则点积不是余弦相似度。

**修复：** 始终用 `F.normalize(embedding, dim=-1)` 归一化后再计算相似度。

### 错误 2：训练批次太小

**现象：** 对比损失效果差——模型无法学习区分正负例。

**原因：** InfoNCE 的负例数量 = batch size - 1。batch=64 时只有 63 个负例——信号太弱。

**修复：** 对比学习需要大批次（>1024）——用梯度累积或分布式训练。

### 错误 3：零样本分类文本模板不统一

**现象：** 相同图像在不同模板下得到不同分类结果。

**原因：** "a photo of a cat" vs "a picture of a cat" 可能嵌入不同位置。

**修复：** 固定标准模板——"a photo of a {class}"。或用多个模板取平均。

---

## 8. 面试考点

### Q1：InfoNCE 损失的本质是什么？（难度：⭐⭐）

**参考答案：**
InfoNCE 是对比学习的核心损失——它最大化正例（匹配的图文对）的相似度，最小化负例（批次中所有不匹配的对）的相似度。本质上是一个 N 分类问题：给定一张图像，从 N 个标题中选出正确的那个。softmax 强制模型将正例的相似度推到比所有负例都高——这学习了一个判别性的嵌入空间。

### Q2：SigLIP 的 sigmoid 损失为什么比 softmax 更高效？（难度：⭐⭐⭐）

**参考答案：**
Softmax 需要计算整个相似矩阵并对每一行归一化——在分布式训练中需要全 gather 所有嵌入到每个副本（通信开销 = O(world_size²)）。SigLIP 的 sigmoid 损失是逐对独立的——每对 (i,j) 的损失只取决于 S[i,j] 和标签 y_ij，不需要全局信息。每个 GPU 可以独立计算自己负责的局部块，然后只需 AllReduce 求和——通信开销 = O(world_size)。这使得 SigLIP 能以低通信成本训练超大批次（32k-512k）。

### Q3：CLIP 的零样本分类为什么有效？（难度：⭐⭐）

**参考答案：**
CLIP 在 4 亿图文对上学习了图像和文本的共享嵌入空间——匹配的图文对距离近，不匹配的距离远。零样本分类时，将类别名称编码为文本嵌入（如 "a photo of a dog"），与图像嵌入计算余弦相似度。即使这些特定图像从未被标注为"dog"，嵌入空间已经学会了"dog"的概念应该嵌入在狗的图像附近。这本质上是**分布外泛化**——CLIP 学到的嵌入空间足够通用，能匹配训练时从未见过的具体类别。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| CLIP | "对比学习的视觉模型" | 用对比损失在共享嵌入空间中对齐图像和文本——零样本分类的基础 |
| InfoNCE | "对比损失" | N 分类交叉熵——让每张图像匹配自己的标题而非其他标题 |
| 温度 τ | "相似度缩放" | 控制 softmax 锐度的可学习参数——影响正负例的分离程度 |
| SigLIP | "sigmoid CLIP" | 用 sigmoid 替代 softmax 的对比学习——可扩展到超大批次 |
| 零样本分类 | "不用标注的分类" | 构建类别文本嵌入，与图像嵌入计算余弦相似度，取 argmax |

---

## 📚 小结

CLIP 用对比学习在 4 亿图文对上学习对齐的嵌入空间——实现了零样本分类、图文检索，并成为所有 VLM 的基础。SigLIP 用 sigmoid 替代 softmax 解决了大规模训练的通信瓶颈。关键超参数：batch size（越大越好）、temperature（影响负例挖掘）、文本模板（统一格式）。2026 年每个 VLM 的视觉塔都是 CLIP/SigLIP 的后代。

---

## ✏️ 练习

1. **【实现】** 用 OpenCLIP 实现零样本 ImageNet 分类——在 100 张图像上测试准确率
2. **【实验】** 对比 CLIP 和 SigLIP 在相同数据上的训练稳定性——谁需要更小的 batch？
3. **【对比】** 用 mCLIP 处理中文文本——与英文 CLIP 对比多语言分类准确率

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| InfoNCE 实现 | `code/main.py` | InfoNCE + sigmoid 成对损失 + 零样本分类 |
| 训练分析 | `outputs/clip-training-analyzer.md` | CLIP/SigLIP 训练超参数分析 |

---

## 📖 参考资料

1. [论文] Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP). ICML, 2021. https://arxiv.org/abs/2103.00020
2. [论文] Zhai et al. "Sigmoid Loss for Language Image Pre-Training" (SigLIP). ICCV, 2023. https://arxiv.org/abs/2303.15343
3. [GitHub] OpenCLIP: https://github.com/mlfoundations/open_clip
4. [论文] Oord et al. "Representation Learning with Contrastive Predictive Coding". arXiv, 2018. https://arxiv.org/abs/1807.03748

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
