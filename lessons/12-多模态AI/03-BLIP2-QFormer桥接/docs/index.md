# 从 CLIP 到 BLIP-2——Q-Former 作为模态桥接

> CLIP 对齐了图像和文本，但不能生成描述、回答问题或对话。BLIP-2（Salesforce，2023）用一个小的可训练桥接解决了这个问题：32 个可学习查询向量通过交叉注意力关注冻结的 ViT 特征，然后直接插入冻结的 LLM 输入流。188M 参数的桥接连接了 11B LLM 和 ViT-g/14。2026 年每个基于适配器的 VLM——MiniGPT-4、InstructBLIP、LLaVA 的表亲——都是它的后代。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 12 · 02（CLIP）、阶段 07（Transformer）| **时间：** ~180 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么冻结视觉编码器和冻结 LLM 之间的可训练桥接比端到端微调更优
- [ ] 实现一个交叉注意力块——可学习查询通过交叉注意力关注外部图像特征
- [ ] 描述 BLIP-2 的两阶段预训练——表示学习和生成学习
- [ ] 对比 Q-Former 和 LLaVA 的 MLP 投影器——各自适用的场景

---

## 1. 问题

CLIP 可以判断"这张图是猫"，但不能说"这张图里有一只猫坐在沙发上"。我们需要一个能将视觉信息转换为 LLM 能理解的文本词元序列的模块。

核心问题：如何将冻结的视觉编码器输出连接到冻结的 LLM 输入，同时只训练一个小型桥接网络？

BLIP-2 的答案：**Q-Former**——32 个可学习查询向量，通过交叉注意力从冻结的 ViT 提取视觉信息，然后作为词元序列送入冻结的 LLM。只需训练 188M 参数的桥接，就能让 11B 的 LLM "看见"图像。

---

## 2. 概念

### 2.1 Q-Former 架构

```
冻结 ViT: 图像 → N 个图块特征 (N, D)
    ↓ 交叉注意力
可学习查询: K 个查询向量 (K, D), K << N
    ↓ 自注意力
Q-Former 输出: K 个视觉词元 (K, D)
    ↓ 线性投影到 LLM 嵌入维度
冻结 LLM: 输入 = [视觉词元] + [文本词元]
```

关键：查询向量是**可学习的**，它们学会了"问"视觉特征中什么信息。

### 2.2 为什么冻结是关键

| 组件 | 训练？ | 原因 |
|------|--------|------|
| ViT | 冻结 | 视觉特征已经通过 CLIP 预训练好 |
| Q-Former | **可训练** | 桥接网络——学习视觉到语言的映射 |
| LLM | 冻结 | 语言能力已经通过预训练好 |

冻结 + 小型可训练桥接 = 低成本 + 高稳定性 + 不会灾难性遗忘。

### 2.3 BLIP-2 两阶段训练

**阶段 1：表示学习（ITC + ITM + ITG）**
- ITC：图像-文本对比学习（类似 CLIP）
- ITM：图像-文本匹配（二分类）
- ITG：图像条件文本生成

**阶段 2：生成学习**
- 只训练 Q-Former
- LLM 接收 Q-Former 输出的视觉词元
- 使用语言建模损失训练

---

## 3. 从零实现

### Step 1：可学习查询 + 交叉注意力

```python
import torch
import torch.nn as nn

class QFormer(nn.Module):
    """简化版 Q-Former。"""
    def __init__(self, num_queries=32, embed_dim=768, num_heads=8):
        super().__init__()
        # 可学习查询
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        # 自注意力
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, image_features):
        """
        Args:
            image_features: (B, N, D) - ViT 输出的图块特征
        Returns:
            visual_tokens: (B, K, D) - Q-Former 输出
        """
        B = image_features.shape[0]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        # 自注意力
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]

        # 交叉注意力：查询关注图像特征
        q = self.norm2(queries)
        queries = queries + self.cross_attn(q, image_features, image_features)[0]

        return self.norm3(queries)
```

### Step 2：桥接 LLM

```python
class VisualBridge(nn.Module):
    """Q-Former + 投影到 LLM 嵌入维度。"""
    def __init__(self, num_queries=32, vit_dim=1024, llm_dim=4096):
        super().__init__()
        self.qformer = QFormer(num_queries=num_queries, embed_dim=vit_dim)
        self.projection = nn.Linear(vit_dim, llm_dim)

    def forward(self, image_features):
        visual_tokens = self.qformer(image_features)  # (B, K, D_vit)
        projected = self.projection(visual_tokens)  # (B, K, D_llm)
        return projected
```

---

## 4. 工具

### 4.1 HuggingFace Transformers

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", device_map="auto"
)
```

### 4.2 工具对比

| 方法 | 参数量 | 质量 | 推理速度 |
|------|--------|------|---------|
| Q-Former (BLIP-2) | 188M | 高 | 中 |
| MLP 投影 (LLaVA) | ~20M | 高 | 快 |
| 全量微调 | 全部 | 最高 | 慢 |

---

## 6. 工程最佳实践

### 6.1 Q-Former vs MLP 选择

| 场景 | Q-Former | MLP |
|------|---------|-----|
| 需要压缩 token 数 | ✅（32 个词元） | ❌（N 个词元） |
| 推理速度要求高 | ❌ | ✅（更简单） |
| 需要精细视觉控制 | ✅（可学习查询） | ❌ |

### 6.2 踩坑经验

- **查询数太少**：K<16 时丢失视觉信息——建议 32-64
- **冻结策略不当**：解冻 ViT 可能提升质量但增加成本——先冻结试效果
- **分辨率受限**：Q-Former 的图块数随分辨率增长——高分辨率需要更大 K

---

## 7. 常见错误

### 错误 1：解冻了冻结的组件

**现象：** 训练后 LLM 的语言能力退化。

**原因：** 解冻了 LLM 权重——少量视觉数据导致灾难性遗忘。

**修复：** 只训练 Q-Former 和投影层——LLM 和 ViT 始终冻结。

### 错误 2：查询数与图块数不匹配

**现象：** 视觉信息丢失——图像中的小物体检测不到。

**原因：** K（查询数）太小——无法覆盖所有图块的信息。

**修复：** K≥32。对于高分辨率图像，使用 K=64 或更多。

---

## 8. 面试考点

### Q1：为什么在冻结模型之间加可训练桥接比端到端微调更好？（难度：⭐⭐）

**参考答案：**
(1) **成本低**：只训练 188M 参数（Q-Former），不训练 11B 的 LLM；(2) **稳定**：冻结的预训练模型提供稳定的特征，训练只需学习模态映射；(3) **不会遗忘**：冻结的 LLM 保留了预训练的语言能力，不会被少量视觉数据"破坏"。

### Q2：Q-Former 和 MLP 投影器的本质区别是什么？（难度：⭐⭐⭐）

**参考答案：**
MLP 投影器是线性的——将每个图块特征独立映射到 LLM 维度，保留 N 个词元。Q-Former 通过 K 个可学习查询做交叉注意力——从 N 个图块中提取最重要的 K 个信息，输出 K 个词元（K<<N）。Q-Former 更节省 token 预算（32 vs 196），但 MLP 更简单、更快。LLaVA 选择 MLP 是因为"更简单胜过更聪明"——简单投影就能达到接近 Q-Former 的质量。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Q-Former | "视觉词元提取器" | 32 个可学习查询通过交叉注意力从冻结 ViT 提取视觉信息的轻量模块 |
| 模态桥接 | "连接视觉和语言" | 在冻结的视觉编码器和冻结的 LLM 之间添加可训练映射层 |
| ITC | "对比预训练" | Image-Text Contrastive——对齐图像和文本嵌入 |
| ITM | "匹配预训练" | Image-Text Matching——判断图文是否匹配 |
| 可学习查询 | "智能的注意力探针" | Q-Former 中可训练的查询向量——学会"问"视觉特征中什么信息 |

---

## 📚 小结

BLIP-2 用 Q-Former 在冻结的 ViT 和 LLM 之间架起桥梁——188M 参数连接 11B LLM 和 ViT-g/14。可学习查询通过交叉注意力提取视觉信息，投影后送入 LLM。冻结 + 小型桥接 = 低成本 + 高稳定性。Q-Former vs MLP：前者压缩 token 预算，后者更快更简单。2026 年的 VLM 大多是这个模式的变体。

---

## ✏️ 练习

1. **【实现】** 实现 Q-Former 的核心——32 个查询通过交叉注意力关注 ViT 特征
2. **【对比】** 用 Q-Former（K=32）和 MLP（N=196 个词元）分别桥接同一个 ViT，对比 token 使用量

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| Q-Former 实现 | `code/main.py` | 可学习查询 + 交叉注意力 + 模态桥接 |

---

## 📖 参考资料

1. [论文] Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models". ICML, 2023. https://arxiv.org/abs/2301.12597
2. [论文] Alayrac et al. "Flamingo: a Visual Language Model for Few-Shot Learning". NeurIPS, 2022. https://arxiv.org/abs/2204.14198
3. [论文] Liu et al. "Visual Instruction Tuning" (LLaVA). NeurIPS, 2023. https://arxiv.org/abs/2304.08485

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
