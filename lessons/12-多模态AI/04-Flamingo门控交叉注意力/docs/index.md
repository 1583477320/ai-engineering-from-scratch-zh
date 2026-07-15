# Flamingo 与门控交叉注意力

> DeepMind 的 Flamingo（2022）率先做了两件事：单个模型可以处理任意交错的图像、视频和文本序列；VLM 可以进行上下文学习——给 3 个 (图像, 描述) 示例和一个查询图像，模型就能描述新图像。核心机制：门控交叉注意力层——插入冻结 LLM 现有层之间，用 tanh 门控在初始化时保持 LLM 的文本能力。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 12 · 03（BLIP-2 Q-Former）| **时间：** ~120 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释门控交叉注意力如何通过 tanh(gate)=0 在初始化时保持冻结 LLM 的文本能力
- [ ] 通过 Perceiver 重采样器：N 个图像图块 → K 个固定"潜在"查询
- [ ] 描述 Flamingo 如何处理交错的图像-文本序列——使用尊重图像位置的因果掩码
- [ ] 构建 Few-shot 多模态提示结构

---

## 1. 问题

BLIP-2 的 Q-Former 能让冻结的 LLM "看见"图像。但 Flamingo 做了更多：(1) 处理交错的图像+视频+文本序列；(2) Few-shot 上下文学习——给 3 个示例就能描述新图像。核心机制是**门控交叉注意力**——在 LLM 每两层之间插入一个交叉注意力层，用 tanh 门控在初始化时让输出为零。

---

## 2. 概念

### 2.1 门控交叉注意力

```
LLM 层 1 → [门控交叉注意力层] → LLM 层 2
    ↑                    ↓
  文本词元          图像特征（来自冻结 ViT）
```

**关键公式：**

```
gated_output = tanh(α) × cross_attention(Q=LLM_hidden, K,V=image_features)
```

α 初始化为 0 → tanh(0) = 0 → 初始化时输出为零 → LLM 的文本能力完整保留。

训练过程中 α 逐渐增大 → 视觉信息逐步注入。

### 2.2 Perceiver 重采样器

将 N 个图像图块压缩为 K 个固定的"潜在查询"（K << N）。这样无论输入图像分辨率如何变化，送入 LLM 的词元数量固定。

### 2.3 交错序列处理

Flamingo 支持交错的 [图像1] [文本1] [图像2] [文本2] ... 序列。图像通过门控交叉注意力注入，文本通过自注意力处理。因果掩码确保每个位置只能看到之前的内容。

### 2.4 Few-shot 上下文学习

```
图像1 → 门控交叉注意力 → [视觉词元1]
文本1: "A cat sitting on a sofa" → 自注意力
图像2 → 门控交叉注意力 → [视觉词元2]
文本2: "A dog running in the park" → 自注意力
图像3 → 门控交叉注意力 → [视觉词元3]
查询: "Describe this image" → 生成回答
```

模型从上下文示例中学习模式，无需梯度更新。

---

## 3. 从零实现

### Step 1：门控交叉注意力

```python
import torch
import torch.nn as nn

class GatedCrossAttention(nn.Module):
    """Flamingo 的门控交叉注意力层。"""
    def __init__(self, llm_dim=768, image_dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(llm_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)
        # 可学习门控参数——初始化为 0
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, text_hidden, image_features):
        """
        text_hidden: (B, T, D_llm) - LLM 当前层的隐藏状态
        image_features: (B, N, D_img) - 来自 ViT 的图像特征
        """
        # tanh(gate) 在 gate=0 时输出 0
        gate_value = torch.tanh(self.gate)

        # 交叉注意力
        normed = self.norm1(text_hidden)
        attn_out = self.cross_attn(normed, image_features, image_features)[0]

        # 门控输出 + 残差
        return text_hidden + gate_value * attn_out
```

### Step 2：Perceiver 重采样器

```python
class PerceiverResampler(nn.Module):
    """将 N 个图块重采样为 K 个潜在查询。"""
    def __init__(self, input_dim=1024, output_dim=768, num_queries=32, num_heads=8):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(num_queries, output_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        """(B, N, D_in) -> (B, K, D_out)"""
        B = image_features.shape[0]
        projected = self.projection(image_features)
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        output = self.cross_attn(queries, projected, projected)[0]
        return output
```

---

## 4. 工具

### 4.1 HuggingFace Transformers

```python
from transformers import AutoProcessor, FlamingoForConditionalGeneration

model = FlamingoForConditionalGeneration.from_pretrained("lmms-lab/Flamingo-3B")
processor = AutoProcessor.from_pretrained("lmms-lab/Flamingo-3B")
```

---

## 6. 工程最佳实践

### 6.1 门控初始化

- α 必须初始化为 0——否则 LLM 文本能力在训练开始时就丢失
- tanh(0) = 0 → 门控输出为零 → LLM 看不到图像 → 保持预训练状态

### 6.2 Few-shot 提示

- 2-4 个 (图像, 描述) 对作为上下文
- 查询图像放在最后
- 系统提示定义任务格式

---

## 7. 常见错误

### 错误 1：门控初始化不为零

**现象：** 训练开始时 LLM 生成混乱文本。

**原因：** α 初始化不为 0 → tanh(α)≠0 → LLM 看到了未处理的图像特征。

### 错误 2：Perceiver 查询数太少

**现象：** 图像中的小物体丢失。

**原因：** K 太小——无法覆盖所有图块信息。

---

## 8. 面试考点

### Q1：门控交叉注意力为什么能保持 LLM 文本能力？（难度：⭐⭐）

**参考答案：**
门控参数 α 初始化为 0 → tanh(0)=0 → 交叉注意力的输出被乘以 0 → 不影响 LLM 的隐藏状态。LLM 在训练开始时完全看不到图像——保持了预训练的文本能力。随着训练，α 逐渐增大，图像信息逐步注入。这比直接微调更安全——LLM 的权重从未被破坏。

### Q2：Flamingo 和 BLIP-2 的桥接方式有什么区别？（难度：⭐⭐⭐）

**参考答案：**
BLIP-2 用 Q-Former——32 个可学习查询从冻结 ViT 提取信息，然后作为词元序列送入冻结 LLM。Flamingo 用门控交叉注意力——在 LLM 每两层之间直接插入交叉注意力层，文本隐藏状态与图像特征交互。BLIP-2 是"先提取再送入"，Flamingo 是"边处理边注入"。Flamingo 更适合交错序列（多图+多文本），BLIP-2 更适合单图问答。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 门控交叉注意力 | "带开关的注意力" | 交叉注意力层的输出乘以 tanh(α)，α 初始化为 0 保持 LLM 预训练能力 |
| Perceiver 重采样器 | "图块压缩器" | 将 N 个图像图块通过交叉注意力压缩为 K 个固定查询（K<<N） |
| Few-shot 上下文学习 | "给几个例子就行" | 模型从几个 (图像, 描述) 示例中学习模式，无需梯度更新 |
| 因果掩码 | "只能看过去" | 在交错序列中确保每个位置只能关注之前的内容 |

---

## 📚 小结

Flamingo 用门控交叉注意力在冻结 LLM 中插入视觉感知——tanh(α) 在初始化时为零，保持文本能力。Perceiver 重采样器压缩图块到固定数量的潜在查询。Few-shot 上下文学习让模型从几个示例中学会描述新图像。这是 2026 年多模态 LLM 的基础架构之一。

---

## ✏️ 练习

1. **【实现】** 实现门控交叉注意力层——验证 gate=0 时输出为零
2. **【实验】** 构建 3-shot 多模态提示——给 3 个 (图像, 描述) 示例，模型描述新图像

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 门控交叉注意力实现 | `code/main.py` | 门控机制 + Perceiver 重采样器 |

---

## 📖 参考资料

1. [论文] Alayrac et al. "Flamingo: a Visual Language Model for Few-Shot Learning". NeurIPS, 2022. https://arxiv.org/abs/2204.14198
2. [论文] Jaegle et al. "Perceiver: General Perception with Iterative Attention". ICML, 2021. https://arxiv.org/abs/2103.03206
3. [论文] Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training". ICML, 2023. https://arxiv.org/abs/2301.12597

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
