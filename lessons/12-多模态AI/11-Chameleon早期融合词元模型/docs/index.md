# Chameleon 与早期融合词元模型

> 我们看过的每个 VLM 都将图像和文本分开处理。视觉词元来自视觉编码器，流入投影器，然后在 LLM 内部与文本相遇。视觉和文本词表从不重叠。Chameleon（Meta，2024 年 5 月）问：如果它们重叠会怎样？训练一个 VQ-VAE 将图像转换为来自共享词表的离散词元序列。每个多模态文档现在是一个序列——文本词元和图像词元交错，一个单一的自回归损失。副作用：模型可以生成混合模态输出——在单次推理调用中交替生成文本和图像词元。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 12 · 05（LLaVA）、阶段 08（生成式 AI）| **时间：** ~180 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释共享词表+单一损失如何改变模型的能力
- [ ] 实现 VQ-VAE 图像 tokenizer——将图像转换为离散词元
- [ ] 构建交错模态输入——文本词元和图像词元混合序列
- [ ] 对比早期融合（Chameleon）和晚期融合（LLaVA）的优劣

---

## 1. 问题

传统 VLM（LLaVA、BLIP-2）将图像和文本分开处理：视觉编码器 → 投影器 → LLM。两个"世界"只在投影器处交汇。这意味着：

- LLM 需要学习视觉词元和文本词元的关系
- 不能生成混合内容（如在文本中嵌入图像）
- 两个词表互相独立

Chameleon 问：**如果图像和文本使用同一个词表会怎样？**

---

## 2. 概念

### 2.1 早期融合 vs 晚期融合

| 方面 | 晚期融合（LLaVA） | 早期融合（Chameleon） |
|------|------------------|---------------------|
| 词表 | 视觉词表 + 文本词表（独立） | 共享词表 |
| 序列 | 视觉词元 + 文本词元（拼接） | 交错（图文交替） |
| 损失 | 视觉损失 + 语言损失 | 单一自回归损失 |
| 生成能力 | 只能生成文本 | 可以生成混合模态 |

### 2.2 VQ-VAE 图像 Tokenizer

```
图像 → [VQ-VAE 编码器] → 离散词元序列 (T, ) → 共享词表中的词元 ID
```

VQ-VAE 将连续图像特征量化为离散词元——类似 AudioCraft 的 EnCodec 对音频做的事。

### 2.3 交错序列

```
文档: [文本词元1] [文本词元2] [图像词元1] [图像词元2] [图像词元3] [文本词元3] ...
       ←── 文本 ──→ ←────── 图像 ──────→ ←── 文本 ──→
```

单一自回归损失处理整个交错序列——模型自然地学会文本和图像的关系。

### 2.4 混合模态生成

模型可以交替生成文本和图像词元：
- "我想画一只猫" → 文本词元
- [图像词元1-64] → 生成的猫图像
- "这是在樱花树下" → 文本词元

---

## 3. 从零实现

### Step 1：VQ-VAE 图像 Tokenizer（简化版）

```python
import torch
import torch.nn as nn

class SimpleVQVAE(nn.Module):
    """简化版 VQ-VAE——将图像转换为离散词元。"""
    def __init__(self, n_embeddings=512, embed_dim=256):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, padding=1),
        )
        # 码本
        self.codebook = nn.Embedding(n_embeddings, embed_dim)
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        z = self.encoder(x)
        # 量化：最近邻匹配码本
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        dists = torch.cdist(z_flat, self.codebook.weight.unsqueeze(0))
        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices)
        return z_q.reshape(B, H, W, -1), indices.reshape(B, H, W)

    def decode(self, z_q):
        z = z_q.permute(0, 3, 1, 2)
        return self.decoder(z)

    def forward(self, x):
        z_q, indices = self.encode(x)
        recon = self.decode(z_q)
        return recon, indices
```

### Step 2：交错序列构建

```python
def build_interleaved_sequence(text_tokens, image_tokens):
    """构建交错的文本+图像序列。"""
    sequence = []
    modalities = []

    for token in text_tokens:
        sequence.append(token)
        modalities.append("text")
    for token in image_tokens:
        sequence.append(token)
        modalities.append("image")

    return sequence, modalities
```

---

## 4. 工具

### 4.1 Meta Chameleon

```python
from transformers import ChameleonForConditionalGeneration

model = ChameleonForConditionalGeneration.from_pretrained(
    "meta-llama/Chameleon-7B"
)
```

### 4.2 Emu3

```python
from transformers import AutoModelForCausalLM
# Emu3 也是早期融合的视觉自回归模型
model = AutoModelForCausalLM.from_pretrained("BAAI/Emu3-Gen")
```

---

## 6. 工程最佳实践

### 6.1 早期融合 vs 晚期融合选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 图像理解/问答 | 晚期融合（LLaVA） | 更简单、更高效 |
| 混合内容生成 | 早期融合（Chameleon） | 需要生成图像词元 |
| 统一架构 | 早期融合 | 一个模型处理所有模态 |

### 6.2 踩坑经验

- **共享词表太小**：图像词元需要的码本比文本大——需要足够大的词表
- **交错序列的注意力掩码**：确保图像词元之间可以互相注意
- **训练不稳定**：早期融合需要小心的学习率和梯度裁剪

---

## 7. 常见错误

### 错误 1：图像词元和文本词元不区分

**现象：** 模型在生成时混淆图像和文本词元。

**原因：** 没有明确的模态标记——模型不知道哪个词元来自图像。

### 错误 2：VQ-VAE 重建质量差

**现象：** 图像词元解码后模糊——视觉信息丢失。

**修复：** 增加码本大小和码本维度——提升重建质量。

---

## 8. 面试考点

### Q1：早期融合和晚期融合的核心区别是什么？（难度：⭐⭐）

**参考答案：**
晚期融合（LLaVA）将图像和文本分开处理——视觉编码器生成视觉词元，投影器映射到 LLM 维度，然后与文本词元拼接送入 LLM。早期融合（Chameleon）用共享词表——图像和文本都用相同的词元表示，交错在同一个序列中，用单一自回归损失训练。早期融合的优势：模型可以生成混合内容（文本中嵌入图像），且架构更统一。劣势：需要训练 VQ-VAE，且码本大小是瓶颈。

### Q2：VQ-VAE 在 Chameleon 中扮演什么角色？（难度：⭐⭐⭐）

**参考答案：**
VQ-VAE 是 Chameleon 的图像 tokenizer——将连续的视觉特征量化为离散词元，这些词元来自共享词表（与文本词元共享）。它解决了一个关键问题：如何让自回归模型处理图像。通过 VQ-VAE，图像被转换为一个固定长度的词元序列（如 64 个词元），然后这个序列可以像文本词元一样被 LLM 处理。码本大小决定了视觉细节的保真度——太小会丢失细节，太大会导致词表爆炸。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 早期融合 | "图文一起处理" | 图像和文本使用共享词表——交错在同一个序列中用单一自回归损失训练 |
| 晚期融合 | "先处理再拼接" | 图像和文本分开处理——在 LLM 内部相遇（如 LLaVA） |
| VQ-VAE | "离散化图像" | 将连续视觉特征量化为离散词元——类似音频的 EnCodec |
| 共享词表 | "一个词表两种模态" | 图像和文本词元来自同一个词表——Chameleon 的核心创新 |
| 交错序列 | "图文混合序列" | 文本词元和图像词元交替排列——单一自回归损失 |

---

## 📚 小结

Chameleon 用共享词表和 VQ-VAE 图像 tokenizer 实现了早期融合——图像和文本作为同一个序列的词元，用单一自回归损失训练。这使模型可以生成混合模态内容。LLaVA（晚期融合）更简单高效，适合图像理解；Chameleon（早期融合）更通用，支持混合生成。2026 年两种范式并存。

---

## ✏️ 练习

1. **【实现】** 构建一个简化 VQ-VAE——将 32×32 图像转换为 8 个离散词元
2. **【对比】** 对比早期融合和晚期融合在图像问答任务上的训练效率

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| VQ-VAE 实现 | `code/main.py` | 图像离散化 + 交错序列构建 |

---

## 📖 参考资料

1. [论文] Team Chameleon. "Chameleon: Mixed-Modal Early-Fusion Foundation Models". arXiv, 2024.
2. [论文] van den Oord et al. "Neural Discrete Representation Learning" (VQ-VAE). NeurIPS, 2017.
3. [论文] Team Emu. "Emu3: Next-Token Prediction is All You Need". arXiv, 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
