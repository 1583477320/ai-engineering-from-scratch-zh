# Transfusion：自回归文本 + 扩散图像的统一 Transformer

> Chameleon 和 Emu3 将所有赌注押在离散词元上。它们有效，但量化的瓶颈可见——图像质量在连续空间扩散模型之下。Transfusion（Meta，2024 年 8 月）选择了相反的赌注：保持图像连续，去掉 VQ-VAE，用两个损失训练一个 Transformer。文本词元用下一个词元预测。图像图块用 flow-matching/diffusion 损失。两个目标优化相同的权重。Stable Diffusion 3（MMDiT）的架构是它的近亲。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 12 · 11（Chameleon）、阶段 08（生成式 AI）| **时间：** ~180 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 在一个 Transformer 中连接两个损失——文本词元的 NTP 和图像图块的扩散 MSE
- [ ] 解释混合注意力掩码如何让一个 Transformer 同时处理离散和连续模态
- [ ] 对比 Transfusion 和 Chameleon/Emu3 的优劣
- [ ] 说明 Transfusion 与 Stable Diffusion 3（MMDiT）的关系

---

## 1. 问题

Chameleon/Emu3 用 VQ-VAE 将图像转换为离散词元——但量化不可避免地丢失信息。图像质量在某些场景下低于连续空间的扩散模型。

Transfusion 的答案：**不要量化。** 保持图像为连续的图块，用扩散/flow-matching 损失训练。文本用下一个词元预测。两种损失在同一个 Transformer 上优化。

---

## 2. 概念

### 2.1 Transfusion 架构

```
输入：文本词元 [t1, t2, ..., tN] + 图像图块 [p1, p2, ..., pM]（连续）

训练目标：
  - 文本词元：下一个词元预测（交叉熵）
  - 图像图块：扩散/flow-matching 损失（MSE）

推理：
  1. 自回归生成文本词元
  2. 逐步去噪生成图像图块（从噪声到清晰）
```

### 2.2 两种损失

| 损失 | 适用 | 目标 |
|------|------|------|
| 下一个词元预测 | 文本词元 | 预测下一个离散 token |
| 扩散 MSE | 图像图块 | 从噪声预测干净图块 |

### 2.3 混合注意力掩码

一个 Transformer 同时处理离散和连续模态——需要特殊的注意力掩码：
- 文本词元：因果注意力（只能看之前）
- 图像图块：双向注意力（可以看所有图块）
- 跨模态：文本可以看图像，但图像不看文本（条件化）

### 2.4 与 Stable Diffusion 3 的关系

Stable Diffusion 3（MMDiT）是 Transfusion 的近亲——使用多模态 DiT 架构，在图像图块上做扩散，文本通过交叉注意力注入。

---

## 3. 从零实现

### Step 1：两种损失的 Transformer

```python
import torch
import torch.nn as nn

class TransfusionTransformer(nn.Module):
    """Transfusion 风格 Transformer——处理离散文本和连续图像。"""
    def __init__(self, vocab_size=32000, image_dim=768, hidden=1024, n_layers=12, n_heads=16):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, hidden)
        self.image_proj = nn.Linear(3 * 16 * 16, hidden)  # 假设 16x16 图块
        self.pos_embed = nn.Embedding(512, hidden)  # 最大序列长度

        encoder_layer = nn.TransformerEncoderLayer(hidden, n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.text_head = nn.Linear(hidden, vocab_size)
        self.image_head = nn.Linear(hidden, 3 * 16 * 16)

    def forward(self, text_tokens=None, image_patches=None, modality_mask=None):
        """处理混合模态输入。"""
        embeddings = []
        if text_tokens is not None:
            embeddings.append(self.text_embed(text_tokens))
        if image_patches is not None:
            embeddings.append(self.image_proj(image_patches))
        x = torch.cat(embeddings, dim=1)

        seq_len = x.shape[1]
        x = x + self.pos_embed(torch.arange(seq_len, device=x.device))

        # 注意力掩码处理
        if modality_mask is not None:
            x = self.transformer(x, mask=modality_mask)
        else:
            x = self.transformer(x)

        text_logits = self.text_head(x) if text_tokens is not None else None
        image_logits = self.image_head(x) if image_patches is not None else None
        return text_logits, image_logits
```

### Step 2：混合损失训练

```python
def transfusion_loss(text_logits, image_logits, text_targets, image_targets, text_mask):
    """Transfusion 的混合损失。"""
    text_loss = F.cross_entropy(
        text_logits.view(-1, text_logits.size(-1)),
        text_targets.view(-1),
        ignore_index=-100
    )
    image_loss = F.mse_loss(image_logits, image_targets)
    # 加权组合
    total_loss = text_loss + image_loss
    return total_loss
```

---

## 4. 工具

### 4.1 Meta MMDiT

Stable Diffusion 3 的 MMDiT 架构与 Transfusion 同源——多模态 DiT，图像做扩散，文本做条件化。

---

## 6. 工程最佳实践

### 6.1 Transfusion vs 离散词元方案

| 方面 | Transfusion | Chameleon/Emu3 |
|------|-------------|----------------|
| 图像表示 | 连续图块 | 离散 VQ-VAE 词元 |
| 损失 | 扩散 MSE + NTP | 纯 NTP |
| 图像质量 | 更高 | 受量化限制 |
| 复杂度 | 更高 | 更简单 |
| 生成方式 | 自回归文本 + 扩散图像 | 全部自回归 |

### 6.2 踩坑经验

- **两种损失的梯度冲突**：文本损失和图像损失的梯度量级不同——需要权重平衡
- **注意力掩码设计**：跨模态注意力的掩码需要精心设计——否则文本和图像混淆

---

## 7. 常见错误

### 错误 1：两种损失不平衡

**现象：** 模型偏向其中一个模态——文本质量下降或图像质量下降。

**修复：** 调整两种损失的权重——通常文本损失:图像损失 ≈ 1:1。

### 错误 2：推理时不知道何时切换模态

**现象：** 模型无法在文本和图像生成之间正确切换。

**修复：** 使用特殊的模态切换 token——告诉模型"接下来生成图像"。

---

## 8. 面试考点

### Q1：Transfusion 和 Chameleon/Emu3 的核心区别是什么？（难度：⭐⭐⭐）

**参考答案：**
Chameleon/Emu3 用 VQ-VAE 将图像转换为离散词元——所有模态都用下一个词元预测。Transfusion 保持图像为连续图块——文本用 NTP，图像用扩散 MSE。Transfusion 的优势：图像质量不受 VQ-VAE 量化损失影响。劣势：架构更复杂——需要管理两种不同的损失和注意力掩码。Stable Diffusion 3 的 MMDiT 是 Transfusion 的近亲。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Transfusion | "混合损失 Transformer" | 一个 Transformer 同时用 NTP（文本）和扩散 MSE（图像）训练 |
| MMDiT | "多模态 DiT" | Stable Diffusion 3 的架构——Transfusion 的近亲 |
| 混合注意力掩码 | "图文不同的掩码" | 文本用因果掩码，图像用双向掩码，跨模态条件化 |

---

## 📚 小结

Transfusion 用混合损失（NTP + 扩散）在一个 Transformer 中同时处理文本和图像——避免了 VQ-VAE 的量化损失。图像保持连续，质量更高。Stable Diffusion 3 的 MMDiT 是其近亲。与 Chameleon/Emu3 相比，Transfusion 更复杂但图像质量更好。

---

## ✏️ 练习

1. **【对比】** 对比 Transfusion 和 Chameleon 的架构复杂度和图像质量
2. **【设计】** 设计 Transfusion 的混合注意力掩码——文本和图像应该如何交互

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 混合损失 Transformer | `code/main.py` | NTP + 扩散 MSE 的双损失训练 |

---

## 📖 参考资料

1. [论文] Zhou et al. "Transfusion: Predict the Next Token and the Diffusion Latent". arXiv, 2024.
2. [论文] Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3). ICML, 2024.
3. [论文] Team Chameleon. "Chameleon: Mixed-Modal Early-Fusion Foundation Models". arXiv, 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
