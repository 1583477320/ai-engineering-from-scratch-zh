# Emu3：下一个词元预测生成图像和视频

> BAAI 的 Emu3（Wang 等人，2024 年 9 月）是 2024 年应该终结"扩散还是自回归"辩论的结果。一个单一的 Llama 风格解码器 Transformer，只在下一个词元预测目标上训练，在文本 + VQ 图像词元 + 3D VQ 视频词元的统一词表上，超越了 SDXL 的图像生成和 LLaVA-1.6 的感知能力。没有 CLIP 损失。没有扩散调度。发表在 Nature 上。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 12 · 11（Chameleon）| **时间：** ~120 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么 Emu3 的单一损失下一个词元目标能超越扩散模型
- [ ] 对比 Emu3 和扩散模型在图像生成中的优劣
- [ ] 说明 VQ-VAE 图像 tokenizer 如何使自回归图像生成可行
- [ ] 理解分类器无关引导在推理中的作用

---

## 1. 问题

自回归语言模型（GPT、Llama）用"预测下一个词"统治了文本生成。但图像生成一直被扩散模型主导——因为连续图像不适合离散的"下一个词"范式。

Emu3 的答案：**用 VQ-VAE 将图像转换为离散词元**——然后用纯 GPT 风格的下一个词元预测生成图像。没有 CLIP 损失，没有扩散调度。只用一个自回归损失就能超越 SDXL。

---

## 2. 概念

### 2.1 Emu3 的核心思想

```
训练：下一个词元预测（teacher forcing）
生成：自回归采样 + 分类器无关引导（CFG）
```

### 2.2 为什么有效

传统观点认为图像生成需要扩散模型——因为图像是连续的。但 Emu3 证明：**只要 tokenizer 足够好，自回归就能生成高质量图像。** VQ-VAE 将连续图像量化为离散词元——这使自回归成为可能。

### 2.3 对比：Emu3 vs 扩散模型

| 方面 | Emu3（自回归） | SDXL（扩散） |
|------|--------------|-------------|
| 训练损失 | 下一个词元预测 | 去噪 MSE |
| 生成方式 | 逐词元采样 | 逐步去噪 |
| 多模态 | 文本+图像+视频统一 | 分离的模型 |
| 推理速度 | 较慢（逐词元） | 较快（并行去噪） |
| 图像质量 | 接近 SDXL | 基准 |

### 2.4 分类器无关引导（CFG）

推理时的关键技巧：同时生成有条件（带提示词）和无条件（无提示词）的输出，加权组合以提升质量。

```
output = unconditional_output + guidance_scale * (conditional_output - unconditional_output)
```

---

## 3. 从零实现

### Step 1：VQ-VAE 图像 Tokenizer（简化）

```python
import torch
import torch.nn as nn

class SimpleImageTokenizer(nn.Module):
    """简化版图像 tokenizer——将图像转换为离散词元。"""
    def __init__(self, n_codes=512, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, padding=1),
        )
        self.codebook = nn.Embedding(n_codes, embed_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Tanh(),
        )

    def encode(self, x):
        z = self.encoder(x)
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        dists = torch.cdist(z_flat, self.codebook.weight.unsqueeze(0))
        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices)
        return z_q.reshape(B, H, W, -1), indices.reshape(B, H, W)

    def decode(self, z_q):
        return self.decoder(z_q.permute(0, 3, 1, 2))
```

### Step 2：自回归图像生成模拟

```python
def autoregressive_generate(model, prompt_tokens, max_new=64, temperature=0.8):
    """自回归生成图像词元。"""
    tokens = list(prompt_tokens)
    for _ in range(max_new):
        logits = model(torch.tensor(tokens).unsqueeze(0))[:, -1, :]
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        tokens.append(next_token)
    return tokens
```

---

## 4. 工具

### 4.1 HuggingFace

```python
from transformers import AutoModelForCausalLM
# Emu3 通过 Emu3-Gen 模型使用
```

---

## 6. 工程最佳实践

### 6.1 Emu3 vs 扩散模型选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 需要多模态统一 | Emu3 | 文本+图像+视频单一模型 |
| 需要最快推理 | 扩散模型 | 并行去噪 |
| 需要高质量 | 两者相当 | 2024 年 Emu3 超越 SDXL |

---

## 7. 常见错误

### 错误 1：自回归图像生成太慢

**现象：** 生成 256×256 图像需要数百次前向传播。

**原因：** 图像词元数量大——VQ-VAE 产生数百个词元。

**修复：** 使用更大 VQ-VAE（减少词元数量）或并行生成策略。

---

## 8. 面试考点

### Q1：为什么 Emu3 只用下一个词元预测就能生成高质量图像？（难度：⭐⭐）

**参考答案：**
关键在于 VQ-VAE tokenizer——将连续图像转换为离散词元。一旦图像变成了词元序列，自回归预测下一个词元就像预测文本一样自然。Emu3 证明了"更好的 tokenizer + 足够的规模"是全部答案——不需要扩散模型的去噪调度。图像生成的质量瓶颈在 tokenizer 而非生成架构。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Emu3 | "自回归图像生成" | 用 GPT 风格的下一个词元预测生成图像——VQ-VAE tokenizer + 自回归 Transformer |
| 分类器无关引导 | "CFG" | 同时生成有条件和无条件输出，加权组合提升质量 |
| VQ-VAE | "图像离散化" | 将连续图像特征量化为离散词元——自回归图像生成的基础 |

---

## 📚 小结

Emu3 用 VQ-VAE 将图像转换为离散词元，然后用纯 GPT 风格的下一个词元预测生成图像。没有 CLIP 损失，没有扩散调度。证明了"更好的 tokenizer + 足够规模"是图像生成的全部答案。与扩散模型质量相当，但架构更统一——文本+图像+视频用同一个模型。

---

## ✏️ 练习

1. **【对比】** 对比 Emu3 和 SDXL 的架构——哪个更简单？
2. **【分析】** 计算 256×256 图像在 VQ-VAE 后产生多少词元——对比文本序列长度

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 图像 Tokenizer | `code/main.py` | VQ-VAE 编码/解码 + 自回归采样 |

---

## 📖 参考资料

1. [论文] Wang et al. "Emu3: Next-Token Prediction is All You Need". Nature, 2024.
2. [论文] van den Oord et al. "Neural Discrete Representation Learning" (VQ-VAE). NeurIPS, 2017.
3. [论文] Salimans et al. "PixelCNN++" (classifier-free guidance). 2017.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
