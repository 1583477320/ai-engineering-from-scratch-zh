# Show-o 与离散扩散统一模型

> Transfusion 混合了连续和离散表示。Show-o（Xie 等人，2024 年 8 月）走了另一条路：文本词元使用因果下一个词元预测，图像词元使用掩码离散扩散（受 MaskGIT 启发）。两者在一个具有混合注意力掩码的 Transformer 中。结果统一了 VQA、文生图、图像修复和混合模态生成——一个骨干、每种模态一个词表、一个损失公式。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 12 · 13（Transfusion）| **时间：** ~120 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释掩码离散扩散——均匀掩码再恢复的调度过程
- [ ] 对比 Show-o 和 Transfusion/Emu3 的架构差异
- [ ] 说明混合注意力掩码如何让一个 Transformer 处理多种模态
- [ ] 理解掩码扩散作为并行图像生成器的优势

---

## 1. 问题

Transfusion 混合了连续和离散表示——但需要管理两种不同的损失。Show-o 问：**能不能保持所有词元都是离散的，但用扩散的方式生成图像？**

Show-o 的答案：图像词元用**掩码离散扩散**——随机掩码图像词元，让 Transformer 一次性恢复多个掩码位置。这比逐词元自回归快得多。

---

## 2. 概念

### 2.1 掩码离散扩散

```
原始图像词元: [v1, v2, v3, v4, v5, v6, v7, v8]
                    ↓ 随机掩码 50%
掩码图像:     [v1, [M], v3, [M], v5, [M], v7, [M]]
                    ↓ Transformer 一次性恢复所有 [M]
完整图像:     [v1, v2, v3, v4, v5, v6, v7, v8]
```

### 2.2 混合注意力掩码

Show-o 用不同的注意力掩码处理不同模态：
- 文本词元：因果掩码（只能看之前）
- 图像词元：无掩码（双向，可以看所有图块）
- 掩码图像词元：可以看未掩码的图像词元和所有文本词元

### 2.3 与 Transfusion/Emu3 对比

| 方面 | Show-o | Transfusion | Emu3 |
|------|--------|-------------|------|
| 图像表示 | 离散 VQ-VAE | 连续图块 | 离散 VQ-VAE |
| 生成方式 | 掩码恢复（并行） | 扩散去噪 | 自回归逐词元 |
| 注意力掩码 | 混合 | 混合 | 因果 |
| 速度 | 较快（并行恢复） | 中等 | 较慢（逐词元） |

### 2.4 掩码调度

掩码率从高到低——早期掩码大部分词元，后期掩码越来越少。这类似扩散模型的噪声调度——从高噪声到低噪声。

---

## 3. 从零实现

### Step 1：掩码离散扩散

```python
import torch
import torch.nn as nn

def mask_tokens(tokens, mask_ratio=0.5):
    """随机掩码图像词元。"""
    mask = torch.bernoulli(torch.full_like(tokens.float(), mask_ratio))
    masked_tokens = tokens * (1 - mask) + mask * 100  # 100 = [MASK] ID
    return masked_tokens, mask


def masked_discrete_diffusion_loss(model, tokens, mask_ratio=0.5):
    """掩码离散扩散损失。"""
    masked_tokens, mask = mask_tokens(tokens, mask_ratio)
    logits = model(masked_tokens)  # Transformer 预测所有位置
    # 只在掩码位置计算损失
    mask_flat = mask.view(-1)
    targets = tokens.view(-1)
    logits_flat = logits.view(-1, logits.size(-1))
    loss = F.cross_entropy(logits_flat, targets, reduction='none')
    loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
    return loss
```

### Step 2：统一生成流程

```python
def show_o_generate(model, prompt_tokens, max_new_tokens=64, temperature=0.8):
    """Show-o 风格生成——混合文本和图像。"""
    generated = list(prompt_tokens)
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([generated]))[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
    return generated
```

---

## 4. 工具

### 4.1 Show-o

```python
from transformers import AutoModelForCausalLM
# Show-o 通过 HuggingFace 使用
```

---

## 6. 工程最佳实践

### 6.1 Show-o vs Transfusion vs Emu3 选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 需要并行图像生成 | Show-o | 掩码恢复比逐词元快 |
| 需要最高图像质量 | Transfusion | 连续图块无量化损失 |
| 需要最简单架构 | Emu3 | 纯自回归 |
| 需要统一多模态 | 任意 | 三者都支持 |

---

## 7. 常见错误

### 错误 1：掩码率不匹配训练和推理

**现象：** 推理时掩码率与训练不一致——图像质量差。

**修复：** 使用与训练相同的掩码调度。

### 错误 2：混合注意力掩码设计不当

**现象：** 文本和图像词元混淆——模型无法区分模态。

**修复：** 确保文本和图像有明确的注意力边界。

---

## 8. 面试考点

### Q1：掩码离散扩散为什么比自回归更快？（难度：⭐⭐）

**参考答案：**
自回归逐词元生成——N 个词元需要 N 次前向传播。掩码扩散一次性恢复所有掩码词元——无论有多少掩码，只需 1 次前向传播。对于图像（通常数百个词元），掩码扩散的加速是显著的。关键是掩码扩散可以并行恢复所有掩码位置，而自回归必须顺序生成。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 掩码离散扩散 | "并行图像生成" | 随机掩码图像词元，让 Transformer 一次性恢复——比自回归快 |
| 混合注意力掩码 | "不同模态不同掩码" | 文本用因果掩码，图像用无掩码，跨模态条件化 |
| Show-o | "统一多模态" | 文本 NTP + 图像掩码扩散——一个 Transformer 处理所有模态 |

---

## 📚 小结

Show-o 用掩码离散扩散生成图像——比自回归更快（并行恢复）。文本用因果 NTP，图像用掩码恢复。混合注意力掩码让一个 Transformer 处理多种模态。与 Transfusion（连续图块+扩散）和 Emu3（离散词元+自回归）形成对比——三者是多模态统一的三种路径。

---

## ✏️ 练习

1. **【对比】** 对比 Show-o（掩码扩散）和 Emu3（自回归）的生成速度
2. **【分析】** 掩码扩散在什么场景下比自回归更有优势？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 掩码扩散实现 | `code/main.py` | 掩码离散扩散 + 混合注意力掩码 |

---

## 📖 参考资料

1. [论文] Xie et al. "Show-o: Unifying Multimodal Understanding and Generation". arXiv, 2024.
2. [论文] Chang et al. "MaskGIT: Masked Generative Image Transformer". CVPR, 2022.
3. [论文] Esser et al. "Scaling Rectified Flow Transformers" (SD3). ICML, 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
