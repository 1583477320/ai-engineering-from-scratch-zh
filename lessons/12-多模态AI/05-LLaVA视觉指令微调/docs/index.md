# LLaVA 与视觉指令微调

> LLaVA（2023）用 2 层 MLP 替代 Q-Former，用朴素词元拼接替代门控交叉注意力。简单到令人怀疑的架构，却成为 2023-2026 年最广泛使用的 VLM。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 12 · 02（CLIP）、阶段 11（指令微调）| **时间：** ~180 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 构建 2 层 MLP 投影器——将 ViT 图块嵌入映射到 LLM 嵌入维度
- [ ] 走过 LLaVA 两阶段训练流程
- [ ] 构建 LLaVA 格式的提示词
- [ ] 解释为什么 MLP 简单方案效果接近 Q-Former

---

## 1. 问题

BLIP-2 的 Q-Former 需要专门的两阶段训练。Flamingo 的门控交叉注意力复杂。LLaVA 的答案：**用一个简单的 2 层 MLP 投影器替代所有复杂桥接。**

核心创新不在架构而在**数据**——GPT-4 生成的 158K 视觉指令对。

---

## 2. 概念

### 2.1 LLaVA 架构

```
图像 → [冻结 ViT] → 图块嵌入 (N, 1024)
    ↓
[MLP: 1024→4096] → 视觉词元 (N, 4096)
    ↓
拼接: [视觉词元] + [文本词元]
    ↓
[冻结/微调 LLM] → 生成文本
```

### 2.2 两阶段训练

| 阶段 | 数据 | 训练对象 | 目标 |
|------|------|---------|------|
| 阶段 1 | 558K 图像-标题对 | 仅 MLP | 视觉词元与文本词元对齐 |
| 阶段 2 | 158K GPT-4 视觉指令对 | LLM + MLP | 遵循视觉指令 |

### 2.3 MLP vs Q-Former

| 方面 | MLP | Q-Former |
|------|-----|----------|
| 参数量 | ~20M | 188M |
| 输出词元数 | N（图块数） | K（固定，如 32） |
| 推理速度 | 快 | 中 |
| 质量 | 接近 | 接近 |
| 实现复杂度 | 简单 | 复杂 |

社区选择 MLP：**"更简单的方案胜过更聪明的方案"**——MLP 质量接近但快得多。

---

## 3. 从零实现

### Step 1：MLP 投影器

```python
import torch
import torch.nn as nn

class LLaVAProjector(nn.Module):
    """LLaVA 的 MLP 投影器——将 ViT 嵌入映射到 LLM 维度。"""
    def __init__(self, vit_dim=1024, llm_dim=4096):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(vit_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, image_features):
        """(B, N, D_vit) -> (B, N, D_llm)"""
        return self.projection(image_features)
```

### Step 2：LLaVA 提示构建

```python
def build_llava_prompt(image_tokens, user_message):
    """构建 LLaVA 格式的提示词。"""
    return f"""<image>{image_tokens}</image>
<|user|>
{user_message}
<|assistant|>"""
```

### Step 3：评估指标

```python
def evaluate_vlm(model, test_cases):
    """评估 VLM 性能。"""
    correct = 0
    for image_features, expected in test_cases:
        prompt = build_llava_prompt(image_features, expected["question"])
        response = model.generate(prompt)
        if expected["answer"].lower() in response.lower():
            correct += 1
    return correct / len(test_cases)
```

---

## 4. 工具

### 4.1 HuggingFace Transformers

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
```

### 4.2 LLaVA 变体

| 模型 | 分辨率 | 特点 |
|------|--------|------|
| LLaVA 1.5 | 336 | 基础版 |
| LLaVA-NeXT | 672 | 高分辨率 |
| LLaVA-OneVision | 动态 | 图像+视频统一 |

---

## 6. 工程最佳实践

### 6.1 MLP vs Q-Former 选择

| 场景 | 推荐 |
|------|------|
| 高吞吐推理 | MLP（更快） |
| 低 token 预算 | Q-Former（32 词元） |
| 快速原型 | MLP（最简单） |

### 6.2 踩坑经验

- **视觉词元数太多**：N 个图块词元占用大量上下文窗口——考虑降低分辨率或增大图块大小
- **训练数据质量差**：GPT-4 生成的数据可能有幻觉——需人工审核
- **ViT 分辨率限制**：训练分辨率和推理分辨率不匹配需要插值

---

## 7. 常见错误

### 错误 1：投影器维度不匹配

**现象：** 训练时报维度错误。

**原因：** ViT 输出维度与 LLM 嵌入维度不同——MLP 必须匹配两者。

### 错误 2：忘记冻结 ViT

**现象：** 训练后 ViT 的视觉特征退化。

**原因：** ViT 与 LLM 一起训练——视觉数据量不足以更新 ViT。

---

## 8. 面试考点

### Q1：LLaVA 的 MLP 为什么效果接近 Q-Former？（难度：⭐⭐）

**参考答案：**
关键不在桥接复杂度而在**训练数据质量**。Q-Former 用 188M 参数压缩信息，MLP 用 20M 参数传递信息。但 LLaVA 的 GPT-4 生成的视觉指令数据（158K 高质量对）提供了足够的监督信号，使得简单投影器也能学到有效的视觉-语言映射。**"数据为王"是 LLaVA 的核心洞察。**

### Q2：LLaVA 的两阶段训练有什么必要性？（难度：⭐⭐⭐）

**参考答案：**
阶段 1（投影器对齐）：只训练 MLP，让视觉词元和文本词元在同一空间对齐。这步不需要指令数据——普通的图像-标题对就行。阶段 2（视觉指令微调）：微调 LLM + MLP，让模型学会遵循视觉指令。两阶段分开的原因：(1) 阶段 1 只需要简单对齐，不需要指令数据；(2) 阶段 2 需要高质量指令数据但不能从零训练 LLM。分开可以分别优化。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| MLP 投影器 | "简单的桥接" | 2 层 MLP 将 ViT 嵌入维度映射到 LLM 嵌入维度 |
| 视觉指令微调 | "用指令训练视觉能力" | 在视觉指令对上微调 LLM，让模型遵循视觉指令 |
| AnyRes | "任意分辨率" | LLaVA 1.5 的多分辨率处理机制 |
| GPT-4 生成数据 | "合成指令数据" | 用 GPT-4 从图像描述生成视觉指令对 |

---

## 📚 小结

LLaVA 用最简单的 MLP 投影器 + 高质量 GPT-4 生成数据构建了最广泛使用的 VLM。两阶段训练：先对齐投影器，再微调 LLM。MLP 简单但效果接近 Q-Former——"更简单胜过更聪明"。LLaVA-NeXT/OneVision 在此基础上扩展了分辨率和视频支持。

---

## ✏️ 练习

1. **【实现】** 实现 MLP 投影器——验证输入输出维度正确
2. **【对比】** 用 Q-Former（K=32）和 MLP（N=196）桥接同一个 ViT，对比生成质量

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| MLP 投影器实现 | `code/main.py` | 2 层 MLP 投影器 + 两阶段训练逻辑 |

---

## 📖 参考资料

1. [论文] Liu et al. "Visual Instruction Tuning" (LLaVA). NeurIPS, 2023. https://arxiv.org/abs/2304.08485
2. [论文] Liu et al. "Improved Baselines with Visual Instruction Tuning" (LLaVA-1.5). CVPR, 2024. https://arxiv.org/abs/2310.03744
3. [论文] Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training". ICML, 2023. https://arxiv.org/abs/2301.12597

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
