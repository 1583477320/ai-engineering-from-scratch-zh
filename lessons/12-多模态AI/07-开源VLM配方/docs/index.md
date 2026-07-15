# 开源 VLM 配方：什么真正重要

> 2024-2026 年的开源 VLM 文献是消融实验表的森林。Apple 的 MM1 测试了 13 种图像编码器、连接器和数据混合组合。Allen AI 的 Molmo 证明了详细的人工标题比 GPT-4V 蒸馏更好。Cambrian-1 比较了 20+ 种编码器。Idefics2 形式化了五轴设计空间。Prismatic VLM 在受控基准上比较了 27 种训练配方。从所有这些噪音中，有一小部分结果跨论文成立：图像编码器比连接器架构更重要，数据混合比两者都重要，详细的人工标题比蒸馏合成数据更好。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 12 · 05（LLaVA）| **时间：** ~180 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 说出 VLM 设计空间的五个轴：图像编码器、连接器、LLM、数据混合、分辨率策略
- [ ] 解释为什么图像编码器的选择比连接器更重要
- [ ] 对比不同数据混合对 VLM 质量的影响
- [ ] 设计一个 VLM 训练配方——选择编码器、连接器和数据

---

## 1. 问题

你想构建一个 VLM。但选择太多：用 CLIP 还是 DINOv2？MLP 还是 Q-Former？什么数据混合？这些选择如何影响性能？论文告诉你"我们的方法最好"——但每个论文都这么说。

**核心洞察：** 从数十篇消融论文中提炼出的关键发现——编码器 > 连接器 > 数据混合。数据质量比数据数量更重要。

---

## 2. 概念

### 2.1 VLM 五轴设计空间

| 轴 | 选项 | 影响 |
|------|------|------|
| **图像编码器** | CLIP / DINOv2 / SigLIP 2 / InternViT | 决定视觉理解的上限 |
| **连接器** | MLP / Q-Former / Perceiver Resampler | 信息传递效率 |
| **LLM** | Llama / Qwen / Mistral | 语言能力基础 |
| **数据混合** | 图像-标题 / 指令 / 视觉指令 | 决定模型能力分布 |
| **分辨率策略** | 固定 / AnyRes / NaViT | 影响 OCR/文档理解 |

### 2.2 跨论文一致发现

| 发现 | 证据来源 |
|------|---------|
| 编码器 > 连接器 | Cambrian-1（20+编码器比较） |
| 数据质量 > 数据数量 | Molmo（人工标题 vs 蒸馏） |
| 人工标题 > 合成数据 | Prismatic VLM（27种配方） |
| 分辨率是关键 | LLaVA-NeXT（AnyRes） |
| 多模态统一训练有效 | LLaVA-OneVision（单图+多图+视频） |

### 2.3 编码器选择指南

| 编码器 | 维度 | 适合场景 | 开源 |
|--------|------|---------|------|
| CLIP ViT-L/14 | 1024 | 通用（零样本好） | ✅ |
| DINOv2 ViT-g | 1536 | 密集预测（分割/深度） | ✅ |
| SigLIP 2 | 1152 | 超大批次训练 | ✅ |
| InternViT-6B | 3200 | 极致质量 | ✅ |

---

## 3. 从零实现

### Step 1：编码器评估框架

```python
def evaluate_encoder(encoder, test_dataset, task="retrieval"):
    """评估图像编码器在特定任务上的表现。"""
    # 提取嵌入
    embeddings = []
    for image in test_dataset:
        with torch.no_grad():
            emb = encoder(image.unsqueeze(0))
            embeddings.append(emb.squeeze())
    embeddings = torch.stack(embeddings)

    # 计算指标
    if task == "retrieval":
        # 图像检索：余弦相似度矩阵
        sim_matrix = embeddings @ embeddings.T
        # 对角线应该是最高分（每个图像与自己最相似）
        hits = 0
        for i in range(len(embeddings)):
            top_k = sim_matrix[i].topk(5).indices.tolist()
            if i in top_k:
                hits += 1
        return hits / len(embeddings)
```

### Step 2：数据混合分析

```python
def analyze_data_mix(caption_lengths, domains):
    """分析数据集的组成。"""
    stats = {
        "avg_caption_length": sum(caption_lengths) / len(caption_lengths),
        "domain_distribution": Counter(domains),
        "total_samples": len(caption_lengths),
    }
    return stats
```

---

## 4. 工具

### 4.1 HuggingFace Hub

```python
from transformers import AutoModel, AutoProcessor

# 加载不同编码器
encoders = {
    "CLIP": "openai/clip-vit-large-patch14-336",
    "DINOv2": "facebook/dinov2-large",
    "SigLIP": "google/siglip-base-patch16-224",
}
```

---

## 6. 工程最佳实践

### 6.1 VLM 构建决策树

```
你的场景需要什么？
├── 通用对话（图像+文本）
│   └── LLaVA 架构 + CLIP ViT-L + 158K 指令数据
├── 高分辨率文档理解
│   └── LLaVA-NeXT AnyRes + 高分辨率编码器
├── 视频理解
│   └── LLaVA-OneVision + 视频帧采样
├── 极致质量
│   └── SigLIP 2 + 大 LLM + 人工标题数据
```

### 6.2 踩坑经验

- **连接器太复杂**：MLP 足够——Q-Former 只在需要压缩词元时有用
- **数据质量差**：GPT-4 生成的标题有幻觉——人工审核是必要的
- **编码器太小**：ViT-B/16 在高分辨率任务上不如 ViT-L/14

---

## 7. 常见错误

### 错误 1：过度优化连接器架构

**现象：** 花大量时间设计复杂的连接器——性能提升有限。

**原因：** 消融研究表明连接器架构的影响远小于数据质量和编码器选择。

### 错误 2：忽略数据清洗

**现象：** 训练数据中有大量噪声、重复、低质量样本——模型学到了错误模式。

---

## 8. 面试考点

### Q1：构建 VLM 时最重要的选择是什么？（难度：⭐⭐）

**参考答案：**
从跨论文的消融研究中，最重要的选择依次是：(1) **图像编码器**——决定了视觉理解的上限（CLIP 用于零样本，DINOv2 用于密集预测）；(2) **数据质量**——详细的人工标题比合成数据更有效；(3) **数据混合比例**——不同数据类型的比例需要调优。连接器架构（MLP vs Q-Former）的影响最小。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 消融实验 | "逐步移除组件" | 控制变量法——每次只改变一个因素，测量其对性能的影响 |
| 编码器 | "视觉大脑" | 将图像转换为嵌入向量的网络——决定视觉理解的上限 |
| 连接器 | "视觉-语言桥梁" | 将视觉嵌入转换为 LLM 可理解的词元 |
| 数据混合 | "训练数据配方" | 不同类型训练数据的比例——影响模型能力分布 |

---

## 📚 小结

VLM 设计空间的五个轴：编码器、连接器、LLM、数据混合、分辨率。跨论文一致的发现：编码器 > 连接器 > 数据质量。MLP 足够作为连接器——简单胜过复杂。数据质量比数据数量更重要。

---

## ✏️ 练习

1. **【对比】** 用 CLIP ViT-L 和 DINOv2 ViT-g 在同一个检索任务上对比性能
2. **【分析】** 统计 LLaVA 训练数据的标题长度分布——长标题 vs 短标题的比例

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 编码器评估 | `code/main.py` | 图像编码器性能对比框架 |

---

## 📖 参考资料

1. [论文] Li et al. "Cambrian-1: A Multimodal Look at Vision Capabilities". arXiv, 2024.
2. [论文] Prasad et al. "Prismatic VLMs: Investigating the Design Space of Visual-Language Models". arXiv, 2024.
3. [论文] Deitke et al. "Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models". arXiv, 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
