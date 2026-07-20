# 综合项目60——投影层与模态对齐（Projection Layer for Modality Alignment）

> 视觉编码器产生图像词元。文本解码器消费文本词元。两者生活在不同向量空间中。一个小型两层 MLP 将图像词元投影到文本嵌入空间，与配对标题的余弦对齐损失将两个空间拉近。那个投影是视觉语言模型中最小的部分，也是迁移中最重要的部分。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 构建将图像特征映射到文本嵌入空间的两层 MLP 投影器
- 构建模拟文本嵌入表（无预训练分词器、无真实语料）
- 计算投影图像词元与配对标题嵌入之间的余弦对齐损失
- 冻结视觉编码器和文本表，单独训练投影层

---

## 1. 问题

你有一个产生维度 `vision_hidden=768` 的视觉编码器。你有一个要连接在上面的文本解码器，其嵌入维度 `text_hidden=512`。解码器期望文本形状的词元。图像词元不是文本形状的——它们生活在编码器在仅视觉预训练期间学到的基中，与解码器的词向量没有关系。

两层 MLP 投影（线性→GELU→线性）弥合了这个差距。它足够小（约 `768×1024 + 1024×512 = 1.3M` 参数），可以在单个 GPU 上几分钟内训练好。在训练期间只有投影层移动——视觉编码器冻结，文本嵌入表冻结。这是 LLaVA 在 2023 年推出的配方。

---

## 2. 核心概念

### 2.1 投影前池化

视觉编码器输出 197 个词元。文本侧有一个标题级嵌入。要对齐它们，每个样本需要一个图像级向量。CLS 汇聚是最简单的——取编码器的第一个词元。平均汇聚在所有 197 个词元上取平均。

### 2.2 为什么两层而不是一层

单个线性层可以旋转和缩放，但如果两个空间存在曲率不匹配，无法修复基。GELU 在两个线性层之间给了投影一个非线性弯折，这在经验上足以对齐 CLIP 风格特征到语言模型嵌入。

| 层 | 形状 | 参数 |
|:--|:-----|:-----|
| fc1 | `(vision_hidden, projection_hidden)` | `768×1024 + 1024` |
| 激活 | GELU | 0 |
| fc2 | `(projection_hidden, text_hidden)` | `1024×512 + 512` |

### 2.3 余弦对齐损失

对齐并不意味着 `image_emb == text_emb`。对齐意味着 `image_emb` 在联合空间中与 `text_emb` 指向相同方向。余弦损失是 `1 - cos_sim(image, text)`，范围从 0（完美对齐）到 2（完全相反）。

### 2.4 冻结编码器

视觉编码器有 86M 参数。文本表另有数百万。从模拟语料训练全部参数是不可行的。冻结两者意味着投影的 1.3M 参数是唯一变化的部分——几百步训练就能降低损失。这正是每个基于适配器的视觉语言模型的操作形状。

---

## 3. 从零实现

```python
"""投影层与模态对齐——MLP 投影器+余弦对齐损失。"""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F


class MLPProjector(nn.Module):
    """两层 MLP 投影：vision_hidden → projection_hidden → text_hidden。"""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 1024, out_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MockTextEmbedding(nn.Module):
    """冻结的模拟文本嵌入表。"""
    def __init__(self, vocab_size: int = 128, dim: int = 512, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.table = nn.Embedding(vocab_size, dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.table(ids)


def cosine_alignment_loss(image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """每对 (image, text) 的余弦对齐损失。"""
    image_norm = F.normalize(image_emb, dim=-1)
    text_norm = F.normalize(text_emb, dim=-1)
    cos_sim = (image_norm * text_norm).sum(dim=-1)
    return (1 - cos_sim).mean()


def make_pair(seed: int, vocab_size: int = 128, cap_len: int = 8,
              vision_dim: int = 768, text_dim: int = 512) -> tuple:
    """生成一对合成 (图像特征, 标题 ID)。"""
    torch.manual_seed(seed)
    image_feat = torch.randn(vision_dim)
    cap_ids = torch.randint(2, vocab_size, (cap_len,))
    return image_feat, cap_ids


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟冻结的视觉编码器输出
    vision_encoder = nn.Linear(768, 768, bias=False)  # placeholder
    for p in vision_encoder.parameters():
        p.requires_grad = False

    projector = MLPProjector().to(device)
    text_embed = MockTextEmbedding().to(device)
    optimizer = torch.optim.Adam(projector.parameters(), lr=1e-3)

    n_pairs = 32
    pairs = [make_pair(i) for i in range(n_pairs)]

    print(f"开始训练投影器（{sum(p.numel() for p in projector.parameters()):,} 参数）")
    for step in range(200):
        total_loss = 0.0
        for img_feat, cap_ids in pairs:
            img_feat = img_feat.unsqueeze(0).to(device)
            cap_ids = cap_ids.unsqueeze(0).to(device)
            with torch.no_grad():
                img_enc = vision_encoder(img_feat)
            img_proj = projector(img_enc)
            cap_emb = text_embed(cap_ids).mean(dim=1)
            loss = cosine_alignment_loss(img_proj, cap_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if step % 50 == 0 or step == 199:
            print(f"  步 {step:3d}: 损失 = {total_loss / n_pairs:.4f}")

    # 验证
    img_feat, cap_ids = pairs[0]
    img_feat = img_feat.unsqueeze(0).to(device)
    cap_ids = cap_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        img_proj = projector(vision_encoder(img_feat))
        cap_emb = text_embed(cap_ids).mean(dim=1)
        cos = F.cosine_similarity(img_proj, cap_emb).item()
    print(f"\n最终余弦相似度: {cos:.4f}（训练前应接近 0，训练后应 > 0）")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 关键术语

| 术语 | 含义 |
|------|------|
| 模态对齐 | 使图像和文本嵌入在联合空间中可比较 |
| 投影头 | 将一个空间映射到另一个的小模块，通常是 2 层 MLP |
| 余弦相似度 | 点积除以 L2 范数的乘积 |
| 冻结编码器 | 视觉和文本模型所有参数设置 `requires_grad=False` |
| 模拟语料 | 合成配对数据，训练无需下载数据集 |

---

## 5. 工程最佳实践

### 5.1 LLaVA 两阶段训练

1. **阶段一**：冻结视觉编码器和 LLM，只训练投影器（约 1-2 小时）
2. **阶段二**：冻结视觉编码器，解冻 LLM 的 LoRA 适配器，联合训练

### 5.2 投影器变体

| 模型 | 投影器类型 | 参数量 |
|:-----|:----------|:-------|
| LLaVA 1.5 | 2 层 MLP (GELU) | ~1.3M |
| BLIP-2 | Q-Former (交叉注意力) | ~188M |
| Qwen-VL | 交叉注意力适配器 | ~200M |

### 5.3 中文场景特别建议

- 中文视觉语言模型（如 Qwen-VL、InternVL）使用相同的投影架构，只是 LLM 后端替换为中文优化的模型（Qwen、InternLM）。
- 中文 OCR 视觉语言模型的投影器需要更多的参数来对齐中文字符的视觉特征与语义嵌入。

---

## 6. 常见错误

- **池化方式选择错误**：CLS 汇聚适合分类任务，但细粒度识别任务中平均汇聚可能更好。
- **未冻结视觉编码器**：反向传播流经 86M 参数显著增加训练时间且不如先训练投影器效果好。
- **余弦损失的温度参数**：默认温度 1.0 可能不是最优。添加可学习的温度参数可以改善收敛。

---

## 📖 参考资料

1. [论文] Liu et al. "LLaVA: Large Language and Vision Assistant". NeurIPS 2023. https://arxiv.org/abs/2304.08485
2. [论文] Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training". NeurIPS 2023. https://arxiv.org/abs/2301.12597
3. [论文] Bai et al. "Qwen-VL: A Versatile Vision-Language Model". 2023. https://arxiv.org/abs/2308.12966
