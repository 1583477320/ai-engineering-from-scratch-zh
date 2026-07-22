# === main.py — Open-Vocabulary CLIP：从零实现图文对比学习 ===
# 对应课程：第 04 阶段 · 第 18 课（Open-Vocabulary CLIP）
# 依赖：torch>=2.0, torchvision>=0.15, pillow, open_clip_torch
# 安装：pip install torch torchvision open_clip_torch


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple

import math
import random


# ================================================================
# 第 1 部分：从零实现简易双塔模型
# ================================================================

class SimpleVisualEncoder(nn.Module):
    """轻量级视觉编码器：用全局平均池化 + 线性层模拟 ViT 的输出。

    真实 CLIP 使用预训练 ViT-B/14 或 ViT-L/14，此处用 2 层 MLP
    接收 ImageNet 预训练特征（3×224×224 → 512），确保 CPU 可运行。
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, embed_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, embed_dim)
        return self.backbone(x)


class SimpleTextEncoder(nn.Module):
    """轻量级文本编码器：字符嵌入 + 池化，模拟 Transformer 编码。

    真实 CLIP 使用带位置编码的 Transformer Decoder，此处用字符
    嵌入的平均池化来模拟文本到向量的映射。
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64, output_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Linear(embed_dim, output_dim)

    def encode_string(self, text: str, max_len: int = 77) -> torch.Tensor:
        """将单条字符串转换为编码张量。"""
        # 简单按字节映射到词汇表索引
        token_ids = []
        for ch in text[:max_len]:
            token_ids.append(ord(ch) % 1000)
        # 补齐
        while len(token_ids) < max_len:
            token_ids.append(0)
        ids = torch.tensor([token_ids], dtype=torch.long)
        embs = self.embedding(ids)                # (1, 77, 64)
        pooled = embs.mean(dim=1)                 # (1, 64)
        return self.projection(pooled)            # (1, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class TwoTowerCLIP(nn.Module):
    """双塔对比学习模型。

    核心设计：图像塔和文本塔分别编码，在同一个嵌入空间中对齐。
    与真实 OpenAI CLIP API 一致：encode_image() / encode_text() / logit_scale。
    """

    def __init__(
        self,
        visual_input_dim: int = 512,
        text_vocab_size: int = 1000,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.visual_encoder = SimpleVisualEncoder(
            input_dim=visual_input_dim, hidden_dim=embed_dim // 2, embed_dim=embed_dim
        )
        self.text_encoder = SimpleTextEncoder(text_vocab_size, embed_dim=embed_dim // 4, output_dim=embed_dim)
        # 可学习温度参数，初始 ln(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def encode_image(self, features: torch.Tensor) -> torch.Tensor:
        """编码图像特征并 L2 归一化。

        Args:
            features: (batch, visual_input_dim)

        Returns:
            归一化后的图像嵌入 (batch, embed_dim)
        """
        emb = self.visual_encoder(features)
        return F.normalize(emb, dim=-1)

    def encode_text_str(self, texts: List[str]) -> torch.Tensor:
        """编码文本列表。

        Args:
            texts: 字符串列表

        Returns:
            归一化后的文本嵌入 (len(texts), embed_dim)
        """
        encoded = [self.text_encoder.encode_string(t) for t in texts]
        stacked = torch.cat(encoded, dim=0)
        return F.normalize(stacked, dim=-1)

    def forward(
        self, image_features: torch.Tensor, text_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """前向传播。

        Args:
            image_features: (batch, visual_input_dim) 原始图像特征
            text_embeds: (batch, text_hidden) 文本嵌入（来自 text_encoder 之前）

        Returns:
            norm_img: 归一化图像嵌入
            norm_txt: 归一化文本嵌入
            scale:    温度标量的 exp
        """
        norm_img = F.normalize(self.visual_encoder(image_features), dim=-1)
        norm_txt = F.normalize(self.text_encoder(text_embeds), dim=-1)
        scale = self.logit_scale.exp()
        return norm_img, norm_txt, scale.item()


# ================================================================
# 第 2 部分：对比损失函数
# ================================================================

def clip_contrastive_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    logit_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CLIP 对称对比损失。

    给定 batch 中 N 对 (图像, 文本)，构建 N×N 相似度矩阵。
    对角线代表匹配对，需有高相似度；非对角线需有低相似度。

    $$L = \\frac{1}{2} (\\text{CE}(S, y) + \\text{CE}(S^T, y))$$

    其中 $S = \\tau \\cdot E_i E_T^T$，$y = [0, 1, ..., N-1]$。

    Args:
        image_emb:    归一化图像嵌入 (N, D)
        text_emb:     归一化文本嵌入 (N, D)
        logit_scale:  温度参数 tau

    Returns:
        loss:          总损失标量
        i2t_ce:        图像→文本交叉熵
        t2i_ce:        文本→图像交叉熵
    """
    N = image_emb.size(0)
    # 相似度矩阵 × 温度
    sim_matrix = logit_scale * (image_emb @ text_emb.T)

    # 标签：对角线为匹配对
    labels = torch.arange(N, device=image_emb.device)

    # 双向交叉熵
    i2t_ce = F.cross_entropy(sim_matrix, labels)
    t2i_ce = F.cross_entropy(sim_matrix.T, labels)

    loss = (i2t_ce + t2i_ce) / 2.0
    return loss, i2t_ce, t2i_ce


# ================================================================
# 第 3 部分：零样本分类
# ================================================================

@torch.no_grad()
def zero_shot_classify(
    model: TwoTowerCLIP,
    image_features: torch.Tensor,
    class_prompts: List[List[str]],
    batch_size: int = 32,
) -> List[List[Tuple[str, float]]]:
    """使用训练好的双塔模型进行零样本分类。

    对每个类别使用多条提示词模板，对文本嵌入取平均以增强鲁棒性。

    Args:
        model:            训练好的双塔模型
        image_features:   测试图像特征 (N, visual_input_dim)
        class_prompts:    每个类别对应的提示词列表，如 [["a photo of a {}"], ...]
        batch_size:       文本编码批次大小

    Returns:
        每个图像的排序后类别列表 [(class_name, similarity_score), ...]
    """
    device = image_features.device

    # 编码所有类别提示词并取平均
    all_class_embeddings = []
    class_names = []

    for cls_idx, prompts in enumerate(class_prompts):
        # 逐批编码同一类别的多条模板
        class_embs = []
        for i in range(0, len(prompts), batch_size):
            batch_texts = prompts[i : i + batch_size]
            emb = model.encode_text_str(batch_texts).to(device)
            class_embs.append(emb)
        avg_emb = torch.cat(class_embs, dim=0).mean(dim=0, keepdim=True)  # (1, D)
        all_class_embeddings.append(avg_emb)
        class_names.append(f"类别_{cls_idx}")

    class_matrix = torch.cat(all_class_embeddings, dim=0)  # (C, D)

    # 编码所有测试图像
    images_normalized = model.encode_image(image_features)  # (N, D)

    # 计算相似度矩阵 (N, C)
    sim_matrix = logit_scale_from_model(model) * (images_normalized @ class_matrix.T)

    # 返回每个图像的前 3 个预测
    top_k = min(3, sim_matrix.size(1))
    results = []
    top_scores, top_indices = torch.topk(sim_matrix, top_k, dim=-1)

    for i in range(sim_matrix.size(0)):
        ranked = [
            (class_names[j.item()], sim_matrix[i, j].item())
            for j in top_indices[i]
        ]
        results.append(ranked)

    return results


def logit_scale_from_model(model: TwoTowerCLIP) -> float:
    """从模型中提取当前温度参数。"""
    return model.logit_scale.exp().item()


# ================================================================
# 第 4 部分：提示词工程 — 多模板平均
# ================================================================

def build_prompt_templates(
    templates: List[str],
    class_names: List[str],
) -> List[List[str]]:
    """为每个类别构建多条提示词模板。

    模板中的 {} 会被替换为类别名称。
    模板数量越多，零样本准确率通常越高（直到收益递减）。
    OpenAI 原始论文使用 80 条模板，在 ImageNet 上提升约 2 个百分点。

    Args:
        templates: 模板列表，每个包含一个 {} 占位符
        class_names: 类别名称列表

    Returns:
        每个类别的提示词列表
    """
    result = []
    for cls_name in class_names:
        prompts = [tmpl.format(cls_name) for tmpl in templates]
        result.append(prompts)
    return result


def template_weighted_predict(
    image_emb: torch.Tensor,
    text_embs_per_class: List[torch.Tensor],
    template_weights: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, int]:
    """对多个模板的预测结果进行加权投票。

    Args:
        image_emb:           图像嵌入 (D,)
        text_embs_per_class: 每个类别的模板嵌入列表 [(M_c, D), ...]
        template_weights:    可选的模板权重；None 时均匀加权

    Returns:
        predicted_label: 预测的类别索引
        confidence:      最高相似度分数
    """
    N_classes = len(text_embs_per_class)
    D = image_emb.size(0)

    # 归一化图像
    image_emb = F.normalize(image_emb, dim=-1)

    # 累计所有模板的相似度
    total_scores = torch.zeros(N_classes, device=image_emb.device)

    for cls_idx, text_embs in enumerate(text_embs_per_class):
        if template_weights:
            w = template_weights[cls_idx]
        else:
            w = 1.0

        # 该类别所有模板的嵌入均值
        avg_emb = text_embs.mean(dim=0)  # (D,)
        score = F.cosine_similarity(image_emb, avg_emb, dim=0) * w
        total_scores[cls_idx] += score

    predicted_label = total_scores.argmax().item()
    confidence = total_scores[predicted_label].item()
    return predicted_label, confidence


# ================================================================
# 第 5 部分：TIP-Adapter — 无需微调的提示适配器
# ================================================================

class TIPAdapter(nn.Module):
    """TIP-Adapter：基于验证集的提示词优化器。

    TIP-Adapter（Prompt-Tuning Adapter）的思想是：不修改预训练模型
    的权重，而是在推理时使用一个小型适配器网络来校准提示词的嵌入。

    适配器学习一组可训练向量，在推理时对文本嵌入做微调：

        text_emb_adapted = original_text_emb + adapter(image_emb)

    这使得文本嵌入可以"感知"到具体的输入图像，同时保持零样本能力。
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        # 适配器的内容向量（可学习）
        self.content_vector = nn.Parameter(torch.randn(1, hidden_dim))
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # 缩放系数 alpha（从 0 开始，初始等价于零样本）
        self.alpha = nn.Parameter(torch.zeros([]))

    def forward(
        self,
        text_embs: torch.Tensor,    # (C, D)
        image_feat: torch.Tensor,   # (B, visual_dim) 来自原始特征
        model: TwoTowerCLIP,
    ) -> torch.Tensor:
        """生成适配后的文本嵌入。

        Args:
            text_embs:  原始文本嵌入 (C, D)
            image_feat: 原始图像特征 (B, visual_dim)
            model:      预训练双塔模型，用于从 image_feat 恢复嵌入

        Returns:
            适配后的文本嵌入 (C, B, D)
        """
        # 从图像特征生成偏移向量
        img_emb = model.encode_image(image_feat)    # (B, D)
        offset = self.projection(self.content_vector.expand(img_emb.size(0), -1))
        offset = img_emb + self.alpha * offset       # (B, D)

        # 广播到 (C, B, D)
        text_adapted = text_embs.unsqueeze(1) + offset.unsqueeze(0)
        return F.normalize(text_adapted, dim=-1)


@torch.no_grad()
def tip_adapter_inference(
    model: TwoTowerCLIP,
    adapter: TIPAdapter,
    image_features: torch.Tensor,
    text_embs: torch.Tensor,
    class_names: List[str],
) -> List[int]:
    """使用 TIP-Adapter 进行推理。

    Args:
        model:         预训练模型
        adapter:       训练好的适配器
        image_features: 图像特征 (B, visual_dim)
        text_embs:     原始文本嵌入 (C, D)
        class_names:   类别名称

    Returns:
        每个图像的预测类别索引
    """
    B = image_features.size(0)
    adapted_embs = adapter(text_embs, image_features, model)  # (C, B, D)

    img_emb = model.encode_image(image_features)               # (B, D)
    # 相似度 (B, C, B) → 取第一维作为最终相似度
    sim = F.normalize(img_emb, dim=-1).unsqueeze(1) @ adapted_embs.permute(1, 0, 2)
    best = sim[:, 0].argmax(dim=-1)                            # (B,)
    return best.tolist()


# ================================================================
# 第 6 部分：图文检索
# ================================================================

def compute_similarity_matrix(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
) -> torch.Tensor:
    """计算图像与文本之间的相似度矩阵。

    两个嵌入都已 L2 归一化，所以点积等于余弦相似度。

    Args:
        image_embs: 图像嵌入 (N_img, D)
        text_embs:  文本嵌入 (N_txt, D)

    Returns:
        相似度矩阵 (N_img, N_txt)，值域 [-1, 1]
    """
    return image_embs @ text_embs.T


@torch.no_grad()
def image_to_text_retrieval(
    model: TwoTowerCLIP,
    image_features: torch.Tensor,
    texts: List[str],
    k: int = 5,
) -> List[List[Tuple[str, float]]]:
    """图像到文本的检索。

    给定一张（或多张）图片和一组候选文本，返回最匹配的文本排名。

    Args:
        model:             双塔模型
        image_features:    图像特征 (B, visual_dim)
        texts:             候选文本列表
        k:                 返回前 k 个匹配

    Returns:
        每张图像的前 k 条匹配结果 [(text, score), ...]
    """
    device = image_features.device
    # 批量编码所有文本
    all_text_embs = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        emb = model.encode_text_str(batch).to(device)
        all_text_embs.append(emb)
    text_matrix = torch.cat(all_text_embs, dim=0)  # (N_txt, D)

    # 编码图像
    img_embs = model.encode_image(image_features)  # (B, D)

    # 相似度矩阵
    sim = img_embs @ text_matrix.T  # (B, N_txt)

    results = []
    for b in range(img_embs.size(0)):
        top_k_idx = sim[b].topk(k).indices.tolist()
        ranked = [(texts[idx], sim[b, idx].item()) for idx in top_k_idx]
        results.append(ranked)

    return results


@torch.no_grad()
def text_to_image_retrieval(
    model: TwoTowerCLIP,
    image_features: torch.Tensor,
    queries: List[str],
    k: int = 5,
) -> List[List[Tuple[int, float]]]:
    """文本到图像的检索。

    Args:
        model:             双塔模型
        image_features:    图像特征 (N_img, visual_dim)
        queries:           查询文本列表
        k:                 返回前 k 个匹配

    Returns:
        每条查询的前 k 个匹配图像索引 [(image_index, score), ...]
    """
    device = image_features.device
    img_embs = model.encode_image(image_features).to(device)  # (N_img, D)

    all_img_embs = []
    for q in queries:
        emb = model.encode_text_str([q]).to(device)  # (1, D)
        all_img_embs.append(emb)
    query_matrix = torch.cat(all_img_embs, dim=0)  # (N_query, D)

    sim = query_matrix @ img_embs.T  # (N_query, N_img)

    results = []
    for q in range(query_matrix.size(0)):
        top_k_idx = sim[q].topk(k).indices.tolist()
        ranked = [(idx, sim[q, idx].item()) for idx in top_k_idx]
        results.append(ranked)

    return results


# ================================================================
# 第 7 部分：合成数据与训练
# ================================================================

class SyntheticClipDataset(Dataset):
    """合成数据集：用于在 CPU 上快速演示 CLIP 训练流程。

    每对样本共享一个语义原型（prototype），正对的距离近、
    负对的距离远。模拟真实场景中的 (image, caption) 对齐。
    """

    def __init__(self, num_pairs: int, num_classes: int, dim: int = 32):
        self.num_pairs = num_pairs
        self.num_classes = num_classes
        self.dim = dim
        # 随机初始化原型
        rng = torch.Generator().manual_seed(42)
        self.prototypes = F.normalize(torch.randn(num_classes, dim, generator=rng), dim=-1)
        self.labels = torch.randint(0, num_classes, (num_pairs,))

    def __len__(self) -> int:
        return self.num_pairs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        label = self.labels[idx].item()
        proto = self.prototypes[label]

        # 模拟图像特征：在原型附近加噪声
        img_feat = torch.randn(512)
        img_feat[: self.dim] = proto + 0.15 * torch.randn(self.dim)

        # 模拟文本特征：也在原型附近加噪声
        txt_feat = torch.randn(128)
        txt_feat[: self.dim] = proto + 0.15 * torch.randn(self.dim)

        return img_feat, txt_feat, label


def train_clip_step(
    model: TwoTowerCLIP,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float, float]:
    """执行一步 CLIP 训练。

    Args:
        model:           双塔模型
        image_features:  图像特征 (B, V_dim)
        text_features:   文本特征 (B, T_dim)
        optimizer:       优化器

    Returns:
        loss, i2t_loss, t2i_loss
    """
    optimizer.zero_grad()
    img_emb, txt_emb, scale = model(image_features, text_features)
    loss, i2t, t2i = clip_contrastive_loss(img_emb, txt_emb, scale)
    loss.backward()
    optimizer.step()
    return loss.item(), i2t.item(), t2i.item()


# ================================================================
# 主程序：完整演示流程
# ================================================================

def main():
    """运行完整的 CLIP 演示：训练 → 零样本分类 → 提示工程 → TIP-Adapter → 检索。"""

    random.seed(0)
    torch.manual_seed(0)

    embed_dim = 64
    num_classes = 5
    model = TwoTowerCLIP(
        visual_input_dim=512,
        text_vocab_size=1000,
        embed_dim=embed_dim,
    )

    print("=" * 60)
    print("第 1 步：随机初始化模型 —  Sanity Check")
    print("=" * 60)
    batch_size = 8
    img_rand = torch.randn(batch_size, 512)
    txt_rand = torch.randn(batch_size, 128)
    img_emb, txt_emb, scale = model(img_rand, txt_rand)
    loss, i2t, t2i = clip_contrastive_loss(img_emb, txt_emb, scale)
    expected_loss = math.log(batch_size)
    print(f"  批次大小: {batch_size}")
    print(f"  温度参数 tau = exp(logit_scale) = {scale:.3f}")
    print(f"  初始损失: {loss.item():.3f} （期望接近 log({batch_size}) = {expected_loss:.3f}）")
    print(f"  图像嵌入形状: {img_emb.shape}")
    print(f"  文本嵌入形状: {txt_emb.shape}")

    print("\n" + "=" * 60)
    print("第 2 步：在合成数据上训练 CLIP")
    print("=" * 60)
    dataset = SyntheticClipDataset(num_pairs=200, num_classes=num_classes, dim=embed_dim)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    num_epochs = 20
    for epoch in range(num_epochs):
        total_loss = 0.0
        for img_b, txt_b, _ in dataloader:
            l, _, _ = train_clip_step(model, img_b, txt_b, optimizer)
            total_loss += l
        avg_loss = total_loss / len(dataloader)
        lr_scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{num_epochs}  loss = {avg_loss:.3f}  "
                  f"tau = {model.logit_scale.exp().item():.3f}")

    print("\n" + "=" * 60)
    print("第 3 步：零样本分类")
    print("=" * 60)

    # 使用原型作为类别的文本嵌入
    class_proto_embs = []
    for c in range(num_classes):
        proto = dataset.prototypes[c]
        txt_embed = torch.randn(128)
        txt_embed[:embed_dim] = proto + 0.1 * torch.randn(embed_dim)
        class_proto_embs.append(txt_embed)
    class_matrix = torch.stack(class_proto_embs)  # (C, 128)

    # 生成测试图像
    test_labels = torch.randint(0, num_classes, (16,))
    test_imgs = []
    for label in test_labels:
        feat = torch.randn(512)
        feat[:embed_dim] = dataset.prototypes[label.item()] + 0.15 * torch.randn(embed_dim)
        test_imgs.append(feat)
    test_imgs = torch.stack(test_imgs)  # (16, 512)

    # 构建类别提示词（多模板）
    templates = [
        "一张 {} 的图片",
        "一幅 {} 的素描",
        "一个 {} 的特写镜头",
        "一个模糊的 {} 图像",
        "一个低分辨率的 {} 照片",
    ]
    class_names = [f"类别_{c}" for c in range(num_classes)]
    class_prompts = build_prompt_templates(templates, class_names)

    predictions = zero_shot_classify(model, test_imgs, class_prompts)

    correct = sum(1 for i, pred_list in enumerate(predictions) if pred_list[0][0] == f"类别_{test_labels[i].item()}")
    print(f"  多模板零样本准确率: {correct}/16 = {correct / 16:.1%}")

    print("\n  示例预测结果（前 3 条）：")
    for i in range(min(3, len(predictions))):
        top3 = predictions[i]
        print(f"    图像 {i}:")
        for rank, (cls_name, score) in enumerate(top3, 1):
            marker = " <- 正确" if cls_name == f"类别_{test_labels[i].item()}" else ""
            print(f"      #{rank} {cls_name} (相似度={score:.3f}){marker}")

    print("\n" + "=" * 60)
    print("第 4 步：单模板 vs 多模板 对比")
    print("=" * 60)

    single_templates = ["一张{}的图片"]
    single_prompts = build_prompt_templates(single_templates, class_names)
    single_preds = zero_shot_classify(model, test_imgs, single_prompts)
    single_correct = sum(1 for i, pred_list in enumerate(single_preds)
                         if pred_list[0][0] == f"类别_{test_labels[i].item()}")
    print(f"  单模板准确率:   {single_correct}/16 = {single_correct / 16:.1%}")
    print(f"  多模板准确率:   {correct}/16 = {correct / 16:.1%}")
    print(f"  提升:           {(correct - single_correct)}/16 = {(correct - single_correct) / 16:.1%}")

    print("\n" + "=" * 60)
    print("第 5 步：TIP-Adapter — 无需微调的提示词适配")
    print("=" * 60)

    adapter = TIPAdapter(embed_dim=embed_dim, hidden_dim=32)
    adapter_optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    # 用一个小型验证集校准适配器
    val_dataset = SyntheticClipDataset(num_pairs=50, num_classes=num_classes, dim=embed_dim)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 准备验证集文本嵌入
    val_class_embs = []
    for c in range(num_classes):
        proto = val_dataset.prototypes[c]
        txt_f = torch.randn(128)
        txt_f[:embed_dim] = proto + 0.1 * torch.randn(embed_dim)
        val_class_embs.append(txt_f)
    val_class_matrix = torch.stack(val_class_embs)  # (C, 128)

    # 训练适配器
    val_num_epochs = 30
    for epoch in range(val_num_epochs):
        for img_b, txt_b, val_labels in val_dataloader:
            adapter_optimizer.zero_grad()
            adapted = adapter(val_class_matrix, img_b, model)  # (C, B, D)
            img_norm = model.encode_image(img_b)              # (B, D)

            # 使用适配后的文本嵌入计算损失
            sim = F.normalize(img_norm, dim=-1).unsqueeze(1) @ adapted.permute(1, 0, 2)
            sim_batch = sim[:, 0]  # (B, C)
            targets = val_labels.to(sim_batch.device)
            adapter_loss = F.cross_entropy(sim_batch, targets)
            adapter_loss.backward()
            adapter_optimizer.step()

    print(f"  适配器训练完成（{val_num_epochs} 轮）")
    print(f"  缩放系数 alpha = {adapter.alpha.item():.4f} "
          f"(接近 0 表示适配器未大幅改变原始嵌入)")

    # 在测试集上比较
    adapter_preds = tip_adapter_inference(
        model, adapter, test_imgs, class_matrix, class_names
    )
    adapter_correct = sum(1 for i, pred_idx in enumerate(adapter_preds)
                          if class_names[pred_idx] == f"类别_{test_labels[i].item()}")
    print(f"  TIP-Adapter 准确率: {adapter_correct}/16 = {adapter_correct / 16:.1%}")

    print("\n" + "=" * 60)
    print("第 6 步：图像到文本检索")
    print("=" * 60)

    # 准备检索语料库
    corpus_texts = [
        "一只橙色的小猫在草地上玩耍",
        "一辆红色的跑车在高速公路上行驶",
        "一座雪山倒映在平静的湖面",
        "一栋现代风格的玻璃幕墙建筑",
        "一群海鸥在黄昏的海岸线上飞翔",
        "一本摊开的书籍和一杯咖啡放在木桌上",
        "一朵绽放的粉色樱花特写",
        "一条繁忙的城市街道夜景",
        "一片金黄色的麦田在风中起伏",
        "一台笔记本电脑显示着代码编辑器界面",
    ]

    # 生成一些测试图像
    retriever_imgs = []
    for _ in range(5):
        feat = torch.randn(512)
        feat[:embed_dim] = dataset.prototypes[random.randint(0, num_classes - 1)]
        feat = feat + 0.1 * torch.randn(512)
        retriever_imgs.append(feat)
    retriever_imgs = torch.stack(retriever_imgs)

    retrievals = image_to_text_retrieval(model, retriever_imgs, corpus_texts, k=3)
    for i, ranked in enumerate(retrievals):
        print(f"  图像 {i} 的前 3 条匹配描述：")
        for rank, (text, score) in enumerate(ranked, 1):
            print(f"    #{rank} \"{text}\" (相似度={score:.3f})")

    print("\n" + "=" * 60)
    print("第 7 步：文本到图像检索")
    print("=" * 60)

    queries = [
        "一只小猫",
        "城市夜景",
        "自然风光",
    ]
    text_retrievals = text_to_image_retrieval(model, retriever_imgs, queries, k=3)
    for q_idx, query in enumerate(queries):
        ranked = text_retrievals[q_idx]
        print(f"  查询 \"{query}\" 的 Top-3 匹配图像索引：")
        for rank, (img_idx, score) in enumerate(ranked, 1):
            print(f"    #{rank} 图像_{img_idx} (相似度={score:.3f})")

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
