# CLIP 对比预训练：InfoNCE + 零样本分类

import torch
import torch.nn.functional as F


# ============================================================================
# 第 1 步：InfoNCE 对比损失
# ============================================================================

def info_nce_loss(image_embeds, text_embeds, temperature=0.07):
    """InfoNCE 对比损失——CLIP 的核心训练目标。"""
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits = image_embeds @ text_embeds.T / temperature
    labels = torch.arange(len(logits), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


# ============================================================================
# 第 2 步：Sigmoid 成对损失（SigLIP）
# ============================================================================

def sigmoid_contrastive_loss(image_embeds, text_embeds, temperature=10.0):
    """SigLIP 的 sigmoid 成对损失——无需全 gather。"""
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    logits = image_embeds @ text_embeds.T * temperature
    N = len(logits)
    labels = torch.eye(N, device=logits.device)
    loss = -labels * F.logsigmoid(logits) - (1 - labels) * F.logsigmoid(-logits)
    return loss.mean()


# ============================================================================
# 第 3 步：零样本分类
# ============================================================================

def zero_shot_classify(image_embed, class_names, text_embeds, class_text_embeds=None):
    """零样本分类。"""
    if class_text_embeds is None:
        class_text_embeds = text_embeds
    image_embed = F.normalize(image_embed, dim=-1)
    class_text_embeds = F.normalize(class_text_embeds, dim=-1)
    similarities = image_embed @ class_text_embeds.T
    predicted_idx = similarities.argmax().item()
    return class_names[predicted_idx], similarities


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("CLIP 对比预训练演示\n")

    # 1. InfoNCE 损失
    image_embeds = torch.randn(8, 256)
    text_embeds = torch.randn(8, 256)
    loss = info_nce_loss(image_embeds, text_embeds, temperature=0.07)
    print(f"InfoNCE 损失 (batch=8): {loss.item():.4f}")

    # 2. SigLIP 损失
    loss_siglip = sigmoid_contrastive_loss(image_embeds, text_embeds, temperature=10.0)
    print(f"SigLIP 损失 (batch=8): {loss_siglip.item():.4f}")

    # 3. 零样本分类模拟
    class_names = ["猫", "狗", "鸟", "车"]
    class_embeds = torch.randn(4, 256)  # 模拟类别嵌入
    image_embed = torch.randn(1, 256) * 0.5  # 模拟图像嵌入（偏向"猫"）
    image_embed[0, :128] += 0.3
    class_embeds[0, :128] += 0.3

    pred, sims = zero_shot_classify(image_embed, class_names, None, class_embeds)
    print(f"\n零样本分类: 预测={pred}, 相似度={sims[0].tolist()}")

    # 4. 温度影响
    print("\n温度对训练的影响:")
    for temp in [0.01, 0.07, 0.1, 0.5]:
        l = info_nce_loss(image_embeds, text_embeds, temperature=temp)
        print(f"  τ={temp}: loss={l.item():.4f}")
