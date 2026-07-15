# Show-o 掩码离散扩散

import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_tokens(tokens, mask_ratio=0.5, mask_id=100):
    """随机掩码图像词元。"""
    mask = torch.bernoulli(torch.full_like(tokens.float(), mask_ratio))
    masked = tokens * (1 - mask) + mask * mask_id
    return masked, mask


def masked_discrete_diffusion_loss(model, tokens, mask_ratio=0.5):
    """掩码离散扩散损失。"""
    masked, mask = mask_tokens(tokens, mask_ratio)
    logits = model(masked)
    mask_flat = mask.view(-1).bool()
    targets = tokens.view(-1)
    logits_flat = logits.view(-1, logits.size(-1))
    loss = F.cross_entropy(logits_flat, targets, reduction='none')
    loss = (loss * mask_float).sum() / mask_float.sum().clamp(min=1)
    return loss


def show_o_generate(model, prompt_tokens, max_new=64, temperature=0.8):
    """Show-o 自回归生成。"""
    generated = list(prompt_tokens)
    for _ in range(max_new):
        logits = model(torch.tensor([generated]))[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
    return generated


if __name__ == "__main__":
    print("Show-o 掩码离散扩散演示\n")
    tokens = torch.randint(0, 512, (1, 64))
    masked, mask = mask_tokens(tokens, mask_ratio=0.5)
    print(f"原始词元: {tokens.shape}")
    print(f"掩码后: {masked.shape}")
    print(f"掩码比例: {mask.mean():.0%}")
    print(f"掩码词元数: {int(mask.sum())}")
