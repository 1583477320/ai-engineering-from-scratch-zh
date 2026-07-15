# Transfusion 双损失 Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransfusionTransformer(nn.Module):
    """简化版 Transfusion——NTP（文本）+ 扩散 MSE（图像）。"""
    def __init__(self, vocab_size=32000, hidden=512, n_layers=6, n_heads=8):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, hidden)
        self.image_proj = nn.Linear(3 * 16 * 16, hidden)
        self.pos_embed = nn.Embedding(256, hidden)

        encoder_layer = nn.TransformerEncoderLayer(hidden, n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.text_head = nn.Linear(hidden, vocab_size)
        self.image_head = nn.Linear(hidden, 3 * 16 * 16)

    def forward(self, text_tokens=None, image_patches=None):
        embeddings = []
        if text_tokens is not None:
            embeddings.append(self.text_embed(text_tokens))
        if image_patches is not None:
            embeddings.append(self.image_proj(image_patches))
        x = torch.cat(embeddings, dim=1)
        x = x + self.pos_embed(torch.arange(x.shape[1], device=x.device))
        x = self.transformer(x)
        text_logits = self.text_head(x) if text_tokens is not None else None
        image_logits = self.image_head(x) if image_patches is not None else None
        return text_logits, image_logits


if __name__ == "__main__":
    print("Transfusion 双损失 Transformer 演示\n")
    model = TransfusionTransformer(vocab_size=32000, hidden=256, n_layers=4, n_heads=4)
    text = torch.randint(0, 32000, (2, 10))
    image = torch.randn(2, 3 * 16 * 16)
    text_logits, image_logits = model(text, image)
    print(f"文本 logits: {text_logits.shape}")
    print(f"图像 logits: {image_logits.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params/1e6:.1f}M")
