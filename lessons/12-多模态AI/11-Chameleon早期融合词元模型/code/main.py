# Chameleon 早期融合：VQ-VAE 图像 Tokenizer + 交错序列

import torch
import torch.nn as nn


class SimpleVQVAE(nn.Module):
    """简化版 VQ-VAE——将图像转换为离散词元。"""
    def __init__(self, n_embeddings=512, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, padding=1),
        )
        self.codebook = nn.Embedding(n_embeddings, embed_dim)
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

    def forward(self, x):
        z_q, indices = self.encode(x)
        return self.decode(z_q), indices


def build_interleaved_sequence(text_tokens, image_tokens):
    """构建交错的文本+图像序列。"""
    sequence = text_tokens + image_tokens
    modalities = ["text"] * len(text_tokens) + ["image"] * len(image_tokens)
    return sequence, modalities


if __name__ == "__main__":
    print("Chameleon VQ-VAE + 交错序列演示\n")
    vqvae = SimpleVQVAE(n_embeddings=512, embed_dim=256)
    img = torch.randn(1, 3, 32, 32)
    recon, indices = vqvae(img)
    print(f"输入: {img.shape}, 词元: {indices.shape}, 重建: {recon.shape}")
    params = sum(p.numel() for p in vqvae.parameters())
    print(f"参数量: {params/1e6:.1f}M")

    seq, mods = build_interleaved_sequence([1, 2, 3], [10, 11, 12])
    print(f"\n交错序列: {seq}")
    print(f"模态标记: {mods}")
